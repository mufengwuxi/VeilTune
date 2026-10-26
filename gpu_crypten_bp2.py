"""Communication-reduced secure backprop (bp2) = gpu_crypten_bp with two kernels replaced.

1) GELU (exact-erf target, not the tanh form). gelu(z) = z/2 + E(z), E even.
   |z| < c (c = 4): E(z) = R(u), gelu'(z) - 1/2 = z * S(u), u = z^2 / c^2, Horner (R deg 7, S deg 6, stacked).
   |z| >= c       : gelu = relu (error <= 1.3e-4), gelu' = [z > 0].
   Gates: [z < 0] and [z^2 < c^2] in ONE batched comparison (2 comparisons / element instead of 3), no exp,
   no 1/(1+e) polynomial. Plaintext fixed-point emulation: max |err| 2.2e-4 (gelu), 1.2e-3 (gelu').
2) Row max for softmax: own binary tournament (1 comparison + 1 multiplication per eliminated element,
   max value only) instead of CrypTen's max(dim) (log-reduction + pairwise + one-hot argmax).
"""
import math
import crypten
import gpu_crypten_v2 as V

import gpu_crypten_bp as BPm
from gpu_crypten_v2 import _gate, neg1p, to_fx
from gpu_crypten import const

GC = 4.0
GA = [0.000165592118, 6.35914196, -16.4498848, 35.5286104, -52.3277043, 48.9141622, -25.8863215, 5.86186901]
GB = [0.793507332, -4.06499135, 12.8734628, -24.3811859, 27.1982983, -16.3942397, 4.10054477]


class SecureOpsBP2(BPm.SecureOpsBP):

    def gelu_save(self, z):
        if not getattr(self, 'NEW_GELU', True):                       # ablation: old tanh-form GELU
            return super().gelu_save(z)
        z2 = z * z                                                    # 1 round
        g = _gate(crypten.stack([z, z2 - GC * GC]))                   # [z<0], [z^2<c^2]: one comparison call
        neg, inn = g[0], g[1]
        u = z2 * const(1.0 / (GC * GC))
        R = u * const(GA[7]) + GA[6]                                  # public coefficient: local
        S = u * const(GB[6]) + GB[5]
        for j in range(6):                                            # 6 rounds
            if j < 5:
                pr = crypten.stack([R, S]) * crypten.stack([u, u])
                R, S = pr[0] + GA[5 - j], pr[1] + GB[4 - j]
            else:
                R = R * u + GA[0]
        sgn = to_fx(neg1p(neg * 2), z)                                # +-1 at fixed-point scale (exact)
        pr = crypten.stack([z, z]) * crypten.stack([sgn, S])          # |z|, z*S(u)
        absz, zS = pr[0], pr[1]
        innf = to_fx(inn, z)
        sel = crypten.stack([innf, innf]) * crypten.stack([R - absz * 0.5, zS - sgn * 0.5])
        out = z * 0.5 + absz * 0.5 + sel[0]
        gp = sgn * 0.5 + 0.5 + sel[1]
        return out, gp

    @staticmethod
    def exp_free(x):
        """exp(x), x <= 0, no clamp comparison: Taylor deg 6 on x/128, 7 squarings. Accurate for x >= -128,
        bounded (-> 0) down to about -400; measured real-key score spread <= 47 (t15). Masked keys are
        zeroed by the public mask afterwards."""
        t = x * const(1.0 / 128)
        pw = V.powers(t, 6)
        acc = x * 0 + 1.0
        for k in range(1, 7):
            acc = acc + pw[k - 1] * const(1.0 / math.factorial(k))
        for _ in range(7):
            acc = acc * acc
        return acc

    @staticmethod
    def max_tree(x):
        d = x.dim() - 1
        while x.size(d) > 1:
            n = x.size(d); h = n // 2
            a, b = x.narrow(d, 0, h), x.narrow(d, h, h)
            m = a + _gate(a - b) * (b - a)                            # max(a, b)
            x = crypten.cat([m, x.narrow(d, 2 * h, 1)], dim=d) if n % 2 else m
        return x

    def softmax_save(self, s, mask):
        s = s + (mask - 1.0) * 30.0
        m = self.max_tree(s) if getattr(self, 'MAX_TREE', True) else s.max(dim=-1, keepdim=True)[0]
        e = self.exp_free(s - m) if getattr(self, 'EXP_FREE', False) else self.exp_neg(s - m)[0]
        e = e * mask
        return e * self.recip_den(e.sum(dim=-1, keepdim=True), s.shape[-1])


def _dp_aggregate(self, gps, C, tau, noise=None):
    """Secure DP-SGD aggregation on per-sample grads gps (16 secret tensors [B, ...], true scale).
    Per sample: drop it obliviously if any |entry| > tau (truncation failure), else clip to L2 norm C
    (scale = C * rsqrt(max(||g||^2, C^2))). Sum, add the dealer's Gaussian noise share (if given), divide by B.
    Nothing is revealed."""
    B = gps[0].shape[0]
    shp = [g.shape[1:] for g in gps]; num = [g[0].numel() for g in gps]
    flat = crypten.cat([g.reshape(B, -1) for g in gps], dim=1)              # [B, P]
    gt = V.cmp_lt_batch(flat, [-tau, tau])
    nbad = (gt[0] + neg1p(gt[1])).sum(dim=1)                                # [B] integer
    okb = V.cmp_lt_batch(to_fx(nbad, flat), [0.5])[0]                       # [B] = [no bad entry]
    n2 = (flat * flat).sum(dim=1)                                           # [B] squared norms
    lt = V.cmp_lt_batch(n2, [C * C])[0]
    m = n2 + lt * ((n2 * 0 + C * C) - n2)                                   # max(n2, C^2)
    scale = (self.rsqrt_newton(m) * C) * okb                                # min(1, C/||g||), 0 if bad
    agg = (flat * scale.unsqueeze(1)).sum(dim=0)                            # [P]
    if noise is not None:
        agg = agg + noise
    agg = agg * (1.0 / B)
    out, o = [], 0
    for sh, n_ in zip(shp, num):
        out.append(agg[o:o + n_].reshape(sh)); o += n_
    return out, okb, n2


SecureOpsBP2.dp_aggregate = _dp_aggregate


# ---- LN with pre-scaled variance (outlier-aware) ----
# RoBERTa-large has residual-stream outliers (|x - mu| up to ~1.5e3 on some non-MASK rows in layers 20-22). The square
# d*d (~2e6) is the largest product in the network, and the local-truncation failure probability grows linearly with
# the product's magnitude (~|x| 2^-31). Squaring d*LN_SC instead (public scale, exact rescale of r afterwards) cuts that
# probability by 1/LN_SC^2 on exactly those entries, at a precision cost of log2(1/LN_SC) bits on var only.
def _ln_save_scaled(self, x, g, b):
    sc = float(getattr(self, 'LN_SC', 0.25))
    if sc >= 1.0:
        return BPm.SecureOpsBP.ln_save(self, x, g, b)
    inv = const(1.0 / x.shape[-1])
    d = x - x.sum(dim=-1, keepdim=True) * inv
    ds = d * const(sc)
    var_s = (ds * ds).sum(dim=-1, keepdim=True) * inv + getattr(self, 'LN_EPS', 0.0) * (sc * sc)
    r = self.rsqrt_newton(var_s) * const(sc)                          # 1/sqrt(var) = sc / sqrt(sc^2 var)
    n = d * r
    return n * g + b, (n, r)


SecureOpsBP2.ln_save = _ln_save_scaled
