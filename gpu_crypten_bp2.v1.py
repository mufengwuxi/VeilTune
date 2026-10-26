"""Communication-reduced secure backprop (bp2) = gpu_crypten_bp with two kernels replaced.

1) GELU (exact-erf target, not the tanh form). gelu(z) = z/2 + E(z), E even.
   |z| < c (c = 4): E(z) = R(u), gelu'(z) - 1/2 = z * S(u), u = z^2 / c^2, Horner (R deg 7, S deg 6, stacked).
   |z| >= c       : gelu = relu (error <= 1.3e-4), gelu' = [z > 0].
   Gates: [z < 0] and [z^2 < c^2] in ONE batched comparison (2 comparisons / element instead of 3), no exp,
   no 1/(1+e) polynomial. Plaintext fixed-point emulation: max |err| 2.2e-4 (gelu), 1.2e-3 (gelu').
2) Row max for softmax: own binary tournament (1 comparison + 1 multiplication per eliminated element,
   max value only) instead of CrypTen's max(dim) (log-reduction + pairwise + one-hot argmax).
"""
import crypten

import gpu_crypten_bp as BPm
from gpu_crypten_v2 import _gate, neg1p, to_fx
from gpu_crypten import const

GC = 4.0
GA = [0.000165592118, 6.35914196, -16.4498848, 35.5286104, -52.3277043, 48.9141622, -25.8863215, 5.86186901]
GB = [0.793507332, -4.06499135, 12.8734628, -24.3811859, 27.1982983, -16.3942397, 4.10054477]


class SecureOpsBP2(BPm.SecureOpsBP):

    def gelu_save(self, z):
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
        m = self.max_tree(s)
        e, _ = self.exp_neg(s - m)
        e = e * mask
        return e * self.recip_den(e.sum(dim=-1, keepdim=True), s.shape[-1])
