"""gpu_crypten_v2: round/byte-reduced CrypTen secure-JVP encoder (stage 2, phase A).

Same model, LoRA layout, analytic tangents and f=16 as gpu_crypten.py; same security
(theta, u, all activations secret; W, biases, mask, positions public). Changes are in how the
non-linear kernels are evaluated and in removing padding:

  1. Batched comparisons: all comparisons that do not depend on each other are evaluated in ONE
     CrypTen comparison call (one A2B+B2A pass, 8 rounds) on a stacked tensor.
  2. rsqrt (LayerNorm): octave ladder (33 thresholds, one batched comparison). Selection is
     one-hot s_k = g_k - g_{k-1} (gates are monotone in the threshold) -> no sequential "decided"
     multiplications. u = sum_k s_k * (v * 2^-k) (powers of two: exact shifts, one round),
     linear initial guess on u in [1/2,1) (max rel err 2.2%), 2 Newton steps written as
     t' = 1.5 t - 0.5 (t^2)(u t) (2 rounds each), r = sum_k s_k * (t * 2^-k/2) (one round).
     ~14 rounds instead of ~197.
  3. Softmax denominator reciprocal: denominator lies in [~1, T]; octave ladder (one batched
     comparison) + 48/17-32/17 w initial guess + 3 Newton steps. ~16 rounds instead of 38.
  4. GELU: the three comparisons (sign, |s|<12 for the sigma' gate, exp clamp) come from one
     batched comparison on [s, s-12, s+12]; the exp clamp gate is derived (1 - inner), so
     exp_neg does no comparison of its own inside GELU. z^3 is reused for z*ds (linear).
  5. Polynomials (exp Taylor degree 8, inv_1p degree 9) evaluated with Estrin-style power
     ladders (all products of one depth in a single stacked multiplication): 3 / 4 rounds
     instead of 8 sequential Horner multiplications.
  6. Packing: token-wise ops (projections, LayerNorm, FFN, GELU) run on the real tokens only
     ([N_real, 256]); attention gathers per-sample padded Q/K/V with public indices (local),
     masked exactly as before, and scatters back. The last layer keeps only the MASK rows.

Gates returned by CrypTen comparisons are integer 0/1 with encoder scale 1; they are only ever
multiplied by secrets (no truncation) or combined with integers, never with fractional public
constants (CrypTen would encode those at scale 1).
"""
import math
import torch
import crypten

import gpu_crypten as G
from gpu_crypten import const

ARITH = None

RS_A, RS_B = 1.787700, 0.809950           # rsqrt initial guess on [1/2,1): max rel err 2.23%
RC_A, RC_B = 48.0 / 17.0, 32.0 / 17.0     # reciprocal initial guess on [1/2,1): max rel err 1/17
RS_K = list(range(-14, int(__import__('os').environ.get('RS_KMAX', '18')) + 1))   # rsqrt ladder 2^k, k = -14..RS_KMAX (default 18; 26 for Adam scale 8192)
EXP_TAYLOR = [1.0 / math.factorial(k) for k in range(9)]


def _gate(x):
    return (x < 0).to(ARITH)


def cmp_lt_batch(x, thresholds):
    """[x < t for t in thresholds] in ONE comparison call; returns stacked gates [K, *x.shape]."""
    return _gate(crypten.stack([x - float(t) for t in thresholds]))


def neg1p(x):
    """1 - x without relying on __rsub__ (exact for integer gates)."""
    return x * -1 + 1


def to_fx(g, ref):
    """Re-encode an integer 0/1 gate (scale 1) at ref's fixed-point scale (exact, local)."""
    r = g.clone()
    r._tensor.encode_as_(ref._tensor)
    return r


def shift(x, k):
    """x * 2^k for integer k, exact for k >= 0 (integer multiply), one truncation for k < 0."""
    if k >= 0:
        return x * (1 << k)
    return x.div(1 << (-k))


def powers(x, n):
    """[x^1..x^n]; every depth is one stacked multiplication (ceil(log2 n) rounds)."""
    p = {1: x}
    have = 1
    while have < n:
        top = p[have]
        lo = list(range(1, min(have, n - have) + 1))
        prod = crypten.stack([p[i] for i in lo]) * top.unsqueeze(0)
        for j, i in enumerate(lo):
            p[have + i] = prod[j]
        have = max(p)
    return [p[i] for i in range(1, n + 1)]


class SecureOpsV2(G.SecureOps):

    # ---- exp(x), x <= 0: clamp at -12, Taylor deg 8 on x/16, 4 squarings ----
    def exp_neg(self, x, xt=None, clamp_gate=None):
        if clamp_gate is None:
            clamp_gate = _gate(x + 12.0)                 # [x < -12]
        xc = x + clamp_gate * ((x * 0 - 12.0) - x)       # select, 1 round
        z = xc * const(1.0 / 16.0)
        pw = powers(z, 8)                                # 3 rounds
        acc = x * 0 + 1.0
        for k in range(1, 9):
            acc = acc + pw[k - 1] * const(EXP_TAYLOR[k])
        for _ in range(4):
            acc = acc * acc
        return acc, (None if xt is None else acc * xt)

    # ---- 1/(1+e) polynomial: identical polynomial to v1's Horner form ----
    def inv_1p(self, e):
        # v1: acc0 = c0 t; acc_j = acc_{j-1} t + c_j (j=1..8)
        #   => p(t) = c0 t^9 + c1 t^7 + c2 t^6 + ... + c7 t + c8
        t = e - 0.5
        pw = powers(t, 9)                                # 4 rounds
        c = self.cinv
        acc = pw[8] * c[0] + c[8]
        for j in range(1, 8):
            acc = acc + pw[8 - j - 1] * c[j]             # c_j * t^(8-j)
        return acc

    def gelu(self, z, zt=None):
        k044, k159 = const(0.044715), const(1.5957691216057308)
        z2 = z * z
        z3 = z2 * z
        s = (z + z3 * k044) * k159
        g = cmp_lt_batch(s, [0.0, 12.0, -12.0])          # one comparison call
        g0, g12, gm12 = g[0], g[1], g[2]                 # [s<0], [s<12], [s<-12]
        inner = g12 - gm12                               # [-12 <= s < 12]
        a = s - (s * g0) * 2                             # |s|, 1 round
        e, _ = self.exp_neg(a * -1, clamp_gate=neg1p(inner))
        qq = self.inv_1p(e)
        pos = neg1p(g0)
        y = (qq * -1 + 1.0) + pos * (qq * 2 - 1.0)       # sigmoid(s), 1 round
        out = z * y
        if zt is None:
            return out, None
        pr = crypten.stack([e, qq]) * crypten.stack([to_fx(inner, qq), qq])   # [e*inner, qq^2], 1 round
        dsig = pr[0] * pr[1]                             # e*qq^2 gated, 1 round
        zds = (z + z3 * const(0.134145)) * k159          # z * ds, linear (z^3 reused)
        return out, (y + dsig * zds) * zt

    # ---- 1/sqrt(v), v > 0 ----
    def rsqrt_newton(self, v):
        K = RS_K
        g = cmp_lt_batch(v, [2.0 ** k for k in K])       # g_j = [v < 2^K_j], monotone in j
        sel = [g[0]] + [g[j] - g[j - 1] for j in range(1, len(K))] + [neg1p(g[len(K) - 1])]
        # sel[0]: v < 2^K0 (scale 2^-K0); sel[j]: 2^K_{j-1} <= v < 2^K_j (scale 2^-K_j, u in [1/2,1));
        # sel[-1]: v >= 2^K_last (scale 2^-(K_last+1))
        exps = K + [K[-1] + 1]
        S = crypten.stack(sel)
        U = crypten.stack([shift(v, -e) for e in exps])
        u = (S * U).sum(dim=0)                           # 1 round
        t = (u * -RS_B) + RS_A                           # linear initial guess
        for _ in range(2):
            pr = crypten.stack([t, u]) * t.unsqueeze(0)  # [t^2, u t], 1 round
            t = t * 1.5 - (pr[0] * pr[1]) * 0.5          # 1 round
        # r = t * 2^(-e/2): even e exact shift; odd e via sqrt(1/2)
        R = []
        tt = t * const(math.sqrt(0.5))
        for e in exps:
            R.append(shift(t, -e // 2) if e % 2 == 0 else shift(tt, -(e - 1) // 2))
        return (S * crypten.stack(R)).sum(dim=0)         # 1 round

    # ---- reciprocal for softmax denominators in [~1, T] ----
    def recip_den(self, x, tmax):
        K = list(range(1, int(math.ceil(math.log2(max(tmax, 2)))) + 1))
        g = cmp_lt_batch(x, [2.0 ** k for k in K])
        sel = [g[0]] + [g[j] - g[j - 1] for j in range(1, len(K))] + [neg1p(g[len(K) - 1])]
        exps = K + [K[-1] + 1]                           # region j scaled by 2^-exps[j]
        S = crypten.stack(sel)
        w = (S * crypten.stack([shift(x, -e) for e in exps])).sum(dim=0)   # w in ~[1/2,1)
        y = (w * -RC_B) + RC_A
        for _ in range(3):
            y = y * ((w * y) * -1 + 2.0)                 # 2 rounds each
        return (S * crypten.stack([shift(y, -e) for e in exps])).sum(dim=0)

    def softmax(self, s, st, mask):
        s = s + (mask - 1.0) * 30.0
        m = s.max(dim=-1, keepdim=True)[0]
        e, _ = self.exp_neg(s - m)
        e = e * mask
        p = e * self.recip_den(e.sum(dim=-1, keepdim=True), s.shape[-1])
        if st is None:
            return p, None
        return p, p * (st - (p * st).sum(dim=-1, keepdim=True))

    # ---- packed encoder ----
    def encoder_packed(self, h0p, lens, pos, theta, u=None):
        """h0p: [N, 256] public-shape plaintext (client input, secret-shared from party 0) holding
        the real tokens of all samples back to back; lens: list of per-sample valid lengths;
        pos: per-sample MASK index (public).
        Returns (h, h_dot) [B, 256] at the MASK rows."""
        W, mm = self.W, lambda a, b: a @ b
        dev = 'cuda'
        B, T = len(lens), max(lens)
        N = int(sum(lens))
        offs = [0]
        for l in lens[:-1]:
            offs.append(offs[-1] + l)
        # public index maps
        pad_idx = torch.zeros(B * T, dtype=torch.long, device=dev)       # padded slot -> packed row
        mask = torch.zeros(B, T, dtype=torch.float64, device=dev)
        for b, l in enumerate(lens):
            pad_idx[b * T: b * T + l] = torch.arange(offs[b], offs[b] + l, device=dev)
            mask[b, :l] = 1.0
        pack_idx = torch.cat([torch.arange(b * T, b * T + l, device=dev) for b, l in enumerate(lens)])
        mask_rows = torch.tensor([offs[b] + int(pos[b]) for b in range(B)], device=dev)
        mk = mask[:, None, None, :]

        x = crypten.cryptensor(h0p, device=dev)
        xt = crypten.cryptensor(torch.zeros_like(h0p), device=dev) if u is not None else None

        def to_pad(a):                                   # [N,256] -> [B,T,256] (local gather)
            return a.index_select(0, pad_idx).reshape(B, T, 256)

        def from_pad(a, n):                              # [B,n,256] -> [N or B,256]
            flat = a.reshape(B * n, 256)
            return flat.index_select(0, pack_idx) if n == T else flat

        for i in range(4):
            w = lambda k: W[f'w{i}_{k}']
            Aq, Bq, Av, Bv = theta[4 * i:4 * i + 4]

            def lora_lin(name, A, Bm, j, xx, xxt):
                base = mm(xx, w(name)) + w(name + 'b')
                inner = mm(xx, A.transpose(0, 1))
                y = base + mm(inner, Bm.transpose(0, 1)) * 4.0
                if xxt is None:
                    return y, None
                dA, dB = u[4 * i + j], u[4 * i + j + 1]
                inner_t = mm(xxt, A.transpose(0, 1)) + mm(xx, dA.transpose(0, 1))
                yt = mm(xxt, w(name)) + (mm(inner_t, Bm.transpose(0, 1)) + mm(inner, dB.transpose(0, 1))) * 4.0
                return y, yt

            last = i == 3
            v_, vt = lora_lin('v', Av, Bv, 2, x, xt)
            k_ = mm(x, w('k')) + w('kb')
            kt = None if xt is None else mm(xt, w('k'))
            if last:                                     # only MASK rows go on
                x = x.index_select(0, mask_rows)
                if xt is not None:
                    xt = xt.index_select(0, mask_rows)
            q_, qt = lora_lin('q', Aq, Bq, 0, x, xt)
            Tq = 1 if last else T
            hd = lambda a, n: a.reshape(B, n, 4, 64).transpose(1, 2)
            Qp = q_.reshape(B, 1, 256) if last else to_pad(q_)
            Q, K, V = hd(Qp, Tq), hd(to_pad(k_), T), hd(to_pad(v_), T)
            S = mm(Q, K.transpose(-1, -2))
            St = None
            if xt is not None:
                Qtp = qt.reshape(B, 1, 256) if last else to_pad(qt)
                St = mm(hd(Qtp, Tq), K.transpose(-1, -2)) + mm(Q, hd(to_pad(kt), T).transpose(-1, -2))
            P, Pt = self.softmax(S, St, mk)
            att = from_pad(mm(P, V).transpose(1, 2).reshape(B, Tq, 256), Tq)
            att_t = None if xt is None else from_pad(
                (mm(Pt, V) + mm(P, hd(to_pad(vt), T))).transpose(1, 2).reshape(B, Tq, 256), Tq)
            pre1 = x + mm(att, w('o')) + w('ob')
            pre1t = None if xt is None else xt + mm(att_t, w('o'))
            h, ht = self.ln(pre1, pre1t, w('ln1'), w('ln1b'))
            z = mm(h, w('ff1')) + w('ff1b')
            zt = None if xt is None else mm(ht, w('ff1'))
            g, gt = self.gelu(z, zt)
            pre2 = h + mm(g, w('ff2')) + w('ff2b')
            pre2t = None if xt is None else ht + mm(gt, w('ff2'))
            x, xt = self.ln(pre2, pre2t, w('ln2'), w('ln2b'))
        return x, xt


def pack(h0, mask):
    """[B,T,256] padded plaintext + [B,T] mask -> ([N,256] packed, lens)."""
    lens = [int(m.sum()) for m in mask]
    rows = [h0[b, :l] for b, l in enumerate(lens)]
    return torch.cat(rows, 0), lens


# ---------------------------------------------------------------------------------------------
# Fully secure tail (system model A, no reveal): MLM head + 2-class cross-entropy + directional
# derivative c, all on secret shares. The client inputs its labels as secrets; nothing is revealed
# to anyone during training (c stays secret and feeds the secure update directly).
# ---------------------------------------------------------------------------------------------
def _sigmoid(self, s):
    """sigma(s) via the same exp/inv_1p kernels as GELU; one batched comparison on [s, s-12, s+12]."""
    g = cmp_lt_batch(s, [0.0, 12.0, -12.0])
    g0, inner = g[0], g[1] - g[2]
    a = s - (s * g0) * 2                                  # |s|
    e, _ = self.exp_neg(a * -1, clamp_gate=neg1p(inner))
    qq = self.inv_1p(e)                                   # 1/(1+e^{-|s|})
    return (qq * -1 + 1.0) + neg1p(g0) * (qq * 2 - 1.0)


def _init_head(self, H):
    """H: plaintext head weights (client_c layout: z = h@W + b; logits = n@dec + decb)."""
    enc = lambda k: G.enc_pub(H[k]).cuda()
    self.hW, self.hb, self.hg, self.hbe = enc('head_W'), enc('head_b'), enc('head_g'), enc('head_be')
    dec = torch.as_tensor(H['head_dec'], dtype=torch.float64)
    decb = torch.as_tensor(H['head_decb'], dtype=torch.float64)
    self.hdd = G.enc_pub((dec[:, 1] - dec[:, 0]).numpy()).cuda()          # logit difference d = s1 - s0
    self.hdb = float(G.enc_pub(np.array(float(decb[1] - decb[0]))))


def _secure_c(self, h, hd, y):
    """h, hd: secret [B, D] at the MASK rows; y: secret {0,1} [B]. Returns secret scalar
    c = (1/B) sum_b (sigma(d_b) - y_b) * d_dot_b  (= dL/dtheta . u for the batch-mean CE loss)."""
    z = h @ self.hW + self.hb
    zt = hd @ self.hW
    g, gt = self.gelu(z, zt)
    n, nt = self.ln(g, gt, self.hg, self.hbe)
    d = n @ self.hdd + self.hdb                           # [B]
    dt = nt @ self.hdd
    p1 = self._sigmoid(d)
    return ((p1 - y) * dt).sum() * (1.0 / h.shape[0])


import numpy as np  # noqa: E402
SecureOpsV2._sigmoid = _sigmoid
SecureOpsV2.init_head = _init_head
SecureOpsV2.secure_c = _secure_c
