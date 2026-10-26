"""Secure backpropagation (VJP) for the v2 packed encoder + secure head, CrypTen 2PC.

Same model / kernels / packing as gpu_crypten_v2 (forward is the v2 primal, no tangent). The forward
saves, per layer: inputs, LoRA inner products, per-head Q/K/V, softmax P, LayerNorm (n, r) and
gelu'(z). Backward reuses those (no new comparisons): LN backward 3 multiplications deep, softmax
backward 2, GELU backward 1, secret x secret matmuls for attention and LoRA; products with public
weights are local. Output: secret gradients for all 16 LoRA tensors (batch-mean CE loss).
"""
import torch
import crypten

import gpu_crypten_v2 as V
from gpu_crypten_v2 import cmp_lt_batch, neg1p, to_fx
from gpu_crypten import const


class SecureOpsBP(V.SecureOpsV2):

    # ---------- forward pieces that keep what backward needs ----------
    def gelu_save(self, z):
        k044, k159 = const(0.044715), const(1.5957691216057308)
        z2 = z * z
        z3 = z2 * z
        s = (z + z3 * k044) * k159
        g = cmp_lt_batch(s, [0.0, 12.0, -12.0])
        g0, inner = g[0], g[1] - g[2]
        a = s - (s * g0) * 2
        e, _ = self.exp_neg(a * -1, clamp_gate=neg1p(inner))
        qq = self.inv_1p(e)
        y = (qq * -1 + 1.0) + neg1p(g0) * (qq * 2 - 1.0)
        out = z * y
        pr = crypten.stack([e, qq]) * crypten.stack([to_fx(inner, qq), qq])
        dsig = pr[0] * pr[1]
        zds = (z + z3 * const(0.134145)) * k159
        gp = y + dsig * zds                               # gelu'(z)
        return out, gp

    def ln_save(self, x, g, b):
        inv = const(1.0 / x.shape[-1])
        d = x - x.sum(dim=-1, keepdim=True) * inv
        var = (d * d).sum(dim=-1, keepdim=True) * inv + getattr(self, 'LN_EPS', 0.0)   # eps public (RoBERTa 1e-5)
        r = self.rsqrt_newton(var)
        n = d * r
        return n * g + b, (n, r)

    def softmax_save(self, s, mask):
        s = s + (mask - 1.0) * 30.0
        m = s.max(dim=-1, keepdim=True)[0]
        e, _ = self.exp_neg(s - m)
        e = e * mask
        return e * self.recip_den(e.sum(dim=-1, keepdim=True), s.shape[-1])

    # ---------- backward pieces ----------
    @staticmethod
    def ln_bwd(gy, g, saved):
        n, r = saved
        inv = const(1.0 / gy.shape[-1])
        gn = gy * g
        m1 = gn.sum(dim=-1, keepdim=True) * inv
        m2 = (gn * n).sum(dim=-1, keepdim=True) * inv
        return r * (gn - m1 - n * m2)

    @staticmethod
    def softmax_bwd(gP, P):
        return P * (gP - (P * gP).sum(dim=-1, keepdim=True))

    # ---------- full forward + backward ----------
    def loss_grad(self, h0p, lens, pos, theta, y):
        """Returns secret grads [16 tensors] of the batch-mean CE loss w.r.t. theta."""
        W, mm = self.W, lambda a, b: a @ b
        dev = 'cuda'
        B, T = len(lens), max(lens)
        offs = [0]
        for l in lens[:-1]:
            offs.append(offs[-1] + l)
        N = int(sum(lens))
        pad_idx = torch.zeros(B * T, dtype=torch.long, device=dev)
        mask = torch.zeros(B, T, dtype=torch.float64, device=dev)
        for b, l in enumerate(lens):
            pad_idx[b * T: b * T + l] = torch.arange(offs[b], offs[b] + l, device=dev)
            mask[b, :l] = 1.0
        pack_idx = torch.cat([torch.arange(b * T, b * T + l, device=dev) for b, l in enumerate(lens)])
        mask_rows = torch.tensor([offs[b] + int(pos[b]) for b in range(B)], device=dev)
        mk = mask[:, None, None, :]
        padmask = mask[:, :, None]                                   # zero the padded query slots in backward
        onehotT = torch.zeros(B, N, dtype=torch.float64, device=dev)
        onehotT[torch.arange(B, device=dev), mask_rows] = 1.0

        to_pad = lambda a: a.index_select(0, pad_idx).reshape(B, T, a.shape[-1])
        from_pad = lambda a, n: (a.reshape(B * n, a.shape[-1]).index_select(0, pack_idx) if n == T
                                 else a.reshape(B * n, a.shape[-1]))
        D, NH = getattr(self, 'D', 256), getattr(self, 'NH', 4)         # model width / heads (BERT-mini default)
        hd = lambda a, n: a.reshape(B, n, NH, D // NH).transpose(1, 2)
        uhd = lambda a, n: a.transpose(1, 2).reshape(B, n, D)

        x = crypten.cryptensor(h0p, device=dev)
        saved = []
        L = len(theta) // 4                                          # number of LoRA (secure) layers
        for i in range(L):
            w = lambda k, i=i: W[f'w{i}_{k}']
            Aq, Bq, Av, Bv = theta[4 * i:4 * i + 4]
            last = (i == L - 1) and getattr(self, 'MASK_ONLY', True)   # ablation: MASK-row-only last layer
            inner_v = mm(x, Av.transpose(0, 1))
            v_ = mm(x, w('v')) + w('vb') + mm(inner_v, Bv.transpose(0, 1)) * 4.0
            k_ = mm(x, w('k')) + w('kb')
            xq = x.index_select(0, mask_rows) if last else x
            inner_q = mm(xq, Aq.transpose(0, 1))
            q_ = mm(xq, w('q')) + w('qb') + mm(inner_q, Bq.transpose(0, 1)) * 4.0
            Tq = 1 if last else T
            Q = hd(q_.reshape(B, 1, D) if last else to_pad(q_), Tq)
            K, Vh = hd(to_pad(k_), T), hd(to_pad(v_), T)
            P = self.softmax_save(mm(Q, K.transpose(-1, -2)), mk)
            att = from_pad(uhd(mm(P, Vh), Tq), Tq)
            hmid, ln1 = self.ln_save(xq + mm(att, w('o')) + w('ob'), w('ln1'), w('ln1b'))
            gz_out, gp = self.gelu_save(mm(hmid, w('ff1')) + w('ff1b'))
            xn, ln2 = self.ln_save(hmid + mm(gz_out, w('ff2')) + w('ff2b'), w('ln2'), w('ln2b'))
            saved.append(dict(x=x, xq=xq, inner_v=inner_v, inner_q=inner_q, Q=Q, K=K, V=Vh, P=P,
                              ln1=ln1, gp=gp, ln2=ln2, Tq=Tq))
            x = xn
        # ---- secure head forward + backward to h ----
        if not getattr(self, 'MASK_ONLY', True):
            x = x.index_select(0, mask_rows)                           # head needs MASK rows only
        zh = x @ self.hW + self.hb
        gh_out, gph = self.gelu_save(zh)
        nh, lnh = self.ln_save(gh_out, self.hg, self.hbe)
        d = nh @ self.hdd + self.hdb
        p1 = self._sigmoid(d)
        if getattr(self, 'FWD_ONLY', False):                          # cost baseline: forward + loss only (ZO-style)
            return [p1]
        LS = float(getattr(self, 'LS', 1.0))                         # loss scaling (fixed-point resolution)
        PS = bool(getattr(self, 'PER_SAMPLE', False))                # per-sample grads (DP-SGD): no 1/B here
        gd = (p1 - y) * (LS if PS else LS / B)                        # LS * dL/dd

        def seg(a, b, Tq_):                                           # per-sample sum_rows a^T b -> [B, da, db]
            if Tq_ == 1:                                              # one row per sample (last-layer MASK rows)
                return a.unsqueeze(2) * b.unsqueeze(1)
            return (to_pad(a) * padmask).transpose(1, 2) @ to_pad(b)
        gnh = gd.unsqueeze(1) * self.hdd                               # [B, D] (public vector)
        gg = self.ln_bwd(gnh, self.hg, lnh)
        gx = (gg * gph) @ self.hW.transpose(0, 1)
        if not getattr(self, 'MASK_ONLY', True):
            gx = (gx.transpose(0, 1) @ onehotT).transpose(0, 1)        # scatter back to all N rows (public map)
        # ---- encoder backward ----
        grads = [None] * (4 * L)
        for i in reversed(range(L)):
            w = lambda k, i=i: W[f'w{i}_{k}']
            S = saved[i]
            Aq, Bq, Av, Bv = theta[4 * i:4 * i + 4]
            Tq, last = S['Tq'], (i == L - 1) and getattr(self, 'MASK_ONLY', True)
            g_pre2 = self.ln_bwd(gx, w('ln2'), S['ln2'])
            g_h = g_pre2 + ((g_pre2 @ w('ff2').transpose(0, 1)) * S['gp']) @ w('ff1').transpose(0, 1)
            g_pre1 = self.ln_bwd(g_h, w('ln1'), S['ln1'])
            g_xq = g_pre1
            g_att = g_pre1 @ w('o').transpose(0, 1)
            gA = hd(g_att.reshape(B, 1, D) if last else to_pad(g_att) * padmask, Tq)
            gP = mm(gA, S['V'].transpose(-1, -2))
            gV = mm(S['P'].transpose(-1, -2), gA)
            gS = self.softmax_bwd(gP, S['P'])
            gQ = mm(gS, S['K'])
            gK = mm(gS.transpose(-1, -2), S['Q'])
            g_q = from_pad(uhd(gQ, Tq), Tq)                          # [Nq, D]
            g_k = from_pad(uhd(gK, T), T)                            # [N, D]
            g_v = from_pad(uhd(gV, T), T)
            # LoRA on q (input xq) and v (input x)
            gin_q = (g_q @ Bq) * 4.0
            gin_v = (g_v @ Bv) * 4.0
            if PS:
                gBq = seg(g_q, S['inner_q'], Tq) * 4.0
                gAq = seg(gin_q, S['xq'], Tq)
                gBv = seg(g_v, S['inner_v'], T) * 4.0
                gAv = seg(gin_v, S['x'], T)
            else:
                gBq = (g_q.transpose(0, 1) @ S['inner_q']) * 4.0
                gAq = gin_q.transpose(0, 1) @ S['xq']
                gBv = (g_v.transpose(0, 1) @ S['inner_v']) * 4.0
                gAv = gin_v.transpose(0, 1) @ S['x']
            grads[4 * i:4 * i + 4] = [gAq, gBq, gAv, gBv]
            if i == 0:
                break
            g_xq = g_xq + g_q @ w('q').transpose(0, 1) + gin_q @ Aq
            g_x = g_k @ w('k').transpose(0, 1) + g_v @ w('v').transpose(0, 1) + gin_v @ Av
            if last:
                g_x = g_x + (g_xq.transpose(0, 1) @ onehotT).transpose(0, 1)
            else:
                g_x = g_x + g_xq
            gx = g_x
        if LS != 1.0:
            grads = [g * (1.0 / LS) for g in grads]
        return grads
