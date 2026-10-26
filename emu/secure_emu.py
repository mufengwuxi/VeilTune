"""Emulated secure blocks (U-split): 4 BERT layers, MPC kernels of sjvp/model.py, every truncation
point passed through FXP.q, forward-mode (primal, tangent) carried explicitly, mirroring the JAX
custom JVPs that SPU executes. Batched: x [B,T,256] with per-sample mask [B,T], pos [B]."""
import torch
C_INV = [0.03226120897959244, -0.049780065730235586, 0.057650254381631766, -0.08583010236400039,
         0.1317100845857373, -0.19765769464931107, 0.29629765286612486, -0.44444230412031277,
         0.6666666475671728]

class Emu:
    def __init__(self, W, fx):
        self.fx = fx
        self.W = {k: fx.enc(v) for k, v in W.items()}  # model owner's weights, encoded once
        self.cinv = [fx.const(c) for c in C_INV]

    # ---- kernels: each returns (y, y_dot) ; y_dot None when no tangent ----
    def exp_neg(self, x, xt=None):
        q = self.fx.q
        z = q(torch.clamp(x, -12.0, 0.0) * (1.0 / 16))
        acc = torch.ones_like(z)
        for k in range(8, 0, -1):
            acc = 1.0 + q(q(z * acc) * self.fx.const(1.0 / k))
        for _ in range(4):
            acc = q(acc * acc)
        return acc, (None if xt is None else q(acc * xt))

    def inv_1p(self, e):
        q = self.fx.q
        t = e - 0.5
        acc = q(self.cinv[0] * t) + self.cinv[1]
        for c in self.cinv[2:]:
            acc = q(acc * t) + c
        return acc

    def gelu(self, z, zt=None):
        q, c = self.fx.q, self.fx.const
        s = q(c(1.5957691216057308) * (z + q(q(q(c(0.044715) * z) * z) * z)))
        pos = s >= 0
        a = torch.where(pos, s, -s)
        e, _ = self.exp_neg(-a)
        qq = self.inv_1p(e)
        y = torch.where(pos, qq, 1.0 - qq)
        out = q(z * y)
        if zt is None:
            return out, None
        dsig = torch.where(a < 12.0, q(q(e * qq) * qq), torch.zeros_like(a))
        ds = q(c(1.5957691216057308) * (1.0 + q(q(c(0.134145) * z) * z)))
        return out, q((y + q(q(z * dsig) * ds)) * zt)

    def softmax(self, s, st, mask):
        q = self.fx.q
        s = s + (mask - 1.0) * 30.0
        m = s.max(-1, keepdim=True).values
        e, _ = self.exp_neg(s - m)
        e = e * mask
        p = q(e * self.fx.recip(e.sum(-1, keepdim=True)))
        if st is None:
            return p, None
        return p, q(p * (st - q(p * st).sum(-1, keepdim=True)))

    def ln(self, x, xt, g, b):
        q = self.fx.q
        d = x - q(x.sum(-1, keepdim=True) * (1.0 / 256))          # LN_PRE = 1 (f16 config)
        var = q(q(d * d).sum(-1, keepdim=True) * (1.0 / 256))
        r = self.fx.rsqrt(var)
        n = q(d * r)
        y = q(n * g) + b
        if xt is None:
            return y, None
        dd = xt - q(xt.sum(-1, keepdim=True) * (1.0 / 256))
        mm = q(q(n * dd).sum(-1, keepdim=True) * (1.0 / 256))
        nt = q(r * (dd - q(n * mm)))
        return y, q(nt * g)

    def mm(self, a, b):
        return self.fx.q(a @ b)

    # ---- network ----
    def encoder(self, h0, mask, pos, theta, u=None):
        """theta/u: list of 16 tensors (A[4,256], B[256,4] for Q,V per layer). Returns h, h_dot at MASK."""
        fx, W, mm = self.fx, self.W, self.mm
        x = fx.enc(h0); xt = torch.zeros_like(x) if u is not None else None
        th = [fx.enc(t) for t in theta]; ut = None if u is None else [fx.enc(v) for v in u]
        B, T, _ = x.shape
        mk = mask[:, None, None, :]
        ar = torch.arange(B, device=x.device)
        for i in range(4):
            w = lambda k: W[f'w{i}_{k}']
            Aq, Bq, Av, Bv = th[4 * i: 4 * i + 4]

            def lora_lin(name, A, Bm, j):
                base = mm(x, w(name)) + w(name + 'b')
                inner = mm(x, A.T); y = base + mm(inner, Bm.T) * 4.0
                if xt is None:
                    return y, None
                dA, dB = ut[4 * i + j], ut[4 * i + j + 1]
                inner_t = mm(xt, A.T) + mm(x, dA.T)
                yt = mm(xt, w(name)) + (mm(inner_t, Bm.T) + mm(inner, dB.T)) * 4.0
                return y, yt
            q_, qt = lora_lin('q', Aq, Bq, 0)
            v_, vt = lora_lin('v', Av, Bv, 2)
            k_ = mm(x, w('k')) + w('kb'); kt = None if xt is None else mm(xt, w('k'))
            last = i == 3
            if last:  # only MASK row of queries / residual stream
                sel = lambda a: a[ar, pos][:, None]
                x = sel(x); q_ = sel(q_)
                if xt is not None:
                    xt = sel(xt); qt = sel(qt)
            Tq = x.shape[1]
            hd = lambda a, n: a.reshape(B, n, 4, 64).transpose(1, 2)
            Q, K, V = hd(q_, Tq), hd(k_, T), hd(v_, T)
            S = mm(Q, K.transpose(-1, -2))
            St = None if xt is None else mm(hd(qt, Tq), K.transpose(-1, -2)) + mm(Q, hd(kt, T).transpose(-1, -2))
            P, Pt = self.softmax(S, St, mk)
            att = mm(P, V).transpose(1, 2).reshape(B, Tq, 256)
            att_t = None if xt is None else (mm(Pt, V) + mm(P, hd(vt, T))).transpose(1, 2).reshape(B, Tq, 256)
            pre1 = x + mm(att, w('o')) + w('ob')
            pre1t = None if xt is None else xt + mm(att_t, w('o'))
            h, ht = self.ln(pre1, pre1t, w('ln1'), w('ln1b'))
            z = mm(h, w('ff1')) + w('ff1b'); zt = None if xt is None else mm(ht, w('ff1'))
            g, gt = self.gelu(z, zt)
            pre2 = h + mm(g, w('ff2')) + w('ff2b')
            pre2t = None if xt is None else ht + mm(gt, w('ff2'))
            x, xt = self.ln(pre2, pre2t, w('ln2'), w('ln2b'))
        return x[:, 0], (None if xt is None else xt[:, 0])
