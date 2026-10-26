"""secure_emu_sigma: copy of emu/secure_emu.py with the SIGMA pre-check changes.

Per FSS_PROBE_SPEC step 4 (originals untouched):
  - GELU tangent replaced by a SIGMA-quantized gelu' lookup table:
    input index = clamp(round(z*2^6), -255, 255)  (TClip=u8, clipBw=8,
    Xt = truncate(x, scale-6) with scale=12 -- source: fss/gpu_gelu.cu
    gpuGelu/gpuKeyGenGelu, backend/sigma.h:149 gpuGelu<T,u8,8>);
    table output quantized at f=12. gelu' saturates to {0,1} beyond +-4,
    matching the clip range.
  - fixed point f=12 (sigma.cu: u64 scale = 12), truncation mode 'sigma'
    (deterministic floor, TrFloor).
  Optional variant (--sigma-primal): also replace the GELU primal by SIGMA's
  relu(x) - residual-LUT(relu-gelu) with the same quantization.
"""
import torch

C_INV = [0.03226120897959244, -0.049780065730235586, 0.057650254381631766, -0.08583010236400039,
         0.1317100845857373, -0.19765769464931107, 0.29629765286612486, -0.44444230412031277,
         0.6666666475671728]

SIGMA_PRIMAL = [False]
TANGENT_LUT = [True]
# gelu' LUT index model: 'round8' (first version: round(z*2^6), signed clip 255),
# 'floor8' (faithful SIGMA gpuGelu: Xt = TrWithSlack(x, scale-6) ~ floor(z*2^6), LUT on
# |Xt| clipped to 2^8-1, negative half via gelu'(-x) = 1 - gelu'(x)),
# 'floor10' (same, 10-bit index at step 2^-8 over the same +-4 range; genLUT(10, 8, scale)).
LUT_MODE = ['round8']
# LayerNorm rsqrt model: 'spu' (emu/fxp.py bit-faithful SPU f_rsqrt, as before) or 'sigma'
# (GPU-MPC fss/gpu_layernorm.cu + llama api.cpp Rsqrt: sum of squares exact at scale 2f,
#  F2BF16 keeps the top 8 significant bits (floor), 13-bit DPF LUT with extradiv = n_embd,
#  output GroupElement(double) cast = floor at scale f).
RSQRT_MODE = ['spu']
W_FRAC = [None]
THETA_FRAC = [None]  # LoRA-state encoding bits (secret; finer grid, truncated back to f after each product)     # public-weight encoding bits (None = same f as everything else)
RSQRT_MBITS = [7]   # mantissa bits kept by F2BF16 (SIGMA: 7 -> 13-bit LUT with 6-bit exponent)


def rsqrt_sigma(d, fx, n=256):
    import math
    S = fx.S
    xi = torch.round((d * d).sum(-1, keepdim=True) * S * S)          # exact at scale 2f
    xi = torch.clamp(xi, min=1.0)
    msb = torch.floor(torch.log2(xi))
    msb = msb - (torch.exp2(msb) > xi).double() + (torch.exp2(msb + 1) <= xi).double()
    sh = torch.clamp(msb - RSQRT_MBITS[0], min=0.0)
    val = torch.floor(xi / torch.exp2(sh)) * torch.exp2(sh)            # mbits+1 significant bits, floor
    return torch.floor(math.sqrt(n) * S * S / torch.sqrt(val)) / S     # floor cast, scale f


def gelu_exact(z):
    return z * 0.5 * (1.0 + torch.erf(z / math_sqrt2()))


def math_sqrt2():
    return 1.4142135623730951


def sigmoid_exact(s):
    return torch.sigmoid(s.double())


def gelu_prime_exact(z):
    k = 1.5957691216057308
    s = k * (z + (0.044715 * z * z) * z)
    sig = sigmoid_exact(s)
    dsig = sig * (1 - sig)
    ds = k * (1.0 + (0.134145 * z) * z)
    return sig + z * dsig * ds


def gelu_prime_lut(z, fx):
    """SIGMA-style quantized gelu' lookup; table output encoded at fx.f (nearest)."""
    mode = LUT_MODE[0]
    if mode == 'round8':
        idx = torch.clamp(torch.round(z * 64.0), -255.0, 255.0)
        return fx.enc(gelu_prime_exact(idx / 64.0))
    bits, k = {'floor8': (8, 6), 'floor10': (10, 8)}[mode]
    t = torch.floor(z * 2.0 ** k)                       # TrWithSlack ~ arithmetic shift
    a = torch.clamp(t.abs(), max=2.0 ** bits - 1)       # |Xt| clipped (u8 / u16 index)
    tab = fx.enc(gelu_prime_exact(a / 2.0 ** k))        # LUT entry for the magnitude
    return torch.where(t >= 0, tab, 1.0 - tab)


def gelu_sigma_primal(z, fx):
    """SIGMA GELU primal: relu(z) - LUT(relu-gelu), same input quantization."""
    idx = torch.clamp(torch.round(z * 64.0), -255.0, 255.0)
    xq = idx / 64.0
    res = torch.relu(xq) - gelu_exact(xq)
    return torch.relu(z) - fx.enc(res)


class Emu:
    def __init__(self, W, fx):
        self.fx = fx
        if W_FRAC[0] is None:
            self.W = {k: fx.enc(v) for k, v in W.items()}
        else:  # public weights encoded at a finer grid; products are truncated back to f by fx.q
            Sw = 2.0 ** W_FRAC[0]
            self.W = {k: torch.round(v * Sw) / Sw for k, v in W.items()}
        self.cinv = [fx.const(c) for c in C_INV]

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
        if SIGMA_PRIMAL[0]:
            out = q(gelu_sigma_primal(z, self.fx))
        else:
            s = q(c(1.5957691216057308) * (z + q(q(q(c(0.044715) * z) * z) * z)))
            pos = s >= 0
            a = torch.where(pos, s, -s)
            e, _ = self.exp_neg(-a)
            qq = self.inv_1p(e)
            y = torch.where(pos, qq, 1.0 - qq)
            out = q(z * y)
        if zt is None:
            return out, None
        # SIGMA-style tangent: gelu'(z) from the quantized LUT (or analytic for diagnostics)
        gp = gelu_prime_lut(z, self.fx) if TANGENT_LUT[0] else None
        if gp is not None:
            return out, q(gp * zt)
        c = self.fx.const
        s = q(c(1.5957691216057308) * (z + q(q(q(c(0.044715) * z) * z) * z)))
        pos = s >= 0
        a_ = torch.where(pos, s, -s)
        e, _ = self.exp_neg(-a_)
        qq = self.inv_1p(e)
        y = torch.where(pos, qq, 1.0 - qq)
        gate = (a_ < 12.0)
        dsig = torch.where(gate, q(q(e * qq) * qq), torch.zeros_like(a_))
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
        d = x - q(x.sum(-1, keepdim=True) * (1.0 / 256))
        if RSQRT_MODE[0] == 'sigma':
            r = rsqrt_sigma(d, self.fx)
        else:
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

    def encoder(self, h0, mask, pos, theta, u=None):
        fx, W, mm = self.fx, self.W, self.mm
        x = fx.enc(h0); xt = torch.zeros_like(x) if u is not None else None
        th = [fx.enc(t) for t in theta] if THETA_FRAC[0] is None else [torch.round(t * 2.0 ** THETA_FRAC[0]) / 2.0 ** THETA_FRAC[0] for t in theta]; ut = None if u is None else [fx.enc(v) for v in u]
        B, T, _ = x.shape
        mk = mask[:, None, None, :]
        ar = torch.arange(B, device=x.device)
        for i in range(4):
            w = lambda k: W[f'w{i}_{k}']
            Aq, Bq, Av, Bv = th[4 * i:4 * i + 4]

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
            if last:
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
