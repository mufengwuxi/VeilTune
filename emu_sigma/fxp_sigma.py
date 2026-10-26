"""fxp_sigma: copy of emu/fxp.py + mode 'sigma' (deterministic floor truncation,
mirroring GPU-MPC TruncateType::TrFloor used by SIGMA LayerNorm paths).
f default 12 per experiments/sigma/sigma.cu (u64 scale = 12)."""
import math
import torch


class FXP:
    def __init__(self, f=12, mode='sigma', on=True, gen=None):
        self.f, self.S, self.mode, self.on, self.gen = f, float(2 ** f), mode, on, gen

    def q(self, x):
        if not self.on:
            return x
        if self.mode == 'nearest':
            return torch.round(x * self.S) / self.S
        if self.mode == 'sigma':
            # GPU-MPC TrFloor: deterministic truncation toward -inf, biased
            return torch.floor(x * self.S) / self.S
        if self.mode == 'cheetah':
            u = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self.gen)
            k = (u > 0.8).double() - (u < 0.2).double()
            return (torch.floor(x * self.S) + k) / self.S
        return torch.floor(x * self.S + torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self.gen)) / self.S

    def enc(self, x):
        return torch.round(x * self.S) / self.S if self.on else x

    def const(self, c):
        c = float(torch.tensor(c, dtype=torch.float32))
        return round(c * self.S) / self.S if self.on else c

    def recip(self, x):
        if not self.on:
            return 1.0 / x
        f, q = self.f, self.q
        xi = torch.clamp(torch.round(x * self.S), min=1.0)
        p = torch.floor(torch.log2(xi))
        p = p - (torch.exp2(p) > xi).double() + (torch.exp2(p + 1) <= xi).double()
        factor = torch.exp2(f - 1 - p)
        c = q(x * factor)
        w = self.const(2.9142) - 2.0 * c
        e = 1.0 - q(c * w)
        r = q(w * (e + 1.0))
        e = q(e * e)
        r = q(r * (e + 1.0))
        return q(r * factor)

    def rsqrt(self, x):
        if not self.on:
            return torch.rsqrt(x)
        f, q = self.f, self.q
        xi = torch.clamp(torch.round(x * self.S), min=1.0)
        p = torch.floor(torch.log2(xi))
        p = p - (torch.exp2(p) > xi).double() + (torch.exp2(p + 1) <= xi).double()
        u = q(xi * torch.exp2(-p - 2))
        u2 = q(u * u); u3 = q(u2 * u); u4 = q(u3 * u)
        cs = [self.const(c) for c in (-15.47994394, 38.4714796, -49.86605845, 26.02942339)]
        r = q(cs[0] * u + cs[1] * u2 + cs[2] * u3 + cs[3] * u4) + self.const(4.14285016)
        k = p + 2 - f
        even = torch.remainder(p + 1, 2) == 0
        sq2 = math.floor(2 ** (f // 2) * math.sqrt(2)) / 2 ** (f // 2) / math.sqrt(2)
        comp = torch.exp2(-k / 2) * torch.where(even, torch.full_like(k, sq2), torch.ones_like(k))
        return q(r * comp)
