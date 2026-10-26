"""Fixed-point emulation of SPU FM64 arithmetic (value domain, float64 on GPU).
q(): every truncation point (secret x secret / secret x non-integer public multiply, matmul output).
Truncation model: stochastic (floor + U[0,1)) ~ Cheetah's probabilistic 1-ulp truncation; 'nearest' optional.
recip()/rsqrt(): bit-faithful re-implementations of SPU 0.9.5 reciprocal_goldschmidt (2 iters) and
f_rsqrt (degree-4 init guess on u in [0.25,0.5) + exact power-of-two compensation with integerised sqrt2)."""
import math, torch

class FXP:
    def __init__(self, f=16, mode='cheetah', on=True, gen=None):
        self.f, self.S, self.mode, self.on, self.gen = f, float(2 ** f), mode, on, gen

    def q(self, x):
        if not self.on:
            return x
        if self.mode == 'nearest':
            return torch.round(x * self.S) / self.S
        if self.mode == 'cheetah':
            # calibrated to SPU 0.9.5 Cheetah FM64 kernels (kern_cmp*.py): floor, plus a 1-ulp
            # error of either sign with prob 0.2 each (matches recip/rsqrt error rms & bias at f16/f24)
            u = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self.gen)
            k = (u > 0.8).double() - (u < 0.2).double()
            return (torch.floor(x * self.S) + k) / self.S
        return torch.floor(x * self.S + torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self.gen)) / self.S

    def enc(self, x):  # input encoding: nearest
        return torch.round(x * self.S) / self.S if self.on else x

    def const(self, c):
        c = float(torch.tensor(c, dtype=torch.float32))  # SPU takes float constants as float32
        return round(c * self.S) / self.S if self.on else c

    def recip(self, x):
        """1/x, x>0 (softmax denominators)."""
        if not self.on:
            return 1.0 / x
        f, q = self.f, self.q
        xi = torch.clamp(torch.round(x * self.S), min=1.0)
        p = torch.floor(torch.log2(xi))
        p = p - (torch.exp2(p) > xi).double() + (torch.exp2(p + 1) <= xi).double()   # exact msb
        factor = torch.exp2(f - 1 - p)                       # value of bitrev(2^p) over 2f bits
        c = q(x * factor)                                    # in [0.5,1)
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
        u = q(xi * torch.exp2(-p - 2))                       # u in [0.25,0.5)
        u2 = q(u * u); u3 = q(u2 * u); u4 = q(u3 * u)
        cs = [self.const(c) for c in (-15.47994394, 38.4714796, -49.86605845, 26.02942339)]
        r = q(cs[0] * u + cs[1] * u2 + cs[2] * u3 + cs[3] * u4) + self.const(4.14285016)
        # comp = 2^{-(p+2-f)/2}; when (p+1) is even SPU uses int(2^{f/2} sqrt2) (f even)
        k = p + 2 - f
        even = torch.remainder(p + 1, 2) == 0
        sq2 = math.floor(2 ** (f // 2) * math.sqrt(2)) / 2 ** (f // 2) / math.sqrt(2)
        comp = torch.exp2(-k / 2) * torch.where(even, torch.full_like(k, sq2), torch.ones_like(k))
        return q(r * comp)
