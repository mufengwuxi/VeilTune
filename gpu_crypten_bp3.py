"""bp3 = bp2 with the GELU polynomial argument clamped to its interval.

bp2 evaluates the Horner polynomials R(u), S(u) on u = z^2/c^2 for ALL z and discards the result outside |z| < c.
For residual-stream outliers u is large, the Horner intermediates reach ~5e7 and local truncation fails often
(t29: 7 of 8 failures). Here u is replaced by u_c = [|z|<c]*(u-1) + 1 (= u inside, 1 outside) before the loop:
one extra batched multiplication (1 round), outputs unchanged (outside the interval the result is discarded).
"""
import crypten
import gpu_crypten_bp2 as BP2
from gpu_crypten_v2 import _gate, neg1p, to_fx
from gpu_crypten import const
GA, GB, GC = BP2.GA, BP2.GB, BP2.GC


class SecureOpsBP3(BP2.SecureOpsBP2):

    def gelu_save(self, z):
        z2 = z * z
        g = _gate(crypten.stack([z, z2 - GC * GC]))
        neg, inn = g[0], g[1]
        innf = to_fx(inn, z)
        u = innf * (z2 * const(1.0 / (GC * GC)) - 1.0) + 1.0          # clamp: u inside, 1 outside (1 round)
        R = u * const(GA[7]) + GA[6]
        S = u * const(GB[6]) + GB[5]
        for j in range(6):
            if j < 5:
                pr = crypten.stack([R, S]) * crypten.stack([u, u])
                R, S = pr[0] + GA[5 - j], pr[1] + GB[4 - j]
            else:
                R = R * u + GA[0]
        sgn = to_fx(neg1p(neg * 2), z)
        pr = crypten.stack([z, z]) * crypten.stack([sgn, S])
        absz, zS = pr[0], pr[1]
        sel = crypten.stack([innf, innf]) * crypten.stack([R - absz * 0.5, zS - sgn * 0.5])
        out = z * 0.5 + absz * 0.5 + sel[0]
        gp = sgn * 0.5 + 0.5 + sel[1]
        return out, gp
