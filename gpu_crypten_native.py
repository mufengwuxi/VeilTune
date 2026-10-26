"""Baseline: the same packed secure backprop step, but every nonlinearity uses CrypTen's built-in functions
(default cfg: exp 8 iters, NR reciprocal 10 iters, inv_sqrt NR 3 iters, sigmoid/tanh via reciprocal, max via
log_reduction + one-hot argmax). Same packing, same last-layer MASK-row trick, same backward formulas
(only the forward kernels differ) -> isolates the effect of our kernels."""
import math
import crypten
import gpu_crypten_bp as BPm
from gpu_crypten import const

K0 = math.sqrt(2.0 / math.pi)


class SecureOpsNative(BPm.SecureOpsBP):

    def gelu_save(self, z):                       # tanh GELU via crypten.tanh, derivative from the same t
        z2 = z * z
        inner = (z + z2 * z * const(0.044715)) * const(K0)
        t = inner.tanh()
        y = (t + 1.0) * 0.5
        out = z * y
        dt = (t * t * -1 + 1.0) * ((z2 * const(3 * 0.044715) + 1.0) * const(K0))
        gp = y + z * dt * 0.5
        return out, gp

    def ln_save(self, x, g, b):
        inv = const(1.0 / x.shape[-1])
        d = x - x.sum(dim=-1, keepdim=True) * inv
        var = (d * d).sum(dim=-1, keepdim=True) * inv
        r = (var + float(getattr(self, 'LN_EPS', 0.0))).inv_sqrt()
        n = d * r
        return n * g + b, (n, r)

    def softmax_save(self, s, mask):
        s = s + (mask - 1.0) * 30.0
        p = s.softmax(-1)
        return p * mask

    def _sigmoid(self, s):
        return s.sigmoid()
