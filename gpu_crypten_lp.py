"""VeilTune-LP = bp2 with low-precision comparisons in GELU gates and in the softmax max tournament.

ltz_lowprec(x, t, L): SecureML local truncation of the arithmetic shares by t bits, local reduction mod 2^L,
pack 64/L lanes per int64 word, lane-isolated SPK adder (log2 L levels of CrypTen's own SPK masks), lane MSB,
B2A single bit. Correct when |x / 2^(f... )| fits: |x_ring| / 2^t < 2^(L-1) - 2, and away from 0 by ~2^t ring units.
GELU: [z<0] and [z^2<16] with t=16, L=32 (sign only matters for |z|>=4; interval tolerant near the boundary).
max tree: [a-b<0] with t=12, L=16 (softmax is shift-invariant; an approximate max within 1/16 is enough).
New file; CrypTen and the existing kernels are unchanged."""
import math, torch, crypten
import crypten.communicator as comm
from crypten.mpc import MPCTensor
from crypten.mpc.primitives.binary import BinarySharedTensor
from crypten.mpc.primitives import circuit, beaver
import gpu_crypten_bp2 as BP2
from gpu_crypten_v2 import neg1p, to_fx
from gpu_crypten import const

_consts = getattr(circuit, '__SPK_circuit_constants')
GELU_T, GELU_L = 16, 32
MAX_T, MAX_L = 12, 16


def _spk_levels(S, P, levels):
    SP = BinarySharedTensor.stack([S, P])
    M, OM, MUL = _consts(SP.device)
    for i in range(levels):
        in_mask, out_mask = M[i], OM[i]
        P0 = SP[1] & out_mask
        S1P1 = SP & in_mask
        S1P1._tensor *= MUL[i]
        update = P0 & S1P1
        SP[1] &= (out_mask ^ -1)
        SP ^= update
    return SP[0], SP[1]


def ltz_lowprec(x, t, L):
    lanes = 64 // L; levels = int(math.log2(L))
    sh = x._tensor.share; rank = comm.get().get_rank()
    raw = getattr(sh, '_tensor', sh)
    tr = (raw >> t) if rank == 0 else -((-raw) >> t)
    v = (tr & ((1 << L) - 1)).flatten()
    n = v.numel(); pad = (-n) % lanes
    if pad:
        v = torch.cat([v, torch.zeros(pad, dtype=torch.long, device=v.device)])
    v = v.view(-1, lanes)
    packed = v[:, 0].clone()
    for j in range(1, lanes):
        packed = packed | (v[:, j] << (j * L))
    if packed.is_cuda:
        from crypten.cuda import CUDALongTensor
        packed = CUDALongTensor(packed)
    xb = [BinarySharedTensor(packed, src=i, device=raw.device) for i in range(2)]
    S = xb[0] & xb[1]; P = xb[0] ^ xb[1]
    carry, _ = _spk_levels(S, P, levels)
    s = P ^ (carry << 1)
    msb = BinarySharedTensor.stack([(s >> (j * L + L - 1)) & 1 for j in range(lanes)])
    bits = beaver.B2A_single_bit(msb)
    braw = getattr(bits.share, '_tensor', bits.share)
    bt = braw.transpose(0, 1).reshape(-1)[:n].reshape(raw.shape)
    if bt.is_cuda:
        from crypten.cuda import CUDALongTensor
        bt = CUDALongTensor(bt)
    out = MPCTensor.from_shares(bt, precision=0)
    out.encoder._scale = 1
    return out


GC, GA, GB = BP2.GC, BP2.GA, BP2.GB


class SecureOpsLP(BP2.SecureOpsBP2):

    def gelu_save(self, z):
        if not getattr(self, 'NEW_GELU', True):
            return super().gelu_save(z)
        z2 = z * z
        g = ltz_lowprec(crypten.stack([z, z2 - GC * GC]), GELU_T, GELU_L)
        neg, inn = g[0], g[1]
        u = z2 * const(1.0 / (GC * GC))
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
        innf = to_fx(inn, z)
        sel = crypten.stack([innf, innf]) * crypten.stack([R - absz * 0.5, zS - sgn * 0.5])
        return z * 0.5 + absz * 0.5 + sel[0], sgn * 0.5 + 0.5 + sel[1]

    @staticmethod
    def max_tree(x):
        d = x.dim() - 1
        while x.size(d) > 1:
            n = x.size(d); h = n // 2
            a, b = x.narrow(d, 0, h), x.narrow(d, h, h)
            m = a + ltz_lowprec(a - b, MAX_T, MAX_L) * (b - a)
            x = crypten.cat([m, x.narrow(d, 2 * h, 1)], dim=d) if n % 2 else m
        return x
