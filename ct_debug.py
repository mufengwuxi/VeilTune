"""Unit-debug SecureOps kernels vs emu (plain torch) under CrypTen 2PC.
Both ranks run the identical op sequence; only rank 0 prints."""
import sys
from pathlib import Path
import numpy as np
import torch
import crypten
import crypten.mpc as mpc

sys.path.insert(0, '/workspace/veiltune')
sys.path.insert(0, '/workspace/veiltune/emu')
ROOT = Path('/workspace/veiltune')


@mpc.run_multiprocess(world_size=2)
def work():
    import gpu_crypten as G
    G.ARITH = crypten.mpc.ptype.arithmetic
    crypten.init()
    rank = crypten.communicator.get().get_rank()
    from fxp import FXP
    from secure_emu import Emu
    torch.manual_seed(7)

    Fd = dict(np.load(ROOT / 'sjvp/fixture_s2000.npz'))
    W = {k: torch.tensor(v, dtype=torch.float64) for k, v in Fd.items() if k.startswith('w')}
    ops = G.SecureOps({k: v.numpy() for k, v in W.items()})
    emu = Emu(W, FXP(f=16, mode='nearest', on=True))
    out = {}

    def err(a, b):
        return round(float((a - b).abs().max().item()), 8), round(float(b.abs().max().item()), 4)

    def ct(x):
        return crypten.cryptensor(G.enc_pub(x).cuda(), device='cuda')

    v = torch.logspace(-3, 4, 16, dtype=torch.float64)
    r_ct = ops.rsqrt_newton(ct(v)).get_plain_text().cpu()
    out['rsqrt_maxerr_vs_exact'] = err(r_ct, v.rsqrt())

    x = -torch.rand(64, dtype=torch.float64) * 12
    y_ct = ops.exp_neg(ct(x))[0].get_plain_text().cpu()
    out['expneg_vs_emu'] = err(y_ct, emu.exp_neg(x)[0])

    e = torch.rand(64, dtype=torch.float64)
    out['inv1p_vs_emu'] = err(ops.inv_1p(ct(e)).get_plain_text().cpu(), emu.inv_1p(e))

    z = torch.randn(64, dtype=torch.float64) * 8
    g_ct, gt_ct = ops.gelu(ct(z), ct(torch.ones_like(z)))
    g_em, gt_em = emu.gelu(z, torch.ones_like(z))
    out['gelu_vs_emu'] = err(g_ct.get_plain_text().cpu(), g_em)
    out['gelu_t_vs_emu'] = err(gt_ct.get_plain_text().cpu(), gt_em)

    T = 12
    s, st = torch.randn(2, T, dtype=torch.float64) * 3, torch.randn(2, T, dtype=torch.float64)
    mask = torch.ones(2, T, dtype=torch.float64)
    mask[1, 8:] = 0
    p_ct, pt_ct = ops.softmax(ct(s), ct(st), mask[:, None, None, :].cuda())
    p_em, pt_em = emu.softmax(s, st, mask[:, None, None, :])
    out['softmax_vs_emu'] = err(p_ct.get_plain_text().cpu(), p_em)
    out['softmax_t_vs_emu'] = err(pt_ct.get_plain_text().cpu(), pt_em)

    x = torch.tensor(Fd['h0'][0], dtype=torch.float64)
    y_ct, yt_ct = ops.ln(ct(x), ct(torch.ones_like(x)), ops.W['w0_ln1'], ops.W['w0_ln1b'])
    y_em, yt_em = emu.ln(x, torch.ones_like(x), W['w0_ln1'], W['w0_ln1b'])
    out['ln_vs_emu'] = err(y_ct.get_plain_text().cpu(), y_em)
    out['ln_t_vs_emu'] = err(yt_ct.get_plain_text().cpu(), yt_em)

    if rank == 0:
        print('DEBUG', out)
    return out


work()
