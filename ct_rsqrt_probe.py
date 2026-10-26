"""Per-point rsqrt probe: where and how big is the error?"""
import sys
from pathlib import Path
import numpy as np
import torch
import crypten
import crypten.mpc as mpc

sys.path.insert(0, '/workspace/veiltune')
ROOT = Path('/workspace/veiltune')


@mpc.run_multiprocess(world_size=2)
def work():
    import gpu_crypten as G
    G.ARITH = crypten.mpc.ptype.arithmetic
    crypten.init()
    rank = crypten.communicator.get().get_rank()
    v = torch.logspace(-3, 4, 16, dtype=torch.float64)
    ct_v = crypten.cryptensor(G.enc_pub(v).cuda(), device='cuda')
    # also probe the ladder alone
    ops = G.SecureOps({})
    r0 = ct_v * 0 + G.const(2.0 ** -8)
    decided = ct_v * 0
    for thr, cand in G.SecureOps._RSQRT_RUNGS:
        gate = (ct_v < thr).to(crypten.mpc.ptype.arithmetic)
        take = gate - gate * decided
        r0 = r0 + take * ((ct_v * 0 + G.const(cand)) - r0)
        decided = decided + take
    out_r0 = r0.get_plain_text().cpu()
    out_r = ops.rsqrt_newton(ct_v).get_plain_text().cpu()
    if rank == 0:
        for i in range(len(v)):
            rstar = float(v[i].rsqrt())
            print(f"v={float(v[i]):10.3e} r*={rstar:10.4f} r0={float(out_r0[i]):10.4f} "
                  f"r={float(out_r[i]):10.4f} rel={abs(float(out_r[i]) - rstar) / rstar:.2e}")
    return {}


work()
