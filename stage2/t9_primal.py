"""v2 encoder: primal-only vs primal+tangent (JVP) cost, b1 and b64 (packed), TTP, rank-1 counters."""
import json, sys, time, types
from pathlib import Path
import numpy as np, torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    Fd, W, H, theta0, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    ops = V.SecureOpsV2(W)
    th = G.to_ct_list(theta0); u = G.to_ct_list(us[0])
    out = {}
    for label, B, vlen in [('b1', 1, 10), ('b64', 64, None)]:
        h0, mask, pos, y = G.build_batch('s2000', B, valid_len=vlen)
        mv = int(mask.sum(-1).max()); h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
        h0p, lens = V.pack(h0, mask)
        for mode in ('primal', 'jvp'):
            rr = []
            for rep in range(4):
                torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
                ops.encoder_packed(h0p, lens, [int(p) for p in pos], th, u if mode == 'jvp' else None)
                torch.cuda.synchronize()
                rr.append(dict(s=time.perf_counter() - t0, rounds=comm.comm_rounds - r0, bytes=comm.comm_bytes - b0))
            out[f'{label}_{mode}'] = dict(s=float(np.median([r['s'] for r in rr[1:]])), rounds=rr[-1]['rounds'], bytes=rr[-1]['bytes'])
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = [r for r in run() if r]
    (ROOT / 'stage2/t9_primal.json').write_text(json.dumps(res, indent=1))
    print('T9_DONE')
