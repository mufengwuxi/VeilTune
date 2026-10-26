"""ONE zero-reveal secure training step (backprop, secure head, SGD update), batch 64, TTP. Timed once."""
import json, sys, time
from pathlib import Path
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))

@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp as BPm
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    Fd, W, H, theta0, _ = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    ops = BPm.SecureOpsBP(W); ops.init_head(H)
    h0, mask, pos, y = G.build_batch('s2000', 64)
    mv = int(mask.sum(-1).max()); h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
    h0p, lens = V.pack(h0, mask)
    th = G.to_ct_list(theta0)
    torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
    ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')
    grads = ops.loss_grad(h0p, lens, [int(p) for p in pos], th, ysec)
    th = [t - g * 3e-4 for t, g in zip(th, grads)]
    torch.cuda.synchronize()
    return dict(rank=rank, s=time.perf_counter() - t0, rounds=comm.comm_rounds - r0, bytes=comm.comm_bytes - b0,
                N_real=int(sum(lens)), T=mv)

if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = {r['rank']: r for r in run() if r}
    print('ONE_STEP', json.dumps(dict(time_s=round(res[0]['s'], 2), rounds=res[1]['rounds'],
          wire_GB=round(res[1]['bytes'] / 1e9, 2), N_real=res[0]['N_real'], T=res[0]['T'])))
