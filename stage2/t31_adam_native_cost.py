"""Cost (rounds, rank-1 bytes) of ONE Adam update on the adapter with CrypTen's built-in operators
(inv_sqrt default: exp-based initial guess + 3 NR iters), no scaling, no guards. Read-only w.r.t. the framework."""
import json, sys, torch, crypten, crypten.mpc as mpc
from pathlib import Path
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12

@mpc.run_multiprocess(world_size=2)
def run():
    crypten.init(); crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    _, _, _, theta, _ = T12.load_weights()
    n = sum(t.numel() for t in theta)
    torch.manual_seed(0)
    g = crypten.cryptensor(torch.randn(n).double().cuda() * 1e-3, device='cuda')
    m = crypten.cryptensor(torch.zeros(n).double().cuda(), device='cuda')
    v = crypten.cryptensor(torch.zeros(n).double().cuda(), device='cuda')
    th = crypten.cryptensor(torch.zeros(n).double().cuda(), device='cuda')
    comm.reset_communication_stats()
    m = m * 0.9 + g * 0.1
    v = v * 0.999 + (g * g) * 0.001
    r = (v + 1e-8).inv_sqrt()
    th = th - (m * r) * 1e-4
    return dict(rank=rank, n=n, rounds=comm.comm_rounds, bytes=comm.comm_bytes)

if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = [r for r in run() if r]
    print('T31', json.dumps(res))
