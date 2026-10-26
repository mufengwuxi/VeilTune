"""Unit check of gpu_crypten_bp2 kernels (2PC, TTP): GELU/GELU' vs exact erf, max_tree vs torch max,
softmax_save vs torch softmax; also old (bp) GELU for reference, plus rounds/bytes of each kernel (rank 1)."""
import json, math, sys
from pathlib import Path
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp as BPm, gpu_crypten_bp2 as BP2
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    RV = lambda t: torch.as_tensor(G.reveal(t).cpu()).double().cuda()
    torch.manual_seed(0)
    z = torch.cat([torch.linspace(-12, 12, 4801), torch.randn(20000) * 2.5, torch.tensor([-40., 40., 60.])]).double().cuda()
    zc = crypten.cryptensor(z, device='cuda')
    out = {}
    ref = z * 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    dref = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    for name, ops in (('bp', BPm.SecureOpsBP({})), ('bp2', BP2.SecureOpsBP2({}))):
        comm.reset_communication_stats()
        o, gp = ops.gelu_save(zc)
        r, b = comm.comm_rounds, comm.comm_bytes
        o, gp = RV(o), RV(gp)
        out[f'gelu_{name}'] = dict(max_err=float((o - ref).abs().max()), max_err_grad=float((gp - dref).abs().max()),
                                   rounds=r, bytes_per_elt=b / z.numel())
    s = (torch.randn(2, 16, 7, 49) * 4).double().cuda()
    mask = torch.ones(2, 1, 1, 49).double().cuda(); mask[1, ..., 30:] = 0
    sc = crypten.cryptensor(s, device='cuda')
    ops2 = BP2.SecureOpsBP2({})
    comm.reset_communication_stats()
    m = ops2.max_tree(sc); r, b = comm.comm_rounds, comm.comm_bytes
    out['max_tree'] = dict(err=float((RV(m) - s.max(-1, keepdim=True)[0]).abs().max()), rounds=r, bytes_per_row=b / (s.numel() / 49))
    comm.reset_communication_stats()
    m0 = sc.max(dim=-1, keepdim=True)[0]; r, b = comm.comm_rounds, comm.comm_bytes
    out['max_crypten'] = dict(err=float((RV(m0) - s.max(-1, keepdim=True)[0]).abs().max()), rounds=r, bytes_per_row=b / (s.numel() / 49))
    for name, ops in (('bp', BPm.SecureOpsBP({})), ('bp2', ops2)):
        comm.reset_communication_stats()
        p = ops.softmax_save(sc, mask); r, b = comm.comm_rounds, comm.comm_bytes
        pr = torch.softmax(s + (mask - 1) * 1e9, -1)
        out[f'softmax_{name}'] = dict(max_err=float((RV(p) - pr).abs().max()), rounds=r, bytes_per_row=b / (s.numel() / 49))
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = {r['rank']: r['out'] for r in run() if r}
    (ROOT / 'stage2/t14_kernels.json').write_text(json.dumps(res, indent=1))
    for k in res[1]:
        print('T14', k, json.dumps(res[1][k]), 'err0', {a: v for a, v in res[0][k].items() if 'err' in a})
