"""Secure backprop check + timing vs secure JVP (same process, alternating, same load).
correctness: <grad_bp, u> must equal the secure JVP c (same theta, batch, u), and match the fixture refs."""
import json, sys, time, types
from pathlib import Path
import numpy as np, torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp as BPm
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    Fd, W, H, theta0, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    ops = BPm.SecureOpsBP(W); ops.init_head(H)
    th = G.to_ct_list(theta0)
    c_same = np.load(ROOT / 'sjvp/c_plain_s2000_mpc.npy')
    out = {'rows': []}
    # ---- correctness on fixture rows (B=1) ----
    for i in range(4):
        n = int(Fd['mask'][i].sum())
        h0p = torch.tensor(Fd['h0'][i, :n], dtype=torch.float64, device='cuda')
        y = crypten.cryptensor(torch.tensor([float(Fd['y'][i])], device='cuda'), src=0, device='cuda')
        grads = ops.loss_grad(h0p, [n], [int(Fd['pos'][i])], th, y)
        gpl = [G.reveal(g).cpu() for g in grads]
        for d in [2, 3, 4, 5]:
            c_bp = float(sum((g * torch.tensor(u)).sum() for g, u in zip(gpl, us[d])))
            h_ct, hd_ct = ops.encoder_packed(h0p, [n], [int(Fd['pos'][i])], th, G.to_ct_list(us[d]))
            c_jvp = float(G.reveal(ops.secure_c(h_ct, hd_ct, y)).item())
            out['rows'].append(dict(sample=i, dir=d, c_bp=c_bp, c_jvp=c_jvp, c_same=float(c_same[i, d])))
    rows = out['rows']
    A = lambda k: np.array([r[k] for r in rows])
    rl2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    out['check'] = dict(bp_vs_same=rl2(A('c_bp'), A('c_same')), jvp_vs_same=rl2(A('c_jvp'), A('c_same')),
                        bp_vs_jvp=rl2(A('c_bp'), A('c_jvp')))
    # ---- timing: full training step, JVP vs BP, alternating ----
    shapes = [t.shape for t in theta0]; sizes = [int(np.prod(s)) for s in shapes]; lr = 3e-4
    for label, B, vlen in [('b1', 1, 10), ('b64', 64, None)]:
        h0, mask, pos, yy = G.build_batch('s2000', B, valid_len=vlen)
        mv = int(mask.sum(-1).max()); h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
        h0p, lens = V.pack(h0, mask); pp = [int(p) for p in pos]
        ysec = crypten.cryptensor(yy.double().cuda(), src=0, device='cuda')
        rec = {'jvp': [], 'bp': []}
        for rep in range(4):
            for mode in ('jvp', 'bp'):
                torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
                if mode == 'jvp':
                    g = torch.Generator(device='cuda').manual_seed(1000 * rank + rep)
                    half = torch.randn(sum(sizes), generator=g, device='cuda', dtype=torch.float64) * (0.5 ** 0.5)
                    U = crypten.cryptensor(half, src=0, device='cuda') + crypten.cryptensor(half, src=1, device='cuda')
                    u, o = [], 0
                    for s, k in zip(shapes, sizes):
                        u.append(U[o:o + k].reshape(s)); o += k
                    h_ct, hd_ct = ops.encoder_packed(h0p, lens, pp, th, u)
                    c = ops.secure_c(h_ct, hd_ct, ysec)
                    dU = U * (c * lr)
                    o, new = 0, []
                    for t, s, k in zip(th, shapes, sizes):
                        new.append(t - dU[o:o + k].reshape(s)); o += k
                else:
                    grads = ops.loss_grad(h0p, lens, pp, th, ysec)
                    new = [t - g * lr for t, g in zip(th, grads)]       # SGD (public lr: local)
                torch.cuda.synchronize()
                if rep > 0:
                    rec[mode].append(dict(s=time.perf_counter() - t0, rounds=comm.comm_rounds - r0, bytes=comm.comm_bytes - b0))
        out[label] = {m: dict(s=float(np.median([r['s'] for r in v])), rounds=v[-1]['rounds'], bytes=v[-1]['bytes'])
                      for m, v in rec.items()}
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = [r for r in run() if r]
    (ROOT / 'stage2/t10_bp.json').write_text(json.dumps(res, indent=1))
    print('T10_DONE')
