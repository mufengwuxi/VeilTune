"""Secure DP-SGD pieces on RoBERTa-large (bp2 kernels, LS=256, exp-free).
  check : B=4: secure per-sample grads vs plaintext FP64 per-sample grads; secure clipped aggregate (no noise)
          vs plaintext clipped mean.
  time  : one secure DP step (per-sample grads + oblivious drop + clip + dealer noise) for B in --batches."""
import argparse, json, math, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12


@mpc.run_multiprocess(world_size=2)
def worker(mode, batches, C, sigma):
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp2 as BP2
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    tail, W, H, theta, C_ = T12.load_weights()
    ops = BP2.SecureOpsBP2(W); ops.init_head(H)
    ops.D, ops.NH, ops.LN_EPS = 1024, 16, G.const(1e-5)
    ops.LS, ops.EXP_FREE, ops.PER_SAMPLE = 256.0, True, True
    RV = lambda t: torch.as_tensor(G.reveal(t).cpu()).double()
    th = [crypten.cryptensor(G.enc_pub(t.numpy()).cuda(), src=0, device='cuda') for t in theta]
    out = {}
    if mode == 'check':
        ix, h0p, lens, pos, y = T12.batch(C_, 4, 1)
        ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')
        gps = ops.loss_grad(h0p.cuda(), lens, pos, th, ysec)
        agg, okb, n2 = ops.dp_aggregate(gps, C, 1e4)
        gps_r = [RV(g) for g in gps]; agg_r = [RV(g) for g in agg]; n2_r = RV(n2)
        if rank == 0:
            torch.set_default_dtype(torch.float64)
            for P in tail.layers:
                for k in list(P):
                    if torch.is_tensor(P[k]): P[k] = P[k].double()
            for k in list(tail.head): tail.head[k] = tail.head[k].double()
            d = C_['train']; lab = C_['label_ids'].cuda()
            ref = []
            for j in range(4):
                i = ix[j:j + 1]; n = int(d['mask'][i].sum())
                thp = [t.clone().cuda().requires_grad_(True) for t in theta]
                lo = tail.forward(thp, d['h'][i, :n].double().cuda(), d['mask'][i, :n].cuda(), d['pos'][i].cuda(), lab)
                ref.append([g.detach().cpu() for g in torch.autograd.grad(F.cross_entropy(lo, d['y'][i].cuda()), thp)])
            sec = torch.stack([torch.cat([g[j].flatten() for g in gps_r]) for j in range(4)])
            pl = torch.stack([torch.cat([g.flatten() for g in r]) for r in ref])
            rel_ps = ((sec - pl).norm(dim=1) / pl.norm(dim=1)).tolist()
            nrm = pl.norm(dim=1)
            clipped = (pl * torch.clamp(C / nrm, max=1.0)[:, None]).mean(0)
            a_ = torch.cat([g.flatten() for g in agg_r])
            out = dict(rel_err_per_sample=rel_ps, norms_plain=nrm.tolist(), norms_secure=n2_r.sqrt().tolist(),
                       agg_rel_err=float((a_ - clipped).norm() / clipped.norm()),
                       agg_cos=float(a_ @ clipped / a_.norm() / clipped.norm()), C=C)
    else:
        for B in batches:
            ix, h0p, lens, pos, y = T12.batch(C_, B, 2)
            P = sum(t.numel() for t in theta)
            noise = crypten.cryptensor((torch.randn(P, dtype=torch.float64) * sigma * C).cuda(), src=0, device='cuda')  # dealer share (offline)
            torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
            ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')
            gps = ops.loss_grad(h0p.cuda(), lens, pos, th, ysec)
            t1 = time.perf_counter(); r1, b1 = comm.comm_rounds, comm.comm_bytes
            agg, okb, n2 = ops.dp_aggregate(gps, C, 1e4, noise)
            new = [t - g * 1e-3 for t, g in zip(th, agg)]
            torch.cuda.synchronize()
            out[f'b{B}'] = dict(s=time.perf_counter() - t0, s_dp=time.perf_counter() - t1, rounds=comm.comm_rounds - r0,
                                rounds_dp=comm.comm_rounds - r1, bytes=comm.comm_bytes - b0, bytes_dp=comm.comm_bytes - b1,
                                peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9)
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'time']); ap.add_argument('--batches', type=int, nargs='*', default=[1, 1, 32, 64])
    ap.add_argument('--C', type=float, default=1.0); ap.add_argument('--sigma', type=float, default=1.0)
    a = ap.parse_args()
    crypten.cfg.mpc.provider = 'TTP'
    res = {r['rank']: r['out'] for r in worker(a.mode, a.batches, a.C, a.sigma) if r}
    (ROOT / f'stage2/t20_dp_{a.mode}.json').write_text(json.dumps(res, indent=1))
    if a.mode == 'check':
        print('T20_CHECK', json.dumps(res[0]))
    else:
        for k in res[1]:
            print('T20_TIME', k, json.dumps(dict(time_s=round(res[0][k]['s'], 2), dp_s=round(res[0][k]['s_dp'], 2), rounds=res[1][k]['rounds'],
                  rounds_dp=res[1][k]['rounds_dp'], wire_GB=round(res[1][k]['bytes'] / 1e9, 2), dp_GB=round(res[1][k]['bytes_dp'] / 1e9, 3),
                  peak_mem_GB=round(res[0][k]['peak_mem_GB'], 1))))
