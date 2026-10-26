"""Cost (rounds, bytes per party) of ONE VeilTune optimizer step as in t16_lp.py: gradient skip check + clip,
scaled Adam with moment range checks, batched rsqrt, q clip, gated update. Independent of the batch.
(; new file, framework untouched; code mirrors t16_lp.py lines ~95-145)"""
import json, math, sys, time, torch, crypten, crypten.mpc as mpc
from pathlib import Path
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12

@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_lp as LP
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic; crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    _, W, H, theta, _ = T12.load_weights()
    ops = LP.SecureOpsLP(W); ops.init_head(H)
    S_, clip, tau, qclip, eps, lr, step = 8192.0, 1.0, 10.0, 4.0, 3e-5, 1e-4, 10
    b1, b2 = 0.9, 0.999
    torch.manual_seed(0)
    th = [crypten.cryptensor(t.double().cuda(), src=0, device='cuda') for t in theta]
    grads = [crypten.cryptensor(torch.randn(t.shape).double().cuda() * 1e-2, src=0, device='cuda') for t in theta]
    flat = lambda ts: crypten.cat([t.flatten() for t in ts])
    def unflat(x, like):
        out_, o = [], 0
        for t in like:
            n_ = t.numel(); out_.append(x[o:o + n_].reshape(t.shape)); o += n_
        return out_
    adam_m = [g * 0 for g in grads]; adam_v = [g * 0 for g in grads]
    res = []
    for rep in range(2):
        torch.cuda.synchronize(); r0, by0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
        gc = flat(grads)
        gt = V.cmp_lt_batch(gc, [-tau, tau, -clip, clip])
        nbad = (gt[0] + V.neg1p(gt[1])).sum()
        ok = V.cmp_lt_batch(V.to_fx(nbad, gc), [0.5])[0]
        gc = gc + gt[2] * ((gc * 0 - clip) - gc) + V.neg1p(gt[3]) * ((gc * 0 + clip) - gc)
        gc = gc * ok
        Gs = [gr * int(S_) for gr in unflat(gc, grads)]
        cb2 = math.sqrt(1 - b2); MB = S_ * clip * 1.01; VB = (S_ * clip) ** 2 * 1.01
        m_new = [m * b1 + g * (1 - b1) for m, g in zip(adam_m, Gs)]
        hs = [g * cb2 for g in Gs]
        v_new = [v * b2 + h * h for v, h in zip(adam_v, hs)]
        mc_old, vc_old = flat(adam_m), flat(adam_v); mc, vc = flat(m_new), flat(v_new)
        gm = V.cmp_lt_batch(mc, [-MB, MB]); gv = V.cmp_lt_batch(vc, [-1.0, VB])
        okm, okv_ = gm[1] - gm[0], gv[1] - gv[0]
        okk = crypten.stack([okm, okv_]) * ok; okm, okv_ = okk[0], okk[1]
        pr = crypten.stack([okm, okv_]) * crypten.stack([mc - mc_old, vc - vc_old])
        adam_m, adam_v = unflat(mc_old + pr[0], Gs), unflat(vc_old + pr[1], Gs)
        lr_t = lr * math.sqrt(1 - b2 ** step) / (1 - b1 ** step); eps_t = (eps * S_) * math.sqrt(1 - b2 ** step)
        qc = flat(adam_m) * ops.rsqrt_newton(flat(adam_v) + eps_t * eps_t)
        gq = V.cmp_lt_batch(qc, [-qclip, qclip])
        qc = qc + gq[0] * ((qc * 0 - qclip) - qc) + V.neg1p(gq[1]) * ((qc * 0 + qclip) - qc)
        qc = qc * ok
        upd = [(q * (lr_t * 1024)).div(1024) for q in unflat(qc, Gs)]
        th = [t - u for t, u in zip(th, upd)]
        torch.cuda.synchronize()
        res.append(dict(s=time.perf_counter() - t0, rounds=comm.comm_rounds - r0, bytes=comm.comm_bytes - by0))
    return dict(rank=rank, res=res)

if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    out = run()
    print('T35', json.dumps(out))
