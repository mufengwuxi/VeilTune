"""End-to-end secure fine-tuning (zero reveal during training), RoBERTa-large layers 20..23 Q/V LoRA r=4, k16 SST-2.
Each step: secure forward + CE + backprop + SGD on shares (bp2 kernels, loss scaling). Every --eval-every steps the
harness reveals theta (what the model owner would get at the end) and evaluates in plaintext; a plaintext FP64 SGD
shadow run with the same init / batches / lr gives the reference trajectory.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12


@mpc.run_multiprocess(world_size=2)
def worker(a):
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp as BPm, gpu_crypten_bp2 as BP2
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    tail, W, H, _, C = T12.load_weights()
    ops = (BP2.SecureOpsBP2 if a.ops == 'bp2' else BPm.SecureOpsBP)(W); ops.init_head(H)
    ops.D, ops.NH, ops.LN_EPS = 1024, 16, G.const(1e-5)
    ops.LS, ops.EXP_FREE = a.ls, a.exp_free
    lab = C['label_ids'].cuda()
    tr = C['train']
    g = torch.Generator().manual_seed(a.seed + 7)                     # same k-shot subset as rl_plain.py
    tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:a.k]]
                       for c in (0, 1)])
    batch_rng = torch.Generator().manual_seed(a.seed + 1000)
    theta0 = [t.double().cpu() for t in tail.init]                   # standard LoRA init (B = 0), owner's
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    start = 0
    if a.resume and (out / 'theta_last.pt').exists():
        ck = torch.load(out / 'theta_last.pt'); theta0, start = ck['theta'], ck['step']; shadow0 = ck['shadow']
        batch_rng.set_state(ck['rng'])
    else:
        shadow0 = [t.clone() for t in theta0]
    th = [crypten.cryptensor(G.enc_pub(t.numpy()).cuda(), src=0, device='cuda') for t in theta0]
    if rank == 0:
        for P in tail.layers:
            for k in list(P):
                if torch.is_tensor(P[k]): P[k] = P[k].double()
        for k in list(tail.head): tail.head[k] = tail.head[k].double()
        shadow = [t.cuda() for t in shadow0]
        log = open(out / 'log.jsonl', 'a')

    def get(split, ix):
        d = C[split]; n = int(d['mask'][ix].sum(1).max())
        return d['h'][ix, :n].double().cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(), d['y'][ix].cuda()

    @torch.no_grad()
    def evaluate(theta, split):
        corr = tot = 0; ls = 0.0
        for ix in torch.arange(len(C[split]['y'])).split(128):
            b = get(split, ix); lo = tail.forward(theta, *b[:3], lab)
            corr += int((lo.argmax(-1) == b[3]).sum()); tot += len(ix); ls += float(F.cross_entropy(lo, b[3], reduction='sum'))
        return corr / tot, ls / tot

    def record(step, extra):
        thr = [G.reveal(t).cpu() for t in th]                         # harness / model-owner output only
        if rank != 0:
            return
        ths = [t.cuda() for t in thr]
        dacc, dloss = evaluate(ths, 'tune'); tacc, _ = evaluate(ths, 'heldout')
        pd, _ = evaluate(shadow, 'tune'); pt, _ = evaluate(shadow, 'heldout')
        a_ = torch.cat([t.flatten() for t in thr]); b_ = torch.cat([t.flatten().cpu() for t in shadow])
        rec = dict(step=step, dev_acc=dacc, dev_loss=dloss, test_acc=tacc, plain_dev_acc=pd, plain_test_acc=pt,
                   rel_dist_to_plain=float((a_ - b_).norm() / max(float(b_.norm()), 1e-12)), **extra)
        log.write(json.dumps(rec) + '\n'); log.flush(); print('T16', json.dumps(rec), flush=True)
        torch.save(dict(theta=thr, shadow=[t.cpu() for t in shadow], step=step, rng=batch_rng.get_state()), out / 'theta_last.pt')

    if start == 0:
        record(0, {})
    bsz = min(a.batch, len(tr_ix)); tsum = 0.0; nstep = 0; loss = torch.tensor(float('nan'))
    for step in range(start + 1, a.steps + 1):
        ix = tr_ix[torch.randperm(len(tr_ix), generator=batch_rng)[:bsz]]
        lens = [int(tr['mask'][i].sum()) for i in ix]
        h0p = torch.cat([tr['h'][i, :l] for i, l in zip(ix, lens)]).double().cuda()
        pos = [int(tr['pos'][i]) for i in ix]; y = tr['y'][ix]
        torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
        ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')          # labels secret-shared
        grads = ops.loss_grad(h0p, lens, pos, th, ysec)
        th = [t - gr * a.lr for t, gr in zip(th, grads)]
        torch.cuda.synchronize(); dt = time.perf_counter() - t0; tsum += dt; nstep += 1
        rr, bb = comm.comm_rounds - r0, comm.comm_bytes - b0
        if rank == 0:                                                 # plaintext FP64 SGD shadow (reference)
            ps = [t.clone().requires_grad_(True) for t in shadow]
            b = get('train', ix)
            loss = F.cross_entropy(tail.forward(ps, *b[:3], lab), b[3])
            gs = torch.autograd.grad(loss, ps)
            shadow = [(t - a.lr * gg).detach() for t, gg in zip(ps, gs)]
        if step % a.eval_every == 0 or step == a.steps:
            record(step, dict(step_s_mean=tsum / nstep, last_rounds=rr, last_bytes=bb, train_loss_plain=float(loss)))
            tsum = 0.0; nstep = 0
    return dict(rank=rank)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, required=True); ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--eval-every', type=int, default=25); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ops', default='bp2'); ap.add_argument('--ls', type=float, default=256.0)
    ap.add_argument('--exp-free', type=int, default=1); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); a.exp_free = bool(a.exp_free)
    crypten.cfg.mpc.provider = 'TTP'
    worker(a)
    print('T16_DONE')
