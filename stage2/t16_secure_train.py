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
    ops.LS, ops.EXP_FREE = a.ls, a.exp_free; ops.LN_SC = a.ln_sc
    lab = C['label_ids'].cuda()
    tr = C['train']
    g = torch.Generator().manual_seed(a.seed + 7)                     # same k-shot subset as rl_plain.py
    tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:a.k]]
                       for c in sorted(set(tr['y'].tolist()))])
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

    def flat(ts):
        return crypten.cat([t.flatten() for t in ts])

    def unflat(x, like):
        out_, o = [], 0
        for t in like:
            n_ = t.numel(); out_.append(x[o:o + n_].reshape(t.shape)); o += n_
        return out_

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
    adam_m = adam_v = None; sh_m = sh_v = None
    diag = []; skips = []; stfix = []; bsz = min(a.batch, len(tr_ix)); tsum = 0.0; nstep = 0; loss = torch.tensor(float('nan'))
    for step in range(start + 1, a.steps + 1):
        ix = tr_ix[torch.randperm(len(tr_ix), generator=batch_rng)[:bsz]]
        lens = [int(tr['mask'][i].sum()) for i in ix]
        h0p = torch.cat([tr['h'][i, :l] for i, l in zip(ix, lens)]).double().cuda()
        pos = [int(tr['pos'][i]) for i in ix]; y = tr['y'][ix]
        torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
        ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')          # labels secret-shared
        grads = ops.loss_grad(h0p, lens, pos, th, ysec)
        gmax = None; ok = None
        if a.diag:                                                    # harness-only diagnostic (not protocol)
            gmax = max(float(torch.as_tensor(G.reveal(gr).cpu()).abs().max()) for gr in grads); diag.append(gmax)
        if a.clip > 0:
            # oblivious guard: if ANY entry |g| > tau (truncation failure), the whole step is zeroed (no reveal);
            # otherwise per-entry clamp to [-clip, clip]. One batched comparison on the 65K entries + 1 scalar one.
            shp = [gr.shape for gr in grads]; num = [gr.numel() for gr in grads]
            gc = crypten.cat([gr.flatten() for gr in grads])
            gt = V.cmp_lt_batch(gc, [-a.tau, a.tau, -a.clip, a.clip])
            nbad = (gt[0] + V.neg1p(gt[1])).sum()
            ok = V.cmp_lt_batch(V.to_fx(nbad, gc), [0.5])[0]           # [no entry out of range]
            gc = gc + gt[2] * ((gc * 0 - a.clip) - gc) + V.neg1p(gt[3]) * ((gc * 0 + a.clip) - gc)
            gc = gc * ok
            if a.diag:
                okv = float(torch.as_tensor(G.reveal(ok).cpu()).flatten()[0]); skips.append(int(okv < 0.5))
            grads, o = [], 0
            for sh, n_ in zip(shp, num):
                grads.append(gc[o:o + n_].reshape(sh)); o += n_
        if a.opt == 'adam':
            # secure Adam on the clipped/guarded gradient, in the scaled domain G = S*g (Adam is scale-invariant;
            # S keeps v above the fixed-point resolution). lr_t and eps_t carry the bias corrections (public).
            S_ = a.adam_scale; b1, b2 = 0.9, 0.999
            Gs = [gr * int(S_) for gr in grads]
            if adam_m is None:
                adam_m = [g * 0 for g in Gs]; adam_v = [g * 0 for g in Gs]
            # Adam state guard (oblivious, element-wise): a local-truncation failure lands ~2^32 away, far outside the
            # legal range (|m| <= S*clip, 0 <= v <= (S*clip)^2), so an out-of-range new entry keeps its previous value.
            # v is accumulated as (c*G)^2 with c = sqrt(1-b2) so the squared operand stays small.
            cb2 = math.sqrt(1 - b2); MB = S_ * a.clip * 1.01; VB = (S_ * a.clip) ** 2 * 1.01
            m_new = [m * b1 + g * (1 - b1) for m, g in zip(adam_m, Gs)]
            hs = [g * cb2 for g in Gs]
            v_new = [v * b2 + h * h for v, h in zip(adam_v, hs)]
            mc_old, vc_old = flat(adam_m), flat(adam_v); mc, vc = flat(m_new), flat(v_new)
            gm = V.cmp_lt_batch(mc, [-MB, MB]); gv = V.cmp_lt_batch(vc, [-1.0, VB])
            okm, okv_ = gm[1] - gm[0], gv[1] - gv[0]
            if a.gate and ok is not None:                               # skipped step: Adam state and theta stay put
                okk = crypten.stack([okm, okv_]) * ok
                okm, okv_ = okk[0], okk[1]
            pr = crypten.stack([okm, okv_]) * crypten.stack([mc - mc_old, vc - vc_old])
            adam_m, adam_v = unflat(mc_old + pr[0], Gs), unflat(vc_old + pr[1], Gs)
            if a.diag:
                nst = float(torch.as_tensor(G.reveal((okm.numel() - okm.sum()) + (okv_.numel() - okv_.sum())).cpu()).flatten()[0]); stfix.append(int(round(nst)))
            tstep = step
            lr_t = a.lr * math.sqrt(1 - b2 ** tstep) / (1 - b1 ** tstep)
            eps_t = (a.adam_eps * S_) * math.sqrt(1 - b2 ** tstep)
            upd = []
            for m, v in zip(adam_m, adam_v):
                r = ops.rsqrt_newton(v + eps_t * eps_t)               # ~ 1/(sqrt(v)+eps_t)
                upd.append(m * r)                                       # normalised step q = m / sqrt(v)
            qc = flat(upd); gq = V.cmp_lt_batch(qc, [-a.qclip, a.qclip])  # clamp q to [-Q, Q] (|q| <= ~3.2 legally)
            qc = qc + gq[0] * ((qc * 0 - a.qclip) - qc) + V.neg1p(gq[1]) * ((qc * 0 + a.qclip) - qc)
            if a.gate and ok is not None:
                qc = qc * ok
            upd = [(q * (lr_t * 1024)).div(1024) for q in unflat(qc, Gs)]  # keep lr precision (lr ~ 1e-4)
            th = [t - u for t, u in zip(th, upd)]
        else:
            th = [t - gr * a.lr for t, gr in zip(th, grads)]
        torch.cuda.synchronize(); dt = time.perf_counter() - t0; tsum += dt; nstep += 1
        rr, bb = comm.comm_rounds - r0, comm.comm_bytes - b0
        if rank == 0:                                                 # plaintext FP64 SGD shadow (reference)
            ps = [t.clone().requires_grad_(True) for t in shadow]
            b = get('train', ix)
            loss = F.cross_entropy(tail.forward(ps, *b[:3], lab), b[3])
            gs = torch.autograd.grad(loss, ps)
            if a.clip > 0:
                gs = [gg.clamp(-a.clip, a.clip) for gg in gs]
                if gmax is not None and gmax > a.tau:                 # shadow skips the same (failed) steps
                    gs = [gg * 0 for gg in gs]
            skip_sh = a.gate and a.clip > 0 and gmax is not None and gmax > a.tau
            if a.opt == 'adam' and skip_sh:
                pass                                                  # gated: nothing moves on a skipped step
            elif a.opt == 'adam':
                b1, b2 = 0.9, 0.999
                if sh_m is None:
                    sh_m = [torch.zeros_like(g) for g in gs]; sh_v = [torch.zeros_like(g) for g in gs]
                sh_m = [b1 * m + (1 - b1) * g for m, g in zip(sh_m, gs)]
                sh_v = [b2 * v + (1 - b2) * g * g for v, g in zip(sh_v, gs)]
                lr_t = a.lr * math.sqrt(1 - b2 ** step) / (1 - b1 ** step); eps_t = a.adam_eps * math.sqrt(1 - b2 ** step)
                shadow = [(t - lr_t * (m / torch.sqrt(v + eps_t * eps_t)).clamp(-a.qclip, a.qclip)).detach() for t, m, v in zip(ps, sh_m, sh_v)]
            else:
                shadow = [(t - a.lr * gg).detach() for t, gg in zip(ps, gs)]
        if step % a.eval_every == 0 or step == a.steps:
            record(step, dict(step_s_mean=tsum / nstep, last_rounds=rr, last_bytes=bb, train_loss_plain=float(loss), diag_gmax=diag, skipped=skips, state_fixed=stfix))
            tsum = 0.0; nstep = 0; diag = []; skips = []; stfix = []
    return dict(rank=rank)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, required=True); ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--eval-every', type=int, default=25); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ops', default='bp2'); ap.add_argument('--ls', type=float, default=256.0)
    ap.add_argument('--exp-free', type=int, default=1); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--opt', choices=['sgd', 'adam'], default='sgd'); ap.add_argument('--adam-scale', type=float, default=256.0); ap.add_argument('--adam-eps', type=float, default=3e-5)
    ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--tau', type=float, default=100.0); ap.add_argument('--diag', type=int, default=1); ap.add_argument('--qclip', type=float, default=4.0); ap.add_argument('--gate', type=int, default=1); ap.add_argument('--ln-sc', type=float, default=0.25)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); a.exp_free = bool(a.exp_free)
    crypten.cfg.mpc.provider = 'TTP'
    worker(a)
    print('T16_DONE')
