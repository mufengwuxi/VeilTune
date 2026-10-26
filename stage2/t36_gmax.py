"""t33 + per-step max|g| (pre-clip) of the plaintext replica; read-only diagnostic 2026-10-21"""
"""Plaintext FP64 replica of the t16_lp shadow (reference) run, without MPC.
mode=replay: skip exactly the steps the secure run skipped (read from its log) -> must reproduce plain_test_acc (sanity).
mode=noskip: apply every step -> effect of step skipping on accuracy. Same k-shot subset, batch order, init, Adam, clipping.
(; new file, framework code untouched)"""
import argparse, json, math, os, sys
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12
ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['replay', 'noskip'], required=True); ap.add_argument('--seclog', required=True)
ap.add_argument('--lr', type=float, required=True); ap.add_argument('--steps', type=int, required=True)
ap.add_argument('--seed', type=int, required=True); ap.add_argument('--k', type=int, default=16); ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--qclip', type=float, default=4.0); ap.add_argument('--adam-eps', type=float, default=3e-5)
ap.add_argument('--eval-every', type=int, default=25); ap.add_argument('--out', required=True)
a = ap.parse_args()
tail, W, H, _, C = T12.load_weights(); del W, H
for P in tail.layers:
    for k in list(P):
        if torch.is_tensor(P[k]): P[k] = P[k].double()
for k in list(tail.head): tail.head[k] = tail.head[k].double()
lab = C['label_ids'].cuda(); tr = C['train']
g = torch.Generator().manual_seed(a.seed + 7)
tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:a.k]] for c in sorted(set(tr['y'].tolist()))])
batch_rng = torch.Generator().manual_seed(a.seed + 1000)
shadow = [t.double().cuda() for t in tail.init]
skipped = []
for l in open(a.seclog):
    if '"state_fixed"' in l:
        skipped += json.loads(l[l.index('{'):])['skipped']
def get(split, ix):
    d = C[split]; n = int(d['mask'][ix].sum(1).max())
    return d['h'][ix, :n].double().cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(), d['y'][ix].cuda()
@torch.no_grad()
def evaluate(theta, split):
    corr = tot = 0
    for ix in torch.arange(len(C[split]['y'])).split(128):
        b = get(split, ix); lo = tail.forward(theta, *b[:3], lab); corr += int((lo.argmax(-1) == b[3]).sum()); tot += len(ix)
    return corr / tot
log = open(Path(a.out).with_suffix('.jsonl'), 'w')
GMAX = []; sh_m = sh_v = None; bsz = min(a.batch, len(tr_ix)); b1, b2 = 0.9, 0.999
for step in range(1, a.steps + 1):
    ix = tr_ix[torch.randperm(len(tr_ix), generator=batch_rng)[:bsz]]
    ps = [t.clone().requires_grad_(True) for t in shadow]
    b = get('train', ix)
    loss = F.cross_entropy(tail.forward(ps, *b[:3], lab), b[3])
    graw = torch.autograd.grad(loss, ps); GMAX.append(max(float(gg.abs().max()) for gg in graw)); gs = [gg.clamp(-a.clip, a.clip) for gg in graw]
    skip = a.mode == 'replay' and step <= len(skipped) and skipped[step - 1] == 1
    if not skip:
        if sh_m is None:
            sh_m = [torch.zeros_like(x) for x in gs]; sh_v = [torch.zeros_like(x) for x in gs]
        sh_m = [b1 * m + (1 - b1) * x for m, x in zip(sh_m, gs)]
        sh_v = [b2 * v + (1 - b2) * x * x for v, x in zip(sh_v, gs)]
        lr_t = a.lr * math.sqrt(1 - b2 ** step) / (1 - b1 ** step); eps_t = a.adam_eps * math.sqrt(1 - b2 ** step)
        shadow = [(t - lr_t * (m / torch.sqrt(v + eps_t * eps_t)).clamp(-a.qclip, a.qclip)).detach() for t, m, v in zip(ps, sh_m, sh_v)]
    if step % a.eval_every == 0 or step == a.steps:
        rec = dict(step=step, mode=a.mode, dev_acc=evaluate(shadow, 'tune'), test_acc=evaluate(shadow, 'heldout'))
        rec['gmax'] = GMAX[-a.eval_every:]; log.write(json.dumps(rec) + '\n'); log.flush(); print('T33', json.dumps(rec), flush=True)
