"""Plaintext SGD (k16, lr 5e-3, 300 steps, seeds 42-44): per-step max |grad entry| and grad norm -> clip bound."""
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
C = RP.build_cache(20); tail = RP.Tail(20, list(range(20, 24)), 4); lab = C['label_ids'].cuda(); tr = C['train']
out = {}
for seed in (42, 43, 44):
    g = torch.Generator().manual_seed(seed + 7)
    tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:16]] for c in (0, 1)])
    th = [t.clone() for t in tail.init]; mx = []; nr = []
    for step in range(300):
        n = int(tr['mask'][tr_ix].sum(1).max())
        ps = [t.clone().requires_grad_(True) for t in th]
        loss = F.cross_entropy(tail.forward(ps, tr['h'][tr_ix, :n].cuda(), tr['mask'][tr_ix, :n].cuda(), tr['pos'][tr_ix].cuda(), lab), tr['y'][tr_ix].cuda())
        gs = torch.autograd.grad(loss, ps)
        mx.append(max(float(x.abs().max()) for x in gs)); nr.append(float(torch.cat([x.flatten() for x in gs]).norm()))
        th = [(t - 5e-3 * x).detach() for t, x in zip(ps, gs)]
    out[seed] = dict(max_entry=max(mx), max_entry_step0=mx[0], max_norm=max(nr), norm_step0=nr[0], med_max_entry=sorted(mx)[150])
print('T17', json.dumps(out))
