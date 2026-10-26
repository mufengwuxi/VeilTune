"""Plaintext ranges inside the secure tail (layers 20..23 + head): attention-score spread per row (real keys),
GELU input range. Justifies clamp-free exp and comparison bit widths."""
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
C = RP.build_cache(20); tail = RP.Tail(20, list(range(20, 24)), 4); lab = C['label_ids'].cuda()
st = dict(spread=[], smax=[], gz=[])
osm, ogelu = torch.softmax, F.gelu
def sm(s, dim=-1, **k):
    fin = s > -1e30
    mx = torch.where(fin, s, torch.full_like(s, -1e38)).amax(-1); mn = torch.where(fin, s, torch.full_like(s, 1e38)).amin(-1)
    st['spread'].append((mx - mn).flatten().cpu()); st['smax'].append(s.where(fin, torch.zeros_like(s)).abs().amax().item())
    return osm(s, dim, **k)
def ge(z, *a, **k):
    st['gz'].append(z.abs().flatten().cpu()); return ogelu(z, *a, **k)
torch.softmax = sm; F.gelu = ge
g = torch.Generator().manual_seed(5)
theta = [t if t.shape[0] == 4 else (torch.randn(t.shape, generator=g) * 0.02).cuda() for t in tail.init]
out = {}
for split in ('train', 'heldout'):
    d = C[split]
    for k in st: st[k] = []
    with torch.no_grad():
        for ix in torch.arange(len(d['y'])).split(128):
            n = int(d['mask'][ix].sum(1).max())
            tail.forward(theta, d['h'][ix, :n].cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(), lab)
    sp = torch.cat(st['spread']); gz = torch.cat(st['gz'])
    out[split] = dict(spread_max=float(sp.max()), spread_q999=float(sp.quantile(0.999)), spread_q99=float(sp.quantile(0.99)),
                      score_absmax=max(st['smax']), gelu_absmax=float(gz.max()), gelu_frac_ge4=float((gz >= 4).float().mean()),
                      gelu_q9999=float(gz.float().quantile(0.9999)) if gz.numel() < 2**24 else float(gz[torch.randperm(gz.numel())[:2**23]].float().quantile(0.9999)))
print('T15', json.dumps(out))
