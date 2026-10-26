"""Plaintext probe (read-only w.r.t. framework): log10-histograms of
 (1) |tanh argument| of the tanh-form GELU and LayerNorm variance in adapted blocks 16-23 (pretrained RoBERTa-large, SST-2 prompts);
 (2) bias-corrected Adam second moment of the adapter during plaintext few-shot fine-tuning (tail model, 16-shot, B=32, lr 1e-4, 100 steps).
Writes stage2/t34_dist.json."""
import json, math, os, sys, numpy as np, torch
from pathlib import Path
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
EDGES = np.linspace(-14, 6, 201)
def hist(x):
    x = np.abs(np.asarray(x, dtype=np.float64)); nz = x[x > 0]
    h, _ = np.histogram(np.log10(nz), bins=EDGES)
    return dict(counts=h.tolist(), n=int(x.size), n_zero=int((x == 0).sum()),
                below=int((np.log10(nz) < EDGES[0]).sum()), above=int((np.log10(nz) >= EDGES[-1]).sum()),
                q=[float(np.quantile(nz, p)) for p in (0.001, 0.01, 0.5, 0.99, 0.999)], max=float(nz.max()), min=float(nz.min()))
out = dict(edges=EDGES.tolist())
# ---- part 1: forward activations
from transformers import AutoModelForMaskedLM, AutoTokenizer
MID = 'FacebookAI/roberta-large'
tok = AutoTokenizer.from_pretrained(MID); m = AutoModelForMaskedLM.from_pretrained(MID).cuda().eval()
acc = dict(var=[], targ=[]); K0 = math.sqrt(2 / math.pi)
def ln_hook(mod, inp):
    x = inp[0].double(); d = x - x.mean(-1, keepdim=True); acc['var'].append(((d * d).mean(-1))[acc['mask']].cpu().numpy())
def g_hook(mod, inp, o):
    z = o.double()[acc['mask']]; acc['targ'].append((K0 * (z + 0.044715 * z ** 3)).abs().flatten().cpu().numpy())
for i in range(16, 24):
    L = m.roberta.encoder.layer[i]
    L.attention.output.LayerNorm.register_forward_pre_hook(ln_hook); L.output.LayerNorm.register_forward_pre_hook(ln_hook)
    L.intermediate.dense.register_forward_hook(g_hook)
sents = []
for n in ('train', 'tune'):
    sents += [d['sentence'] + ' It was <mask> .' for d in json.loads((ROOT / f'data/{n}.json').read_text())]
with torch.no_grad():
    for i in range(0, len(sents), 64):
        b = tok(sents[i:i + 64], return_tensors='pt', padding=True).to('cuda'); acc['mask'] = b['attention_mask'].bool(); m(**b)
out['ln_var'] = hist(np.concatenate(acc['var'])); out['tanh_arg'] = hist(np.concatenate(acc['targ'])); out['n_sent'] = len(sents)
del m; torch.cuda.empty_cache()
# ---- part 2: plaintext Adam second moment on the adapter
import torch.nn.functional as F, t12_rl_secure as T12
tail, W, H, theta, C = T12.load_weights()
for P in tail.layers:
    for k in list(P):
        if torch.is_tensor(P[k]): P[k] = P[k].double().cuda()
for k in list(tail.head): tail.head[k] = tail.head[k].double().cuda()
torch.set_default_dtype(torch.float64)
th = [t.double().cuda().clone().requires_grad_(True) for t in theta]
opt = torch.optim.Adam(th, lr=1e-4)
tr = C['train']; g = torch.Generator().manual_seed(49)
ix_all = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:16]] for c in sorted(set(tr['y'].tolist()))])
vs = []; lab = C['label_ids'].cuda()
for step in range(1, 101):
    ix = ix_all[torch.randperm(len(ix_all), generator=g)[:32]]
    n = int(tr['mask'][ix].sum(1).max())
    lo = tail.forward(th, tr['h'][ix, :n].double().cuda(), tr['mask'][ix, :n].cuda(), tr['pos'][ix].cuda(), lab)
    loss = F.cross_entropy(lo, tr['y'][ix].cuda()); opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 2, 5, 10, 20, 50, 100):
        vh = torch.cat([opt.state[t]['exp_avg_sq'].flatten() / (1 - 0.999 ** step) for t in th]).cpu().numpy(); vs.append(vh)
out['adam_v'] = hist(np.concatenate(vs)); out['adam_note'] = 'bias-corrected v at steps 1,2,5,10,20,50,100; lr 1e-4; B=32; 16-shot SST-2'
(ROOT / 'stage2/t34_dist.json').write_text(json.dumps(out))
print('T34', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'counts'} for k, v in out.items() if isinstance(v, dict)}))
