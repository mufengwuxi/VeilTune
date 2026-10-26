"""Plaintext reproduction of SecP-Tuning's training algorithm (BBT-style black-box prompt tuning).

SecP-Tuning = forward-only prompt tuning optimized by CMA-ES, as in BBT (Sun et al., ICML'22):
  - 50 continuous prompt tokens prepended after <s> in the input embedding
  - prompt = A z, z in R^500, A fixed random normal with std = std(word emb) / (sqrt(500) * sigma)
  - CMA-ES, popsize 20, sigma 1, budget 8000 forward evaluations (400 generations)
  - objective: cross-entropy over label words at <mask> on the whole k-shot training set
Secure execution and its softmax approximation (RFA) only add error, so exact-softmax plaintext
is an upper bound on SecP-Tuning's accuracy. Same data (k-shot sample, dev, test), templates and
label words as rl_plain.py; checkpoint selected on dev every 25 generations.
"""
import argparse, json, math, os, time
from pathlib import Path
ROOT = Path('/workspace/veiltune')
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np, torch, torch.nn.functional as F, cma
from transformers import AutoModelForMaskedLM, AutoTokenizer
import rl_plain as R   # TASK env picks task; reuse encode/templates/label words/data paths

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--kshot', type=int, default=16)
ap.add_argument('--n-prompt', type=int, default=50)
ap.add_argument('--dim', type=int, default=500)
ap.add_argument('--pop', type=int, default=20)
ap.add_argument('--sigma', type=float, default=1.0)
ap.add_argument('--budget', type=int, default=8000)
ap.add_argument('--eval-every', type=int, default=25)
ap.add_argument('--chunk', type=int, default=64)
ap.add_argument('--out', required=True)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
log = open(out / 'log.jsonl', 'w')

tok = R.AutoTokenizer.from_pretrained(R.MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(R.MODEL_ID, torch_dtype=torch.float16).cuda().eval()
emb = model.roberta.embeddings.word_embeddings.weight            # (V, 1024)
lab = torch.tensor([tok.encode(w, add_special_tokens=False)[0] for w in R.LABEL_WORDS]).cuda()

def load(name):
    ids, mask, pos, y = R.encode(json.loads((R.DATA / f'{name}.json').read_text()), tok)
    return dict(ids=ids, mask=mask, pos=pos, y=y)
D = {s: load(s) for s in ['train', 'tune', 'heldout']}
tr = D['train']                                                     # same k-shot sample as rl_plain.py
g = torch.Generator().manual_seed(a.seed + 7)
tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:a.kshot]]
                   for c in sorted(set(tr['y'].tolist()))])
TR = {k: v[tr_ix] for k, v in tr.items()}

std_hat = float(emb.float().std())
pg = torch.Generator(device='cuda').manual_seed(a.seed + 5000)
A = torch.randn(a.n_prompt * emb.shape[1], a.dim, device='cuda', generator=pg) * (std_hat / (math.sqrt(a.dim) * a.sigma))
P = a.n_prompt

@torch.no_grad()
def logits(d, prompts):
    """d: split dict; prompts: (K, P, 1024) -> (K, n, C) label-word logits at <mask>."""
    n = len(d['y']); L = int(d['mask'].sum(1).max())
    ids, mask, pos = d['ids'][:, :L].cuda(), d['mask'][:, :L].cuda(), d['pos'].cuda()
    we = emb[ids]                                                   # (n, L, H)
    res = []
    for p in prompts:
        outk = []
        for s in range(0, n, a.chunk):
            e = we[s:s + a.chunk]; m = mask[s:s + a.chunk]; b = e.shape[0]
            x = torch.cat([e[:, :1], p.half().unsqueeze(0).expand(b, -1, -1), e[:, 1:]], 1)
            mm = torch.cat([m[:, :1], torch.ones(b, P, dtype=m.dtype, device=m.device), m[:, 1:]], 1)
            h = model.roberta(inputs_embeds=x, attention_mask=mm).last_hidden_state
            hm = h[torch.arange(b, device=h.device), pos[s:s + a.chunk] + P]
            outk.append(model.lm_head(hm).float()[:, lab])
        res.append(torch.cat(outk))
    return torch.stack(res)

def zp(z): return (A @ torch.as_tensor(z, device='cuda', dtype=torch.float32)).view(1, P, -1)

def evaluate(name, z):
    lo = logits(D[name], zp(z))[0]; y = D[name]['y'].cuda(); pr = lo.argmax(-1)
    acc = float((pr == y).float().mean())
    f1 = np.mean([2 * float(((pr == c) & (y == c)).sum()) / max(1, float((pr == c).sum() + (y == c).sum())) for c in range(len(lab))])
    return acc, float(f1), float(F.cross_entropy(lo, y))

def record(gen, z, extra=None):
    da, df, dl = evaluate('tune', z); ta, tf, _ = evaluate('heldout', z)
    r = dict(gen=gen, evals=gen * a.pop, dev_acc=da, dev_f1=df, dev_loss=dl, test_acc=ta, test_f1=tf, **(extra or {}))
    log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)

es = cma.CMAEvolutionStrategy(np.zeros(a.dim), a.sigma, {'popsize': a.pop, 'seed': a.seed + 1, 'verbose': -9})
record(0, np.zeros(a.dim))
ytr = TR['y'].cuda(); t0 = time.time()
for gen in range(1, a.budget // a.pop + 1):
    Z = es.ask()
    pr = torch.cat([zp(z) for z in Z])
    lo = logits(TR, pr)                                             # (pop, n, C)
    losses = [float(F.cross_entropy(lo[k], ytr)) for k in range(a.pop)]
    es.tell(Z, losses)
    if gen % a.eval_every == 0:
        record(gen, es.mean, dict(train_loss_best=min(losses), sec=round(time.time() - t0, 1)))
np.save(out / 'z_final.npy', es.mean)
json.dump(vars(a) | dict(task=R.TASK, n_train=len(tr_ix)), open(out / 'args.json', 'w'))
print('DONE', flush=True)
