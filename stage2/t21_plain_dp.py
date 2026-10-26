"""Plaintext DP-SGD reference on the full 1024-sample set (RoBERTa-large tail, Q/V LoRA r=4 on layers 20-23):
Poisson sampling q = 64/1024, per-sample clipping C, Gaussian noise sigma*C, sum/(q*n). sigma from an RDP
accountant (Poisson-subsampled Gaussian, integer orders, delta=1e-5). eps=inf -> clipping only."""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
ROOT = Path('/workspace/veiltune')


def rdp_eps(q, sigma, T, delta, orders=range(2, 257)):
    best = float('inf')
    for a in orders:
        terms = [math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1) + (a - k) * math.log1p(-q)
                 + k * math.log(q) + (k * k - k) / (2 * sigma * sigma) for k in range(a + 1)]
        m = max(terms); lse = m + math.log(sum(math.exp(t - m) for t in terms))
        rdp = T * lse / (a - 1)
        best = min(best, rdp + math.log(1 / delta) / (a - 1))
    return best


def sigma_for(eps, q, T, delta):
    lo, hi = 0.3, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if rdp_eps(q, mid, T, delta) <= eps else (mid, hi)
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=float('inf')); ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--lr', type=float, default=0.05); ap.add_argument('--C', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--eval-every', type=int, default=50)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    C = RP.build_cache(20); tail = RP.Tail(20, list(range(20, 24)), 4); lab = C['label_ids'].cuda()
    n = len(C['train']['y']); q = 64 / n; delta = 1e-5
    sigma = 0.0 if math.isinf(a.eps) else sigma_for(a.eps, q, a.steps, delta)
    out = ROOT / f'dp_plain/eps{a.eps}_lr{a.lr}_C{a.C}_s{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'w')
    def get(split, ix):
        d = C[split]; m = int(d['mask'][ix].sum(1).max())
        return d['h'][ix, :m].cuda(), d['mask'][ix, :m].cuda(), d['pos'][ix].cuda(), d['y'][ix].cuda()
    @torch.no_grad()
    def evaluate(th, split):
        corr = 0
        for ix in torch.arange(len(C[split]['y'])).split(256):
            b = get(split, ix); corr += int((tail.forward(th, *b[:3], lab).argmax(-1) == b[3]).sum())
        return corr / len(C[split]['y'])
    g = torch.Generator().manual_seed(a.seed); gn = torch.Generator(device='cuda').manual_seed(a.seed + 1)
    th = [t.clone() for t in tail.init]
    rec = lambda **kw: (log.write(json.dumps(kw) + '\n'), log.flush(), print(json.dumps(kw), flush=True))
    rec(step=0, sigma=sigma, eps=a.eps, dev=evaluate(th, 'tune'), test=evaluate(th, 'heldout'))
    t0 = time.time(); clipfrac = []
    for step in range(1, a.steps + 1):
        ix = torch.nonzero(torch.rand(n, generator=g) < q).flatten()
        acc = [torch.zeros_like(t) for t in th]
        for j in ix.split(1):
            b = get('train', j)
            ps = [t.clone().requires_grad_(True) for t in th]
            gs = torch.autograd.grad(F.cross_entropy(tail.forward(ps, *b[:3], lab), b[3]), ps)
            nr = float(torch.sqrt(sum((x * x).sum() for x in gs)))
            s = min(1.0, a.C / max(nr, 1e-12)); clipfrac.append(s < 1)
            acc = [u + x * s for u, x in zip(acc, gs)]
        if sigma > 0:
            acc = [u + torch.randn(u.shape, device='cuda', generator=gn) * sigma * a.C for u in acc]
        th = [(t - a.lr * u / (q * n)).detach() for t, u in zip(th, acc)]
        if step % a.eval_every == 0 or step == a.steps:
            rec(step=step, dev=evaluate(th, 'tune'), test=evaluate(th, 'heldout'), clip_frac=sum(clipfrac) / max(len(clipfrac), 1),
                sec=round(time.time() - t0, 1)); clipfrac = []


if __name__ == '__main__':
    main()
