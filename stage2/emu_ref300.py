"""Stage-2 T4 reference: emu/train_emu.py (md5 57ac5f9e...) re-run for the first N steps with
per-step logging of c (and loss), eval at 1/100/200/300. Identical training code path as the
CPU line (same modules, same seed/batch/direction streams, same FXP generator seed); only the
logging cadence differs. Use --check to compare step 1/200 against the CPU-line train.jsonl."""
import argparse, json, sys, time, types
from pathlib import Path
import torch
sys.path.insert(0, '/workspace/veiltune'); sys.path.insert(0, '/workspace/veiltune/emu')
_argv = sys.argv; sys.argv = [sys.argv[0]]
import round5_float as R
from fxp import FXP
from secure_emu import Emu
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='jvp'); ap.add_argument('--fxp', default='16')
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--batch', type=int, default=64); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--eval-at', type=int, nargs='*', default=[1, 100, 200, 300])
ap.add_argument('--unbiased-update', action='store_true')
ap.add_argument('--out', required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
args = types.SimpleNamespace(row='zo', rank=4, lr=a.lr, seed=a.seed, batch=a.batch, steps=a.steps, eval_every=10 ** 9,
                             epsilon=1e-3, qdirs=1, save_at=[], output=str(out / '_r5'))
E = R.MiniExp(args)
model = E.model
on = a.fxp != 'off'
fx = FXP(int(a.fxp) if on else 16, on=on, gen=torch.Generator(device='cuda').manual_seed(a.seed + 3000))
W = {}
for i, layer in enumerate(model.bert.encoder.layer):
    s = layer.attention.self
    g = lambda m: m.base if hasattr(m, 'base') else m
    for k, m in dict(q=g(s.query), k=g(s.key), v=g(s.value), o=layer.attention.output.dense,
                     ff1=layer.intermediate.dense, ff2=layer.output.dense).items():
        f = 0.125 if k == 'k' else 1.0
        W[f'w{i}_{k}'] = m.weight.detach().double().T.contiguous() * f
        W[f'w{i}_{k}b'] = m.bias.detach().double() * f
    for k, m in dict(ln1=layer.attention.output.LayerNorm, ln2=layer.output.LayerNorm).items():
        W[f'w{i}_{k}'] = m.weight.detach().double(); W[f'w{i}_{k}b'] = m.bias.detach().double()
emu = Emu(W, fx)
hd = model.cls.predictions
Hw = dict(W=hd.transform.dense.weight.detach().double(), b=hd.transform.dense.bias.detach().double(),
          g=hd.transform.LayerNorm.weight.detach().double(), be=hd.transform.LayerNorm.bias.detach().double(),
          dec=hd.decoder.weight.detach().double()[E.label_ids], decb=hd.bias.detach().double()[E.label_ids])

def client_loss(h, y):
    z = torch.nn.functional.gelu(h @ Hw['W'].T + Hw['b'])
    z = torch.nn.functional.layer_norm(z, (256,), Hw['g'], Hw['be'], 1e-12)
    return torch.nn.functional.cross_entropy(z @ Hw['dec'].T + Hw['decb'], y)

theta = [fx.enc(m.clone()) for m in E.master]
log = open(out / 'train_steps.jsonl', 'w'); t0 = time.time()

def evaluate():
    E.master = [t.clone() for t in theta]; E.set_parameters()
    with torch.no_grad():
        return E.evaluate()

acc0 = evaluate()
log.write(json.dumps(dict(step=0, acc=acc0)) + '\n')
for step in range(1, a.steps + 1):
    ix = E.batch()
    ids, mask, pos, y = (t[ix].cuda() for t in E.data['train'])
    n = int(mask.sum(1).max())
    with torch.no_grad():
        h0 = model.bert.embeddings(input_ids=ids[:, :n]).double()
    m = mask[:, :n].double()
    u = [torch.randn(p.shape, device='cuda', generator=E.direction_rng).double() for p in E.params]
    with torch.no_grad():
        h, ht = emu.encoder(h0, m, pos, theta, u)
    hh = h.detach().requires_grad_(True)
    loss = client_loss(hh, y); gr, = torch.autograd.grad(loss, hh)
    c = float((gr * ht).sum()); L = float(loss)
    with torch.no_grad():
        lc = fx.q(torch.tensor(a.lr * c, dtype=torch.float64, device='cuda'))
        corr = (0.5 / fx.S) if (on and a.unbiased_update) else 0.0
        theta = [t - (fx.q(lc * v) + corr) for t, v in zip(theta, u)]
    rec = dict(step=step, c=c, loss=L, batch_head=[int(i) for i in ix[:4]])
    if step in a.eval_at:
        rec['acc'] = evaluate()
        torch.save([t.cpu() for t in theta], out / f'theta_{step}.pt')
    log.write(json.dumps(rec) + '\n'); log.flush()
    if step in a.eval_at:
        print(json.dumps(rec), round(time.time() - t0, 1), flush=True)
print('DONE', flush=True)
