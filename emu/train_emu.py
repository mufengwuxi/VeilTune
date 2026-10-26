"""U-split secure ZO/JVP LoRA training with the secure blocks EMULATED in fixed point (GPU).
Client (plaintext float): embeddings, real MLM head + label words, loss, c.
Server (emulated SPU, FM64/f, Cheetah-calibrated truncation): 4 blocks with MPC kernels, LoRA state,
update theta <- theta - q(q(lr*c)*u). Same batch/direction streams as round5_float (seed 42).
arms: --fxp off (FP64, MPC kernels) | --fxp 16 ;  --mode jvp | zo"""
import argparse, json, sys, time, types
from pathlib import Path
import torch
sys.path.insert(0, '/workspace/veiltune'); sys.path.insert(0, '/workspace/veiltune/emu')
sys.argv_saved = sys.argv; sys.argv = [sys.argv[0]]
import round5_float as R
from fxp import FXP
from secure_emu import Emu
sys.argv = sys.argv_saved

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['jvp', 'zo'], required=True)
ap.add_argument('--fxp', default='16')
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--eps', type=float, default=1e-3)
ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--eval-every', type=int, default=200); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--save-at', type=int, nargs='*', default=[2000, 10000, 20000])
ap.add_argument('--out', required=True)
ap.add_argument('--unbiased-update', action='store_true', help='add public +0.5 ulp to each truncated update term (cancels floor-truncation bias)')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
args = types.SimpleNamespace(row='zo', rank=4, lr=a.lr, seed=a.seed, batch=a.batch, steps=a.steps, eval_every=a.eval_every,
                             epsilon=a.eps, qdirs=1, save_at=[], output=str(out / '_r5'))
E = R.MiniExp(args)
model = E.model
on = a.fxp != 'off'
fx = FXP(int(a.fxp) if on else 16, on=on, gen=torch.Generator(device='cuda').manual_seed(a.seed + 3000))
# server weights (1/sqrt(d_h) folded into frozen W_K), FP64
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

def client_loss(h, y):  # real MLM transform (exact erf GELU) + 2 label-word rows, plaintext
    z = torch.nn.functional.gelu(h @ Hw['W'].T + Hw['b'])
    z = torch.nn.functional.layer_norm(z, (256,), Hw['g'], Hw['be'], 1e-12)
    return torch.nn.functional.cross_entropy(z @ Hw['dec'].T + Hw['decb'], y)

theta = [fx.enc(m.clone()) for m in E.master]           # secret LoRA state (fixed point)
log = open(out / 'train.jsonl', 'w')
best = (-1, 0); t0 = time.time()

def evaluate():
    E.master = [t.clone() for t in theta]; E.set_parameters()
    with torch.no_grad():
        return E.evaluate()

for step in range(1, a.steps + 1):
    ix = E.batch()
    ids, mask, pos, y = (t[ix].cuda() for t in E.data['train'])
    n = int(mask.sum(1).max())
    with torch.no_grad():
        h0 = model.bert.embeddings(input_ids=ids[:, :n]).double()     # client, plaintext
    m = mask[:, :n].double()
    u = [torch.randn(p.shape, device='cuda', generator=E.direction_rng).double() for p in E.params]
    with torch.no_grad():
        if a.mode == 'jvp':
            h, ht = emu.encoder(h0, m, pos, theta, u)
        else:
            hp, _ = emu.encoder(h0, m, pos, [t + a.eps * v for t, v in zip(theta, u)])
            hm, _ = emu.encoder(h0, m, pos, [t - a.eps * v for t, v in zip(theta, u)])
    if a.mode == 'jvp':
        hh = h.detach().requires_grad_(True)
        loss = client_loss(hh, y); gr, = torch.autograd.grad(loss, hh)
        c = float((gr * ht).sum()); L = float(loss)
    else:
        with torch.no_grad():
            Lp, Lm = float(client_loss(hp, y)), float(client_loss(hm, y))
        c = (Lp - Lm) / (2 * a.eps); L = (Lp + Lm) / 2
    with torch.no_grad():
        lc = fx.q(torch.tensor(a.lr * c, dtype=torch.float64, device='cuda'))   # secret lr*c
        corr = (0.5 / fx.S) if (on and a.unbiased_update) else 0.0
        theta = [t - (fx.q(lc * v) + corr) for t, v in zip(theta, u)]
    if step % a.eval_every == 0 or step == 1:
        acc = evaluate()
        if acc > best[0]: best = (acc, step)
        rec = dict(step=step, acc=acc, loss=L, c=c, best=best[0], best_step=best[1], sec=round(time.time() - t0, 1))
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)
    if step in a.save_at:
        torch.save([t.cpu() for t in theta], out / f'theta_{step}.pt')
json.dump(dict(args=vars(a), best=best[0], best_step=best[1]), open(out / 'summary.json', 'w'))
print('DONE', best, flush=True)
