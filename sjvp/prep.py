"""Export everything the secure-JVP test needs (main torch env).
Per checkpoint: h0 for N public train samples, encoder weights (1/8 folded into W_K),
LoRA A/B (round5 layout), client MLM-head weights (2 label rows), Q directions,
and FP64 HF references: L, c_ref = dL/dtheta.u (JVP), plus c_zo64 for eps."""
import argparse, sys, types, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/workspace/veiltune')
sys.argv = [sys.argv[0]] + sys.argv[1:]
import round5_float as R
from torch.func import functional_call, jvp

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--master', default=None)
p.add_argument('--n', type=int, default=4); p.add_argument('--dirs', type=int, default=16)
p.add_argument('--eps', type=float, default=1e-3)
a = p.parse_args()
args = types.SimpleNamespace(row='zo', rank=4, lr=0.0, seed=42, batch=1, steps=0, eval_every=1,
                             epsilon=a.eps, qdirs=1, save_at=[], output=f'/tmp/sjvp_{a.tag}')
E = R.MiniExp(args)
if a.master:
    E.master = [m.cuda().double() for m in torch.load(a.master)]
E.set_parameters()
model = E.model.double().cpu()
names = E.param_names
params = [q.detach().double().cpu() for q in E.params]
ix = torch.randperm(len(E.data['train'][0]), generator=torch.Generator().manual_seed(4242))[:a.n]
ids, mask, pos, y = (t[ix] for t in E.data['train'])
lab = E.label_ids.cpu()
gen = torch.Generator().manual_seed(777)
us = [[torch.randn(q.shape, generator=gen, dtype=torch.float64) for q in params] for _ in range(a.dirs)]

def loss_i(i, ps):
    lg = functional_call(model, dict(zip(names, ps)), (), dict(input_ids=ids[i:i+1], attention_mask=mask[i:i+1])).logits
    return torch.nn.functional.cross_entropy(lg[0, pos[i]].index_select(-1, lab)[None], y[i:i+1])

L = np.zeros(a.n); cref = np.zeros((a.n, a.dirs)); czo = np.zeros((a.n, a.dirs))
with torch.no_grad():
    for i in range(a.n):
        for d, u in enumerate(us):
            l, c = jvp(lambda *ps: loss_i(i, ps), tuple(params), tuple(u))
            L[i], cref[i, d] = l.item(), c.item()
            lp = loss_i(i, [q + a.eps * v for q, v in zip(params, u)]).item()
            lm = loss_i(i, [q - a.eps * v for q, v in zip(params, u)]).item()
            czo[i, d] = (lp - lm) / (2 * a.eps)
    h0 = model.bert.embeddings(input_ids=ids).numpy()
out = dict(h0=h0, mask=mask.numpy().astype('float64'), pos=pos.numpy(), y=y.numpy(),
           L_ref=L, c_ref=cref, c_zo64=czo, eps=a.eps)
enc = model.bert.encoder.layer
for i, layer in enumerate(enc):
    s = layer.attention.self
    get = lambda m: m.base if hasattr(m, 'base') else m
    W = dict(q=get(s.query), k=get(s.key), v=get(s.value), o=layer.attention.output.dense,
             ff1=layer.intermediate.dense, ff2=layer.output.dense)
    for k, m in W.items():
        f = 0.125 if k == 'k' else 1.0
        out[f'w{i}_{k}'] = m.weight.detach().numpy().T * f
        out[f'w{i}_{k}b'] = m.bias.detach().numpy() * f
    for k, m in dict(ln1=layer.attention.output.LayerNorm, ln2=layer.output.LayerNorm).items():
        out[f'w{i}_{k}'] = m.weight.detach().numpy(); out[f'w{i}_{k}b'] = m.bias.detach().numpy()
hd = model.cls.predictions
out.update(head_W=hd.transform.dense.weight.detach().numpy().T, head_b=hd.transform.dense.bias.detach().numpy(),
           head_g=hd.transform.LayerNorm.weight.detach().numpy(), head_be=hd.transform.LayerNorm.bias.detach().numpy(),
           head_dec=hd.decoder.weight.detach()[lab].numpy().T, head_decb=hd.bias.detach()[lab].numpy())
assert names == [f'bert.encoder.layer.{l}.attention.self.{m}.{x}' for l in range(4) for m in ('query', 'value') for x in 'AB'], names
for j, q in enumerate(params): out[f'theta{j}'] = q.numpy()
for d, u in enumerate(us):
    for j, v in enumerate(u): out[f'u{d}_{j}'] = v.numpy()
np.savez(f'/workspace/veiltune/sjvp/fixture_{a.tag}.npz', **out)
print(a.tag, 'L', L.round(4), 'c_ref[0,:4]', cref[0, :4], 'zo-vs-ref rel',
      np.linalg.norm(czo - cref) / np.linalg.norm(cref), 'act', model.config.hidden_act)
