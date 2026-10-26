"""Sanity gate: emu (plaintext torch, SPU-faithful f=16) vs c_plain references.
Reproduces the CPU-line error ballpark before any CrypTen work."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/workspace/veiltune')
sys.path.insert(0, str(ROOT / 'emu'))
sys.path.insert(0, str(ROOT / 'sjvp'))
from fxp import FXP
from secure_emu import Emu

tag = sys.argv[1] if len(sys.argv) > 1 else 's2000'
samples = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0'])]
dirs = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0', '1'])]

F = dict(np.load(ROOT / f'sjvp/fixture_{tag}.npz'))
W = {k: torch.tensor(v, dtype=torch.float64) for k, v in F.items() if k.startswith('w')}
H = {k: v for k, v in F.items() if k.startswith('head')}
theta = [torch.tensor(F[f'theta{j}'], dtype=torch.float64) for j in range(16)]
us = [[torch.tensor(F[f'u{d}_{j}'], dtype=torch.float64) for j in range(16)] for d in range(16)]
c_same = np.load(ROOT / f'sjvp/c_plain_{tag}_mpc.npy')

torch.manual_seed(0)
fx = FXP(f=16, mode='cheetah', on=True)
emu = Emu(W, fx)


def client_c(h, hdot, y):
    h = torch.tensor(h, dtype=torch.float64, requires_grad=True)
    with torch.enable_grad():
        z = torch.nn.functional.gelu(h @ torch.tensor(H['head_W']) + torch.tensor(H['head_b']), approximate='none')
        c = z - z.mean(-1, keepdim=True)
        var = (c * c).mean(-1, keepdim=True) + 1e-12
        z = c / var.sqrt() * torch.tensor(H['head_g']) + torch.tensor(H['head_be'])
        logits = z @ torch.tensor(H['head_dec']) + torch.tensor(H['head_decb'])
        loss = -torch.log_softmax(logits, -1)[0, y]
        g, = torch.autograd.grad(loss, h)
    return (g * torch.tensor(hdot, dtype=torch.float64)).sum().item()


rows = []
for i in samples:
    mask = torch.tensor(F['mask'][i], dtype=torch.float64)
    pos = torch.tensor([int(F['pos'][i])])
    n = int(F['mask'][i].sum())
    h0 = torch.tensor(F['h0'][i:i + 1, :n])
    for d in dirs:
        h, hd = emu.encoder(h0, mask[:n][None, :], pos, theta, us[d])
        c = client_c(h.numpy(), hd.numpy(), int(F['y'][i]))
        rel = abs(c - c_same[i, d]) / abs(c_same[i, d])
        rel_ref = abs(c - F['c_ref'][i, d]) / abs(F['c_ref'][i, d])
        rows.append(dict(sample=i, dir=d, c=c, c_same=float(c_same[i, d]),
                         rel_same=rel, rel_ref=rel_ref))
        print(rows[-1], flush=True)

cs = np.array([r['c'] for r in rows]); sm = np.array([r['c_same'] for r in rows])
rf = np.array([abs(r['c'] - F['c_ref'][r['sample'], r['dir']]) / abs(F['c_ref'][r['sample'], r['dir']]) for r in rows])
print('EMU_SUMMARY', dict(n=len(rows),
                          rel_l2_same=float(np.linalg.norm(cs - sm) / np.linalg.norm(sm)),
                          max_rel_same=float(max(r['rel_same'] for r in rows)),
                          sign_vs_ref=float((np.sign(cs) == np.sign(
                              np.array([F['c_ref'][r['sample'], r['dir']] for r in rows]))).mean())))
