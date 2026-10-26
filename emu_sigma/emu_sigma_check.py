"""Step-4 pre-check driver: SIGMA-quantization (f=12, floor truncation, gelu'
LUT) on the 16-group protocol (samples 0-3 x dirs 2-5) + dirs 0/1 reference.
Run twice: --sigma-primal false (spec primary) and true (extra info)."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/workspace/veiltune')
sys.path.insert(0, str(ROOT / 'emu_sigma'))
import fxp_sigma
import secure_emu_sigma as SE

ap = argparse.ArgumentParser()
ap.add_argument('--sigma-primal', default='false')
ap.add_argument('--mode', default='sigma')
ap.add_argument('--f', type=int, default=12)
ap.add_argument('--tangent-lut', default='true')
ap.add_argument('--lut-mode', default='round8', choices=['round8', 'floor8', 'floor10'])
ap.add_argument('--rsqrt', default='spu', choices=['spu', 'sigma'])
ap.add_argument('--rsqrt-mbits', type=int, default=7)
ap.add_argument('--w-frac', type=int, default=None)
ap.add_argument('--theta-frac', type=int, default=None)
ap.add_argument('--tan-scale', type=float, default=1.0, help='public kappa: run with kappa*u, client divides c by kappa (JVP is linear in u)')
a = ap.parse_args()
SE.SIGMA_PRIMAL[0] = a.sigma_primal == 'true'
SE.TANGENT_LUT[0] = a.tangent_lut == 'true'
SE.LUT_MODE[0] = a.lut_mode
SE.RSQRT_MODE[0] = a.rsqrt
SE.RSQRT_MBITS[0] = a.rsqrt_mbits
SE.W_FRAC[0] = a.w_frac
SE.THETA_FRAC[0] = a.theta_frac

F = dict(np.load(ROOT / 'sjvp/fixture_s2000.npz'))
W = {k: torch.tensor(v, dtype=torch.float64) for k, v in F.items() if k.startswith('w')}
H = {k: v for k, v in F.items() if k.startswith('head')}
theta = [torch.tensor(F[f'theta{j}'], dtype=torch.float64) for j in range(16)]
us = [[torch.tensor(F[f'u{d}_{j}'], dtype=torch.float64) for j in range(16)] for d in range(16)]
c_same = np.load(ROOT / 'sjvp/c_plain_s2000_mpc.npy')

torch.manual_seed(0)
fx = fxp_sigma.FXP(f=a.f, mode=a.mode, on=True)
emu = SE.Emu(W, fx)


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
for i in [0, 1, 2, 3]:
    mask = torch.tensor(F['mask'][i], dtype=torch.float64)
    pos = torch.tensor([int(F['pos'][i])])
    n = int(F['mask'][i].sum())
    h0 = torch.tensor(F['h0'][i:i + 1, :n])
    for d in ([0, 1, 2, 3, 4, 5] if i == 0 else [2, 3, 4, 5]):
        h, hd = emu.encoder(h0, mask[:n][None, :], pos, theta, [a.tan_scale * v for v in us[d]])
        c = client_c(h.numpy(), hd.numpy(), int(F['y'][i])) / a.tan_scale
        rows.append(dict(sample=i, dir=d, c=c, c_same=float(c_same[i, d]),
                         rel_same=abs(c - c_same[i, d]) / abs(c_same[i, d]),
                         rel_ref=abs(c - F['c_ref'][i, d]) / abs(F['c_ref'][i, d])))

g16 = [r for r in rows if not (r['sample'] == 0 and r['dir'] in (0, 1))]
cs = np.array([r['c'] for r in g16]); sm = np.array([r['c_same'] for r in g16])
rf = np.array([F['c_ref'][r['sample'], r['dir']] for r in g16])
summary = dict(f=a.f, mode=a.mode, sigma_primal=a.sigma_primal, tangent_lut=a.tangent_lut, lut_mode=a.lut_mode, rsqrt=a.rsqrt, rsqrt_mbits=a.rsqrt_mbits, tan_scale=a.tan_scale, w_frac=a.w_frac, theta_frac=a.theta_frac,
               n=len(g16),
               rel_l2_same=float(np.linalg.norm(cs - sm) / np.linalg.norm(sm)),
               rel_l2_ref=float(np.linalg.norm(cs - rf) / np.linalg.norm(rf)),
               max_rel_same=float(max(r['rel_same'] for r in g16)),
               sign_ref=float((np.sign(cs) == np.sign(rf)).mean()),
               dirs01=[{k: r[k] for k in ('sample', 'dir', 'rel_same', 'rel_ref')}
                       for r in rows if r['sample'] == 0 and r['dir'] in (0, 1)])
print('SIGMA_PRECHECK', json.dumps(summary, indent=1))
out = ROOT / 'emu_sigma' / (f'precheck_f{a.f}_{a.mode}_lut{a.tangent_lut}_primal{a.sigma_primal}' + ('' if a.lut_mode == 'round8' else f'_{a.lut_mode}') + ('' if a.rsqrt == 'spu' else '_rsqrtsigma' + ('' if a.rsqrt_mbits == 7 else f'm{a.rsqrt_mbits}')) + ('' if a.tan_scale == 1.0 else f'_k{int(a.tan_scale)}') + ('' if a.w_frac is None else f'_w{a.w_frac}') + ('' if a.theta_frac is None else f'_t{a.theta_frac}') + '.json')
out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
