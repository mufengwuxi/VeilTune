"""FP64 plaintext: (a) exact JAX vs torch HF reference; (b) MPC-kernel JAX vs reference."""
import sys, json, numpy as np, jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, '/workspace/veiltune/sjvp')
from model import *
res = {}
for tag in ['init', 's2000']:
    F, W, H, theta, us = load(f'/workspace/veiltune/sjvp/fixture_{tag}.npz')
    for kind in ['exact', 'mpc']:
        cs = np.zeros_like(F['c_ref']); Ls = np.zeros(len(F['y']))
        for i in range(len(F['y'])):
            f = lambda th: encoder(F['h0'][i:i+1], W, th, F['mask'][i], int(F['pos'][i]), kind)
            for d, u in enumerate(us):
                h, hd = jax.jvp(f, (theta,), (u,))
                Ls[i], cs[i, d] = jax.jvp(lambda hh: client_loss(hh, H, int(F['y'][i])), (h,), (hd,))
        ref = F['c_ref']
        res[f'{tag}_{kind}'] = dict(L_maxdiff=float(np.abs(Ls - F['L_ref']).max()),
            c_rel=float(np.linalg.norm(cs - ref) / np.linalg.norm(ref)),
            pearson=float(np.corrcoef(cs.ravel(), ref.ravel())[0, 1]),
            sign=float((np.sign(cs) == np.sign(ref)).mean()))
        np.save(f'/workspace/veiltune/sjvp/c_plain_{tag}_{kind}.npy', cs)
        print(tag, kind, res[f'{tag}_{kind}'], flush=True)
json.dump(res, open('/workspace/veiltune/sjvp/plain_check.json', 'w'), indent=1)
