"""Secure U-split step in the SPU simulator (Cheetah 2PC, FM64).
mode=jvp: secure blocks return (h, h_dot) at MASK; client c = dL/dh . h_dot.
mode=zo : secure blocks return (h(th+eps u), h(th-eps u)); client c = (L+ - L-)/2eps.
Compares c with (i) same-kernel FP64 plaintext JVP and (ii) exact HF FP64 reference."""
import argparse, json, sys, time, re, collections
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, '/workspace/veiltune/sjvp')
import model as M
from model import encoder, client_loss, load
import spu.libspu as libspu, spu.utils.simulation as spsim

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='s2000'); ap.add_argument('--bits', type=int, required=True)
ap.add_argument('--mode', choices=['jvp', 'zo'], required=True)
ap.add_argument('--samples', type=int, nargs='+', default=[0]); ap.add_argument('--dirs', type=int, nargs='+', default=[0])
ap.add_argument('--repeats', type=int, default=1); ap.add_argument('--eps', type=float, default=1e-3)
ap.add_argument('--out', required=True); ap.add_argument('--trim', action='store_true'); ap.add_argument('--lnpre', type=float, default=64.0); ap.add_argument('--last-row', action='store_true')
a = ap.parse_args()
M.LN_PRE[0] = a.lnpre
F, W, H, theta, us = load(f'/workspace/veiltune/sjvp/fixture_{a.tag}.npz')
c_same = np.load(f'/workspace/veiltune/sjvp/c_plain_{a.tag}_mpc.npy')
cfg = libspu.RuntimeConfig(protocol=libspu.ProtocolKind.CHEETAH, field=libspu.FieldType.FM64,
                           fxp_fraction_bits=a.bits)
sim = spsim.Simulator(2, cfg)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = []

def client_c(i, r0, r1):
    y = int(F['y'][i]); r0 = np.asarray(r0, np.float64); r1 = np.asarray(r1, np.float64)
    with jax.experimental.enable_x64():
        Hd = {k: jnp.asarray(v, jnp.float64) for k, v in H.items()}
        if a.mode == 'jvp':
            _, c = jax.jvp(lambda h: client_loss(h, Hd, y), (jnp.asarray(r0),), (jnp.asarray(r1),))
            return float(c)
        return float((client_loss(jnp.asarray(r0), Hd, y) - client_loss(jnp.asarray(r1), Hd, y)) / (2 * a.eps))

import jax.experimental
for i in a.samples:
    mask, pos = F['mask'][i].astype(np.float32), int(F['pos'][i])
    if a.mode == 'jvp':
        def fn(h0, Wd, th, u):
            return jax.jvp(lambda t: encoder(h0, Wd, t, mask, pos, trim=a.trim, last_row=a.last_row), (th,), (u,))
    else:
        def fn(h0, Wd, th, u):
            tp = [t + a.eps * v for t, v in zip(th, u)]; tm = [t - a.eps * v for t, v in zip(th, u)]
            return encoder(h0, Wd, tp, mask, pos, trim=a.trim, last_row=a.last_row), encoder(h0, Wd, tm, mask, pos, trim=a.trim, last_row=a.last_row)
    sf = spsim.sim_jax(sim, fn)
    f32 = lambda z: np.asarray(z, np.float32)
    for d in a.dirs:
        for rep in range(a.repeats):
            t0 = time.time()
            r0, r1 = sf(f32(F['h0'][i:i+1]), {k: f32(v) for k, v in W.items()},
                        [f32(t) for t in theta], [f32(v) for v in us[d]])
            dt = time.time() - t0
            c = client_c(i, r0, r1)
            row = dict(sample=i, dir=d, rep=rep, c=c, c_same=float(c_same[i, d]), c_ref=float(F['c_ref'][i, d]),
                       rel_same=abs(c - c_same[i, d]) / abs(c_same[i, d]), rel_ref=abs(c - F['c_ref'][i, d]) / abs(F['c_ref'][i, d]),
                       seconds=dt)
            rows.append(row); print(json.dumps(row), flush=True)
            (out / 'rows.json').write_text(json.dumps(rows, indent=1))
    if i == a.samples[0]:
        txt = sf.pphlo
        (out / 'pphlo.mlir').write_text(txt)
        ops = collections.Counter(re.findall(r'pphlo\.(\w+)', txt))
        (out / 'ops.json').write_text(json.dumps(dict(ops.most_common()), indent=1))
cs = np.array([r['c'] for r in rows]); sm = np.array([r['c_same'] for r in rows]); rf = np.array([r['c_ref'] for r in rows])
summ = dict(mode=a.mode, bits=a.bits, tag=a.tag, n=len(rows),
            rel_l2_same=float(np.linalg.norm(cs - sm) / np.linalg.norm(sm)),
            rel_l2_ref=float(np.linalg.norm(cs - rf) / np.linalg.norm(rf)),
            max_rel_same=float(max(r['rel_same'] for r in rows)),
            sign_ref=float((np.sign(cs) == np.sign(rf)).mean()),
            mean_seconds=float(np.mean([r['seconds'] for r in rows])),
            trunc_allow_msb_error=bool(cfg.trunc_allow_msb_error))
(out / 'summary.json').write_text(json.dumps(summ, indent=1)); print('SUMMARY', json.dumps(summ), flush=True)
