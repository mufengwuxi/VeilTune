"""Which CrypTen-native kernel breaks at RoBERTa-large ranges? max abs / rel error per kernel vs float64."""
import json, math, sys
from pathlib import Path
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_native as NAT, gpu_crypten_bp2 as BP2
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    RV = lambda t: torch.as_tensor(G.reveal(t).cpu()).double()
    torch.manual_seed(0); out = {}
    nat, ours = NAT.SecureOpsNative({}), BP2.SecureOpsBP2({}); ours.EXP_FREE = True
    C = lambda x: crypten.cryptensor(x.cuda(), device='cuda')
    z = torch.linspace(-36, 36, 14401).double()
    ref = z * 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    for n, o in (('native', nat), ('ours', ours)):
        y, _ = o.gelu_save(C(z)); e = (RV(y) - ref).abs()
        out[f'gelu_{n}'] = dict(max_abs=float(e.max()), at=float(z[e.argmax()]))
    v = torch.logspace(-3, 5, 801).double()
    for n, f in (('native', lambda c: c.inv_sqrt()), ('ours', lambda c: ours.rsqrt_newton(c))):
        r = RV(f(C(v))); rel = ((r - v.rsqrt()) / v.rsqrt()).abs()
        bad = v[rel > 0.05]
        out[f'rsqrt_{n}'] = dict(max_rel=float(rel.max()), ok_range=[float(v[rel < 0.01].min()), float(v[rel < 0.01].max())], n_bad=int((rel > 0.05).sum()))
    for spread in (5.0, 20.0, 47.0):
        s = (torch.rand(4, 16, 8, 49).double() * spread); s[..., 0] = 0
        mask = torch.ones(4, 1, 1, 49).double()
        pr = torch.softmax(s, -1)
        for n, o in (('native', nat), ('ours', ours)):
            p = RV(o.softmax_save(C(s), mask.cuda()))
            out[f'softmax_{n}_spread{int(spread)}'] = dict(max_abs=float((p - pr).abs().max()))
    sden = torch.tensor([1.0, 3.0, 10.0, 30.0, 49.0, 100.0, 1000.0]).double()
    out['reciprocal_native'] = dict(zip([str(float(x)) for x in sden], RV(C(sden).reciprocal()).tolist()))
    x = torch.linspace(-20, 20, 4001).double()
    for n, o in (('native', nat), ('ours', ours)):
        e = (RV(o._sigmoid(C(x))) - torch.sigmoid(x)).abs()
        out[f'sigmoid_{n}'] = dict(max_abs=float(e.max()), at=float(x[e.argmax()]))
    return dict(rank=crypten.communicator.get().get_rank(), out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = {r['rank']: r['out'] for r in run() if r}
    (ROOT / 'baselines/t19_native_kernels.json').write_text(json.dumps(res, indent=1))
    for k, v in res[0].items():
        print('T19', k, json.dumps(v))
