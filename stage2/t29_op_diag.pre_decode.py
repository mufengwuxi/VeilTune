"""Per-call-site truncation-failure counter for the secure loss_grad (2PC, TTP).
Monkeypatches ArithmeticSharedTensor.mul / matmul / div: when DIAG is on, reveals the operands and the result,
recomputes the op in float64 and counts output elements whose error exceeds max(1, 1e-3*|ref|) (a local-truncation
failure is off by ~2^31 in fixed point, far above rounding). Aggregates by the first caller frame inside the
gpu_crypten*/stage2 sources. Runs TASK's k-shot batches with the trained adapter from --theta (if given).
Usage: TASK=rte SPLIT=16 python t29_op_diag.py --batches 2 --theta sec_train_adam/rte_sp16_s42/theta_last.pt"""
import argparse, json, sys, traceback
from collections import defaultdict
from pathlib import Path
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12


@mpc.run_multiprocess(world_size=2)
def worker(a):
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp2 as BP2
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor as AST
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    rank = crypten.communicator.get().get_rank()
    tail, W, H, _, C = T12.load_weights()
    ops = BP2.SecureOpsBP2(W); ops.init_head(H)
    ops.D, ops.NH, ops.LN_EPS = 1024, 16, G.const(1e-5)
    ops.LS, ops.EXP_FREE, ops.LN_SC = a.ls, True, a.ln_sc

    state = dict(on=False)
    stats = defaultdict(lambda: [0, 0, 0, 0.0])                     # calls, elements, bad, max|ref|

    def site():
        for fr in reversed(traceback.extract_stack()[:-2]):
            f = fr.filename
            if ('gpu_crypten' in f or 'stage2' in f) and 't29_op_diag' not in f:
                return f'{Path(f).name}:{fr.lineno}:{fr.name}'
        return '?'

    def plain(x):
        return x.get_plain_text().double() if isinstance(x, AST) else (
            x.double() if torch.is_tensor(x) else torch.tensor(float(x), dtype=torch.float64))

    def wrap(name, ref_fn):
        orig = getattr(AST, name)

        def f(self, y, *args, **kw):
            out = orig(self, y, *args, **kw)
            if not state['on'] or isinstance(y, int) or (torch.is_tensor(y) and not torch.is_floating_point(y)):
                return out
            state['on'] = False
            try:
                xs, ys, got = plain(self), plain(y).to(self.share.device), plain(out)
                ref = ref_fn(xs, ys)
                bad = ((got - ref).abs() > torch.clamp(ref.abs() * 1e-3, min=1.0)).sum().item()
                clean = float(xs.abs().max()) < 1e6 and float(ys.abs().max()) < 1e6   # inputs not yet corrupted
                key = f'{name}@{site()}'
                s = stats[key]
                s[0] += 1; s[1] += ref.numel(); s[3] = max(s[3], float(ref.abs().max()) if clean else s[3])
                if clean:
                    s[2] += bad
                    if bad and not state.get('first'):
                        state['first'] = (key, bad, float(xs.abs().max()), float(ys.abs().max()) if ys.numel() else 0.0)
            finally:
                state['on'] = True
            return out
        setattr(AST, name, f)

    wrap('mul', lambda x, y: x * y)
    wrap('matmul', lambda x, y: x @ y)
    wrap('div', lambda x, y: x / y)

    tr = C['train']
    g = torch.Generator().manual_seed(a.seed + 7)
    tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:16]]
                       for c in sorted(set(tr['y'].tolist()))])
    theta0 = [t.double().cpu() for t in tail.init]
    if a.theta:
        theta0 = [t.double().cpu() for t in torch.load(ROOT / a.theta)['theta']]
    th = [crypten.cryptensor(G.enc_pub(t.numpy()).cuda(), src=0, device='cuda') for t in theta0]
    rng = torch.Generator().manual_seed(a.seed + 1000)
    nbad_grad = []
    for bi in range(a.batches):
        ix = tr_ix[torch.randperm(len(tr_ix), generator=rng)[:a.batch]]
        lens = [int(tr['mask'][i].sum()) for i in ix]
        h0p = torch.cat([tr['h'][i, :l] for i, l in zip(ix, lens)]).double().cuda()
        pos = [int(tr['pos'][i]) for i in ix]
        ysec = crypten.cryptensor(tr['y'][ix].double().cuda(), src=0, device='cuda')
        state['first'] = None
        state['on'] = True
        grads = ops.loss_grad(h0p, lens, pos, th, ysec)
        state['on'] = False
        gm = max(float(torch.as_tensor(G.reveal(gr).cpu()).abs().max()) for gr in grads)
        nbad_grad.append((gm, state['first']))
    rows = sorted(([k] + v for k, v in stats.items()), key=lambda r: -r[3])
    return dict(rank=rank, rows=rows, gmax=nbad_grad)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=2); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--ls', type=float, default=256.0)
    ap.add_argument('--ln-sc', type=float, default=0.25); ap.add_argument('--theta', default='')
    ap.add_argument('--out', default='stage2/t29_op_diag.json')
    a = ap.parse_args()
    crypten.cfg.mpc.provider = 'TTP'
    res = [r for r in worker(a) if r]
    r1 = [r for r in res if r['rank'] == 1][0]
    (ROOT / a.out).write_text(json.dumps(r1, indent=1))
    print('T29 gmax per batch', r1['gmax'])
    tot = sum(r[3] for r in r1['rows'])
    print('T29 total bad elements', tot)
    for r in r1['rows']:
        if r[3] > 0 or r[4] > 1e3:
            print('T29', r[0], 'calls', r[1], 'elems', r[2], 'bad', r[3], 'max|ref|', round(r[4], 1))
