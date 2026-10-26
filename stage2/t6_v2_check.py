"""Stage-2 phase A check for gpu_crypten_v2 (TTP provider).
  precision : fixture_s2000, samples 0-3 x dirs 2-5 (16 groups) + sample 0 dirs 0/1, c vs c_plain_mpc / c_ref
  equiv     : real batch B=8: v1 padded encoder vs v2 packed encoder -> rel L2 of h, h_dot, c
  profile   : rounds / P0-P1 wire bytes (rank1 log) / step time per op context, b1_L10 and b64 (packed)
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


def setup():
    import gpu_crypten as G
    import gpu_crypten_v2 as V
    crypten.init()
    G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    return G, V


@mpc.run_multiprocess(world_size=2)
def precision():
    G, V = setup()
    Fd, W, H, theta, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    Ht = {k: torch.tensor(v) for k, v in H.items()}
    c_same = np.load(ROOT / 'sjvp/c_plain_s2000_mpc.npy')
    ops = V.SecureOpsV2(W)
    th = G.to_ct_list(theta)
    rows = []
    for i in range(4):
        n = int(Fd['mask'][i].sum())
        h0p = torch.tensor(Fd['h0'][i, :n], dtype=torch.float64, device='cuda')
        for d in ([0, 1, 2, 3, 4, 5] if i == 0 else [2, 3, 4, 5]):
            t0 = time.time()
            h_ct, hd_ct = ops.encoder_packed(h0p, [n], [int(Fd['pos'][i])], th, G.to_ct_list(us[d]))
            h, hd = G.reveal(h_ct).cpu(), G.reveal(hd_ct).cpu()
            c = G.client_c(h, hd, Ht, int(Fd['y'][i]))
            rows.append(dict(sample=i, dir=d, c=c, c_same=float(c_same[i, d]), c_ref=float(Fd['c_ref'][i, d]),
                             rel_same=abs(c - c_same[i, d]) / abs(c_same[i, d]), sec=time.time() - t0))
    g16 = [r for r in rows if not (r['sample'] == 0 and r['dir'] in (0, 1))]
    cs = np.array([r['c'] for r in g16]); sm = np.array([r['c_same'] for r in g16]); rf = np.array([r['c_ref'] for r in g16])
    return dict(rank=crypten.communicator.get().get_rank(), rows=rows, summary=dict(
        n=len(g16), rel_l2_same=float(np.linalg.norm(cs - sm) / np.linalg.norm(sm)),
        rel_l2_ref=float(np.linalg.norm(cs - rf) / np.linalg.norm(rf)),
        max_rel_same=float(max(r['rel_same'] for r in g16)), sign_ref=float((np.sign(cs) == np.sign(rf)).mean()),
        dirs01=[(r['dir'], r['rel_same']) for r in rows if r['sample'] == 0 and r['dir'] in (0, 1)]))


@mpc.run_multiprocess(world_size=2)
def equiv(B):
    G, V = setup()
    Fd, W, H, theta, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    v1, v2 = G.SecureOps(W), V.SecureOpsV2(W)
    h0, mask, pos, y = G.build_batch('s2000', B)
    mv = int(mask.sum(-1).max())
    h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
    th, u = G.to_ct_list(theta), G.to_ct_list(us[0])
    a, ad = v1.encoder(h0, mask, pos.cuda(), th, u)
    h0p, lens = V.pack(h0, mask)
    b, bd = v2.encoder_packed(h0p, lens, [int(p) for p in pos], th, u)
    A, Ad, Bv, Bd = (G.reveal(t).cpu() for t in (a, ad, b, bd))
    rel = lambda p, q: float((p - q).norm() / q.norm())
    return dict(rank=crypten.communicator.get().get_rank(), B=B, lens=lens,
                rel_h=rel(Bv, A), rel_hdot=rel(Bd, Ad),
                per_sample_rel_hdot=[rel(Bd[i], Ad[i]) for i in range(B)])


@mpc.run_multiprocess(world_size=2)
def profile(which):
    G, V = setup()
    crypten.cfg.communicator.verbose = True
    rank = crypten.communicator.get().get_rank()
    comm = crypten.communicator.get()
    ctx, prim = ['linear'], [None]
    acc = defaultdict(lambda: [0, 0])
    orig_log = comm._log_communication

    def _log(nelement):
        r0, b0 = comm.comm_rounds, comm.comm_bytes
        orig_log(nelement)
        a = acc[(ctx[-1], prim[0] or 'other')]
        a[0] += comm.comm_rounds - r0; a[1] += comm.comm_bytes - b0
    comm._log_communication = _log

    def wrap_prim(owner, name, tag):
        f = getattr(owner, name)
        def w(*a, **k):
            outer = prim[0] is None
            if outer: prim[0] = tag
            try:
                return f(*a, **k)
            finally:
                if outer: prim[0] = None
        setattr(owner, name, w)
    from crypten.mpc import MPCTensor
    from crypten.cryptensor import CrypTensor
    import crypten.mpc.primitives.beaver as BV
    wrap_prim(MPCTensor, '_ltz', 'cmp'); wrap_prim(CrypTensor, 'max', 'max'); wrap_prim(CrypTensor, 'reciprocal', 'recip')
    for n, t in (('mul', 'mul'), ('square', 'mul'), ('matmul', 'matmul')):
        wrap_prim(BV, n, t)

    def wrap_ctx(cls, name):
        f = getattr(cls, name)
        def w(self, *a, **k):
            ctx.append(name)
            try:
                return f(self, *a, **k)
            finally:
                ctx.pop()
        setattr(cls, name, w)
    for n in ('exp_neg', 'inv_1p', 'gelu', 'rsqrt_newton', 'softmax', 'ln', 'recip_den'):
        if hasattr(V.SecureOpsV2, n):
            wrap_ctx(V.SecureOpsV2, n)

    Fd, W, H, theta, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    ops = V.SecureOpsV2(W)
    out = {}
    for label, B, vlen in [('b1_L10', 1, 10), ('b64', 64, None)]:
        h0, mask, pos, y = G.build_batch('s2000', B, valid_len=vlen)
        mv = int(mask.sum(-1).max())
        h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
        h0p, lens = V.pack(h0, mask)
        th, u = G.to_ct_list(theta), G.to_ct_list(us[0])
        times = []
        for rep in range(3):                     # rep 0 = warm-up
            acc.clear(); comm.reset_communication_stats()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            h_ct, hd_ct = ops.encoder_packed(h0p, lens, [int(p) for p in pos], th, u)
            G.reveal(h_ct, dst=0); G.reveal(hd_ct, dst=0)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        out[label] = dict(B=B, T=mv, N_real=int(sum(lens)), N_padded=B * mv, step_s=times[1:],
                          rounds=comm.comm_rounds, bytes=comm.comm_bytes,
                          rows=[dict(ctx=c, prim=p, rounds=v[0], bytes=v[1]) for (c, p), v in sorted(acc.items())])
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    mode = sys.argv[1]
    if mode == 'precision':
        res = [r for r in precision() if r]
    elif mode == 'equiv':
        res = [r for r in equiv(int(sys.argv[2]) if len(sys.argv) > 2 else 8) if r]
    else:
        res = [r for r in profile(mode) if r]
    (ROOT / f'stage2/t6_v2_{mode}.json').write_text(json.dumps(res, indent=1, default=str))
    r0 = [r for r in res if r.get('rank') == 0][0]
    print('T6_DONE', mode, json.dumps(r0.get('summary', {k: r0[k] for k in r0 if k not in ('rows', 'out')}), default=str)[:1500])
