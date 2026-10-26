"""RoBERTa-large secure step (t12 setup): per-operation breakdown of P0-P1 rounds / bytes (rank-1 log) and
wall time. Keys: (phase fwd|bwd, outer kernel, innermost kernel, primitive). One step at B=--B. OPS=bp|bp2, LS env."""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'stage2'))
import t12_rl_secure as T12


@mpc.run_multiprocess(world_size=2)
def worker(B):
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp as BPm, gpu_crypten_bp2 as BP2, gpu_crypten_native as NAT, gpu_crypten_lp as LP
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    st = dict(phase='fwd', prim=None); stack = []
    acc = defaultdict(lambda: [0, 0, 0])
    tacc = defaultdict(float)
    orig_log = comm._log_communication

    def _log(nel):
        r0, b0 = comm.comm_rounds, comm.comm_bytes
        orig_log(nel)
        k = (st['phase'], stack[0] if stack else 'linear', stack[-1] if stack else 'linear', st['prim'] or 'other')
        a = acc[k]; a[0] += comm.comm_rounds - r0; a[1] += comm.comm_bytes - b0; a[2] += 1
    comm._log_communication = _log

    def wrap_prim(owner, name, tag):
        f = getattr(owner, name)
        def w(*a, **k):
            outer = st['prim'] is None
            if outer: st['prim'] = tag
            try: return f(*a, **k)
            finally:
                if outer: st['prim'] = None
        setattr(owner, name, w)
    from crypten.mpc import MPCTensor
    from crypten.cryptensor import CrypTensor
    import crypten.mpc.primitives.beaver as BV
    wrap_prim(MPCTensor, '_ltz', 'cmp'); wrap_prim(CrypTensor, 'max', 'max')
    for n, t in (('mul', 'mul'), ('square', 'mul'), ('matmul', 'matmul')):
        wrap_prim(BV, n, t)

    def wrap_ctx(cls, name, static=False):
        f = cls.__dict__[name]
        f = f.__func__ if isinstance(f, staticmethod) else f
        def w(*a, **k):
            if name == 'ln_bwd': st['phase'] = 'bwd'
            stack.append(name)
            if len(stack) == 1: torch.cuda.synchronize(); t0 = time.perf_counter()
            try: return f(*a, **k)
            finally:
                if len(stack) == 1:
                    torch.cuda.synchronize(); tacc[(st['phase'], name)] += time.perf_counter() - t0
                stack.pop()
        setattr(cls, name, staticmethod(w) if static else w)
    for cls in (V.SecureOpsV2, BPm.SecureOpsBP, BP2.SecureOpsBP2, NAT.SecureOpsNative, LP.SecureOpsLP):
        for n, f in list(cls.__dict__.items()):
            if n in ('exp_neg', 'inv_1p', 'rsqrt_newton', 'recip_den', 'gelu_save', 'ln_save', 'softmax_save', '_sigmoid'):
                wrap_ctx(cls, n)
            if n in ('ln_bwd', 'softmax_bwd', 'max_tree', 'exp_free'):
                wrap_ctx(cls, n, static=True)

    tail, W, H, theta, C = T12.load_weights()
    ops = {'bp2': BP2.SecureOpsBP2, 'native': NAT.SecureOpsNative, 'lp': LP.SecureOpsLP}.get(os.environ.get('OPS'), BPm.SecureOpsBP)(W); ops.init_head(H)
    ops.D, ops.NH, ops.LN_EPS = 1024, 16, G.const(1e-5)
    ops.LS = float(os.environ.get('LS', '1'))
    ops.EXP_FREE = os.environ.get('EXP_FREE') == '1'
    tail = None; torch.cuda.empty_cache()
    th = [crypten.cryptensor(G.enc_pub(t.numpy()).cuda(), src=0, device='cuda') for t in theta]
    ix, h0p, lens, pos, y = T12.batch(C, B, 2)
    ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')
    acc.clear(); tacc.clear(); st['phase'] = 'fwd'; comm.reset_communication_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    grads = ops.loss_grad(h0p.cuda(), lens, pos, th, ysec)
    new = [t - g * 1e-4 for t, g in zip(th, grads)]
    torch.cuda.synchronize(); tot = time.perf_counter() - t0
    return dict(rank=rank, total_s=tot, rounds=comm.comm_rounds, bytes=comm.comm_bytes, N_real=int(sum(lens)), T=max(lens),
                rows=[dict(phase=k[0], outer=k[1], inner=k[2], prim=k[3], rounds=v[0], bytes=v[1], calls=v[2])
                      for k, v in sorted(acc.items(), key=lambda kv: -kv[1][1])],
                time=[dict(phase=k[0], outer=k[1], s=v) for k, v in sorted(tacc.items(), key=lambda kv: -kv[1])])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=64); a = ap.parse_args()
    crypten.cfg.mpc.provider = 'TTP'
    res = {r['rank']: r for r in worker(a.B) if r}
    (ROOT / f"stage2/t13_rl_profile_b{a.B}_{os.environ.get('OPS', 'bp')}_ef{os.environ.get('EXP_FREE', '0')}.json").write_text(json.dumps(res, indent=1))
    r1, r0 = res[1], res[0]
    print('T13 total_s', round(r0['total_s'], 2), 'rounds', r1['rounds'], 'GB', round(r1['bytes'] / 1e9, 2), 'N', r1['N_real'])
    for r in r1['rows'][:40]:
        print('T13', r['phase'], r['outer'], r['inner'], r['prim'], r['rounds'], round(r['bytes'] / 1e9, 3), r['calls'])
    for r in r0['time']:
        print('T13T', r['phase'], r['outer'], round(r['s'], 2))
