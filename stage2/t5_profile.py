"""T5: per-operator round / byte / time attribution of the CrypTen secure JVP encoder.
Op context  = innermost SecureOps kernel (exp_neg, inv_1p, gelu, rsqrt_newton, softmax, ln) or
              'linear' (everything else in encoder: matmuls with public W, LoRA, attention mms).
Primitive   = outermost CrypTen primitive active when the communication happens:
              cmp (MPCTensor._ltz: lt/ge/gt and inside max), max, recip, mul (beaver mul/square),
              matmul (beaver matmul), reveal, other.
TTP traffic (party0 <-> dealer) is attributed to primitive 'dealer'.
Communicator stats are per party (CrypTen logs send+recv, so wire bytes = sum over parties / 2)."""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def worker(shapes):
    import gpu_crypten as G
    crypten.init()
    G.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    rank = crypten.communicator.get().get_rank()
    comm = crypten.communicator.get()
    ctx, prim = ['linear'], [None]
    acc = defaultdict(lambda: [0, 0, 0.0])          # (ctx, prim) -> rounds, bytes, comm_time
    tacc = defaultdict(float)                        # ctx -> wall time (exclusive of children)

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
    import crypten.mpc.primitives.arithmetic as AR
    wrap_prim(MPCTensor, '_ltz', 'cmp')
    wrap_prim(CrypTensor, 'max', 'max')
    wrap_prim(CrypTensor, 'reciprocal', 'recip')
    for n, t in (('mul', 'mul'), ('square', 'mul'), ('matmul', 'matmul')):
        wrap_prim(BV, n, t)
    wrap_prim(G, 'reveal', 'reveal')
    from crypten.mpc.provider.ttp_provider import TTPClient
    if rank == 0 and crypten.mpc.ttp_required():
        cli = TTPClient.get(); wrap_prim(cli, 'ttp_request', 'dealer')

    def wrap_ctx(cls, name):
        f = getattr(cls, name)
        def w(self, *a, **k):
            ctx.append(name); t0 = time.perf_counter()
            try:
                return f(self, *a, **k)
            finally:
                dt = time.perf_counter() - t0
                ctx.pop(); tacc[name] += dt
                if ctx[-1] != 'linear': tacc[ctx[-1]] -= dt     # keep times exclusive
        setattr(cls, name, w)
    for n in ('exp_neg', 'inv_1p', 'gelu', 'rsqrt_newton', 'softmax', 'ln'):
        wrap_ctx(G.SecureOps, n)

    Fd, W, H, theta, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    ops = G.SecureOps(W)
    out = {}
    for label, B, vlen in shapes:
        h0, mask, pos, y = G.build_batch('s2000', B, valid_len=vlen)
        mv = int(mask.sum(-1).max())
        h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
        th, u = G.to_ct_list(theta), G.to_ct_list(us[0])
        for rep in range(2):                 # rep 0 = warm-up
            acc.clear(); tacc.clear()
            comm.reset_communication_stats()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            h_ct, hd_ct = ops.encoder(h0, mask, pos.cuda(), th, u)
            G.reveal(h_ct, dst=0); G.reveal(hd_ct, dst=0)
            torch.cuda.synchronize(); step = time.perf_counter() - t0
        tot = time.perf_counter()
        out[label] = dict(B=B, T=mv, step_s=step,
                          total_rounds=comm.comm_rounds, total_bytes=comm.comm_bytes,
                          rows=[dict(ctx=c, prim=p, rounds=v[0], bytes=v[1]) for (c, p), v in sorted(acc.items())],
                          ctx_time_excl=dict(tacc))
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = sys.argv[1] if len(sys.argv) > 1 else 'TTP'
    res = [r for r in worker([('b1_L10', 1, 10), ('b64', 64, None)]) if r]
    Path(ROOT / 'stage2/t5_profile.json').write_text(json.dumps(res, indent=1, default=str))
    print('T5_DONE')
