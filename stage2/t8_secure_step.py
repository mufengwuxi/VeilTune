"""Full secure training step (system model A, nothing revealed during training), timed.
Step = client shares packed layer-0 activations + labels -> joint secret direction u ->
v2 encoder with tangent -> secure head + CE + c -> secure update theta <- theta - lr*c*u.
u: each party inputs N(0, 1/2) noise, u = u0 + u1 (timing prototype; production: dealer-generated).
Reports per-phase time and P0<->P1 rounds/bytes (rank-1 counters, TTP traffic excluded)."""
import json, sys, time, types
from pathlib import Path
import numpy as np, torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def run(cfgs, steps, lr):
    import gpu_crypten as G, gpu_crypten_v2 as V
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    _argv = sys.argv; sys.argv = [sys.argv[0]]
    import round5_float as R
    sys.argv = _argv
    Fd, W, H, theta0, _ = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    ops = V.SecureOpsV2(W); ops.init_head(H)
    args = types.SimpleNamespace(row='zo', rank=4, lr=lr, seed=42, batch=64, steps=steps, eval_every=10 ** 9,
                                 epsilon=1e-3, qdirs=1, save_at=[], output=f'/tmp/t8_r5_{rank}')
    E = R.MiniExp(args)
    shapes = [t.shape for t in theta0]
    sizes = [int(np.prod(s)) for s in shapes]
    out = {}
    for label, B in cfgs:
        E.args.batch = B
        theta = G.to_ct_list(theta0)
        recs = []
        for step in range(steps + 1):                              # step 0 = warm-up
            ix = E.batch()
            ids, mask, pos, y = (t[ix] for t in E.data['train'])
            n = int(mask.sum(1).max())
            with torch.no_grad():
                h0 = E.model.bert.embeddings(input_ids=ids[:, :n].cuda()).double()
            h0p, lens = V.pack(h0, mask[:, :n].cuda())
            ph = {}
            def tick(name, t0, r0, b0):
                torch.cuda.synchronize()
                ph[name] = dict(s=time.perf_counter() - t0, rounds=comm.comm_rounds - r0, bytes=comm.comm_bytes - b0)
                return time.perf_counter(), comm.comm_rounds, comm.comm_bytes
            torch.cuda.synchronize(); T0 = time.perf_counter()
            t0, r0, b0 = time.perf_counter(), comm.comm_rounds, comm.comm_bytes
            ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')            # client labels (secret)
            g = torch.Generator(device='cuda').manual_seed(1000 * rank + step)
            half = [torch.randn(sum(sizes), generator=g, device='cuda', dtype=torch.float64) * (0.5 ** 0.5)]
            U = crypten.cryptensor(half[0], src=0, device='cuda') + crypten.cryptensor(half[0], src=1, device='cuda')
            u, o = [], 0
            for s, k in zip(shapes, sizes):
                u.append(U[o:o + k].reshape(s)); o += k
            t0, r0, b0 = tick('input', t0, r0, b0)
            h_ct, hd_ct = ops.encoder_packed(h0p, lens, [int(p) for p in pos], theta, u)  # h0p shared from P0
            t0, r0, b0 = tick('encoder', t0, r0, b0)
            c = ops.secure_c(h_ct, hd_ct, ysec)
            t0, r0, b0 = tick('head', t0, r0, b0)
            dU = U * (c * lr)                                       # one secure multiplication for all 16 tensors
            o, new = 0, []
            for t, s, k in zip(theta, shapes, sizes):
                new.append(t - dU[o:o + k].reshape(s)); o += k
            theta = new
            t0, r0, b0 = tick('update', t0, r0, b0)
            total = time.perf_counter() - T0
            if step > 0:
                recs.append(dict(step=step, total_s=total, N_real=int(sum(lens)), T=n, phases=ph))
        out[label] = recs
    # correctness spot check: reveal theta change norm (harness only)
    d = [G.reveal(t).cpu() - torch.tensor(t0_) for t, t0_ in zip(theta, theta0)]
    out['theta_delta_norm'] = float(torch.sqrt(sum((x ** 2).sum() for x in d)))
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = [r for r in run([('b1', 1), ('b64', 64)], 5, 3e-4) if r]
    (ROOT / 'stage2/t8_secure_step.json').write_text(json.dumps(res, indent=1, default=str))
    print('T8_DONE')
