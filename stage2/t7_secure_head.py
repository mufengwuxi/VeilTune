"""Secure head check: c computed fully in MPC (labels secret, nothing revealed) vs
(a) client plaintext head on revealed h (old U-split) and (b) the fixture references.
Also measures the head's share of rounds / bytes / time (b1 fixture rows and a b64 batch)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


@mpc.run_multiprocess(world_size=2)
def run():
    import gpu_crypten as G, gpu_crypten_v2 as V
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    Fd, W, H, theta, us = G.load_fixture(ROOT / 'sjvp/fixture_s2000.npz')
    Ht = {k: torch.tensor(v) for k, v in H.items()}
    c_same = np.load(ROOT / 'sjvp/c_plain_s2000_mpc.npy')
    ops = V.SecureOpsV2(W); ops.init_head(H)
    th = G.to_ct_list(theta)
    rows = []
    for i in range(4):
        n = int(Fd['mask'][i].sum())
        h0p = torch.tensor(Fd['h0'][i, :n], dtype=torch.float64, device='cuda')
        y = crypten.cryptensor(torch.tensor([float(Fd['y'][i])], device='cuda'), src=0, device='cuda')
        for d in ([0, 1, 2, 3, 4, 5] if i == 0 else [2, 3, 4, 5]):
            r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
            h_ct, hd_ct = ops.encoder_packed(h0p, [n], [int(Fd['pos'][i])], th, G.to_ct_list(us[d]))
            r1, b1 = comm.comm_rounds, comm.comm_bytes; t1 = time.perf_counter()
            c_ct = ops.secure_c(h_ct, hd_ct, y)
            r2, b2 = comm.comm_rounds, comm.comm_bytes; t2 = time.perf_counter()
            c_sec = float(G.reveal(c_ct).item())                     # test harness only
            h, hd = G.reveal(h_ct).cpu(), G.reveal(hd_ct).cpu()
            c_cli = G.client_c(h, hd, Ht, int(Fd['y'][i]))
            rows.append(dict(sample=i, dir=d, c_sec=c_sec, c_cli=c_cli, c_same=float(c_same[i, d]),
                             c_ref=float(Fd['c_ref'][i, d]),
                             enc_rounds=r1 - r0, enc_bytes=b1 - b0, enc_s=t1 - t0,
                             head_rounds=r2 - r1, head_bytes=b2 - b1, head_s=t2 - t1))
    g16 = [r for r in rows if not (r['sample'] == 0 and r['dir'] in (0, 1))]
    A = lambda k: np.array([r[k] for r in g16])
    rl2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    summ = dict(n=len(g16),
                secure_head_vs_same=rl2(A('c_sec'), A('c_same')), secure_head_vs_ref=rl2(A('c_sec'), A('c_ref')),
                client_head_vs_same=rl2(A('c_cli'), A('c_same')),
                secure_vs_client=rl2(A('c_sec'), A('c_cli')),
                sign_secure=float((np.sign(A('c_sec')) == np.sign(A('c_ref'))).mean()),
                max_rel_secure=float(np.max(np.abs(A('c_sec') - A('c_same')) / np.abs(A('c_same')))),
                b1_enc_rounds=float(np.median(A('enc_rounds'))), b1_head_rounds=float(np.median(A('head_rounds'))),
                b1_enc_bytes=float(np.median(A('enc_bytes'))), b1_head_bytes=float(np.median(A('head_bytes'))),
                b1_enc_s=float(np.median(A('enc_s'))), b1_head_s=float(np.median(A('head_s'))))
    # b64 batch (packed), head share
    h0, mask, pos, yy = G.build_batch('s2000', 64)
    mv = int(mask.sum(-1).max()); h0, mask = h0[:, :mv].double().cuda(), mask[:, :mv].double().cuda()
    h0p, lens = V.pack(h0, mask)
    ysec = crypten.cryptensor(yy.double().cuda(), src=0, device='cuda')
    for rep in range(2):
        torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
        h_ct, hd_ct = ops.encoder_packed(h0p, lens, [int(p) for p in pos], th, G.to_ct_list(us[0]))
        torch.cuda.synchronize(); r1, b1 = comm.comm_rounds, comm.comm_bytes; t1 = time.perf_counter()
        c_ct = ops.secure_c(h_ct, hd_ct, ysec)
        torch.cuda.synchronize(); r2, b2 = comm.comm_rounds, comm.comm_bytes; t2 = time.perf_counter()
    summ.update(b64_enc_rounds=r1 - r0, b64_head_rounds=r2 - r1, b64_enc_bytes=b1 - b0, b64_head_bytes=b2 - b1,
                b64_enc_s=t1 - t0, b64_head_s=t2 - t1, b64_N_real=int(sum(lens)), b64_T=mv)
    return dict(rank=rank, summary=summ, rows=rows)


if __name__ == '__main__':
    crypten.cfg.mpc.provider = 'TTP'
    res = [r for r in run() if r]
    (ROOT / 'stage2/t7_secure_head.json').write_text(json.dumps(res, indent=1, default=str))
    s0 = [r for r in res if r['rank'] == 0][0]['summary']; s1 = [r for r in res if r['rank'] == 1][0]['summary']
    print('T7_DONE rank0', json.dumps(s0)); print('T7_DONE rank1', json.dumps({k: s1[k] for k in s1 if 'bytes' in k or 'rounds' in k}))
