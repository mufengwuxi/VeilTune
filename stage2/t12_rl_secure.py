"""RoBERTa-large, last 4 layers (20..23) with Q/V LoRA r=4: secure zero-reveal training step (CrypTen 2PC,
TTP dealer, f=16) = v2 primal forward + secure head/CE + secure backprop + SGD. Client-side frozen prefix:
layer-20 hidden states from rl_cache/h20.pt (computed once in plaintext by the client).

modes:
  check  : one batch (B=4): secure LoRA gradient vs plaintext FP64 autograd gradient through the functional
           tail (exact-erf GELU, exact softmax/LN) -> cosine, relative L2 error (per tensor and overall)
  time   : one full secure step each for B in --batches (alone on the GPU); rounds/bytes from rank 1
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, crypten, crypten.mpc as mpc
ROOT = Path('/workspace/veiltune'); sys.path.insert(0, str(ROOT))


def load_weights():
    """W (layers 20..23 -> keys w0..w3, gpu_crypten layout), H (head), label_ids, LoRA init (16 tensors)."""
    import rl_plain as RP
    SPLIT = int(os.environ.get('SPLIT', '20'))
    tail = RP.Tail(SPLIT, list(range(SPLIT, 24)), 4)
    W = {}
    for i, P in enumerate(tail.layers):
        f = 1.0 / math.sqrt(64)                                       # 1/sqrt(d_head) folded into W_K
        for k, (w, b) in dict(q=('Wq', 'bq'), k=('Wk', 'bk'), v=('Wv', 'bv'), o=('Wo', 'bo'),
                              ff1=('Wf1', 'bf1'), ff2=('Wf2', 'bf2')).items():
            s = f if k == 'k' else 1.0
            W[f'w{i}_{k}'] = (P[w].double().T * s).contiguous().cpu().numpy()
            W[f'w{i}_{k}b'] = (P[b].double() * s).cpu().numpy()
        W[f'w{i}_ln1'], W[f'w{i}_ln1b'] = P['g1'].double().cpu().numpy(), P['b1'].double().cpu().numpy()
        W[f'w{i}_ln2'], W[f'w{i}_ln2b'] = P['g2'].double().cpu().numpy(), P['b2'].double().cpu().numpy()
    C = RP.build_cache(SPLIT)
    lab = C['label_ids']
    hh = tail.head
    H = dict(head_W=hh['W'].double().T.cpu().numpy(), head_b=hh['b'].double().cpu().numpy(),
             head_g=hh['g'].double().cpu().numpy(), head_be=hh['be'].double().cpu().numpy(),
             head_dec=hh['dec'][lab.cuda()].double().T.cpu().numpy(), head_decb=hh['decb'][lab.cuda()].double().cpu().numpy())
    g = torch.Generator().manual_seed(5)
    theta = []
    for t in tail.init:                       # A from the standard init, B small random (B = 0 would zero most grads)
        theta.append(t.double().cpu() if t.shape[0] == 4 else torch.randn(t.shape, generator=g, dtype=torch.float64) * 0.02)
    return tail, W, H, theta, C


def batch(C, B, seed):
    tr = C['train']
    ix = torch.randperm(len(tr['y']), generator=torch.Generator().manual_seed(seed))[:B]
    lens = [int(tr['mask'][i].sum()) for i in ix]
    h0p = torch.cat([tr['h'][i, :l] for i, l in zip(ix, lens)]).double()
    return ix, h0p, lens, [int(tr['pos'][i]) for i in ix], tr['y'][ix]


@mpc.run_multiprocess(world_size=2)
def worker(mode, batches):
    import gpu_crypten as G, gpu_crypten_v2 as V, gpu_crypten_bp as BPm, gpu_crypten_bp2 as BP2, gpu_crypten_native as NAT, os
    crypten.init(); G.ARITH = V.ARITH = crypten.mpc.ptype.arithmetic
    crypten.cfg.communicator.verbose = True
    comm = crypten.communicator.get(); rank = comm.get_rank()
    tail, W, H, theta, C = load_weights()
    ops = {'bp2': BP2.SecureOpsBP2, 'native': NAT.SecureOpsNative}.get(os.environ.get('OPS'), BPm.SecureOpsBP)(W); ops.init_head(H)
    ops.D, ops.NH, ops.LN_EPS = 1024, 16, G.const(1e-5)
    ops.LS = float(os.environ.get('LS', '1'))
    ops.EXP_FREE = os.environ.get('EXP_FREE') == '1'
    for flag in ('NEW_GELU', 'MAX_TREE', 'MASK_ONLY'):              # ablation switches (default on)
        setattr(ops, flag, os.environ.get(flag, '1') == '1')
    ops.FWD_ONLY = os.environ.get('FWD_ONLY') == '1'
    ops.LN_SC = float(os.environ.get('LN_SC', '0.25'))
    if os.environ.get('SECRET_BASE') == '1':                     # baseline: base weights secret (owned by model owner)
        sec = lambda t: crypten.cryptensor(t, src=0, device='cuda')
        ops.W = {k: sec(v) for k, v in ops.W.items()}
        ops.hW, ops.hb, ops.hg, ops.hbe, ops.hdd = (sec(t) for t in (ops.hW, ops.hb, ops.hg, ops.hbe, ops.hdd))
    if mode == 'time':                                   # free the plaintext copy before timing
        tail = None; torch.cuda.empty_cache()
    out = {}
    th = [crypten.cryptensor(G.enc_pub(t.numpy()).cuda(), src=0, device='cuda') for t in theta]  # model owner's LoRA (secret)
    if mode == 'check':
        ix, h0p, lens, pos, y = batch(C, 4, 1)
        ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')
        grads = ops.loss_grad(h0p.cuda(), lens, pos, th, ysec)
        gs = [G.reveal(g).cpu() for g in grads]                                   # harness only
        if rank == 0:
            import torch.nn.functional as F
            thp = [t.clone().cuda().requires_grad_(True) for t in theta]
            d = C['train']; n = int(d['mask'][ix].sum(1).max())
            torch.set_default_dtype(torch.float64)
            tl = tail
            for P in tl.layers:
                for k in list(P):
                    if torch.is_tensor(P[k]): P[k] = P[k].double()
            for k in list(tl.head): tl.head[k] = tl.head[k].double()
            lo = tl.forward(thp, d['h'][ix, :n].double().cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(),
                            C['label_ids'].cuda())
            loss = F.cross_entropy(lo, d['y'][ix].cuda())
            gref = torch.autograd.grad(loss, thp)
            gref = [g.detach().cpu() for g in gref]
            a = torch.cat([g.flatten() for g in gs]); b = torch.cat([g.flatten() for g in gref])
            out['check'] = dict(cos=float(a @ b / a.norm() / b.norm()), rel_l2=float((a - b).norm() / b.norm()),
                                per_tensor_rel=[float((x - y_).norm() / y_.norm()) for x, y_ in zip(gs, gref)],
                                grad_norm_ref=float(b.norm()), loss_ref=float(loss))
    else:
        for B in batches:
            ix, h0p, lens, pos, y = batch(C, B, 2)
            if os.environ.get('PAD') == '1':                              # ablation: no packing (pad every row to T)
                T_ = max(lens); rows, o = [], 0
                for l in lens:
                    rows += [h0p[o:o + l], torch.zeros(T_ - l, h0p.shape[1], dtype=h0p.dtype)]; o += l
                h0p, lens = torch.cat(rows), [T_] * len(lens)
            torch.cuda.synchronize(); r0, b0 = comm.comm_rounds, comm.comm_bytes; t0 = time.perf_counter()
            ysec = crypten.cryptensor(y.double().cuda(), src=0, device='cuda')
            grads = ops.loss_grad(h0p.cuda(), lens, pos, th, ysec)
            if not ops.FWD_ONLY:
                new = [t - g * 1e-4 for t, g in zip(th, grads)]
            torch.cuda.synchronize()
            out[f'b{B}'] = dict(s=time.perf_counter() - t0, rounds=comm.comm_rounds - r0, bytes=comm.comm_bytes - b0,
                                N_real=int(sum(lens)), T=max(lens),
                                peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9)
    return dict(rank=rank, out=out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'time'])
    ap.add_argument('--batches', type=int, nargs='*', default=[1, 32, 64])
    a = ap.parse_args()
    crypten.cfg.mpc.provider = 'TTP'
    res = {r['rank']: r['out'] for r in worker(a.mode, a.batches) if r}
    (ROOT / f"stage2/t12_rl_{a.mode}{os.environ.get('OPS', '')}_ls{os.environ.get('LS', '1')}_ef{os.environ.get('EXP_FREE', '0')}_sb{os.environ.get('SECRET_BASE', '0')}_{os.environ.get('TAG', '')}.json").write_text(json.dumps(res, indent=1))
    if a.mode == 'check':
        print('T12_CHECK', json.dumps(res[0]['check']))
    else:
        for k in res[0]:
            print('T12_TIME', k, json.dumps(dict(time_s=round(res[0][k]['s'], 2), rounds=res[1][k]['rounds'],
                  wire_GB=round(res[1][k]['bytes'] / 1e9, 2), N_real=res[0][k]['N_real'], T=res[0][k]['T'],
                  peak_mem_GB=round(res[0][k]['peak_mem_GB'], 1))))
