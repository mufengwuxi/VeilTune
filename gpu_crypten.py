"""CrypTen GPU secure-JVP for the U-split 4-block BERT (f=16, 64-bit ring).

Mirror of emu/secure_emu.py under CrypTen 2PC: two parties in one process
group on one GPU. Operator set and analytic tangents follow sjvp/model.py
exactly. Documented substitutions (per GPU_HANDOFF):
  - reciprocal (softmax denominator): CrypTen built-in MPCTensor.reciprocal()
    (replaces SPU reciprocal_goldschmidt);
  - rsqrt (LayerNorm): CrypTen built-in sqrt() then reciprocal();
  - LN prescale S = 1 (f=16 config; S cancels in exact arithmetic);
  - exp_neg clamp: one comparison, select(x < -12, -12, x); every call site
    passes x <= 0 so the upper clamp at 0 is the identity;
  - public operands pre-quantized to f=16 in plain torch (float32 first for
    scalars), identical to emu FXP.enc/const.

Secrets: theta (LoRA state), u (direction), and everything derived.
Public: W, biases, h0, mask, pos, all constants.

Modes: precision (deliverables 1+2), timing (deliverable 3).
Stage 2: --provider TTP (default; separate TTPServer process) | TFP; exact reveal().
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import crypten
import crypten.mpc as mpc

ROOT = Path('/workspace/veiltune')
C_INV = [0.03226120897959244, -0.049780065730235586, 0.057650254381631766, -0.08583010236400039,
         0.1317100845857373, -0.19765769464931107, 0.29629765286612486, -0.44444230412031277,
         0.6666666475671728]
F = 16
SCALE = 2.0 ** F
ARITH = None  # set after crypten import: crypten.mpc.ptype.arithmetic


def enc_pub(x):
    t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    return torch.round(t * SCALE) / SCALE


def const(c):
    return float(round(float(np.float32(c)) * SCALE)) / SCALE


class SecureOps:
    def __init__(self, W):
        self.W = {k: enc_pub(v).cuda() for k, v in W.items()}
        self.cinv = [const(c) for c in C_INV]

    @staticmethod
    def _select(gate, a, b):
        """gate: arithmetic secret in {0,1}; a,b: secrets. gate?a:b"""
        return b + gate * (a - b)

    # ---- kernels (y, y_dot); tangents mirror sjvp/model.py custom_jvps ----
    def exp_neg(self, x, xt=None):
        gate = (x < -12.0).to(ARITH)
        xc = self._select(gate, x * 0 - 12.0, x)
        z = xc * const(1.0 / 16.0)
        acc = x * 0 + 1.0
        for k in range(8, 0, -1):
            acc = z * acc * const(1.0 / k) + 1.0
        for _ in range(4):
            acc = acc * acc
        return acc, (None if xt is None else acc * xt)

    def inv_1p(self, e):
        t = e - 0.5
        acc = t * self.cinv[0]
        for c in self.cinv[1:]:
            acc = acc * t + c
        return acc

    def gelu(self, z, zt=None):
        k044, k159 = const(0.044715), const(1.5957691216057308)
        s = (z + z * z * k044 * z) * k159
        pos = (s >= 0).to(ARITH)
        a = self._select(pos, s, s * 0 - s)          # |s|
        e, _ = self.exp_neg(-a)
        qq = self.inv_1p(e)
        one = s * 0 + 1.0
        y = self._select(pos, qq, one - qq)          # sigmoid(s)
        out = z * y
        if zt is None:
            return out, None
        gate = (a < 12.0).to(ARITH)
        dsig = self._select(gate, (e * qq) * qq, e * 0)   # sig' zeroed for |s|>=12
        ds = (z * z * const(0.134145) + 1.0) * k159
        return out, (y + z * dsig * ds) * zt

    _RSQRT_RUNGS = [(2.0 ** j, 2.0 ** (-(j + 2) // 2)) for j in range(-14, 19, 2)]

    def rsqrt_newton(self, v):
        """High-precision scale-uniform 1/sqrt(v).
        Ladder picks r0 = 2^{-(j+2)/2} for v in [2^{j-2}, 2^j) (first-true gate,
        monotone flag), so u = v r0^2 lies in [1/16, 1/4) and t* = u^{-1/2}
        in (2, 4]. Newton on t with t0 = 2 (e0 in [-0.5, 0), e' = -1.5e^2 -
        0.5e^3 -> ~1e-10 after 8 iters) runs entirely at O(1) scale, so the
        f=16 truncation noise per iteration is scale-independent. Returns
        r0 * t. A plain Newton on r inherits ulp noise of the large early
        iterates (0.4-7% error at the range ends)."""
        r0 = v * 0 + const(2.0 ** -9)          # above the ladder (v >= 2^18)
        decided = v * 0                          # arithmetic 0/1 flag
        for thr, cand in self._RSQRT_RUNGS:
            gate = (v < thr).to(ARITH)
            take = gate - gate * decided         # first true gate only
            r0 = r0 + take * ((v * 0 + const(cand)) - r0)
            decided = decided + take
        u = v * (r0 * r0)                        # in [1/16, 1/4)
        t = u * 0 + const(2.0)                   # t0/t* in [0.5, 1)
        half, three = const(0.5), const(3.0)
        for _ in range(8):
            t = t * (three - u * (t * t)) * half
        return r0 * t

    def softmax(self, s, st, mask):
        s = s + (mask - 1.0) * 30.0
        m = s.max(dim=-1, keepdim=True)[0]
        e, _ = self.exp_neg(s - m)
        e = e * mask
        p = e * e.sum(dim=-1, keepdim=True).reciprocal()
        if st is None:
            return p, None
        return p, p * (st - (p * st).sum(dim=-1, keepdim=True))

    def ln(self, x, xt, g, b):
        inv256 = const(1.0 / 256.0)
        d = x - x.sum(dim=-1, keepdim=True) * inv256
        var = (d * d).sum(dim=-1, keepdim=True) * inv256
        r = self.rsqrt_newton(var)
        n = d * r
        y = n * g + b
        if xt is None:
            return y, None
        dd = xt - xt.sum(dim=-1, keepdim=True) * inv256
        mm = (n * dd).sum(dim=-1, keepdim=True) * inv256
        nt = r * (dd - n * mm)
        return y, nt * g

    # ---- encoder: mirrors emu/secure_emu.py; last_row only (MASK row) ----
    def encoder(self, h0, mask, pos, theta, u=None):
        """h0 [B,T,256] public f16 torch; mask [B,T] public; pos [B] public ints.
        theta/u: lists of 16 secret cryptensors. Returns (h, h_dot) [B,256]."""
        W, mm = self.W, lambda a, b: a @ b
        x = crypten.cryptensor(h0, device='cuda')
        xt = crypten.cryptensor(torch.zeros_like(h0), device='cuda') if u is not None else None
        B, T, _ = h0.shape
        mk = mask[:, None, None, :]
        ar = torch.arange(B, device='cuda')
        onehot = torch.zeros(B, T, device='cuda')
        onehot[ar, pos] = 1.0                                     # public MASK selector

        def sel_row(a):                                           # [B,T,D]->[B,1,D] at pos
            return (a * onehot[..., None]).sum(dim=1, keepdim=True)

        for i in range(4):
            w = lambda k: W[f'w{i}_{k}']
            Aq, Bq, Av, Bv = theta[4 * i:4 * i + 4]

            def lora_lin(name, A, Bm, j):
                base = mm(x, w(name)) + w(name + 'b')
                inner = mm(x, A.transpose(0, 1))
                y = base + mm(inner, Bm.transpose(0, 1)) * 4.0
                if xt is None:
                    return y, None
                dA, dB = u[4 * i + j], u[4 * i + j + 1]
                inner_t = mm(xt, A.transpose(0, 1)) + mm(x, dA.transpose(0, 1))
                yt = mm(xt, w(name)) + (mm(inner_t, Bm.transpose(0, 1)) + mm(inner, dB.transpose(0, 1))) * 4.0
                return y, yt

            q_, qt = lora_lin('q', Aq, Bq, 0)
            v_, vt = lora_lin('v', Av, Bv, 2)
            k_ = mm(x, w('k')) + w('kb')
            kt = None if xt is None else mm(xt, w('k'))
            if i == 3:                                            # only the MASK row
                x, q_ = sel_row(x), sel_row(q_)
                if xt is not None:
                    xt, qt = sel_row(xt), sel_row(qt)
            Tq = x.shape[1]
            hd = lambda a, n: a.reshape(B, n, 4, 64).transpose(1, 2)
            Q, K, V = hd(q_, Tq), hd(k_, T), hd(v_, T)
            S = mm(Q, K.transpose(-1, -2))
            St = None if xt is None else mm(hd(qt, Tq), K.transpose(-1, -2)) + mm(Q, hd(kt, T).transpose(-1, -2))
            P, Pt = self.softmax(S, St, mk)
            att = mm(P, V).transpose(1, 2).reshape(B, Tq, 256)
            att_t = None if xt is None else (mm(Pt, V) + mm(P, hd(vt, T))).transpose(1, 2).reshape(B, Tq, 256)
            pre1 = x + mm(att, w('o')) + w('ob')
            pre1t = None if xt is None else xt + mm(att_t, w('o'))
            h, ht = self.ln(pre1, pre1t, w('ln1'), w('ln1b'))
            z = mm(h, w('ff1')) + w('ff1b')
            zt = None if xt is None else mm(ht, w('ff1'))
            g, gt = self.gelu(z, zt)
            pre2 = h + mm(g, w('ff2')) + w('ff2b')
            pre2t = None if xt is None else ht + mm(gt, w('ff2'))
            x, xt = self.ln(pre2, pre2t, w('ln2'), w('ln2b'))
        return x[:, 0], (None if xt is None else xt[:, 0])


def reveal(ct, dst=None):
    """Exact decode: reveal ring integers and divide by the scale in float64.
    CrypTen's get_plain_text() divides negatives by (scale-1) and goes through
    float32, giving a -1.0 error for values within |v| ulp above a negative
    integer (stage-2 T1: 22 of 200k values)."""
    t = ct._tensor.reveal(dst=dst)
    if dst is not None and crypten.communicator.get().get_rank() != dst:
        return None
    return t.double() / ct.encoder.scale


def client_c(h, hdot, H, y):
    """Plaintext FP64 client: exact MLM head; c = dL/dh . hdot."""
    h = h.double().clone().requires_grad_()
    with torch.enable_grad():
        z = torch.nn.functional.gelu(h @ H['head_W'].double() + H['head_b'].double(), approximate='none')
        c = z - z.mean(-1, keepdim=True)
        var = (c * c).mean(-1, keepdim=True) + 1e-12
        z = c / var.sqrt() * H['head_g'].double() + H['head_be'].double()
        logits = z @ H['head_dec'].double() + H['head_decb'].double()
        loss = -torch.log_softmax(logits, -1)[0, y]
        g, = torch.autograd.grad(loss, h)
    return (g * hdot.double()).sum().item()


def load_fixture(path):
    Fd = dict(np.load(path))
    W = {k: v for k, v in Fd.items() if k.startswith('w')}
    H = {k: v for k, v in Fd.items() if k.startswith('head')}
    theta = [Fd[f'theta{j}'] for j in range(16)]
    nd = len([k for k in Fd if k.startswith('u') and k.endswith('_0')])
    us = [[Fd[f'u{d}_{j}'] for j in range(16)] for d in range(nd)]
    return Fd, W, H, theta, us


def to_ct_list(arrs):
    return [crypten.cryptensor(enc_pub(a).cuda(), device='cuda') for a in arrs]


def build_batch(tag, B, valid_len=None):
    """Real batches from data/*.json via the tarball round5_float loader.
    valid_len (B=1 mode): pad/truncate so that exactly valid_len positions are
    valid and the MASK sits at valid_len-2. Returns h0,mask,pos,y,valid_lens."""
    import sys
    sys.path.insert(0, str(ROOT))
    import round5_float as R
    import types
    args = types.SimpleNamespace(row='zo', rank=4, lr=0.0, seed=42, batch=1, steps=0,
                                 eval_every=1, epsilon=1e-3, qdirs=1, save_at=[], output='/tmp/gpu_line_exp')
    E = R.MiniExp(args)
    E.set_parameters()
    n = len(E.data['train'][0])
    ids = E.data['train'][0][:B]
    if valid_len is not None:  # craft: keep first valid_len-2 text tokens + suffix
        tok = E.tok
        from round5_float import LENGTH, encode as _e
        rows = []
        data = json.loads((ROOT / 'data/train.json').read_text())
        for r in data[:B]:
            text = tok.encode(r['sentence'], add_special_tokens=False)[:valid_len - 6]
            rows.append({'sentence': tok.decode(text), 'label': r['label']})
        enc = _e(rows, tok)
        ids, mask, pos, y = enc
        assert int(mask[0].sum()) == valid_len and int(pos[0]) == valid_len - 3, (int(mask[0].sum()), int(pos[0]))
    else:
        mask, pos, y = E.data['train'][1][:B], E.data['train'][2][:B], E.data['train'][3][:B]
    with torch.no_grad():
        h0 = E.model.bert.embeddings(input_ids=ids.cuda()).detach().cpu()
    return h0, mask.float(), pos, y


@mpc.run_multiprocess(world_size=2)
def _worker(mode, tag, samples, dirs, timing_B, timing_len, out_path):
    global ARITH
    crypten.init()
    ARITH = crypten.mpc.ptype.arithmetic
    rank = crypten.communicator.get().get_rank()
    torch.manual_seed(1234)
    out = {'rank': rank}

    Fd, W, H, theta, us = load_fixture(ROOT / f'sjvp/fixture_{tag}.npz')
    ops = SecureOps(W)
    c_same = np.load(ROOT / f'sjvp/c_plain_{tag}_mpc.npy')
    c_exact = np.load(ROOT / f'sjvp/c_plain_{tag}_exact.npy')

    if mode == 'precision':
        theta_ct = to_ct_list(theta)
        rows = []
        for i in samples:
            mask = torch.tensor(Fd['mask'][i], dtype=torch.float64, device='cuda')
            pos = torch.tensor([int(Fd['pos'][i])], device='cuda')
            h0 = torch.tensor(Fd['h0'][i:i + 1], dtype=torch.float64, device='cuda')
            n_valid = int(Fd['mask'][i].sum())
            h0, mask = h0[:, :n_valid], mask[None, :n_valid]  # public trim, [B,T]
            y = int(Fd['y'][i])
            for d in dirs:
                u_ct = to_ct_list(us[d])
                t0 = time.time()
                h_ct, hd_ct = ops.encoder(h0, mask, pos, theta_ct, u_ct)
                h = reveal(h_ct).cpu()                   # [1,256]
                hdot = reveal(hd_ct).cpu()
                dt = time.time() - t0
                c = client_c(h, hdot, {k: torch.tensor(v) for k, v in H.items()}, y)
                rows.append(dict(sample=i, dir=d, c=c,
                                 c_same=float(c_same[i, d]), c_ref=float(Fd['c_ref'][i, d]),
                                 rel_same=abs(c - c_same[i, d]) / abs(c_same[i, d]),
                                 rel_ref=abs(c - Fd['c_ref'][i, d]) / abs(Fd['c_ref'][i, d]),
                                 seconds=dt))
                if rank == 0:
                    print(json.dumps(rows[-1]), flush=True)
        if rank == 0:
            cs = np.array([r['c'] for r in rows]); sm = np.array([r['c_same'] for r in rows])
            rf = np.array([r['c_ref'] for r in rows])
            out['summary'] = dict(
                n=len(rows),
                rel_l2_same=float(np.linalg.norm(cs - sm) / np.linalg.norm(sm)),
                rel_l2_ref=float(np.linalg.norm(cs - rf) / np.linalg.norm(rf)),
                max_rel_same=float(max(r['rel_same'] for r in rows)),
                sign_ref=float((np.sign(cs) == np.sign(rf)).mean()),
                mean_seconds=float(np.mean([r['seconds'] for r in rows])))
            out['rows'] = rows

    elif mode == 'timing':
        crypten.cfg.communicator.verbose = True
        comm_obj = crypten.communicator.get()
        # instrument the tuple provider: dealer(offline-equivalent) time spent
        # inside the step = generation + sharing of beaver tuples (TFP: party 0
        # acts as the trusted dealer, generation is inline)
        prov = crypten.mpc.get_default_provider()
        dealer = {'t': 0.0, 'calls': {}}

        def _wrap(fn, name):
            def w(*a, **k):
                t0 = time.perf_counter()
                r = fn(*a, **k)
                dt = time.perf_counter() - t0
                dealer['t'] += dt
                d = dealer['calls'].setdefault(name, [0, 0.0])
                d[0] += 1
                d[1] += dt
                return r
            return w
        for name in [n for n in dir(prov) if not n.startswith('_')]:
            fn = getattr(prov, name)
            if callable(fn) and name != 'NAME':
                setattr(prov, name, _wrap(fn, name))

        ttp = {'n': 0, 't': 0.0, 'rounds': 0, 'bytes': 0}
        if crypten.mpc.ttp_required() and rank == 0:
            from crypten.mpc.provider.ttp_provider import TTPClient
            cli = TTPClient.get()
            _orig = cli.ttp_request

            def _ttp_req(*a, **k):
                r0, b0 = comm_obj.comm_rounds, comm_obj.comm_bytes
                t0 = time.perf_counter()
                r = _orig(*a, **k)
                ttp['t'] += time.perf_counter() - t0
                ttp['n'] += 1
                ttp['rounds'] += comm_obj.comm_rounds - r0
                ttp['bytes'] += comm_obj.comm_bytes - b0
                return r
            cli.ttp_request = _ttp_req

        results = {}
        shapes = [('b1_L10', 1, 10), ('b64', 64, None)]
        for label, B, vlen in shapes:
            if label == 'b1_L10':
                h0, mask, pos, y = build_batch(tag, 1, valid_len=10)
            else:
                h0, mask, pos, y = build_batch(tag, B)
            max_valid = int(mask.sum(-1).max())
            h0, mask = h0[:, :max_valid].double(), mask[:, :max_valid].double()
            theta_ct = to_ct_list(theta)
            u_ct = to_ct_list(us[0])
            pos_t = pos.clone()
            times = []
            for rep in range(4):
                torch.cuda.synchronize()
                t0 = time.time()
                h_ct, hd_ct = ops.encoder(h0.cuda(), mask.cuda(), pos_t.cuda(), theta_ct, u_ct)
                h = reveal(h_ct, dst=0)
                hd = reveal(hd_ct, dst=0)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            # instrumented step for dealer/comm stats
            dealer['t'], dealer['calls'] = 0.0, {}
            ttp.update(n=0, t=0.0, rounds=0, bytes=0)
            comm_obj.reset_communication_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            h_ct, hd_ct = ops.encoder(h0.cuda(), mask.cuda(), pos_t.cuda(), theta_ct, u_ct)
            h = reveal(h_ct, dst=0)
            hd = reveal(hd_ct, dst=0)
            torch.cuda.synchronize()
            step_t = time.time() - t0
            stats = comm_obj.get_communication_stats()
            out.setdefault('per_rank', {})[label] = dict(
                step_s=step_t, dealer_s=dealer['t'],
                online_net_s=step_t - dealer['t'],
                tuples={k: v[0] for k, v in dealer['calls'].items()},
                comm_rounds=stats.get('rounds'), comm_bytes=stats.get('bytes'),
                comm_time_s=stats.get('time'),
                ttp_requests=ttp['n'], ttp_request_s=ttp['t'],
                ttp_rounds=ttp['rounds'], ttp_bytes=ttp['bytes'],
                p01_rounds=stats.get('rounds') - ttp['rounds'],
                p01_bytes=stats.get('bytes') - ttp['bytes'])
            if rank == 0:
                results[label] = dict(provider=crypten.cfg.mpc.provider, batch=B, seq=int(h0.shape[1]),
                                      max_valid=max_valid, times_s=times[1:],
                                      step_s_median=float(np.median(times[1:])),
                                      warmup_s=times[0])
        if rank == 0:
            out['timing'] = results
    if rank == 0:
        (Path(out_path)).write_text(json.dumps(out, indent=1, default=str))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['precision', 'timing'], required=True)
    p.add_argument('--tag', default='s2000')
    p.add_argument('--samples', type=int, nargs='+', default=[0])
    p.add_argument('--dirs', type=int, nargs='+', default=[0, 1])
    p.add_argument('--timing-b', type=int, default=1)
    p.add_argument('--timing-len', type=int, default=10)
    p.add_argument('--provider', choices=['TTP', 'TFP'], default='TTP')
    p.add_argument('--out', default='/workspace/veiltune/gpu_line_out.json')
    a = p.parse_args()
    crypten.cfg.mpc.provider = a.provider      # TTP: independent TTPServer process (dealer)
    res = _worker(a.mode, a.tag, a.samples, a.dirs, a.timing_b, a.timing_len, a.out)
    res = [r for r in (res or []) if r]          # drop the TTPServer's None
    if a.mode == 'timing' and res:
        merged = dict(res[0])
        per_rank = {f'rank{r["rank"]}': r['per_rank'] for r in res if 'per_rank' in r}
        b0 = sum(pr['comm_bytes'] for pr in per_rank.get('rank0', {}).values())
        b1 = sum(pr['comm_bytes'] for pr in per_rank.get('rank1', {}).values())
        merged['comm_total_bytes_both_ranks'] = b0 + b1
        merged['comm_total_MB'] = round((b0 + b1) / 1e6, 2)
        # per label: P0<->P1 only (TTP traffic split out). CrypTen logs send+recv per
        # party, so wire bytes = (rank0 + rank1) / 2; each direction carries half of that.
        merged['per_label'] = {}
        for lab in per_rank.get('rank0', {}):
            r0, r1 = per_rank['rank0'][lab], per_rank['rank1'][lab]
            merged['per_label'][lab] = dict(
                rounds_rank0=r0['comm_rounds'], rounds_rank1=r1['comm_rounds'],
                p01_rounds=r0['p01_rounds'], p01_logged_bytes_both=r0['p01_bytes'] + r1['p01_bytes'],
                p01_wire_bytes=(r0['p01_bytes'] + r1['p01_bytes']) / 2,
                ttp_requests=r0['ttp_requests'], ttp_request_s=r0['ttp_request_s'],
                ttp_logged_bytes=r0['ttp_bytes'], dealer_s_rank0=r0['dealer_s'], dealer_s_rank1=r1['dealer_s'],
                step_s_instrumented=r0['step_s'])
        merged['per_rank'] = per_rank
        Path(a.out).write_text(json.dumps(merged, indent=1, default=str))
        print('DONE', json.dumps({k: merged[k] for k in ('timing', 'per_label') if k in merged},
                                 default=str)[:2500])
    else:
        print('DONE', json.dumps(res[0] if res and res[0] else {}, default=str)[:2000])


if __name__ == '__main__':
    main()
