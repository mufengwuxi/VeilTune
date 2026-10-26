"""Leakage study for system model A (client = data owner, sees h and h_dot at MASK rows each step;
LoRA theta belongs to the model owner and must stay hidden from the client).

Protocol simulated in plaintext (the MPC only changes how h, h_dot are computed, not what the client
sees): each step the client receives h_t = f(x_B; theta_t) and h_dot_t = J_theta f(x_B; theta_t) u_t for
its batch, computes c_t locally, and the update theta <- theta - lr c_t u_t happens inside MPC
(u_t secret).

Scenarios
  s1: theta_0 = standard LoRA init (B = 0, A from the owner's secret seed); client data = train.json (1024)
  s2: theta_0 = owner's adapter, first-order pretrained on the owner's private data (data/owner.json,
      4000 SST-2 sentences disjoint from all client/dev/test data); client data = 16 per class

Client-side attacks (client knows: public base weights, its own inputs + labels, its transcript)
  fit-h  : fit theta_hat to the observed h of the last K steps (MSE on h), from the client's own
           standard init. Measures parameter recovery (Delta W = B A scale, gauge-invariant) and the
           function obtained.
  fit-h+y: same, plus the client's own labels (cross-entropy).
  local  : client-only baseline: first-order LoRA on its own labelled data (no transcript).
Harm = test accuracy / agreement with theta_T of the attack model above the 'local' baseline.
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP

ROOT = Path('/workspace/veiltune')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['s1', 's2'], required=True)
    ap.add_argument('--steps', type=int, default=1000, help='protocol (JVP) steps observed by the client')
    ap.add_argument('--lr', type=float, default=3e-4, help='protocol JVP learning rate')
    ap.add_argument('--train', choices=['jvp', 'fo', 'zo'], default='jvp', help='how the protocol trains: jvp (old), fo = backprop AdamW (SAP/Pencil), zo = two-point SPSA (SecP-Tuning)')
    ap.add_argument('--zo-eps', type=float, default=1e-3)
    ap.add_argument('--Ks', type=int, nargs='*', default=[1, 20, 100], help='attack uses the last K transcript steps')
    ap.add_argument('--reveal', choices=['h', 'logits'], default='h', help='what the client observes each step')
    ap.add_argument('--public', type=int, default=0, help='client also sends N public unlabeled sentences each of the last max(Ks) steps and observes their outputs')
    ap.add_argument('--attack-iters', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out = ROOT / f"rl_leak3/{'' if RP.TASK == 'sst2' else RP.TASK + '_'}{a.train}_{a.scenario}_{a.reveal}{'_pub%d' % a.public if a.public else ''}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'w')

    def rec(**kw):
        log.write(json.dumps(kw) + '\n'); log.flush(); print(json.dumps(kw), flush=True)

    C = RP.build_cache(20, extra=('owner',))
    tail = RP.Tail(20, list(range(20, 24)), 4)
    lab = C['label_ids'].cuda()

    def get(split, ix):
        d = C[split]
        n = int(d['mask'][ix].sum(1).max())
        return d['h'][ix, :n].cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(), d['y'][ix].cuda()

    f32 = lambda th: [t.float() for t in th]

    @torch.no_grad()
    def evaluate(th, split='heldout', ref=None):
        corr = agree = tot = 0
        for ix in torch.arange(len(C[split]['y'])).split(256):
            b = get(split, ix)
            p = tail.forward(f32(th), *b[:3], lab).argmax(-1)
            corr += int((p == b[3]).sum()); tot += len(ix)
            if ref is not None:
                agree += int((p == tail.forward(f32(ref), *b[:3], lab).argmax(-1)).sum())
        return corr / tot, (agree / tot if ref is not None else None)

    def first_order(th0, split, ix_all, lr, steps, bsz, seed):
        """AdamW LoRA training on (split, ix_all) with dev-based selection (tune.json)."""
        params = [t.float().clone().requires_grad_(True) for t in th0]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        g = torch.Generator().manual_seed(seed)
        best = (-1, None)
        for s in range(1, steps + 1):
            ix = ix_all[torch.randperm(len(ix_all), generator=g)[:bsz]]
            b = get(split, ix)
            loss = F.cross_entropy(tail.forward(params, *b[:3], lab), b[3])
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 50 == 0 or s == steps:
                dev, _ = evaluate([p.detach() for p in params], 'tune')
                if dev > best[0]:
                    best = (dev, [p.detach().double().clone() for p in params])
        return best[1], best[0]

    def lora_dW(th):
        """Gauge-invariant per-module Delta W = B A * scale (8 modules: 4 layers x {q,v})."""
        return [(th[2 * k + 1].double() @ th[2 * k].double()) * tail.scale for k in range(len(th) // 2)]

    # ---------------- theta_0 ----------------
    std_init = [t.double().clone() for t in tail.init]          # owner's secret seed (seed 0 in Tail)
    tr = C['train']
    if a.scenario == 's1':
        theta0 = std_init
        cl_split, cl_ix = 'train', torch.arange(len(tr['y']))
        bsz = 64
    else:
        t0 = time.time()
        theta0, dev0 = first_order(std_init, 'owner', torch.arange(len(C['owner']['y'])), 1e-4, 800, 64, a.seed + 11)
        rec(stage='owner_pretrain', dev=dev0, test=evaluate(theta0)[0], sec=round(time.time() - t0, 1))
        g = torch.Generator().manual_seed(a.seed + 7)
        cl_split = 'train'
        cl_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:16]]
                           for c in sorted(set(tr['y'].tolist()))])
        bsz = 32
    rec(stage='theta0', test=evaluate(theta0)[0], n_client=len(cl_ix))

    # ---------------- protocol run, recording the client's view ----------------
    batch_rng = torch.Generator().manual_seed(a.seed + 1000)
    dir_rng = torch.Generator(device='cuda').manual_seed(a.seed + 2000)
    theta = [t.clone() for t in theta0]
    transcript = []                                           # last K steps: (ix, h, hdot, c)
    pub_ix = None
    if a.public:                                              # public unlabeled pool: train sentences not in the client's set
        pool = torch.tensor([i for i in range(len(tr['y'])) if i not in set(cl_ix.tolist())])
        pub_ix = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(a.seed + 77))[:a.public]]
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ix = cl_ix[torch.randperm(len(cl_ix), generator=batch_rng)[:bsz]]
        b = get(cl_split, ix)
        if a.train == 'jvp':
            u = [torch.randn(t.shape, device='cuda', generator=dir_rng) for t in theta]
            fh = lambda *th: tail.forward(list(th), *b[:3], lab, return_h=True)
            h, hdot = torch.func.jvp(fh, tuple(f32(theta)), tuple(u))       # revealed to the client
            hh = h.detach().requires_grad_(True)
            loss = F.cross_entropy(tail.head_logits(hh, lab), b[3])
            gh, = torch.autograd.grad(loss, hh)
            c = float((gh * hdot).sum())                                       # client computes c
            theta = [t - a.lr * c * v.double() for t, v in zip(theta, u)]     # inside MPC
        elif a.train == 'fo':                                     # SAP / Pencil: backprop, owner applies AdamW
            if step == 1:
                fo_params = [t.float().clone().requires_grad_(True) for t in theta]
                fo_opt = torch.optim.AdamW(fo_params, lr=a.lr, weight_decay=0.0)
            h = tail.forward(fo_params, *b[:3], lab, return_h=True)
            loss = F.cross_entropy(tail.head_logits(h, lab), b[3])
            fo_opt.zero_grad(); loss.backward(); fo_opt.step()
            h = h.detach(); c = float(loss)
            theta = [p.detach().double().clone() for p in fo_params]
        else:                                                     # SecP-Tuning style: two forward passes (SPSA)
            u = [torch.randn(t.shape, device='cuda', generator=dir_rng) for t in theta]
            with torch.no_grad():
                hp_ = tail.forward(f32([t + a.zo_eps * v for t, v in zip(theta, u)]), *b[:3], lab, return_h=True)
                hm_ = tail.forward(f32([t - a.zo_eps * v for t, v in zip(theta, u)]), *b[:3], lab, return_h=True)
                lp = F.cross_entropy(tail.head_logits(hp_, lab), b[3]); lm = F.cross_entropy(tail.head_logits(hm_, lab), b[3])
                h = tail.forward(f32(theta), *b[:3], lab, return_h=True)
            c = float((lp - lm) / (2 * a.zo_eps))
            theta = [t - a.lr * c * v.double() for t, v in zip(theta, u)]
        obs = h.detach() if a.reveal == 'h' else tail.head_logits(h.detach(), lab).detach()
        transcript.append((ix, obs, None, c))
        if a.public and step > a.steps - max(a.Ks):
            with torch.no_grad():
                pobs = []
                for pc in pub_ix.split(128):
                    bp = get(cl_split, pc)
                    hp_ = tail.forward(f32(theta), *bp[:3], lab, return_h=True)
                    pobs.append(hp_ if a.reveal == 'h' else tail.head_logits(hp_, lab))
            transcript[-1] = (torch.cat([ix, pub_ix]), torch.cat([obs] + pobs), None, c)
        transcript = transcript[-max(a.Ks):]
    thetaT = theta
    accT = evaluate(thetaT)[0]
    rec(stage='protocol', steps=a.steps, test_thetaT=accT, sec=round(time.time() - t0, 1))

    # ---------------- client-only baseline ----------------
    local, dev_l = first_order(std_init, cl_split, cl_ix, 1e-4, 600 if a.scenario == 's1' else 300, bsz, a.seed + 21)
    acc_l, agr_l = evaluate(local, ref=thetaT)
    rec(stage='baseline_local', test=acc_l, agree_thetaT=agr_l)
    rec(stage='baseline_zeroshot', test=evaluate([torch.zeros_like(t) if i % 2 else t for i, t in enumerate(std_init)], ref=thetaT)[0])

    # ---------------- attacks: fit observed h ----------------
    dWT = lora_dW(thetaT)
    def attack(use_labels, init_seed, K):
        obs_ix = torch.cat([t[0] for t in transcript[-K:]]); obs_h = torch.cat([t[1] for t in transcript[-K:]])
        gen = torch.Generator().manual_seed(1000 + init_seed)   # client does not know the owner's A seed
        H = 1024; bound = 1.0 / math.sqrt(H)
        th = []
        for k in range(len(std_init) // 2):
            th += [torch.empty(4, H).uniform_(-bound, bound, generator=gen).cuda(), torch.zeros(H, 4, device='cuda')]
        params = [t.clone().requires_grad_(True) for t in th]
        opt = torch.optim.Adam(params, lr=1e-3)
        n = len(obs_ix); g = torch.Generator().manual_seed(init_seed)
        hist = []
        for it in range(1, a.attack_iters + 1):
            sel = torch.randperm(n, generator=g)[:min(64, n)]
            b = get(cl_split, obs_ix[sel])
            hp = tail.forward(params, *b[:3], lab, return_h=True)
            pred = hp if a.reveal == 'h' else tail.head_logits(hp, lab)
            loss = F.mse_loss(pred, obs_h[sel])
            if use_labels:                                   # labels only for the client's own samples
                lm = torch.isin(obs_ix[sel], cl_ix).cuda()
                if lm.any():
                    loss = loss + F.cross_entropy(tail.head_logits(hp, lab)[lm], b[3][lm])
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 500 == 0 or it == a.attack_iters:
                thd = [p.detach().double() for p in params]
                dW = lora_dW(thd)
                rel = [float((x - y).norm() / y.norm()) for x, y in zip(dW, dWT)]
                acc, agr = evaluate(thd, ref=thetaT)
                hist.append(dict(it=it, mse=float(F.mse_loss(pred, obs_h[sel])), dW_rel_err_mean=sum(rel) / len(rel),
                                 dW_rel_err=rel, test=acc, agree_thetaT=agr))
        return hist[-1], hist

    for K in a.Ks:
        for use_labels in (False, True):
            final, hist = attack(use_labels, 1, K)
            rec(stage=f'attack_fit_{a.reveal}' + ('+y' if use_labels else ''), K=K, reveal=a.reveal, **final)
            json.dump(hist, open(out / f"attack_{a.reveal}{'y' if use_labels else ''}_K{K}_hist.json", 'w'), indent=1)
    # reference: how far is theta_T from theta_0 / a random LoRA, in the same metric
    rel0 = [float((x - y).norm() / y.norm()) for x, y in zip(lora_dW(theta0), dWT)]
    rec(stage='reference', dW_rel_err_theta0_vs_T=sum(rel0) / len(rel0), dW_norm_T=[float(x.norm()) for x in dWT])
    json.dump(vars(a), open(out / 'args.json', 'w'))
    print('LEAK_DONE', flush=True)


if __name__ == '__main__':
    main()
