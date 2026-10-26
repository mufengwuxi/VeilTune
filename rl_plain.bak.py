"""RoBERTa-large plaintext study for the U-split secure LoRA fine-tuning (system model A).

Client computes embeddings + the 20 frozen layers locally (no LoRA there, public weights), so the
layer-20 hidden states are fixed per sample and cached once. Training touches only layers 20..23
(Q/V LoRA, rank r, alpha 16) + the MLM head (client side), written functionally so the same code
path serves first-order (autograd), forward-gradient (torch.func.jvp) and ZO (two-point SPSA).

Prompt (LM-BFF / MeZO SST-2): <s> sentence It was <mask> . </s>, label words ' terrible' / ' great'.
Data: data/train.json (1024) or a k-shot subset of it; dev = data/tune.json (512);
test = data/heldout.json (872 = SST-2 validation).
"""
import argparse, json, math, os, time
from pathlib import Path

ROOT = Path('/workspace/veiltune')
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = 'FacebookAI/roberta-large'
LENGTH = 64
CACHE = ROOT / 'rl_cache'


def encode(rows, tok):
    pre = tok.encode(' It was', add_special_tokens=False)
    post = tok.encode('.', add_special_tokens=False)
    ids_l, mask_l, pos_l, y_l = [], [], [], []
    for r in rows:
        s = tok.encode(r['sentence'].strip(), add_special_tokens=False)
        s = s[:LENGTH - len(pre) - len(post) - 3]
        ids = [tok.bos_token_id] + s + pre + [tok.mask_token_id] + post + [tok.eos_token_id]
        n = len(ids)
        ids_l.append(ids + [tok.pad_token_id] * (LENGTH - n)); mask_l.append([1] * n + [0] * (LENGTH - n))
        pos_l.append(ids.index(tok.mask_token_id)); y_l.append(r['label'])
    return [torch.tensor(x) for x in (ids_l, mask_l, pos_l, y_l)]


@torch.no_grad()
def build_cache(split_at):
    """Layer-`split_at` input hidden states for all splits (client-side frozen prefix)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f'h{split_at}.pt'
    if f.exists():
        return torch.load(f)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                                 attn_implementation='eager').cuda().eval()
    out = {}
    for name in ['train', 'tune', 'heldout']:
        ids, mask, pos, y = encode(json.loads((ROOT / f'data/{name}.json').read_text()), tok)
        hs = []
        for ix in torch.arange(len(ids)).split(128):
            o = model.roberta(input_ids=ids[ix].cuda(), attention_mask=mask[ix].cuda(), output_hidden_states=True)
            hs.append(o.hidden_states[split_at].float().cpu())
        out[name] = dict(h=torch.cat(hs), mask=mask, pos=pos, y=y, ids=ids)
    lw = [tok.encode(' terrible', add_special_tokens=False), tok.encode(' great', add_special_tokens=False)]
    assert all(len(w) == 1 for w in lw), lw
    out['label_ids'] = torch.tensor([w[0] for w in lw])
    torch.save(out, f)
    return out


class Tail:
    """Functional layers split_at..23 + MLM head of RoBERTa-large (weights frozen, FP32)."""

    def __init__(self, split_at, lora_layers, rank, alpha=16):
        m = AutoModelForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                                 attn_implementation='eager').cuda().eval()
        self.cfg = m.config
        self.eps = m.config.layer_norm_eps
        self.layers = []
        for i in range(split_at, m.config.num_hidden_layers):
            L = m.roberta.encoder.layer[i]
            a, o = L.attention.self, L.attention.output
            g = lambda t: t.detach()
            self.layers.append(dict(
                i=i, Wq=g(a.query.weight), bq=g(a.query.bias), Wk=g(a.key.weight), bk=g(a.key.bias),
                Wv=g(a.value.weight), bv=g(a.value.bias), Wo=g(o.dense.weight), bo=g(o.dense.bias),
                g1=g(o.LayerNorm.weight), b1=g(o.LayerNorm.bias),
                Wf1=g(L.intermediate.dense.weight), bf1=g(L.intermediate.dense.bias),
                Wf2=g(L.output.dense.weight), bf2=g(L.output.dense.bias),
                g2=g(L.output.LayerNorm.weight), b2=g(L.output.LayerNorm.bias)))
        h = m.lm_head
        self.head = dict(W=h.dense.weight.detach(), b=h.dense.bias.detach(), g=h.layer_norm.weight.detach(),
                         be=h.layer_norm.bias.detach(), dec=h.decoder.weight.detach(), decb=h.bias.detach())
        self.lora_layers = lora_layers
        self.scale = alpha / rank
        H = m.config.hidden_size
        gen = torch.Generator().manual_seed(0)
        self.init = []                    # per LoRA layer: Aq [r,H], Bq [H,r], Av, Bv  (A kaiming-uniform, B=0)
        for _ in lora_layers:
            for _ in ('q', 'v'):
                A = torch.empty(rank, H)
                bound = 1.0 / math.sqrt(H)   # kaiming_uniform_(a=sqrt(5)) on [r,H] -> U(-1/sqrt(H), 1/sqrt(H))
                A.uniform_(-bound, bound, generator=gen)
                self.init += [A.cuda(), torch.zeros(H, rank, device='cuda')]
        del m
        torch.cuda.empty_cache()

    def forward(self, theta, h, mask, pos, label_ids):
        B, T, H = h.shape
        nh, hd = self.cfg.num_attention_heads, H // self.cfg.num_attention_heads
        add = (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        k = 0
        for P in self.layers:
            last = P['i'] == self.layers[-1]['i']
            q = h @ P['Wq'].T + P['bq']
            v = h @ P['Wv'].T + P['bv']
            if P['i'] in self.lora_layers:
                Aq, Bq, Av, Bv = theta[k:k + 4]; k += 4
                q = q + (h @ Aq.T) @ Bq.T * self.scale
                v = v + (h @ Av.T) @ Bv.T * self.scale
            kk = h @ P['Wk'].T + P['bk']
            if last:                                   # only the MASK row continues (U-split head input)
                sel = lambda t: t[torch.arange(B, device=t.device), pos][:, None]
                q, hq = sel(q), sel(h)
            else:
                hq = h
            Tq = q.shape[1]
            sp = lambda t, n: t.view(B, n, nh, hd).transpose(1, 2)
            s = sp(q, Tq) @ sp(kk, T).transpose(-1, -2) / math.sqrt(hd) + add
            p = torch.softmax(s, -1)
            att = (p @ sp(v, T)).transpose(1, 2).reshape(B, Tq, H)
            x = F.layer_norm(hq + att @ P['Wo'].T + P['bo'], (H,), P['g1'], P['b1'], self.eps)
            z = F.gelu(x @ P['Wf1'].T + P['bf1'])
            h = F.layer_norm(x + z @ P['Wf2'].T + P['bf2'], (H,), P['g2'], P['b2'], self.eps)
        hm = h[:, 0]                                   # [B,H] MASK rows
        hh = self.head
        z = F.layer_norm(F.gelu(hm @ hh['W'].T + hh['b']), (H,), hh['g'], hh['be'], self.eps)
        return z @ hh['dec'][label_ids].T + hh['decb'][label_ids]   # [B,2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['fo', 'fg', 'zo', 'zeroshot', 'check'], required=True)
    ap.add_argument('--data', default='full', help="'full' or 'k<N>' (N per class from train.json)")
    ap.add_argument('--split-at', type=int, default=20)
    ap.add_argument('--lora-layers', type=int, nargs='*', default=None, help='default: split_at..23')
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    C = build_cache(a.split_at)
    lora_layers = a.lora_layers if a.lora_layers is not None else list(range(a.split_at, 24))
    tail = Tail(a.split_at, lora_layers, a.rank)
    lab = C['label_ids'].cuda()
    tr = C['train']
    if a.data == 'full':
        tr_ix = torch.arange(len(tr['y']))
    else:
        kshot = int(a.data[1:]); g = torch.Generator().manual_seed(a.seed + 7)
        tr_ix = torch.cat([torch.nonzero(tr['y'] == c).flatten()[torch.randperm(int((tr['y'] == c).sum()), generator=g)[:kshot]]
                           for c in (0, 1)])
    batch_rng = torch.Generator().manual_seed(a.seed + 1000)
    dir_rng = torch.Generator(device='cuda').manual_seed(a.seed + 2000)

    def get(split, ix):
        d = C[split]
        n = int(d['mask'][ix].sum(1).max())
        return d['h'][ix, :n].cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(), d['y'][ix].cuda()

    master = [t.double().clone() for t in tail.init]
    fwd = lambda th, b: tail.forward([t.float() for t in th], *b[:3], lab)

    @torch.no_grad()
    def evaluate(split):
        corr = tot = 0; ls = 0.0
        for ix in torch.arange(len(C[split]['y'])).split(256):
            b = get(split, ix)
            lo = fwd(master, b)
            corr += int((lo.argmax(-1) == b[3]).sum()); tot += len(ix); ls += float(F.cross_entropy(lo, b[3], reduction='sum'))
        return corr / tot, ls / tot

    if a.method == 'check':   # functional tail vs HF full forward (LoRA B = 0)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        m = AutoModelForMaskedLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
        d = C['tune']; ix = torch.arange(16)
        with torch.no_grad():
            lo_hf = m(input_ids=d['ids'][ix].cuda(), attention_mask=d['mask'][ix].cuda()).logits
            lo_hf = lo_hf[torch.arange(16), d['pos'][ix].cuda()][:, lab]
            lo_me = fwd(master, get('tune', ix))
        print('CHECK max|diff| =', float((lo_hf - lo_me).abs().max()), 'max|logit| =', float(lo_hf.abs().max()))
        return
    out = Path(a.out or f'{ROOT}/rl_runs/{a.method}_{a.data}_lr{a.lr}_s{a.seed}')
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'w')

    def record(step, extra=None):
        dacc, dloss = evaluate('tune'); tacc, _ = evaluate('heldout')
        rec = dict(step=step, dev_acc=dacc, dev_loss=dloss, test_acc=tacc, **(extra or {}))
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)
        return dacc
    record(0)
    if a.method == 'zeroshot':
        return
    opt = None
    if a.method == 'fo':
        params = [t.float().clone().requires_grad_(True) for t in master]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    bsz = min(a.batch, len(tr_ix))
    t0 = time.time(); csum = 0.0
    for step in range(1, a.steps + 1):
        ix = tr_ix[torch.randperm(len(tr_ix), generator=batch_rng)[:bsz]]
        b = get('train', ix)
        if a.method == 'fo':
            loss = F.cross_entropy(tail.forward(params, *b[:3], lab), b[3])
            opt.zero_grad(); loss.backward(); opt.step()
            master = [p.detach().double() for p in params]
            c = float(loss)
        else:
            u = [torch.randn(t.shape, device='cuda', generator=dir_rng, dtype=torch.float32) for t in master]
            if a.method == 'fg':
                thf = tuple(t.float() for t in master)
                f = lambda *th: F.cross_entropy(tail.forward(list(th), *b[:3], lab), b[3])
                _, c = torch.func.jvp(f, thf, tuple(u))
                c = float(c)
            else:
                with torch.no_grad():
                    lp = F.cross_entropy(fwd([t + a.eps * v for t, v in zip(master, u)], b), b[3])
                    lm = F.cross_entropy(fwd([t - a.eps * v for t, v in zip(master, u)], b), b[3])
                c = float(lp - lm) / (2 * a.eps)
            master = [t - a.lr * c * v.double() for t, v in zip(master, u)]
        csum += abs(c)
        if step % a.eval_every == 0 or step == a.steps:
            record(step, dict(mean_abs_c=csum / a.eval_every, sec=round(time.time() - t0, 1)))
            csum = 0.0
    torch.save([t.cpu() for t in master], out / 'master_final.pt')
    json.dump(vars(a) | dict(n_train=len(tr_ix)), open(out / 'args.json', 'w'))


if __name__ == '__main__':
    main()
