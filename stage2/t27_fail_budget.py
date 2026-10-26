"""Plaintext failure budget: sum |value| * 2^-31 over truncated tensors in layers 16..23 forward (per batch of 32),
per tensor kind, on TASK's tune split. Kinds: ln_sq (d*d before LN mean), attn_scores (QK^T/sqrt(d)), ffn_pre (pre-GELU),
ffn_out, q/k/v, attn_out. Each fixed-point product with |result| p fails w.p. ~p*2^-31."""
import json, os, sys, torch
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
from transformers import AutoModelForMaskedLM
N = int(os.environ.get('N', '48'))
C = RP.build_cache(16); d = C['tune']
m = AutoModelForMaskedLM.from_pretrained(RP.MODEL_ID, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
acc = {}
def add(k, v): acc[k] = acc.get(k, 0.0) + float(v)
mask = None
def ln_hook(name):
    def h(mod, inp):
        x = inp[0][mask].double(); dv = x - x.mean(-1, keepdim=True); add('ln_sq', (dv * dv).sum())
    return h
def out_hook(name):
    def h(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        add(name, o[mask].double().abs().sum())
    return h
for i in range(16, 24):
    L = m.roberta.encoder.layer[i]
    L.attention.output.LayerNorm.register_forward_pre_hook(ln_hook('ln1')); L.output.LayerNorm.register_forward_pre_hook(ln_hook('ln2'))
    for nm, mod in (('q', L.attention.self.query), ('k', L.attention.self.key), ('v', L.attention.self.value),
                    ('attn_out', L.attention.output.dense), ('ffn_pre', L.intermediate.dense), ('ffn_out', L.output.dense)):
        mod.register_forward_hook(out_hook(nm))
scores = []
def attn_scores_hook(mod, inp, out):
    pass
ix = torch.arange(N)
am = d['mask'][ix].cuda()
mask = am.bool()
with torch.no_grad():
    o = m(input_ids=d['ids'][ix].cuda(), attention_mask=am, output_attentions=False, output_hidden_states=True)
    # attention scores: recompute QK^T for layers 16..23 from hidden states
    for i in range(16, 24):
        L = m.roberta.encoder.layer[i].attention.self; h = o.hidden_states[i]
        q = L.query(h).view(N, -1, 16, 64).transpose(1, 2); k = L.key(h).view(N, -1, 16, 64).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / 8.0
        mm = (mask[:, None, :, None] & mask[:, None, None, :])
        add('attn_scores', s.double().abs()[mm.expand_as(s)].sum())
scale = 2.0 ** -31 * 32 / N
res = {k: v * scale for k, v in acc.items()}
res['TOTAL'] = sum(res.values()); res['tokens_per_sample'] = float(am.sum() / N)
print('T27', RP.TASK, json.dumps({k: round(v, 4) for k, v in res.items()}))
