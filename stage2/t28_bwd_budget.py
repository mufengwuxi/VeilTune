"""Backward failure budget (plaintext): sum |grad| * LS * 2^-31 of the activations' gradients in layers 16..23
(loss = batch-mean CE at the MASK position over the label words, as in the secure head), per batch of 32."""
import json, os, sys, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
from transformers import AutoModelForMaskedLM
N = int(os.environ.get('N', '32')); LS = 256.0
C = RP.build_cache(16); d = C['tune']; lab = C['label_ids'].cuda()
m = AutoModelForMaskedLM.from_pretrained(RP.MODEL_ID, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
for p in m.parameters(): p.requires_grad_(False)
outs = []
def keep(name):
    def h(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if o.requires_grad: o.retain_grad(); outs.append((name, o))
    return h
for i in range(16, 24):
    L = m.roberta.encoder.layer[i]
    for nm, mod in (('q', L.attention.self.query), ('k', L.attention.self.key), ('v', L.attention.self.value),
                    ('attn_out', L.attention.output.dense), ('ffn_pre', L.intermediate.dense), ('ffn_out', L.output.dense),
                    ('ln1_out', L.attention.output.LayerNorm), ('ln2_out', L.output.LayerNorm)):
        mod.register_forward_hook(keep(nm))
ix = torch.arange(N); am = d['mask'][ix].cuda(); mask = am.bool()
m.roberta.embeddings.word_embeddings.weight.requires_grad_(True)
logits = m(input_ids=d['ids'][ix].cuda(), attention_mask=am).logits
lo = logits[torch.arange(N), d['pos'][ix].cuda()][:, lab]
loss = F.cross_entropy(lo, d['y'][ix].cuda()); loss.backward()
acc = {}
for nm, o in outs:
    if o.grad is not None: acc[nm] = acc.get(nm, 0.0) + float(o.grad[mask].double().abs().sum()) * LS
scale = 2.0 ** -31 * 32 / N
res = {k: round(v * scale, 4) for k, v in acc.items()}; res['TOTAL'] = round(sum(res.values()), 4)
print('T28', RP.TASK, json.dumps(res))
