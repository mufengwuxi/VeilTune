"""Plaintext probe: max LayerNorm variance and max |GELU input| in the 8 adapted blocks (16-23) of RoBERTa-large
on SST-2 train+tune sentences with the LM-BFF prompt. Read-only."""
import json, torch
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer
ROOT = Path('/workspace/veiltune'); MID = 'FacebookAI/roberta-large'
tok = AutoTokenizer.from_pretrained(MID); m = AutoModelForMaskedLM.from_pretrained(MID).cuda().eval()
st = dict(var_max=0.0, d_max=0.0, gelu_in_max=0.0, var_q999=[])
def ln_hook(mod, inp):
    x = inp[0].float(); d = x - x.mean(-1, keepdim=True); v = (d * d).mean(-1)
    msk = st['mask']; v = v[msk]; st['var_max'] = max(st['var_max'], float(v.max())); st['d_max'] = max(st['d_max'], float(d[msk].abs().max()))
    st['var_q999'].append(v.flatten().cpu())
def g_hook(mod, inp, out):
    st['gelu_in_max'] = max(st['gelu_in_max'], float(out[st['mask']].abs().max()))
for i in range(16, 24):
    L = m.roberta.encoder.layer[i]
    L.attention.output.LayerNorm.register_forward_pre_hook(ln_hook); L.output.LayerNorm.register_forward_pre_hook(ln_hook)
    L.intermediate.dense.register_forward_hook(g_hook)
sents = []
for n in ('train', 'tune'):
    sents += [d['sentence'] + ' It was <mask> .' for d in json.loads((ROOT / f'data/{n}.json').read_text())]
sents = sents[:2000]
with torch.no_grad():
    for i in range(0, len(sents), 128):
        b = tok(sents[i:i + 128], return_tensors='pt', padding=True).to('cuda'); st['mask'] = b['attention_mask'].bool()
        m(**b)
v = torch.cat(st.pop('var_q999')); st.pop('mask')
st.update(n=len(sents), var_q50=float(v.quantile(0.5)), var_q99=float(v.quantile(0.99)), var_frac_gt100=float((v > 100).float().mean()))
print('T33', json.dumps(st))
