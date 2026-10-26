"""Plaintext: magnitudes entering the LayerNorms of layers 20..23 (pre-LN residual sums), MASK rows vs other rows."""
import json, torch, sys
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
from transformers import AutoModelForMaskedLM
C = RP.build_cache(20); d = C['tune']
m = AutoModelForMaskedLM.from_pretrained(RP.MODEL_ID, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
rec = {}
def mk(name):
    def hook(mod, inp):
        x = inp[0]; rec.setdefault(name, []).append(x.detach())
    return hook
for i in (20, 21, 22, 23):
    L = m.roberta.encoder.layer[i]
    L.attention.output.LayerNorm.register_forward_pre_hook(mk(f'L{i}_ln1')); L.output.LayerNorm.register_forward_pre_hook(mk(f'L{i}_ln2'))
ix = torch.arange(128)
with torch.no_grad():
    m(input_ids=d['ids'][ix].cuda(), attention_mask=d['mask'][ix].cuda())
mask = d['mask'][ix].bool().cuda(); pos = d['pos'][ix].cuda()
isM = torch.zeros_like(mask); isM[torch.arange(128), pos] = True
out = {}
for k, v in rec.items():
    x = v[0]; dev = x - x.mean(-1, keepdim=True); var = (dev * dev).mean(-1)
    for nm, sel in (('mask', isM), ('other', mask & ~isM)):
        out[f'{k}_{nm}'] = dict(absmax=float(x[sel].abs().max()), var_max=float(var[sel].max()), var_med=float(var[sel].median()))
print('T24', json.dumps(out))
