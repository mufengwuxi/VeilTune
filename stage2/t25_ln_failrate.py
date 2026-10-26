"""Plaintext estimate of local-truncation failures in the secure LN variance (d*d), per batch of 32 SST-2 inputs,
for LoRA layers 16..23 (both LNs per layer), with and without pre-scaling d by LN_SC before squaring.
Failure model: each fixed-point product with |result| = p fails with prob ~ p * 2^-31 (64-bit ring, f=16)."""
import json, torch, sys
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
from transformers import AutoModelForMaskedLM
C = RP.build_cache(16); d = C['tune']
m = AutoModelForMaskedLM.from_pretrained(RP.MODEL_ID, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
rec = {}
def mk(name):
    def hook(mod, inp):
        rec[name] = inp[0].detach().double()
    return hook
for i in range(16, 24):
    L = m.roberta.encoder.layer[i]
    L.attention.output.LayerNorm.register_forward_pre_hook(mk(f'L{i}_ln1')); L.output.LayerNorm.register_forward_pre_hook(mk(f'L{i}_ln2'))
N = 128; ix = torch.arange(N)
with torch.no_grad():
    m(input_ids=d['ids'][ix].cuda(), attention_mask=d['mask'][ix].cuda())
mask = d['mask'][ix].bool().cuda()
out = {}; tot = {1.0: 0.0, 0.5: 0.0, 0.25: 0.0, 0.125: 0.0}
for k in sorted(rec):
    x = rec[k][mask]; dev = x - x.mean(-1, keepdim=True); sq = dev * dev
    row = {}
    for sc in tot:
        e = float((sq * sc * sc).sum()) * 2.0 ** -31 * 32 / N; row[f'E_sc{sc}'] = e; tot[sc] += e
    top = sq.flatten().sort(descending=True)[0]; n = top.numel()
    row['share_top0.1pct'] = float(top[:max(1, n // 1000)].sum() / top.sum()); row['max_d'] = float(dev.abs().max())
    out[k] = row
out['TOTAL_per_batch32'] = tot
out['P_step_any_fail'] = {sc: 1 - float(torch.exp(torch.tensor(-v))) for sc, v in tot.items()}
print('T25', json.dumps(out))
json.dump(out, open('/workspace/veiltune/stage2/t25_ln_failrate.json', 'w'), indent=1)
