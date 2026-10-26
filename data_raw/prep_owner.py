"""owner.json (model owner's private pretraining data) for Yelp and RTE, disjoint from the
client pool samples, dev (tune) and test (heldout).
Yelp: 4000 training examples after the pool and tune (same shuffle as prep.py), as for SST-2/AG News.
RTE: the official train set is exhausted by pool (2000) + tune (490), so the owner uses the pool examples that none of
the client subsets (16 per class, seeds 42-44, same sampler as rl_leak.py) contains."""
import json, random
from pathlib import Path
import pyarrow.parquet as pq, torch
R = Path('/workspace/veiltune'); D = R / 'data_raw'
T = pq.read_table(D / 'yelp/train.parquet').to_pylist(); rng = random.Random(0); rng.shuffle(T)
own = [dict(sentence=r['text'], label=int(r['label'])) for r in T[2512:6512]]
(R / 'data_yelp/owner.json').write_text(json.dumps(own)); print('yelp owner', len(own))
pool = json.loads((R / 'data_rte/train.json').read_text()); y = torch.tensor([r['label'] for r in pool])
used = set()
for s in (42, 43, 44):
    g = torch.Generator().manual_seed(s + 7)
    for c in sorted(set(y.tolist())):
        ix = torch.nonzero(y == c).flatten()[torch.randperm(int((y == c).sum()), generator=g)[:16]]
        used |= set(ix.tolist())
own = [r for i, r in enumerate(pool) if i not in used]
(R / 'data_rte/owner.json').write_text(json.dumps(own)); print('rte owner', len(own), 'excluded', len(used))
