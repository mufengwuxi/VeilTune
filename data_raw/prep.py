"""parquet -> data_<task>/{train,tune,heldout}.json in rl_plain format. train = 2000-example pool (k-shot subsets are
drawn from it), tune = 512 disjoint train examples (dev), heldout = official validation (MRPC/RTE) or a fixed 2000
subsample of the official test set (Yelp-P, AG News)."""
import json, random, sys
from pathlib import Path
import pyarrow.parquet as pq
R = Path('/workspace/veiltune'); D = R / 'data_raw'
def rows(p):
    t = pq.read_table(p).to_pylist(); return t
cfg = {'mrpc': ('mrpc/train-00000-of-00001.parquet', 'mrpc/validation-00000-of-00001.parquet', None),
       'rte': ('rte/train-00000-of-00001.parquet', 'rte/validation-00000-of-00001.parquet', None),
       'yelp': ('yelp/train.parquet', 'yelp/test.parquet', 2000),
       'agnews': ('data/train-00000-of-00001.parquet', 'data/test-00000-of-00001.parquet', 2000)}
for task in sys.argv[1:]:
    tr, te, nte = cfg[task]
    T, E = rows(D / tr), rows(D / te)
    conv = (lambda r: dict(sentence=r['text'], label=int(r['label']))) if task in ('yelp', 'agnews') else \
           (lambda r: dict(sentence1=r['sentence1'], sentence2=r['sentence2'], label=int(r['label'])))
    rng = random.Random(0); rng.shuffle(T)
    pool, tune = T[:2000], T[2000:2512]
    if nte: rng.shuffle(E); E = E[:nte]
    out = R / f'data_{task}'; out.mkdir(exist_ok=True)
    for n, s in (('train', pool), ('tune', tune), ('heldout', E)):
        (out / f'{n}.json').write_text(json.dumps([conv(r) for r in s]))
    print(task, len(pool), len(tune), len(E), 'labels', sorted(set(r['label'] for r in pool)))
