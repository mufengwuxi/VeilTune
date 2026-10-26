import json, re, subprocess, sys, statistics as st
from pathlib import Path
W = Path('/workspace/fss_runs/work')
def parse(p):
    d = {}
    for line in open(p):
        m = re.match(r'([A-Za-z ]+)=(\d+) (us|B)', line.strip())
        if m: d[m.group(1).strip()] = int(m.group(2))
    return d
import os
plan = [tuple([a.split(':')[0], int(a.split(':')[1])]) for a in sys.argv[2:]]
RAW = sys.argv[1]
reps = 3
res = []
for m, n in plan:
    gb = int(os.environ.get('POOL_GB', '12' if (m == 'bert-base' or n >= 512) else '10'))
    for r in range(reps):
        out = subprocess.run(['/workspace/fss_runs/run_sigma.sh', m, str(n), str(gb), '12'],
                             capture_output=True, text=True).stdout
        ok = 'EXIT p0=0 p1=0' in out
        row = dict(model=m, seq=n, rep=r, ok=ok)
        if ok:
            for p in (0, 1):
                dd = W / f'output/P{p}/models/{m}-{n}'
                row[f'p{p}_dealer'] = parse(dd / 'dealer.txt'); row[f'p{p}_eval'] = parse(dd / 'evaluator.txt')
        else:
            row['tail'] = out[-800:]
        res.append(row)
        print(json.dumps({k: row[k] for k in ('model', 'seq', 'rep', 'ok')} |
                         ({'online_s': row['p0_eval']['Total time'] / 1e6} if ok else {})), flush=True)
        json.dump(res, open(RAW, 'w'), indent=1)
        if not ok: break
print('BENCH_DONE', flush=True)
