"""Macro-F1 (and accuracy) on heldout for the dev-selected run per seed. Usage: TASK=agnews python t26_f1.py <glob-prefix>
e.g. rl_tasks/agnews_fo_k16_sp16 or zo8/agnews ; selection = best final dev_acc among lrs per seed."""
import glob, json, sys, torch
sys.path.insert(0, '/workspace/veiltune')
import rl_plain as RP
pref = sys.argv[1]; split_at = 16
C = RP.build_cache(split_at); tail = RP.Tail(split_at, list(range(split_at, 24)), 4); lab = C['label_ids'].cuda()
d = C['heldout']; ncls = len(lab)
def L(f): return [json.loads(l) for l in open(f)]
res = {}
for s in (42, 43, 44):
    runs = [f for f in glob.glob(f'{pref}_s{s}/log.jsonl') if glob.glob(f.replace('log.jsonl', 'master_final.pt'))]
    if not runs: continue
    best = max(runs, key=lambda f: L(f)[-1]['dev_acc'])
    th = [t.float().cuda() for t in torch.load(best.replace('log.jsonl', 'master_final.pt'))]
    pred = []
    with torch.no_grad():
        for ix in torch.arange(len(d['y'])).split(256):
            n = int(d['mask'][ix].sum(1).max())
            lo = tail.forward(th, d['h'][ix, :n].cuda(), d['mask'][ix, :n].cuda(), d['pos'][ix].cuda(), lab)
            pred.append(lo.argmax(-1).cpu())
    p = torch.cat(pred); y = d['y']
    f1s = []
    for c in range(ncls):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    res[s] = dict(run=best.split('/')[-2], acc=float((p == y).float().mean()), macro_f1=sum(f1s) / ncls, f1_per_class=f1s)
    print('T26', s, json.dumps(res[s]), flush=True)
m = lambda k: sum(r[k] for r in res.values()) / len(res)
print('T26_MEAN', pref, 'acc', round(m('acc') * 100, 2), 'macroF1', round(m('macro_f1') * 100, 2))
json.dump(res, open(f'/workspace/veiltune/stage2/t26_{pref.replace("/", "_")}.json', 'w'), indent=1)
