#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16 TASK=agnews
while [ ! -f plain_noskip/done ]; do sleep 60; done
mkdir -p agnews_lrsel
for lr in 1e-4 3e-4; do for s in 42 43 44; do
  python stage2/t33_plain_noskip.py --mode noskip --seclog sec_train_lp/sst2_s42.log --lr $lr --steps 150 --seed $s --out agnews_lrsel/lr${lr}_s${s} > agnews_lrsel/lr${lr}_s${s}.log 2>&1
done; done
LR=$(python3 - <<'PY'
import json,glob,statistics as st
best=None
for lr in ['1e-4','3e-4']:
    d=[json.loads(open(f).read().strip().splitlines()[-1])['dev_acc'] for f in sorted(glob.glob(f'agnews_lrsel/lr{lr}_s4?.jsonl'))]
    m=st.mean(d); print(lr,m,file=open('agnews_lrsel/choice.txt','a'))
    if best is None or m>best[1]: best=(lr,m)
print(best[0])
PY
)
echo "chosen lr $LR" >> agnews_lrsel/choice.txt
C="--ops lp --opt adam --adam-scale 8192 --k 16 --batch 32 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume"
for s in 42 43 44; do
  python stage2/t16_lp.py $C --lr $LR --steps 150 --eval-every 25 --seed $s --out sec_train_lp/agnews_sp16_s${s} > sec_train_lp/agnews_s${s}.log 2>&1
done; echo DONE > sec_train_lp/done_agnews
