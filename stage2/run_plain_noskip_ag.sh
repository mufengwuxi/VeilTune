#!/bin/bash
# AG News (C-class head) plaintext replay + no-skip control, same as run_plain_noskip.sh; one at a time
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 SPLIT=16 TASK=agnews
for s in 42 43 44; do for mode in replay noskip; do
  python stage2/t33_plain_noskip.py --mode $mode --seclog sec_train_lp/agnews_mc_s${s}.log --lr 3e-4 --steps 150 --seed $s --out plain_noskip/agnews_s${s}_${mode} > plain_noskip/agnews_s${s}_${mode}.log 2>&1
done; done; echo DONE > plain_noskip/done_ag
