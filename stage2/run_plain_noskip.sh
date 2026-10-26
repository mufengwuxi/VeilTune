#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 SPLIT=16
while [ ! -f ablation_lp/done ]; do sleep 60; done
run(){ task=$1; lr=$2; st=$3; s=$4; mode=$5; tv=""; [ "$task" != sst2 ] && tv="TASK=$task"
  env $tv python stage2/t33_plain_noskip.py --mode $mode --seclog sec_train_lp/${task}_s${s}.log --lr $lr --steps $st --seed $s --out plain_noskip/${task}_s${s}_${mode} > plain_noskip/${task}_s${s}_${mode}.log 2>&1; }
run sst2 1e-4 150 42 replay
for s in 42 43 44; do run sst2 1e-4 150 $s noskip; done
for s in 42 43 44; do run rte 3e-4 100 $s replay; run rte 3e-4 100 $s noskip; done
for s in 42 43 44; do run yelp 1e-4 150 $s replay; run yelp 1e-4 150 $s noskip; done
for s in 43 44; do run sst2 1e-4 150 $s replay; done
echo DONE > plain_noskip/done
