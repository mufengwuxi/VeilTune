#!/bin/bash
cd /workspace/veiltune
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT=16 OUTD=rl_leak8
J=(); for s in 42 43 44; do
  J+=("--train fo --lr 1e-4 --reveal h --seed $s" "--train fo --lr 1e-4 --reveal logits --seed $s" "--train fo --lr 1e-4 --reveal logits --public 512 --seed $s" "--train cma --cma-scale 0.1 --reveal logits --seed $s" "--train cma --cma-scale 0.1 --reveal logits --public 512 --seed $s"); done
order=(); for s in 0 1 2; do for t in rte yelp; do for k in 0 1 2 3 4; do order+=("$t $((s*5+k))"); done; done; done
for e in "${order[@]}"; do set -- $e; t=$1; i=$2
  [ -f rl_leak8/logs/${t}$i.log ] && continue
  while [ $(ps -eo cmd | grep -v grep | grep -c "python rl_leak.py") -ge 6 ]; do sleep 20; done
  TASK=$t python rl_leak.py --scenario s2 --Ks 1 100 ${J[$i]} > rl_leak8/logs/${t}$i.log 2>&1 &
  sleep 5
done
wait; echo "ALLDONE2 $(date)" >> rl_leak8/yr_queue.log
