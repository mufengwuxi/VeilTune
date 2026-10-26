#!/bin/bash
cd /workspace/veiltune
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p rl_leak2/logs
J=()
for s in 42 43 44; do for r in h logits; do J+=("--scenario s2 --reveal $r --seed $s --Ks 1 5 20 100"); done; done
i=0
for j in "${J[@]}"; do
  python rl_leak.py $j > rl_leak2/logs/job$i.log 2>&1 &
  i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 5; done
done
wait; echo DONE > rl_leak2/done
