#!/bin/bash
cd /workspace/veiltune
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT=16 OUTD=rl_leak8
jobs_for(){ for s in 42 43 44; do
  echo "--train fo --lr 1e-4 --reveal h --seed $s"
  echo "--train fo --lr 1e-4 --reveal logits --seed $s"
  echo "--train fo --lr 1e-4 --reveal logits --public 512 --seed $s"
  echo "--train cma --cma-scale 0.1 --reveal logits --seed $s"
  echo "--train cma --cma-scale 0.1 --reveal logits --public 512 --seed $s"; done; }
# first job of each task alone (builds the owner cache), then the rest in parallel
for t in yelp rte; do mapfile -t J < <(jobs_for); TASK=$t python rl_leak.py --scenario s2 --Ks 1 100 ${J[0]} > rl_leak8/logs/${t}0.log 2>&1; echo "$t first $(date)" >> rl_leak8/yr_queue.log; done
for t in yelp rte; do mapfile -t J < <(jobs_for)
  for i in $(seq 1 14); do TASK=$t python rl_leak.py --scenario s2 --Ks 1 100 ${J[$i]} > rl_leak8/logs/${t}$i.log 2>&1 &
    while [ $(jobs -r | wc -l) -ge 5 ]; do sleep 10; done; done; done
wait; echo DONE > rl_leak8/done_yr; echo "ALLDONE $(date)" >> rl_leak8/yr_queue.log
