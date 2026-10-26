#!/bin/bash
cd /workspace/veiltune
export HF_HUB_OFFLINE=1
jobs_list=()
for d in full k16; do
  b=64; [ $d = k16 ] && b=32
  for lr in 1e-4 3e-4 1e-3; do jobs_list+=("--method fo --data $d --batch $b --lr $lr --steps 2000 --eval-every 100"); done
  for lr in 1e-4 3e-4 1e-3 3e-3 1e-2; do jobs_list+=("--method zo --data $d --batch $b --lr $lr --steps 10000 --eval-every 500"); done
  for lr in 3e-4 1e-3 3e-3; do jobs_list+=("--method fg --data $d --batch $b --lr $lr --steps 10000 --eval-every 500"); done
done
mkdir -p rl_runs/sweep1_logs
i=0
for j in "${jobs_list[@]}"; do
  python rl_plain.py $j > rl_runs/sweep1_logs/job$i.log 2>&1 &
  i=$((i+1))
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 2; done
done
wait
echo DONE > rl_runs/sweep1.done
