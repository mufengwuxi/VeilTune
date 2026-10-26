#!/bin/bash
# waits for the CrypTen timing (t9) to finish so it is not perturbed, then runs the missing plaintext jobs
cd /workspace/veiltune
export HF_HUB_OFFLINE=1
while [ ! -f stage2/t9_primal.json ] && pgrep -f t9_primal.py >/dev/null; do sleep 5; done
J=()
# full data: forward-gradient (JVP) + longer ZO at the stable lr
for lr in 1e-4 3e-4; do J+=("--method fg --data full --batch 64 --lr $lr --steps 30000 --eval-every 1000"); done
J+=("--method zo --data full --batch 64 --lr 1e-4 --steps 30000 --eval-every 1000 --out rl_runs/zo_full_lr0.0001_s42_long")
# k16 (MeZO setting): FO / ZO / JVP
for lr in 1e-4 3e-4 1e-3; do J+=("--method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50"); done
for lr in 3e-5 1e-4 3e-4; do J+=("--method zo --data k16 --batch 32 --lr $lr --steps 20000 --eval-every 1000"); done
for lr in 1e-4 3e-4; do J+=("--method fg --data k16 --batch 32 --lr $lr --steps 20000 --eval-every 1000"); done
mkdir -p rl_runs/sweep2_logs; i=0
for j in "${J[@]}"; do
  python rl_plain.py $j > rl_runs/sweep2_logs/job$i.log 2>&1 &
  i=$((i+1)); while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 3; done
done
wait; echo DONE > rl_runs/sweep2.done
