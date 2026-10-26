#!/bin/bash
# Plaintext RoBERTa-large study (last-4-layer Q/V LoRA r=4). Runs alone on the GPU (no CrypTen jobs in parallel).
cd /workspace/veiltune
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
J=()
# k16 (16 per class, batch = whole set of 32): first-order, 3 lrs x 3 seeds
for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do
  J+=("--method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --out rl_runs3/fo_k16_lr${lr}_s$s"); done; done
# full 1024: first-order, 3 seeds
for s in 42 43 44; do J+=("--method fo --data full --batch 64 --lr 1e-4 --steps 2000 --eval-every 100 --seed $s --out rl_runs3/fo_full_lr1e-4_s$s"); done
# zeroth-order / forward-gradient for the step-count comparison
for lr in 3e-5 1e-4 3e-4; do J+=("--method zo --data k16 --batch 32 --lr $lr --steps 20000 --eval-every 1000 --seed 42 --out rl_runs3/zo_k16_lr${lr}_s42"); done
J+=("--method fg --data k16 --batch 32 --lr 1e-4 --steps 20000 --eval-every 1000 --seed 42 --out rl_runs3/fg_k16_lr1e-4_s42")
J+=("--method zo --data full --batch 64 --lr 1e-4 --steps 30000 --eval-every 1000 --seed 42 --out rl_runs3/zo_full_lr1e-4_s42")
J+=("--method fg --data full --batch 64 --lr 1e-4 --steps 30000 --eval-every 1000 --seed 42 --out rl_runs3/fg_full_lr1e-4_s42")
mkdir -p rl_runs3/logs; i=0
for j in "${J[@]}"; do
  python rl_plain.py $j > rl_runs3/logs/job$i.log 2>&1 &
  i=$((i+1)); while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 3; done
done
wait; echo DONE > rl_runs3/done
