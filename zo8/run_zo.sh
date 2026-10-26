#!/bin/bash
# ZO-SGD (SPSA) plaintext 8 layers, k16; task lr seeds
cd /workspace/veiltune
T=$1; shift
for s in 42 43 44; do for lr in "$@"; do
  o=zo8/${T}_lr${lr}_s${s}_4k
  [ -f $o/master_final.pt ] && continue
  TASK=$T nice -n 10 python rl_plain.py --method zo --data k16 --split-at 16 --lr $lr --eps 1e-3 --batch 32 --steps 4000 --eval-every 100 --seed $s --out $o > $o.log 2>&1
done; done
