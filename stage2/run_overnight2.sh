#!/bin/bash
# User-approved 2026-10-15 14:43 UTC: after RTE gate -> Yelp secure s42 -> CMA-ES (BBT-style, LoRA subspace) plaintext baseline, 4 tasks x 3 seeds.
cd /workspace/veiltune
while [ ! -f sec_train_adam/done_rte_gate ]; do sleep 60; done
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK=yelp SPLIT=16 python stage2/t16_secure_train.py --opt adam --lr 1e-4 --k 16 --batch 32 --steps 150 --eval-every 25 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume --out sec_train_adam/yelp_sp16_s42_gate > sec_train_adam/yelp_s42_gate.log 2>&1
echo DONE > sec_train_adam/done_yelp_gate
SC=$(cat cma8/scale.txt 2>/dev/null || echo 1)
for T in sst2 yelp agnews rte; do for s in 42 43 44; do
  o=cma8/${T}_sc${SC}_s${s}; [ -f $o/master_final.pt ] && continue
  TASK=$T python rl_plain.py --method cma --cma-scale $SC --data k16 --split-at 16 --batch 64 --steps 400 --eval-every 25 --seed $s --out $o > $o.log 2>&1
done; done
echo DONE > cma8/done
