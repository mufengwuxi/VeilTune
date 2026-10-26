#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26
C="--ops lp --opt adam --adam-scale 8192 --k 16 --batch 32 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume"
for s in 42 43 44; do
  TASK=rte SPLIT=16 python stage2/t16_lp.py $C --lr 3e-4 --steps 100 --eval-every 25 --seed $s --out sec_train_lp/rte_sp16_s${s} > sec_train_lp/rte_s${s}.log 2>&1
done; echo DONE > sec_train_lp/done_rte
for s in 42 43 44; do
  TASK=yelp SPLIT=16 python stage2/t16_lp.py $C --lr 1e-4 --steps 150 --eval-every 25 --seed $s --out sec_train_lp/yelp_sp16_s${s} > sec_train_lp/yelp_s${s}.log 2>&1
done; echo DONE > sec_train_lp/done_yelp
