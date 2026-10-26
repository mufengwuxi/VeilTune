#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16 TASK=agnews
C="--ops lpmc --opt adam --adam-scale 8192 --k 16 --batch 32 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume"
for s in 42 43 44; do
  python stage2/t16_lp_mc.py $C --lr 3e-4 --steps 150 --eval-every 25 --seed $s --out sec_train_lp/agnews_mc_sp16_s${s} > sec_train_lp/agnews_mc_s${s}.log 2>&1
done; echo DONE > sec_train_lp/done_agnews_mc
