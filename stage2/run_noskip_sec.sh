#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26
C="--ops lp --opt adam --adam-scale 8192 --k 16 --batch 32 --clip 1.0 --tau 1e9 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume"
TASK=rte SPLIT=16 python stage2/t16_lp.py $C --lr 3e-4 --steps 100 --eval-every 25 --seed 42 --out sec_train_noskip/rte_sp16_s42 > sec_train_noskip/rte_s42.log 2>&1
echo DONE > sec_train_noskip/done_rte
