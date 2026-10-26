#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26
TASK=rte SPLIT=16 python stage2/t30_secure_train_qdiag.py --ops bp2 --qdiag 1 --opt adam --adam-scale 8192 --lr 3e-4 --k 16 --batch 32 --steps 20 --eval-every 10 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --out sec_train_adam/rte_s42_qdiag_fix > sec_train_adam/rte_s42_qdiag_fix.log 2>&1
echo DONE > sec_train_adam/done_qdiag_fix
