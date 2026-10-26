#!/bin/bash
# -> (2) RTE s42 gate run (bp2, as in the paper runs) with exact check of the Adam q = m/sqrt(v+eps^2), 40 steps.
cd /workspace/veiltune
while [ ! -f sec_train_adam/done_yelp_gate ]; do sleep 60; done
sleep 180
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK=rte SPLIT=16 python stage2/t30_op_diag_clamp.py --gelu-clamp 1 --batches 4 --theta sec_train_adam/rte_sp16_s42/theta_last.pt --out stage2/t30_rte_clamp.json > stage2/t30_rte_clamp.log 2>&1
echo DONE > stage2/done_t30_clamp
TASK=rte SPLIT=16 python stage2/t30_secure_train_qdiag.py --ops bp2 --qdiag 1 --opt adam --lr 3e-4 --k 16 --batch 32 --steps 40 --eval-every 20 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --out sec_train_adam/rte_s42_qdiag > sec_train_adam/rte_s42_qdiag.log 2>&1
echo DONE > sec_train_adam/done_rte_qdiag
