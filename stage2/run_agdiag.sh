#!/bin/bash
# 2026-10-21: diagnose AG News secure divergence: gradient error vs loss scale
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16 TASK=agnews
for ls in 256 32 8; do env OPS=lp LS=$ls EXP_FREE=1 TAG=ag_ls$ls python stage2/t12_lp.py check > agdiag/check_ls$ls.log 2>&1; done
env TASK=sst2 OPS=lp LS=256 EXP_FREE=1 TAG=sst_ls256 python stage2/t12_lp.py check > agdiag/check_sst2_ls256.log 2>&1
echo DONE > agdiag/done
