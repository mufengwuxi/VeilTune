#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
env TASK=agnews OPS=lpmc LS=256 EXP_FREE=1 TAG=ag_mc python stage2/t12_lp_mc.py check > agdiag/check_ag_mc.log 2>&1
env OPS=lpmc LS=256 EXP_FREE=1 TAG=sst_mc python stage2/t12_lp_mc.py check > agdiag/check_sst2_mc.log 2>&1
echo DONE > agdiag/done_mc
