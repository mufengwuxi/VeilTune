#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
run(){ tag=$1; shift; env "$@" TAG=$tag python stage2/t12_lp_tb.py time --batches 32 > ablation_tb/time_$tag.log 2>&1; }
run full        OPS=lp LS=256 EXP_FREE=1
run fwd_only    OPS=lp LS=256 EXP_FREE=1 FWD_ONLY=1
run no_pack     OPS=lp LS=256 EXP_FREE=1 PAD=1
run no_maskonly OPS=lp LS=256 EXP_FREE=1 MASK_ONLY=0
run builtin_ops OPS=native LS=256 EXP_FREE=0
run all_off     OPS=native LS=1 EXP_FREE=0 PAD=1 MASK_ONLY=0 LN_SC=1 NEW_GELU=0 MAX_TREE=0
echo DONE > ablation_tb/done
