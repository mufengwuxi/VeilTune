#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
while [ ! -f ablation_lp/done ]; do sleep 60; done
python stage2/t35_adam_cost.py > ablation_lp/adam_cost.log 2>&1
env OPS=native LS=1 EXP_FREE=0 PAD=1 MASK_ONLY=0 LN_SC=1 NEW_GELU=0 MAX_TREE=0 TAG=all_off python stage2/t12_lp.py check > ablation_lp/check_all_off.log 2>&1
echo DONE > ablation_lp/done_extra
bash stage2/run_plain_noskip.sh > plain_noskip/queue.log 2>&1
bash stage2/run_agnews_sec.sh > agnews_lrsel_queue.log 2>&1
