#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26
while kill -0 17901 2>/dev/null; do sleep 30; done
for o in lp bp2; do
  SPLIT=16 python stage2/t16_lp.py --ops $o --opt adam --adam-scale 8192 --lr 1e-4 --k 16 --batch 32 --steps 3 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0 --out timing/step_${o}_blackwell > timing/step_${o}_blackwell.log 2>&1
done
echo DONE > timing/done_lp_timing
