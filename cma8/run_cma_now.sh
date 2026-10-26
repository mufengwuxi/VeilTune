#!/bin/bash
cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SC=0.3
for T in sst2 yelp agnews rte; do for s in 42 43 44; do
  o=cma8/${T}_sc${SC}_s${s}; [ -f $o/master_final.pt ] && continue
  TASK=$T python rl_plain.py --method cma --cma-scale $SC --data k16 --split-at 16 --batch 64 --steps 400 --eval-every 25 --seed $s --out $o > $o.log 2>&1
done; done
echo DONE > cma8/done
