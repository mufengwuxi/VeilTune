#!/bin/bash
# AG News leakage at main config (8 layers, rank 4, SecP via CMA-ES). Starts after rl_leak8 finishes.
cd /workspace/veiltune; while [ ! -f rl_leak8/done ]; do sleep 60; done
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT=16 OUTD=rl_leak8 TASK=agnews
J=()
for s in 42 43 44; do
  J+=("--train fo --lr 1e-4 --reveal h --seed $s")
  J+=("--train fo --lr 1e-4 --reveal logits --seed $s")
  J+=("--train fo --lr 1e-4 --reveal logits --public 512 --seed $s")
  J+=("--train cma --cma-scale 0.1 --reveal logits --seed $s")
  J+=("--train cma --cma-scale 0.1 --reveal logits --public 512 --seed $s")
done
i=0; for j in "${J[@]}"; do python rl_leak.py --scenario s2 --Ks 1 100 $j > rl_leak8/logs/ag$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 10; done; done; wait; echo DONE > rl_leak8/done_ag
