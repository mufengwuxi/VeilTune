#!/bin/bash
# Leakage study at the main configuration (split 16 = 8 LoRA layers, rank 4). SecP-Tuning modeled with CMA-ES (BBT-style subspace).
cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT=16 OUTD=rl_leak8
SC=${SC:-0.1}
J=()
for s in 42 43 44; do
  J+=("--train fo --lr 1e-4 --reveal h --seed $s")
  J+=("--train fo --lr 1e-4 --reveal logits --seed $s")
  J+=("--train fo --lr 1e-4 --reveal logits --public 512 --seed $s")
  J+=("--train cma --cma-scale $SC --reveal logits --seed $s")
  J+=("--train cma --cma-scale $SC --reveal logits --public 512 --seed $s")
done
i=0; for j in "${J[@]}"; do python rl_leak.py --scenario s2 --Ks 1 20 100 $j > rl_leak8/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 10; done; done; wait; echo DONE > rl_leak8/done
