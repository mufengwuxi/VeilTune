#!/bin/bash
cd /workspace/veiltune
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT=16 OUTD=rl_leak8
TASK=rte  python rl_leak.py --scenario s2 --Ks 1 100 --train fo --lr 1e-4 --reveal logits --public 512 --seed 44 > rl_leak8/logs/rte12.log 2>&1 &
sleep 5
TASK=yelp python rl_leak.py --scenario s2 --Ks 1 100 --train fo --lr 1e-4 --reveal logits --seed 44 > rl_leak8/logs/yelp11.log 2>&1 &
sleep 5
TASK=yelp python rl_leak.py --scenario s2 --Ks 1 100 --train fo --lr 1e-4 --reveal logits --public 512 --seed 44 > rl_leak8/logs/yelp12.log 2>&1 &
wait; echo "ALLDONE44b $(date)" >> rl_leak8/s44b_queue.log
