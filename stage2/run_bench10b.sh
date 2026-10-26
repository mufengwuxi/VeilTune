#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
while ! grep -q ALLDONE bench/queue.log; do sleep 30; done
C="--opt adam --adam-scale 8192 --k 16 --batch 32 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0"
mkdir -p bench/sst2_wan200
env TASK=sst2 PROF_OUT=bench/sst2_wan200 NET_MBPS=200 NET_RTT_MS=40 NETEM_OUT=bench/sst2_wan200 python stage2/t37_netem.py stage2/t40_prof.py --ops lp --lr 1e-4 $C --steps 2 --out bench/sst2_wan200_run > bench/sst2_wan200.log 2>&1
echo "sst2_wan200 $(date)" >> bench/queue.log; echo ALLDONE_B >> bench/queue.log
