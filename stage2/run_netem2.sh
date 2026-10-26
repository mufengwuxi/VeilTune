#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
C="--ops lp --opt adam --adam-scale 8192 --lr 1e-4 --k 16 --batch 32 --steps 3 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0"
python stage2/t16_lp.py $C --out timing_net/direct/run > timing_net/direct.log 2>&1
run(){ tag=$1; mb=$2; rtt=$3; mkdir -p timing_net/$tag
  NET_MBPS=$mb NET_RTT_MS=$rtt NETEM_OUT=timing_net/$tag python stage2/t37_netem.py stage2/t16_lp.py $C --out timing_net/$tag/run > timing_net/$tag.log 2>&1; }
run none3 0 0; run wan200 200 40; run wan100 100 80
echo DONE > timing_net/done2
