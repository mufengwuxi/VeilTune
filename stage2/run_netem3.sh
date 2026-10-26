#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
CS="--ops lp --opt adam --adam-scale 8192 --lr 1e-4 --k 16 --batch 32 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0"
CA="--ops lpmc --opt adam --adam-scale 8192 --lr 3e-4 --k 16 --batch 32 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0"
run(){ tag=$1; mb=$2; rtt=$3; task=$4; st=$5; mkdir -p timing_net5/$tag
  if [ "$task" = ag ]; then S=stage2/t16_lp_mc.py; C="$CA"; T="TASK=agnews"; else S=stage2/t16_lp.py; C="$CS"; T=""; fi
  env $T NET_MBPS=$mb NET_RTT_MS=$rtt NETEM_OUT=timing_net5/$tag python stage2/t37_netem.py $S $C --steps $st --out timing_net5/$tag/run > timing_net5/$tag.log 2>&1; }
run sst_s1 0 0 sst 1; run sst_s3 0 0 sst 3
run ag_s1 0 0 ag 1; run ag_s3 0 0 ag 3
run ag_lan 3000 0.8 ag 3
echo DONE > timing_net5/done
