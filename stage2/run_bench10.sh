#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
C="--opt adam --adam-scale 8192 --k 16 --batch 32 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0"
run(){ tag=$1; task=$2; ops=$3; lr=$4; st=$5; mb=$6; rtt=$7; mkdir -p bench/$tag
  if [ "$mb" = 0 ]; then PRE=""; else PRE="stage2/t37_netem.py"; fi
  env TASK=$task PROF_OUT=bench/$tag NET_MBPS=$mb NET_RTT_MS=$rtt NETEM_OUT=bench/$tag python $PRE stage2/t40_prof.py --ops $ops --lr $lr $C --steps $st --out bench/${tag}_run > bench/$tag.log 2>&1; echo "$tag $(date)" >> bench/queue.log; }
run rte        rte    lp   3e-4 3 0 0
run rte_lan    rte    lp   3e-4 2 3000 0.8
run agnews_lan agnews lpmc 3e-4 2 3000 0.8
run yelp_lan   yelp   lp   1e-4 2 3000 0.8
echo LANDONE >> bench/queue.log
run rte_wan200    rte    lp   3e-4 2 200 40
run agnews_wan200 agnews lpmc 3e-4 2 200 40
run yelp_wan200   yelp   lp   1e-4 2 200 40
echo ALLDONE >> bench/queue.log
