#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
C="--ops lp --opt adam --k 16 --batch 32 --clip 1.0 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume --lr 1e-4 --steps 150 --eval-every 25 --seed 42"
run(){ tag=$1; shift; python stage2/t16_lp.py $C "$@" --out hparam/sst2_$tag > hparam/sst2_$tag.log 2>&1; echo "$tag $(date)" >> hparam/queue.log; }
if [ "$1" = A ]; then
  while kill -0 6888 2>/dev/null; do sleep 20; done; echo "tau1000 $(date)" >> hparam/queue.log
  run ls1024 --adam-scale 8192 --ls 1024 --tau 10
  run S2048  --adam-scale 2048 --ls 256  --tau 10
  echo WA_DONE >> hparam/queue.log
else
  run ls64   --adam-scale 8192 --ls 64    --tau 10
  run S32768 --adam-scale 32768 --ls 256 --tau 10
  echo WB_DONE >> hparam/queue.log
fi
