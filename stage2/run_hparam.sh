#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
# H2: one-step gradient error vs loss scale (B=4, three batches)
for ls in 1 16 64 256 1024 4096; do for b in 1 2 3; do
  env OPS=lp LS=$ls EXP_FREE=1 BSEED=$b TAG=hp_b$b python stage2/t39_hp_check.py check > hparam/check_ls${ls}_b${b}.log 2>&1
done; done
echo H2DONE >> hparam/queue.log
# H1: end-to-end secure training on SST-2, seed 42, one hyperparameter changed at a time
C="--ops lp --opt adam --k 16 --batch 32 --clip 1.0 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume --lr 1e-4 --steps 150 --eval-every 25 --seed 42"
run(){ tag=$1; shift; python stage2/t16_lp.py $C "$@" --out hparam/sst2_$tag > hparam/sst2_$tag.log 2>&1; echo "$tag $(date)" >> hparam/queue.log; }
run tau1      --adam-scale 8192 --ls 256  --tau 1
run tau1000   --adam-scale 8192 --ls 256  --tau 1000
run ls64      --adam-scale 8192 --ls 64   --tau 10
run ls1024    --adam-scale 8192 --ls 1024 --tau 10
run S2048     --adam-scale 2048 --ls 256  --tau 10
run S32768    --adam-scale 32768 --ls 256 --tau 10
echo ALLDONE >> hparam/queue.log
