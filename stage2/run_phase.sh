#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26 SPLIT=16
while [ ! -f timing_net5/done ]; do sleep 20; done
python stage2/t38_phase.py --ops lp --opt adam --adam-scale 8192 --lr 1e-4 --k 16 --batch 32 --steps 5 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0 --out timing_net5/phase_sst > timing_net5/phase_sst.log 2>&1
echo DONE > timing_net5/done_phase
