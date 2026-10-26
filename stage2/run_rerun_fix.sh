#!/bin/bash
cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_KMAX=26
C="--opt adam --adam-scale 8192 --k 16 --batch 32 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume"
for s in 42 43 44; do
  TASK=rte SPLIT=16 python stage2/t16_secure_train_v2.py $C --lr 3e-4 --steps 100 --eval-every 25 --seed $s --out sec_train_adam/rte_sp16_s${s}_fix > sec_train_adam/rte_s${s}_fix.log 2>&1
done; echo DONE > sec_train_adam/done_rte_fix
for s in 42 43 44; do
  SPLIT=16 python stage2/t16_secure_train_v2.py $C --lr 1e-4 --steps 150 --eval-every 25 --seed $s --out sec_train_adam/sst2_sp16_s${s}_fix > sec_train_adam/sst2_s${s}_fix.log 2>&1
done; echo DONE > sec_train_adam/done_sst2_fix
for s in 42 43 44; do
  TASK=yelp SPLIT=16 python stage2/t16_secure_train_v2.py $C --lr 1e-4 --steps 150 --eval-every 25 --seed $s --out sec_train_adam/yelp_sp16_s${s}_fix > sec_train_adam/yelp_s${s}_fix.log 2>&1
done; echo DONE > sec_train_adam/done_yelp_fix
SPLIT=16 python stage2/t16_secure_train_v2.py --opt adam --adam-scale 8192 --lr 1e-4 --k 16 --batch 32 --steps 3 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0 --out timing/step_gate_fix > timing/step_gate_fix.log 2>&1
echo DONE > timing/done_fix
