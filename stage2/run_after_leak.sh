#!/bin/bash
# (1) t29 RTE exact-decode failure diag, (2) phase timing alone on GPU, (3) RTE secure gate 3 seeds.
cd /workspace/veiltune
while [ ! -f sec_train_adam/done_sst2_gate ] || [ ! -f rl_leak8/done_ag_fix ]; do sleep 60; done
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK=rte SPLIT=16 python stage2/t29_op_diag.py --batches 4 --theta sec_train_adam/rte_sp16_s42/theta_last.pt --out stage2/t29_rte_exact.json > stage2/t29_rte_exact.log 2>&1
echo DONE > stage2/done_t29_exact
export OPS=bp2 LS=256 EXP_FREE=1 SPLIT=16
FWD_ONLY=1 TAG=fwd_only_b32 python stage2/t12_rl_secure.py time --batches 1 32 > timing/fwd_only.log 2>&1
TAG=lossgrad_b32 python stage2/t12_rl_secure.py time --batches 1 32 > timing/lossgrad.log 2>&1
python stage2/t16_secure_train.py --opt adam --lr 1e-4 --k 16 --batch 32 --steps 3 --eval-every 1000 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 0 --out timing/step_gate > timing/step_gate.log 2>&1
echo DONE > timing/done
unset OPS LS EXP_FREE
for s in 42 43 44; do TASK=rte SPLIT=16 python stage2/t16_secure_train.py --opt adam --lr 3e-4 --k 16 --batch 32 --steps 100 --eval-every 25 --seed $s --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --gate 1 --diag 1 --resume --out sec_train_adam/rte_sp16_s${s}_gate > sec_train_adam/rte_s${s}_gate.log 2>&1; done
echo DONE > sec_train_adam/done_rte_gate
