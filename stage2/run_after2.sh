cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 OPS=bp2 SPLIT=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LN_SC=1 PAD=1 MASK_ONLY=0 NEW_GELU=0 MAX_TREE=0 EXP_FREE=0 LS=1 TAG=all_off python stage2/t12_rl_secure.py time --batches 1 1 32 > ablation/time_all_off_b32.log 2>&1
SPLIT=16 python stage2/t16_secure_train.py --opt adam --lr 1e-4 --k 16 --batch 32 --steps 3 --eval-every 3 --seed 42 --clip 1.0 --tau 1 --qclip 4 --ln-sc 0.25 --diag 0 --out sec_train_adam/time_diag0 > sec_train_adam/time_diag0.log 2>&1
echo DONE > ablation/done_after2
