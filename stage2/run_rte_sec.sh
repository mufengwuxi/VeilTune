cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TASK=rte PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT=16 python stage2/t16_secure_train.py --opt adam --lr 3e-4 --k 16 --batch 32 --steps 100 --eval-every 25 --seed 42 --clip 1.0 --tau 10 --qclip 4 --ln-sc 0.25 --diag 1 --resume --out sec_train_adam/rte_sp16_s42 > sec_train_adam/rte_s42.log 2>&1
echo DONE > sec_train_adam/done_rte
