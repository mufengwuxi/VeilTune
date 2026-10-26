cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
bash stage2/run_ablation.sh
mkdir -p sec_train_adam
for s in 42 43 44; do
  SPLIT=16 python stage2/t16_secure_train.py --opt adam --lr 1e-4 --k 16 --batch 32 --steps 200 --eval-every 25 --seed $s --clip 1.0 --tau 100 --diag 1 --resume --out sec_train_adam/sst2_sp16_s$s > sec_train_adam/sst2_s$s.log 2>&1
done
echo DONE > sec_train_adam/done_sst2
