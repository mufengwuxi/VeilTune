cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
for s in 42 43 44; do
  python stage2/t16_secure_train.py --lr 5e-3 --steps 300 --seed $s --clip 1.0 --tau 100 --diag 1 --eval-every 25 --resume --out sec_train/k16_lr5e-3_s$s > sec_train/k16_s$s.log 2>&1
done
echo DONE > sec_train/done
