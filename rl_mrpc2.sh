cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
# MRPC capacity probe: (a) rank 8 at 8 layers; (b) 12 LoRA layers rank 4. '?' template.
for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do
 TASK=mrpcq python rl_plain.py --method fo --data k16 --batch 32 --lr $lr --steps 600 --eval-every 50 --seed $s --split-at 16 --rank 8 --out rl_tasks/mrpcq_r8_fo_k16_sp16_lr${lr}_s$s > rl_tasks/logs/mrpcq_r8_${lr}_$s.log 2>&1
done; done
TASK=mrpcq python -c "import rl_plain as R; R.build_cache(12)" > rl_tasks/logs/cache_mrpcq12.log 2>&1
for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do
 TASK=mrpcq python rl_plain.py --method fo --data k16 --batch 32 --lr $lr --steps 600 --eval-every 50 --seed $s --split-at 12 --out rl_tasks/mrpcq_fo_k16_sp12_lr${lr}_s$s > rl_tasks/logs/mrpcq_sp12_${lr}_$s.log 2>&1
done; done
echo DONE > rl_tasks/done_mrpc2
