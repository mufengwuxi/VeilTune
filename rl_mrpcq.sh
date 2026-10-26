cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK=mrpcq python -c "import rl_plain as R; R.build_cache(16)" > rl_tasks/logs/cache_mrpcq.log 2>&1
TASK=mrpcq python rl_plain.py --method zeroshot --data k16 --split-at 16 --out rl_tasks/mrpcq_zeroshot > rl_tasks/logs/mrpcq_zs.log 2>&1
for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do
 TASK=mrpcq python rl_plain.py --method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --split-at 16 --out rl_tasks/mrpcq_fo_k16_sp16_lr${lr}_s$s > rl_tasks/logs/mrpcq_${lr}_$s.log 2>&1
done; done
TASK=mrpcq python stage2/t26_f1.py rl_tasks/mrpcq_fo_k16_sp16 > stage2/t26_mrpcq_fo.log 2>&1
echo DONE > rl_tasks/done_mrpcq
