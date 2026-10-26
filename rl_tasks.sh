cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; mkdir -p rl_tasks/logs
for t in mrpc rte agnews; do TASK=$t python -c "import rl_plain as R; R.build_cache(16)" > rl_tasks/logs/cache_$t.log 2>&1; done
J=(); for t in mrpc rte agnews; do for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do J+=("$t --method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --split-at 16 --out rl_tasks/${t}_fo_k16_sp16_lr${lr}_s$s"); done; done
  J+=("$t --method zeroshot --data k16 --split-at 16 --out rl_tasks/${t}_zeroshot"); done
i=0; for j in "${J[@]}"; do set -- $j; t=$1; shift; TASK=$t python rl_plain.py "$@" > rl_tasks/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 5; done; done; wait; echo DONE > rl_tasks/done
