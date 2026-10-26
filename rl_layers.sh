cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; mkdir -p rl_layers/logs
python -c "import rl_plain as R; R.build_cache(16); R.build_cache(12); R.build_cache(8)" > rl_layers/logs/cache.log 2>&1
J=(); for sp in 16 12 8; do for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do J+=("--method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --split-at $sp --out rl_layers/fo_k16_sp${sp}_lr${lr}_s$s"); done; done; done
i=0; for j in "${J[@]}"; do python rl_plain.py $j > rl_layers/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 5; done; done; wait; echo DONE > rl_layers/done
