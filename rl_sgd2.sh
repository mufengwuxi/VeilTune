cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; mkdir -p rl_sgd/logs
J=(); for s in 42 43 44; do for lr in 2e-3 5e-3 1e-2; do J+=("--method fo --opt sgd --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --out rl_sgd/k16_lr${lr}_s$s"); done; done
i=0; for j in "${J[@]}"; do python rl_plain.py $j > rl_sgd/logs/j2_$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 3; done; done; wait; echo DONE > rl_sgd/done2
