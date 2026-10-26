cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; mkdir -p rl_sgd/logs
J=(); for lr in 1e-2 3e-2 1e-1 3e-1 1 3; do J+=("--method fo --opt sgd --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed 42 --out rl_sgd/k16_lr${lr}_s42"); done
i=0; for j in "${J[@]}"; do python rl_plain.py $j > rl_sgd/logs/job$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 3; done; done; wait; echo DONE > rl_sgd/done
