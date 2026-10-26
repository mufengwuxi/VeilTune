cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; mkdir -p dp_plain/logs
J=(); for lr in 0.05 0.1 0.02; do for e in inf 8 3 1; do J+=("--eps $e --lr $lr --C 1.0 --steps 500 --seed 42"); done; done
i=0; for j in "${J[@]}"; do python stage2/t21_plain_dp.py $j > dp_plain/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 5; done; done; wait; echo DONE > dp_plain/done
