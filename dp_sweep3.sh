cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
J=(); for s in 43 44; do J+=("--eps inf --lr 0.1 --seed $s" "--eps 8 --lr 0.1 --seed $s" "--eps 3 --lr 0.05 --seed $s" "--eps 1 --lr 0.02 --seed $s"); done
i=200; for j in "${J[@]}"; do python stage2/t21_plain_dp.py $j --C 1.0 --steps 500 > dp_plain/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done; done; wait; echo DONE > dp_plain/done3
