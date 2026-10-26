cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TASK=agnews
python -c "import rl_plain as R; R.build_cache(20, extra=('owner',))" > rl_leak3/logs/ag_cache.log 2>&1
J=()
for s in 42 43 44; do
  J+=("--train fo --lr 1e-4 --reveal h --seed $s")
  J+=("--train fo --lr 1e-4 --reveal logits --seed $s")
  J+=("--train fo --lr 1e-4 --reveal logits --public 512 --seed $s")
  J+=("--train zo --lr 1e-4 --reveal logits --seed $s")
  J+=("--train zo --lr 1e-4 --reveal logits --public 512 --seed $s")
done
i=0; for j in "${J[@]}"; do python rl_leak.py --scenario s2 --Ks 1 100 $j > rl_leak3/logs/ag_j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 10; done; done; wait; echo DONE > rl_leak3/done_ag
