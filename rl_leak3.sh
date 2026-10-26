cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
J=()
for s in 42 43 44; do
  J+=("--train fo --lr 1e-4 --reveal h --seed $s")                    # SAP
  J+=("--train fo --lr 1e-4 --reveal logits --seed $s")               # Pencil (passive)
  J+=("--train fo --lr 1e-4 --reveal logits --public 512 --seed $s")  # Pencil (client adds public inputs)
  J+=("--train zo --lr 1e-4 --reveal logits --seed $s")               # SecP-Tuning (passive)
  J+=("--train zo --lr 1e-4 --reveal logits --public 512 --seed $s")  # SecP-Tuning (public inputs)
done
i=0; for j in "${J[@]}"; do python rl_leak.py --scenario s2 --Ks 1 20 100 $j > rl_leak3/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 10; done; done; wait; echo DONE > rl_leak3/done
