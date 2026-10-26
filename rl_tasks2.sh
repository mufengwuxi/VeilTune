cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK=yelp python -c "import rl_plain as R; R.build_cache(16)" > rl_tasks/logs/cache_yelp.log 2>&1
J=()
for d in agnews_fo_k16_sp16_lr1e-3_s42 agnews_fo_k16_sp16_lr1e-3_s43 agnews_fo_k16_sp16_lr1e-3_s44 agnews_fo_k16_sp16_lr1e-4_s43 agnews_fo_k16_sp16_lr1e-4_s44 agnews_fo_k16_sp16_lr3e-4_s43 agnews_fo_k16_sp16_lr3e-4_s44; do
  lr=$(echo $d | sed 's/.*_lr\([^_]*\)_s.*/\1/'); s=${d##*_s}; J+=("agnews --method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --split-at 16 --out rl_tasks/$d"); done
J+=("agnews --method zeroshot --data k16 --split-at 16 --out rl_tasks/agnews_zeroshot")
for s in 42 43 44; do for lr in 1e-4 3e-4 1e-3; do J+=("yelp --method fo --data k16 --batch 32 --lr $lr --steps 1000 --eval-every 50 --seed $s --split-at 16 --out rl_tasks/yelp_fo_k16_sp16_lr${lr}_s$s"); done; done
J+=("yelp --method zeroshot --data k16 --split-at 16 --out rl_tasks/yelp_zeroshot")
i=100; for j in "${J[@]}"; do set -- $j; t=$1; shift; TASK=$t python rl_plain.py "$@" > rl_tasks/logs/j$i.log 2>&1 & i=$((i+1)); while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 5; done; done; wait; echo DONE > rl_tasks/done2
