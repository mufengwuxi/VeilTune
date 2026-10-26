cd /workspace/veiltune; export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TASK=agnews PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f rl_leak3/done_ag ]; do sleep 20; done
complete() { python - "$1" <<'P'
import json,sys
try:
    L=[json.loads(l) for l in open(sys.argv[1])]
    ok=sum(1 for r in L if r['stage'].startswith('attack') and r.get('K')==100)>=2
except Exception: ok=False
print('1' if ok else '0')
P
}
i=0
for s in 42 43 44; do
 for cfg in "fo h 0" "fo logits 0" "fo logits 512" "zo logits 0" "zo logits 512"; do
  set -- $cfg; tr=$1; rv=$2; pb=$3
  d=rl_leak3/agnews_${tr}_s2_${rv}$([ $pb -gt 0 ] && echo _pub$pb)_s$s
  [ "$(complete $d/log.jsonl)" = 1 ] && continue
  python rl_leak.py --scenario s2 --Ks 1 100 --train $tr --lr 1e-4 --reveal $rv --public $pb --seed $s > rl_leak3/logs/ag2_${tr}_${rv}_${pb}_$s.log 2>&1 &
  i=$((i+1)); while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 10; done
 done
done; wait; echo DONE > rl_leak3/done_ag2
