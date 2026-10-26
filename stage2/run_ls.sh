export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
for cfg in "bp 1" "bp 256" "bp2 256" "bp 4096" "bp2 4096"; do set -- $cfg
  OPS=$1 LS=$2 timeout 1200 python stage2/t12_rl_secure.py check 2>&1 | grep T12_CHECK | sed "s/^/$1 LS=$2 /" >> stage2/ls_sweep.log
done
