export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 OPS=bp2 LS=256
for seed in a b; do EXP_FREE=0 timeout 1200 python stage2/t12_rl_secure.py check 2>&1 | grep T12_CHECK | sed "s/^/ef0 /" >> stage2/ef.log; EXP_FREE=1 timeout 1200 python stage2/t12_rl_secure.py check 2>&1 | grep T12_CHECK | sed "s/^/ef1 /" >> stage2/ef.log; done
EXP_FREE=1 timeout 1200 python stage2/t13_rl_profile.py --B 64 > stage2/t13_bp2_ef1.log 2>&1
