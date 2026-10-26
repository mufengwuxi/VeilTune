cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 SPLIT=16 RS_KMAX=26
OPS=native EXP_FREE=0 LS=256 python stage2/t13_rl_profile.py --B 32 > stage2/t13_b32_native.log 2>&1
OPS=bp2 EXP_FREE=1 LS=256 python stage2/t13_rl_profile.py --B 32 > stage2/t13_b32_bp2.log 2>&1
echo DONE > stage2/done_t13_b32
