cd /workspace/veiltune
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
OPS=bp LS=1 EXP_FREE=0 python stage2/t12_rl_secure.py time --batches 1 1 32 64 > stage2/t12_time_5090_bp.log 2>&1
OPS=bp2 LS=256 EXP_FREE=1 python stage2/t12_rl_secure.py time --batches 1 1 32 64 > stage2/t12_time_5090_bp2.log 2>&1
echo DONE > stage2/t12_time_5090.done
