cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
while [ ! -f sec_train/done ]; do sleep 30; done
sleep 20
mkdir -p baselines
# 1) CrypTen-native kernels: precision check (LS=256) then timing
OPS=native LS=256 EXP_FREE=0 timeout 3600 python stage2/t12_rl_secure.py check > baselines/native_check.log 2>&1
OPS=native LS=256 EXP_FREE=0 timeout 3600 python stage2/t12_rl_secure.py time --batches 1 1 32 64 > baselines/native_time.log 2>&1
# 2) secret base weights (our bp2 kernels), timing
OPS=bp2 LS=256 EXP_FREE=1 SECRET_BASE=1 timeout 3600 python stage2/t12_rl_secure.py time --batches 1 1 32 64 > baselines/secretbase_time.log 2>&1
# 3) per-op profile of the native baseline (bytes/rounds by phase)
OPS=native LS=256 EXP_FREE=0 timeout 3600 python stage2/t13_rl_profile.py --B 64 > baselines/native_profile.log 2>&1
echo DONE > baselines/done
