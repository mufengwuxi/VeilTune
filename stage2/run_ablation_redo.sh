cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 OPS=bp2 LS=256 EXP_FREE=1 SPLIT=16
FWD_ONLY=1 TAG=fwd_only python stage2/t12_rl_secure.py time --batches 1 1 32 64 > ablation/time_fwd_only.log 2>&1
LN_SC=1 PAD=1 MASK_ONLY=0 NEW_GELU=0 MAX_TREE=0 EXP_FREE=0 LS=1 TAG=all_off python stage2/t12_rl_secure.py time --batches 1 1 32 64 > ablation/time_all_off.log 2>&1
TAG=full_lnsc python stage2/t12_rl_secure.py time --batches 1 1 32 64 > ablation/time_full_lnsc.log 2>&1
LS=256 TAG=full_lnsc python stage2/t12_rl_secure.py check > ablation/check_full_lnsc.log 2>&1
echo DONE > ablation/done_redo
