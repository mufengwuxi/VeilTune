cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_HOME=/workspace/veiltune/cache/huggingface SPLIT=16 RS_KMAX=26
OPS=lp LS=256 EXP_FREE=1 TAG=lp python stage2/t12_lp.py check > stage2/lp_check.log 2>&1
OPS=lp LS=256 EXP_FREE=1 python stage2/t13_lp.py --B 32 > stage2/lp_t13.log 2>&1
OPS=bp2 LS=256 EXP_FREE=1 python stage2/t13_lp.py --B 32 > stage2/lp_t13_bp2.log 2>&1
echo DONE > stage2/done_lp1
