cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 OPS=bp2 LS=256 EXP_FREE=1 SPLIT=16
mkdir -p ablation
run() { tag=$1; shift; env "$@" TAG=$tag python stage2/t12_rl_secure.py time --batches 1 1 32 64 > ablation/time_$tag.log 2>&1; [ "$tag" = fwd_only ] || env "$@" TAG=$tag python stage2/t12_rl_secure.py check > ablation/check_$tag.log 2>&1; }
run full
run no_pack PAD=1
run no_maskonly MASK_ONLY=0
run old_gelu NEW_GELU=0
run no_maxtree MAX_TREE=0
run no_expfree EXP_FREE=0
run no_ls LS=1
run fwd_only FWD_ONLY=1
run all_off PAD=1 MASK_ONLY=0 NEW_GELU=0 MAX_TREE=0 EXP_FREE=0 LS=1
echo DONE > ablation/done
