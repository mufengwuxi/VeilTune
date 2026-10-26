cd /workspace/veiltune; export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
while [ ! -f sec_train_adam/done_sst2 ]; do sleep 30; done
bash stage2/run_ablation_redo.sh
echo READY > sec_train_adam/after_ready
