#!/bin/bash
# ag0 (SAP s42) crashed on a cache-build race; rerun it after rl_leak8_ag finishes.
cd /workspace/veiltune; while [ ! -f rl_leak8/done_ag ]; do sleep 60; done
export HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT=16 OUTD=rl_leak8 TASK=agnews
mv rl_leak8/logs/ag0.log rl_leak8/logs/ag0_crash.log
python rl_leak.py --scenario s2 --Ks 1 100 --train fo --lr 1e-4 --reveal h --seed 42 > rl_leak8/logs/ag0.log 2>&1
echo DONE > rl_leak8/done_ag_fix
