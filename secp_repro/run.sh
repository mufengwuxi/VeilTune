#!/bin/bash
cd /workspace/veiltune; export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for T in sst2 yelp agnews rte; do for s in 42 43 44; do
  o=secp_repro/${T}_s${s}; [ -f $o/z_final.npy ] && continue
  TASK=$T python secp_repro.py --seed $s --out $o > $o.log 2>&1
done; done
echo DONE > secp_repro/done
