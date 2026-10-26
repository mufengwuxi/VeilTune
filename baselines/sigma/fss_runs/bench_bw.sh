#!/bin/sh
cd /workspace/fss_runs
export SIGMA_BIN=/workspace/fss_build/sigma_lowmem_f SIGMA_SCALE=12
for ex in 0 8 13; do for n in 16 32; do for r in 1 2 3; do
  rm -rf work/output/P0/models/bert-mini-$n work/output/P1/models/bert-mini-$n
  echo "### bwextra=$ex seq=$n rep=$r"
  SIGMA_BW_EXTRA=$ex ./run_sigma.sh bert-mini $n 10 12 | grep -E "EXIT|Total time|Key size|Total Comm"
done; done; done
echo BENCHBW_DONE
