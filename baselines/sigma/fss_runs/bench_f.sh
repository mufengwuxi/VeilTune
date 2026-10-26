#!/bin/sh
cd /workspace/fss_runs
export SIGMA_BIN=/workspace/fss_build/sigma_lowmem_f
for cfg in "12 0" "16 8" "16 4"; do set -- $cfg
  for n in 16 32; do for r in 1 2 3; do
    echo "### scale=$1 bwextra=$2 seq=$n rep=$r"
    SIGMA_SCALE=$1 SIGMA_BW_EXTRA=$2 ./run_sigma.sh bert-mini $n 10 12 | grep -E "EXIT|Total time|Key size|Total Comm|Gelu|Softmax|Layernorm|MHA|Matmul|Truncate|Comm time|Transfer"
    grep -h "scale=" logs/bert-mini_${n}_p0.log
  done; done
done
echo BENCHF_DONE
