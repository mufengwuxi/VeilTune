#!/bin/sh
# usage: run_sigma.sh <model> <seq> [pool_gb] [threads]
M=$1; N=$2; GB=${3:-12}; T=${4:-12}
W=/workspace/fss_runs/work; BIN=${SIGMA_BIN:-/workspace/fss_build/sigma_lowmem}
L=/workspace/fss_runs/logs; mkdir -p $L
cd $W
export SIGMA_POOL_GB=$GB
timeout 1200 $BIN $M $N 0 127.0.0.1 $T > $L/${M}_${N}_p0.log 2>&1 &
P0=$!
sleep 1
timeout 1200 $BIN $M $N 1 127.0.0.1 $T > $L/${M}_${N}_p1.log 2>&1
E1=$?
wait $P0; E0=$?
echo "EXIT p0=$E0 p1=$E1"
for p in 0 1; do echo "--- P$p dealer"; cat output/P$p/models/$M-$N/dealer.txt; echo "--- P$p evaluator"; cat output/P$p/models/$M-$N/evaluator.txt; done
