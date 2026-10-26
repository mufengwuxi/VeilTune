#!/bin/sh
cd /workspace/veiltune
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for k in 1 16 256; do for m in floor8 floor10; do
  python emu_sigma/emu_sigma_check.py --f 12 --lut-mode $m --sigma-primal true --rsqrt sigma --tan-scale $k > emu_sigma/run_f12_${m}_full_k$k.log 2>&1
done; done
echo ALLDONE > emu_sigma/run_kappa.done
