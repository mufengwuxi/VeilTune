#!/bin/sh
cd /workspace/veiltune
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for cfg in "12 floor8 false" "16 floor8 false" "16 floor10 false" "16 floor10 true" "16 floor8 true"; do
  set -- $cfg
  python emu_sigma/emu_sigma_check.py --f $1 --lut-mode $2 --sigma-primal $3 --rsqrt sigma > emu_sigma/run_f$1_$2_primal$3_rsqrtsigma.log 2>&1
done
python emu_sigma/emu_sigma_check.py --f 16 --tangent-lut false --rsqrt sigma > emu_sigma/run_f16_analytic_rsqrtsigma.log 2>&1
echo ALLDONE > emu_sigma/run_rsqrt.done
