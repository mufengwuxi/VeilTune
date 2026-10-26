#!/bin/sh
cd /workspace/veiltune
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for mb in 10 12; do
  python emu_sigma/emu_sigma_check.py --f 16 --lut-mode floor10 --rsqrt sigma --rsqrt-mbits $mb > emu_sigma/run_f16_floor10_rsqrtm$mb.log 2>&1
  python emu_sigma/emu_sigma_check.py --f 16 --lut-mode floor10 --sigma-primal true --rsqrt sigma --rsqrt-mbits $mb > emu_sigma/run_f16_floor10_primal_rsqrtm$mb.log 2>&1
done
echo ALLDONE > emu_sigma/run_mbits.done
