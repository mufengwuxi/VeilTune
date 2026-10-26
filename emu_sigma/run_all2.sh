#!/bin/sh
cd /workspace/veiltune
for f in 12 16; do for m in floor8 floor10; do
  python emu_sigma/emu_sigma_check.py --f $f --lut-mode $m > emu_sigma/run_f${f}_${m}.log 2>&1
  python emu_sigma/emu_sigma_check.py --f $f --sigma-primal true --lut-mode $m > emu_sigma/run_f${f}_primal_${m}.log 2>&1
done; done
echo ALLDONE > emu_sigma/run_all2.done
