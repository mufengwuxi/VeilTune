#!/bin/sh
cd /workspace/veiltune
while [ ! -f emu_sigma/run_lutmodes.done ]; do sleep 10; done
for f in 12 16; do for m in floor8 floor10; do
  python emu_sigma/emu_sigma_check.py --f $f --sigma-primal true --lut-mode $m > emu_sigma/run_f${f}_primal_${m}.log 2>&1
done; done
echo ALLDONE > emu_sigma/run_primal.done
