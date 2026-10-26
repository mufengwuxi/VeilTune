Results produced on a second machine (RTX 4090 48GB) on 2026-10-14, transcribed here
because that machine became unreachable. Code = gpu_crypten_bp.py md5 a156dec8 (no loss scaling), OPS=bp.

t12 time (alone on GPU, no warm-up; b1 includes warm-up):
  b1  time 6.96 s  rounds 780  wire 0.40 GB  N_real 13   T 13  peak 1.4 GB
  b32 time 25.29 s rounds 824  wire 27.57 GB N_real 628  T 49  peak 3.0 GB
  b64 time 41.10 s rounds 824  wire 52.40 GB N_real 1156 T 49  peak 5.1 GB
t12 check (B=4, LS=1): cos 0.998219, rel_l2 0.0597, grad_norm_ref 2.7359, loss_ref 0.46155
t13 profile b64 (OPS=bp): total 30.34 s, 824 rounds, 52.4 GB. Top: gelu cmp 18.83 GB, softmax max 11.41 GB,
  gelu exp mul 4.65, gelu mul 4.19, softmax exp cmp 3.21, gelu inv_1p 2.79, softmax exp mul 2.38, bwd matmul 1.64.
  time: gelu_save 11.77 s, softmax_save 8.08 s, ln_save 1.10 s, ln_bwd 0.28 s, softmax_bwd 0.22 s.
