#!/bin/sh
cd /workspace/gpu-mpc-src/GPU-MPC
export CUDA_VERSION=12.8 GPU_ARCH=120
echo "START $(date)"
make sigma; echo "SIGMA_EXIT=$? $(date)"
make tests/fss/dpf_lut 2>/dev/null || make dpf_lut; echo "DPFLUT_EXIT=$? $(date)"
