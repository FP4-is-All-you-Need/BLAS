#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
nvcc=${NVCC:-nvcc}
arch=${CUDA_ARCH:-103a}
mkdir -p "$root/lib"
flags=(-std=c++20 -O3 --cudart shared -gencode "arch=compute_${arch},code=sm_${arch}" -Xcompiler=-fPIC -shared)
"$nvcc" "${flags[@]}" "$root/src/complex/complex.cu" -lcublasLt -o "$root/lib/libcomplex.so"
"$nvcc" "${flags[@]}" "$root/src/complex/harness.cu" -o "$root/lib/libharness.so"
"$nvcc" "${flags[@]}" "$root/src/complex/emulation.cu" -lcublas -o "$root/lib/libemulation.so"
