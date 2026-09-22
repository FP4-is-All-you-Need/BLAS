#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
nvcc=${NVCC:-nvcc}
arch=${CUDA_ARCH:-103a}
mkdir -p "$root/bin" "$root/lib"
flags=(-std=c++20 -O3 --cudart shared -gencode "arch=compute_${arch},code=sm_${arch}")
"$nvcc" "${flags[@]}" -I"$root/include" -DAWE_HAS_OZAKI2=1 -Xcompiler=-fPIC -shared "$root/src/awe_blas.cu" "$root/src/fp64/backend.cu" -lcublasLt -o "$root/lib/libawe_f64.so"
"$nvcc" "${flags[@]}" -I"$root/include" "$root/tests/fp64_smoke.cu" -L"$root/lib" -lawe_f64 -o "$root/bin/fp64_smoke"
"$nvcc" "${flags[@]}" -I"$root/include" "$root/tests/fp64_acc_check.cu" -L"$root/lib" -lawe_f64 -o "$root/bin/fp64_acc_check"
