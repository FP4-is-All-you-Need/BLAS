# GEMM

Exact matrix multiplication on FP4 matrix products. A C API and a small C++ ownership wrapper provide plans, caller-owned workspace and execution statistics. The encodings come from the catalog in [`../../../../AWE/Solutions/`](../../../../AWE/Solutions/). This version issues its FP4 matrix products through cuBLASLt (NVFP4). The CMake project, target and header names are `awe_blas`.

| Path | Backend |
|---|---|
| `src/awe_blas.cu`, `src/ozaki2_backend.cu`, `ozaki2/` | INT8 → INT64 or INT32: input-aware AWE, radix-13 limbs, and a modular backend (residue system, `AWE_METHOD_OZAKI2`). Built by CMake below. |
| `src/fp64/`, `scripts/build_fp64.sh` | FP64: residue construction of Ozaki scheme II on a 24-, 53- or 55-bit integer image (`AWE_FP64_BITS`). Builds `lib/libawe_f64.so`, `bin/fp64_smoke`, `bin/fp64_acc_check`. |
| `src/complex/`, `scripts/build_complex.sh` | Complex FP64: 2M decomposition on the FP64 path, C interface (`create`, `run`, `error` in `src/complex/complex.cu`). Builds `lib/libcomplex.so`. |

## INT8

Inputs cover **[-128,127]**. The default output is INT64. INT32 is selectable for **K <= 131071**. Both matrix layouts, both transpose choices, padded leading dimensions and partial internal blocks are supported. Only alpha=1 and beta=0 are implemented. The FP64 entry point returns `AWE_NOT_SUPPORTED`.

## Build and Run the Examples

Requires CMake 3.24 or newer, a C++17 compiler, a CUDA 13.2 or newer development toolkit, and a GPU/cuBLASLt combination supporting NVFP4. Put the intended `nvcc` on PATH or select it with `CMAKE_CUDA_COMPILER`. B300 is compute capability 10.3 (`103a`); RTX-class Blackwell GPUs are `120`. Successful compilation alone does not validate another GPU.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=103a
cmake --build build -j
./build/awe_example_c
./build/awe_example_cpp
```

Each example allocates device storage, creates a plan, computes a small matrix product, checks every output and releases resources. Both print PASS and exit with zero on success. The examples and `awe_smoke` execute GPU work. Use an exclusive GPU allocation when measuring performance.

CMake options `AWE_BUILD_EXAMPLES`, `AWE_BUILD_TESTS`, `AWE_BUILD_BENCHMARKS` and `AWE_ENABLE_OZAKI2` default to ON. The optional GEMMul8 comparison requires a separately built source tree specified by `AWE_GEMMUL8_ROOT`; GEMMul8 is not bundled.

## Install and Use from Another Project

```sh
cmake --install build --prefix "$PWD/install"
cmake -S examples -B consumer-build -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build consumer-build -j
./consumer-build/awe_example_c
./consumer-build/awe_example_cpp
```

An application uses `find_package(awe_blas CONFIG REQUIRED)` and links `awe::awe_blas`. CUDA runtime headers and linkage are provided by the exported target. C callers do not need to compile their source with nvcc. Headers and ownership rules are in the API reference.

## Documentation

- [API reference](docs/API.md): every function, argument group, enumeration, input contract, workspace rules, guaranteed K and payout period.
- Complete [C example](examples/minimal.c) and [C++ example](examples/minimal.cpp).

## Timing

Execution includes input inspection, encoding, all GEMMs and integer reconstruction. Plan creation, allocation and transfers are outside the execution statistic. Three streams permit overlap where the selected execution path has independent work. The default fused AWE path completes all encoding before GEMM and reconstructs the whole output afterward. Stage times must not be summed to estimate total time.

`awe_smoke` checks logical outputs against CPU INT64 reference products, including layouts, transposes, input extremes, output padding and internal period boundaries. `awe_bench_i8` checks every output element against cuBLASLt INT8 on larger shapes and reports ordinary and profiled timings separately. Report the GPU, the CUDA and cuBLASLt versions, the shape, the output type and the clock state next to every timing.

## License

MIT; see [LICENSE](LICENSE).
