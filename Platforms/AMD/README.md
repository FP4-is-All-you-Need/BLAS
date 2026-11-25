# AMD Instinct MI350X: FP64 GEMM on MXFP6 Matrix Cores

The MI350X runs MXFP6 at the same rate as MXFP4 ([9.2 PFLOP/s each](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)),
so this engine multiplies FP6 (E2M3) values instead of FP4 ones.

One engine: `HIP-Inline/GEMM/GFX950_K128/D53_MXFP6_CRT/` computes C = A B in FP64 (8192 x 8192 x
8192, column-major, alpha = 1, beta = 0) from exact integer products on the MXFP6 (E2M3) Matrix Cores
of gfx950. Each input row of A and column of B is scaled to a 53-bit integer grid, encoded as residues
modulo 19 moduli (30 FP6 product planes, [table](HIP-Inline/GEMM/GFX950_K128/D53_MXFP6_CRT/table_d53_k8192.h)),
multiplied in one MXFP6 launch that holds partial residues in registers and writes each modulus once,
and reconstructed with the Chinese remainder theorem (CRT) and one rounding to binary64. The output equals the exactly rounded
dot product of the quantized inputs.

The 19 moduli: 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 47 and 49 need 1 FP6 product each, 121 needs
2, and 1223, 1231, 1249, 1279 and 1283 need 3. The three-product moduli are chosen so that every FP6
value of their products has magnitude 36 or less (E2M3 values times 8), which keeps K = 8192 in one
K segment (see [the engine page](HIP-Inline/GEMM/GFX950_K128/D53_MXFP6_CRT/README.md)).

## Build and run

Requires ROCm 7.2 (hipcc, the ROCm clang assembler, rocBLAS; hipBLASLt for the benchmark) and
CMake 3.31.

```sh
cmake -S . -B Build-MI350X -DAWE_BACKEND=MI350X
cmake --build Build-MI350X --parallel 2
ctest --test-dir Build-MI350X --output-on-failure          # CPU test of table, residues, CRT, rounding
Build-MI350X/Bin/awe_mxfp6_check --full                    # GPU: exact samples, rocBLAS bound, inputs unchanged
Build-MI350X/Bin/awe_mxfp6_bench --hipblaslt               # GPU: timings against rocBLAS and hipBLASLt
```

With CMake 3.21 to 3.30, configure the engine directory on its own instead of the first line:
`cmake -S Platforms/AMD/HIP-Inline/GEMM/GFX950_K128/D53_MXFP6_CRT -B Build-MI350X`. It gives the same
targets and output paths.

The kernel code object `Build-MI350X/Lib/awe_mxfp6_modular.co` is assembled from
`AITER/modular.s` ([origin and changes](HIP-Inline/GEMM/GFX950_K128/D53_MXFP6_CRT/THIRD_PARTY.md)).
`-DAWE_ENABLE_GPU_TESTS=ON` registers the GPU check with CTest. From C++, construct
`awe_mxfp6::Dgemm8192 plan(stream, module_path)` and call `plan.run(a, b, c, lda, ldb, ldc)`
([plan.h](HIP-Inline/GEMM/GFX950_K128/D53_MXFP6_CRT/plan.h)).

## Measured result

<a href="../../Figures/amd-mi350x-dgemm-times.png"><img src="../../Figures/amd-mi350x-dgemm-times.png" alt="MI350X DGEMM e2e times from device-resident FP64 inputs to synchronized FP64 output at 8192 cubed. AWE MXFP6 took 18.10 ms (shift 0.27, encode 4.00, MXFP6 products and residue updates 12.48, CRT and synchronization 1.35 ms). rocBLAS took 20.64 ms. GEMMul8 INT8 took 12.12 ms. Sampled ULP errors (p95, maximum) were AWE MXFP6 (0, 0), rocBLAS (151, 214,787), GEMMul8 INT8 (0, 0)." width="700"></a>

The e2e time covers the call from device-resident FP64 inputs to synchronized FP64 output.
The figure shows medians for 8192³ with 2 s between calls. The same MI350X and inputs
were used for all methods. GEMMul8 was built separately and timed in the same runs as the other libraries.

| Calls | Methods timed together | AWE (ms) | rocBLAS (ms) | GEMMul8 INT8 (ms) |
|:---|:---|---:|---:|---:|
| 2 s pause | AWE, rocBLAS | 18.08 | 20.64 | — |
| back to back | AWE, rocBLAS | 16.70 | 18.99 | — |
| 2 s pause | all | 18.10 | 20.64 | 12.12 |
| back to back | all | 16.80 | 18.93 | 10.46 |

The all-method runs included hipBLASLt FP64 DGEMM. With a 2 s pause it took 19.10 ms. Its sampled p95 was 159 ULP against 151 for rocBLAS, so it did not qualify for the figure.

<details><summary>Conditions and checks</summary>

One MI350X (gfx950), ROCm 7.2.0, rocBLAS 5.2.0, hipBLASLt 1.2 and GEMMul8 commit 5f3fae8
(INT8, 15 moduli, fast mode). Uniform FP64 inputs on a 53-bit grid in [−1, 1), seed 20261021,
M = N = K = 8192, α = 1, β = 0. The measurement driver extends `awe_mxfp6_bench` with a separately
built GEMMul8, an optional pause and the reverse starting order. It synchronizes the stream before
and after each call. Each run rotates method order through three warm-ups and eleven timed rounds.
The figure and table pool 22 calls per method across the two opposite orders. The four AWE stages
come from the two central calls after sorting AWE e2e times. The last stage includes reconstruction and
host synchronization.

Using the correctly rounded dot products of the original inputs as the reference for 1,024 sampled outputs,
a method passes if both it and rocBLAS have finite maximum errors and its 95th-percentile and maximum
errors in units of the last place (ULP) are each no greater than the corresponding rocBLAS value. The exactness check
compared all 67,108,864 AWE outputs to the exactly rounded dot of the quantized inputs for each of
the uniform, normal and wide-exponent distributions. `awe_mxfp6_check --full` also checks every
uniform output against the quantization and rounding bound.

</details>

## Limits

> [!WARNING]
> Only M = N = K = 8192, FP64, column-major, no transpose, alpha = 1, beta = 0.

> [!NOTE]
> - About 5.7 GB of device memory per plan, in addition to A, B and C.
> - Inputs with NaN or infinity are rejected (the call throws).
> - A value 2^-e times the largest of its row of A or column of B keeps about 53 - e significant bits.
> - Uniform and normal inputs met the sampled ULP criterion. With wide-exponent inputs the
>   1,024 sampled errors against the original dot products had p95 248,975 and maximum 42,995,406 ULP
>   for AWE, versus 70 and 4,166 for rocBLAS. All 67,108,864 AWE outputs still matched the
>   exactly rounded dot products of the quantized inputs.
> - Separate builds with the 29-product table at K = 4096 ran back to back. 4096 × 4096 × 4096 took 3.04 ms against 2.44 ms for rocBLAS. 8192 × 8192 × 4096 took 9.95 ms against 9.41 ms for rocBLAS.

<details><summary>Details</summary>

- The operand layout, the segment schedule and the kernel grid are written for the 8192 shape. The
  modulus set is exact for K <= 8192 at 53 bits.
- The plan allocates 5.67 GB, including 2.55 GB for the residue planes.
- Each row of A and column of B shares one power-of-two scale.
- The MXFP6 kernel uses four wavefronts per 256 × 256 output tile, three stages in the local data share and a fixed E8M0 scale of 130.

</details>
