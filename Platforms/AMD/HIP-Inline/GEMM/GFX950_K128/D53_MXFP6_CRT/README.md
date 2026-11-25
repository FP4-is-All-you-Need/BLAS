# D53_MXFP6_CRT: FP64 GEMM (8192³) on gfx950 MXFP6 Matrix Cores

See [the platform page](../../../../README.md) for the result, build and limits. This page maps the files.

| File | Role |
|---|---|
| `plan.h` | `Dgemm8192`: allocation, the 4 phases of a call, event timings |
| `crt.h` | row/column shifts; CRT reconstruction in sum form and the final rounding |
| `encode.h` | encoder kernel: FP64 -> residues modulo 19 moduli -> FP6 product planes in the kernel layout |
| `quantize.h` | x 2^shift rounded to the nearest integer (ties to even) as a 96-bit integer; residue modulo p with 32-bit arithmetic |
| `layout.h` | AITER operand layout, segment caps from the exact-accumulator bound, the 2 lookup tables |
| `aiter_module.h` | loads the code object and packs a device array of 448-byte segment records |
| `table_d53_k8192.h` | generated: moduli, product weights, FP6 code tables, CRT constants |
| `table_d53_k4096.h`, `table_select.h` | generated 29-product table, used when K <= 4096; the choice of table by K |
| `wide_integer.h`, `word_round.h` | multi-word integers and rounding to binary64 |
| `AITER/` | kernel assembly and metadata ([THIRD_PARTY.md](THIRD_PARTY.md)) |
| `Tests/host_exact_test.cpp` | The CPU test checks table bounds, all outputs of small matrix products, and whether each form (the weighted FP6 products for one modulus `p`) sums to `r × s` modulo `p` for every pair of residues `r` and `s`. |
| `Tests/gpu_check.hip` | GPU: sampled exact outputs, accuracy criterion against rocBLAS, full-output bound |
| `Tools/bench.hip`, `Tools/common.h` | timing against rocBLAS and hipBLASLt; shared host helpers |

One call runs, on one stream:

1. `shifts_a_partial` + `shifts_finish` (A rows) and `shifts_b` (B columns): shift = 52 - exponent of the
   largest magnitude of the line.
2. `encode_tile<false>` (A) and `encode_tile<true>` (B): 16 rows or columns by 64 K elements per
   block. Each block quantizes its input tile once and writes all 30 product planes.
3. One launch of `f6gemm_awe_modular_v2`: each workgroup holds the partial residues for its
   256 × 256 output tile in registers and stores them once per modulus.
4. `reconstruct_kernel`: 19 residues -> centred integer -> binary64 times 2^-(shift_A + shift_B).

Each segment must accumulate exact FP32 integers. Of the 30 products, 29 span K = 8192 and the
second product of modulus 121 uses two segments of 4096. `layout.h` derives these bounds from the
code tables. The kernel flags any noninteger accumulator or magnitude of 2²⁴ or greater.
