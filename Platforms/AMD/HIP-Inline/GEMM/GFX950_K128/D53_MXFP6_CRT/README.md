# D53_MXFP6_CRT: FP64 GEMM (8192³) on gfx950 MXFP6 Matrix Cores

See [the platform page](../../../../README.md) for the result, build and limits. This page maps the files.

| File | Role |
|---|---|
| `plan.h` | `Dgemm8192`: allocation, the 4 phases of a call, event timings |
| `crt.h` | row/column shifts; CRT reconstruction in sum form and the final rounding |
| `encode.h` | encoder kernel: FP64 -> residues modulo 19 moduli -> FP6 product planes in the kernel layout |
| `quantize.h` | x 2^shift rounded to the nearest integer (ties to even) as a 96-bit integer; residue modulo p with 32-bit arithmetic |
| `layout.h` | AITER operand layout, segment caps from the exact-accumulator bound, the 2 lookup tables |
| `aiter_module.h` | loads the code object and fills its 448-byte argument block |
| `table_d53_k8192.h` | generated: moduli, product weights, FP6 code tables, CRT constants |
| `wide_integer.h`, `word_round.h` | multi-word integers and rounding to binary64 |
| `AITER/` | kernel assembly and metadata ([THIRD_PARTY.md](THIRD_PARTY.md)) |
| `Tests/host_exact_test.cpp` | CPU: table bounds, plane identity of every form, small shapes end to end |
| `Tests/gpu_check.hip` | GPU: sampled exact outputs, accuracy criterion against rocBLAS, full-output bound |
| `Tools/bench.hip`, `Tools/common.h` | timing against rocBLAS and hipBLASLt; shared host helpers |

One call runs, on one stream:

1. `shifts_a_partial` + `shifts_finish` (A rows) and `shifts_b` (B columns): shift = 52 - exponent of the
   largest magnitude of the line.
2. `encode_kernel<false>` (A) and `encode_kernel<true>` (B): 256 x 64 blocks of 32 rows x 128 K; each
   block writes its tile into all 29 product planes.
3. 44 launches of `f6gemm_awe_modular_v1`, one per (product, K segment); the first launch of a modulus
   writes its residue plane, the others add to it.
4. `reconstruct_kernel`: 19 residues -> centred integer -> binary64 times 2^-(shift_A + shift_B).

The accumulator of a launch must be an exact FP32 integer, so products whose FP6 magnitudes are large
run K in 2 segments of 4096 (15 products) and the others in 1 segment of 8192 (14 products);
`layout.h` derives this from the code tables, and the kernel sets an error flag if an accumulator is not
an integer of magnitude below 2^24.
