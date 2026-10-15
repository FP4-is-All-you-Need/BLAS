# C interface

The public header is [Include/AWE/awe.h](../Include/AWE/awe.h). Start with the
[README's supported operations and common forms](../README.md#supported-operations)
and [minimal call](../README.md#build-and-minimal-use).

## Arrays, calls and errors

Initialize `awe_descriptor`, set positive M/N/K and all 3 leading dimensions,
then validate or create a plan. M and N are capped at 65536 to bound host tile
arithmetic. Leading dimensions count elements. 0-size operations are rejected.
All arrays are column-major device allocations on the plan's active device.
Execution completes before returning. The API neither owns caller arrays nor
accepts an external CUDA stream; callers must complete input writes first.
Concurrent execute calls are serialized per engine module. Calling the same module
from multiple threads does not increase throughput. Destruction must not
race any use of that plan, and must occur on its creation device. Direct calls to
legacy engine exports must not be mixed with frontend calls.

`awe_complex_float` stores 2 adjacent float components. `awe_i8gemm` takes
`int8_t` inputs and writes `int32_t` outputs. Each routine has a typed `awe_d*`,
`awe_z*`, `awe_s*`, `awe_cgemm` or `awe_i8gemm` entry point in the header.
Unavailable backend, shape, width or engine choices return `AWE_NOT_SUPPORTED`.
`awe_last_error()` supplies the diagnostic. Read it before another API call.

| Operation | Shape and output details beyond the common forms |
|---|---|
| GEMM | A(M,K) B(K,N); padded leading dimensions allowed |
| SYMM / HEMM | Square; reconstruct A from lower triangle; HEMM ignores imaginary diagonal |
| SYRK / HERK | A(N,K); B absent; lower C=A A^T or A A^H; upper C retained |
| SYR2K / HER2K | A(N,K) B(N,K); lower C=A B^T+B A^T or A B^H+B A^H; upper C retained |
| TRMM | Square; C=tril(A) B; separate output; strict upper A ignored |
| DTRSM | K64 only; C=A^{-1} B; K=M, M<=4096, N<=2048; padded leading dimensions allowed |

Except for GEMM and DTRSM, matrices are contiguous: M=N, lda=ldc=N, ldb=K,
except rank-2k where ldb=N. TRMM, SYMM and HEMM also require K=N.
Set ldb=K even for rank-k descriptors. Complex alpha/beta must be exactly 1+0i
and 0+0i. Unsupported forms return an error.

C must not overlap either input, including padding. Inputs are read-only.
Used elements must satisfy the finite-input quantization contract. Only unused
upper-triangular input may contain NaNs. An error after execution starts does
not promise unchanged output. HERK/HER2K produce a real diagonal.

DTRSM requires lda, ldb and ldc >= M, a nonzero diagonal and finite inputs.
B is retained. Creation can reject workspace above 16 GiB or 75% of free GPU
memory. The 57-bit update uses 128-row blocks; the 53-bit update uses 2048-row
blocks. Diagonal blocks use cuBLAS FP64, updates use the selected K64 GEMM and
FP64 subtraction. A solve contained in 1 diagonal block does not call GEMM;
effective GEMM fields then report 0 or -1. Integer coverage does not promise
accuracy of arbitrary solves; see [tested cases](validation.md#results-and-known-failures).

The frontend checks the active device before loading an engine. This build has
K96 sm_103a and K64 sm_100a images. A K64 source guard accepting capability 10.3
does not make its sm_100a cubin portable to B300.

## Options and ABI compatibility

Initialize `awe_options` with `awe_options_init`, then pass it to
`awe_plan_create_with_options`. `struct_size` records the supplied layout.
On the recorded 64-bit host it is 48 bytes. A 32-byte input uses the default
width and engine; a 40-byte input keeps its width and uses no engine name.
Recompile callers before using the new initializer, which writes the full size.
Descriptor layout, previous enum values and effective-options layout are unchanged.

`quantize_bits=0` uses the default width. Otherwise it must be supported for the
routine and geometry. `engine_name=NULL` selects the default source. A non-null
name is read during create only; the plan does not retain the caller's pointer.
The name, routine, backend, shape, width and modulus set are checked before loading.
An explicit nonzero width must agree with the name.

Names state type, integer width, modulus count and use: `s23_m11_level3` means
single precision with 23-bit integers and 11 moduli. Suffixes distinguish K tables
or variants. Inputs never select an engine or precision automatically.
The public engine-name table below is the complete list of accepted choices.

Pacing, modulus chunks, item order, MINB, pingpong and encoder policy accept only
`AWE_ENGINE_DEFAULT`. Other numeric values are rejected before device access.
Use `AWE_ENGINE_DEFAULT` for these fields and inspect the created plan with
`awe_plan_effective_options`. It applies the recorded measurement settings where
available; the table lists the retained settings for the other engines.

| Engine / operation | Pingpong | Basis |
|---|---:|---|
| K96/K64 DGEMM d17/d18 | 1 | DGEMM runner / K64 calibration |
| K96 DSYRK d18/d21 | 1 | Measured SYRK adapter runner |
| K96 other D/Z/C imports and K64 d21 | 0 | Earlier runner settings were not recovered; retain 0 |
| K96 S imports | 0 | Retained adapter settings |
| K64 S and complex imports | 1 | Measurement runners; table compatibility checked |

Every public plan disables poisoning. INT8 has no real-engine pingpong option;
its effective settings report 0 moduli and 1 chunk, with graph replay and input
reuse disabled. [Measured defaults](Provenance/measured_defaults.json) retain
source evidence.

`awe_plan_effective_options` and the JSON returned by `awe_plan_info` report
actual created-plan settings: pace distance (-1 disabled), chunk counts, CRT
chunks, item order, MINB (-1 inapplicable, 0 encoder default), pingpong,
B-padding suppression, PDL, shift form, extra encoding work, complex encoder
selection (-1 for real) and modulus count. Info also reports poison, public
`engine_name` and library `engine` separately.

The frontend ignores `L8_*`, `ZL8_ENC`, `ZL8_HERK` and `AWE_FP64_BITS` configuration
variables. It does not change process environment variables. Legacy
`d/z_create_ld` and `d/z_create_operation` retain their environment behavior;
benchmark-only variables are not API options.

## Default selection

The engine-name table gives each choice's exact K bound. This table states the
selection rule; a default that requires an explicit name returns
`AWE_NOT_SUPPORTED`. A width alone does not bypass that requirement.

| Backend | Routine | Default rule |
|---|---|---|
| K96 | DGEMM | 53 bits: 17 moduli through K=4096, then 18 through K=8254; larger K rejected. Explicit 51/52 bits use the CRT revision. |
| K64 | DGEMM | 53 bits: 17 moduli through K=8192; larger AUTO rejected. Select 21 moduli explicitly above 8192. |
| K96 | DSYRK / DTRMM | 53 bits: 18 moduli through K=8254, then 21 |
| K64 | D Level-3 products | 53 bits: 18 moduli through K=8192, then 21 |
| K96 | DSYMM / DSYR2K | General 53-bit engine |
| K96 | ZGEMM / ZSYMM / ZHEMM / ZSYR2K / ZHER2K / ZTRMM | 53 bits |
| K96 | ZSYRK | 53 bits through K=4096; 52 above |
| K96 | ZHERK | 51 bits; the K4096 source for small K |
| K64 | Z routines | 53 bits |
| K96 | SGEMM / SSYMM | 23 / 22 bits |
| K96 | SSYRK | 23 bits through K=4096 (variant a); 24 above |
| K96 | SSYR2K | 21 bits |
| K96 | STRMM | 23 bits through K=4096; 22 above |
| K64 | SGEMM / SSYMM | 23 / 22 bits; engine name required at every K, including K>4096 |
| K64 | SSYRK | 23 bits through K=4096; above 4096, engine name required (24-bit source) |
| K64 | SSYR2K | 23 bits through K=4096; above 4096, engine name required (21-bit source) |
| K64 | STRMM | 23 bits through K=4096; above 4096, engine name required (22-bit source) |
| K96 | CGEMM | 22 bits |
| K64 | CGEMM | 24 bits (`c24_m11_gemm`); 20/22/23 bits require an explicit engine name |
| K64 | DTRSM | 57 bits, requiring an explicit name; 53 is an explicit width choice |
| K96/K64 | INT8 GEMM | 8 bits; only AUTO moduli |

At a fixed width, the normal-input source is used where a matching measurement
exists. K96 small SSYRK variant b is the recorded uniform choice. Small SSYR2K
keeps 21 bits without a matching normal-input 4096 measurement. These choices
are not guarantees for every input. [Measurements by engine](performance.md#measurements-by-engine)
record distributions, timings and missing binary links.

K64's `d53_m17_gemm` uses the calibrated 17-modulus source identified in
[the import record](Provenance/engines.json). [Calibration](Provenance/k64_calibration.json)
covered the recorded shapes without an intermediate-K sweep.
[Host table checks](Provenance/table_bounds.json) limit K64's 18-modulus source
to K=8192: at 8193, the bias used to convert an accumulator to an integer exceeds
the table's safe range.

## All public engine names

K ranges are selection limits, not measured shapes. "Name required" marks sources
that require `engine_name`; NULL or a width alone returns `AWE_NOT_SUPPORTED`
for those choices. [Validation](validation.md) lists their GPU test results.
The [reconstruction guide](representations.md#reconstruction-and-accumulation-bounds)
distinguishes integer coverage from floating-point input accuracy.

### K96 names

| Name | Routines | K | Bits / moduli | Name required |
|---|---|---|---:|---|
| `d53_m17_gemm` | DGEMM | 1–8192 | 53 / 17 | no |
| `d53_m18_gemm` | DGEMM | 1–8254 | 53 / 18 | no |
| `d53_m18_syrk` | DSYRK | 1–8254 | 53 / 18 | no |
| `d53_m21_syrk` | DSYRK | 1–16384 | 53 / 21 | no |
| `d53_m18_syrk_trmm` | DTRMM | 1–8254 | 53 / 18 | no |
| `d53_m21_syrk_trmm` | DTRMM | 1–16384 | 53 / 21 | no |
| `z53_m20_syrk_trmm` | ZTRMM, ZSYRK | 1–16384 | 53 / 20 | no |
| `z51_m19_herk` | ZHERK | 1–16384 | 51 / 19 | no |
| `z53_m20_level3` | ZGEMM, ZHEMM, ZHER2K, ZSYMM, ZSYR2K | 1–16384 | 53 / 20 | no |
| `z52_m19_syrk` | ZSYRK | 1–8958 | 52 / 19 | no |
| `z50_m18_syrk` | ZSYRK | 1–15925 | 50 / 18 | no |
| `z49_m19_herk` | ZHERK | 1–16384 | 49 / 19 | no |
| `z48_m18_level3` | ZGEMM | 1–15187 | 48 / 18 | no |
| `d53_m18_level3` | DGEMM, DSYMM, DSYR2K | 1–8254 | 53 / 18 | no |
| `d51_m17_level3` | DGEMM | 1–8720 | 51 / 17 | no |
| `d52_m17_level3` | DGEMM | 1–8560 | 52 / 17 | no |
| `s20_m10_level3_fused` | SSYMM | 1–8192 | 20 / 10 | no |
| `s23_m12_level3_fused` | SSYRK | 1–8192 | 23 / 12 | no |
| `s24_m12_level3_fused` | SSYRK | 1–8192 | 24 / 12 | no |
| `s22_m11_level3_fused` | STRMM | 1–8192 | 22 / 11 | no |
| `s22_m11_level3` | SSYMM | 1–8192 | 22 / 11 | no |
| `s21_m10_level3` | SSYR2K | 1–8192 | 21 / 10 | no |
| `s23_m9_gemm` | SGEMM | 1–8192 | 23 / 9 | no |
| `s21_m10_gemm` | SGEMM | 1–8192 | 21 / 10 | no |
| `s20_m10_syr2k` | SSYR2K | 1–8192 | 20 / 10 | no |
| `c22_m10_gemm` | CGEMM | 1–16384 | 22 / 10 | no |
| `c20_m9_gemm` | CGEMM | 1–16384 | 20 / 9 | no |
| `i8_gemm` | I8GEMM | 1–16384 | 8 / 0 | no |
| `s21_m10_level3_k4096` | SSYMM, STRMM | 1–4096 | 21 / 10 | no |
| `s22_m11_syr2k_k4096` | SSYR2K | 1–4096 | 22 / 11 | no |
| `s23_m11_syrk_k4096_a` | SSYRK | 1–4096 | 23 / 11 | no |
| `s23_m11_syrk_k4096_b` | SSYRK | 1–4096 | 23 / 11 | no |
| `s22_m11_symm_k4096` | SSYMM | 1–4096 | 22 / 11 | no |
| `s23_m11_trmm_k4096` | STRMM | 1–4096 | 23 / 11 | no |
| `s22_m8_gemm_k4096` | SGEMM | 1–4096 | 22 / 8 | no |
| `c23_m10_gemm` | CGEMM | 1–4096 | 23 / 10 | no |
| `z51_m19_herk_k4096` | ZHERK | 1–4096 | 51 / 19 | no |
| `z53_m20_herk` | ZHERK | 1–16384 | 53 / 20 | no |
| `d51_m17_gemm_crt` | DGEMM | 1–8720 | 51 / 17 | no |
| `d52_m17_gemm_crt` | DGEMM | 1–8560 | 52 / 17 | no |
| `d53_m18_gemm_crt` | DGEMM | 1–8254 | 53 / 18 | no |

### K64 names

| Name | Routines | K | Bits / moduli | Name required |
|---|---|---|---:|---|
| `d53_m17_gemm` | DGEMM | 1–8192 | 53 / 17 | no |
| `d53_m18_gemm` | DGEMM, DSYRK, DTRMM, DSYMM, DSYR2K | 1–8192 | 53 / 18 | no |
| `d53_m21_gemm` | DGEMM, DSYRK, DTRMM, DSYMM, DSYR2K | 1–16384 | 53 / 21 | no |
| `z53_m20_level3` | ZGEMM, ZTRMM, ZHERK, ZHEMM, ZHER2K, ZSYMM, ZSYRK, ZSYR2K | 1–16384 | 53 / 20 | no |
| `i8_gemm` | I8GEMM | 1–16384 | 8 / 0 | no |
| `s21_m10_level3` | SGEMM, SSYMM, STRMM | 1–4096 | 21 / 10 | no |
| `s23_m11_level3` | SSYRK, SSYR2K, STRMM | 1–4096 | 23 / 11 | no |
| `s23_m11_trmm_gemm` | STRMM | 1–4096 | 23 / 11 | yes |
| `s20_m10_level3` | SSYMM, SSYR2K | 4097–8192 | 20 / 10 | yes |
| `s21_m10_k8192` | SGEMM, SSYR2K | 4097–8192 | 21 / 10 | yes |
| `s22_m11_k4096` | SSYMM, SSYR2K | 1–4096 | 22 / 11 | yes |
| `s22_m11_k8192` | SSYMM, STRMM | 4097–8192 | 22 / 11 | yes |
| `s23_m12_k8192` | SSYRK | 4097–8192 | 23 / 12 | yes |
| `s24_m12_level3` | SSYRK | 4097–8192 | 24 / 12 | yes |
| `s23_m9_gemm` | SGEMM | 1–8192 | 23 / 9 | yes |
| `c20_m9_gemm` | CGEMM | 1–16384 | 20 / 9 | yes |
| `c22_m10_gemm` | CGEMM | 1–16384 | 22 / 10 | yes |
| `c23_m10_gemm` | CGEMM | 1–4096 | 23 / 10 | yes |
| `c24_m11_gemm` | CGEMM | 1–16384 | 24 / 11 | no |
| `d53_m18_trsm` | DTRSM | 1–4096 | 53 / 18 | no |
| `d57_m20_trsm` | DTRSM | 1–4096 | 57 / 20 | yes |

## Implementation-specific behavior

For odd N or N<=224, fused SSYRK uses 1 partition; other fused shapes retain
2 partitions. The fused path encodes views without temporary float matrices,
with 1 chunk, no pacing and plain encoders with MINB=3. SYR2K adds 2 rounded
float products. DSYR2K forms X=A B^T and writes lower C=X+X^T; complex rank-2k
uses the corresponding transpose or conjugate transpose.

The named K64 `s23_m11_trmm_gemm` path shares host operand preparation with K96;
[validation](validation.md#results-and-known-failures) lists the checked settings and results.

INT8 checks its accumulation bounds per call and prepares shape workspace on the
first call. Its exact integer fallback and bounds are described in
[representations](representations.md#reconstruction-and-accumulation-bounds).
Leading dimensions must fit `int`. Warmup is needed before measuring execute.

## Earlier API compatibility

The library in [Legacy/20260922](../Legacy/20260922/) is outside the current build.
The current API does not replace its INT64 output, row-major/transposed INT8,
caller-owned workspace, C++ plan wrapper, installed CMake package or standalone
GEMMul8 comparison executable. Source-directory changes preserve current numeric
ABI values and library basenames; they do not provide the earlier API.
