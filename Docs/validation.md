# Correctness checks and results

## Run the checks

Run CPU tests from the repository root:

```sh
python3 -m unittest discover -s Tests
```

Python 3 and NumPy are required. The CPU suite uses fake modules and exact
integer/rational references; it does not load CUDA engines. CTest invokes this
suite as shown in the [README](../README.md#build-and-minimal-use).

The quantized-input and boundary checker runs on the indicated supported GPU:

```sh
python3 Tools/awe_validation.py --gpu --backend 96 --library Build-K96/Lib/libawe.so --output k96.json
python3 Tools/awe_validation.py --gpu --backend 64 --library Build-K64/Lib/libawe.so --output k64.json
python3 Tools/awe_validation.py --compare k64.json k96.json
```

Use `--cudart /path/to/libcudart.so` and `--device N` when needed.
CMake enables GPU tests only with `AWE_ENABLE_GPU_TESTS=ON`.
The boundary suite has no recorded GPU run.

The typed Level-3 driver can list its cases without loading CUDA:

```sh
python3 Tests/GPU/b300.py --group small
python3 Tests/GPU/b200.py --group boundaries
```

`b200.py` selects K64. `--group` also accepts `4096`, `8192` and `boundaries`;
`--routine` restricts the list.

## What the tests mean

The quantized-input reference uses per-row/column binary exponents, nearest-even
integer quantization, unbounded integer dot products, rational scaling and FP64
rounding. Complex components share scales. It checks the quantized contract,
not equality to unquantized cuBLAS output.

The Level-3 small cases use exact dyadic inputs and compare every output with
an independent rational reference. DTRSM uses exact forward substitution and
reports errors; it does not promise exact floating-point solves. Large cases
use original inputs before quantization and sampled unrounded reference values.
ULP uses each real or imaginary reference component's binade. Required targets
compare both p95 and maximum error with native cuBLAS on the same input.
Other width/input pairs are record-only. DTRSM53 is record-only, including small
cases. This metric differs from the rounded reference in the
[earlier performance measurements](performance.md#measurement-figures-and-conditions).

Checks cover unchanged input bytes, allocation canaries, leading-dimension
padding and retained upper output triangles. Unused triangular input is NaN;
HEMM ignores the imaginary diagonal. Reports include settings, library and input
hashes, output hashes, versions, errors and first mismatches. Numeric completion,
full-reference matches, native-goal passes and record-only results are distinct.
A record-only result is not an accuracy qualification. The worker records a failed
case and continues; unstarted cases and timeouts remain distinct.

[Historical reproduction specifications](../Tests/GPU/historical_cases.json)
require the earlier inputs, reference points, metric and native settings.
New seed-193 inputs do not reproduce those measurements. The historical output
hash checks below do not recompute their accuracy metrics.

## Results and known failures

| Run | Numeric executions | Full small reference matches | Native-goal passes outside small matches | Record-only | Default rejection checks | Failures | Timeout / not started |
|---|---:|---:|---:|---:|---:|---:|---|
| B300, 14 October 2026 | 201 | 120 | 38 | 40 | 0 | 9 | 1 / 4 |
| B200, 14 October 2026 | 153 | 90 | 45 | 14 | 4 | 11 | 0 / 0 |
| GB200, 15 October 2026 | 112 | 105 | 3 | 4 | 4 | 0 | 0 / 0 |

Failures with saved numbers count as numeric execution. The original `pass`
field included record-only cases. The 15 October run had 116 standard cases
and 4 separate historical-input hash checks. Full case lists, tested widths,
source trees, compiler/runtime versions and errors are in the
[14 October](Results/GPU_RESULTS_20261014.json) and
[15 October](Results/GPU_RESULTS_20261015.json) records. Raw archives are not
included; those records retain their SHA-256 values.

In the 14 October runs, Inline/Template inputs, settings and output hashes
matched in 6 B300 pairs and 4 B200 pairs; B200 Template d17 at 4096³ was not run.
The 4 B200 forced 18/21-modulus pairs also matched: 16³, 17³ and 4096³ normal/uniform.

### Tested widths

The table is derived from the 14 October case records. The 15 October retry
adds full small-output passes for B200 CGEMM20/22/23/24; its large C24 cases
were record-only. These later cases are listed in the retry JSON.

Small means full correctly rounded output comparison on dyadic 16³ and 17³
inputs (INT8: 16³ and 20³). Large columns include only cases that both enforced
and passed the native p95/max goal on 4096³ inputs. Blank cells make no pass claim.
They must be read with the failures below. A passing small case does not prove
accuracy on other inputs or shapes.

| GPU | Routine | Small bits | 4096 normal bits | 4096 uniform bits |
|---|---|---|---|---|
| B300 | ZGEMM | 48, 53 | 53 | 53 |
| B300 | ZHEMM | 53 | 53 | 53 |
| B300 | ZHERK | 49, 51, 53 | 51 |  |
| B300 | ZHER2K | 53 | 53 | 53 |
| B300 | ZSYMM | 53 | 53 | 53 |
| B300 | ZSYRK | 50, 52, 53 | 53 | 53 |
| B300 | ZSYR2K | 53 |  |  |
| B300 | ZTRMM | 53 | 53 | 53 |
| B300 | DGEMM | 51, 52, 53 | 53 | 51 |
| B300 | DSYMM | 53 | 53 | 53 |
| B300 | DSYRK | 53 | 53 | 53 |
| B300 | DSYR2K | 53 | 53 | 53 |
| B300 | DTRMM | 53 | 53 | 53 |
| B300 | SGEMM | 21, 22, 23 | 23 | 22 |
| B300 | SSYMM | 20, 21, 22 | 22 |  |
| B300 | SSYRK | 23, 24 | 23 | 23 |
| B300 | SSYR2K | 20, 21, 22 |  | 22 |
| B300 | STRMM | 21, 22 |  | 21 |
| B300 | CGEMM | 20, 22, 23 |  | 23 |
| B300 | I8GEMM | 8 | 8 | 8 |
| B200 | ZGEMM | 53 | 53 | 53 |
| B200 | ZHEMM | 53 | 53 | 53 |
| B200 | ZHERK | 53 | 53 | 53 |
| B200 | ZHER2K | 53 | 53 | 53 |
| B200 | ZSYMM | 53 | 53 | 53 |
| B200 | ZSYRK | 53 | 53 | 53 |
| B200 | ZSYR2K | 53 | 53 | 53 |
| B200 | ZTRMM | 53 | 53 | 53 |
| B200 | DGEMM | 53 | 53 | 53 |
| B200 | DSYMM | 53 | 53 | 53 |
| B200 | DSYRK | 53 | 53 | 53 |
| B200 | DSYR2K | 53 | 53 | 53 |
| B200 | DTRMM | 53 | 53 | 53 |
| B200 | SGEMM | 21, 23 | 23 |  |
| B200 | SSYMM | 21, 22 | 22 |  |
| B200 | SSYRK | 23 | 23 | 23 |
| B200 | SSYR2K | 22, 23 | 23 | 22 |
| B200 | STRMM | 21, 23 | 23 | 21 |
| B200 | CGEMM |  |  |  |
| B200 | I8GEMM | 8 | 8 | 8 |

### Failures

| Failure | Current evidence |
|---|---|
| B300 STRMM23 create | Operation 3 was passed to a base GEMM accepting 1. Corrected host code passed CPU and B200 shared-path checks at dyadic 16³/17³. Device code matches the base engine measured on B300 and the support library from the first B300 API run. The corrected B300 frontend has not run again. |
| B200 CGEMM20/22/23 | K96 geometry in the common header caused K64 errors. Corrected versions passed 16³/17³ full checks, but their 4096³ failures were not retested. They remain explicit choices. |
| B300 ZSYRK52 uniform timeout | Only 10.5 s remained; recent cases needed 20.13–22.89 s. No worker evidence was saved; the cause of the timeout is unknown. ZSYRK50 normal/uniform and ZSYR2K53 normal/uniform were not started. |
| 4096³ native-goal failures | B300 SSYMM21 uniform, SSYR2K21 normal, CGEMM22 normal; B200 SGEMM21 uniform and SSYMM21 uniform. All remain failures. |

The failed SSYMM21 uniform inputs/outputs match across GPUs. B200 SGEMM21's
failed output matches a B300 record-only case. K96 SSYR2K normal missed the
maximum-error target at 20, 21 and 22 bits on the 4096³ input; no replacement
passed. Its 21-bit choice came from an 8192³ measurement. Wider record-only
candidates leave the defaults unchanged.

C24's Arm build passed the measured-library device comparison and host-table
checks. It passed full small checks and matched all 4 earlier normal/uniform
4096³/8192³ output hashes on 3 calls each. The earlier helper and input hashes
were checked. These large hash checks are not independent CPU recomputations.
The retry selected C24 explicitly; CPU tests checked the later default switch.
No device code changed in that switch. K<=16384 is a bound, not an exhaustive
GPU test range. [C24 evidence](Provenance/k64_cgemm_measured.json) identifies the build.

The shared STRMM cases require `operation=3`, `create_operation=1` and
`native_adapter=false`. Their unused upper input triangle contains NaNs.
The usual K64 native-adapter path also passed its listed small cases.
No old/new performance regression, full boundary-K batch or concurrent-thread
GPU test is recorded. DTRSM57 passed the native goal on full-output rational
checks at 16×16, 257×4 and 2049×2, using fixed sparse triangular inputs. The first runs used short CPU-wall timers;
the retry collected no performance timings. Earlier speed comparisons were not repeated.

## Common inputs across GPUs

The 14 October B300 records and 15 October B200 records have 48 common small-input
cases; all 48 AWE output hashes match. The 6 CGEMM keys in this set are
C20/C22/C23 at 16³ and 17³; the first B200 run saved no hashes for them.
The 6 earlier CGEMM differences are covered by the
[CGEMM failure row](#results-and-known-failures). The key is routine, shape,
effective width, distribution, seed and input SHA. Repeated cases and engine
variants with the same key count once; every output for the key must agree.
The keys and all contributing case IDs are in the retry evidence.

All 48 inputs are dyadic small cases: 24 at 16³, 23 at 17³ and 1 INT8 case at
20³. All their outputs are correctly rounded. The retry did not repeat the earlier large
suite, and its 24-bit large CGEMM cases have no matching-width B300 record.
The historical-input checks use a different hash method and are excluded from
this count. No earlier B200 large cases were merged into the retry. Thus this
48/48 comparison and the initial 82/88 comparison cover different sets of inputs.
The initial comparison included 40 matching outputs among 46 inputs at 4096³;
its other 42 common small inputs all matched.

## Solution checks

```sh
python3 AWE/Solutions/S_2xE2M1/Collected_20260920/verify_collected.py
```

This checks coefficient identity, storage membership, encoding, the correction
partition and every retained input pair using Python integers. It does not
certify floating-point execution, scale formats or minimum product counts.
[The schema](../AWE/Solutions/schema.md) defines the record contract.
