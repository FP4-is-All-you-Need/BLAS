# C and C++ API

Include `awe_blas.h` (C) or `awe_blas.hpp` (C++ ownership wrapper). Matrix and workspace pointers are device memory; descriptors, options and query results are host objects. Structures carry `struct_size` for ABI checking; use the init functions.

## Operation

`C = op(A) * op(B)` with `alpha=1`, `beta=0`. `op(A)` is `m` by `k`, `op(B)` is `k` by `n`, `C` is `m` by `n`. Transposes `AWE_OP_N`, `AWE_OP_T`; layouts `AWE_COLUMN_MAJOR`, `AWE_ROW_MAJOR`. Minimum leading dimensions:

| Layout | `lda`, A not transposed | `lda`, A transposed | `ldb`, B not transposed | `ldb`, B transposed | `ldc` |
|---|---:|---:|---:|---:|---:|
| Column-major | max(1,m) | max(1,k) | max(1,k) | max(1,n) | max(1,m) |
| Row-major | max(1,k) | max(1,m) | max(1,n) | max(1,k) | max(1,n) |

Inputs may contain every INT8 value, -128 through 127. M and N are limited to 1048576, K and leading dimensions to `INT_MAX`. M=0 or N=0 is a no-op; K=0 writes zeros.

## Enumerations

| Type | Values |
|---|---|
| `awe_method` | `AWE_METHOD_AUTO` (0), `AWE_METHOD_AWE` (1): input-aware, one or six planes. `AWE_METHOD_RADIX13` (2): three radix-13 limbs, nine planes. `AWE_METHOD_OZAKI2` (3): modular backend. |
| `awe_data_type` | `AWE_I8_I64` (0), default. `AWE_I8_I32` (2): plan creation rejects K > 131071 with `AWE_INPUT_OUT_OF_RANGE`. `AWE_F64_F64` (1) is not supported. |

## Plan Options

| Field | Default | Meaning |
|---|---|---|
| `method` | `AWE_METHOD_AUTO` | Backend. |
| `stream_count` | 3 | 1 serial stream, or 3 streams for encoding, GEMM and reconstruction. |
| `buffer_count` | 3 | Reusable buffers, 1 through 8; the modular backend also accepts 9 through 11. Query workspace again after changing it. |
| `k_chunk` | 0 | Payout period; 0 selects it automatically. Query the actual period with `awe_plan_get_info`. |
| `profiling` | 0 | Nonzero records stage times. |
| `heuristic_index` | 0 | cuBLASLt heuristic candidate, 0 through 15. |
| `matmul_workspace_bytes` | 64 MiB | cuBLASLt scratch budget, included in the workspace query. |

## Exactness

Each plane product is accumulated in floating point over one payout period, converted to an integer, and combined by integer arithmetic; the result is exact for any K. `awe_plan_get_info` reports `guaranteed_k` (the bound for one period, 116480 for AWE and radix-13) and `payout_period` (the period used).

## Functions

| Function | Result |
|---|---|
| `awe_gemm_desc_init(desc)` | Column-major, no transpose, INT8 to INT64, zero dimensions. |
| `awe_options_init(options)` | Defaults above. |
| `awe_plan_create(&plan, &desc, &options)` | Validates and creates the plan; `options` may be null. |
| `awe_plan_get_workspace_size(plan, &bytes)` | Device workspace to allocate (256-byte aligned). |
| `awe_plan_get_info(plan, &info)` | Method, weights, plane count, `guaranteed_k`, `payout_period`, padded dimensions, `algorithm_id`, `workspace_bytes`. Query again after execution for input-dependent plane selection. |
| `awe_plan_execute(plan, a, b, c, workspace, bytes, stream, stats)` | Runs the product; `stats` may be null. |
| `awe_gemm_i8(...)`, `awe_gemm_i8_i32(...)` | Typed variants of `awe_plan_execute`. |
| `awe_plan_get_last_stats(plan, &stats)` | Statistics of the last execution. |
| `awe_plan_destroy(plan)` | Releases the plan. |
| `awe_status_string(status)` | Static text for a status. |

A plan serves one call at a time. Execution is ordered on the supplied stream and completes before returning. CUDA graph capture is unsupported. On failure C may be partly written.

## Status Values

| Value | Code | Meaning |
|---|---:|---|
| `AWE_SUCCESS` | 0 | |
| `AWE_INVALID_ARGUMENT` | 1 | Pointer, structure size, dimensions, strides, option value, alignment or overlap. |
| `AWE_NOT_SUPPORTED` | 2 | Type, backend, alpha/beta, graph capture, or no usable cuBLASLt algorithm. |
| `AWE_INSUFFICIENT_WORKSPACE` | 3 | Workspace smaller than the plan requires. |
| `AWE_CUDA_ERROR` | 4 | CUDA or cuBLAS failed. |
| `AWE_INTERNAL_ERROR` | 5 | Internal failure. |
| `AWE_INPUT_OUT_OF_RANGE` | 6 | Contract violation, including INT32 output with K above 131071. |

## C++ Wrapper

`awe::plan` owns one plan (move-only). `create(desc, options)`, `workspace_size(bytes)` and `execute(...)` return the C statuses; `get()` returns the borrowed C handle. See [minimal.c](../examples/minimal.c) and [minimal.cpp](../examples/minimal.cpp).

## Environment Variables

Read at plan creation, AWE backend only. `AWE_FUSION=1` (default) fuses encoding and reconstruction with the GEMMs for a single payout period; `AWE_FUSION=0` selects the per-plane path. `AWE_FUSED_TILE_N` (default 0 = whole width; otherwise a multiple of 128) limits the output width per reconstruction tile.
