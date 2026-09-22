#ifndef AWE_BLAS_H
#define AWE_BLAS_H
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
#define AWE_ABI_VERSION 1

typedef enum { AWE_SUCCESS=0, AWE_INVALID_ARGUMENT=1, AWE_NOT_SUPPORTED=2,
 AWE_INSUFFICIENT_WORKSPACE=3, AWE_CUDA_ERROR=4, AWE_INTERNAL_ERROR=5,
 AWE_INPUT_OUT_OF_RANGE=6 } awe_status;
typedef enum { AWE_METHOD_AUTO=0, AWE_METHOD_AWE=1, AWE_METHOD_RADIX13=2,
 AWE_METHOD_OZAKI2=3 } awe_method;
/* AWE_F64_F64 is not implemented. Valid FP64 plan requests
 * return AWE_NOT_SUPPORTED for every method, including empty matrices. */
typedef enum { AWE_I8_I64=0, AWE_F64_F64=1, AWE_I8_I32=2 } awe_data_type;
typedef enum { AWE_OP_N=0, AWE_OP_T=1 } awe_operation;
typedef enum { AWE_COLUMN_MAJOR=0, AWE_ROW_MAJOR=1 } awe_layout;
/* C = alpha op(A) op(B) + beta C; only alpha=1, beta=0 are supported.
 * lda/ldb/ldc are physical column/row strides in elements for the layout.
 * INT8 covers [-128,127] without approximation. Output is signed INT64.
 * AWE_I8_I32 selects INT32 and rejects K > 131071 at plan creation.
 * M=0 or N=0 is a successful no-op. K=0 writes zero to logical C.
 * Negative dimensions are rejected. Leading dimensions remain at least 1.
 * Padding outside the logical matrix must not be read or written. */
typedef struct {
 uint32_t struct_size;
 int64_t m,n,k,lda,ldb,ldc;
 awe_operation trans_a,trans_b;
 awe_data_type data_type;
 awe_layout layout;
 double alpha,beta;
} awe_gemm_desc;
typedef struct {
 uint32_t struct_size;
 awe_method method;
 int stream_count; /* 1 (serial) or 3 (encode, GEMM, reconstruction). */
 int buffer_count; /* 1..8; Ozaki II accepts 1..11. Default 3. */
 int k_chunk; /* 0: automatic; otherwise positive, bounded by exactness. */
 int profiling; /* Nonzero enables per-stage event timing. */
 int heuristic_index; /* Index among successful cuBLASLt heuristics; default 0. */
 size_t matmul_workspace_bytes; /* cuBLASLt scratch limit; default 64 MiB. */
} awe_options;
typedef struct {
 uint32_t struct_size;
 awe_method method;
 int weights[3], face_count, guaranteed_k, payout_period, stream_count;
 int padded_m,padded_n,padded_k,algorithm_id;
 size_t workspace_bytes;
 int input_min,input_max; /* Both INT8 operands; inclusive. */
} awe_plan_info;
typedef struct {
 uint32_t struct_size;
 double total_ms,selection_ms,encode_ms,gemm_ms,reconstruction_ms,wait_ms;
 int face_count,chunk_count;
 /* Stage sums may overlap; do not add them to estimate total_ms.
  * wait_ms counts waits on the GEMM stream for encoded input. */
} awe_execution_stats;
typedef struct awe_plan awe_plan;
/* Sets column-major, no transpose, INT8->INT64, alpha=1, beta=0.
 * Set dimensions and leading dimensions before creating the plan. */
void awe_gemm_desc_init(awe_gemm_desc* desc);
void awe_options_init(awe_options* options);
const char* awe_status_string(awe_status status);
awe_status awe_plan_create(awe_plan** plan,const awe_gemm_desc* desc,const awe_options* options);
awe_status awe_plan_get_workspace_size(const awe_plan* plan,size_t* bytes);
awe_status awe_plan_get_info(const awe_plan* plan,awe_plan_info* info);
awe_status awe_plan_get_last_stats(const awe_plan* plan,awe_execution_stats* stats);
/* Synchronous with respect to this plan's work. The supplied stream orders
 * preceding input writes and subsequent output consumers. No device-wide
 * synchronization. One plan/workspace cannot be used concurrently.
 * Workspace must be a device pointer aligned to 256 bytes, of queried size.
 * Plan creation allocates descriptors/events and temporary query storage;
 * execution allocates no device
 * storage. Execution includes input selection, scale setup, encode, GEMM,
 * and reconstruction. Creation and input transfers are excluded.
 * On error C may be partially written; no fallback calculation is performed. */
awe_status awe_plan_execute(awe_plan* plan,const void* a,const void* b,void* c,
 void* workspace,size_t workspace_bytes,cudaStream_t stream,awe_execution_stats* stats);
awe_status awe_gemm_i8(awe_plan* plan,const int8_t* a,const int8_t* b,int64_t* c,
 void* workspace,size_t workspace_bytes,cudaStream_t stream,awe_execution_stats* stats);
awe_status awe_gemm_i8_i32(awe_plan* plan,const int8_t* a,const int8_t* b,int32_t* c,
 void* workspace,size_t workspace_bytes,cudaStream_t stream,awe_execution_stats* stats);
/* ABI entry: always returns AWE_NOT_SUPPORTED, including null
 * arguments. No pointers are dereferenced and no work is submitted. */
awe_status awe_gemm_f64(awe_plan* plan,const double* a,const double* b,double* c,
 void* workspace,size_t workspace_bytes,cudaStream_t stream,awe_execution_stats* stats);
void awe_plan_destroy(awe_plan* plan);
#ifdef __cplusplus
}
#endif
#endif
