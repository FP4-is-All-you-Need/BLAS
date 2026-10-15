#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HAYASHI_STATUS_SUCCESS = 0,
    HAYASHI_STATUS_INVALID_VALUE = 1,
    HAYASHI_STATUS_NOT_SUPPORTED = 2,
    HAYASHI_STATUS_ALLOC_FAILED = 3,
    HAYASHI_STATUS_CUBLAS_ERROR = 4,
    HAYASHI_STATUS_CUDA_ERROR = 5
} hayashiStatus_t;

typedef struct hayashiContext *hayashiHandle_t;

hayashiStatus_t hayashiCreate(hayashiHandle_t *handle);
hayashiStatus_t hayashiDestroy(hayashiHandle_t handle);
hayashiStatus_t hayashiSetStream(hayashiHandle_t handle, cudaStream_t stream);
// upper bound on the bytes of FP32 piece buffers (default 4 GiB). Fewer buffers = the pieces are added in several
// passes.
hayashiStatus_t hayashiSetPieceBufferLimit(hayashiHandle_t handle, size_t bytes);

hayashiStatus_t hayashiGemmI8TN(hayashiHandle_t handle, int m, int n, int k, const int32_t *alpha, const int8_t *A,
                                int lda, const int8_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc);

#define HAYASHI_OP_N 0
#define HAYASHI_OP_T 1
hayashiStatus_t hayashiGemmI8(hayashiHandle_t handle, int transa, int transb, int m, int n, int k, const int32_t *alpha,
                              const int8_t *A, int lda, long long strideA, const int8_t *B, int ldb, long long strideB,
                              const int32_t *beta, int32_t *C, int ldc, long long strideC, int batchCount);
hayashiStatus_t hayashiGemmI8TNStridedBatched(hayashiHandle_t handle, int m, int n, int k, const int32_t *alpha,
                                              const int8_t *A, int lda, long long strideA, const int8_t *B, int ldb,
                                              long long strideB, const int32_t *beta, int32_t *C, int ldc,
                                              long long strideC, int batchCount);
int hayashiLastBatchedByLoop(hayashiHandle_t handle);

// Quantizing entry (797): A, B are BF16 (k x m / k x n column-major, lda, ldb multiples of 16, 16-byte aligned) with
// per-row inverse scales invScaleA[m], invScaleB[n].  Each value is quantized as q = round-to-nearest-even(x *
// invScale) clamped to -102..102 inside the encoder (no INT8 array is written), and C (INT32) = sum_k qA * qB exactly
// (the caller multiplies by the scales).  +-103 cannot occur, so none of the +-103 work runs.  Finite inputs only.
hayashiStatus_t hayashiGemmBF16QuantTN(hayashiHandle_t handle, int m, int n, int k, const void *A, int lda,
                                       const float *invScaleA, const void *B, int ldb, const float *invScaleB,
                                       int32_t *C, int ldc);

hayashiStatus_t hayashiSetBackend(hayashiHandle_t handle, int backend);
int hayashiLastBackend(hayashiHandle_t handle);
hayashiStatus_t hayashiGetLastSettings(hayashiHandle_t handle, int *s6);
hayashiStatus_t hayashiSetTailSplit(hayashiHandle_t handle, int on);
int hayashiLastTailSplit(hayashiHandle_t handle);
hayashiStatus_t hayashiSetPlus103Mode(hayashiHandle_t handle, int mode);
int hayashiLastPlus103Dense(hayashiHandle_t handle);
hayashiStatus_t hayashiGetDenseCounts(hayashiHandle_t handle, long long *out2);
hayashiStatus_t hayashiSetFewCorr(hayashiHandle_t handle, int on);
hayashiStatus_t hayashiGetLastV5(hayashiHandle_t handle, long long *out10);
hayashiStatus_t hayashiGetLastF5(hayashiHandle_t handle, long long *out8);
hayashiStatus_t hayashiGetLastRecords(hayashiHandle_t handle, long long *out4);
hayashiStatus_t hayashiGemmI8BatchedHost(hayashiHandle_t handle, int transa, int transb, int m, int n, int k,
                                         const int32_t *alpha, const int8_t *const *A, int lda, const int8_t *const *B,
                                         int ldb, const int32_t *beta, int32_t *const *C, int ldc, int batchCount);

hayashiStatus_t hayashiSetProfiling(hayashiHandle_t handle, int on);
hayashiStatus_t hayashiGetLastStageTimes(hayashiHandle_t handle, float *ms4);
hayashiStatus_t hayashiSetTransposeVariant(hayashiHandle_t handle, int variant);
hayashiStatus_t hayashiSetDirectOutput(hayashiHandle_t handle, int on);
hayashiStatus_t hayashiSetReuseB(hayashiHandle_t handle, int on);
hayashiStatus_t hayashiSetEncodeVariant(hayashiHandle_t handle, int variant);
hayashiStatus_t hayashiSetE2(hayashiHandle_t handle, int on);
// Whether the last backend-1 call ran e2 (waits for the stream).
hayashiStatus_t hayashiGetLastKernel(hayashiHandle_t handle, int *e2used);
hayashiStatus_t hayashiSetTable(hayashiHandle_t handle, int which);

hayashiStatus_t hayashiSetGraphMode(hayashiHandle_t handle, int on);

hayashiStatus_t hayashiGetLastInfo(hayashiHandle_t handle, long long *count103A, long long *count103B, int *overflow);

#ifdef __cplusplus
}
#endif
