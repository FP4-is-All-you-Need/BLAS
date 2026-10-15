#include <cuda_runtime.h>
namespace {
__global__ void transpose_into(const float *A, size_t n, size_t k, float *Y, size_t ldo,
                               size_t row0) {
    __shared__ float t[32][33];
    const size_t i0 = size_t(blockIdx.x) * 32, p0 = size_t(blockIdx.y) * 32;
    for (int y = threadIdx.y; y < 32; y += 8) {
        size_t i = i0 + threadIdx.x, p = p0 + y;
        if (i < n && p < k)
            t[y][threadIdx.x] = A[i + p * n];
    }
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) {
        size_t p = p0 + threadIdx.x, i = i0 + y;
        if (i < n && p < k)
            Y[row0 + p + i * ldo] = t[threadIdx.x][y];
    }
}
__global__ void matrix_copy(const float *A, size_t n, float *X, int symmetric) {
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n * n;
         t += size_t(gridDim.x) * blockDim.x) {
        const size_t i = t % n, j = t / n;
        X[t] = i >= j ? A[t] : (symmetric ? A[j + i * n] : 0.0f);
    }
}
__global__ void commit(const float *P, size_t n, float *C, int lower, int xxt) {
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n * n;
         t += size_t(gridDim.x) * blockDim.x) {
        const size_t i = t % n, j = t / n;
        if (!lower || i >= j)
            C[t] = xxt ? __fadd_rn(P[t], P[j + i * n]) : P[t];
    }
}
} // namespace
extern "C" void support_float_transpose(dim3 g, dim3 b, cudaStream_t s, const float *a, size_t n,
                                        size_t k, float *y, size_t ld, size_t row) {
    transpose_into<<<g, b, 0, s>>>(a, n, k, y, ld, row);
}
extern "C" void support_float_matrix(dim3 g, dim3 b, cudaStream_t s, const float *a, size_t n,
                                     float *x, int sym) {
    matrix_copy<<<g, b, 0, s>>>(a, n, x, sym != 0);
}
extern "C" void support_float_commit(dim3 g, dim3 b, cudaStream_t s, const float *p, size_t n,
                                     float *c, int lower, int xxt) {
    commit<<<g, b, 0, s>>>(p, n, c, lower != 0, xxt != 0);
}
