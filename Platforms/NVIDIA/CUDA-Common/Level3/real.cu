// Recorded Level-3 support kernels; see Docs/Provenance/level3_support.json.
#include <cuda_runtime.h>
#include <cstdint>
namespace {
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31);
}
__global__ void fill_normal(double* x, size_t count, uint64_t seed) {   
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + seed; z = mix64(z);
    uint64_t w = z * 0xD1342543DE82EF95ull + 0x2545F4914F6CDD1Dull; w = (w ^ (w >> 29)) * 0xBF58476D1CE4E5B9ull; w ^= w >> 32;
    const double u1 = (double(z >> 11) + 0.5) / 9007199254740992.0, u2 = (double(w >> 11) + 0.5) / 9007199254740992.0;
    x[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
  }
}
__global__ void hash_kernel(const double* x, size_t rows, size_t cols, int lower_only, unsigned long long* out) {
  unsigned long long acc = 0;
  const size_t count = rows * cols;
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < count; t += size_t(gridDim.x) * blockDim.x) {
    const size_t i = t % rows, j = t / rows;
    if (lower_only && i < j) continue;
    const double v = x[t] == 0.0 ? 0.0 : x[t];
    acc ^= (unsigned long long)__double_as_longlong(v) * 0x9E3779B97F4A7C15ull + (unsigned long long)t * 0xD1342543DE82EF95ull;
  }
  atomicXor(out, acc);
}
__global__ void gather(const double* x, size_t ld, const int* ri, const int* ci, int count, double* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < count) out[t] = x[size_t(ri[t]) + size_t(ci[t]) * ld];
}
__global__ void transpose_nk(const double* __restrict__ A, size_t n, size_t k, double* __restrict__ At) {   
  __shared__ double t[32][33];
  const size_t i0 = size_t(blockIdx.x) * 32, p0 = size_t(blockIdx.y) * 32;
  for (int y = threadIdx.y; y < 32; y += 8) { const size_t i = i0 + threadIdx.x, p = p0 + y; if (i < n && p < k) t[y][threadIdx.x] = A[i + p * n]; }
  __syncthreads();
  for (int y = threadIdx.y; y < 32; y += 8) { const size_t p = p0 + threadIdx.x, i = i0 + y; if (i < n && p < k) At[p + i * k] = t[threadIdx.x][y]; }
}
__global__ void transpose_into(const double* __restrict__ A, size_t n, size_t k, double* __restrict__ Y, size_t ldo, size_t row0) {   
  __shared__ double t[32][33];
  const size_t i0 = size_t(blockIdx.x) * 32, p0 = size_t(blockIdx.y) * 32;
  for (int y = threadIdx.y; y < 32; y += 8) { const size_t i = i0 + threadIdx.x, p = p0 + y; if (i < n && p < k) t[y][threadIdx.x] = A[i + p * n]; }
  __syncthreads();
  for (int y = threadIdx.y; y < 32; y += 8) { const size_t p = p0 + threadIdx.x, i = i0 + y; if (i < n && p < k) Y[row0 + p + i * ldo] = t[threadIdx.x][y]; }
}
__global__ void symm_full(const double* __restrict__ A, size_t n, double* __restrict__ S) {   
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < n * n; t += size_t(gridDim.x) * blockDim.x) {
    const size_t i = t % n, j = t / n; S[t] = i >= j ? A[t] : A[j + i * n]; }
}
__global__ void copy_lower(const double *input, size_t n, double *output) {
    for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         t < n * n; t += size_t(gridDim.x) * blockDim.x) {
        if (t % n >= t / n) output[t] = input[t];
    }
}
__global__ void xxt_lower(const double* __restrict__ X, size_t n, double* __restrict__ C) {   
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < n * n; t += size_t(gridDim.x) * blockDim.x) {
    const size_t i = t % n, j = t / n; if (i >= j) C[t] = X[t] + X[j + i * n]; }
}
__global__ void fill_nan(double* __restrict__ x, size_t count) {   
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < count; t += size_t(gridDim.x) * blockDim.x) x[t] = __longlong_as_double(0x7ff8dead0000beefll); }
__global__ void nan_lower(const double* __restrict__ C, size_t n, unsigned long long* out) {   
  unsigned long long c = 0;
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < n * n; t += size_t(gridDim.x) * blockDim.x) { const size_t i = t % n, j = t / n; if (i >= j && C[t] != C[t]) ++c; }
  atomicAdd(out, c); }
__global__ void nan_full(const double* __restrict__ C, size_t count, unsigned long long* out) {   
  unsigned long long c = 0;
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < count; t += size_t(gridDim.x) * blockDim.x) if (C[t] != C[t]) ++c;
  atomicAdd(out, c); }
__global__ void tril_copy(const double* __restrict__ A, size_t n, double* __restrict__ L) {   
  for (size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x; t < n * n; t += size_t(gridDim.x) * blockDim.x) {
    const size_t i = t % n, j = t / n; L[t] = i >= j ? A[t] : 0.0; }
}
}
extern "C" void support_fill_normal(dim3 grid,dim3 block,cudaStream_t stream,double* x, size_t count, uint64_t seed) { fill_normal<<<grid,block,0,stream>>>(x,count,seed); }
extern "C" void support_hash_kernel(dim3 grid,dim3 block,cudaStream_t stream,const double* x, size_t rows, size_t cols, int lower_only, unsigned long long* out) { hash_kernel<<<grid,block,0,stream>>>(x,rows,cols,lower_only,out); }
extern "C" void support_gather(dim3 grid,dim3 block,cudaStream_t stream,const double* x, size_t ld, const int* ri, const int* ci, int count, double* out) { gather<<<grid,block,0,stream>>>(x,ld,ri,ci,count,out); }
extern "C" void support_transpose_nk(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ A, size_t n, size_t k, double* __restrict__ At) { transpose_nk<<<grid,block,0,stream>>>(A,n,k,At); }
extern "C" void support_transpose_into(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ A, size_t n, size_t k, double* __restrict__ Y, size_t ldo, size_t row0) { transpose_into<<<grid,block,0,stream>>>(A,n,k,Y,ldo,row0); }
extern "C" void support_symm_full(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ A, size_t n, double* __restrict__ S) { symm_full<<<grid,block,0,stream>>>(A,n,S); }
extern "C" void support_xxt_lower(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ X, size_t n, double* __restrict__ C) { xxt_lower<<<grid,block,0,stream>>>(X,n,C); }
extern "C" void support_fill_nan(dim3 grid,dim3 block,cudaStream_t stream,double* __restrict__ x, size_t count) { fill_nan<<<grid,block,0,stream>>>(x,count); }
extern "C" void support_nan_lower(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ C, size_t n, unsigned long long* out) { nan_lower<<<grid,block,0,stream>>>(C,n,out); }
extern "C" void support_nan_full(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ C, size_t count, unsigned long long* out) { nan_full<<<grid,block,0,stream>>>(C,count,out); }
extern "C" void support_tril_copy(dim3 grid,dim3 block,cudaStream_t stream,const double* __restrict__ A, size_t n, double* __restrict__ L) { tril_copy<<<grid,block,0,stream>>>(A,n,L); }
extern "C" void support_copy_lower(dim3 grid, dim3 block, cudaStream_t stream,
                                    const double *input, size_t n, double *output) {
    copy_lower<<<grid, block, 0, stream>>>(input, n, output);
}
