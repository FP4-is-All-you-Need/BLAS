#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace shash {

__host__ __device__ __forceinline__ unsigned long long
mix64(unsigned long long z) { // splitmix64's finalizer
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ unsigned long long bits_of(double v) {
    return (unsigned long long)__double_as_longlong(v);
}
__device__ __forceinline__ unsigned long long bits_of(double2 v) {
    return mix64((unsigned long long)__double_as_longlong(v.x)) ^
           (unsigned long long)__double_as_longlong(v.y);
}
// the per-element term: position and value mixed, XOR-combined (order-free, so the reduction order does not matter)
__device__ __forceinline__ unsigned long long term(unsigned long long pos, unsigned long long v) {
    return mix64(v * 0x9E3779B97F4A7C15ull + pos * 0xD1342543DE82EF95ull + 0x2545F4914F6CDD1Dull);
}
__device__ __forceinline__ unsigned long long block_xor(unsigned long long h) {
    __shared__ unsigned long long part[32];
    for (int o = 16; o; o >>= 1) {
        h ^= __shfl_xor_sync(0xffffffffu, h, o);
    }
    if ((threadIdx.x & 31) == 0) {
        part[threadIdx.x >> 5] = h;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        h = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0ull;
        for (int o = 16; o; o >>= 1) {
            h ^= __shfl_xor_sync(0xffffffffu, h, o);
        }
    }
    return h;
}

// one sample per thread (n / 256 blocks: the scattered loads are all in flight at once), XOR-combined into *out (zeroed first)
template <class T>
__global__ void __launch_bounds__(256)
    sample_kernel(const T *__restrict__ x, int64_t ld, int rows, int cols, int n,
                  unsigned long long seed, unsigned long long *__restrict__ out) {
    unsigned long long h = 0;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < n) {
        const unsigned long long z = mix64(seed + (unsigned long long)s * 0x9E3779B97F4A7C15ull);
        const unsigned i = unsigned((z & 0xFFFFFFFFull) % unsigned(rows)),
                       j = unsigned((z >> 32) % unsigned(cols));
        // keyed by the sample's own word z, not by its position: two samples that land on the same element must not cancel under
        // XOR (they did at 97 x 229 with n = 4,096 before this); z fixes the position, so the value still depends only on the matrix
        h = term(z, bits_of(x[size_t(j) * size_t(ld) + i]));
    }
    if (s == 0) {
        h ^= mix64(((unsigned long long)rows << 32 | unsigned(cols)) ^
                   (unsigned long long)n * 0x94D049BB133111EBull);
    }
    h = block_xor(h);
    if (threadIdx.x == 0) {
        atomicXor(out, h);
    }
}

template <class T>
__global__ void __launch_bounds__(256) full_kernel(const T *__restrict__ x, int64_t ld, int rows,
                                                   int cols, unsigned long long *__restrict__ out) {
    unsigned long long h = 0;
    for (int j = blockIdx.x; j < cols;
         j += gridDim.x) { // a column per block step, its rows across the threads (no division)
        for (int i = threadIdx.x; i < rows; i += blockDim.x) {
            h ^= term((unsigned long long)j * unsigned(rows) + unsigned(i),
                      bits_of(x[size_t(j) * size_t(ld) + i]));
        }
    }
    h = block_xor(h);
    if (threadIdx.x == 0) {
        atomicXor(out, h);
    }
}

// n positions (fixed by rows, cols, n and seed) -> *out (device memory, zeroed here first); ceil(n / 256) blocks. The caller reads
// *out (8 bytes; a pinned buffer and cudaMemcpyAsync keep the read cheap) or compares it on the device. rows, cols >= 1, n >= 1.
template <class T>
inline cudaError_t hash_sample(const T *x, int64_t ld, int rows, int cols, int n,
                               unsigned long long *out, cudaStream_t stream = 0,
                               unsigned long long seed = 0x6A09E667F3BCC909ull) {
    if (rows < 1 || cols < 1 || n < 1 || ld < rows) {
        return cudaErrorInvalidValue;
    }
    cudaError_t e = cudaMemsetAsync(out, 0, 8, stream);
    if (e != cudaSuccess) {
        return e;
    }
    sample_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(x, ld, rows, cols, n, seed, out);
    return cudaGetLastError();
}
// every element -> *out (device memory, zeroed here first)
template <class T>
inline cudaError_t hash_full(const T *x, int64_t ld, int rows, int cols, unsigned long long *out,
                             cudaStream_t stream = 0) {
    if (rows < 1 || cols < 1 || ld < rows) {
        return cudaErrorInvalidValue;
    }
    cudaError_t e = cudaMemsetAsync(out, 0, 8, stream);
    if (e != cudaSuccess) {
        return e;
    }
    full_kernel<T><<<1024, 256, 0, stream>>>(x, ld, rows, cols, out);
    return cudaGetLastError();
}

} // namespace shash
