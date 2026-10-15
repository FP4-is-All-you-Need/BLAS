#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef __CUDACC__
#define __host__
#define __device__
#endif
// The logical operand is indexed as (outer row, contraction index).
// 0 keeps the inherited memory layout; 1 reads transposed B;
// 2 reads a lower triangular A; 3 reflects the lower triangular A.
struct S3InputView {
    int mode = 0, n = 0;
    template <class T>
    __host__ __device__ T load(const T *x, size_t index, int64_t ld, int side_b) const {
        if (!mode) {
            return x[index];
        }
        const size_t i = side_b ? index / size_t(ld) : index % size_t(ld);
        const size_t p = side_b ? index % size_t(ld) : index / size_t(ld);
        if (mode == 1) {
            return x[i + p * size_t(n)];
        }
        if (mode == 2) {
            return i >= p ? x[i + p * size_t(n)] : T{};
        }
        return i >= p ? x[i + p * size_t(n)] : x[p + i * size_t(n)];
    }
};
