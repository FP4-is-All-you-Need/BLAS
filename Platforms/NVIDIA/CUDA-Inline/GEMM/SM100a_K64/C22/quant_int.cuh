#pragma once

#include <cstdint>

namespace qint {

__device__ __forceinline__ unsigned long long absbits(double z) {
    return (unsigned long long)__double_as_longlong(z) & 0x7fffffffffffffffull;
}
__device__ __forceinline__ bool finite_bits(unsigned long long a) {
    return (a >> 52) != 0x7ffull;
}
__device__ __forceinline__ int ilogb_bits(unsigned long long a) { // a finite and nonzero
    const int e = int(a >> 52);
    return e ? e - 1023 : (63 - __clzll((long long)a)) - 1074;
}
// __double2ll_rn(ldexp(z, shift)) for finite z with |z| * 2^shift < 2^62.
__device__ __forceinline__ long long scale_round(double z, int shift) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(z),
                             a = b & 0x7fffffffffffffffull;
    if (!a) {
        return 0;
    }
    const int e = int(a >> 52);
    const unsigned long long m = (a & ((1ull << 52) - 1)) | (e ? (1ull << 52) : 0ull);
    const int s = (e ? e : 1) - 1075 + shift; // |z| 2^shift = m 2^s
    unsigned long long v;
    if (s >= 0) {
        v = m << s;
    } else if (s < -54) {
        v = 0; // m < 2^53, so the value is below 1/4
    } else {
        const int d = -s;
        v = m >> d;
        const unsigned long long rem = m & ((1ull << d) - 1), half = 1ull << (d - 1);
        if (rem > half || (rem == half && (v & 1))) {
            ++v;
        }
    }
    return (b >> 63) ? -(long long)v : (long long)v;
}

// A: rows of a column-major matrix. R rows per block, 256 / R column lanes; out is row major (out[o * k + q]).
template <int R>
__global__ void __launch_bounds__(256)
    quant_cols(const double *__restrict__ x, long long *__restrict__ out, int *exponents,
               int *invalid, int outer, int k, int64_t ld, int bits) {
    constexpr int C = 256 / R;
    __shared__ unsigned long long red[C][R];
    __shared__ long long tile[R][C + 1];
    __shared__ int shifts[R];
    const int t = threadIdx.x, r = t % R, c = t / R, o0 = blockIdx.x * R, o = o0 + r;
    unsigned long long mx = 0;
    bool bad = false;
    if (o < outer) {
        for (int q = c; q < k; q += C) {
            const unsigned long long a = absbits(x[size_t(q) * ld + o]);
            if (!finite_bits(a)) {
                bad = true;
            } else {
                mx = a > mx ? a : mx;
            }
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[c][r] = mx;
    __syncthreads();
    if (t < R) {
        unsigned long long m = 0;
        for (int i = 0; i < C; ++i) {
            m = red[i][t] > m ? red[i][t] : m;
        }
        const int sh = m == 0 ? 0 : bits - 1 - ilogb_bits(m);
        shifts[t] = sh;
        if (o0 + t < outer) {
            exponents[o0 + t] = sh;
        }
    }
    __syncthreads();
    const int sh = shifts[r], wr = t / C, wc = t % C;
    for (int q0 = 0; q0 < k; q0 += C) {
        const int q = q0 + c;
        long long v = 0;
        if (o < outer && q < k) {
            const double z = x[size_t(q) * ld + o];
            v = finite_bits(absbits(z)) ? scale_round(z, sh) : 0;
        }
        tile[r][c] = v;
        __syncthreads();
        if (o0 + wr < outer && q0 + wc < k) {
            out[size_t(o0 + wr) * k + q0 + wc] = tile[wr][wc];
        }
        __syncthreads();
    }
}

// B: rows that are contiguous in memory (element (row o, column q) at o * ld + q). One block per row.
__global__ void __launch_bounds__(256)
    quant_rows(const double *__restrict__ x, long long *__restrict__ out, int *exponents,
               int *invalid, int outer, int k, int64_t ld, int bits) {
    __shared__ unsigned long long red[256];
    __shared__ int shift_s;
    const int o = blockIdx.x, t = threadIdx.x;
    const double *row = x + size_t(o) * ld;
    unsigned long long mx = 0;
    bool bad = false;
    for (int q = t; q < k; q += 256) {
        const unsigned long long a = absbits(row[q]);
        if (!finite_bits(a)) {
            bad = true;
        } else {
            mx = a > mx ? a : mx;
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[t] = mx;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) {
            red[t] = red[t + s] > red[t] ? red[t + s] : red[t];
        }
        __syncthreads();
    }
    if (t == 0) {
        const int sh = red[0] == 0 ? 0 : bits - 1 - ilogb_bits(red[0]);
        shift_s = sh;
        exponents[o] = sh;
    }
    __syncthreads();
    const int sh = shift_s;
    for (int q = t; q < k; q += 256) {
        const double z = row[q];
        out[size_t(o) * k + q] = finite_bits(absbits(z)) ? scale_round(z, sh) : 0;
    }
}

} // namespace qint
