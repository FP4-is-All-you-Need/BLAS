#pragma once
// Scaling shifts and the CRT reconstruction.
//
// Reconstruction uses the sum form: C = S - k P with S = sum_i y_i Q_i, Q_i = P / p_i and
// y_i = r_i (Q_i^-1 mod p_i) mod p_i. k = round(sum_i y_i / p_i) is computed in FP64: S / P equals that
// sum exactly, its FP64 error is below 19 * 2^-52, and |C| stays well below P / 2 because the table
// guarantees 2 K 2^(2 b) < P, so k is exact. C is the centred integer dot product; its sign, magnitude
// and the final rounding are then the same as a digit-by-digit (Garner) reconstruction.
#include "common.h"
#include "table_d53_k8192.h"
#include "wide_integer.h"
#include "word_round.h"
#include <cstdint>

namespace awe_mxfp6 {
namespace table = fp6_catalog_table;
namespace crt {

constexpr unsigned F = table::form_count;
constexpr unsigned NW = sizeof(table::Big) / 4;
using Wide = awe_wide::Integer<NW + 1>; // S < 19 P needs one more word than P

template <unsigned N> constexpr uint32_t big_mod(const awe_wide::Integer<N> &a, uint32_t p) {
    uint64_t r = 0;
    for (int i = int(N) - 1; i >= 0; --i) {
        r = ((r << 32) | a.w[i]) % p;
    }
    return uint32_t(r);
}
template <unsigned N> constexpr awe_wide::Integer<N + 1> widen(const awe_wide::Integer<N> &a) {
    awe_wide::Integer<N + 1> r{};
    for (unsigned i = 0; i < N; ++i) {
        r.w[i] = a.w[i];
    }
    return r;
}
template <unsigned N>
constexpr awe_wide::Integer<N> big_div(const awe_wide::Integer<N> &a, uint32_t p) {
    awe_wide::Integer<N> q{};
    uint64_t r = 0;
    for (int i = int(N) - 1; i >= 0; --i) {
        const uint64_t cur = (r << 32) | a.w[i];
        q.w[i] = uint32_t(cur / p);
        r = cur % p;
    }
    return q;
}
constexpr uint32_t mod_inverse(uint32_t a, uint32_t p) {
    int64_t t = 0, nt = 1, r = p, nr = a % p;
    while (nr) {
        const int64_t q = r / nr, tt = t - q * nt, rr = r - q * nr;
        t = nt;
        nt = tt;
        r = nr;
        nr = rr;
    }
    return r == 1 ? uint32_t(t < 0 ? t + p : t) : 0u;
}

struct SumTable {
    Wide q[F];
    uint32_t c[F];
};
constexpr SumTable make_sum_table() {
    SumTable t{};
    for (unsigned i = 0; i < F; ++i) {
        const auto qi = big_div(table::modulus_product, table::host_forms[i].p);
        t.q[i] = widen(qi);
        t.c[i] = mod_inverse(big_mod(qi, table::host_forms[i].p), table::host_forms[i].p);
    }
    return t;
}
inline constexpr SumTable sums = make_sum_table();

// Template recursion keeps every table access at a compile-time index, so no table lives in memory.
template <unsigned I = 0>
AWE_MXFP6_HD_INLINE void sum_terms(Wide &acc, double &f, const uint16_t *residues, size_t index,
                                   size_t stride) {
    if constexpr (I < F) {
        constexpr uint32_t p = table::host_forms[I].p, c = sums.c[I];
        static_assert(c != 0u, "Q_i invertible mod p_i");
        constexpr auto q = sums.q[I];
        constexpr double inv = 1.0 / double(p);
        const uint32_t y = uint32_t(uint64_t(residues[size_t(I) * stride + index]) * c % p);
        acc = awe_wide::add_multiply(acc, q, y);
        f = __builtin_fma(double(y), inv, f);
        sum_terms<I + 1>(acc, f, residues, index, stride);
    }
}

// One output element on the host: residues of all forms (stride elements apart) -> rounded binary64
// times 2^exponent. reconstruct_kernel writes these steps out in its own body, which fixes its
// instruction schedule.
AWE_MXFP6_HD_INLINE double reconstruct_element(const uint16_t *residues, size_t index,
                                               size_t stride, int exponent) {
    Wide s{};
    double f = 0.0;
    sum_terms(s, f, residues, index, stride);
    const uint32_t k = uint32_t(__builtin_rint(f));
    constexpr Wide P = widen(table::modulus_product);
    const Wide kp = awe_wide::add_multiply(Wide{}, P, k);
    Wide d{};
    uint64_t borrow = 0;
#pragma unroll
    for (unsigned w = 0; w < NW + 1; ++w) {
        const uint64_t rhs = uint64_t(kp.w[w]) + borrow;
        d.w[w] = uint32_t(uint64_t(s.w[w]) - rhs);
        borrow = uint64_t(s.w[w]) < rhs;
    }
    const bool negative = borrow != 0;
    if (negative) {
        uint64_t carry = 1;
#pragma unroll
        for (unsigned w = 0; w < NW + 1; ++w) {
            const uint64_t v = uint64_t(~d.w[w]) + carry;
            d.w[w] = uint32_t(v);
            carry = v >> 32;
        }
    }
    table::Big magnitude{};
#pragma unroll
    for (unsigned w = 0; w < NW; ++w) {
        magnitude.w[w] = d.w[w];
    }
    return awe_word_round::to_double(magnitude, negative, exponent);
}

// shift = (bits - 1) - exponent of the row maximum: the largest value becomes a bits-bit integer.
// largest holds the raw magnitude bits; the subnormal branch shifts a copy, so the zero test below
// still sees the maximum.
AWE_MXFP6_HD_INLINE int shift_of(uint64_t largest, int bits) {
    uint64_t raw = largest;
    int exponent = 0;
    if (raw) {
        const unsigned encoded = unsigned(raw >> 52);
        if (encoded) {
            exponent = int(encoded) - 1023;
        } else {
            exponent = -1074;
            while (raw >>= 1) {
                ++exponent;
            }
        }
    }
    return largest ? bits - 1 - exponent : 0;
}

#if defined(__HIPCC__)
// Same steps as reconstruct_element, written out in the kernel body.
__global__ __launch_bounds__(256) void reconstruct_kernel(const uint16_t *residues, double *output,
                                                          const int *sa, const int *sb, int m,
                                                          int n, int ld) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(m) * n) {
        return;
    }
    Wide s{};
    double f = 0.0;
    sum_terms(s, f, residues, i, size_t(m) * n);
    const uint32_t k = uint32_t(__builtin_rint(f));
    constexpr Wide P = widen(table::modulus_product);
    const Wide kp = awe_wide::add_multiply(Wide{}, P, k);
    Wide d{};
    uint64_t borrow = 0;
#pragma unroll
    for (unsigned w = 0; w < NW + 1; ++w) {
        const uint64_t rhs = uint64_t(kp.w[w]) + borrow;
        d.w[w] = uint32_t(uint64_t(s.w[w]) - rhs);
        borrow = uint64_t(s.w[w]) < rhs;
    }
    const bool negative = borrow != 0;
    if (negative) {
        uint64_t carry = 1;
#pragma unroll
        for (unsigned w = 0; w < NW + 1; ++w) {
            const uint64_t v = uint64_t(~d.w[w]) + carry;
            d.w[w] = uint32_t(v);
            carry = v >> 32;
        }
    }
    table::Big magnitude{};
#pragma unroll
    for (unsigned w = 0; w < NW; ++w) {
        magnitude.w[w] = d.w[w];
    }
    output[(i / m) * ld + i % m] =
        awe_word_round::to_double(magnitude, negative, -sa[i % m] - sb[i / m]);
}

// A side: 256 consecutive rows per block with K split over blockIdx.y, so reads are coalesced along
// the column-major rows; the per-row maximum of the raw magnitude bits is combined with atomicMax.
__global__ __launch_bounds__(256) void shifts_a_partial(const double *input,
                                                        unsigned long long *largest, int *invalid,
                                                        int outer, int k, int ld, int kchunk) {
    const int o = blockIdx.x * 256 + threadIdx.x;
    if (o >= outer) {
        return;
    }
    const int k0 = blockIdx.y * kchunk, k1 = min(k, k0 + kchunk);
    unsigned long long best = 0;
    bool bad = false;
    for (int kk = k0; kk < k1; ++kk) {
        const unsigned long long raw =
            (unsigned long long)__double_as_longlong(input[size_t(kk) * ld + o]) &
            0x7fffffffffffffffull;
        bad |= (raw >> 52) == 2047;
        best = (raw >> 52) == 2047 ? best : (raw > best ? raw : best);
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    if (best) {
        atomicMax(largest + o, best);
    }
}

__global__ void shifts_finish(const unsigned long long *largest, int *shifts, int outer, int bits) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outer) {
        return;
    }
    shifts[o] = shift_of(largest[o], bits);
}

// B side: one block per column; the column's K values are contiguous, so the block reads them directly.
// outer is not read (the grid has one block per column); it keeps the kernel argument layout.
__global__ void shifts_b(const double *input, int *shifts, int *invalid, int outer, int k, int ld,
                         int bits) {
    __shared__ uint64_t maximum[256];
    const unsigned o = blockIdx.x, lane = threadIdx.x;
    uint64_t largest = 0;
    for (int kk = lane; kk < k; kk += 256) {
        const uint64_t raw =
            uint64_t(__double_as_longlong(input[size_t(o) * ld + kk])) & 0x7fffffffffffffffull;
        if ((raw >> 52) == 2047) {
            atomicExch(invalid, 1);
        } else if (raw > largest) {
            largest = raw;
        }
    }
    maximum[lane] = largest;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (lane < stride && maximum[lane + stride] > maximum[lane]) {
            maximum[lane] = maximum[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        shifts[o] = shift_of(maximum[0], bits);
    }
}
#endif

} // namespace crt
} // namespace awe_mxfp6
