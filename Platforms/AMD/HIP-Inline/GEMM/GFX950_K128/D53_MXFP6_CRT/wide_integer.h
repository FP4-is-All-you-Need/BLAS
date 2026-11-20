#pragma once
// Unsigned multi-word integers for the CRT: the 19-modulus product needs 121 bits, so 4 words hold it
// and 5 words hold the sum form used by the reconstruction kernel.
#include "common.h"
#include <cstdint>
#include <cstring>

namespace awe_wide {

template <unsigned Words> struct Integer {
    static_assert(Words >= 1 && Words <= 8, "supported arithmetic width is 32..256 bits");
    uint32_t w[Words]{};
};

template <unsigned N> AWE_MXFP6_HD inline int compare(Integer<N> a, Integer<N> b) {
    for (int i = int(N) - 1; i >= 0; --i) {
        if (a.w[i] != b.w[i]) {
            return a.w[i] > b.w[i] ? 1 : -1;
        }
    }
    return 0;
}

// Caller proves a >= b.
template <unsigned N> AWE_MXFP6_HD inline Integer<N> subtract(Integer<N> a, Integer<N> b) {
    Integer<N> out;
    uint64_t borrow = 0;
    for (unsigned i = 0; i < N; ++i) {
        const uint64_t rhs = uint64_t(b.w[i]) + borrow;
        out.w[i] = uint32_t(uint64_t(a.w[i]) - rhs);
        borrow = uint64_t(a.w[i]) < rhs;
    }
    return out;
}

template <unsigned N> AWE_MXFP6_HD inline Integer<N> half(Integer<N> a) {
    Integer<N> out;
    for (unsigned i = 0; i < N; ++i) {
        out.w[i] = (a.w[i] >> 1) | (i + 1 < N ? a.w[i + 1] << 31 : 0u);
    }
    return out;
}

// p is a nonzero 32-bit value, so each 64-bit partial dividend fits.
template <unsigned N> AWE_MXFP6_HD inline uint32_t remainder(Integer<N> a, uint32_t p) {
    uint64_t r = 0;
    for (int i = int(N) - 1; i >= 0; --i) {
        r = ((r << 32) | a.w[i]) % p;
    }
    return uint32_t(r);
}

// a + b * digit. Caller proves the result fits in 32 * N bits.
template <unsigned N>
AWE_MXFP6_HD inline Integer<N> add_multiply(Integer<N> a, Integer<N> b, uint32_t digit) {
    uint64_t carry = 0;
    for (unsigned i = 0; i < N; ++i) {
        const uint64_t v = uint64_t(b.w[i]) * digit + a.w[i] + carry;
        a.w[i] = uint32_t(v);
        carry = v >> 32;
    }
    return a;
}

template <unsigned N> AWE_MXFP6_HD inline bool bit(Integer<N> a, int i) {
    return i >= 0 && i < int(32 * N) && ((a.w[i / 32] >> (i % 32)) & 1u);
}

template <unsigned N> AWE_MXFP6_HD inline int bit_length(Integer<N> a) {
    for (int i = int(N) - 1; i >= 0; --i) {
        if (a.w[i]) {
            int length = 0;
            uint32_t v = a.w[i];
            while (v) {
                ++length;
                v >>= 1;
            }
            return 32 * i + length;
        }
    }
    return 0;
}

} // namespace awe_wide
