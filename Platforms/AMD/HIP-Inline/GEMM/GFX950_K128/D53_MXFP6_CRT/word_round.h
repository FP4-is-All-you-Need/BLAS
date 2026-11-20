#pragma once
// Round a multi-word magnitude times 2^e to the nearest binary64 (ties to even), including the
// subnormal range and overflow to infinity. Works word by word so no intermediate FP64 rounding occurs.
#include "wide_integer.h"
#include <cstdint>
#include <cstring>

namespace awe_word_round {

template <unsigned N> AWE_MXFP6_HD inline int word_bit_length(awe_wide::Integer<N> a) {
    static_assert(N >= 1 && N <= 8, "word count 1..8");
    for (int i = int(N) - 1; i >= 0; --i) {
        if (a.w[i]) {
            static_assert(sizeof(unsigned int) == 4, "32-bit unsigned required");
            return 32 * i + 32 - __builtin_clz(static_cast<unsigned int>(a.w[i]));
        }
    }
    return 0;
}

// Bits [shift, shift + Kept] of a, rounded to nearest even on the bits below shift.
template <unsigned Kept, unsigned N>
AWE_MXFP6_HD inline uint64_t word_rounded_shift(awe_wide::Integer<N> a, int shift) {
    static_assert(N >= 1 && N <= 8, "word count 1..8");
    static_assert(Kept < 63, "extraction plus rounding fits uint64");
    uint64_t out = 0;
    const int64_t lo = shift, hi = lo + int64_t(Kept) + 1;
    for (unsigned j = 0; j < N; ++j) {
        const int64_t base = int64_t(j) * 32;
        const int64_t first = lo > base ? lo : base;
        const int64_t last = hi < base + 32 ? hi : base + 32;
        if (first >= last) {
            continue;
        }
        const unsigned width = unsigned(last - first);
        const uint32_t mask = width == 32 ? UINT32_MAX : ((uint32_t(1) << width) - 1);
        const uint64_t piece = (a.w[j] >> unsigned(first - base)) & mask;
        out |= piece << unsigned(first - lo);
    }
    if (lo > 0) {
        const int64_t g = lo - 1;
        bool guard = false, sticky = false;
        if (g < int64_t(N) * 32) {
            guard = ((a.w[unsigned(g) / 32] >> (unsigned(g) % 32)) & 1u) != 0;
        }
        for (unsigned j = 0; j < N; ++j) {
            const int64_t count = g - int64_t(j) * 32;
            if (count >= 32) {
                sticky |= a.w[j] != 0;
            } else if (count > 0) {
                sticky |= (a.w[j] & ((uint32_t(1) << unsigned(count)) - 1)) != 0;
            }
        }
        if (guard && (sticky || (out & 1))) {
            ++out;
        }
    }
    return out;
}

template <unsigned Fraction, unsigned Exponent, unsigned N>
AWE_MXFP6_HD inline uint64_t word_scaled_bits(awe_wide::Integer<N> a, bool negative, int exponent) {
    static_assert((Fraction == 52 && Exponent == 11) || (Fraction == 23 && Exponent == 8),
                  "binary64/binary32 only");
    constexpr int bias = (1 << (Exponent - 1)) - 1, precision = Fraction + 1;
    constexpr uint64_t fraction_mask = (uint64_t(1) << Fraction) - 1;
    constexpr uint64_t inf = ((uint64_t(1) << Exponent) - 1) << Fraction;
    const uint64_t sign = uint64_t(negative) << (Fraction + Exponent);
    const int length = word_bit_length(a);
    if (!length) {
        return sign;
    }
    int64_t e = int64_t(length) - 1 + int64_t(exponent);
    if (e > bias) {
        return sign | inf;
    }
    if (e < -bias - int(Fraction)) {
        return sign;
    }
    if (e < 1 - bias) {
        const int64_t distance = int64_t(1 - bias - int(Fraction)) - int64_t(exponent);
        return sign | word_rounded_shift<precision>(a, int(distance));
    }
    uint64_t significand = word_rounded_shift<precision>(a, length - precision);
    if (significand == (uint64_t(1) << precision)) {
        significand >>= 1;
        ++e;
    }
    if (e > bias) {
        return sign | inf;
    }
    return sign | (uint64_t(e + bias) << Fraction) | (significand & fraction_mask);
}

template <unsigned N>
AWE_MXFP6_HD inline double to_double(awe_wide::Integer<N> a, bool sign, int e) {
    const uint64_t raw = word_scaled_bits<52, 11>(a, sign, e);
    double value;
#if defined(__HIP_DEVICE_COMPILE__)
    value = __longlong_as_double(static_cast<long long>(raw));
#else
    std::memcpy(&value, &raw, sizeof(value));
#endif
    return value;
}

} // namespace awe_word_round
