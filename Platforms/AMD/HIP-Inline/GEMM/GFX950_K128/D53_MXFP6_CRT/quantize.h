#pragma once
// Input quantization: x -> RNE(x * 2^shift) as a signed 96-bit integer, and its residue modulo p.
// The row (A) or column (B) shift is 52 minus the exponent of the largest magnitude, so every
// quantized value fits in 53 bits and the dot product of two such rows is exact in the CRT range.
#include "common.h"
#include <cstdint>

namespace awe_mxfp6 {

struct Prepared {
    uint32_t words[3]{};
    bool negative = false;
    bool valid = false;
};

struct Residue {
    uint32_t residue = 0;
    bool valid = false;
};

// Nonfinite input or a magnitude above 96 bits returns valid = false. Both signed zeros keep the sign.
AWE_MXFP6_HD inline Prepared prepare(uint64_t raw, int shift) {
    Prepared out;
    out.negative = (raw >> 63) != 0;
    const unsigned encoded = unsigned((raw >> 52) & 2047);
    if (encoded == 2047) {
        return out;
    }
    uint64_t mantissa = raw & 0x000fffffffffffffull;
    if (encoded) {
        mantissa |= uint64_t(1) << 52;
    }
    if (!mantissa) {
        out.valid = true;
        return out;
    }
    const int exponent = encoded ? int(encoded) - 1075 : -1074;
    const int64_t h = int64_t(exponent) + int64_t(shift);
    if (h >= 0) {
        static_assert(sizeof(unsigned long long) == 8, "64-bit clz input");
        const int length = 64 - __builtin_clzll(static_cast<unsigned long long>(mantissa));
        if (h > 96 - length) {
            return out;
        }
        const unsigned whole = unsigned(h) / 32, offset = unsigned(h) % 32;
        for (unsigned j = 0; j < 2; ++j) {
            const uint32_t part = uint32_t(mantissa >> (32 * j));
            const unsigned target = whole + j;
            if (target < 3) {
                out.words[target] |= part << offset;
            }
            if (offset && target + 1 < 3) {
                out.words[target + 1] |= part >> (32 - offset);
            }
        }
    } else {
        // The binary64 exponent plus an int shift cannot reach INT64_MIN.
        const int64_t distance = -h;
        uint64_t integer = 0;
        if (distance < 54) {
            const unsigned d = unsigned(distance);
            integer = mantissa >> d;
            const uint64_t remainder = mantissa & ((uint64_t(1) << d) - 1), half = uint64_t(1)
                                                                                   << (d - 1);
            if (remainder > half || (remainder == half && (integer & 1))) {
                ++integer;
            }
        }
        out.words[0] = uint32_t(integer);
        out.words[1] = uint32_t(integer >> 32);
    }
    out.valid = true;
    return out;
}

// Residue of the prepared integer modulo P with 32-bit arithmetic only, no branch. Each word is reduced
// first, so r1 * R + r0 <= (P - 1) * P < 2^32 and (t mod P) + r2 * R2 < 2^32 for P <= 65535.
template <uint32_t P> AWE_MXFP6_HD inline Residue residue(const Prepared &q) {
    static_assert(P >= 2 && P <= 65535, "residue modulus must be 2..65535");
    constexpr uint32_t R = uint32_t((uint64_t(1) << 32) % P);
    constexpr uint32_t R2 = uint32_t(uint64_t(R) * R % P);
    uint32_t t = q.words[1] % P * R + q.words[0] % P;
    t = t % P + q.words[2] % P * R2;
    t %= P;
    t = q.negative && t ? P - t : t;
    return {q.valid ? t : 0u, q.valid};
}

// Same value as residue<P> for a runtime modulus; used by the host test, not by the kernels.
AWE_MXFP6_HD inline Residue residue_runtime(const Prepared &q, uint32_t p) {
    if (!q.valid || p < 2 || p > 65535) {
        return {};
    }
    const uint32_t r = uint32_t((uint64_t(1) << 32) % p), r2 = uint32_t(uint64_t(r) * r % p);
    const uint64_t sum =
        uint64_t(q.words[0]) + uint64_t(q.words[1]) * r + uint64_t(q.words[2]) * r2;
    uint32_t result = uint32_t(sum % uint64_t(p));
    if (q.negative && result) {
        result = p - result;
    }
    return {result, true};
}

#if defined(__HIPCC__)
AWE_MXFP6_HD inline uint64_t raw_binary64(double value) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __double_as_longlong(value);
#else
    uint64_t r;
    __builtin_memcpy(&r, &value, 8);
    return r;
#endif
}
#else
inline uint64_t raw_binary64(double value) {
    uint64_t r;
    __builtin_memcpy(&r, &value, 8);
    return r;
}
#endif

} // namespace awe_mxfp6
