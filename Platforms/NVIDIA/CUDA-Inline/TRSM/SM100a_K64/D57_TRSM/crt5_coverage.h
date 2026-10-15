#pragma once
#include <cstdint>

// Compare the five-word modulus product with 2*K*(2^bits-1)^2 without truncation.
inline bool crt5_covers(const unsigned p[5], unsigned bits, unsigned k) {
    if (!k || !bits || bits > 63) return false;
    using U128 = unsigned __int128;
    const U128 q = (U128(1) << bits) - 1;
    const U128 square = q * q;
    unsigned bound[5]{};
    uint64_t carry = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const U128 v = U128(unsigned(square >> (32 * i))) * (uint64_t(k) * 2) + carry;
        bound[i] = unsigned(v);
        carry = uint64_t(v >> 32);
    }
    if (carry > UINT32_MAX) return false;
    bound[4] = unsigned(carry);
    for (int i = 4; i >= 0; --i) {
        if (p[i] != bound[i]) return p[i] > bound[i];
    }
    return false;
}
