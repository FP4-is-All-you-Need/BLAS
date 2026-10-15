#pragma once

#include <cstdint>
#ifndef CRTF_NMOD
#define CRTF_NMOD 13
#endif
using CrtU128 = __uint128_t;

struct CrtF32Const {
    unsigned w[CRTF_NMOD][3];
    unsigned long long m;
    CrtU128 P;
};

__host__ __device__ inline CrtU128 crtf_reduce(const unsigned (&r)[CRTF_NMOD],
                                               const CrtF32Const &c) {
    unsigned long long a0 = 0, a1 = 0, a2 = 0;
#pragma unroll
    for (int q = 0; q < CRTF_NMOD; ++q) {
        a0 += (unsigned long long)c.w[q][0] * r[q];
        a1 += (unsigned long long)c.w[q][1] * r[q];
        a2 += (unsigned long long)c.w[q][2] * r[q];
    }

    const unsigned long long c1 = a1 + (a0 >> 32);
    const unsigned long long lo = (a0 & 0xffffffffull) | (c1 << 32);
    const unsigned long long hi = a2 + (c1 >> 32);
    const CrtU128 s = ((CrtU128)hi << 64) | lo;

#ifdef __CUDA_ARCH__
    const unsigned long long lo_m_hi = __umul64hi(lo, c.m);
#else
    const unsigned long long lo_m_hi = (unsigned long long)(((CrtU128)lo * c.m) >> 64);
#endif
    const CrtU128 t = (CrtU128)hi * c.m + lo_m_hi;
    const CrtU128 qe = t >> 64;
    CrtU128 u = s - qe * c.P;
    if (u >= c.P)
        u -= c.P;
    if (u >= c.P)
        u -= c.P;
    return u;
}

__host__ __device__ inline unsigned crtf_round_bits(CrtU128 x, bool neg, int e) {
    if (!x)
        return 0u;
    const unsigned long long hi = (unsigned long long)(x >> 64), lo = (unsigned long long)x;
#ifdef __CUDA_ARCH__
    const int top = hi ? 127 - __clzll(hi) : 63 - __clzll(lo);
#else
    const int top = hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(lo);
#endif
    const int E = top + e;
    const unsigned sign = neg ? 0x80000000u : 0u;
    if (E > 127)
        return sign | 0x7f800000u;
    const int ulp = (E - 23 > -149) ? E - 23 : -149;
    const int drop = ulp - e;
    unsigned long long M;
    if (drop <= 0) {
        M = (unsigned long long)(x << (-drop));
    } else if (drop > top + 1) {
        M = 0;
    } else {
        const CrtU128 kept = x >> drop, rem = x - (kept << drop), half = (CrtU128)1 << (drop - 1);
        M = (unsigned long long)kept;
        if (rem > half || (rem == half && (M & 1)))
            ++M;
    }
    if (M < (1ull << 23))
        return sign | (unsigned)M;
    int ue = ulp;
    if (M == (1ull << 24)) {
        M >>= 1;
        ++ue;
    }
    const int biased = ue + 23 + 127;
    if (biased >= 255)
        return sign | 0x7f800000u;
    return sign | ((unsigned)biased << 23) | ((unsigned)M & 0x7fffffu);
}

__host__ __device__ inline float crtf_bits_to_float(unsigned b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(b);
#else
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
#endif
}

__host__ __device__ inline unsigned crtf_element_bits(const unsigned (&r)[CRTF_NMOD],
                                                      const CrtF32Const &c, int exponent) {
    const CrtU128 u = crtf_reduce(r, c);
    const bool neg = u > c.P / 2;
    return crtf_round_bits(neg ? c.P - u : u, neg, exponent);
}

#ifndef __CUDACC_RTC__

inline bool crtf_make_consts(const unsigned *p, const unsigned *lam0inv, CrtF32Const &c) {
    CrtU128 P = 1;
    for (int q = 0; q < CRTF_NMOD; ++q)
        P *= p[q];
    if (!(P > ((CrtU128)1 << 64) && P < ((CrtU128)1 << 66)))
        return false;
    c.P = P;
    c.m = (unsigned long long)(((CrtU128)0 - 1) / P);
    for (int q = 0; q < CRTF_NMOD; ++q) {
        const CrtU128 base = P / p[q];
        unsigned inv = 1;
        while ((unsigned)(base % p[q]) * inv % p[q] != 1)
            ++inv;
        inv = inv * lam0inv[q] % p[q];
        CrtU128 w = 0;
        for (unsigned i = 0; i < inv; ++i) {
            w += base;
            if (w >= P)
                w -= P;
        }
        for (int j = 0; j < 3; ++j)
            c.w[q][j] = (unsigned)(w >> (32 * j));
        if ((w >> 96) != 0)
            return false;
    }
    return true;
}
#endif
