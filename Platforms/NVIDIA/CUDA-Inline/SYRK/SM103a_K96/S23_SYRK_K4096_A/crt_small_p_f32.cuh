#pragma once

#include <cstdint>
#ifndef CRTF_NMOD
#define CRTF_NMOD 13
#endif
using CrtU128 = __uint128_t;

struct CrtF32Const {
    unsigned w[CRTF_NMOD][3]; // W_q in three 32-bit words (W_q < 2^66 < 2^96)
    unsigned long long m;     // floor(2^128 / P)
    CrtU128 P;
    unsigned quotient_shift; // 32 for P <= 2^64; 64 for the original range
};

// s mod P from the residues (the payout stores lambda0 R; lambda0^-1 is in W_q)
__host__ __device__ inline CrtU128 crtf_reduce(const unsigned (&r)[CRTF_NMOD],
                                               const CrtF32Const &c) {
    unsigned long long a0 = 0, a1 = 0, a2 = 0;
#pragma unroll
    for (int q = 0; q < CRTF_NMOD; ++q) {
        a0 += (unsigned long long)c.w[q][0] * r[q];
        a1 += (unsigned long long)c.w[q][1] * r[q];
        a2 += (unsigned long long)c.w[q][2] * r[q];
    }
    // a0, a1, a2 < 13 x 2^32 x 2^8 < 2^44: carry into a 128-bit s (bits 0..31 from a0, 32..63 from a1 + carry, 64.. from a2 + carry)
    const unsigned long long c1 = a1 + (a0 >> 32);
    const unsigned long long lo = (a0 & 0xffffffffull) | (c1 << 32);
    const unsigned long long hi = a2 + (c1 >> 32);
    const CrtU128 s = ((CrtU128)hi << 64) | lo;
    // q = floor(s m / 2^128): s m = hi m 2^64 + lo m
#ifdef __CUDA_ARCH__
    const unsigned long long lo_m_hi = __umul64hi(lo, c.m);
#else
    const unsigned long long lo_m_hi = (unsigned long long)(((CrtU128)lo * c.m) >> 64);
#endif
    const CrtU128 t = (CrtU128)hi * c.m + lo_m_hi;
    const CrtU128 qe = t >> c.quotient_shift;
    CrtU128 u = s - qe * c.P;
    if (u >= c.P) {
        u -= c.P;
    }
    if (u >= c.P) {
        u -= c.P;
    }
    return u;
}

// round(x 2^e) to binary32 bits, x >= 0 an integer < 2^127, nearest-even; neg sets the sign (x = 0 gives +0)
__host__ __device__ inline unsigned crtf_round_bits(CrtU128 x, bool neg, int e) {
    if (!x) {
        return 0u;
    }
    const unsigned long long hi = (unsigned long long)(x >> 64), lo = (unsigned long long)x;
#ifdef __CUDA_ARCH__
    const int top = hi ? 127 - __clzll(hi) : 63 - __clzll(lo);
#else
    const int top = hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(lo);
#endif
    const int E = top + e; // the value is in [2^E, 2^(E+1))
    const unsigned sign = neg ? 0x80000000u : 0u;
    if (E > 127) {
        return sign | 0x7f800000u; // at or above 2^128: infinity (rounding cannot bring it back)
    }
    const int ulp =
        (E - 23 > -149) ? E - 23 : -149; // the exponent of the last kept bit (subnormals: 2^-149)
    const int drop = ulp - e;            // bits of x below the last kept bit
    unsigned long long M;                // the rounded magnitude in units of 2^ulp (<= 2^24)
    if (drop <= 0) {
        M = (unsigned long long)(x << (-drop)); // exact (x 2^-drop < 2^24 here)
    } else if (drop > top + 1) {
        M = 0; // below half of the smallest unit
    } else {
        const CrtU128 kept = x >> drop, rem = x - (kept << drop), half = (CrtU128)1 << (drop - 1);
        M = (unsigned long long)kept;
        if (rem > half || (rem == half && (M & 1))) {
            ++M;
        }
    }
    if (M < (1ull << 23)) {
        return sign | (unsigned)M; // subnormal (or zero)
    }
    int ue = ulp;
    if (M == (1ull << 24)) {
        M >>= 1;
        ++ue;
    } // rounding carried into the next binade
    const int biased = ue + 23 + 127;
    if (biased >= 255) {
        return sign | 0x7f800000u;
    }
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

// the whole element: residues -> signed integer -> float C value
__host__ __device__ inline unsigned crtf_element_bits(const unsigned (&r)[CRTF_NMOD],
                                                      const CrtF32Const &c, int exponent) {
    const CrtU128 u = crtf_reduce(r, c);
    const bool neg = u > c.P / 2;
    return crtf_round_bits(neg ? c.P - u : u, neg, exponent);
}

#ifndef __CUDACC_RTC__
// host: the constants (as l8rt_plan.cuh's W_q) for moduli p[] and the table's lambda0^-1 (lam0inv[]); false if P is out of range
inline bool crtf_make_consts(const unsigned *p, const unsigned *lam0inv, CrtF32Const &c) {
    CrtU128 P = 1;
    static_assert(CRTF_NMOD <= 64, "small-P bound needs at most 64 moduli");
    for (int q = 0; q < CRTF_NMOD; ++q) {
        if (p[q] < 2 || p[q] > 255 || !lam0inv[q] || lam0inv[q] >= p[q]) {
            return false;
        }
        for (int j = 0; j < q; ++j) {
            unsigned a = p[q], b = p[j];
            while (b) {
                unsigned z = a % b;
                a = b;
                b = z;
            }
            if (a != 1) {
                return false;
            }
        }
        P *= p[q];
        if (P >= ((CrtU128)1 << 66)) {
            return false;
        }
    }
    if (!(P > ((CrtU128)1 << 32) && P < ((CrtU128)1 << 66))) {
        return false;
    }
    c.quotient_shift = P <= ((CrtU128)1 << 64) ? 32 : 64;
    c.P = P;
    c.m =
        c.quotient_shift == 32
            ? (unsigned long long)(((CrtU128)1 << 96) / P)
            : (unsigned long long)(((CrtU128)0 - 1) /
                                   P); // floor((2^128 - 1) / P) = floor(2^128 / P) (P is not a power of 2)
    for (int q = 0; q < CRTF_NMOD; ++q) {
        const CrtU128 base = P / p[q];
        unsigned inv = 1;
        while ((unsigned)(base % p[q]) * inv % p[q] != 1) {
            ++inv;
        }
        inv = inv * lam0inv[q] % p[q];
        CrtU128 w = 0;
        for (unsigned i = 0; i < inv; ++i) {
            w += base;
            if (w >= P) {
                w -= P;
            }
        }
        for (int j = 0; j < 3; ++j) {
            c.w[q][j] = (unsigned)(w >> (32 * j));
        }
        if ((w >> 96) != 0) {
            return false;
        }
    }
    return true;
}
#endif
