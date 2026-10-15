#pragma once
// Small-modulus-product path; the existing RN32 path is unchanged.
// For 2^32 < P <= 2^64 use m=floor(2^96/P), q=floor(s*m/2^96).
// With <=64 moduli, residues <256 and W<P, s<2^78<2^96.
// q underestimates floor(s/P) by at most one; correction is exact.
// Above 2^64 retain the original 128-bit reciprocal reduction.
// Upstream SHA256: 30adba7cd9b09dc641cc50a6ac589080a01dd8d0efc316b0072c8cabf2485640
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

    unsigned long long wlo[CRTF_NMOD]; // W_q mod 2^64
    unsigned wfix[CRTF_NMOD];          // round(W_q 2^20 / P)
    unsigned long long plo;            // P mod 2^64
    unsigned
        guard; // >= max over residues of |sum r_q wfix_q - s 2^20 / P| (units of 2^-20), plus 1
    int fast_ready; // 1 when wfix, guard and the 32-bit sum bound hold for these moduli
    int fast;       // 1 only after crtf_set_fast (fast_ready and |value| < 2^63 proven)
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
__host__ __device__ inline unsigned crtf_element_bits_exact(const unsigned (&r)[CRTF_NMOD],
                                                            const CrtF32Const &c, int exponent) {
    const CrtU128 u = crtf_reduce(r, c);
    const bool neg = u > c.P / 2;
    return crtf_round_bits(neg ? c.P - u : u, neg, exponent);
}

#ifndef CRTF_USE_I2F
#define CRTF_USE_I2F 0
#endif
__host__ __device__ inline unsigned crtf_u63_bits(unsigned long long a, int exponent) {
#ifdef __CUDA_ARCH__
    const int lz = __clzll(a);
#else
    const int lz = __builtin_clzll(a);
#endif
    const int e0 = 63 - lz + 127 + exponent; // biased exponent of the leading bit
    if (e0 < 2 || e0 > 253) {
        return 0u;
    }
    const unsigned long long t = a << lz; // leading bit at 63
    const unsigned hi = (unsigned)(t >> 32), lo = (unsigned)t;
    const unsigned m = hi >> 8,
                   rest = hi & 0xFFu; // m: 24 bits with the leading bit; rest: guard bit and 7 more
    const unsigned up = (rest > 0x80u || (rest == 0x80u && (lo != 0u || (m & 1u)))) ? 1u : 0u;
    return ((unsigned)(e0 - 1) << 23) + m +
           up; // m + up = 2^24 carries into the exponent (at most 254)
}

__host__ __device__ inline unsigned crtf_element_bits(const unsigned (&r)[CRTF_NMOD],
                                                      const CrtF32Const &c, int exponent) {
    if (c.fast) {
        unsigned long long s = 0;
        unsigned f = 0;
#pragma unroll
        for (int q = 0; q < CRTF_NMOD; ++q) {
            s += c.wlo[q] * (unsigned long long)r[q];
            f += c.wfix[q] * r[q];
        }
        const unsigned frac = f & 0xFFFFFu, half = 1u << 19;
        const unsigned d = frac > half ? frac - half : half - frac;
        if (d > c.guard) {
            const unsigned long long n =
                (unsigned long long)(f >> 20) + (frac > half ? 1ull : 0ull);
            const long long y = (long long)(s - n * c.plo);
            if (y == 0) {
                return 0u;
            }
#if CRTF_USE_I2F
#ifdef __CUDA_ARCH__
            const unsigned b = __float_as_uint(__ll2float_rn(y));
#else
            const float v = (float)y; // x86-64 cvtsi2ss: round to nearest even
            unsigned b;
            __builtin_memcpy(&b, &v, 4);
#endif
            const int nb = int((b >> 23) & 0xFFu) + exponent;
            if (nb >= 2 && nb <= 253) {
                return unsigned(int(b) + exponent * (1 << 23));
            }
#else
            const unsigned long long a =
                y < 0 ? 0ull - (unsigned long long)y : (unsigned long long)y;
            const unsigned b = crtf_u63_bits(a, exponent);
            if (b) {
                return b | (y < 0 ? 0x80000000u : 0u);
            }
#endif
        }
    }
    return crtf_element_bits_exact(r, c, exponent);
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
    c.fast_ready = 0;
    c.fast = 0;
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

        c.wlo[q] = (unsigned long long)w;
        const CrtU128 num = (w << 20) + P / 2;
        const CrtU128 wf = num / P;
        c.wfix[q] = wf > ((CrtU128)1 << 20) ? 0u : (unsigned)wf;
        if (wf > ((CrtU128)1 << 20)) {
            c.fast_ready = -1; // never for w < P; kept as a refusal, not a failure
        }
    }
    c.plo = (unsigned long long)P;
    c.fast = 0;
    { // guard: sum (p_q - 1) |wfix_q P - w_q 2^20| / P, rounded up, plus 1; and the sum f must stay below 2^32
        CrtU128 err = 0, fmax = 0;
        for (int q = 0; q < CRTF_NMOD; ++q) {
            CrtU128 w = 0;
            for (int j = 2; j >= 0; --j) {
                w = (w << 32) | c.w[q][j];
            }
            const CrtU128 a = (CrtU128)c.wfix[q] * P, b = w << 20;
            err += (CrtU128)(p[q] - 1) * (a > b ? a - b : b - a);
            fmax += (CrtU128)(p[q] - 1) * c.wfix[q];
        }
        const CrtU128 g = (err + P - 1) / P + 1;
        // the fast path is refused (fast_ready 0), never the constants: every configuration that worked before still works
        const bool ok = c.fast_ready != -1 && fmax < ((CrtU128)1 << 32) && g < (1u << 18);
        c.guard = ok ? (unsigned)g : 0u;
        c.fast_ready = ok ? 1 : 0;
    }
    return true;
}

inline void crtf_set_fast(CrtF32Const &c, int bits, long long k) {
    int cl = 0;
    while ((1ll << cl) < k) {
        ++cl;
    }
    const bool small_p = c.P < ((CrtU128)1 << 64);
    c.fast = (c.fast_ready == 1 && (small_p || (bits >= 1 && 2 * bits + cl <= 62))) ? 1 : 0;
}
#endif
