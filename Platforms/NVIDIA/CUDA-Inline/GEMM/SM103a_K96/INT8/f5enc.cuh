// Encode all signed INT8 values with weights (1,7,28) and six FP4 products.
// Each call allows FP32 accumulation only when K*tmax <= 2^24 or the product
// of the two maximum row energies is <= 2^48. Otherwise dp4a computes exact sums.
// The stricter 2^22 / 2^44 bounds allow the fast integer conversion.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include "enc1526.h"
namespace f5lut {
#include "lut1728.h"
}
namespace f5e {
constexpr int kCoef[6] = {1, 14, 56, 1, 4, 56}; // units 37/2 (faces 0-2) and 7/2 (faces 3-5)
constexpr unsigned kTmax0 = 974, kTmax1 = 1152;

inline unsigned code_of(int c) { // E2M1 code of c / 2; 0xFF when c / 2 is not an E2M1 value
    const int a = c < 0 ? -c : c;
    const int m = a <= 4 ? a : a == 6 ? 5 : a == 8 ? 6 : a == 12 ? 7 : -1;
    if (m < 0)
        return 0xFFu;
    return c == 0 ? 0u : unsigned(m) | (c < 0 ? 8u : 0u);
}
// t[256 side + u] for the byte u (value int8(u)): low word byte j = code(face 2j+1) << 4 | code(face 2j) (byte 3 = 0),
// high word = E0 | E1 << 16 (each < 4,096, so 16 of them add up without carrying into the other half).  tm = {tmax0,
// tmax1}.
inline bool build_table(unsigned long long *t, unsigned *tm) {
    for (int side = 0; side < 2; ++side) {
        const signed char (*L)[6] = side ? f5lut::LUT_B : f5lut::LUT_A;
        for (int u = 0; u < 256; ++u) {
            const signed char *f = L[int(int8_t(uint8_t(u))) + 128];
            unsigned lo = 0, e[2] = {0, 0};
            for (int j = 0; j < 6; ++j) {
                const unsigned c = code_of(f[j]);
                if (c == 0xFFu)
                    return false;
                lo |= c << (4 * j);
                e[j / 3] += unsigned(kCoef[j] * f[j] * f[j]);
            }
            if (e[0] >= 4096 || e[1] >= 4096)
                return false;
            t[256 * side + u] = (unsigned long long)lo | ((unsigned long long)(e[0] | e[1] << 16) << 32);
        }
    }
    unsigned mx[2] = {0, 0};
    for (int x = 0; x < 256; ++x)
        for (int y = 0; y < 256; ++y)
            for (int g = 0; g < 2; ++g) {
                unsigned s = 0;
                for (int j = 3 * g; j < 3 * g + 3; ++j)
                    s += unsigned(kCoef[j] * std::abs(int(f5lut::LUT_A[x][j])) * std::abs(int(f5lut::LUT_B[y][j])));
                mx[g] = std::max(mx[g], s);
            }
    tm[0] = mx[0];
    tm[1] = mx[1];
    return mx[0] == kTmax0 && mx[1] == kTmax1;
}

struct F5Side {
    const int8_t *raw;
    unsigned char *out;
    int rows, rows_pad, rt, ld, cuts, padded;
    unsigned *e0;
    unsigned *e1;
};
// One block task = one band of rt rows x 8 cuts; a warp task = 8 rows x 4 cuts (lane & 7 = row, lane >> 3 = cut), as
// encode_t3. padded is a multiple of 4 and c0 of 8, so the 4 cuts of a warp task are all below padded or all at / above
// it: the shuffles run with the whole warp.  side.padded = 0 skips that side (B reused).
template <int MINB, int ILP, bool STREAM>
__global__ void __launch_bounds__(256, MINB)
    encode_f5(F5Side a, F5Side b, const unsigned long long *__restrict__ lut_g) {
    __shared__ unsigned long long lut[512];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    lut[threadIdx.x + 256] = lut_g[threadIdx.x + 256];
    __syncthreads();
    const int cga = (a.padded + 7) / 8, cgb = (b.padded + 7) / 8;
    const long long ta = (long long)(a.rows_pad / a.rt) * cga, tb = (long long)(b.rows_pad / b.rt) * cgb;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bool odd = (lane >> 3) & 1;
    for (long long q = blockIdx.x; q < ta + tb; q += gridDim.x) {
        const bool isb = q >= ta;
        const F5Side &S = isb ? b : a;
        const long long qq = isb ? q - ta : q;
        const int cg = isb ? cgb : cga;
        const int band = int(qq / cg), c0 = int(qq % cg) * 8, ntask = (S.rt / 8) * 2;
        const unsigned long long *L = lut + (isb ? 256 : 0);
        for (int tau0 = w; tau0 < ntask; tau0 += 8 * ILP) {
            uint4 xs[ILP];
            int rs[ILP], cs[ILP];
#pragma unroll
            for (int i = 0; i < ILP; ++i) { // all loads of this round first
                const int tau = tau0 + 8 * i, rg = tau >> 1, ch = tau & 1;
                rs[i] = band * S.rt + rg * 8 + (lane & 7);
                cs[i] = c0 + ch * 4 + (lane >> 3);
                xs[i] = make_uint4(0, 0, 0, 0);
                if (tau < ntask && cs[i] < S.padded && rs[i] < S.rows && cs[i] < S.cuts)
                    xs[i] = __ldcs(reinterpret_cast<const uint4 *>(S.raw + size_t(rs[i]) * S.ld + size_t(cs[i]) * 16));
            }
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                const int tau = tau0 + 8 * i;
                if (tau >= ntask)
                    break; // the same for the whole warp
                const int rg = tau >> 1, r = rs[i], c = cs[i];
                if (c0 + (tau & 1) * 4 >= S.padded)
                    continue; // the whole warp (see above)
                const uint32_t w4[4] = {xs[i].x, xs[i].y, xs[i].z, xs[i].w};
                uint32_t ev[16], es = 0;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const unsigned long long t = L[(w4[e >> 2] >> (8 * (e & 3))) & 0xFFu];
                    ev[e] = uint32_t(t);
                    es += uint32_t(t >> 32);
                }
                uint32_t o0[6], o1[6];
                enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[0]), o0);
                enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[8]), o1);
                uint32_t x0[3], x1[3]; // the even cut's lane gets faces 0-2 of the odd cut; the odd cut's lane gets
                                       // faces 3-5 of the even cut
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    x0[j] = __shfl_xor_sync(0xFFFFFFFFu, odd ? o0[j] : o0[3 + j], 8);
                    x1[j] = __shfl_xor_sync(0xFFFFFFFFu, odd ? o1[j] : o1[3 + j], 8);
                }
                uint4 v0, v1, v2;
                if (!odd) {
                    v0 = make_uint4(o0[0], o1[0], o0[1], o1[1]);
                    v1 = make_uint4(o0[2], o1[2], x0[0], x1[0]);
                    v2 = make_uint4(x0[1], x1[1], x0[2], x1[2]);
                } // D0: own f0 f1 f2, odd's f0 f1 f2
                else {
                    v0 = make_uint4(x0[0], x1[0], x0[1], x1[1]);
                    v1 = make_uint4(x0[2], x1[2], o0[3], o1[3]);
                    v2 = make_uint4(o0[4], o1[4], o0[5], o1[5]);
                } // D1: even's f3 f4 f5, own f3 f4 f5
                uint4 *d = reinterpret_cast<uint4 *>(S.out + size_t(band) * S.padded * S.rt * 48 +
                                                     size_t(c) * S.rt * 48 + size_t(rg) * 384 + size_t(lane & 7) * 16);
                if (STREAM) {
                    __stcs(d, v0);
                    __stcs(d + 8, v1);
                    __stcs(d + 16, v2);
                } else {
                    d[0] = v0;
                    d[8] = v1;
                    d[16] = v2;
                }
                if (S.e0) { // this (row, cut)'s E0 / E1 (16 values, each half < 65,536), the 4 cuts of the row added,
                            // one atomic per row and group
                    unsigned s0 = es & 0xFFFFu, s1 = es >> 16;
                    s0 += __shfl_xor_sync(0xFFFFFFFFu, s0, 8);
                    s0 += __shfl_xor_sync(0xFFFFFFFFu, s0, 16);
                    s1 += __shfl_xor_sync(0xFFFFFFFFu, s1, 8);
                    s1 += __shfl_xor_sync(0xFFFFFFFFu, s1, 16);
                    if ((lane >> 3) == 0) {
                        if (s0)
                            atomicAdd(S.e0 + r, s0);
                        if (s1)
                            atomicAdd(S.e1 + r, s1);
                    }
                }
            }
        }
    }
}
// gate[0]: bit 0 / 1 = fast conversion of D0 / D1, bit 2 = both groups exact (the kernel runs); fl[0] = 1 when not
// (exact_fallback writes C), fl[1] = 0; st[0..4] = max rows A E0, E1, max cols B E0, E1, gate[0] (hayashiGetLastF5).
__global__ void __launch_bounds__(1024)
    f5_gate(const unsigned *__restrict__ e0a, const unsigned *__restrict__ e1a, int na,
            const unsigned *__restrict__ e0b, const unsigned *__restrict__ e1b, int nb, int k, unsigned tmax0,
            unsigned tmax1, unsigned *gate, unsigned long long *fl, unsigned *st) {
    __shared__ unsigned sh[4][32];
    unsigned v[4] = {0, 0, 0, 0};
    for (int i = threadIdx.x; i < na; i += 1024) {
        v[0] = max(v[0], e0a[i]);
        v[1] = max(v[1], e1a[i]);
    }
    for (int i = threadIdx.x; i < nb; i += 1024) {
        v[2] = max(v[2], e0b[i]);
        v[3] = max(v[3], e1b[i]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        for (int o = 16; o; o >>= 1)
            v[j] = max(v[j], __shfl_xor_sync(0xFFFFFFFFu, v[j], o));
    if ((threadIdx.x & 31) == 0)
        for (int j = 0; j < 4; ++j)
            sh[j][threadIdx.x >> 5] = v[j];
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = sh[j][threadIdx.x];
            for (int o = 16; o; o >>= 1)
                v[j] = max(v[j], __shfl_xor_sync(0xFFFFFFFFu, v[j], o));
        }
        if (threadIdx.x == 0) {
            unsigned g = 4u;
            for (int j = 0; j < 2; ++j) {
                const unsigned long long bt = (unsigned long long)k * (j ? tmax1 : tmax0),
                                         pr = (unsigned long long)v[j] * v[2 + j]; // < 2^64: each max < 2^32
                const bool hi = __umul64hi((unsigned long long)v[j], (unsigned long long)v[2 + j]) !=
                                0ull; // (never with k <= 16,384)
                const bool exact = bt <= (1ull << 24) || (!hi && pr <= (1ull << 48)),
                           fast = bt <= (1ull << 22) || (!hi && pr <= (1ull << 44));
                if (!exact)
                    g &= ~4u;
                if (fast)
                    g |= 1u << j;
            }
            gate[0] = g;
            fl[0] = (g & 4u) ? 0ull : 1ull;
            fl[1] = 0ull;
            st[0] = v[0];
            st[1] = v[1];
            st[2] = v[2];
            st[3] = v[3];
            st[4] = g;
        }
    }
}
} // namespace f5e
