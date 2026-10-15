#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "v5enc.cuh"
namespace v5f {
typedef unsigned long long u64;
constexpr int GroupsPerWin = 16, Threads = 128, PartWords = 8, GatherMax = 512;
__device__ u64 g_tab[2][256];                // by the unsigned byte b (value (int8)b)
__device__ signed char g_near[256];          // by the unsigned byte: the nearest covered value
__device__ u64 g_dig[EXT5_HI - EXT5_LO + 1]; // digit j's code in byte j (j = 0..5)
__device__ unsigned short g_vm[256];

inline unsigned code4h(int c) {
    const int a = c < 0 ? -c : c;
    const unsigned m = a <= 4 ? unsigned(a) : (a == 6 ? 5u : (a == 8 ? 6u : 7u));
    return c == 0 ? 0u : (m | (c < 0 ? 8u : 0u));
}
inline cudaError_t init_tables() {
    u64 tab[2][256];
    signed char nearv[256];
    static u64 dig[EXT5_HI - EXT5_LO + 1];
    unsigned short vm[256];
    const int S15[15] = {0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12};
    const int lam_abs[5] = {120, 120, 24, 40, 15};
    for (int f = 0; f < 5; ++f)
        if (LUT5_1_10_13[128][f] != 0)
            return cudaErrorInvalidValue; // the faces of 0
    for (int j = 0; j < 6; ++j)
        if (EXT5_DIGITS[0 - EXT5_LO][j] != 0)
            return cudaErrorInvalidValue; // the digits of 0
    if (EXT5_LO != -16 * 128 || EXT5_HI != 16 * 127)
        return cudaErrorInvalidValue;
    for (int b = 0; b < 256; ++b) {
        const int x = (signed char)b, xi = x + 128;
        const bool cov = LUT5_1_10_13_COVERED[xi] != 0;
        long long e = 0;
        for (int f = 0; f < 5; ++f)
            e += (long long)lam_abs[f] * LUT5_1_10_13[xi][f] * LUT5_1_10_13[xi][f];
        if (e >= (1 << 15))
            return cudaErrorInvalidValue; // E in 15 bits (the maximum is 22,311)
        for (int side = 0; side < 2; ++side) {
            unsigned lo = 0, hi = 0;
            for (int f = 0; f < 5; ++f) {
                const int c = side == 0 && LUT5_1_10_13_LAMBDA[f] < 0 ? -LUT5_1_10_13[xi][f] : LUT5_1_10_13[xi][f];
                const unsigned cd = code4h(c);
                if (f < 4)
                    lo |= cd << (8 * f);
                else
                    hi |= cd;
            }
            if (cov)
                hi |= unsigned(e) << 8;
            else
                hi |= 1u << 31;
            tab[side][b] = u64(lo) | (u64(hi) << 32);
        }
        int best = 0, bd = 1 << 30;
        for (int v = -128; v < 128; ++v)
            if (LUT5_1_10_13_COVERED[v + 128]) {
                const int dd = v > x ? v - x : x - v;
                if (dd < bd) {
                    bd = dd;
                    best = v;
                }
            }
        nearv[b] = (signed char)best;
        unsigned m = 0;
        for (int t = 0; t < 15; ++t) {
            const int y = x - S15[t];
            if (y >= -128 && y <= 127 && LUT5_1_10_13_COVERED[y + 128])
                m |= 1u << t;
        }
        vm[b] = (unsigned short)m;
    }
    for (int s = EXT5_LO; s <= EXT5_HI; ++s) {
        u64 d = 0;
        for (int j = 0; j < 6; ++j)
            d |= u64(code4h(EXT5_DIGITS[s - EXT5_LO][j])) << (8 * j);
        dig[s - EXT5_LO] = d;
    }
    cudaError_t e;
    if ((e = cudaMemcpyToSymbol(g_tab, tab, sizeof tab)) != cudaSuccess)
        return e;
    if ((e = cudaMemcpyToSymbol(g_near, nearv, sizeof nearv)) != cudaSuccess)
        return e;
    if ((e = cudaMemcpyToSymbol(g_vm, vm, sizeof vm)) != cudaSuccess)
        return e;
    return cudaMemcpyToSymbol(g_dig, dig, sizeof dig);
}

struct EncArgs {
    const int8_t *raw[2];
    long long ld[2];
    int rows[2], rows_true[2], vec[2]; // vec: 2 = 16-byte loads, 1 = 4-byte, 0 = bytes
    int kr, ng, gx, ya;                // kr: the raw k; ng = Kp / 96 groups; gx = windows (grid.x); ya = A's CTAs in y
    unsigned char *faces[2];
    int stride_cuts; // the padded cut count
    unsigned char *ext[2];
    int xub, xu;
    u64 *sh8[2];  // rows x ng
    int *wcnt[2]; // rows x gx: the leftovers of (row, window)
    unsigned *kmask;
    const int *kbase;
    int8_t *xk;
    unsigned *kmaskA;
    int8_t *xt;
    int part_base;
    int *rowtot[2];
    unsigned *part; // per CTA PartWords: max |shift|, blocks shifted, leftovers, 0, raw +-103, the window maximum of E
};

__device__ __forceinline__ void tr4(unsigned a, unsigned b, unsigned c, unsigned d, unsigned &o0, unsigned &o1,
                                    unsigned &o2, unsigned &o3) {
    const unsigned ablo = __byte_perm(a, b, 0x5140), abhi = __byte_perm(a, b, 0x7362), cdlo = __byte_perm(c, d, 0x5140),
                   cdhi = __byte_perm(c, d, 0x7362);
    o0 = __byte_perm(ablo, cdlo, 0x5410);
    o1 = __byte_perm(ablo, cdlo, 0x7632);
    o2 = __byte_perm(abhi, cdhi, 0x5410);
    o3 = __byte_perm(abhi, cdhi, 0x7632);
}
__device__ __forceinline__ unsigned g4(unsigned a, unsigned b, unsigned c, unsigned d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}
__device__ __forceinline__ int dsum(unsigned q0, unsigned q1, unsigned q2, unsigned q3) {
    return __dp4a(int(q0), 0x01010101,
                  __dp4a(int(q1), 0x01010101, __dp4a(int(q2), 0x01010101, __dp4a(int(q3), 0x01010101, 0))));
}

__device__ __forceinline__ int s15_of(int t) {
    const int m = int((0xC8643210u >> (4 * ((t + 1) >> 1))) & 15u);
    return (t & 1) ? m : -m;
}
__device__ __forceinline__ unsigned n103_of(unsigned q0, unsigned q1, unsigned q2,
                                            unsigned q3) { // the values +-103 among 16 (0xFF a match)
    return (__popc(__vcmpeq4(q0, 0x67676767u)) + __popc(__vcmpeq4(q0, 0x99999999u)) +
            __popc(__vcmpeq4(q1, 0x67676767u)) + __popc(__vcmpeq4(q1, 0x99999999u)) +
            __popc(__vcmpeq4(q2, 0x67676767u)) + __popc(__vcmpeq4(q2, 0x99999999u)) +
            __popc(__vcmpeq4(q3, 0x67676767u)) + __popc(__vcmpeq4(q3, 0x99999999u))) >>
           3;
}

// the lookups of one block: L[i] / H[i] = the codes of values 2i, 2i + 1 (faces 0..3 / face 4 in byte 0), es = sum of
// the hi words, ho = their or
template <int STR = 1>
__device__ __forceinline__ void look(const u64 *tab, unsigned q0, unsigned q1, unsigned q2, unsigned q3,
                                     unsigned (&L)[8], unsigned (&H)[8], unsigned &es, unsigned &ho) {
    const unsigned q[4] = {q0, q1, q2, q3};
    es = 0;
    ho = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned w = q[i >> 1], s = 16 * (i & 1);
        const u64 e0 = tab[((w >> s) & 255) * STR], e1 = tab[((w >> (s + 8)) & 255) * STR];
        const unsigned l0 = unsigned(e0), h0 = unsigned(e0 >> 32), l1 = unsigned(e1), h1 = unsigned(e1 >> 32);
        L[i] = l0 + 16 * l1;
        H[i] = h0 + 16 * h1;
        es += h0 + h1;
        ho |= h0 | h1;
    }
}

__device__ __forceinline__ void push(unsigned &lo, unsigned &hi, unsigned src, unsigned sel) {
    lo = __funnelshift_r(lo, hi, 8);
    hi = __byte_perm(hi, src, sel);
}

#ifndef V5F_PRIV_VM
#define V5F_PRIV_VM 0
#endif
#ifndef V5F_PRIV_TAB
#define V5F_PRIV_TAB 0
#endif
#ifndef V5F_XT4
#define V5F_XT4 0
#endif
constexpr int DynSmem = V5F_PRIV_TAB ? 256 * 16 * 8 : 0;
#ifdef V5F_MINB
__global__ void __launch_bounds__(Threads, V5F_MINB) encode_fused(const EncArgs a) {
#else
__global__ void __launch_bounds__(Threads) encode_fused(const EncArgs a) {
#endif
    __shared__ signed char nearv[256];
    __shared__ unsigned red[Threads / 32][5];
#if V5F_PRIV_VM
    __shared__ unsigned vmp[256 * 32];
#else
    __shared__ unsigned short vmt[256];
#endif
#if V5F_PRIV_TAB
    extern __shared__ u64 tab16[];
#else
    __shared__ u64 tab[256];
#endif
    const int side = blockIdx.y >= unsigned(a.ya) ? 1 : 0, yb = int(blockIdx.y) - (side ? a.ya : 0);
    for (int i = threadIdx.x; i < 256; i += Threads) {
        nearv[i] = g_near[i];
#if !V5F_PRIV_VM
        vmt[i] = g_vm[i];
#endif
#if !V5F_PRIV_TAB
        tab[i] = g_tab[side][i];
#endif
    }
#if V5F_PRIV_VM
    for (int i = threadIdx.x; i < 256 * 32; i += Threads)
        vmp[i] = g_vm[i >> 5];
#endif
#if V5F_PRIV_TAB
    for (int i = threadIdx.x; i < 256 * 16; i += Threads)
        tab16[i] = g_tab[side][i >> 4];
    const u64 *tabl = tab16 + (threadIdx.x & 15);
#define V5F_LOOK look<16>(tabl,
#else
#define V5F_LOOK look<1>(tab,
#endif
#if V5F_PRIV_VM
#define V5F_VM(b) vmp[((b) << 5) | (threadIdx.x & 31)]
#else
#define V5F_VM(b) vmt[b]
#endif
    __syncthreads();
    const int r = yb * Threads + threadIdx.x;
    unsigned mx = 0, nsh = 0, nlt = 0, n103 = 0, ewin = 0;
    if (r < a.rows[side]) {
        const int R = side ? v5::HalfN : v5::CtaRows, cutb = R * 48, tile = r / R, row = r - tile * R;
        const size_t cell = size_t(row >> 3) * 384 + (row & 7) * 16;
        unsigned char *fbase = a.faces[side] + size_t(tile) * a.stride_cuts * cutb + cell;
        unsigned char *xbase = a.ext[side] + size_t(tile) * a.xu * cutb + cell;
        const bool real = r < a.rows_true[side];
        const int8_t *src = a.raw[side] + size_t(real ? r : 0) * a.ld[side];
        const int vec = a.vec[side], kr = a.kr;
        const int g0 = blockIdx.x * GroupsPerWin, g1 = min(a.ng, g0 + GroupsPerWin);
        unsigned dl[6] = {0, 0, 0, 0, 0, 0}, dh[6] = {0, 0, 0, 0, 0, 0}, sl = 0, shv = 0;
        auto store_unit = [&](int u) {
            const int du = side ? a.xub + u : u, su = side ? u : a.xub + u;
            unsigned char *pd = xbase + size_t(du) * cutb;
            unsigned char *ps = xbase + size_t(su) * cutb;
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                *reinterpret_cast<uint4 *>(pd + 128 * c) =
                    make_uint4(dl[2 * c], dh[2 * c], dl[2 * c + 1], dh[2 * c + 1]);
                *reinterpret_cast<uint4 *>(ps + 128 * c) = make_uint4(sl, shv, sl, shv);
            }
        };
        for (int g = g0; g < g1; ++g) {
            unsigned w[24];
            const int k0 = 96 * g;
            if (!real) {
#pragma unroll
                for (int i = 0; i < 24; ++i)
                    w[i] = 0;
            } else if (vec == 2 && k0 + 96 <= kr) {
                const uint4 *p = reinterpret_cast<const uint4 *>(src + k0);
#pragma unroll
                for (int i = 0; i < 6; ++i) {
                    const uint4 t = __ldg(p + i);
                    w[4 * i] = t.x;
                    w[4 * i + 1] = t.y;
                    w[4 * i + 2] = t.z;
                    w[4 * i + 3] = t.w;
                }
            } else if (vec == 1 && k0 + 96 <= kr) {
                const unsigned *p = reinterpret_cast<const unsigned *>(src + k0);
#pragma unroll
                for (int i = 0; i < 24; ++i)
                    w[i] = __ldg(p + i);
            } else {
#pragma unroll
                for (int i = 0; i < 24; ++i) {
                    unsigned t = 0;
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int k = k0 + 4 * i + b;
                        if (k < kr)
                            t |= unsigned((unsigned char)src[k]) << (8 * b);
                    }
                    w[i] = t;
                }
            }
            if (side == 0 && a.xt) {
#if V5F_XT4
                const int lq = threadIdx.x & 3, lb = (threadIdx.x & 31) & ~3;
                const unsigned sel = unsigned(lq) | (unsigned(4 + lq) << 4);
                unsigned *xp = reinterpret_cast<unsigned *>(a.xt + (size_t(k0) + lq) * a.rows[0] + (r & ~3));
                const size_t xs = size_t(a.rows[0]); // in bytes; a word per 4 rows
#pragma unroll
                for (int i = 0; i < 24; ++i) {
                    const unsigned v0 = __shfl_sync(0xFFFFFFFFu, w[i], lb), v1 = __shfl_sync(0xFFFFFFFFu, w[i], lb + 1),
                                   v2 = __shfl_sync(0xFFFFFFFFu, w[i], lb + 2),
                                   v3 = __shfl_sync(0xFFFFFFFFu, w[i], lb + 3);
                    *reinterpret_cast<unsigned *>(reinterpret_cast<int8_t *>(xp) + size_t(4 * i) * xs) =
                        __byte_perm(__byte_perm(v0, v1, sel), __byte_perm(v2, v3, sel), 0x5410);
                }
            }
#else
                int8_t *xp = a.xt + size_t(k0) * a.rows[0] + r;
                const size_t xs = size_t(a.rows[0]);
#pragma unroll
                for (int i = 0; i < 96; ++i)
                    xp[i * xs] = (int8_t)(w[i >> 2] >> (8 * (i & 3)));
            }
#endif
            if (side == 0 && a.xk) {
#pragma unroll
                for (int t3 = 0; t3 < 3; ++t3) {
                    unsigned bits = __ldg(a.kmask + 3 * g + t3);
                    if (!bits)
                        continue;
                    const unsigned all = bits;
                    const int base = __ldg(a.kbase + 3 * g + t3);
                    while (bits) {
                        const int b = __ffs(bits) - 1;
                        bits &= bits - 1;
                        const int kk = 32 * t3 + b, q = kk >> 2;
                        unsigned wd = 0;
#pragma unroll
                        for (int i = 0; i < 24; ++i)
                            wd = i == q ? w[i] : wd;
                        a.xk[size_t(base + __popc(all & ((1u << b) - 1))) * a.rows[0] + r] =
                            (int8_t)(wd >> (8 * (kk & 3)));
                    }
                }
            }
            unsigned char *fg = fbase + size_t(5 * g) * cutb;
            u64 pend = 0;
            unsigned pkl = 0, pkh = 0, nlg = 0;
            unsigned dpl = 0, dph = 0, spv = 0;
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                unsigned q0 = w[4 * j], q1 = w[4 * j + 1], q2 = w[4 * j + 2], q3 = w[4 * j + 3];
                const int sr = dsum(q0, q1, q2, q3);
                unsigned va = 0x7FFFu;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const unsigned wq = i < 4 ? q0 : i < 8 ? q1 : i < 12 ? q2 : q3;
                    va &= V5F_VM((wq >> (8 * (i & 3))) & 255);
                }
                int pick = 0;
                if (!(va & 1u)) { // a hole at shift 0
                    n103 += n103_of(q0, q1, q2, q3);
                    if (va) {
                        pick = s15_of(__ffs(va) - 1);
                        const unsigned rp =
                            unsigned(pick & 255) * 0x01010101u; // every y in [-128, 127]: the byte wrap is exact
                        q0 = __vsub4(q0, rp);
                        q1 = __vsub4(q1, rp);
                        q2 = __vsub4(q2, rp);
                        q3 = __vsub4(q3, rp);
                    } else { // no shift covers the block: the holes go to their nearest covered value (the leftovers)
                        unsigned o[4] = {q0, q1, q2, q3}, lm = 0;
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const unsigned sh8b = 8 * (i & 3), b = (o[i >> 2] >> sh8b) & 255u;
                            if (!(V5F_VM(b) & 1u)) {
                                o[i >> 2] = (o[i >> 2] & ~(255u << sh8b)) | (unsigned((unsigned char)nearv[b]) << sh8b);
                                lm |= 1u << i;
                            }
                        }
                        q0 = o[0];
                        q1 = o[1];
                        q2 = o[2];
                        q3 = o[3];
                        nlg += __popc(lm);
                        unsigned *km = side ? a.kmask : a.kmaskA;
                        if (km)
                            atomicOr(km + ((k0 + 16 * j) >> 5), lm << ((k0 + 16 * j) & 31));
                    }
                }
                const int sz = dsum(q0, q1, q2, q3);
                unsigned L[8], H[8], es, ho;
        V5F_LOOK q0, q1, q2, q3, L, H, es, ho);
        ewin += es >> 8;
        const unsigned ap = unsigned(pick < 0 ? -pick : pick);
        mx = ap > mx ? ap : mx;
        nsh += pick != 0;
        if (j < 4)
            pkl |= unsigned(pick & 255) << (8 * j);
        else
            pkh |= unsigned(pick & 255) << (8 * (j - 4));
        // the five pieces: face f's codes of values 0..7 (lo) / 8..15 (hi)
        unsigned Fl[5], Fh[5];
        tr4(L[0], L[1], L[2], L[3], Fl[0], Fl[1], Fl[2], Fl[3]);
        tr4(L[4], L[5], L[6], L[7], Fh[0], Fh[1], Fh[2], Fh[3]);
        Fl[4] = g4(H[0], H[1], H[2], H[3]);
        Fh[4] = g4(H[4], H[5], H[6], H[7]);
#pragma unroll
        for (int f = 0; f < 5; ++f) {
            const int flat = 5 * j + f; // cut z = flat / 6 of the group, slot flat % 6; two slots a 16-byte cell
            if ((flat & 1) == 0)
                pend = u64(Fl[f]) | (u64(Fh[f]) << 32);
            else {
                const int q = flat >> 1;
                *reinterpret_cast<uint4 *>(fg + size_t(q / 3) * cutb + 128 * (q % 3)) =
                    make_uint4(unsigned(pend), unsigned(pend >> 32), Fl[f], Fh[f]);
            }
        }
        // the extension: the digits of the raw block sum (A) / of y'' (B), the shift's code; a pair of blocks = one
        // byte a position
        const u64 d = __ldg(&g_dig[(side ? sz : sr) - EXT5_LO]);
        const unsigned cd = v5::code4(pick);
        if ((j & 1) == 0) {
            dpl = unsigned(d);
            dph = unsigned(d >> 32);
            spv = cd;
        } else {
            dpl += 16 * unsigned(d);
            dph += 16 * unsigned(d >> 32);
            spv |= cd << 4;
            push(dl[0], dh[0], dpl, 0x4321);
            push(dl[1], dh[1], dpl, 0x5321);
            push(dl[2], dh[2], dpl, 0x6321);
            push(dl[3], dh[3], dpl, 0x7321);
            push(dl[4], dh[4], dph, 0x4321);
            push(dl[5], dh[5], dph, 0x5321);
            push(sl, shv, spv, 0x4321);
            const int P = 3 * g + (j >> 1);
            if ((P & 7) == 7)
                store_unit(P >> 3);
        }
            }
            nlt += nlg;
            a.sh8[side][size_t(r) * a.ng + g] = u64(pkl) | (u64(pkh | (nlg << 16)) << 32);
        }
        // the row's last unit when 3 ng is not a multiple of 8: zero pairs up to the unit's end
        const int pn = 3 * g1;
        if (g1 == a.ng && (pn & 7)) {
            for (int t = pn & 7; t < 8; ++t) {
                push(dl[0], dh[0], 0u, 0x4321);
                push(dl[1], dh[1], 0u, 0x4321);
                push(dl[2], dh[2], 0u, 0x4321);
                push(dl[3], dh[3], 0u, 0x4321);
                push(dl[4], dh[4], 0u, 0x4321);
                push(dl[5], dh[5], 0u, 0x4321);
                push(sl, shv, 0u, 0x4321);
            }
            store_unit(pn >> 3);
        }
        a.wcnt[side][size_t(r) * a.gx + blockIdx.x] = int(nlt);
        if (nlt && a.rowtot[side])
            atomicAdd(a.rowtot[side] + r, int(nlt));
        if (!real)
            ewin = 0;
    }
    // the CTA's counters: warp reductions, then Shared Memory, one plain store a word
    const unsigned lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    mx = __reduce_max_sync(0xFFFFFFFFu, mx);
    nsh = __reduce_add_sync(0xFFFFFFFFu, nsh);
    nlt = __reduce_add_sync(0xFFFFFFFFu, nlt);
    n103 = __reduce_add_sync(0xFFFFFFFFu, n103);
    ewin = __reduce_max_sync(0xFFFFFFFFu, ewin);
    if (lane == 0) {
        red[wp][0] = mx;
        red[wp][1] = nsh;
        red[wp][2] = nlt;
        red[wp][3] = n103;
        red[wp][4] = ewin;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned o[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < Threads / 32; ++i) {
            o[0] = max(o[0], red[i][0]);
            o[1] += red[i][1];
            o[2] += red[i][2];
            o[3] += red[i][3];
            o[4] = max(o[4], red[i][4]);
        }
        unsigned *pt = a.part + size_t(a.part_base + blockIdx.y * gridDim.x + blockIdx.x) * PartWords;
        pt[0] = o[0];
        pt[1] = o[1];
        pt[2] = o[2];
        pt[3] = 0;
        pt[4] = o[3];
        pt[5] = o[4];
        pt[6] = 0;
        pt[7] = 0;
    }
}

// rowptr (rows + 1) from the (row, window) counts: one CTA a side (blockIdx.x = 0: A, 1: B), the row totals scanned
// 1,024 rows at a time
__global__ void __launch_bounds__(1024)
    rows_scan_w(const int *wca, int rowsa, int *rpa, const int *wcb, int rowsb, int *rpb, int gx) {
    __shared__ int buf[1024];
    __shared__ int carry;
    const int *wc = blockIdx.x ? wcb : wca;
    const int rows = blockIdx.x ? rowsb : rowsa;
    int *rowptr = blockIdx.x ? rpb : rpa;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    for (int r0 = 0; r0 < rows; r0 += 1024) {
        const int r = r0 + threadIdx.x;
        int v = 0;
        if (r < rows)
            for (int w = 0; w < gx; ++w)
                v += wc[size_t(r) * gx + w];
        buf[threadIdx.x] = v;
        __syncthreads();
        for (int d = 1; d < 1024; d <<= 1) {
            const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
            __syncthreads();
            buf[threadIdx.x] += t;
            __syncthreads();
        }
        if (r < rows)
            rowptr[r] = carry + buf[threadIdx.x] - v;
        __syncthreads();
        if (threadIdx.x == 1023)
            carry += buf[1023];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        rowptr[rows] = carry;
}
__global__ void __launch_bounds__(1024)
    rows_scan_t(const int *tota, int rowsa, int *rpa, const int *totb, int rowsb, int *rpb) {
    __shared__ int buf[1024];
    const int *tot = blockIdx.x ? totb : tota;
    const int rows = blockIdx.x ? rowsb : rowsa;
    int *rowptr = blockIdx.x ? rpb : rpa;
    const int per = (rows + 1023) / 1024, r0 = threadIdx.x * per, r1 = min(rows, r0 + per);
    int v = 0;
    for (int r = r0; r < r1; ++r)
        v += tot[r];
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int d = 1; d < 1024; d <<= 1) {
        const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
        __syncthreads();
        buf[threadIdx.x] += t;
        __syncthreads();
    }
    int c = buf[threadIdx.x] - v;
    for (int r = r0; r < r1; ++r) {
        rowptr[r] = c;
        c += tot[r];
    }
    if (threadIdx.x == 1023)
        rowptr[rows] = buf[1023];
}
// the leftovers of (row, window) go to entries[rowptr[r] + the row's earlier windows ..] in k order: (k << 8) | (d +
// 128), d = raw - its nearest covered value; only blocks with shift 0 have leftovers (a block with a covering shift has
// none; with shift 0 found, no value is a hole)
struct LeftSide {
    const int8_t *raw;
    long long ld;
    int rows, rows_true;
    const u64 *sh8;
    const int *wcnt;
    const int *rowptr;
    int *ent;
    long long cap;
};
__global__ void __launch_bounds__(256) write_left_w(const LeftSide la, const LeftSide lb, int ng, int gx, int kr) {
    const int na = la.rows * gx, total = na + lb.rows * gx;
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < total; q += gridDim.x * blockDim.x) {
        const bool sb = q >= na;
        const LeftSide &s = sb ? lb : la;
        const int qq = sb ? q - na : q;
        if (!s.wcnt[qq])
            continue;
        const int r = qq / gx, wi = qq - r * gx;
        int j = s.rowptr[r];
        for (int w = 0; w < wi; ++w)
            j += s.wcnt[size_t(r) * gx + w];
        const int8_t *z = s.raw + size_t(r) * s.ld;
        for (int g = wi * GroupsPerWin; g < min(ng, (wi + 1) * GroupsPerWin); ++g) {
            const u64 w8 = s.sh8[size_t(r) * ng + g];
            if (!((w8 >> 48) & 255))
                continue;
            for (int b = 0; b < 6; ++b) {
                if ((w8 >> (8 * b)) & 255)
                    continue;
                for (int i = 0; i < 16; ++i) {
                    const int k = 96 * g + 16 * b + i;
                    const int x = (r < s.rows_true && k < kr) ? int(z[k]) : 0;
                    if (g_tab[sb][x & 255] >> 63) {
                        if (j < s.cap)
                            s.ent[j] = (k << 8) | (x - g_near[x & 255] + 128);
                        ++j;
                    }
                }
            }
        }
    }
} // beyond cap: the gate sends the call to the exact fallback
__device__ __forceinline__ int y2_of(const int8_t *B, long long ldb, int n_true, int ng, const u64 *sh8b, int kr, int n,
                                     int k) {
    const int x = (n < n_true && k < kr) ? int(B[size_t(n) * ldb + k]) : 0;
    const int g = k / 96, b = (k - 96 * g) >> 4;
    const int pick = n < n_true ? int((signed char)(sh8b[size_t(n) * ng + g] >> (8 * b))) : 0;
    const int y = x - pick;
    return (g_tab[1][y & 255] >> 63) ? int(g_near[y & 255]) : y;
}
// y'' transposed [K][np] for the read-out of A's leftovers: only the rows k that A's entries name when there are at
// most GatherMax entries (a column of B each), else all of it; both kernels read A's count on the device and one of
// them returns at once
__global__ void __launch_bounds__(256)
    y2t_gather(const int *aent, const int *a_total, long long capa, const int8_t *B, long long ldb, int n_true, int np,
               int ng, const u64 *sh8b, int kr, int8_t *y2t) {
    const int tot = *a_total;
    if (tot == 0 || tot > GatherMax || tot > capa)
        return;
    const int work = tot * np; // <= GatherMax x np
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < work; t += gridDim.x * blockDim.x) {
        const int e = t / np, n = t - e * np;
        const int k = aent[e] >> 8;
        y2t[size_t(k) * np + n] = (int8_t)y2_of(B, ldb, n_true, ng, sh8b, kr, n, k);
    }
}
__global__ void __launch_bounds__(256) y2t_full(const int *a_total, const int8_t *B, long long ldb, int n_true, int np,
                                                int ng, const u64 *sh8b, int kr, int K, int8_t *y2t) {
    const int tot = *a_total;
    if (tot <= GatherMax)
        return;
    __shared__ int8_t t[32][33];
    const int tk = (K + 31) / 32, tn = (np + 31) / 32;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    for (int tl = blockIdx.x; tl < tk * tn; tl += gridDim.x) {
        const int k0 = (tl % tk) * 32, n0 = (tl / tk) * 32;
        for (int i = ty; i < 32; i += 8) {
            const int n = n0 + i, k = k0 + tx;
            t[i][tx] = (n < np && k < K) ? (int8_t)y2_of(B, ldb, n_true, ng, sh8b, kr, n, k) : 0;
        }
        __syncthreads();
        for (int i = ty; i < 32; i += 8) {
            const int k = k0 + i, n = n0 + tx;
            if (k < K && n < np)
                y2t[size_t(k) * np + n] = t[tx][i];
        }
        __syncthreads();
    }
}
// the gate (one CTA): the counters of A's CTAs [0, na) and B's [na, na + nb) into stat (A [0..4], B [8..12], as
// v5enc.cuh) and the proof's maxima; then v5enc.cuh's rule: flag[0] = 1 (the exact fallback) when a shift is over 12 or
// the leftovers exceed their cap; the segment is 1,536 k (80 units) when max_rows S_A x max_cols S_B <= 2^48, else 672
// k (35)
__global__ void __launch_bounds__(256)
    gate_fused(const unsigned *part, int na, int nbc, const int *totA, const int *totB, long long capA, long long capB,
               unsigned *stat, unsigned long long *proof, unsigned long long *flag, int *segc) {
    __shared__ unsigned red[2][5][8];
    unsigned o[2][5] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    for (int c = threadIdx.x; c < na + nbc; c += blockDim.x) {
        const int s = c >= na;
        const unsigned *p = part + size_t(c) * PartWords;
        o[s][0] = max(o[s][0], p[0]);
        o[s][1] += p[1];
        o[s][2] += p[2];
        o[s][3] += p[4];
        o[s][4] = max(o[s][4], p[5]);
    }
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int s = 0; s < 2; ++s) {
        o[s][0] = __reduce_max_sync(0xFFFFFFFFu, o[s][0]);
        o[s][1] = __reduce_add_sync(0xFFFFFFFFu, o[s][1]);
        o[s][2] = __reduce_add_sync(0xFFFFFFFFu, o[s][2]);
        o[s][3] = __reduce_add_sync(0xFFFFFFFFu, o[s][3]);
        o[s][4] = __reduce_max_sync(0xFFFFFFFFu, o[s][4]);
        if (lane == 0)
            for (int i = 0; i < 5; ++i)
                red[s][i][wp] = o[s][i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned t[2][5];
        for (int s = 0; s < 2; ++s) {
            t[s][0] = 0;
            t[s][1] = 0;
            t[s][2] = 0;
            t[s][3] = 0;
            t[s][4] = 0;
            for (int w2 = 0; w2 < 8; ++w2) {
                t[s][0] = max(t[s][0], red[s][0][w2]);
                t[s][1] += red[s][1][w2];
                t[s][2] += red[s][2][w2];
                t[s][3] += red[s][3][w2];
                t[s][4] = max(t[s][4], red[s][4][w2]);
            }
            unsigned *st = stat + 8 * s;
            st[0] = t[s][0];
            st[1] = t[s][1];
            st[2] = t[s][2];
            st[3] = 0;
            st[4] = t[s][3];
        }
        proof[0] = t[0][4];
        proof[1] = t[1][4];
        const bool bad = t[0][0] > 12 || t[1][0] > 12 || *totA > capA || *totB > capB;
        const unsigned long long sa = t[0][4], sb = t[1][4];
        const bool ok1536 =
            sa == 0 || sb == 0 || (sa <= 3000000000ull && sb <= 3000000000ull && sa * sb <= (1ull << 48));
        flag[0] = bad ? 1ull : 0ull;
        flag[1] = 0;
        *segc = ok1536 ? 80 : 35;
    }
}
// B's leftovers (f5_fixb of the source, reading the original A with its lda): C[n][m] += sum over column n's entries
// (k, d) of d x A[m][k]; a CTA per column (grid-stride over the columns), the columns without entries skipped at once
// (v5enc.cuh's fixb launched m / 256 x n CTAs)
__global__ void __launch_bounds__(256) fixb_cols(int *C, long long ldc, int mt, int nt, const int8_t *A, long long lda,
                                                 const int *bcol, const int *bent, const unsigned long long *flag) {
    if (flag[0])
        return;
    for (int n = blockIdx.x; n < nt; n += gridDim.x) {
        const int q0 = bcol[n], q1 = bcol[n + 1];
        if (q0 == q1)
            continue;
        for (int m0 = 0; m0 < mt; m0 += 4 * 256) {
            int sm[4] = {0, 0, 0, 0};
            for (int e = q0; e < q1; ++e) {
                const int ent = __ldg(bent + e), d = (ent & 255) - 128;
                const int8_t *ak = A + (ent >> 8);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int m = m0 + 256 * u + threadIdx.x;
                    if (m < mt)
                        sm[u] += d * int(__ldg(ak + size_t(m) * lda));
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int m = m0 + 256 * u + threadIdx.x;
                if (m < mt)
                    C[size_t(n) * ldc + m] += sm[u];
            }
        }
    }
}
__global__ void __launch_bounds__(1024) kprefix(const unsigned *kmask, int nw, int *kbase) {
    __shared__ int buf[1024];
    __shared__ int carry;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    for (int w0 = 0; w0 < nw; w0 += 1024) {
        const int w = w0 + threadIdx.x;
        const int v = w < nw ? __popc(kmask[w]) : 0;
        buf[threadIdx.x] = v;
        __syncthreads();
        for (int d = 1; d < 1024; d <<= 1) {
            const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
            __syncthreads();
            buf[threadIdx.x] += t;
            __syncthreads();
        }
        if (w < nw)
            kbase[w] = carry + buf[threadIdx.x] - v;
        __syncthreads();
        if (threadIdx.x == 1023)
            carry += buf[1023];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        kbase[nw] = carry;
}
__global__ void __launch_bounds__(256)
    fixb_cols2(int *C, long long ldc, int mt, int nt, const int8_t *xk, long long ldx, const unsigned *kmask,
               const int *kbase, const int *bcol, const int *bent, const unsigned long long *flag) {
    if (flag[0])
        return;
    for (int n = blockIdx.x; n < nt; n += gridDim.x) {
        const int q0 = bcol[n], q1 = bcol[n + 1];
        if (q0 == q1)
            continue;
        for (int m0 = 0; m0 < mt; m0 += 4 * 256) {
            int sm[4] = {0, 0, 0, 0};
            for (int e = q0; e < q1; ++e) {
                const int ent = __ldg(bent + e), d = (ent & 255) - 128, k = ent >> 8, wd = k >> 5;
                const int8_t *xr =
                    xk + size_t(__ldg(kbase + wd) + __popc(__ldg(kmask + wd) & ((1u << (k & 31)) - 1))) * ldx;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int m = m0 + 256 * u + threadIdx.x;
                    if (m < mt)
                        sm[u] += d * int(__ldg(xr + m));
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int m = m0 + 256 * u + threadIdx.x;
                if (m < mt)
                    C[size_t(n) * ldc + m] += sm[u];
            }
        }
    }
}
__global__ void __launch_bounds__(1024)
    post1(const int *tota, int rowsa, int *rpa, const int *totb, int rowsb, int *rpb, const unsigned *part, int na,
          int nbc, long long capA, long long capB, unsigned *stat, unsigned long long *proof, unsigned long long *flag,
          int *segc, const unsigned *kmaskA, int nw, int *alist, int *acnt) {
    __shared__ int buf[1024];
    __shared__ unsigned red[2][5][32];
    __shared__ int bad_s;
    for (int sd = 0; sd < 2; ++sd) {
        const int *tot = sd ? totb : tota;
        const int rows = sd ? rowsb : rowsa;
        int *rowptr = sd ? rpb : rpa;
        const int per = (rows + 1023) / 1024, r0 = threadIdx.x * per, r1 = min(rows, r0 + per);
        int v = 0;
        for (int r = r0; r < r1; ++r)
            v += tot[r];
        buf[threadIdx.x] = v;
        __syncthreads();
        for (int d = 1; d < 1024; d <<= 1) {
            const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
            __syncthreads();
            buf[threadIdx.x] += t;
            __syncthreads();
        }
        int c = buf[threadIdx.x] - v;
        for (int r = r0; r < r1; ++r) {
            rowptr[r] = c;
            c += tot[r];
        }
        if (threadIdx.x == 1023)
            rowptr[rows] = buf[1023];
        __syncthreads();
    }
    unsigned o[2][5] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    for (int c = threadIdx.x; c < na + nbc; c += blockDim.x) {
        const int sd = c >= na;
        const unsigned *p = part + size_t(c) * PartWords;
        o[sd][0] = max(o[sd][0], p[0]);
        o[sd][1] += p[1];
        o[sd][2] += p[2];
        o[sd][3] += p[4];
        o[sd][4] = max(o[sd][4], p[5]);
    }
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int sd = 0; sd < 2; ++sd) {
        o[sd][0] = __reduce_max_sync(0xFFFFFFFFu, o[sd][0]);
        o[sd][1] = __reduce_add_sync(0xFFFFFFFFu, o[sd][1]);
        o[sd][2] = __reduce_add_sync(0xFFFFFFFFu, o[sd][2]);
        o[sd][3] = __reduce_add_sync(0xFFFFFFFFu, o[sd][3]);
        o[sd][4] = __reduce_max_sync(0xFFFFFFFFu, o[sd][4]);
        if (lane == 0)
            for (int i = 0; i < 5; ++i)
                red[sd][i][wp] = o[sd][i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned t[2][5];
        for (int sd = 0; sd < 2; ++sd) {
            t[sd][0] = 0;
            t[sd][1] = 0;
            t[sd][2] = 0;
            t[sd][3] = 0;
            t[sd][4] = 0;
            for (int w2 = 0; w2 < 32; ++w2) {
                t[sd][0] = max(t[sd][0], red[sd][0][w2]);
                t[sd][1] += red[sd][1][w2];
                t[sd][2] += red[sd][2][w2];
                t[sd][3] += red[sd][3][w2];
                t[sd][4] = max(t[sd][4], red[sd][4][w2]);
            }
            unsigned *st = stat + 8 * sd;
            st[0] = t[sd][0];
            st[1] = t[sd][1];
            st[2] = t[sd][2];
            st[3] = 0;
            st[4] = t[sd][3];
        }
        proof[0] = t[0][4];
        proof[1] = t[1][4];
        const bool bad = t[0][0] > 12 || t[1][0] > 12 || rpa[rowsa] > capA || rpb[rowsb] > capB;
        const unsigned long long sa = t[0][4], sb = t[1][4];
        const bool ok1536 =
            sa == 0 || sb == 0 || (sa <= 3000000000ull && sb <= 3000000000ull && sa * sb <= (1ull << 48));
        flag[0] = bad ? 1ull : 0ull;
        flag[1] = 0;
        *segc = ok1536 ? 80 : 35;
        bad_s = bad ? 1 : 0;
    }
    __syncthreads();
    if (bad_s) {
        for (int i = threadIdx.x; i <= rowsa; i += 1024)
            rpa[i] = 0;
        for (int i = threadIdx.x; i <= rowsb; i += 1024)
            rpb[i] = 0;
    }
    const int tot =
        bad_s ? 0 : rpa[rowsa]; // (read before any thread could zero it: the zeroing above only runs when bad_s)
    if (tot == 0 || tot > GatherMax) {
        if (threadIdx.x == 0)
            *acnt = tot == 0 ? 0 : -1;
        return;
    }
    const int per = (nw + 1023) / 1024, w0 = threadIdx.x * per, w1 = min(nw, w0 + per);
    int v = 0;
    for (int w = w0; w < w1; ++w)
        v += __popc(kmaskA[w]);
    __syncthreads();
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int d = 1; d < 1024; d <<= 1) {
        const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
        __syncthreads();
        buf[threadIdx.x] += t;
        __syncthreads();
    }
    int c = buf[threadIdx.x] - v;
    for (int w = w0; w < w1; ++w) {
        unsigned bits = kmaskA[w];
        while (bits) {
            const int b = __ffs(bits) - 1;
            bits &= bits - 1;
            alist[c++] = 32 * w + b;
        }
    }
    if (threadIdx.x == 1023)
        *acnt = buf[1023];
}
__global__ void __launch_bounds__(256)
    post2(const LeftSide la, const LeftSide lb, int ng, int gx, int kr, int nl, const int *alist, const int *acnt,
          const int8_t *B, long long ldb, int n_true, int np, const u64 *sh8b, int K, int8_t *y2t) {
    if (int(blockIdx.x) < nl) {
        const int na = la.rows * ng, total = na + lb.rows * ng;
        for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < total; q += nl * blockDim.x) {
            const bool sb = q >= na;
            const LeftSide &s = sb ? lb : la;
            const int qq = sb ? q - na : q;
            const u64 w8 = s.sh8[qq];
            if (!((w8 >> 48) & 255))
                continue;
            const int r = qq / ng, g = qq - r * ng, wi = g / GroupsPerWin;
            int j = s.rowptr[r];
            for (int w = 0; w < wi; ++w)
                j += s.wcnt[size_t(r) * gx + w];
            for (int g2 = wi * GroupsPerWin; g2 < g; ++g2)
                j += int((s.sh8[size_t(r) * ng + g2] >> 48) & 255);
            const int8_t *z = s.raw + size_t(r) * s.ld;
            for (int b = 0; b < 6; ++b) {
                if ((w8 >> (8 * b)) & 255)
                    continue;
                for (int i = 0; i < 16; ++i) {
                    const int k = 96 * g + 16 * b + i;
                    const int x = (r < s.rows_true && k < kr) ? int(z[k]) : 0;
                    if (g_tab[sb][x & 255] >> 63) {
                        if (j < s.cap)
                            s.ent[j] = (k << 8) | (x - g_near[x & 255] + 128);
                        ++j;
                    }
                }
            }
        }
        return;
    }
    const int ac = *acnt, yb = int(blockIdx.x) - nl, ny = int(gridDim.x) - nl;
    if (ac == 0)
        return;
    if (ac > 0) {
        const int work = ac * np;
        for (int t = yb * blockDim.x + threadIdx.x; t < work; t += ny * blockDim.x) {
            const int e = t / np, n = t - e * np, k = alist[e];
            y2t[size_t(k) * np + n] = (int8_t)y2_of(B, ldb, n_true, ng, sh8b, kr, n, k);
        }
        return;
    }
    __shared__ int8_t t[32][33];
    const int tk = (K + 31) / 32, tn = (np + 31) / 32;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    for (int tl = yb; tl < tk * tn; tl += ny) {
        const int k0 = (tl % tk) * 32, n0 = (tl / tk) * 32;
        for (int i = ty; i < 32; i += 8) {
            const int n = n0 + i, k = k0 + tx;
            t[i][tx] = (n < np && k < K) ? (int8_t)y2_of(B, ldb, n_true, ng, sh8b, kr, n, k) : 0;
        }
        __syncthreads();
        for (int i = ty; i < 32; i += 8) {
            const int k = k0 + i, n = n0 + tx;
            if (k < K && n < np)
                y2t[size_t(k) * np + n] = t[tx][i];
        }
        __syncthreads();
    }
}
__global__ void __launch_bounds__(256) fixb_xt(int *C, long long ldc, int mt, int nt, const int8_t *xt, long long ldx,
                                               const int *bcol, const int *bent, const unsigned long long *flag) {
    if (flag[0])
        return;
    for (int n = blockIdx.x; n < nt; n += gridDim.x) {
        const int q0 = bcol[n], q1 = bcol[n + 1];
        if (q0 == q1)
            continue;
        for (int m0 = 0; m0 < mt; m0 += 4 * 256) {
            int sm[4] = {0, 0, 0, 0};
            for (int e = q0; e < q1; ++e) {
                const int ent = __ldg(bent + e), d = (ent & 255) - 128;
                const int8_t *xr = xt + size_t(ent >> 8) * ldx;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int m = m0 + 256 * u + threadIdx.x;
                    if (m < mt)
                        sm[u] += d * int(__ldg(xr + m));
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int m = m0 + 256 * u + threadIdx.x;
                if (m < mt)
                    C[size_t(n) * ldc + m] += sm[u];
            }
        }
    }
}
inline cudaError_t init_kernels() {
    return DynSmem ? cudaFuncSetAttribute(encode_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, DynSmem)
                   : cudaSuccess;
}
} // namespace v5f
