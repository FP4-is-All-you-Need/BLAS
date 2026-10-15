#pragma once

#include <type_traits>
#include <cuda_bf16.h>
#include "l8_encode.cuh"
#ifndef L8E_CS_TABLES_FILE // the counting form's tables (FX 1024) belong to the DGEMM's set; another set passes its own or a dummy
#define L8E_CS_TABLES_FILE "l8_cs_tables.h"
#endif
#include L8E_CS_TABLES_FILE

namespace l8e {

// Compile-time constants read from l8_tables.h (any set on the product / plane
// frontier runs through the compile-time form). Device code only takes them in constant expressions (constexpr locals).
__host__ __device__ constexpr int ct_mod(int q) {
    return l8_mod_ct(q);
}
__host__ __device__ constexpr int ct_planes(int q) {
    return l8_planes_ct(q);
}
__host__ __device__ constexpr int ct_lutoff(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += 2 * ct_mod(i);
    }
    return s;
}
__host__ __device__ constexpr int ct_pbase(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += ct_planes(i);
    }
    return s;
}
// FX 16: the u8 copy of the 1- and 2-plane moduli's entries, after the u16 LUT (byte offsets)
__host__ __device__ constexpr int ct_lut8off(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += ct_planes(i) < 3 ? 2 * ct_mod(i) : 0;
    }
    return s;
}
constexpr int Lut8Bytes = ct_lut8off(L8_NMOD);      // 1,600
constexpr int LutXBytes = LutWords * 2 + Lut8Bytes; // 8,040
// the words the FX 16 copy moves and where the pair tables start: LutXBytes rounded up to 4 (8,040 already is; another table set's
// u8 copy may end mid-word: a 4,910-byte table with 13 moduli lost its last 2 bytes when copied as LutXBytes / 4 words)
constexpr int LutXPad = (LutXBytes + 3) / 4 * 4;
constexpr int LutSmemX = (LutXBytes + 15) / 16 * 16;
// FX 256 groups the 1-plane moduli q 0..3 into two pairs
// (0, 2) and (1, 3), each one residue mod p_a p_b (lazy, [0, 2 p_a p_b)) and one u8 lookup whose byte holds both codes (a in bits
// 0..3, b in bits 4..7), so a pair costs what one 1-plane modulus did. The pair tables follow lut_x's u8 copy (byte LutXPad).
// ch < 2^24 and lo < 2^30, so ch (2^30 mod P) + lo < 2^32 needs P <= 191 (7 x 9 = 63, 8 x 11 = 88).
__host__ __device__ constexpr int ct_pairmod(int i) {
    return ct_mod(i) * ct_mod(i + 2);
}
__host__ __device__ constexpr int ct_pairoff(int i) {
    return i == 0 ? 0 : 2 * ct_pairmod(0);
}
constexpr int LutPairBytes = (2 * ct_pairmod(0) + 2 * ct_pairmod(1) + 3) / 4 * 4; // 304
constexpr int LutX2Bytes = LutXPad + LutPairBytes;                                // 8,344
constexpr int LutSmemX2 = (LutX2Bytes + 15) / 16 * 16;
// FX 1024: the counted moduli's u32 tables after the pair tables, then the block's row sums (32 rows x L8CS_N, u32)
constexpr int LutX3Bytes = LutX2Bytes + L8CS_ENTRIES * 4;
constexpr int CsRowWords = 32 * L8CS_N;
constexpr int LutSmemX3 = (LutX3Bytes + 15) / 16 * 16 + (CsRowWords * 4 + 15) / 16 * 16;

__host__ __device__ constexpr int ct_planes_ns(int q) {
    return ct_planes(q) == 3 ? 2 : ct_planes(q);
}
__host__ __device__ constexpr int ct_pbase_ns(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += ct_planes_ns(i);
    }
    return s;
}
constexpr int L8NoSumPlanes = ct_pbase_ns(L8_NMOD); // 38
__host__ __device__ constexpr int cs_index(int q) {
    for (int c = 0; c < L8CS_N; ++c) {
        if (l8cs_q[c] == q) {
            return c;
        }
    }
    return -1;
}
__host__ __device__ constexpr bool
cs_tables_ok() { // one counted accumulator per modulus, none among the pair moduli, u32 aligned
    for (int c = 0; c < L8CS_N; ++c) {
        if (l8cs_q[c] < 4 || l8cs_q[c] >= L8_NMOD || ct_planes(l8cs_q[c]) < 2) {
            return false;
        }
        if (c && l8cs_q[c] <= l8cs_q[c - 1]) {
            return false;
        }
        if (l8cs_off[c] != (c ? l8cs_off[c - 1] + 2 * ct_mod(l8cs_q[c - 1]) : 0)) {
            return false;
        }
    }
    return LutX2Bytes % 4 == 0 &&
           L8CS_ENTRIES == l8cs_off[L8CS_N - 1] + 2 * ct_mod(l8cs_q[L8CS_N - 1]);
}
// FX 512: A's staging loop divides by nr = rows of the part's 256 threads (8 per row) + 1, which is
// 32 for all 4 parts; as a constant the division e / nr is a multiply-high instead of I2F / MUFU.RCP / F2I and corrections
__host__ __device__ constexpr bool staging_rows_are_32() {
    for (unsigned part = 0; part < 4; ++part) {
        if ((part * 256 + 255) / 8 - (part * 256) / 8 + 1 != 32) {
            return false;
        }
    }
    return true;
}
static_assert(staging_rows_are_32(), "l8e: FX 512's constant");
__host__ __device__ constexpr bool ct_pairs_ok() {
    return ct_planes(0) == 1 && ct_planes(1) == 1 && ct_planes(2) == 1 && ct_planes(3) == 1 &&
           ct_pairmod(0) <= 191 && ct_pairmod(1) <= 191;
}
static_assert(ct_lutoff(L8_NMOD) == LutWords && ct_pbase(L8_NMOD) == L8_NPLANES, "l8e: ct tables");
inline bool
ct_tables_match() { // l8_tables.h's device-side copies against its arrays (both written by gen_l8_tables.py)
    for (int q = 0; q < L8_NMOD; ++q) {
        if (ct_mod(q) != l8_mod[q] || ct_planes(q) != l8_planes[q]) {
            return false;
        }
    }
    return true;
}

// programmatic dependent launch (PTX griddepcontrol, sm_90+): no-ops when the grid was not launched as a dependent
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <int I, int N, class F> __device__ __forceinline__ void static_for(F &&f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        static_for<I + 1, N>(f);
    }
}

constexpr int AStageX = (64 * AStride + 8) * 8; // FX & 1: + one double per chunk of 8 k
__host__ __device__ constexpr int lut_smem_x(int fx) {
    return (fx & 1024) ? LutSmemX3 : (fx & 256) ? LutSmemX2 : (fx & 16) ? LutSmemX : LutSmem;
}
__host__ __device__ constexpr int dyn_smem_x(int side_b, int fx) {
    return lut_smem_x(fx) + (side_b ? 0 : ((fx & 1) ? AStageX : AStage));
}

__device__ __forceinline__ unsigned long long widen_bits(float f) {
    const unsigned b = __float_as_uint(f), a = b & 0x7fffffffu, e = a >> 23, m = a & 0x7fffffu;
    unsigned long long w;
    if (e == 255u) {
        w = 0x7ff0000000000000ull | ((unsigned long long)m << 29);
    } else if (e) {
        w = ((unsigned long long)(e + 896u) << 52) | ((unsigned long long)m << 29);
    } else if (m) {
        const int h =
            31 - __clz(m); // the highest set bit, 0..22: the value is 2^(h - 149) x (m / 2^h)
        w = ((unsigned long long)(h + 874) << 52) |
            (((unsigned long long)m << (52 - h)) & ((1ull << 52) - 1));
    } else {
        w = 0;
    }
    return w | ((unsigned long long)(b >> 31) << 63);
}
__device__ __forceinline__ unsigned long long abs64(double z) {
    return absbits_i(z);
}
__device__ __forceinline__ unsigned long long abs64(float z) {
    return widen_bits(z) & 0x7fffffffffffffffull;
}
__device__ __forceinline__ long long q1t(double z, int sh) {
    return q1i(z, sh);
}
__device__ __forceinline__ long long q1t(float z, int sh) {
    return q1i(__longlong_as_double((long long)widen_bits(z)), sh);
}
// A bf16 occupies the upper half of the float with the same value.
__device__ __forceinline__ float bf16_as_float(__nv_bfloat16 z) {
    return __uint_as_float(unsigned(__bfloat16_as_ushort(z)) << 16);
}
__device__ __forceinline__ unsigned long long abs64(__nv_bfloat16 z) {
    return abs64(bf16_as_float(z));
}
__device__ __forceinline__ long long q1t(__nv_bfloat16 z, int sh) {
    return q1t(bf16_as_float(z), sh);
}

template <int PR, bool CT, int FX = 0, int MINB = 3, class T = double>
__global__ __launch_bounds__(256, MINB) void encode_l8x(const T *__restrict__ x, int64_t ld,
                                                        const int *__restrict__ shifts,
                                                        unsigned char *__restrict__ packed,
                                                        const ModC *__restrict__ mods,
                                                        const unsigned short *__restrict__ lut,
                                                        int side_b, const Geom g) {
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *slut = reinterpret_cast<unsigned short *>(dsm);
    T *sa = reinterpret_cast<T *>(
        dsm +
        lut_smem_x(
            FX)); // [96][AStride] (a float input uses the first half of the double-sized space)
    static_assert(!(FX & 24) || (CT && PR == 0),
                  "FX 8 / 16 are for the compile-time form without probes");
    static_assert(!(FX & 256) || ((FX & 16) && ct_pairs_ok()),
                  "FX 256 is on FX 16's LUT, with four 1-plane moduli first");
    static_assert(!(FX & 1024) || ((FX & 16) && (FX & 8) && ct_pairs_ok() && cs_tables_ok()),
                  "FX 1024 is on FX 24's LUT");

    static_assert(!(FX & (2048 | 4096 | 8192)) ||
                      ((FX & 1024) && __builtin_popcount(FX & (2048 | 4096 | 8192)) == 1),
                  "one FX 1024 probe at a time");
    static_assert(!(FX & 16384) || (CT && PR == 0),
                  "FX 16384 is for the compile-time form without probes");
    unsigned *const cs_s =
        reinterpret_cast<unsigned *>(dsm + (LutX3Bytes + 15) / 16 * 16); // FX 1024: [32][L8CS_N]
    static_assert(((FX >> 5) & 7) == 0 || ((FX >> 5) & 7) == 1 || ((FX >> 5) & 7) == 2 ||
                      ((FX >> 5) & 7) == 4,
                  "one PDL form");
    if constexpr ((FX & 128) != 0) {
        pdl_wait();
        pdl_trigger();
    }
    if constexpr ((FX & 64) != 0) {
        pdl_trigger();
    }
    struct EndWait {
        __device__ ~EndWait() {
            if constexpr ((FX & 64) != 0) {
                pdl_wait();
            }
        }
    } end_wait; // every exit of the block
    const unsigned part = blockIdx.x, cut = blockIdx.y, rank = blockIdx.z & 1u,
                   tile = blockIdx.z >> 1;
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 8, chunk = linear % 8,
                   k0 = cut * 64 + chunk * 8;
    const unsigned pos =
        side_b ? tile * Cols + rank * (Cols / 2) + row : tile * Side + rank * (Side / 2) + row;
    const bool live = k0 < unsigned(g.k) && (side_b ? row < 112 : pos < unsigned(g.m));
    const int inner = int(tile_byte(row, chunk * 4));
    unsigned char *const tbase = packed + size_t(tile) * tile_block_bytes(g.cuts, g.planes) + inner;
    if constexpr ((PR & 8) != 0) { // stores only
        const unsigned seed = (linear * 0x9E3779B9u) ^ (blockIdx.y * 0x85EBCA6Bu) ^ blockIdx.z;
#pragma unroll 1
        for (int q = 0; q < L8_NMOD; ++q) {
            const ModC c = mods[q];
            for (int p = 0; p < c.planes; ++p) {
                *reinterpret_cast<unsigned *>(
                    tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, p)) =
                    live ? seed ^ unsigned(q * 3 + p) : 0u;
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < ((FX & 1024)  ? LutX3Bytes / 4
                                   : (FX & 256) ? LutX2Bytes / 4
                                   : (FX & 16)  ? LutXPad / 4
                                                : LutWords / 2);
         i += 256) {
        reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
    }
    if constexpr ((FX & 1024) != 0) {
        for (int i = threadIdx.x; i < CsRowWords; i += 256) {
            cs_s[i] = 0; // before the block's first barrier below
        }
    }
    const unsigned rlo_part = (part * 256) / 8; // the block's first row (32 rows from here)
    long long v[8];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 8, rhi = (part * 256 + 255) / 8,
                       nr = (FX & 512) ? 32u : rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        if constexpr ((FX & 512) != 0) {

#pragma unroll 4
            for (unsigned e = threadIdx.x; e < 32u * 64u; e += 256) {
                const unsigned kk = e / 32u, r = e % 32u, kg = cut * 64 + kk;
                sa[kk * AStride + ((FX & 1) ? kk / 8 : 0) + r] =
                    kg < unsigned(g.k) && abase + r < unsigned(g.m) ? x[size_t(kg) * ld + abase + r]
                                                                    : T{};
            }
        } else {
            for (unsigned e = threadIdx.x; e < nr * 64; e += 256) {
                const unsigned kk = e / nr, r = e % nr, kg = cut * 64 + kk;
                sa[kk * AStride + ((FX & 1) ? kk / 8 : 0) + r] =
                    kg < unsigned(g.k) && abase + r < unsigned(g.m) ? x[size_t(kg) * ld + abase + r]
                                                                    : T{};
            }
        }
        __syncthreads();
        if constexpr ((FX & 32) != 0) {
            pdl_wait();
            pdl_trigger();
        } // the shifts come from the grids before
        const int sh = live ? shifts[pos] : 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] =
                live ? q1t(sa[(chunk * 8 + j) * AStride + ((FX & 1) ? chunk : 0) + (row - rlo)], sh)
                     : 0;
        }
    } else {
        __syncthreads(); // the LUT copy
        if constexpr ((FX & 32) != 0) {
            pdl_wait();
            pdl_trigger();
        }
        if (live) {
            const unsigned src_row = pos % unsigned(g.n);
            const int sh = shifts[src_row];
            const T *src = x + size_t(src_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = k0 + j < unsigned(g.k) ? q1t(src[j], sh) : 0;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = 0;
            }
        }
    }
    if (!live) { // padding: code 0 in every plane (FX 1024: no return, the block's row sums need every thread at the barrier)
        if ((FX & 4) != 0 && side_b &&
            row >= 112) { // never read by the MMA; zeroed once by the caller
            if constexpr ((FX & 1024) == 0) {
                return;
            }
        } else if constexpr ((FX & 16384) != 0) { // the 38-plane layout
            static_for<0, L8_NMOD>([&](auto qc) {
                constexpr int q = decltype(qc)::value;
#pragma unroll
                for (int p = 0; p < ct_planes_ns(q); ++p)
                    *reinterpret_cast<unsigned *>(tbase + plane_offset(g.cuts, ct_pbase_ns(q),
                                                                       ct_planes_ns(q), rank, cut,
                                                                       p)) = 0u;
            });
        } else {
#pragma unroll 1
            for (int q = 0; q < L8_NMOD; ++q) {
                const ModC c = mods[q];
                for (int p = 0; p < c.planes; ++p) {
                    *reinterpret_cast<unsigned *>(
                        tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, p)) = 0u;
                }
            }
        }
        if constexpr ((FX & 1024) == 0) {
            return;
        }
    }
    auto body = [&]() {
        unsigned lo[8], ch[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            lo[j] = unsigned(v[j]) & 0x3FFFFFFFu;
            ch[j] = unsigned(int(v[j] >> 30) + fzqe2::P2Offset);
        }
        // the codes of one modulus, packed and stored (planes 1 to 3; PR & 4 folds them into one store)
        auto emit = [&](const unsigned (&cd)[8], int planes, unsigned char *qb) {
            if constexpr ((FX & 8) != 0) {
                if (planes ==
                    3) { // byte 0 (planes 0, 1) and byte 1 (plane 2) of two codes per byte_perm
                    const unsigned xe = __byte_perm(cd[0], cd[2], 0x5140),
                                   ye = __byte_perm(cd[4], cd[6], 0x5140);
                    const unsigned xo = __byte_perm(cd[1], cd[3], 0x5140),
                                   yo = __byte_perm(cd[5], cd[7], 0x5140);
                    const unsigned ev = __byte_perm(xe, ye, 0x5410),
                                   eh = __byte_perm(xe, ye, 0x7632);
                    const unsigned od = __byte_perm(xo, yo, 0x5410),
                                   oh = __byte_perm(xo, yo, 0x7632);
                    *reinterpret_cast<unsigned *>(qb) =
                        (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
                    *reinterpret_cast<unsigned *>(qb + TileBytes) =
                        ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
                    *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) =
                        eh | (oh << 4); // plane 2's codes are < 16
                    return;
                }
            }
            const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                           od = gather4(cd[1], cd[3], cd[5], cd[7]);
            unsigned w0 = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4), w1 = 0, w2 = 0;
            if (planes >= 2)
                w1 = ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
            if (planes == 3) {
                const unsigned eh = gather4(cd[0] >> 8, cd[2] >> 8, cd[4] >> 8, cd[6] >> 8),
                               oh = gather4(cd[1] >> 8, cd[3] >> 8, cd[5] >> 8, cd[7] >> 8);
                w2 = (eh & 0x0F0F0F0Fu) | ((oh & 0x0F0F0F0Fu) << 4);
            }
            if constexpr ((PR & 4) != 0) {
                *reinterpret_cast<unsigned *>(qb) = w0 ^ w1 ^ w2;
            } else {
                *reinterpret_cast<unsigned *>(qb) = w0;
                if (planes >= 2)
                    *reinterpret_cast<unsigned *>(qb + TileBytes) = w1;
                if (planes == 3)
                    *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) = w2;
            }
        };
        auto codes = [&](unsigned (&cd)[8], const unsigned short *tab, unsigned rs, unsigned negp,
                         unsigned mm) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned t = (PR & 1) ? ((lo[j] ^ ch[j]) & 7u)
                                      : bqn_asm(mad_asm(ch[j], rs, lo[j]), negp, mm);
                if constexpr ((FX & 2) != 0 && (PR & 1) == 0)
                    t = min(t, t + negp); // [0, 2p) -> [0, p)
                cd[j] = (PR & 2) ? t : unsigned(tab[t]);
            }
        };
        unsigned sum_all = 0; // FX 4096 only
        if constexpr (CT) {
            const unsigned cs = 2u * unsigned(g.cuts) * unsigned(TileBytes),
                           rc = (rank * unsigned(g.cuts) + cut) * unsigned(TileBytes);
            static_for<0, L8_NMOD>([&](auto qc) {
                constexpr int q = decltype(qc)::value;
                constexpr unsigned p = unsigned(ct_mod(q));
                constexpr unsigned mm = unsigned((1ull << 32) / p), negp = 0u - p,
                                   rs = unsigned((1ull << 30) % p);
                constexpr int planes = ct_planes(q), lutoff = ct_lutoff(q);
                // the planes written and their base: FX 16384 writes 2 of a 3-plane modulus's planes (the LUT's codes stay 3-plane)
                constexpr int wplanes = (FX & 16384) ? ct_planes_ns(q) : planes,
                              pbase = (FX & 16384) ? ct_pbase_ns(q) : ct_pbase(q);
                unsigned cd[8];
                if constexpr (
                    (FX & 256) != 0 &&
                    q < 4) { // q 0 / 1 store the pairs (0, 2) / (1, 3); q 2 / 3 were stored with them
                    if constexpr (q < 2) {
                        constexpr unsigned P = unsigned(ct_pairmod(q)),
                                           mmP = unsigned((1ull << 32) / P), negP = 0u - P,
                                           rsP = unsigned((1ull << 30) % P);
                        const unsigned char *tabp =
                            reinterpret_cast<const unsigned char *>(slut) + LutXPad + ct_pairoff(q);
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            cd[j] = tabp[bqn_asm(mad_asm(ch[j], rsP, lo[j]), negP, mmP)];
                        const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                                       od = gather4(cd[1], cd[3], cd[5], cd[7]);
                        constexpr unsigned pb0 = (FX & 16384) ? ct_pbase_ns(q) : ct_pbase(q),
                                           pb2 =
                                               (FX & 16384) ? ct_pbase_ns(q + 2) : ct_pbase(q + 2);
                        *reinterpret_cast<unsigned *>(tbase + (cs * pb0 + rc)) =
                            (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
                        *reinterpret_cast<unsigned *>(tbase + (cs * pb2 + rc)) =
                            ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
                    }
                } else if constexpr ((FX & 1024) != 0 &&
                                     cs_index(q) >= 0) { // counted: u32 entries = code | E << 16
                    constexpr int c = cs_index(q), off = l8cs_off[c];
                    const unsigned *tab32 =
                        reinterpret_cast<const unsigned *>(
                            reinterpret_cast<const unsigned char *>(slut) + LutX2Bytes) +
                        off;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        cd[j] = tab32[bqn_asm(mad_asm(ch[j], rs, lo[j]), negp, mm)];
                    emit(cd, wplanes, tbase + (cs * unsigned(pbase) + rc * unsigned(wplanes)));
                    if constexpr ((FX & 8192) == 0) {
                        const unsigned sum = ((cd[0] + cd[1]) + (cd[2] + cd[3])) +
                                             ((cd[4] + cd[5]) + (cd[6] + cd[7]));
                        if constexpr ((FX & 4096) != 0)
                            sum_all += sum >> 16;
                        else
                            atomicAdd(&cs_s[(row - rlo_part) * L8CS_N + c], sum >> 16);
                    }
                } else if constexpr ((FX & 16) != 0 &&
                                     planes < 3) { // the u8 copy: the residue is the byte offset
                    constexpr int lut8off = ct_lut8off(q);
                    const unsigned char *tab8 =
                        reinterpret_cast<const unsigned char *>(slut) + LutWords * 2 + lut8off;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        cd[j] = tab8[bqn_asm(mad_asm(ch[j], rs, lo[j]), negp, mm)];
                    emit(cd, wplanes, tbase + (cs * unsigned(pbase) + rc * unsigned(wplanes)));
                } else {
                    codes(cd, slut + lutoff, rs, negp, mm);
                    emit(cd, wplanes, tbase + (cs * unsigned(pbase) + rc * unsigned(wplanes)));
                }
            });
            if constexpr ((FX & 4096) != 0)
                atomicAdd(&cs_s[(row - rlo_part) * L8CS_N], sum_all);
        } else {
#pragma unroll 1
            for (int q = 0; q < L8_NMOD; ++q) {
                const ModC c = mods[q];
                unsigned cd[8];
                codes(cd, slut + c.lutoff, c.rs, c.negp, c.mmagic);
                emit(cd, c.planes, tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, 0));
            }
        }
    };
    if constexpr ((FX & 8192) != 0) {
        if (live) {
            body();
        }
    } else if constexpr ((FX & 1024) != 0) {
        if (live) {
            body();
        }
        __syncthreads();
        // once per (row, accumulator): A's rows below m; B's rows below 112 and below n (the wrapped copies of the last tile add
        // nothing); a row shared with the neighbouring part's block gets both blocks' parts
        const unsigned base_row =
            side_b ? tile * Cols + rank * (Cols / 2) : tile * Side + rank * (Side / 2);
        unsigned *const out = g.cs + (side_b ? size_t(g.m) * L8CS_N : size_t(0));
        for (int i = threadIdx.x; i < CsRowWords; i += 256) {
            const unsigned rr = rlo_part + unsigned(i) / L8CS_N, gr = base_row + rr, val = cs_s[i];
            const bool ok = side_b ? (rr < 112 && gr < unsigned(g.n)) : gr < unsigned(g.m);
            if (val && ok && rr < 128 && ((FX & 2048) == 0 || cut == 0)) {
                atomicAdd(out + size_t(gr) * L8CS_N + unsigned(i) % L8CS_N, val);
            }
        }
    } else {
        body();
    }
}

// FX 1024's check (one block per counted accumulator c): res[2c] = max over A's m rows of S_c, res[2c + 1] = max over B's n
// columns, and bit c of res[2 L8CS_N] when res[2c] x res[2c + 1] > (2^24 - 1)^2 (Cauchy-Schwarz: the accumulator's sum over K
// of sum_j w_j |A_j| |B_j| <= sqrt(S_A S_B)). res is zeroed with the sums.
__global__ void __launch_bounds__(1024)
    cs_check_k(const unsigned *__restrict__ cs, int m, int n, unsigned *res) {
    __shared__ unsigned red[2][32];
    const int c = blockIdx.x;
    unsigned ma = 0, mb = 0;
    for (int r = threadIdx.x; r < m; r += blockDim.x) {
        ma = max(ma, cs[size_t(r) * L8CS_N + c]);
    }
    for (int r = threadIdx.x; r < n; r += blockDim.x) {
        mb = max(mb, cs[size_t(m + r) * L8CS_N + c]);
    }
    for (int o = 16; o; o /= 2) {
        ma = max(ma, __shfl_xor_sync(0xffffffffu, ma, o));
        mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    }
    if ((threadIdx.x & 31) == 0) {
        red[0][threadIdx.x / 32] = ma;
        red[1][threadIdx.x / 32] = mb;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ma = mb = 0;
        for (int w = 0; w < int(blockDim.x / 32); ++w) {
            ma = max(ma, red[0][w]);
            mb = max(mb, red[1][w]);
        }
        res[2 * c] = ma;
        res[2 * c + 1] = mb;
        const unsigned long long lim = (1ull << 24) - 1;
        if ((unsigned long long)ma * mb > lim * lim) {
            atomicOr(res + 2 * L8CS_N, 1u << c);
        }
    }
}

// FX 16's LUT of one side: the Encoder's u16 LUT (the same entries, rebuilt from Encoder::codes as init does) followed by the
// u8 copy of the 1- and 2-plane moduli's entries (LutXBytes bytes)
inline std::vector<unsigned char> lut_x_host(const Encoder &e, int side_b) {
    if (e.a_direct || !ct_tables_match()) {
        throw std::runtime_error("l8e: lut_x is the 59-product form of l8_tables.h");
    }
    // FX 16 copies the first LutXPad, FX 256 LutX2Bytes, FX 1024 LutX3Bytes (the counted moduli's u32 entries, code | E << 16)
    std::vector<unsigned char> h(ct_pairs_ok() && cs_tables_ok() ? LutX3Bytes
                                 : ct_pairs_ok()                 ? LutX2Bytes
                                                                 : LutXPad,
                                 0);
    if (ct_pairs_ok() && cs_tables_ok()) {
        for (int c = 0; c < L8CS_N; ++c) {
            const int q = l8cs_q[c], p = ct_mod(q);
            for (int xx = 0; xx < 2 * p; ++xx) {
                const int r = int((((long long)xx - (long long)((1ull << 53) % p)) % p + p) % p);
                const unsigned code = e.codes(side_b, q, r),
                               ev = side_b ? l8cs_eb[l8cs_off[c] + xx] : l8cs_ea[l8cs_off[c] + xx];
                if (code > 0xFFFFu || ev > 0xFFFFu) {
                    throw std::runtime_error("l8e: a counted entry above 16 bits");
                }
                const unsigned w = code | (ev << 16);
                for (int b = 0; b < 4; ++b) {
                    h[LutX2Bytes + 4 * (l8cs_off[c] + xx) + b] = (unsigned char)(w >> (8 * b));
                }
            }
        }
    }
    if (ct_pairs_ok()) {
        for (
            int i = 0; i < 2;
            ++i) { // FX 256's pair tables: entry xx holds the codes of the residues of xx - 2^53 mod p_a / p_b
            const int qa = i, qb = i + 2, pa = ct_mod(qa), pb = ct_mod(qb);
            for (int xx = 0; xx < 2 * ct_pairmod(i); ++xx) {
                const int ra =
                    int((((long long)xx - (long long)((1ull << 53) % pa)) % pa + pa) % pa);
                const int rb =
                    int((((long long)xx - (long long)((1ull << 53) % pb)) % pb + pb) % pb);
                const unsigned ca = e.codes(side_b, qa, ra), cb = e.codes(side_b, qb, rb);
                if (ca > 15u || cb > 15u) {
                    throw std::runtime_error("l8e: a 1-plane code above 4 bits");
                }
                h[LutXPad + ct_pairoff(i) + xx] = (unsigned char)(ca | (cb << 4));
            }
        }
    }
    for (int q = 0; q < L8_NMOD; ++q) {
        const unsigned p = unsigned(ct_mod(q));
        const long long rot = (long long)((1ull << 53) % p);
        for (unsigned xx = 0; xx < 2 * p; ++xx) {
            const int r = int((((long long)xx - rot) % (long long)p + p) % p);
            const unsigned c = e.codes(side_b, q, r);
            h[2 * (ct_lutoff(q) + xx)] = (unsigned char)(c & 255u);
            h[2 * (ct_lutoff(q) + xx) + 1] = (unsigned char)(c >> 8);
            if (ct_planes(q) < 3) {
                if (c > 255u) {
                    throw std::runtime_error("l8e: a 1- or 2-plane code above 8 bits");
                }
                h[LutWords * 2 + ct_lut8off(q) + xx] = (unsigned char)c;
            }
        }
    }
    return h;
}

// launch on an Encoder's tables (the 59-product form for CT); FX 16 needs lutx = that side's lut_x_host on the device
template <int PR, bool CT, int FX = 0, int MINB = 3, class T = double>
inline void launch_x(const Encoder &e, const T *x, int64_t ld, const int *shifts,
                     unsigned char *packed, int side_b, cudaStream_t stream = 0,
                     const unsigned char *lutx = nullptr) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(encode_l8x<PR, CT, FX, MINB, T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_smem_x(0, FX)) != cudaSuccess) {
            throw std::runtime_error("l8e: encode_l8x attribute");
        }
        attr = true;
    }
    if (CT && (e.a_direct || !ct_tables_match())) {
        throw std::runtime_error("l8e: encode_l8x<CT> is the 59-product form of l8_tables.h");
    }
    if ((FX & 16) && !lutx) {
        throw std::runtime_error("l8e: FX 16 needs the lut_x of this side");
    }
    const unsigned short *tab =
        (FX & 16) ? reinterpret_cast<const unsigned short *>(lutx) : e.dlut[side_b];
    const dim3 grid(4, unsigned(e.geo.cuts), unsigned(e.tiles(side_b) * 2));
    encode_l8x<PR, CT, FX, MINB, T><<<grid, 256, dyn_smem_x(side_b, FX), stream>>>(
        x, ld, shifts, packed, e.dmods, tab, side_b, e.geo);
}

} // namespace l8e
