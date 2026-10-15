#pragma once

#include <type_traits>
#include "quant_int.cuh"
#include "fp64k_encode.cuh"
#include "fp64kc_parts.h"

namespace fzqe2 {
using namespace fp64k;

constexpr int XS = 576;

constexpr int Z2M = 3 * FP64K_NMOD - FP64KC_NPART;
constexpr int ABlockRows = 43, AStride = ABlockRows + 1;
constexpr int AStageBytes = 96 * AStride * 16;

__host__ __device__ constexpr int lut_stride(int lazy) {
    return lazy ? XS : FP64K_LUT_STRIDE;
}

constexpr int XSA = 1024;
__host__ __device__ constexpr int part_off(int amin, int pidx) {
    return !amin            ? pidx * XS
           : pidx < 2 * Z2M ? (pidx / 2) * (XS + XSA) + (pidx & 1) * XS
                            : Z2M * (XS + XSA) + (pidx - 2 * Z2M) * XS;
}
__host__ __device__ constexpr int lut_bytes(int lazy, int p2 = 0, int amin = 0) {
    return p2 ? part_off(amin, FP64KC_NPART) : FP64K_NMOD * lut_stride(lazy);
}
__host__ __device__ constexpr int lut_smem(int lazy, int p2 = 0, int amin = 0) {
    return (lut_bytes(lazy, p2, amin) + 15) / 16 * 16;
}

__host__ __device__ constexpr int dyn_smem(int lazy, int lutsm, int side_b, int p2 = 0,
                                           int amin = 0) {
    return (lutsm ? lut_smem(lazy, p2, amin) : 0) + (side_b ? 0 : AStageBytes);
}

constexpr int P2Offset = 1 << 23;

__device__ __forceinline__ long long q1(double v, int sh) {
    return qint::finite_bits(qint::absbits(v)) ? qint::scale_round(v, sh) : 0;
}

__device__ __forceinline__ unsigned long long absbits_i(double z) {
    unsigned long long r;
    asm("and.b64 %0, %1, 0x7fffffffffffffff;"
        : "=l"(r)
        : "l"((unsigned long long)__double_as_longlong(z)));
    return r;
}
__device__ __forceinline__ long long q1i(double z, int shift) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(z), a = absbits_i(z);
    if (!qint::finite_bits(a) || !a)
        return 0;
    const int e = int(a >> 52);
    const unsigned long long m = (a & ((1ull << 52) - 1)) | (e ? (1ull << 52) : 0ull);
    const int s = (e ? e : 1) - 1075 + shift;
    unsigned long long v;
    if (s >= 0)
        v = m << s;
    else if (s < -54)
        v = 0;
    else {
        const int d = -s;
        v = m >> d;
        const unsigned long long rem = m & ((1ull << d) - 1), half = 1ull << (d - 1);
        if (rem > half || (rem == half && (v & 1)))
            ++v;
    }
    return (b >> 63) ? -(long long)v : (long long)v;
}

__device__ __forceinline__ unsigned tsum(long long v, int rc1, int rc2, unsigned ebias) {
    const int v0 = int(v & 0x3FFFF), v1 = int((v >> 18) & 0x3FFFF), v2 = int(v >> 36);
    return unsigned(v0 + v1 * rc1 + v2 * rc2) + ebias;
}
__device__ __forceinline__ unsigned bq(unsigned t, unsigned p, unsigned mmagic) {
    return t - __umulhi(t, mmagic) * p;
}

__device__ __forceinline__ unsigned bqn(unsigned t, unsigned negp, unsigned mmagic) {
    return t + __umulhi(t, mmagic) * negp;
}

__device__ __forceinline__ unsigned mad_asm(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
__device__ __forceinline__ unsigned bqn_asm(unsigned t, unsigned negp, unsigned mmagic) {
    unsigned r;
    asm("{.reg .u32 q; mul.hi.u32 q, %1, %3; mad.lo.u32 %0, q, %2, %1;}"
        : "=r"(r)
        : "r"(t), "r"(negp), "r"(mmagic));
    return r;
}

__device__ __forceinline__ unsigned bq16_asm(unsigned a, unsigned negp, unsigned m16) {
    unsigned r;
    asm("{.reg .u32 q; mul.lo.u32 q, %1, %3; shr.u32 q, q, 16; mad.lo.u32 %0, q, %2, %1;}"
        : "=r"(r)
        : "r"(a), "r"(negp), "r"(m16));
    return r;
}
__device__ __forceinline__ unsigned amin_asm(unsigned rr, unsigned twop, unsigned a) {
    unsigned r;
    asm("{.reg .u32 x; add.u32 x, %1, %1; add.u32 x, x, %2; sub.u32 %0, x, %3;}"
        : "=r"(r)
        : "r"(rr), "r"(twop), "r"(a));
    return r;
}
__device__ __forceinline__ unsigned fix(unsigned r, unsigned p) {
    return min(r, r - p);
}

__device__ __forceinline__ unsigned gather4(unsigned c0, unsigned c1, unsigned c2, unsigned c3) {
    return __byte_perm(__byte_perm(c0, c1, 0x0040), __byte_perm(c2, c3, 0x0040), 0x5410);
}

template <int LAZY, int PACK, int LUTSM, int DEAD, int PROBE, int P2 = 0, int AMIN = 0, int ASM = 0,
          int B16 = 0, int FP = 0, int QI = 0>
__global__ __launch_bounds__(256, 2) void encode_c2(
    const double2 *__restrict__ x, int64_t ld, const int *__restrict__ shifts,
    unsigned char *__restrict__ packed, const Fp64kMod *__restrict__ mods,
    const unsigned char *__restrict__ lut, const int *__restrict__ sqrtm1, int side_b,
    unsigned *__restrict__ sink, const unsigned char *__restrict__ lut_base,
    const uint4 *__restrict__ rsplit, const float4 *__restrict__ fcon) {
    static_assert(!P2 || LAZY, "P2 uses the extended LUT");
    static_assert(!AMIN || P2 >= 2, "AMIN needs the bqn forms");
    static_assert(!ASM || P2 >= 2, "ASM is for the bqn forms");
    static_assert(!B16 || ASM, "B16 is an ASM form");
    static_assert(!FP || ((P2 == 2 || P2 == 4 || P2 == 5) && AMIN && ASM),
                  "FP is built on the q30a / q62a / q60a amin forms");
    static_assert((P2 != 4 && P2 != 5) || ASM, "P2 = 4 and 5 are ASM forms");
    extern __shared__ __align__(16) unsigned char dsm[];
    constexpr int LS = lut_stride(LAZY);
    unsigned char *slut = dsm;
    double2 *sa_c = reinterpret_cast<double2 *>(dsm + (LUTSM ? lut_smem(LAZY, P2, AMIN) : 0));
    const unsigned unit = blockIdx.x / 3, part = blockIdx.x % 3;
    const unsigned cut = unit % Cuts, rank = (unit / Cuts) % 2, tile = unit / (Cuts * 2);
    const unsigned linear = part * 256 + threadIdx.x;
    const unsigned row = linear / 6, chunk = linear % 6;
    const unsigned k0 = cut * 96 + chunk * 16;
    const bool live = (k0 + 15 < K) && (!side_b || row < 112);
    if constexpr (LUTSM) {
        for (int i = threadIdx.x; i < lut_bytes(LAZY, P2, AMIN) / 4; i += 256)
            reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
        if (side_b)
            __syncthreads();
    }
    const unsigned char *tab = LUTSM ? slut : lut;
    long long vr[16], vi[16];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 6, rhi = (part * 256 + 255) / 6, nr = rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        for (unsigned e = threadIdx.x; e < nr * 96; e += 256) {
            const unsigned kk = e / nr, r = e % nr, kg = cut * 96 + kk;
            sa_c[kk * AStride + r] =
                kg < unsigned(K) ? x[size_t(kg) * ld + abase + r] : make_double2(0.0, 0.0);
        }
        __syncthreads();
        const int sh = live ? shifts[abase + (row - rlo)] : 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const double2 z =
                live ? sa_c[(chunk * 16 + j) * AStride + (row - rlo)] : make_double2(0.0, 0.0);
            vr[j] = live ? (QI ? q1i(z.x, sh) : q1(z.x, sh)) : 0;
            vi[j] = live ? (QI ? q1i(z.y, sh) : q1(z.y, sh)) : 0;
        }
    } else {
        if (live) {
            const unsigned source_row = (tile * Cols + rank * (Cols / 2) + row) % N;
            const int sh = shifts[source_row];
            const double2 *src = x + size_t(source_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const double2 z = src[j];
                vr[j] = QI ? q1i(z.x, sh) : q1(z.x, sh);
                vi[j] = QI ? q1i(z.y, sh) : q1(z.y, sh);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j)
                vr[j] = vi[j] = 0;
        }
    }
    unsigned lor[16], chr[16], loi[16], chi[16];
    if constexpr (P2 == 5) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            lor[j] = unsigned(vr[j]) & 0x3FFFFFFFu;
            chr[j] = unsigned(int(vr[j] >> 30)) + (1u << 30);
            loi[j] = unsigned(vi[j]) & 0x3FFFFFFFu;
            chi[j] = unsigned(int(vi[j] >> 30)) + (1u << 30);
        }
    } else if constexpr (P2 == 4) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            lor[j] = unsigned(vr[j]);
            chr[j] = unsigned(int(vr[j] >> 32)) + (1u << 30);
            loi[j] = unsigned(vi[j]);
            chi[j] = unsigned(int(vi[j] >> 32)) + (1u << 30);
        }
    } else if constexpr (P2 == 2) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            lor[j] = unsigned(vr[j]) & 0x3FFFFFFFu;
            chr[j] = unsigned(int(vr[j] >> 30) + P2Offset);
            loi[j] = unsigned(vi[j]) & 0x3FFFFFFFu;
            chi[j] = unsigned(int(vi[j] >> 30) + P2Offset);
        }
    } else if constexpr (P2) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            lor[j] = unsigned(vr[j]);
            chr[j] = unsigned(int(vr[j] >> 32) + P2Offset);
            loi[j] = unsigned(vi[j]);
            chi[j] = unsigned(int(vi[j] >> 32) + P2Offset);
        }
    }

    auto resid = [&](int j, int h, const Fp64kMod &m, unsigned p, unsigned negp, unsigned rs,
                     unsigned m16) -> unsigned {
        const unsigned lo = h ? loi[j] : lor[j], ch = h ? chi[j] : chr[j];
        if constexpr (P2 == 5)
            return bqn_asm(mad_asm(bqn_asm(ch, negp, m.mmagic), rs, lo), negp, m.mmagic);
        else if constexpr (P2 == 4)
            return bq16_asm(mad_asm(bqn_asm(ch, negp, m.mmagic), rs, bqn_asm(lo, negp, m.mmagic)),
                            negp, m16);
        else if constexpr (P2 == 2 && ASM)
            return bqn_asm(mad_asm(ch, rs, lo), negp, m.mmagic);
        else if constexpr (P2 == 2)
            return bqn(lo + ch * rs, negp, m.mmagic);
        else if constexpr (P2 == 3)
            return bqn(bqn(lo, negp, m.mmagic) + ch * rs, negp, m.mmagic);
        else if constexpr (P2 == 1)
            return bq(bq(lo, p, m.mmagic) + ch * rs, p, m.mmagic);
        else
            return bq(tsum(h ? vi[j] : vr[j], m.rc1, m.rc2, m.ebias), p, m.mmagic);
    };
    const int inner = int(tile_byte(row, chunk * 8));
    unsigned char *const tbase = packed + size_t(tile) * FP64KC_NPART * PackedBytes;
    const size_t off0 = tile_offset(rank, cut, 0) * TileBytes + inner,
                 off1 = tile_offset(rank, cut, 1) * TileBytes + inner;
    unsigned acc = 0;
    auto put = [&](int pidx, unsigned lo0, unsigned hi0, unsigned lo1, unsigned hi1) {
        if constexpr (PROBE == 2) {
            acc ^= lo0 ^ (hi0 * 3u) ^ (lo1 * 5u) ^ (hi1 * 7u) ^ unsigned(pidx);
        } else {
            unsigned char *base = tbase + size_t(pidx) * PackedBytes;
            reinterpret_cast<uint2 *>(base + off0)[0] = make_uint2(lo0, hi0);
            reinterpret_cast<uint2 *>(base + off1)[0] = make_uint2(lo1, hi1);
        }
    };
    if constexpr (PROBE == 3) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc ^= unsigned(vr[j]) ^ unsigned(vr[j] >> 32) * 3u ^ unsigned(vi[j]) * 5u ^
                   unsigned(vi[j] >> 32) * 7u;
        sink[size_t(blockIdx.x) * 256 + threadIdx.x] = acc;
        return;
    }
    if constexpr (DEAD) {

        if (DEAD == 2 && side_b && row >= 112)
            return;
        if (!live) {
#pragma unroll 1
            for (int pidx = 0; pidx < FP64KC_NPART; ++pidx) {
                const int qm = pidx < 2 * Z2M ? pidx / 2 : Z2M + (pidx - 2 * Z2M) / 3;
                const unsigned code = P2 ? lut_base[qm * FP64K_LUT_STRIDE] : tab[qm * LS];
                const unsigned w0 = (code & 15u) * 0x11111111u, w1 = (code >> 4) * 0x11111111u;
                put(pidx, w0, w0, w1, w1);
            }
            if constexpr (PROBE == 2)
                sink[size_t(blockIdx.x) * 256 + threadIdx.x] = acc;
            return;
        }
    }

    auto modulus = [&](auto code_of, auto np_tag, int pidx0) {
        constexpr int NP = decltype(np_tag)::value;
        unsigned l0[NP][2], l1[NP][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if constexpr (PACK) {
                unsigned ev[NP], od[NP];
#pragma unroll
                for (int par = 0; par < 2; ++par) {
                    unsigned c[4][NP];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        code_of(8 * h + par + 2 * i, c[i]);
#pragma unroll
                    for (int k = 0; k < NP; ++k)
                        (par ? od : ev)[k] = gather4(c[0][k], c[1][k], c[2][k], c[3][k]);
                }
#pragma unroll
                for (int k = 0; k < NP; ++k) {
                    l0[k][h] = (ev[k] & 0x0F0F0F0Fu) | ((od[k] & 0x0F0F0F0Fu) << 4);
                    l1[k][h] = ((ev[k] >> 4) & 0x0F0F0F0Fu) | (od[k] & 0xF0F0F0F0u);
                }
            } else {
#pragma unroll
                for (int k = 0; k < NP; ++k)
                    l0[k][h] = l1[k][h] = 0;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    unsigned c[NP];
                    code_of(8 * h + e, c);
#pragma unroll
                    for (int k = 0; k < NP; ++k) {
                        l0[k][h] |= (c[k] & 15u) << (4 * e);
                        l1[k][h] |= (c[k] >> 4) << (4 * e);
                    }
                }
            }
        }
#pragma unroll
        for (int k = 0; k < NP; ++k)
            put(pidx0 + k, l0[k][0], l0[k][1], l1[k][0], l1[k][1]);
    };
#pragma unroll 1
    for (int qm = 0; qm < Z2M; ++qm) {
        const Fp64kMod m = mods[qm];
        const unsigned p = unsigned(m.p), jj = unsigned(sqrtm1[qm]), pj = p - jj;
        const uint4 mc = P2 >= 2 ? rsplit[qm] : make_uint4(0u, 0u, 0u, 0u);
        const unsigned negp = mc.y, m16 = mc.z;
        const unsigned rs = P2 == 1 ? unsigned(0x100000000ull % p) : mc.x;
        const unsigned char *table = P2 ? tab + part_off(AMIN, 2 * qm) : tab + qm * LS;
        const unsigned char *tableb = P2 ? tab + part_off(AMIN, 2 * qm + 1) : table;
        const float4 fq = FP ? fcon[qm] : make_float4(0.f, 0.f, 0.f, 0.f);
        modulus(
            [&](int j, unsigned (&c)[2]) {
                const unsigned rr = resid(j, 0, m, p, negp, rs, m16),
                               ri = resid(j, 1, m, p, negp, rs, m16);
                unsigned a, b;
                if constexpr (FP) {
                    const float rrf = __fsub_rn(__uint_as_float(rr | 0x4B000000u), 8388608.0f);
                    const float rif = __fsub_rn(__uint_as_float(ri | 0x4B000000u), 8388608.0f);
                    const float af = __fmaf_rn(fq.x, rif, rrf);
                    const float qf = __fsub_rn(__fmaf_rn(af, fq.y, 12582912.0f), 12582912.0f);
                    const float rf = __fmaf_rn(-qf, fq.z, af);
                    a = __float_as_uint(__fadd_rn(rf, fq.w)) - 0x4B000000u;
                    b = __float_as_uint(__fmaf_rn(2.0f, rrf, __fsub_rn(fq.w, rf))) - 0x4B000000u;
                } else if constexpr (P2 >= 2 && ASM) {
                    if constexpr (B16)
                        a = bq16_asm(mad_asm(jj, ri, rr), negp, m16);
                    else
                        a = bqn_asm(mad_asm(jj, ri, rr), negp, m.mmagic);
                    if constexpr (AMIN)
                        b = amin_asm(rr, 2 * p, a);
                    else if constexpr (B16)
                        b = bq16_asm(mad_asm(pj, ri, rr), negp, m16);
                    else
                        b = bqn_asm(mad_asm(pj, ri, rr), negp, m.mmagic);
                } else if constexpr (P2 >= 2) {
                    a = bqn(rr + jj * ri, negp, m.mmagic);
                    if constexpr (AMIN)
                        b = 2 * rr + 2 * p - a;
                    else
                        b = bqn(rr + pj * ri, negp, m.mmagic);
                } else {
                    a = bq(rr + jj * ri, p, m.mmagic);
                    b = bq(rr + pj * ri, p, m.mmagic);
                }
                if constexpr (!LAZY) {
                    a = fix(a, p);
                    b = fix(b, p);
                }
                if constexpr (PROBE == 1) {
                    c[0] = a & 0xffu;
                    c[1] = b & 0xffu;
                } else {
                    c[0] = table[a];
                    c[1] = tableb[b];
                }
            },
            std::integral_constant<int, 2>{}, 2 * qm);
    }
#pragma unroll 1
    for (int qm = Z2M; qm < FP64K_NMOD; ++qm) {
        const Fp64kMod m = mods[qm];
        const unsigned p = unsigned(m.p);
        const uint4 mc = P2 >= 2 ? rsplit[qm] : make_uint4(0u, 0u, 0u, 0u);
        const unsigned negp = mc.y, m16 = mc.z;
        const unsigned rs = P2 == 1 ? unsigned(0x100000000ull % p) : mc.x;
        const int pidx0 = 2 * Z2M + 3 * (qm - Z2M);
        const unsigned char *table = P2 ? tab + part_off(AMIN, pidx0) : tab + qm * LS;
        const unsigned char *table2 = P2 ? tab + part_off(AMIN, pidx0 + 1) : table;
        const unsigned char *table3 = P2 ? tab + part_off(AMIN, pidx0 + 2) : table;
        modulus(
            [&](int j, unsigned (&c)[3]) {
                unsigned rr = resid(j, 0, m, p, negp, rs, m16),
                         ri = resid(j, 1, m, p, negp, rs, m16);
                unsigned s;
                if constexpr (LAZY)
                    s = rr + ri;
                else {
                    rr = fix(rr, p);
                    ri = fix(ri, p);
                    s = fix(rr + ri, p);
                }
                if constexpr (PROBE == 1) {
                    c[0] = rr & 0xffu;
                    c[1] = ri & 0xffu;
                    c[2] = s & 0xffu;
                } else {
                    c[0] = table[rr];
                    c[1] = table2[ri];
                    c[2] = table3[s];
                }
            },
            std::integral_constant<int, 3>{}, pidx0);
    }
    if constexpr (PROBE == 2)
        sink[size_t(blockIdx.x) * 256 + threadIdx.x] = acc;
}

} // namespace fzqe2
