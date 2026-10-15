#pragma once

#include <stdexcept>
#include <vector>
#include <cstdlib>
#include "zl8_lut.h"

namespace zl8e {
using namespace fp64k;
using l8e::Geom;
using l8e::plane_offset;
using l8e::tile_block_bytes;
using fzqe2::q1i;
using fzqe2::gather4;
using fzqe2::bqn_asm;
using fzqe2::mad_asm;

struct ZModC {
    unsigned mmagic, negp, rs, rot;
    int p, planes, pbase, lutoff, role, j, pad0, pad1;
}; // per modulus
constexpr int LutWordsZ = (ZL8_LUT_WORDS + 1) / 2 * 2;
constexpr int LutSmemZ = (LutWordsZ * 2 + 15) / 16 * 16;
constexpr int ARowsZ = 32, AStrideZ = ARowsZ + 1;
constexpr int AStageZ = 64 * AStrideZ; // float2
__host__ __device__ constexpr int dyn_smem_z(int side_b) {
    return LutSmemZ + (side_b ? 0 : AStageZ * 8);
}

__device__ __forceinline__ unsigned long long cabsbits(float2 z) {
    const unsigned long long a = qint::absbits(z.x), b = qint::absbits(z.y);
    return a > b ? a : b;
}
__device__ __forceinline__ bool cfinite(float2 z) {
    return qint::finite_bits(qint::absbits(z.x)) && qint::finite_bits(qint::absbits(z.y));
}
// fused_zqe.cuh's shift_cols_c / shift_rows_c, verbatim (the complex engine's shifts)
template <int R>
__global__ void __launch_bounds__(256)
    shift_cols_c(const float2 *__restrict__ x, int *exponents, int *invalid, int outer, int k,
                 int64_t ld, int bits) {
    constexpr int C = 256 / R;
    __shared__ unsigned long long red[C][R];
    const int t = threadIdx.x, r = t % R, c = t / R, o0 = blockIdx.x * R, o = o0 + r;
    unsigned long long mx = 0;
    bool bad = false;
    if (o < outer) {
        for (int q = c; q < k; q += C) {
            const float2 z = x[size_t(q) * ld + o];
            if (!cfinite(z)) {
                bad = true;
            } else {
                const unsigned long long a = cabsbits(z);
                mx = a > mx ? a : mx;
            }
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[c][r] = mx;
    __syncthreads();
    if (t < R) {
        unsigned long long m = 0;
        for (int i = 0; i < C; ++i) {
            m = red[i][t] > m ? red[i][t] : m;
        }
        if (o0 + t < outer) {
            exponents[o0 + t] = m == 0 ? 0 : bits - 1 - qint::ilogb_bits(m);
        }
    }
}
__global__ void __launch_bounds__(256)
    shift_rows_c(const float2 *__restrict__ x, int *exponents, int *invalid, int outer, int k,
                 int64_t ld, int bits) {
    __shared__ unsigned long long red[256];
    const int o = blockIdx.x, t = threadIdx.x;
    const float2 *row = x + size_t(o) * ld;
    unsigned long long mx = 0;
    bool bad = false;
    for (int q = t; q < k; q += 256) {
        const float2 z = row[q];
        if (!cfinite(z)) {
            bad = true;
        } else {
            const unsigned long long a = cabsbits(z);
            mx = a > mx ? a : mx;
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[t] = mx;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) {
            red[t] = red[t + s] > red[t] ? red[t + s] : red[t];
        }
        __syncthreads();
    }
    if (t == 0) {
        exponents[o] = red[0] == 0 ? 0 : bits - 1 - qint::ilogb_bits(red[0]);
    }
}

// the exact residue of v (|v| < 2^53) mod p from lo / ch (ch = (v >> 30) + 2^23): lazy Barrett in [0, 2p), one correction, and the
// offset's residue rot = 2^53 mod p taken off
__device__ __forceinline__ unsigned exact_res(unsigned lo, unsigned ch, const ZModC &c) {
    unsigned t = bqn_asm(mad_asm(ch, c.rs, lo), c.negp, c.mmagic);
    t = min(t, t + c.negp); // [0, p)
    return t >= c.rot ? t - c.rot : t + unsigned(c.p) - c.rot;
}
__device__ __forceinline__ unsigned modp_small(unsigned t, const ZModC &c) { // t < 2^32 -> [0, p)
    unsigned r = bqn_asm(t, c.negp, c.mmagic);
    return min(r, r + c.negp);
}

__global__ __launch_bounds__(256, 2) void encode_zl8(const float2 *__restrict__ x, int64_t ld,
                                                     const int *__restrict__ shifts,
                                                     unsigned char *__restrict__ packed,
                                                     const ZModC *__restrict__ mods,
                                                     const unsigned short *__restrict__ lut,
                                                     int side_b, const Geom g) {
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *slut = reinterpret_cast<unsigned short *>(dsm);
    float2 *sa = reinterpret_cast<float2 *>(dsm + LutSmemZ); // Shared rows use the encoder stride.
    const unsigned part = blockIdx.x, cut = blockIdx.y, rank = blockIdx.z & 1u,
                   tile = blockIdx.z >> 1;
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 8, chunk = linear % 8,
                   k0 = cut * 64 + chunk * 8;
    const unsigned pos =
        side_b ? tile * Cols + rank * (Cols / 2) + row : tile * Side + rank * (Side / 2) + row;
    const bool live = k0 < unsigned(g.k) && (side_b ? row < 112 : pos < unsigned(g.m));
    for (int i = threadIdx.x; i < LutWordsZ / 2; i += 256) {
        reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
    }
    long long vr[8], vi[8];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 8, rhi = (part * 256 + 255) / 8, nr = rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        for (unsigned e = threadIdx.x; e < nr * 64; e += 256) {
            const unsigned kk = e / nr, r = e % nr, kg = cut * 64 + kk;
            sa[kk * AStrideZ + r] = kg < unsigned(g.k) && abase + r < unsigned(g.m)
                                        ? x[size_t(kg) * ld + abase + r]
                                        : make_float2(0.0, 0.0);
        }
        __syncthreads();
        const int sh = live ? shifts[pos] : 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 z = sa[(chunk * 8 + j) * AStrideZ + (row - rlo)];
            vr[j] = live ? q1i(z.x, sh) : 0;
            vi[j] = live ? q1i(z.y, sh) : 0;
        }
    } else {
        __syncthreads(); // the LUT copy
        if (live) {
            const unsigned src_row = pos % unsigned(g.n);
            const int sh = shifts[src_row];
            const float2 *src = x + size_t(src_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 z = k0 + j < unsigned(g.k) ? src[j] : make_float2(0.0, 0.0);
                vr[j] = q1i(z.x, sh);
                vi[j] = q1i(z.y, sh);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                vr[j] = vi[j] = 0;
            }
        }
    }
    const int inner = int(tile_byte(row, chunk * 4));
    unsigned char *const tbase = packed + size_t(tile) * tile_block_bytes(g.cuts, g.planes) + inner;
    if (!live) { // padding: code 0 in every plane of every part
#pragma unroll 1
        for (int q = 0; q < ZL8_NMODC; ++q) {
            const ZModC c = mods[q];
            const int np = c.role == 0 ? 2 : 3;
            for (int pr = 0; pr < np; ++pr) {
                for (int p = 0; p < c.planes; ++p) {
                    *reinterpret_cast<unsigned *>(
                        tbase +
                        plane_offset(g.cuts, c.pbase + pr * c.planes, c.planes, rank, cut, p)) = 0u;
                }
            }
        }
        return;
    }
    unsigned lor[8], chr[8], loi[8], chi[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        lor[j] = unsigned(vr[j]) & 0x3FFFFFFFu;
        chr[j] = unsigned(int(vr[j] >> 30) + fzqe2::P2Offset);
        loi[j] = unsigned(vi[j]) & 0x3FFFFFFFu;
        chi[j] = unsigned(int(vi[j] >> 30) + fzqe2::P2Offset);
    }
#pragma unroll 1
    for (int q = 0; q < ZL8_NMODC; ++q) {
        const ZModC c = mods[q];
        const unsigned short *tab = slut + c.lutoff;
        unsigned rr[8], ri[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rr[j] = exact_res(lor[j], chr[j], c);
            ri[j] = exact_res(loi[j], chi[j], c);
        }
        const int np = c.role == 0 ? 2 : 3;
        for (int pr = 0; pr < np; ++pr) {
            unsigned cd[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned xv;
                if (c.role == 0) { // 2M: c+ = r_re + j r_im, c- = r_re + (p - j) r_im
                    xv = modp_small(rr[j] + (pr == 0 ? unsigned(c.j) : unsigned(c.p - c.j)) * ri[j],
                                    c);
                } else { // 3M: P0 = r_re, P1 = r_im, P2 = r_re + r_im
                    xv = pr == 0 ? rr[j] : pr == 1 ? ri[j] : modp_small(rr[j] + ri[j], c);
                }
                cd[j] = tab[xv];
            }
            const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                           od = gather4(cd[1], cd[3], cd[5], cd[7]);
            unsigned char *const qb =
                tbase + plane_offset(g.cuts, c.pbase + pr * c.planes, c.planes, rank, cut, 0);
            *reinterpret_cast<unsigned *>(qb) = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
            if (c.planes >= 2) {
                *reinterpret_cast<unsigned *>(qb + TileBytes) =
                    ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
            }
            if (c.planes == 3) {
                const unsigned eh = gather4(cd[0] >> 8, cd[2] >> 8, cd[4] >> 8, cd[6] >> 8),
                               oh = gather4(cd[1] >> 8, cd[3] >> 8, cd[5] >> 8, cd[7] >> 8);
                *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) =
                    (eh & 0x0F0F0F0Fu) | ((oh & 0x0F0F0F0Fu) << 4);
            }
        }
    }
}

using fzqe2::bq16_asm;
__host__ __device__ constexpr int zl8f_part_lut_size(int q, int pr) {
    return zl8_role_ct(q) == 1 && pr == 2 ? 4 * zl8_p_ct(q) : 2 * zl8_p_ct(q);
}
__host__ __device__ constexpr int zl8f_lut_off(int q, int pr) {
    int o = 0;
    for (int a = 0; a < q; ++a) {
        for (int b = 0; b < (zl8_role_ct(a) == 0 ? 2 : 3); ++b) {
            o += zl8f_part_lut_size(a, b);
        }
    }
    for (int b = 0; b < pr; ++b) {
        o += zl8f_part_lut_size(q, b);
    }
    return o;
}
constexpr int LutWordsF = (zl8f_lut_off(ZL8_NMODC, 0) + 7) / 8 * 8;
constexpr int LutSmemF = LutWordsF * 2;
__host__ __device__ constexpr int dyn_smem_f(int side_b) {
    return LutSmemF + (side_b ? 0 : AStageZ * 8);
}
__host__ __device__ constexpr int zl8f_pbase(int q) {
    int o = 0;
    for (int a = 0; a < q; ++a) {
        o += (zl8_role_ct(a) == 0 ? 2 : 3) * zl8_planes_ct(a);
    }
    return o;
}
// every per-modulus constant of encode_zl8f in one compile-time table (a local constexpr of the kernel, indexed by the unrolled q)
struct ZL8FTab {
    int p[ZL8_NMODC], role[ZL8_NMODC], j[ZL8_NMODC], pl[ZL8_NMODC], pb[ZL8_NMODC],
        off[ZL8_NMODC][3];
};
__host__ __device__ constexpr ZL8FTab zl8f_tab() {
    ZL8FTab t{};
    for (int q = 0; q < ZL8_NMODC; ++q) {
        t.p[q] = zl8_p_ct(q);
        t.role[q] = zl8_role_ct(q);
        t.j[q] = zl8_j_ct(q);
        t.pl[q] = zl8_planes_ct(q);
        t.pb[q] = zl8f_pbase(q);
        for (int pr = 0; pr < 3; ++pr) {
            t.off[q][pr] = zl8f_lut_off(q, pr);
        }
    }
    return t;
}

__global__ __launch_bounds__(256, 3) void encode_zl8f(const float2 *__restrict__ x, int64_t ld,
                                                      const int *__restrict__ shifts,
                                                      unsigned char *__restrict__ packed,
                                                      const unsigned short *__restrict__ lutf,
                                                      int side_b, const Geom g) {
    constexpr ZL8FTab T = zl8f_tab();
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *slut = reinterpret_cast<unsigned short *>(dsm);
    float2 *sa = reinterpret_cast<float2 *>(dsm + LutSmemF); // Shared rows use the encoder stride.
    const unsigned part = blockIdx.x, cut = blockIdx.y, rank = blockIdx.z & 1u,
                   tile = blockIdx.z >> 1;
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 8, chunk = linear % 8,
                   k0 = cut * 64 + chunk * 8;
    const unsigned pos =
        side_b ? tile * Cols + rank * (Cols / 2) + row : tile * Side + rank * (Side / 2) + row;
    const bool live = k0 < unsigned(g.k) && (side_b ? row < 112 : pos < unsigned(g.m));
    for (int i = threadIdx.x; i < LutWordsF / 8; i += 256) {
        reinterpret_cast<uint4 *>(slut)[i] = reinterpret_cast<const uint4 *>(lutf)[i];
    }
    long long vr[8], vi[8];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 8, rhi = (part * 256 + 255) / 8, nr = rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        for (unsigned e = threadIdx.x; e < nr * 64; e += 256) {
            const unsigned kk = e / nr, r = e % nr, kg = cut * 64 + kk;
            sa[kk * AStrideZ + r] = kg < unsigned(g.k) && abase + r < unsigned(g.m)
                                        ? x[size_t(kg) * ld + abase + r]
                                        : make_float2(0.0, 0.0);
        }
        __syncthreads();
        const int sh = live ? shifts[pos] : 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 z = sa[(chunk * 8 + j) * AStrideZ + (row - rlo)];
            vr[j] = live ? q1i(z.x, sh) : 0;
            vi[j] = live ? q1i(z.y, sh) : 0;
        }
    } else {
        __syncthreads(); // the LUT copy
        if (live) {
            const unsigned src_row = pos % unsigned(g.n);
            const int sh = shifts[src_row];
            const float2 *src = x + size_t(src_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 z = k0 + j < unsigned(g.k) ? src[j] : make_float2(0.0, 0.0);
                vr[j] = q1i(z.x, sh);
                vi[j] = q1i(z.y, sh);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                vr[j] = vi[j] = 0;
            }
        }
    }
    const int inner = int(tile_byte(row, chunk * 4));
    unsigned char *const tbase = packed + size_t(tile) * tile_block_bytes(g.cuts, g.planes) + inner;
    if (!live) { // padding: code 0 in every plane of every part (as encode_zl8)
#pragma unroll
        for (int q = 0; q < ZL8_NMODC; ++q) {
            const int pl = T.pl[q], np = T.role[q] == 0 ? 2 : 3, pb = T.pb[q];
#pragma unroll
            for (int pr = 0; pr < 3; ++pr) {
                if (pr >= np) {
                    break;
                }
#pragma unroll
                for (int p = 0; p < 3; ++p) {
                    if (p >= pl) {
                        break;
                    }
                    *reinterpret_cast<unsigned *>(
                        tbase + plane_offset(g.cuts, pb + pr * pl, pl, rank, cut, p)) = 0u;
                }
            }
        }
        return;
    }
    unsigned lor[8], chr[8], loi[8], chi[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        lor[j] = unsigned(vr[j]) & 0x3FFFFFFFu;
        chr[j] = unsigned(int(vr[j] >> 30) + fzqe2::P2Offset);
        loi[j] = unsigned(vi[j]) & 0x3FFFFFFFu;
        chi[j] = unsigned(int(vi[j] >> 30) + fzqe2::P2Offset);
    }
    auto store = [&](const unsigned (&cd)[8], int pb0, int pl) {
        const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                       od = gather4(cd[1], cd[3], cd[5], cd[7]);
        unsigned char *const qb = tbase + plane_offset(g.cuts, pb0, pl, rank, cut, 0);
        *reinterpret_cast<unsigned *>(qb) = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
        if (pl >= 2) {
            *reinterpret_cast<unsigned *>(qb + TileBytes) =
                ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
        }
        if (pl == 3) {
            const unsigned eh = gather4(cd[0] >> 8, cd[2] >> 8, cd[4] >> 8, cd[6] >> 8),
                           oh = gather4(cd[1] >> 8, cd[3] >> 8, cd[5] >> 8, cd[7] >> 8);
            *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) =
                (eh & 0x0F0F0F0Fu) | ((oh & 0x0F0F0F0Fu) << 4);
        }
    };
#pragma unroll
    for (int q = 0; q < ZL8_NMODC; ++q) {
        const unsigned p = unsigned(T.p[q]), negp = 0u - p, mm = unsigned((1ull << 32) / p),
                       rs = unsigned((1ull << 30) % p);
        const unsigned m16 = (1u << 16) / p;
        const int pl = T.pl[q], pb = T.pb[q];
        unsigned tr[8], ti[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            tr[j] = bqn_asm(mad_asm(chr[j], rs, lor[j]), negp, mm);
            ti[j] = bqn_asm(mad_asm(chi[j], rs, loi[j]), negp, mm);
        }
        unsigned cd[8];
        if (T.role[q] == 0) { // 2M: c+, c-
            const unsigned jp = unsigned(T.j[q]), jm = p - jp;
            const unsigned short *t0 = slut + T.off[q][0];
            const unsigned short *t1 = slut + T.off[q][1];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cd[j] = t0[bq16_asm(tr[j] + jp * ti[j], negp, m16)];
            }
            store(cd, pb, pl);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cd[j] = t1[bq16_asm(tr[j] + jm * ti[j], negp, m16)];
            }
            store(cd, pb + pl, pl);
        } else { // 3M: P0 = r_re, P1 = r_im, P2 = r_re + r_im
            const unsigned short *t0 = slut + T.off[q][0];
            const unsigned short *t1 = slut + T.off[q][1];
            const unsigned short *t2 = slut + T.off[q][2];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cd[j] = t0[tr[j]];
            }
            store(cd, pb, pl);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cd[j] = t1[ti[j]];
            }
            store(cd, pb + pl, pl);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cd[j] = t2[tr[j] + ti[j]];
            }
            store(cd, pb + 2 * pl, pl);
        }
    }
}

struct ZEncoder {
    Geom geo{};
    int bits = 53;
    int fast = 1;
    unsigned short *dlutf = nullptr;
    bool a_direct = false;
    unsigned flips[2][L8K_NMOD] =
        {}; // the complex set has no sign flips (checked by make_zl8_tables.py)
    ZModC *dmods = nullptr;
    unsigned short *dlut = nullptr;
    void set_flips_for_k(int) {}
    int tiles(int side_b) const {
        return side_b ? (geo.n + Cols - 1) / Cols : (geo.m + Side - 1) / Side;
    }
    size_t packed_side_bytes(int side_b) const {
        return size_t(tiles(side_b)) * tile_block_bytes(geo.cuts, geo.planes);
    }
    void init(int bits_, int m, int n, int k, bool = false) {
        if (bits_ > 53) {
            throw std::runtime_error("zl8e: bits <= 53 in this build");
        }
        if (m <= 0 || n <= 0 || k <= 0) {
            throw std::runtime_error("zl8e: sizes");
        }
        std::vector<ZModC> hm(ZL8_NMODC);
        int pb = 0;
        for (int q = 0; q < ZL8_NMODC; ++q) {
            const ZL8Mod &z = zl8_modc_k8192[q];
            const unsigned p = unsigned(z.p);
            if (zl8_planes_of_mod[q] < 1 || zl8_planes_of_mod[q] > 3) {
                throw std::runtime_error("zl8e: planes");
            }
            hm[q] = ZModC{unsigned((1ull << 32) / p),
                          0u - p,
                          unsigned((1ull << 30) % p),
                          unsigned((1ull << 53) % p),
                          int(p),
                          zl8_planes_of_mod[q],
                          pb,
                          zl8_lut_off[q],
                          z.role,
                          z.j,
                          0,
                          0};
            pb += (z.role == 0 ? 2 : 3) * zl8_planes_of_mod[q];
        }
        bits = bits_;
        geo = Geom{m, n, k, l8e::cuts_of(k), pb};
        if (pb != zl8f_pbase(ZL8_NMODC)) {
            throw std::runtime_error("zl8e: the constant plane bases differ from the table's");
        }
        for (int q = 0; q < ZL8_NMODC; ++q) {
            if (zl8_planes_ct(q) != zl8_planes_of_mod[q] || zl8_j_ct(q) != zl8_modc_k8192[q].j ||
                zl8_p_ct(q) != zl8_modc_k8192[q].p) {
                throw std::runtime_error(
                    "zl8e: the constant tables differ from zl8_tables_rt.h / zl8_lut.h");
            }
        }
        std::vector<unsigned short> hl(LutWordsZ, 0);
        for (int i = 0; i < ZL8_LUT_WORDS; ++i) {
            hl[i] = zl8_lut[i];
        }

        std::vector<unsigned short> hf(LutWordsF, 0);
        for (int q = 0; q < ZL8_NMODC; ++q) {
            const long long p = zl8_p_ct(q),
                            rot = (long long)((1ull << 53) % (unsigned long long)p),
                            j = zl8_j_ct(q);
            const int np = zl8_role_ct(q) == 0 ? 2 : 3;
            for (int pr = 0; pr < np; ++pr) {
                const long long off = zl8_role_ct(q) == 0 ? rot * (pr == 0 ? 1 + j : 1 + p - j) % p
                                      : pr == 2           ? 2 * rot % p
                                                          : rot;
                const int o = zl8f_lut_off(q, pr), sz = zl8f_part_lut_size(q, pr);
                for (int i = 0; i < sz; ++i) {
                    hf[o + i] = zl8_lut[zl8_lut_off[q] + (((i - off) % p) + p) % p];
                }
            }
        }
        if (const char *e = awe_additional_env("ZL8_ENC")) {
            fast = atoi(e) != 0 ? 1 : 0;
        }
        release();
        if (cudaMalloc(&dmods, sizeof(ZModC) * ZL8_NMODC) != cudaSuccess ||
            cudaMalloc(&dlut, LutWordsZ * 2) != cudaSuccess ||
            cudaMalloc(&dlutf, LutWordsF * 2) != cudaSuccess ||
            cudaMemcpy(dmods, hm.data(), sizeof(ZModC) * ZL8_NMODC, cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(dlut, hl.data(), LutWordsZ * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dlutf, hf.data(), LutWordsF * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaFuncSetAttribute(encode_zl8, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_smem_z(0)) != cudaSuccess ||
            cudaFuncSetAttribute(encode_zl8f, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_smem_f(0)) != cudaSuccess) {
            throw std::runtime_error("zl8e: device setup failed");
        }
    }
    void shift(const float *x, int *shifts, int *invalid, int side_b, int64_t ld,
               cudaStream_t stream = 0) const {
        const float2 *z = reinterpret_cast<const float2 *>(x);
        if (side_b) {
            shift_rows_c<<<geo.n, 256, 0, stream>>>(z, shifts, invalid, geo.n, geo.k, ld, bits);
        } else {
            shift_cols_c<8>
                <<<(geo.m + 7) / 8, 256, 0, stream>>>(z, shifts, invalid, geo.m, geo.k, ld, bits);
        }
    }
    void launch(const float *x, int64_t ld, const int *shifts, unsigned char *packed, int side_b,
                cudaStream_t stream = 0) const {
        const dim3 grid(4, unsigned(geo.cuts), unsigned(tiles(side_b) * 2));
        if (fast) {
            encode_zl8f<<<grid, 256, dyn_smem_f(side_b), stream>>>(
                reinterpret_cast<const float2 *>(x), ld, shifts, packed, dlutf, side_b, geo);
        } else {
            encode_zl8<<<grid, 256, dyn_smem_z(side_b), stream>>>(
                reinterpret_cast<const float2 *>(x), ld, shifts, packed, dmods, dlut, side_b, geo);
        }
    }
    void release() {
        cudaFree(dmods);
        cudaFree(dlut);
        cudaFree(dlutf);
        dmods = nullptr;
        dlut = nullptr;
        dlutf = nullptr;
    }
};
} // namespace zl8e
