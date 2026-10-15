#pragma once

#include <stdexcept>
#include <vector>
#include "fused_zqe2.cuh"

#ifndef L8E_TABLES_FILE
#define L8E_TABLES_FILE "l8_tables.h"
#endif
#ifndef L8E_FLIPS_FILE
#define L8E_FLIPS_FILE "l8_flips.h"
#endif
#include L8E_TABLES_FILE
#include L8E_FLIPS_FILE

namespace l8e {
using namespace fp64k;
using fzqe2::q1i;
using fzqe2::gather4;
using fzqe2::bqn_asm;
using fzqe2::mad_asm;
using fzqe2::absbits_i;

struct Geom { // planes: per side, 50 or 38
    int m, n, k, cuts, planes;
    unsigned *cs =
        nullptr; // FX 1024 (l8_probe.cuh): the row sums of the counted accumulators, A's m rows then B's n columns
};
__host__ __device__ constexpr int cuts_of(int k) {
    return ((k + 95) / 96 + 1) / 2 * 2;
}
struct ModC {
    unsigned mmagic, negp, rs, lutoff;
    int planes, pbase, pad0, pad1;
}; // per modulus, 32 bytes
constexpr int LutWords = [] {
    int s = 0;
    for (int q = 0; q < L8_NMOD; ++q) {
        s += 2 * l8_mod[q];
    }
    return s;
}(); // u16 entries
constexpr int LutSmem = (LutWords * 2 + 15) / 16 * 16;
constexpr int ARows = 23, AStride = ARows + 1; // rows of a 256-thread block: ceil(256 / 12) + 1
constexpr int AStage = 96 * AStride * 8;       // doubles
__host__ __device__ constexpr int dyn_smem(int side_b) {
    return LutSmem + (side_b ? 0 : AStage);
}
__host__ __device__ inline size_t tile_block_bytes(int cuts, int planes) {
    return size_t(2) * cuts * planes * TileBytes;
}

// ---- the shift pass, real (fzqe_shift's forms: 16 rows x 4 loads for A, 8 loads for B; |x| by an opaque AND) ----
__device__ __forceinline__ bool rfinite(unsigned long long a) {
    return a < 0x7FF0000000000000ull;
}
template <int R, int U>
__global__ void __launch_bounds__(256)
    shift_cols_r(const double *__restrict__ x, int *exponents, int *invalid, int outer, int k,
                 int64_t ld, int bits) {
    constexpr int C = 256 / R;
    __shared__ unsigned long long red[C][R];
    const int t = threadIdx.x, r = t % R, c = t / R, o0 = blockIdx.x * R, o = o0 + r;
    unsigned long long mx[U] = {};
    bool bad = false;
    if (o < outer) {
        int q = c;
        for (; q + (U - 1) * C < k; q += U * C) {
            unsigned long long a[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                a[u] = absbits_i(x[size_t(q + u * C) * ld + o]);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (!rfinite(a[u])) {
                    bad = true;
                } else {
                    mx[u] = a[u] > mx[u] ? a[u] : mx[u];
                }
            }
        }
        for (; q < k; q += C) {
            const unsigned long long a = absbits_i(x[size_t(q) * ld + o]);
            if (!rfinite(a)) {
                bad = true;
            } else {
                mx[0] = a > mx[0] ? a : mx[0];
            }
        }
    }
#pragma unroll
    for (int u = 1; u < U; ++u) {
        mx[0] = mx[u] > mx[0] ? mx[u] : mx[0];
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[c][r] = mx[0];
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
template <int U>
__global__ void __launch_bounds__(256)
    shift_rows_r(const double *__restrict__ x, int *exponents, int *invalid, int outer, int k,
                 int64_t ld, int bits) {
    __shared__ unsigned long long red[256];
    const int o = blockIdx.x, t = threadIdx.x;
    const double *row = x + size_t(o) * ld;
    unsigned long long mx[U] = {};
    bool bad = false;
    int q = t;
    for (; q + (U - 1) * 256 < k; q += U * 256) {
        unsigned long long a[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            a[u] = absbits_i(row[q + u * 256]);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!rfinite(a[u])) {
                bad = true;
            } else {
                mx[u] = a[u] > mx[u] ? a[u] : mx[u];
            }
        }
    }
    for (; q < k; q += 256) {
        const unsigned long long a = absbits_i(row[q]);
        if (!rfinite(a)) {
            bad = true;
        } else {
            mx[0] = a > mx[0] ? a : mx[0];
        }
    }
#pragma unroll
    for (int u = 1; u < U; ++u) {
        mx[0] = mx[u] > mx[0] ? mx[u] : mx[0];
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[t] = mx[0];
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

// byte offset of plane p of modulus (pbase, planes) for (rank, cut) inside a tile block, before the in-tile byte
__host__ __device__ __forceinline__ size_t plane_offset(int cuts, int pbase, int planes,
                                                        unsigned rank, unsigned cut, int p) {
    return size_t(2) * cuts * TileBytes * pbase +
           ((size_t(rank) * cuts + cut) * planes + p) * TileBytes;
}

__global__ __launch_bounds__(256, 3) void encode_l8(const double *__restrict__ x, int64_t ld,
                                                    const int *__restrict__ shifts,
                                                    unsigned char *__restrict__ packed,
                                                    const ModC *__restrict__ mods,
                                                    const unsigned short *__restrict__ lut,
                                                    int side_b, const Geom g) {
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *slut = reinterpret_cast<unsigned short *>(dsm);
    double *sa = reinterpret_cast<double *>(dsm + LutSmem); // [96][AStride]
    const unsigned part = blockIdx.x, cut = blockIdx.y, rank = blockIdx.z & 1u,
                   tile = blockIdx.z >> 1;
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 12, chunk = linear % 12,
                   k0 = cut * 96 + chunk * 8;
    const unsigned pos =
        side_b ? tile * Cols + rank * (Cols / 2) + row : tile * Side + rank * (Side / 2) + row;
    const bool live = k0 < unsigned(g.k) && (side_b ? row < 112 : pos < unsigned(g.m));
    for (int i = threadIdx.x; i < LutWords / 2; i += 256) {
        reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
    }
    long long v[8];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 12, rhi = (part * 256 + 255) / 12, nr = rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        for (unsigned e = threadIdx.x; e < nr * 96; e += 256) {
            const unsigned kk = e / nr, r = e % nr, kg = cut * 96 + kk;
            sa[kk * AStride + r] = kg < unsigned(g.k) && abase + r < unsigned(g.m)
                                       ? x[size_t(kg) * ld + abase + r]
                                       : 0.0;
        }
        __syncthreads();
        const int sh = live ? shifts[pos] : 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = live ? q1i(sa[(chunk * 8 + j) * AStride + (row - rlo)], sh) : 0;
        }
    } else {
        __syncthreads(); // the LUT copy
        if (live) {
            const unsigned src_row = pos % unsigned(g.n);
            const int sh = shifts[src_row];
            const double *src = x + size_t(src_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = k0 + j < unsigned(g.k) ? q1i(src[j], sh) : 0;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = 0;
            }
        }
    }
    const int inner = int(tile_byte(row, chunk * 4));
    unsigned char *const tbase = packed + size_t(tile) * tile_block_bytes(g.cuts, g.planes) + inner;
    if (!live) { // padding: code 0 in every plane
#pragma unroll 1
        for (int q = 0; q < L8_NMOD; ++q) {
            const ModC c = mods[q];
            for (int p = 0; p < c.planes; ++p) {
                *reinterpret_cast<unsigned *>(
                    tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, p)) = 0u;
            }
        }
        return;
    }
    unsigned lo[8], ch[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        lo[j] = unsigned(v[j]) & 0x3FFFFFFFu;
        ch[j] = unsigned(int(v[j] >> 30) + fzqe2::P2Offset);
    }
#pragma unroll 1
    for (int q = 0; q < L8_NMOD; ++q) {
        const ModC c = mods[q];
        const unsigned short *tab = slut + c.lutoff;
        unsigned cd[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            cd[j] = tab[bqn_asm(mad_asm(ch[j], c.rs, lo[j]), c.negp, c.mmagic)];
        }
        const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                       od = gather4(cd[1], cd[3], cd[5], cd[7]);
        unsigned char *const qb = tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, 0);
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

// FP4 E2M1 code of an integer plane value (value / 2 is the FP4 value); -1 when not representable
inline int fp4_code(int v) {
    static const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    const int a = v < 0 ? -v : v;
    for (int i = 0; i < 8; ++i) {
        if (mag[i] == a) {
            return v < 0 && a ? (8 | i) : i;
        }
    }
    return -1;
}

struct Encoder {
    Geom geo{};
    int bits = 53;
    unsigned flips[2][L8_NMOD] = {}; // per side, per modulus: bit p negates plane p
    ModC *dmods = nullptr;
    unsigned short *dlut[2] = {nullptr, nullptr};
    void set_flip(int side_b, int q, unsigned mask) {
        flips[side_b][q] = mask;
    }
    // The plan in l8_flips.h selects A-side flips of the smallest planned K >= k, for the 59-product form (call
    // before init). a_direct = true (the 71-product form): mechanism A moduli get no flips from this plan (its weights are for the
    // 3-product form); a plan for the 4-product form with negative weights sets them with set_flip after this, before init.
    void set_flips_for_k(int k, bool a_direct_plan = false) {
        const unsigned *f = k <= 4096    ? l8_flip_a_K4096
                            : k <= 8192  ? l8_flip_a_K8192
                            : k <= 16384 ? l8_flip_a_K16384
                                         : nullptr;
        if (!f) {
            throw std::runtime_error("l8e: no plan for K > 16384");
        }
        for (int q = 0; q < L8_NMOD; ++q) {
            flips[0][q] = a_direct_plan && l8_kind[q] == 1 ? 0u : f[q];
            flips[1][q] = 0;
        }
    }
    int tiles(int side_b) const {
        return side_b ? (geo.n + Cols - 1) / Cols : (geo.m + Side - 1) / Side;
    }
    size_t packed_side_bytes(int side_b) const {
        return size_t(tiles(side_b)) * tile_block_bytes(geo.cuts, geo.planes);
    }
    bool a_direct = false;
    int planes_of(int q) const {
        return l8_kind[q] == 1 && a_direct ? 2 : l8_planes[q];
    }
    // the planes' codes of residue r of modulus q on one side (host; the tables' contents)
    unsigned codes(int side_b, int q, int r) const {
        const int a0 = l8_limbs[l8_limb_off[q] + r][0], a1 = l8_limbs[l8_limb_off[q] + r][1];
        const int val[3] = {a0, a1, a0 + a1};
        unsigned w = 0;
        for (int p = 0; p < planes_of(q); ++p) {
            const int s = (flips[side_b][q] >> p) & 1 ? -val[p] : val[p];
            const int cd = fp4_code(s);
            if (cd < 0) {
                throw std::runtime_error("l8e: plane value not in FP4");
            }
            w |= unsigned(cd) << (4 * p);
        }
        return w;
    }
    void init(int bits_, int m, int n, int k, bool a_direct_ = false) {
        if (bits_ > 53) {
            throw std::runtime_error("l8e: bits <= 53");
        }
        if (m <= 0 || n <= 0 || k <= 0) {
            throw std::runtime_error("l8e: sizes");
        }
        a_direct = a_direct_;
        int total = 0;
        for (int q = 0; q < L8_NMOD; ++q) {
            // a flip must name a plane this form has (the 71-product form keeps 2 planes for mechanism A)
            if ((flips[0][q] | flips[1][q]) >> planes_of(q)) {
                throw std::runtime_error("l8e: a flip names a plane the form does not have");
            }
            total += planes_of(q);
        }
        bits = bits_;
        geo = Geom{m, n, k, cuts_of(k), total};
        std::vector<ModC> hm(L8_NMOD);
        std::vector<unsigned short> hl[2] = {std::vector<unsigned short>(LutWords),
                                             std::vector<unsigned short>(LutWords)};
        int off = 0, pb = 0;
        for (int q = 0; q < L8_NMOD; ++q) {
            const unsigned p = unsigned(l8_mod[q]);
            if (l8_planes[q] < 1 || l8_planes[q] > 3 || (l8_kind[q] == 1) != (l8_planes[q] == 3)) {
                throw std::runtime_error("l8e: table");
            }
            hm[q] = ModC{unsigned((1ull << 32) / p),
                         0u - p,
                         unsigned((1ull << 30) % p),
                         unsigned(off),
                         planes_of(q),
                         pb,
                         0,
                         0};
            const long long rot =
                (long long)((1ull << 53) % p); // ch carries + 2^23 = + 2^53 / 2^30 (mod p)
            for (int side = 0; side < 2; ++side) {
                for (unsigned xx = 0; xx < 2 * p; ++xx) {
                    const int r = int((((long long)xx - rot) % (long long)p + p) % p);
                    hl[side][off + xx] = (unsigned short)codes(side, q, r);
                }
            }
            off += 2 * int(p);
            pb += planes_of(q);
        }
        release();
        if (cudaMalloc(&dmods, sizeof(ModC) * L8_NMOD) != cudaSuccess ||
            cudaMalloc(&dlut[0], LutWords * 2) != cudaSuccess ||
            cudaMalloc(&dlut[1], LutWords * 2) != cudaSuccess ||
            cudaMemcpy(dmods, hm.data(), sizeof(ModC) * L8_NMOD, cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(dlut[0], hl[0].data(), LutWords * 2, cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(dlut[1], hl[1].data(), LutWords * 2, cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaFuncSetAttribute(encode_l8, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_smem(0)) != cudaSuccess) {
            throw std::runtime_error("l8e: device setup failed");
        }
    }

    void shift(const double *x, int *shifts, int *invalid, int side_b, int64_t ld,
               cudaStream_t stream = 0) const {
        if (side_b) {
            shift_rows_r<1><<<geo.n, 256, 0, stream>>>(x, shifts, invalid, geo.n, geo.k, ld, bits);
        } else {
            shift_cols_r<8, 1>
                <<<(geo.m + 7) / 8, 256, 0, stream>>>(x, shifts, invalid, geo.m, geo.k, ld, bits);
        }
    }
    void launch(const double *x, int64_t ld, const int *shifts, unsigned char *packed, int side_b,
                cudaStream_t stream = 0) const {
        const dim3 grid(6, unsigned(geo.cuts), unsigned(tiles(side_b) * 2));
        encode_l8<<<grid, 256, dyn_smem(side_b), stream>>>(x, ld, shifts, packed, dmods,
                                                           dlut[side_b], side_b, geo);
    }
    void release() {
        cudaFree(dmods);
        cudaFree(dlut[0]);
        cudaFree(dlut[1]);
        dmods = nullptr;
        dlut[0] = dlut[1] = nullptr;
    }
};

} // namespace l8e
