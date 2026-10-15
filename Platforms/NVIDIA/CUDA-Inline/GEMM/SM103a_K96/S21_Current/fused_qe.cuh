#pragma once

#include "quant_int.cuh"
#include "fp64k_encode.cuh"

namespace fqe {
using namespace fp64k;

// the shifts of qint::quant_cols<R> (A: rows of a column-major matrix), without the int64 output
template <int R>
__global__ void __launch_bounds__(256)
    shift_cols(const double *__restrict__ x, int *exponents, int *invalid, int outer, int k,
               int64_t ld, int bits) {
    constexpr int C = 256 / R;
    __shared__ unsigned long long red[C][R];
    const int t = threadIdx.x, r = t % R, c = t / R, o0 = blockIdx.x * R, o = o0 + r;
    unsigned long long mx = 0;
    bool bad = false;
    if (o < outer) {
        for (int q = c; q < k; q += C) {
            const unsigned long long a = qint::absbits(x[size_t(q) * ld + o]);
            if (!qint::finite_bits(a)) {
                bad = true;
            } else {
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
        const int sh = m == 0 ? 0 : bits - 1 - qint::ilogb_bits(m);
        if (o0 + t < outer) {
            exponents[o0 + t] = sh;
        }
    }
}

// the shifts of qint::quant_rows (B: contiguous rows, one block per row), without the int64 output
__global__ void __launch_bounds__(256)
    shift_rows(const double *__restrict__ x, int *exponents, int *invalid, int outer, int k,
               int64_t ld, int bits) {
    __shared__ unsigned long long red[256];
    const int o = blockIdx.x, t = threadIdx.x;
    const double *row = x + size_t(o) * ld;
    unsigned long long mx = 0;
    bool bad = false;
    for (int q = t; q < k; q += 256) {
        const unsigned long long a = qint::absbits(row[q]);
        if (!qint::finite_bits(a)) {
            bad = true;
        } else {
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

#ifndef FQE_NB
#define FQE_NB 0
#endif
__device__ __forceinline__ long long scale_round_nb(double z, int shift) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(z),
                             a = b & 0x7fffffffffffffffull;
    const int e = int(a >> 52);
    const unsigned long long m = (a & ((1ull << 52) - 1)) | (e ? (1ull << 52) : 0ull);
    const int s = (e ? e : 1) - 1075 + shift;
    const unsigned d = unsigned(min(max(-s, 0), 63)), l = unsigned(min(max(s, 0), 63));
    const unsigned long long vr = m >> d, rem = m & ((1ull << d) - 1ull), half = (1ull << d) >> 1;
    const bool up = d != 0 && (rem > half || (rem == half && (vr & 1ull)));
    const unsigned long long v = s >= 0 ? (m << l) : vr + (up ? 1ull : 0ull);
    return (b >> 63) ? -(long long)v : (long long)v;
}
__device__ __forceinline__ long long
quant_one(double z, int sh) { // as qint::quant_cols / quant_rows write it
    return qint::finite_bits(qint::absbits(z))
               ? (FQE_NB ? scale_round_nb(z, sh) : qint::scale_round(z, sh))
               : 0;
}

// Rows of A one block touches: the block's 256 threads cover linear = part * 256 + t, row = linear / 6 (128 rows x 6 chunks
// over the 3 parts), so at most 43 rows.
constexpr int ABlockRows = 43, AStride = ABlockRows + 1;

#ifndef FQE_R32
#define FQE_R32 0
#endif
#ifndef FQE_LUTSM
#define FQE_LUTSM 0
#endif
__device__ __forceinline__ int reduce32(long long v, const Fp64kMod &m) {
    const int v0 = int(v & 0x3FFFF), v1 = int((v >> 18) & 0x3FFFF), v2 = int(v >> 36);
    return enc_barrett(unsigned(v0 + v1 * m.rc1 + v2 * m.rc2 + int(m.ebias)), m.p, m.mmagic);
}
#ifndef FQE_MINB
#define FQE_MINB 4 // 4 = fp64k_encode's occupancy (48 B spill on sm_103a); 3 = no spill
#endif
__global__ __launch_bounds__(256, FQE_MINB) void encode_fused(
    const double *__restrict__ x, int64_t ld, const int *__restrict__ shifts,
    unsigned char *__restrict__ packed, const Fp64kMod *__restrict__ mods,
    const unsigned char *__restrict__ lut, int side_b) {
    const unsigned unit = blockIdx.x / 3, part = blockIdx.x % 3;
    const unsigned cut = unit % Cuts, rank = (unit / Cuts) % 2, tile = unit / (Cuts * 2);
    const unsigned linear = part * 256 + threadIdx.x;
    const unsigned row = linear / 6, chunk = linear % 6;
    const unsigned k0 = cut * 96 + chunk * 16;
    const bool live = (k0 + 15 < K) && (!side_b || row < 112);
    long long v[16];
    if (!side_b) {
        // stage this block's rows x 96 k of A (column major) through Shared Memory: consecutive threads read consecutive rows
        __shared__ double sa[96][AStride];
        const unsigned rlo = (part * 256) / 6, rhi = (part * 256 + 255) / 6, nr = rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        for (unsigned e = threadIdx.x; e < nr * 96; e += 256) {
            const unsigned kk = e / nr, r = e % nr, kg = cut * 96 + kk;
            sa[kk][r] = kg < unsigned(K) ? x[size_t(kg) * ld + abase + r] : 0.0;
        }
        __syncthreads();
        if (live) {
            const int sh = shifts[abase + (row - rlo)];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                v[j] = quant_one(sa[chunk * 16 + j][row - rlo], sh);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                v[j] = 0;
            }
        }
    } else {
        if (live) {
            const unsigned source_row = (tile * Cols + rank * (Cols / 2) + row) % N;
            const int sh = shifts[source_row];
            const double2 *src =
                reinterpret_cast<const double2 *>(x + size_t(source_row) * ld + k0);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const double2 w = src[j];
                v[2 * j] = quant_one(w.x, sh);
                v[2 * j + 1] = quant_one(w.y, sh);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                v[j] = 0;
            }
        }
    }
    // from here as fp64k_encode
#if FQE_LUTSM
    __shared__ unsigned char slut[FP64K_NMOD * FP64K_LUT_STRIDE];
    __shared__ Fp64kMod smods[FP64K_NMOD];
    for (int i = threadIdx.x; i < FP64K_NMOD * FP64K_LUT_STRIDE; i += 256) {
        slut[i] = lut[i];
    }
    if (threadIdx.x < FP64K_NMOD) {
        smods[threadIdx.x] = mods[threadIdx.x];
    }
    __syncthreads();
    const Fp64kMod *mtab = smods;
    const unsigned char *ltab = slut;
#else
    const Fp64kMod *mtab = mods;
    const unsigned char *ltab = lut;
#endif
    const int inner = int(tile_byte(row, chunk * 8));
#pragma unroll 1
    for (int qm = 0; qm < FP64K_NMOD; ++qm) {
        const Fp64kMod m = mtab[qm];
        const unsigned char *table = ltab + size_t(qm) * FP64K_LUT_STRIDE;
        unsigned lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned code = table[FQE_R32 ? reduce32(v[j], m) : reduce_quantized(v[j], m)];
            unsigned c0 = code & 15u, c1 = code >> 4;
            if (j < 8) {
                lo0 |= c0 << (4 * j);
                lo1 |= c1 << (4 * j);
            } else {
                hi0 |= c0 << (4 * (j - 8));
                hi1 |= c1 << (4 * (j - 8));
            }
        }
        unsigned char *base = packed + (size_t(tile) * FP64K_NMOD + qm) * PackedBytes;
        reinterpret_cast<uint2 *>(base + tile_offset(rank, cut, 0) * TileBytes + inner)[0] =
            make_uint2(lo0, hi0);
        reinterpret_cast<uint2 *>(base + tile_offset(rank, cut, 1) * TileBytes + inner)[0] =
            make_uint2(lo1, hi1);
    }
}

} // namespace fqe
