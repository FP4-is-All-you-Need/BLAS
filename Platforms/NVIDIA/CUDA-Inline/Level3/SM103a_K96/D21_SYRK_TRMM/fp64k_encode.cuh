#pragma once
// Encode quantized int64 values directly into the packed per-modulus limb layout.
// Input integers come from the FP64 quantizer.
// A block writes all moduli to read the quantized input only once.
// The LUT maps residues modulo p to two limbs.
// tile_byte and tile_offset define the packed layout; stores are eight bytes wide.
#include "fp64k_common.h"

namespace fp64k {

// Barrett reduction requires t < 2^30.
__device__ __forceinline__ int enc_barrett(unsigned t, int p, unsigned mmagic) {
    unsigned q = (unsigned)(((unsigned long long)t * mmagic) >> 32);
    unsigned r = t - q * (unsigned)p;
    if (r >= (unsigned)p) {
        r -= (unsigned)p;
    }
    return (int)r;
}

// Reduce quantized v modulo p; |v| < 2^bits <= 2^55.
// Use the same split as reduce_quantized, with constants from the generated table.
__device__ __forceinline__ int reduce_quantized(long long v, const Fp64kMod &m) {
    int v0 = (int)(v & 0x3FFFF);
    int v1 = (int)((v >> 18) & 0x3FFFF);
    long long v2 = v >> 36;
    long long t = (long long)v0 + (long long)v1 * m.rc1 + v2 * m.rc2 + (long long)m.ebias;
    return enc_barrett((unsigned)t, m.p, m.mmagic);
}

// Grid covers tiles, ranks, cuts, and row groups; each block has 256 threads.
// Each thread handles 16 k values and writes two limbs per modulus in eight-byte stores.
__global__ __launch_bounds__(256, 4) void fp64k_encode(const long long *__restrict__ q,
                                                       unsigned char *__restrict__ packed,
                                                       const Fp64kMod *__restrict__ mods,
                                                       const unsigned char *__restrict__ lut,
                                                       int side_b) {
    const unsigned unit = blockIdx.x / 3, part = blockIdx.x % 3;
    const unsigned cut = unit % Cuts, rank = (unit / Cuts) % 2, tile = unit / (Cuts * 2);
    const unsigned linear = part * 256 + threadIdx.x;
    const unsigned row = linear / 6, chunk = linear % 6;
    const unsigned k0 = cut * 96 + chunk * 16;
    // Each CTA covers 112 B columns; the remaining 16 rows are padding ignored by payout.
    const bool live = (k0 + 15 < K) && (!side_b || row < 112);
    long long v[16];
    if (live) {
        const unsigned source_row = side_b ? (tile * Cols + rank * (Cols / 2) + row) % N
                                           : tile * Side + rank * (Side / 2) + row;
        const int4 *src = reinterpret_cast<const int4 *>(q + size_t(source_row) * K + k0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int4 w = src[j];
            v[2 * j] = (long long)(unsigned)w.x | ((long long)w.y << 32);
            v[2 * j + 1] = (long long)(unsigned)w.z | ((long long)w.w << 32);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = 0; // Padding maps residue zero to FP4 code zero.
        }
    }
    const int inner = int(tile_byte(row, chunk * 8));
#pragma unroll 1
    for (int qm = 0; qm < FP64K_NMOD; ++qm) {
        const Fp64kMod m = mods[qm];
        const unsigned char *table = lut + size_t(qm) * FP64K_LUT_STRIDE;
        unsigned lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned code = table[reduce_quantized(v[j], m)];
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

} // namespace fp64k
