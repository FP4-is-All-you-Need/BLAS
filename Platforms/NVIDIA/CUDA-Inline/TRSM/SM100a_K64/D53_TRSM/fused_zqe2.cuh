#pragma once
// Fused complex encoding shares the reference quantization and packed layout.
//   2M / 3M loops apart: the 2M moduli and the 3M moduli run in two loops, so the per-element branch on
//        j = sqrt(-1) that the compiler left in the reference loop body (both paths, per element) is gone.
// Barrett without the last correction: for t < 2^30, q = umulhi(t, floor(2^32 / p)) is floor(t / p) or one less, so
// t - q p is in [0, 2p) (fp64k_encode.cuh's enc_barrett then subtracts p once). The re / im residues stay in [0, 2p);
// the 2M combinations rr + j ri < 2p + 168 * 2p < 2^30 are valid Barrett inputs.
// LAZY = 0: each residue that indexes the LUT is brought to [0, p) by min(r, r - p) (unsigned; 2 instructions instead of
//        compare, select, subtract), and the LUT is the reference (stride FP64K_LUT_STRIDE).
// LAZY = 1: no correction before the LUT; the LUT is extended (host: lutx[q][r] = lut[q][r mod p], r < XS): 2M residues
// are in [0, 2p) (p <= 169, 338 entries), 3M residues in [0, 2p) and r + i in [0, 4p) (p <= 141, 564 entries).
// PACK = 1: the 16 codes (one byte each) are gathered with byte permutes into words of the even and odd elements, then
// limb 0 = (even & 0x0F0F0F0F) | ((odd & 0x0F0F0F0F) << 4), limb 1 = ((even >> 4) & 0x0F0F0F0F) | (odd & 0xF0F0F0F0).
//        PACK = 0: the reference per-element shift and or.
// LUTSM = 1: the LUT is copied to Shared Memory once per block (32-bit addresses instead of 64-bit ones per lookup).
//   DEAD = 2: as 1, but B rows 112..127 are not written at all (only if the GEMM never loads them).
// DEAD = 1: threads with no live input (B rows 112..127 of each half tile, the K padding) write the codes of residue 0
//        for every part and skip the arithmetic (the bytes are the same: the reference encodes zeros there).
// PROBE (not byte-identical, for finding the limit): 1 = no LUT (code = residue byte), 2 = no stores (one 4-byte xor per
// thread to sink), 3 = load + quantize only.
#include <type_traits>
#include "quant_int.cuh"
#include "fp64k_encode.cuh"
#include "fp64kc_parts.h"

namespace fzqe2 {
using namespace fp64k;

constexpr int XS = 576; // extended LUT stride: 2 * 169 = 338 (2M), 4 * 141 = 564 (3M)
// the 2M moduli come first (checked on the host); parts = 2 n2M + 3 n3M with n2M + n3M = FP64K_NMOD, so n2M = 3 NMOD - parts
// Shared-memory rows are derived from the active part count.
constexpr int Z2M = 3 * FP64K_NMOD - FP64KC_NPART;
constexpr int ABlockRows = 43, AStride = ABlockRows + 1;
constexpr int AStageBytes = 96 * AStride * 16; // 67,584 (as fzqe::ASharedBytes)

__host__ __device__ constexpr int lut_stride(int lazy) {
    return lazy ? XS : FP64K_LUT_STRIDE;
}
// P2: one extended LUT per part, rotated by the part's constant offset (see encode_c2). AMIN widens the parts to 1,024 entries
// (A- = 2 rr + 2p - A+ < 6p <= 1,014).
// With AMIN only the A- parts need 1,024 entries: a 2M pair takes XS + XSA bytes, a 3M part XS (31,040 bytes for 43 parts).
constexpr int XSA = 1024;
__host__ __device__ constexpr int
part_off(int amin, int pidx) { // byte offset of part pidx's LUT (pidx = NPART: total)
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
// dynamic Shared Memory for a launch: the LUT (if LUTSM) then the A staging (A only)
__host__ __device__ constexpr int dyn_smem(int lazy, int lutsm, int side_b, int p2 = 0,
                                           int amin = 0) {
    return (lutsm ? lut_smem(lazy, p2, amin) : 0) + (side_b ? 0 : AStageBytes);
}
// P2: the high half of a quantized value is offset by 2^23 (so it is in [0, 2^24] for |v| <= 2^55); every re / im residue
// then carries + 2^55 (mod p), folded into the rotated LUTs on the host.
constexpr int P2Offset = 1 << 23;

__device__ __forceinline__ long long q1(double v, int sh) {
    return qint::finite_bits(qint::absbits(v)) ? qint::scale_round(v, sh) : 0;
}
// q1 with the sign bit cleared by an opaque 64-bit AND: from `bits & 0x7fff...` of a value loaded as a double, the compiler
// makes `DADD R, -RZ, |R|` (one FP64 instruction per value in every encoder, too; B300's FP64 unit is slow,
// SASS). The arithmetic and the result are qint::scale_round's.
__device__ __forceinline__ unsigned long long absbits_i(double z) {
    unsigned long long r;
    asm("and.b64 %0, %1, 0x7fffffffffffffff;"
        : "=l"(r)
        : "l"((unsigned long long)__double_as_longlong(z)));
    return r;
}
__device__ __forceinline__ long long q1i(double z, int shift) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(z), a = absbits_i(z);
    if (!qint::finite_bits(a) || !a) {
        return 0;
    }
    const int e = int(a >> 52);
    const unsigned long long m = (a & ((1ull << 52) - 1)) | (e ? (1ull << 52) : 0ull);
    const int s = (e ? e : 1) - 1075 + shift;
    unsigned long long v;
    if (s >= 0) {
        v = m << s;
    } else if (s < -54) {
        v = 0;
    } else {
        const int d = -s;
        v = m >> d;
        const unsigned long long rem = m & ((1ull << d) - 1), half = 1ull << (d - 1);
        if (rem > half || (rem == half && (v & 1))) {
            ++v;
        }
    }
    return (b >> 63) ? -(long long)v : (long long)v;
}
// the unreduced sum of reduce_quantized, in 32 bits: the same low 32 bits as fp64k_encode.cuh's 64-bit form, and that
// value is in [0, 2^30) there.
__device__ __forceinline__ unsigned tsum(long long v, int rc1, int rc2, unsigned ebias) {
    const int v0 = int(v & 0x3FFFF), v1 = int((v >> 18) & 0x3FFFF), v2 = int(v >> 36);
    return unsigned(v0 + v1 * rc1 + v2 * rc2) + ebias;
}
__device__ __forceinline__ unsigned bq(unsigned t, unsigned p,
                                       unsigned mmagic) { // [0, 2p) for t < 2^30
    return t - __umulhi(t, mmagic) * p;
}
// the same value as bq with the product added instead of subtracted: negp = 0 - p (mod 2^32), so t + q negp = t - q p.
// In an earlier build, SASS spent one IMAD.MOV per bq on negating q (96 of 395 IMAD in the 2M loop).
__device__ __forceinline__ unsigned bqn(unsigned t, unsigned negp,
                                        unsigned mmagic) { // [0, 2p) for t < 2^32
    return t + __umulhi(t, mmagic) * negp;
}
// ASM = 1 (P2 >= 2): the same three steps as opaque PTX, so NVVM cannot reassociate them (with plain C++ it expanded
// 2 rr through the Barrett form and reused the ch * rs products: 284 IMAD in the 2M loop where 144 were meant).
__device__ __forceinline__ unsigned mad_asm(unsigned a, unsigned b, unsigned c) { // a b + c
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
// B16: Barrett for a < 2^16 with m16 = floor(2^16 / p): q = (a m16) >> 16 is floor(a / p) or one less, so the result is in
// [0, 2p); a multiply, a shift and a multiply-add instead of IMAD.HI (which the times suggest costs about 2 IMAD).
__device__ __forceinline__ unsigned bq16_asm(unsigned a, unsigned negp, unsigned m16) {
    unsigned r;
    asm("{.reg .u32 q; mul.lo.u32 q, %1, %3; shr.u32 q, q, 16; mad.lo.u32 %0, q, %2, %1;}"
        : "=r"(r)
        : "r"(a), "r"(negp), "r"(m16));
    return r;
}
__device__ __forceinline__ unsigned amin_asm(unsigned rr, unsigned twop,
                                             unsigned a) { // 2 rr + 2p - a on the ALU
    unsigned r;
    asm("{.reg .u32 x; add.u32 x, %1, %1; add.u32 x, x, %2; sub.u32 %0, x, %3;}"
        : "=r"(r)
        : "r"(rr), "r"(twop), "r"(a));
    return r;
}
__device__ __forceinline__ unsigned fix(unsigned r, unsigned p) {
    return min(r, r - p);
} // [0, 2p) -> [0, p)

// 4 codes (one byte each, in the low byte of each word) -> one word, code i in byte i
__device__ __forceinline__ unsigned gather4(unsigned c0, unsigned c1, unsigned c2, unsigned c3) {
    return __byte_perm(__byte_perm(c0, c1, 0x0040), __byte_perm(c2, c3, 0x0040), 0x5410);
}

// P2 = 1 (needs LAZY = 1): a quantized value v is kept as its two 32-bit halves, lo = v mod 2^32 and ch = (v >> 32) + 2^23,
//   instead of three 18-bit pieces (2 registers per value instead of 3). Per modulus:
// r = bq(bq(lo) + ch * (2^32 mod p)) in [0, 2p) (bq(lo) < 2p for any 32-bit lo; the sum < 2p + 2^24 * 168 < 2^32),
// r = v + 2^55 (mod p). The LUT of part k is rotated by that constant carried into the part: 2M A+ k (1 + j),
// A- k (1 - j), 3M r and i k, r + i 2k (k = 2^55 mod p). Same instructions per value as the 18-bit form.
// P2 = 3: as P2 = 1 with bqn and 2^32 mod p from the host table rsplit (P2 = 1 called a 64-bit modulo per modulus).
// 0 - p also comes from that table: computed in the kernel, the compiler folds t + q (0 - p) back into t - q p and spends an
// IMAD.MOV on -q again.
// P2 = 2 (|v| <= 2^53 only, i.e. bits <= 53): lo = v mod 2^30, ch = (v >> 30) + 2^23 in [0, 2^24]; t = lo + ch (2^30 mod p)
// < 2^30 + 2^24 * 168 < 2^32, one bqn: 3 IMAD per value instead of 5. r = v + 2^53 (mod p); rsplit = 2^30 mod p.
// B16 = 1 (ASM): the A+- Barrett by bq16 (a = rr + j ri < 2p + 168 * 2p = 57,122 < 2^16 for p <= 169).
// AMIN = 1 (P2 >= 2): A- = 2 rr + 2p - A+ in (0, 6p) on the integer ALU instead of an IMAD and a bqn (3 IMAD). Its
// constant is k (1 - j) as before, so the same rotated LUT, 1,024 entries per part.
// P2 = 4 (bits <= 62; ASM): lo = v mod 2^32, ch = (v >> 32) + 2^30 in [0, 2^31]; both halves reduced (bqn, [0, 2p)),
// t = ch_r (2^32 mod p) + lo_r < 2p + 2p (p - 1) < 2^16, r = bq16(t) in [0, 2p). r = v + 2^62 (mod p): the LUTs are
//   rotated by 2^62 mod p. Two IMAD.HI per value instead of one (q30 needs |v| <= 2^53).
// P2 = 5 (bits <= 60; ASM): lo = v mod 2^30, ch = (v >> 30) + 2^30 in [0, 2^31]; ch reduced first (bqn, [0, 2p)), then
// t = lo + ch_r (2^30 mod p) < 2^30 + 2p (p - 1) < 2^31 and one more bqn: 2 IMAD.HI + 3 IMAD per value (P2 = 4 has
// 2 + 5, P2 = 2 has 1 + 2 but needs |v| <= 2^53). r = v + 2^60 (mod p): the LUTs are rotated by 2^60 mod p.
// FP = 1 (P2 = 2, 4 or 5, AMIN, ASM; 2M moduli only): A+- from rr, ri in [0, 2p) on the FP32 pipes instead of IMAD. Every step is exact:
// rrf, rif = rr, ri as floats (bits | 0x4B000000, minus 2^23); af = fma(j, rif, rrf) = rr + j ri < 2^16;
// q = fma(af, 1/p, 1.5 2^23) - 1.5 2^23 = the integer nearest af (1/p)_fp32, and |af (1/p)_fp32 - af / p| < 1e-4, so
// r = fma(-q, p, af) is exact with |r| <= floor(p / 2) (p odd);
// A+ index = r + OFF, A- index = 2 rr + OFF - r (OFF = floor(p / 2) + 2), both read as the bits of (index + 2^23).
// A+ index = A+ + k (1 + j) + OFF and A- index = A- + k (1 - j) + OFF (mod p): the host rotates the 2M LUTs by that
// (fcon = per modulus (j, fp32(1/p), p, OFF + 2^23) as floats). Index ranges [1, p + 3] and [1, 5p + 3] (AMIN's layout).
template <int LAZY, int PACK, int LUTSM, int DEAD, int PROBE, int P2 = 0, int AMIN = 0, int ASM = 0,
          int B16 = 0, int FP = 0,
          int QI = 0> // QI: the quantization without the FP64 abs (q1i)
__global__ __launch_bounds__(256, 2) void encode_c2(
    const double2 *__restrict__ x, int64_t ld, const int *__restrict__ shifts,
    unsigned char *__restrict__ packed, const Fp64kMod *__restrict__ mods,
    const unsigned char *__restrict__ lut, const int *__restrict__ sqrtm1, int side_b,
    unsigned *__restrict__ sink, const unsigned char *__restrict__ lut_base,
    const uint4
        *__restrict__ rsplit, // per modulus: (2^32 or 2^30 mod p, 0 - p, floor(2^16 / p), 0); P2 >= 2
    const float4 *__restrict__ fcon) { // per modulus: (j, fp32(1/p), p, OFF + 2^23); FP only
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
    double2 *sa_c =
        reinterpret_cast<double2 *>(dsm + (LUTSM ? lut_smem(LAZY, P2, AMIN) : 0)); // Shared rows use the encoder stride.
    const unsigned unit = blockIdx.x / 3, part = blockIdx.x % 3;
    const unsigned cut = unit % Cuts, rank = (unit / Cuts) % 2, tile = unit / (Cuts * 2);
    const unsigned linear = part * 256 + threadIdx.x;
    const unsigned row = linear / 6, chunk = linear % 6;
    const unsigned k0 = cut * 96 + chunk * 16;
    const bool live = (k0 + 15 < K) && (!side_b || row < 112);
    if constexpr (LUTSM) {
        for (int i = threadIdx.x; i < lut_bytes(LAZY, P2, AMIN) / 4; i += 256) {
            reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
        }
        if (side_b) {
            __syncthreads(); // A syncs after its staging below
        }
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
            for (int j = 0; j < 16; ++j) {
                vr[j] = vi[j] = 0;
            }
        }
    }
    unsigned lor[16], chr[16], loi[16], chi[16]; // P2 only
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
    // one residue of a quantized value (re: h = 0, im: h = 1) of element j, in [0, 2p)
    auto resid = [&](int j, int h, const Fp64kMod &m, unsigned p, unsigned negp, unsigned rs,
                     unsigned m16) -> unsigned {
        const unsigned lo = h ? loi[j] : lor[j], ch = h ? chi[j] : chr[j];
        if constexpr (P2 == 5) { // rs = 2^30 mod p
            return bqn_asm(mad_asm(bqn_asm(ch, negp, m.mmagic), rs, lo), negp, m.mmagic);
        } else if constexpr (P2 == 4) { // rs = 2^32 mod p
            return bq16_asm(mad_asm(bqn_asm(ch, negp, m.mmagic), rs, bqn_asm(lo, negp, m.mmagic)),
                            negp, m16);
        } else if constexpr (P2 == 2 && ASM) {
            return bqn_asm(mad_asm(ch, rs, lo), negp, m.mmagic);
        } else if constexpr (P2 == 2) {
            return bqn(lo + ch * rs, negp, m.mmagic);
        } else if constexpr (P2 == 3) {
            return bqn(bqn(lo, negp, m.mmagic) + ch * rs, negp, m.mmagic);
        } else if constexpr (P2 == 1) {
            return bq(bq(lo, p, m.mmagic) + ch * rs, p, m.mmagic);
        } else {
            return bq(tsum(h ? vi[j] : vr[j], m.rc1, m.rc2, m.ebias), p, m.mmagic);
        }
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
        for (int j = 0; j < 16; ++j) {
            acc ^= unsigned(vr[j]) ^ unsigned(vr[j] >> 32) * 3u ^ unsigned(vi[j]) * 5u ^
                   unsigned(vi[j] >> 32) * 7u;
        }
        sink[size_t(blockIdx.x) * 256 + threadIdx.x] = acc;
        return;
    }
    if constexpr (DEAD) {
        // DEAD=2 skips B rows 112..127 only when the GEMM never reads them.
        // K padding is still written because the GEMM reads the last cut.
        if (DEAD == 2 && side_b && row >= 112) {
            return;
        }
        if (!live) {
#pragma unroll 1
            for (int pidx = 0; pidx < FP64KC_NPART; ++pidx) {
                const int qm = pidx < 2 * Z2M
                                   ? pidx / 2
                                   : Z2M + (pidx - 2 * Z2M) / 3; // fp64kc_part_mod (host array)
                const unsigned code = P2 ? lut_base[qm * FP64K_LUT_STRIDE] : tab[qm * LS];
                const unsigned w0 = (code & 15u) * 0x11111111u, w1 = (code >> 4) * 0x11111111u;
                put(pidx, w0, w0, w1, w1);
            }
            if constexpr (PROBE == 2) {
                sink[size_t(blockIdx.x) * 256 + threadIdx.x] = acc;
            }
            return;
        }
    }
    // Per modulus, the codes of NP parts for element j come from code_of(j, c). They are packed as they come: PACK = 0 ors
    // each nibble into its word; PACK = 1 gathers the even and the odd elements of each half by 4 and splits
    // the nibbles once per word. Only a few codes are live at a time (the 18-bit pieces of vr / vi take most registers).
    auto modulus = [&](auto code_of, auto np_tag, int pidx0) {
        constexpr int NP = decltype(np_tag)::value;
        unsigned l0[NP][2], l1[NP][2]; // limb 0 / 1, elements 0..7 / 8..15
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
                    if constexpr (B16) {
                        a = bq16_asm(mad_asm(jj, ri, rr), negp, m16);
                    } else {
                        a = bqn_asm(mad_asm(jj, ri, rr), negp, m.mmagic);
                    }
                    if constexpr (AMIN) {
                        b = amin_asm(rr, 2 * p, a);
                    } else if constexpr (B16) {
                        b = bq16_asm(mad_asm(pj, ri, rr), negp, m16);
                    } else {
                        b = bqn_asm(mad_asm(pj, ri, rr), negp, m.mmagic);
                    }
                } else if constexpr (P2 >= 2) {
                    a = bqn(rr + jj * ri, negp, m.mmagic);
                    if constexpr (AMIN) {
                        b = 2 * rr + 2 * p - a;
                    } else {
                        b = bqn(rr + pj * ri, negp, m.mmagic);
                    }
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
                if constexpr (LAZY) {
                    s = rr + ri;
                } else {
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
    if constexpr (PROBE == 2) {
        sink[size_t(blockIdx.x) * 256 + threadIdx.x] = acc;
    }
}

} // namespace fzqe2
