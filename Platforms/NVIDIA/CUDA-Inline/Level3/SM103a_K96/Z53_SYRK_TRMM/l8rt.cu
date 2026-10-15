// Runtime-sized AWE GEMM with explicit m, n, k, lda, ldb, and ldc.
// The generated tables select the moduli, plane layout, accumulator weights, and products.
// The kernel and CRT constants precede the host plan and entry points below.
// See l8k_kernel_rt.cuh, l8rt_plan.cuh, and the encoder for the arithmetic contracts.

#include <cuda.h> // CUtensorMap
#include "Reference/awe_backend.h"
// CUBLAS_EMU=1 runs cuBLAS's FP64 emulation on the same A and B after the verdict.
#ifndef CUBLAS_EMU
#define CUBLAS_EMU 0
#endif
#if CUBLAS_EMU
#include <cublas_v2.h>
#endif
// SUSTAINED=1 adds 300 back-to-back calls of Ours and of cuBLAS fixed 53 bits with NVML clock / power samples.
#ifndef SUSTAINED
#define SUSTAINED 0
#endif
#if SUSTAINED
#include <nvml.h>
#include <thread>
#include <chrono>
#endif
#include "Reference/f64_backend.cuh"
#include "ptx_common.cuh"
#include "fp64k_common.h"
#include "duald40_load.cuh"
#include "fp64k_encode.cuh"
// PAYOUT_CRT=1 moves the CRT into the payout.
#ifndef PAYOUT_CRT
#define PAYOUT_CRT 0
#endif
// CRT_LIN=1 replaces crt_store by crt_lin (no lookup table).
#ifndef CRT_LIN
#define CRT_LIN 0
#endif
// CRT_LIN=2 = integer-built double; CRT_PROBE=1 times the CRT kernel in parts after the verdict.
#ifndef CRT_PROBE
#define CRT_PROBE 0
#endif
// RESID_U8=1 writes and reads the residues as 1 byte (p <= 169) in the same buffer.
#ifndef RESID_U8
#define RESID_U8 0
#endif
// QUANT_INT=1 quantizes our inputs with quant_int.cuh (same integers, see test_quant.cu).
#ifndef QUANT_INT
#define QUANT_INT 0
#endif
// FUSE_QE=1 computes shifts first, then quantizes FP64 input inside the encoder.
// Avoiding the qa/qb round trip is intended to preserve quantized integers and packed bytes.
#ifndef FUSE_QE
#define FUSE_QE 0
#endif
#include "quant_int.cuh"
#include "fused_qe.cuh"
#include "l8_encode.cuh"
#ifndef L8_FAST_ENC
#define L8_FAST_ENC 0
#endif
#if L8_FAST_ENC // The FastEncoder interface preserves the reference encoder layout.
#include "l8_encode_fast.cuh"
#endif
#if ZL8 // the complex GEMM on this kernel (make_zl8_tables.py, zl8_encode.cuh)
#include "zl8_tables_rt.h"
#include "zl8_encode.cuh"
#else
#include "l8k_tables_rt.h" // Runtime K-plan tables.
#endif
#ifndef L8_ADIRECT
#define L8_ADIRECT 0
#endif
#ifndef L8_SLOTS
#define L8_SLOTS 3
#endif
#ifndef L8_FOLD_TILES
#define L8_FOLD_TILES 0 // > 0: the HBM diagnostic (reads folded into L2; output not a product)
#endif
// SUPPLY_DIRECT=1: tensor copies use `.cta_group::2` and complete on rank 0's full barrier.
#ifndef SUPPLY_DIRECT
#define SUPPLY_DIRECT 0
#endif
#if SUPPLY_DIRECT
#define FULL_COUNT "1"
#else
#define FULL_COUNT "2"
#endif
#if RESID_U8
using ResT = unsigned char;
#else
using ResT = unsigned short;
#endif

namespace fp64k {

#ifndef FK_TILES
#define FK_TILES 16
#endif
constexpr int TilesPerCluster = FK_TILES;
static_assert(16 % TilesPerCluster == 0, "FK_TILES must divide 16");
constexpr int GridClusters = Tiles / TilesPerCluster;
// L8: one part = one modulus of l8k_tables.h; a ring slot holds L8K_FTS plane tiles per side
constexpr int kParts = L8K_NMOD;
constexpr int kSlots = L8_SLOTS, kOperand = L8K_FTS * TileBytes, kGroup = 2 * kOperand,
              kShared = kSlots * kGroup;
constexpr int kBoxTiles = 2; // the supply's tensor copy: 2 plane tiles (48 rows of 256 bytes)
static_assert(PAYOUT_CRT == 5 && RESID_U8 && CRT_LIN == 3 && SUPPLY_DIRECT,
              "the L8 kernel is written for this form only");

// Map cluster q and its local iteration to the fixed tile schedule.
__device__ __forceinline__ unsigned stream_tile(unsigned q, int k) {
    return ((q / 74) * TilesPerCluster + k) * 74 + q % 74;
}
__device__ __forceinline__ unsigned tile_row(unsigned tile) {
    return (tile / 296) * 8 + tile % 8;
}
__device__ __forceinline__ unsigned tile_col(unsigned tile) {
    return (tile % 296) / 8;
}

struct Metrics {
    unsigned long long core, ns, a_wait, b_wait, commit_wait, issue, payout, payout_read;
};
// residues of one tile (128 rows x 224 columns, 1 byte each) behind the slot ring.
constexpr int ResBytes = PAYOUT_CRT == 3 ? 7 * 256 * 16 : 0;
struct KernelMetrics {
    unsigned long long total;
    unsigned smid;
};

__device__ __forceinline__ unsigned long long tile_desc(unsigned address) {
    return (address >> 4) | (8ull << 16) | (24ull << 32) | (1ull << 46);
}
template <bool Collector, int Step>
__device__ __forceinline__ void face_mma(unsigned tm, unsigned long long a, unsigned long long b,
                                         unsigned as, unsigned bs, bool acc) {
#define FACE_MMA(QUAL)                                                                             \
    asm volatile("{.reg .pred p;setp.ne.u32 p,%6,0;tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_" \
                 "scale.block16" QUAL " [%0],%1,%2,%3,[%4],[%5],p;}" ::"r"(tm),                    \
                 "l"(a), "l"(b), "r"(0x90380480u), "r"(as), "r"(bs), "r"(int(acc))                 \
                 : "memory")
    if constexpr (!Collector) {
        FACE_MMA("");
    } else if constexpr (Step == 0) {
        FACE_MMA(".collector::a::fill");
    } else {
        FACE_MMA(".collector::a::lastuse");
    }
#undef FACE_MMA
}
__device__ __forceinline__ unsigned smid() {
    unsigned x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
    return x;
}

#if CRT_LIN
// crtw_q[r] = W_q r mod P, W_q = B_q inv_q, B_q = P / p_q. p_q B_q = P, so W_q r = B_q s_q (mod P) with
// s_q = (inv_q r) mod p_q < p_q; B_q s_q < P keeps one conditional subtraction per modulus, as in crt_store.
__constant__ unsigned long long crt_b[L8K_NMOD][2];
__constant__ unsigned crt_inv[L8K_NMOD], crt_pm[L8K_NMOD], crt_magic[L8K_NMOD];
// f64_impl::round_scaled with the double assembled from integers when the result is normal
// (ldexp of an exact mantissa is then exact, so the bits agree); otherwise the original.
__device__ __forceinline__ double round_scaled_int(f64_impl::U128 x, bool neg, int exponent) {
    if (!x) {
        return 0.;
    }
    const unsigned long long hi = (unsigned long long)(x >> 64), lo = (unsigned long long)x;
    const int top = hi ? 127 - __clzll(hi) : 63 - __clzll(lo);
    const int drop = max(0, top - 52);
    unsigned long long mant = (unsigned long long)(x >> drop);
    if (drop) {
        const f64_impl::U128 rem = x - ((f64_impl::U128)mant << drop), half = f64_impl::U128(1)
                                                                              << (drop - 1);
        if (rem > half || (rem == half && (mant & 1))) {
            ++mant;
        }
    }
    const int lz = __clzll(mant) - 11; // -1 only when mant == 2^53
    const unsigned long long m = lz >= 0 ? mant << lz : mant >> 1;
    const int e = drop + exponent - lz + 52 + 1023;
    if (e < 1 || e > 2046) {
        return f64_impl::round_scaled(x, neg, exponent);
    }
    return __longlong_as_double((long long)(((unsigned long long)e << 52) |
                                            (m & ((1ull << 52) - 1)) |
                                            ((unsigned long long)neg << 63)));
}
// Round: 1 = f64_impl::round_scaled, 2 = round_scaled_int, 0 = none (probe only: raw bits, not a result).
template <int Round>
__global__ void __launch_bounds__(256)
    crt_lin(const unsigned short *__restrict__ resid, size_t stride, double *__restrict__ out,
            const int *__restrict__ ea, const int *__restrict__ eb, f64_impl::U128 P) {
    constexpr size_t Groups8 = Count / 8;
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < Groups8;
         g += size_t(blockDim.x) * gridDim.x) {
        const size_t at = g * 8; // column major, ld = M: rows i0..i0+7 of column j
        const unsigned i0 = unsigned(at % M), j = unsigned(at / M);
        f64_impl::U128 u[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            u[e] = 0;
        }
#pragma unroll 2
        for (int q = 0; q < kParts; ++q) {
            const uint4 v =
                __ldcs(reinterpret_cast<const uint4 *>(resid + size_t(q) * stride + at));
            const unsigned w[4] = {v.x, v.y, v.z, v.w};
            const f64_impl::U128 b = (f64_impl::U128(crt_b[q][1]) << 64) | crt_b[q][0];
            const unsigned inv = crt_inv[q], p = crt_pm[q], mg = crt_magic[q];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned x = ((w[e / 2] >> (16 * (e % 2))) & 0xffffu) * inv; // < 169 * 169
                unsigned s = x - unsigned(((unsigned long long)x * mg) >> 32) * p;
                if (s >= p) {
                    s -= p;
                }
                u[e] += b * s;
                if (u[e] >= P) {
                    u[e] -= P;
                }
            }
        }
        const int ej = eb[j];
        double o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const bool neg = u[e] > P / 2;
            if constexpr (Round == 0) {
                o[e] = __longlong_as_double(
                    (long long)((unsigned long long)u[e] ^ (unsigned long long)(u[e] >> 64)));
            } else if constexpr (Round == 1) {
                o[e] = f64_impl::round_scaled(neg ? P - u[e] : u[e], neg, -ea[i0 + e] - ej);
            } else {
                o[e] = round_scaled_int(neg ? P - u[e] : u[e], neg, -ea[i0 + e] - ej);
            }
        }
        double2 *dst = reinterpret_cast<double2 *>(out + at);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            dst[e] = make_double2(o[2 * e], o[2 * e + 1]);
        }
    }
}
// no reduction per modulus. sum_q r_q W_q (W_q < P, 4 x 32-bit limbs) in four 64-bit column sums
// (each < 2^43), one carry, then qh = umulhi64(sum >> 96, floor(2^64 / ((P >> 96) + 1))) <= floor(sum / P),
// u = sum - qh P (mod 2^128) and two conditional subtractions: the same value in [0, P) as crt_store.
__constant__ unsigned crt_wl[L8K_NMOD][4];
__constant__ unsigned long long crt_pt_magic;
template <int Round>
__global__ void __launch_bounds__(256)
    crt_limb(const unsigned short *__restrict__ resid, size_t stride, double *__restrict__ out,
             const int *__restrict__ ea, const int *__restrict__ eb, f64_impl::U128 P) {
    constexpr size_t Groups8 = Count / 8;
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < Groups8;
         g += size_t(blockDim.x) * gridDim.x) {
        const size_t at = g * 8;
        const unsigned i0 = unsigned(at % M), j = unsigned(at / M);
        unsigned long long acc[8][4];
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                acc[e][k] = 0;
            }
#pragma unroll 2
        for (int q = 0; q < kParts; ++q) {
            const uint4 v =
                __ldcs(reinterpret_cast<const uint4 *>(resid + size_t(q) * stride + at));
            const unsigned w[4] = {v.x, v.y, v.z, v.w};
            const unsigned wl[4] = {crt_wl[q][0], crt_wl[q][1], crt_wl[q][2], crt_wl[q][3]};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const unsigned r = (w[e / 2] >> (16 * (e % 2))) & 0xffffu;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    acc[e][k] += (unsigned long long)wl[k] * r;
                }
            }
        }
        const int ej = eb[j];
        double o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned long long c = acc[e][0] >> 32, v1 = acc[e][1] + c;
            c = v1 >> 32;
            const unsigned long long v2 = acc[e][2] + c;
            c = v2 >> 32;
            const unsigned long long v3 = acc[e][3] + c; // = sum >> 96
            const f64_impl::U128 lo =
                (f64_impl::U128(((v3 & 0xffffffffull) << 32) | (v2 & 0xffffffffull)) << 64) |
                (((v1 & 0xffffffffull) << 32) | (acc[e][0] & 0xffffffffull));
            const unsigned long long qh = __umul64hi(v3, crt_pt_magic);
            f64_impl::U128 u = lo - f64_impl::U128(qh) * P;
            if (u >= P) {
                u -= P;
            }
            if (u >= P) {
                u -= P;
            }
            const bool neg = u > P / 2;
            if constexpr (Round == 0) {
                o[e] = __longlong_as_double(
                    (long long)((unsigned long long)u ^ (unsigned long long)(u >> 64)));
            } else {
                o[e] = round_scaled_int(neg ? P - u : u, neg, -ea[i0 + e] - ej);
            }
        }
        double2 *dst = reinterpret_cast<double2 *>(out + at);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            dst[e] = make_double2(o[2 * e], o[2 * e + 1]);
        }
    }
}
#if PAYOUT_CRT == 5
// the CRT of columns [c0, c1) of this thread's row of one tile, from the 18 residues this thread wrote
// (same arithmetic as crt_limb<2>).
__device__ __forceinline__ void crt_cols(const unsigned short *resid, double *cout, const int *ea,
                                         const int *eb, f64_impl::U128 P, unsigned tm_row,
                                         unsigned tn, unsigned rank, unsigned t, int c0, int c1) {
    const unsigned row = tm_row * Side + rank * (Side / 2) + (t % 128);
    const int er = -ea[row];
    for (int c = c0; c < c1; ++c) {
        const unsigned global_n = tn * Cols + (t / 128) * 112 + c;
        if (global_n >= unsigned(N)) {
            continue;
        }
        const size_t index = size_t(global_n) * M + row;
        unsigned r[L8K_NMOD];
#pragma unroll
        for (int q = 0; q < kParts; ++q) {
            r[q] = __ldcg(reinterpret_cast<const ResT *>(resid) + size_t(q) * Count + index);
        }
        unsigned long long acc[4] = {0, 0, 0, 0};
#pragma unroll
        for (int q = 0; q < kParts; ++q)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                acc[k] += (unsigned long long)crt_wl[q][k] * r[q];
            }
        unsigned long long cy = acc[0] >> 32, v1 = acc[1] + cy;
        cy = v1 >> 32;
        const unsigned long long v2 = acc[2] + cy;
        cy = v2 >> 32;
        const unsigned long long v3 = acc[3] + cy;
        const f64_impl::U128 lo =
            (f64_impl::U128(((v3 & 0xffffffffull) << 32) | (v2 & 0xffffffffull)) << 64) |
            (((v1 & 0xffffffffull) << 32) | (acc[0] & 0xffffffffull));
        f64_impl::U128 u = lo - f64_impl::U128(__umul64hi(v3, crt_pt_magic)) * P;
        if (u >= P) {
            u -= P;
        }
        if (u >= P) {
            u -= P;
        }
        const bool neg = u > P / 2;
        cout[index] = round_scaled_int(neg ? P - u : u, neg, er - eb[global_n]);
    }
}
#endif
#if CRT_PROBE
// probe: the bytes crt_lin moves (18 x uint4 in, 64 B out per thread) without its arithmetic.
__global__ void __launch_bounds__(256)
    crt_stream(const unsigned short *__restrict__ resid, size_t stride, double *__restrict__ out) {
    constexpr size_t Groups8 = Count / 8;
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < Groups8;
         g += size_t(blockDim.x) * gridDim.x) {
        const size_t at = g * 8;
        uint4 x = make_uint4(0, 0, 0, 0);
#pragma unroll 2
        for (int q = 0; q < kParts; ++q) {
            const uint4 v =
                __ldcs(reinterpret_cast<const uint4 *>(resid + size_t(q) * stride + at));
            x.x += v.x;
            x.y += v.y;
            x.z += v.z;
            x.w += v.w;
        }
        double2 *dst = reinterpret_cast<double2 *>(out + at);
        const double2 d = make_double2(__longlong_as_double(((long long)x.y << 32) | x.x),
                                       __longlong_as_double(((long long)x.w << 32) | x.z));
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            dst[e] = d;
        }
    }
}
#endif
#endif

#include "l8k_kernel_rt.cuh"

#if CUBLAS_EMU
// how far another FP64 result is from ours: count of different doubles and the largest distance in ulps.
__global__ void ulp_diff(const double *ours, const double *other, size_t n,
                         unsigned long long *out) {
    unsigned long long diff = 0, worst = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) {
        long long a = __double_as_longlong(ours[i]), b = __double_as_longlong(other[i]);
        if (a == b) {
            continue;
        }
        ++diff;
        if (a < 0) {
            a = (long long)0x8000000000000000ull - a;
        }
        if (b < 0) {
            b = (long long)0x8000000000000000ull - b;
        }
        const unsigned long long d =
            a > b ? (unsigned long long)(a - b) : (unsigned long long)(b - a);
        worst = d > worst ? d : worst;
    }
    if (diff) {
        atomicAdd(out, diff);
    }
    if (worst) {
        atomicMax(out + 1, worst);
    }
}
#endif

// Compare outputs on the device to avoid copying the full matrix to the host.
// XOR reduction makes the hash independent of block execution order.
__global__ void compare_out(const double *a, const double *b, size_t n, unsigned long long *out) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        unsigned long long x = __double_as_longlong(a[i]), y = __double_as_longlong(b[i]);
        atomicXor(out + 2, x * 0x9E3779B97F4A7C15ull + i * 0xD1342543DE82EF95ull);
        atomicXor(out + 3, y * 0x9E3779B97F4A7C15ull + i * 0xD1342543DE82EF95ull);
        if (x != y) {
            atomicAdd(out, 1ull);
            atomicMin(out + 1, (unsigned long long)i);
        }
    }
}

// Fixed-seed splitmix64 and Box-Muller generate identical input arrays for both paths.
__global__ void fill_normal(double *x, size_t n, unsigned long long seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        unsigned long long z = (i + 1) * 0x9E3779B97F4A7C15ull + seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        unsigned long long w = z * 0xD1342543DE82EF95ull + 0x2545F4914F6CDD1Dull;
        w = (w ^ (w >> 29)) * 0xBF58476D1CE4E5B9ull;
        w ^= w >> 32;
        double u1 = ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double u2 = ((w >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        x[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    }
}
} // namespace fp64k

using namespace fp64k;
using U128 = f64_impl::U128;
#include "l8k_host.cuh"
#if ZL8 // the complex table check (the real one compares with l8_tables.h)
#include "zl8_host.cuh"
#endif

static U128 mulmod(U128 a, int b, U128 p) {
    U128 s = 0;
    for (; b; b >>= 1) {
        if (b & 1) {
            s += a;
            if (s >= p) {
                s -= p;
            }
        }
        a += a;
        if (a >= p) {
            a -= p;
        }
    }
    return s;
}
#include "l8rt_plan.cuh"
#ifndef DGEMM_LIB
#define DGEMM_LIB 0
#endif
#if ZL8 // the z_* library
#include "zl8_dtab.cuh"
#elif DGEMM_LIB // the d_* library (l8rt_dtab.cuh) instead of main
#include "l8rt_dtab.cuh"
#else
#include "l8rt_main.cuh"
#endif
