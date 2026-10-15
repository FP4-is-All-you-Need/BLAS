

#include <cuda.h>
#include "Reference/awe_backend.h"

#ifndef CUBLAS_EMU
#define CUBLAS_EMU 0
#endif
#if CUBLAS_EMU
#include <cublas_v2.h>
#endif

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

#ifndef PAYOUT_CRT
#define PAYOUT_CRT 0
#endif

#ifndef CRT_LIN
#define CRT_LIN 0
#endif

#ifndef CRT_PROBE
#define CRT_PROBE 0
#endif

#ifndef RESID_U8
#define RESID_U8 0
#endif

#ifndef QUANT_INT
#define QUANT_INT 0
#endif

#ifndef FUSE_QE
#define FUSE_QE 0
#endif
#include "quant_int.cuh"
#include "fused_qe.cuh"
#include "l8_encode.cuh"
#ifndef L8_FAST_ENC
#define L8_FAST_ENC 0
#endif
#if L8_FAST_ENC
#include "l8_encode_fast.cuh"
#endif
#include "l8k_tables.h"
#ifndef L8_ADIRECT
#define L8_ADIRECT 0
#endif
#ifndef L8_SLOTS
#define L8_SLOTS 3
#endif
#ifndef L8_FOLD_TILES
#define L8_FOLD_TILES 0 // > 0: the HBM diagnostic (reads folded into L2; output not a product)
#endif

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
static_assert(L8K_CUTS == Cuts, "l8k_tables.h was made for another K");
static_assert(PAYOUT_CRT == 5 && RESID_U8 && CRT_LIN == 3 && SUPPLY_DIRECT,
              "the L8 kernel is written for this form only");

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

__constant__ unsigned long long crt_b[L8K_NMOD][2];
__constant__ unsigned crt_inv[L8K_NMOD], crt_pm[L8K_NMOD], crt_magic[L8K_NMOD];

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

#include "l8k_kernel.cuh"

#if CUBLAS_EMU

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
int main(int argc, char **argv) try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const int runs = argc > 1 ? atoi(argv[1]) : 4;
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || prop.minor != 3) {
        throw std::runtime_error("requires B300 sm_103a");
    }
    int runtime = 0;
    ck(cudaRuntimeGetVersion(&runtime));
    int bits = 53;
    if (const char *s = getenv("AWE_FP64_BITS")) {
        bits = atoi(s);
    }

    U128 P = 1;
    const L8kMod *L8M = L8_ADIRECT ? l8k_mods71 : l8k_mods59;
    const int l8_planes = L8_ADIRECT ? L8K_PLANES71 : L8K_PLANES59,
              l8_products = L8_ADIRECT ? L8K_PRODUCTS71 : L8K_PRODUCTS59;
    const int l8_accs = L8_ADIRECT ? L8K_ACCS71 : L8K_ACCS59;
    for (int i = 0; i < L8K_NMOD; ++i) {
        P *= (unsigned)L8M[i].p;
    }
    int pbits = 0;
    {
        U128 z = P;
        while (z) {
            ++pbits;
            z >>= 1;
        }
    }
    printf(
        "{\"kind\":\"environment\",\"engine\":\"l8k\",\"variant\":%d,\"gpu\":\"%s\",\"cuda_runtime\":%d,"
        "\"m\":%d,\"n\":%d,\"k\":%d,\"bits\":%d,\"moduli\":%d,\"planes_per_side\":%d,\"mma_products\":%d,"
        "\"accumulators_per_tile\":%d,\"product_bits\":%d,\"tile_m\":%d,\"tile_n\":%d,\"cuts\":%d,\"face_tiles_per_slot\":%d,"
        "\"slots\":%d,\"tiles_per_cluster\":%d,\"grid_clusters\":%d,\"cta_group\":2,\"lambda\":\"per accumulator\","
        "\"fold_tiles\":%d,\"encoder\":\"%s\","
        "\"reference\":\"r1b f64 (21 moduli, 59 faces, cuBLASLt)\"}\n",
        l8_products, prop.name, runtime, M, N, K, bits, L8K_NMOD, l8_planes, l8_products, l8_accs,
        pbits, Side, Cols, Cuts, L8K_FTS, kSlots, TilesPerCluster, GridClusters, L8_FOLD_TILES,
        L8_FAST_ENC ? "fast" : "base");

    double *A, *B, *Cref, *Cours;
    ck(cudaMalloc(&A, size_t(M) * K * 8));
    ck(cudaMalloc(&B, size_t(K) * N * 8));
    ck(cudaMalloc(&Cref, Count * 8));
    ck(cudaMalloc(&Cours, Count * 8));
    fill_normal<<<4096, 256>>>(A, size_t(M) * K, 20260921ull);
    fill_normal<<<4096, 256>>>(B, size_t(K) * N, 76543210ull);
    ck(cudaDeviceSynchronize());

    awe_gemm_desc gd{};
    gd.struct_size = sizeof(gd);
    gd.m = M;
    gd.n = N;
    gd.k = K;
    gd.lda = M;
    gd.ldb = K;
    gd.ldc = M;
    gd.trans_a = AWE_OP_N;
    gd.trans_b = AWE_OP_N;
    gd.data_type = AWE_F64_F64;
    gd.layout = AWE_COLUMN_MAJOR;
    gd.alpha = 1;
    gd.beta = 0;
    awe_options go{};
    go.struct_size = sizeof(go);
    go.method = AWE_METHOD_OZAKI2;
    go.stream_count = 3;
    go.buffer_count = 3;
    go.k_chunk = 0;
    go.profiling = 1;
    go.heuristic_index = 0;
    go.matmul_workspace_bytes = size_t(64) << 20;
    f64_impl::Engine eng;
    eng.init(gd, go);
    void *refwork = nullptr;
    ck(cudaMalloc(&refwork, eng.info.workspace_bytes));
    double ref_best = 1e30;
    for (int r = 0; r < runs; ++r) {
        awe_execution_stats st{};
        st.struct_size = sizeof(st);
        eng.execute(A, B, Cref, refwork, eng.info.workspace_bytes, 0, &st);
        printf(
            "{\"kind\":\"reference\",\"run\":%d,\"warmup\":%s,\"total_ms\":%.6f,\"encode_ms\":%.6f,"
            "\"gemm_ms\":%.6f,\"reconstruction_ms\":%.6f,\"faces\":%d}\n",
            r, r == 0 ? "true" : "false", st.total_ms, st.encode_ms, st.gemm_ms,
            st.reconstruction_ms, st.face_count);
        if (r) {
            ref_best = std::min<double>(ref_best, st.total_ms);
        }
    }
    ck(cudaFree(refwork));
    eng.drain();

    l8k_check_tables(L8M, L8K_NMOD, K, Cuts, L8K_FTS, kBoxTiles,
                     L8_ADIRECT != 0); // blead 15330 (b): before any launch
    int *ea, *eb, *bad;
    int *hbad;
    ck(cudaMalloc(&ea, M * 4));
    ck(cudaMalloc(&eb, N * 4));
    ck(cudaMalloc(&bad, 4));
    ck(cudaMallocHost(&hbad, 4));
#if L8_FAST_ENC
    l8e::FastEncoder enc;
#else
    l8e::Encoder enc;
#endif
    if (!L8_ADIRECT) {
        enc.set_flips_for_k(K);
    }
    enc.init(bits, M, N, K, L8_ADIRECT != 0);
    if (enc.geo.planes != l8_planes || enc.geo.cuts != Cuts) {
        throw std::runtime_error("encoder geometry differs from the engine's");
    }
    for (int qi = 0; qi < L8K_NMOD; ++qi) {
        if (enc.flips[0][qi] != (L8_ADIRECT ? 0u : l8k_flips59[qi]) || enc.flips[1][qi] != 0u) {
            throw std::runtime_error("encoder sign flips differ from the engine's table");
        }
    }
    const unsigned block_tiles = unsigned(2 * Cuts * l8_planes);
    const size_t bytesA = enc.packed_side_bytes(0), bytesB = enc.packed_side_bytes(1);
    if (bytesA != size_t(TileRows) * block_tiles * TileBytes ||
        bytesB != size_t(TileCols) * block_tiles * TileBytes) {
        throw std::runtime_error("packed sizes differ from the engine's");
    }
    unsigned char *pa, *pb;
    unsigned short *resid;
    ck(cudaMalloc(&pa, bytesA));
    ck(cudaMalloc(&pb, bytesB));

    CUtensorMap tmA, tmB;
    memset(&tmA, 0, sizeof tmA);
    memset(&tmB, 0, sizeof tmB);
    {
        typedef CUresult (*EncodeTiled)(
            CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
            const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
            CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
        void *fn = nullptr;
        cudaDriverEntryPointQueryResult qr;
        ck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &qr));
        if (!fn || qr != cudaDriverEntryPointSuccess) {
            throw std::runtime_error("cuTensorMapEncodeTiled not found");
        }
        static_assert(TileBytes % 256 == 0, "256-byte rows");
        auto make = [&](CUtensorMap *m, void *base, size_t bytes) {
            const cuuint64_t gdim[2] = {256, cuuint64_t(bytes / 256)};
            const cuuint64_t gs[1] = {256};
            const cuuint32_t box[2] = {256, cuuint32_t(kBoxTiles * TileBytes / 256)};
            const cuuint32_t es[2] = {1, 1};
            CUresult r = ((EncodeTiled)fn)(
                m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, gdim, gs, box, es,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (r != CUDA_SUCCESS) {
                throw std::runtime_error("cuTensorMapEncodeTiled failed " + std::to_string(int(r)));
            }
        };
        make(&tmA, pa, bytesA);
        make(&tmB, pb, bytesB);
    }
    ck(cudaMalloc(&resid, Count * kParts * 2));
    f64_impl::U128 *crt_state;
    ck(cudaMalloc(&crt_state, 16)); // unused by the PAYOUT_CRT=5 kernel
    L8kMod *dmods;
    ck(cudaMalloc(&dmods, sizeof(L8kMod) * L8K_NMOD));
    ck(cudaMemcpy(dmods, L8M, sizeof(L8kMod) * L8K_NMOD, cudaMemcpyHostToDevice));

    std::vector<U128> hcrtw(size_t(kParts) * FP64K_LUT_STRIDE, 0);
    for (int qi = 0; qi < kParts; ++qi) {
        int p = L8M[qi].p;
        if (p > FP64K_LUT_STRIDE) {
            throw std::runtime_error("modulus above the table stride");
        }
        U128 base = P / (unsigned)p;
        int inv = 1;
        while ((unsigned)(base % (unsigned)p) * inv % p != 1) {
            ++inv;
        }
        inv = int((unsigned)inv * L8M[qi].lam0inv % unsigned(p));
        U128 weight = base * (unsigned)inv;
        for (int r = 0; r < p; ++r) {
            hcrtw[size_t(qi) * FP64K_LUT_STRIDE + r] = mulmod(weight, r, P);
        }
    }
    U128 *dcrtw;
    ck(cudaMalloc(&dcrtw, hcrtw.size() * sizeof(U128)));
    ck(cudaMemcpy(dcrtw, hcrtw.data(), hcrtw.size() * sizeof(U128), cudaMemcpyHostToDevice));
    {
        if (P >> 125) {
            throw std::runtime_error("crt_limb assumes P < 2^125");
        }
        unsigned long long hb[L8K_NMOD][2];
        unsigned hinv[L8K_NMOD], hp[L8K_NMOD], hmg[L8K_NMOD];
        for (int qi = 0; qi < kParts; ++qi) {
            int p = L8M[qi].p;
            U128 base = P / (unsigned)p;
            int inv = 1;
            while ((unsigned)(base % (unsigned)p) * inv % p != 1) {
                ++inv;
            }
            if (base * (unsigned)p != P) {
                throw std::runtime_error("p does not divide P");
            }
            hb[qi][0] = (unsigned long long)base;
            hb[qi][1] = (unsigned long long)(base >> 64);
            hinv[qi] = unsigned(inv) * L8M[qi].lam0inv % unsigned(p);
            hp[qi] = unsigned(p);
            hmg[qi] = unsigned((1ull << 32) / unsigned(p));
        }
        ck(cudaMemcpyToSymbol(crt_b, hb, sizeof(hb)));
        ck(cudaMemcpyToSymbol(crt_inv, hinv, sizeof(hinv)));
        ck(cudaMemcpyToSymbol(crt_pm, hp, sizeof(hp)));
        ck(cudaMemcpyToSymbol(crt_magic, hmg, sizeof(hmg)));
        unsigned hwl[L8K_NMOD][4];
        for (int qi = 0; qi < kParts; ++qi) {
            const U128 w =
                hcrtw[size_t(qi) * FP64K_LUT_STRIDE + 1]; // W_q lambda0^-1 mod P = crtw_q[1]
            for (int k = 0; k < 4; ++k) {
                hwl[qi][k] = unsigned(w >> (32 * k));
            }
        }
        const unsigned long long pt1 = (unsigned long long)(P >> 96) + 1;
        const unsigned long long hmagic = (unsigned long long)((U128(1) << 64) / pt1);
        ck(cudaMemcpyToSymbol(crt_wl, hwl, sizeof(hwl)));
        ck(cudaMemcpyToSymbol(crt_pt_magic, &hmagic, sizeof(hmagic)));
    }
    std::vector<f64_impl::CrtMod> hcm(kParts);
    for (int qi = 0; qi < kParts; ++qi) {
        hcm[qi] =
            f64_impl::CrtMod{resid + size_t(qi) * Count, dcrtw + size_t(qi) * FP64K_LUT_STRIDE};
    }
    f64_impl::CrtMod *dcm;
    ck(cudaMalloc(&dcm, kParts * sizeof(f64_impl::CrtMod)));
    ck(cudaMemcpy(dcm, hcm.data(), kParts * sizeof(f64_impl::CrtMod), cudaMemcpyHostToDevice));

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2 * GridClusters);
    cfg.blockDim = dim3(384);
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cfg.stream = 0;
    cudaFuncAttributes ka{};
    ck(cudaFuncGetAttributes(&ka, l8k_gemm_kernel<true>));
    if (ka.sharedSizeBytes + kShared > 227 * 1024) {
        throw std::runtime_error("Shared capacity");
    }
    ck(cudaFuncSetAttribute(l8k_gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            kShared));
    cfg.dynamicSmemBytes = kShared;
    printf("{\"kind\":\"kernel\",\"registers\":%d,\"static_shared\":%zu,\"dynamic_shared\":%d}\n",
           ka.numRegs, ka.sharedSizeBytes, kShared);

    KernelMetrics *km;
    Metrics *mm;
    ck(cudaMalloc(&km, size_t(Tiles) * 2 * sizeof(KernelMetrics)));
    ck(cudaMalloc(&mm, size_t(Tiles) * kParts * 2 * sizeof(Metrics)));

    cudaEvent_t e0, e1, e2, e3, e4;
    ck(cudaEventCreate(&e0));
    ck(cudaEventCreate(&e1));
    ck(cudaEventCreate(&e2));
    ck(cudaEventCreate(&e3));
    ck(cudaEventCreate(&e4));
    double ours_best = 1e30;
    for (int r = 0; r < runs; ++r) {
        ck(cudaMemsetAsync(Cours, 0xa5, Count * 8, 0));
        ck(cudaMemsetAsync(pa, 0xa5, bytesA, 0));
        ck(cudaMemsetAsync(pb, 0xa5, bytesB, 0));
        ck(cudaMemsetAsync(resid, 0xa5, Count * kParts * 2, 0));
        ck(cudaMemsetAsync(bad, 0, 4, 0));
        ck(cudaMemsetAsync(km, 0, size_t(Tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(mm, 0, size_t(Tiles) * kParts * 2 * sizeof(Metrics), 0));
        ck(cudaEventRecord(e0, 0));
        enc.shift(A, ea, bad, 0, gd.lda);
        enc.shift(B, eb, bad, 1, gd.ldb);
        ck(cudaEventRecord(e1, 0));
        enc.launch(A, gd.lda, ea, pa, 0);
        enc.launch(B, gd.ldb, eb, pb, 1);
        ck(cudaEventRecord(e2, 0));
        ck(cudaLaunchKernelEx(&cfg, l8k_gemm_kernel<true>, (const unsigned char *)pa,
                              (const unsigned char *)pb, resid, (const L8kMod *)dmods, block_tiles,
                              km, mm, crt_state, (const f64_impl::U128 *)dcrtw, Cours,
                              (const int *)ea, (const int *)eb, P, tmA, tmB));
        ck(cudaGetLastError());
        ck(cudaEventRecord(e3, 0));
        if (!PAYOUT_CRT) {
#if CRT_LIN
#if CRT_LIN == 3
            crt_limb<2><<<unsigned((Count / 8 + 255) / 256), 256, 0, 0>>>(
                resid, Count, Cours, (const int *)ea, (const int *)eb, P);
#else
            crt_lin<CRT_LIN == 2 ? 2 : 1><<<unsigned((Count / 8 + 255) / 256), 256, 0, 0>>>(
                resid, Count, Cours, (const int *)ea, (const int *)eb, P);
#endif
#else
            f64_impl::crt_store<<<f64_impl::blocks(Count), 256, 0, 0>>>(kParts, dcm, Cours, M, N, M,
                                                                        gd.ldc, false, P, ea, eb);
#endif
            ck(cudaGetLastError());
        }
        ck(cudaEventRecord(e4, 0));
        ck(cudaEventSynchronize(e4));
        ck(cudaMemcpy(hbad, bad, 4, cudaMemcpyDeviceToHost));
        if (*hbad) {
            throw std::runtime_error("input out of range");
        }
        float q_ms, e_ms, g_ms, c_ms, tot;
        ck(cudaEventElapsedTime(&q_ms, e0, e1));
        ck(cudaEventElapsedTime(&e_ms, e1, e2));
        ck(cudaEventElapsedTime(&g_ms, e2, e3));
        ck(cudaEventElapsedTime(&c_ms, e3, e4));
        ck(cudaEventElapsedTime(&tot, e0, e4));
        printf("{\"kind\":\"ours\",\"run\":%d,\"warmup\":%s,\"total_ms\":%.6f,\"quantize_ms\":%.6f,"
               "\"encode_ms\":%.6f,\"gemm_ms\":%.6f,\"crt_ms\":%.6f}\n",
               r, r == 0 ? "true" : "false", tot, q_ms, e_ms, g_ms, c_ms);
        if (r) {
            ours_best = std::min<double>(ours_best, tot);
        }
    }

    unsigned long long *dcmp, hcmp[4];
    ck(cudaMalloc(&dcmp, 4 * sizeof(unsigned long long)));
    ck(cudaMemset(dcmp, 0, 4 * sizeof(unsigned long long)));
    {
        unsigned long long big = ~0ull;
        ck(cudaMemcpy(dcmp + 1, &big, 8, cudaMemcpyHostToDevice));
    }
    compare_out<<<4096, 256>>>(Cref, Cours, Count, dcmp);
    ck(cudaGetLastError());
    ck(cudaMemcpy(hcmp, dcmp, sizeof(hcmp), cudaMemcpyDeviceToHost));
    int match = hcmp[0] == 0;
    printf("{\"kind\":\"verdict\",\"byte_match\":%s,\"elements\":%zu,\"mismatched\":%llu,"
           "\"first_mismatch\":%llu,\"ref_hash\":\"%016llx\",\"ours_hash\":\"%016llx\"}\n",
           match ? "true" : "false", Count, hcmp[0], match ? 0ull : hcmp[1], hcmp[2], hcmp[3]);
    printf("{\"kind\":\"summary\",\"reference_best_ms\":%.6f,\"ours_best_ms\":%.6f,\"ratio\":%.4f,"
           "\"byte_match\":%s}\n",
           ref_best, ours_best, ours_best / ref_best, match ? "true" : "false");

#if CRT_LIN && CRT_PROBE

    {
        const size_t pad = Count + 4160; // planes 2 * pad bytes apart instead of 2^27
        unsigned short *rpad;
        ck(cudaMalloc(&rpad, pad * kParts * 2));
        for (int q = 0; q < kParts; ++q) {
            ck(cudaMemcpy(rpad + size_t(q) * pad, resid + size_t(q) * Count, Count * 2,
                          cudaMemcpyDeviceToDevice));
        }
        const unsigned nb = unsigned((Count / 8 + 255) / 256);
        const int *cea = (const int *)ea;
        const int *ceb = (const int *)eb;
        auto timeit = [&](const char *name, auto &&launch) {
            float best = 1e30f;
            for (int r = 0; r < 4; ++r) {
                ck(cudaEventRecord(e0, 0));
                launch();
                ck(cudaGetLastError());
                ck(cudaEventRecord(e1, 0));
                ck(cudaEventSynchronize(e1));
                float ms;
                ck(cudaEventElapsedTime(&ms, e0, e1));
                if (r) {
                    best = std::min(best, ms);
                }
            }
            printf("{\"kind\":\"probe\",\"name\":\"%s\",\"best_ms\":%.6f}\n", name, best);
        };
        timeit("crt_store", [&] {
            f64_impl::crt_store<<<f64_impl::blocks(Count), 256, 0, 0>>>(kParts, dcm, Cours, M, N, M,
                                                                        gd.ldc, false, P, ea, eb);
        });
        timeit("crt_lin_round_f64",
               [&] { crt_lin<1><<<nb, 256, 0, 0>>>(resid, Count, Cours, cea, ceb, P); });
        timeit("crt_lin_round_int",
               [&] { crt_lin<2><<<nb, 256, 0, 0>>>(resid, Count, Cours, cea, ceb, P); });
        timeit("crt_lin_no_round",
               [&] { crt_lin<0><<<nb, 256, 0, 0>>>(resid, Count, Cours, cea, ceb, P); });
        timeit("crt_limb_no_round",
               [&] { crt_limb<0><<<nb, 256, 0, 0>>>(resid, Count, Cours, cea, ceb, P); });
        timeit("crt_limb_round_int",
               [&] { crt_limb<2><<<nb, 256, 0, 0>>>(resid, Count, Cours, cea, ceb, P); });
        timeit("bytes_only", [&] { crt_stream<<<nb, 256, 0, 0>>>(resid, Count, Cours); });
        timeit("bytes_only_planes_offset",
               [&] { crt_stream<<<nb, 256, 0, 0>>>(rpad, pad, Cours); });
        timeit("crt_lin_round_int_planes_offset",
               [&] { crt_lin<2><<<nb, 256, 0, 0>>>(rpad, pad, Cours, cea, ceb, P); });
        timeit("write_only_memset", [&] { ck(cudaMemsetAsync(Cours, 0, Count * 8, 0)); });
        ck(cudaFree(rpad));
    }
#endif

#if CUBLAS_EMU

    {
        auto cb = [](cublasStatus_t s, const char *what) {
            if (s != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("cuBLAS ") + what + " " +
                                         std::to_string(int(s)));
            }
        };
        cublasHandle_t h;
        cb(cublasCreate(&h), "create");
        int ver = 0;
        cb(cublasGetVersion(h, &ver), "version");
        double *Ce;
        ck(cudaMalloc(&Ce, Count * 8));
        const size_t ewsb = size_t(4) << 30;
        void *ews;
        ck(cudaMalloc(&ews, ewsb));
        unsigned long long *dd;
        ck(cudaMalloc(&dd, 2 * sizeof(unsigned long long)));
        const double one = 1.0, zero = 0.0;

        for (int setting = 0; setting < (ver >= 130800 ? 1 : 0); ++setting) {
            const bool fixed = true;
            cb(cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH), "math mode");
            cb(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER), "strategy");
            cb(cublasSetFixedPointEmulationMantissaControl(
                   h, fixed ? CUDA_EMULATION_MANTISSA_CONTROL_FIXED
                            : CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC),
               "control");
            if (fixed) {
                cb(cublasSetFixedPointEmulationMaxMantissaBitCount(h, 53), "bits");
            }
            cb(cublasSetWorkspace(h, ews, ewsb), "workspace");
            float ms[4] = {0, 0, 0, 0}, best = 1e30f;
            for (int r = 0; r < 4; ++r) {
                ck(cudaMemsetAsync(Ce, 0xa5, Count * 8, 0));
                ck(cudaEventRecord(e0, 0));
                cb(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, gd.lda, B, gd.ldb,
                               &zero, Ce, M),
                   "dgemm");
                ck(cudaEventRecord(e1, 0));
                ck(cudaEventSynchronize(e1));
                ck(cudaEventElapsedTime(&ms[r], e0, e1));
                if (r) {
                    best = std::min(best, ms[r]);
                }
            }
            ck(cudaMemset(dd, 0, 2 * sizeof(unsigned long long)));
            ulp_diff<<<4096, 256>>>(Cours, Ce, Count, dd);
            ck(cudaGetLastError());
            unsigned long long hd[2];
            ck(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
            printf(
                "{\"kind\":\"cublas_emu\",\"setting\":\"%s\",\"cublas_version\":%d,\"best_ms\":%.6f,"
                "\"runs_ms\":[%.6f,%.6f,%.6f,%.6f],\"workspace_bytes\":%zu,\"elements\":%zu,"
                "\"differ_from_ours\":%llu,\"max_ulp_from_ours\":%llu}\n",
                fixed ? "cublas_fixed53" : "cublas_dynamic", ver, best, ms[0], ms[1], ms[2], ms[3],
                ewsb, Count, hd[0], hd[1]);
        }
        if (ver < 130800) {
            printf(
                "{\"kind\":\"cublas_emu\",\"status\":\"skipped_version\",\"cublas_version\":%d}\n",
                ver);
        }
        cudaFree(dd);
        cudaFree(ews);
        cudaFree(Ce);
        cublasDestroy(h);
    }
#endif

#if SUSTAINED

    {
        nvmlDevice_t nd{};
        bool nv_ok = false;
        {
            char bus[32] = {0};
            if (cudaDeviceGetPCIBusId(bus, sizeof(bus), 0) == cudaSuccess &&
                nvmlInit_v2() == NVML_SUCCESS &&
                nvmlDeviceGetHandleByPciBusId_v2(bus, &nd) == NVML_SUCCESS) {
                nv_ok = true;
            }
        }
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            size_t n = v.size();
            return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0;
        };
        struct Res {
            std::vector<double> ms, clk, pw;
            double energy_j = -1, wall_s = 0;
        };
        auto sample = [&](Res &r) {
            if (!nv_ok) {
                return;
            }
            unsigned mhz = 0, mw = 0;
            if (nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) {
                r.clk.push_back(mhz);
            }
            if (nvmlDeviceGetPowerUsage(nd, &mw) == NVML_SUCCESS) {
                r.pw.push_back(mw / 1000.0);
            }
        };
        auto run_block = [&](int R, auto &&launch) {
            Res r;
            std::vector<cudaEvent_t> st(R), en(R);
            for (int i = 0; i < R; ++i) {
                ck(cudaEventCreate(&st[i]));
                ck(cudaEventCreate(&en[i]));
            }
            unsigned long long e0 = 0, e1 = 0;
            bool eok = nv_ok && nvmlDeviceGetTotalEnergyConsumption(nd, &e0) == NVML_SUCCESS;
            const auto w0 = std::chrono::steady_clock::now();
            for (int i = 0; i < R; ++i) {
                ck(cudaEventRecord(st[i], 0));
                launch();
                ck(cudaEventRecord(en[i], 0));
                sample(r);
            }
            while (cudaEventQuery(en[R - 1]) == cudaErrorNotReady) {
                sample(r);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            ck(cudaEventSynchronize(en[R - 1]));
            r.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
            if (eok && nvmlDeviceGetTotalEnergyConsumption(nd, &e1) == NVML_SUCCESS) {
                r.energy_j = (e1 - e0) / 1000.0;
            }
            r.ms.resize(R);
            for (int i = 0; i < R; ++i) {
                float x;
                ck(cudaEventElapsedTime(&x, st[i], en[i]));
                r.ms[i] = x;
            }
            for (int i = 0; i < R; ++i) {
                cudaEventDestroy(st[i]);
                cudaEventDestroy(en[i]);
            }
            return r;
        };
        auto report = [&](const char *who, const char *phase, const Res &r) {
            const size_t R = r.ms.size();
            const double f10 = median(std::vector<double>(r.ms.begin(), r.ms.begin() + 10));
            const double l30 = median(std::vector<double>(r.ms.end() - 30, r.ms.end()));
            printf(
                "{\"kind\":\"sustained\",\"who\":\"%s\",\"phase\":\"%s\",\"reps\":%zu,\"first10_median_ms\":%.6f,"
                "\"last30_median_ms\":%.6f,\"all_median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"last30_over_first10\":%.6f,"
                "\"sm_clock_median_mhz\":%.1f,\"power_median_w\":%.1f,\"nvml_samples\":%zu,\"energy_j\":%.3f,"
                "\"energy_per_call_j\":%.5f,\"wall_s\":%.4f,\"mean_power_w\":%.1f}\n",
                who, phase, R, f10, l30, median(r.ms), *std::min_element(r.ms.begin(), r.ms.end()),
                *std::max_element(r.ms.begin(), r.ms.end()), l30 / f10, median(r.clk), median(r.pw),
                r.clk.size(), r.energy_j, r.energy_j >= 0 ? r.energy_j / R : -1.0, r.wall_s,
                r.energy_j >= 0 ? r.energy_j / r.wall_s : -1.0);
        };
        auto ours = [&] {
            enc.shift(A, ea, bad, 0, gd.lda);
            enc.shift(B, eb, bad, 1, gd.ldb);
            enc.launch(A, gd.lda, ea, pa, 0);
            enc.launch(B, gd.ldb, eb, pb, 1);
            ck(cudaLaunchKernelEx(&cfg, l8k_gemm_kernel<true>, (const unsigned char *)pa,
                                  (const unsigned char *)pb, resid, (const L8kMod *)dmods,
                                  block_tiles, km, mm, crt_state, (const f64_impl::U128 *)dcrtw,
                                  Cours, (const int *)ea, (const int *)eb, P, tmA, tmB));
            ck(cudaGetLastError());
            if (!PAYOUT_CRT) {
#if CRT_LIN
#if CRT_LIN == 3
                crt_limb<2><<<unsigned((Count / 8 + 255) / 256), 256, 0, 0>>>(
                    resid, Count, Cours, (const int *)ea, (const int *)eb, P);
#else
                crt_lin<CRT_LIN == 2 ? 2 : 1><<<unsigned((Count / 8 + 255) / 256), 256, 0, 0>>>(
                    resid, Count, Cours, (const int *)ea, (const int *)eb, P);
#endif
#else
                f64_impl::crt_store<<<f64_impl::blocks(Count), 256, 0, 0>>>(
                    kParts, dcm, Cours, M, N, M, gd.ldc, false, P, ea, eb);
#endif
                ck(cudaGetLastError());
            }
        };
        cublasHandle_t h2;
        if (cublasCreate(&h2) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate");
        }
        int ver = 0;
        cublasGetVersion(h2, &ver);
        if (ver < 130800) {
            printf(
                "{\"kind\":\"sustained\",\"status\":\"skipped_version\",\"cublas_version\":%d}\n",
                ver);
        } else {
            double *Ce;
            ck(cudaMalloc(&Ce, Count * 8));
            const size_t ewsb = size_t(4) << 30;
            void *ews;
            ck(cudaMalloc(&ews, ewsb));
            const double one = 1.0, zero = 0.0;
            cublasSetMathMode(h2, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
            cublasSetEmulationStrategy(h2, CUBLAS_EMULATION_STRATEGY_EAGER);
            cublasSetFixedPointEmulationMantissaControl(h2, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
            cublasSetFixedPointEmulationMaxMantissaBitCount(h2, 53);
            cublasSetWorkspace(h2, ews, ewsb);
            auto cub = [&] {
                if (cublasDgemm(h2, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, gd.lda, B, gd.ldb,
                                &zero, Ce, M) != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublasDgemm");
                }
            };
            cub();
            ck(cudaDeviceSynchronize());                           // warm-up (plan, workspace)
            std::this_thread::sleep_for(std::chrono::seconds(20)); // cold GPU
            const Res rc = run_block(300, cub);
            report("cublas_fixed53", "first_from_cold", rc);
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const Res ro = run_block(300, ours);
            report("ours", "second_after_cublas", ro);
            { // the output after 300 calls is still the reference, byte for byte
                unsigned long long *d2, hh[4];
                ck(cudaMalloc(&d2, 4 * sizeof(unsigned long long)));
                ck(cudaMemset(d2, 0, 4 * sizeof(unsigned long long)));
                {
                    unsigned long long big = ~0ull;
                    ck(cudaMemcpy(d2 + 1, &big, 8, cudaMemcpyHostToDevice));
                }
                compare_out<<<4096, 256>>>(Cref, Cours, Count, d2);
                ck(cudaGetLastError());
                ck(cudaMemcpy(hh, d2, sizeof(hh), cudaMemcpyDeviceToHost));
                printf(
                    "{\"kind\":\"sustained_verdict\",\"byte_match\":%s,\"mismatched\":%llu,\"ours_hash\":\"%016llx\"}\n",
                    hh[0] == 0 ? "true" : "false", hh[0], hh[3]);
                cudaFree(d2);
            }
            // alternating with both hot: 4 rounds of (Ours 50, cuBLAS 50)
            std::vector<double> ao, ac;
            double eo = 0, ec = 0;
            bool eok = true;
            for (int round = 0; round < 4; ++round) {
                const Res x = run_block(50, ours);
                ao.insert(ao.end(), x.ms.begin(), x.ms.end());
                const Res y = run_block(50, cub);
                ac.insert(ac.end(), y.ms.begin(), y.ms.end());
                if (x.energy_j < 0 || y.energy_j < 0) {
                    eok = false;
                } else {
                    eo += x.energy_j;
                    ec += y.energy_j;
                }
            }
            printf(
                "{\"kind\":\"sustained_alt\",\"rounds\":4,\"per_round\":50,\"ours_median_ms\":%.6f,\"cublas_median_ms\":%.6f,"
                "\"ours_over_cublas\":%.6f,\"ours_energy_per_call_j\":%.5f,\"cublas_energy_per_call_j\":%.5f}\n",
                median(ao), median(ac), median(ao) / median(ac), eok ? eo / ao.size() : -1.0,
                eok ? ec / ac.size() : -1.0);
            cudaFree(ews);
            cudaFree(Ce);
        }
        cublasDestroy(h2);
        if (nv_ok) {
            nvmlShutdown();
        }
    }
#endif

    {
        std::vector<KernelMetrics> hkm(size_t(Tiles) * 2);
        std::vector<Metrics> hmm(size_t(Tiles) * kParts * 2);
        ck(cudaMemcpy(hkm.data(), km, hkm.size() * sizeof(KernelMetrics), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(hmm.data(), mm, hmm.size() * sizeof(Metrics), cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> tot;
        tot.reserve(Tiles);
        for (int i = 0; i < Tiles; ++i) {
            tot.push_back(hkm[size_t(i) * 2].total);
        }
        std::sort(tot.begin(), tot.end());
        unsigned long long core = 0, aw = 0, bw = 0, iss = 0, pay = 0, pread = 0;
        for (size_t i = 0; i < hmm.size(); i += 2) {
            core += hmm[i].core;
            aw += hmm[i].a_wait;
            bw += hmm[i].b_wait;
            iss += hmm[i].issue;
            pay += hmm[i].payout;
            pread += hmm[i].payout_read;
        }
        const double n = double(Tiles) * kParts;
        printf(
            "{\"kind\":\"cycles\",\"tile_total_median\":%llu,\"per_modulus_core_mean\":%.1f,"
            "\"a_wait_mean\":%.1f,\"b_wait_mean\":%.1f,\"issue_mean\":%.1f,\"payout_mean\":%.1f,\"payout_read_mean\":%.1f,"
            "\"mma_bound_per_modulus\":%d}\n",
            tot[Tiles / 2], core / n, aw / n, bw / n, iss / n, pay / n, pread / n,
            Cuts * l8_products * 112 / L8K_NMOD);
    }
    return match ? 0 : 4;
} catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
} catch (f64_impl::Failure f) {
    fprintf(stderr, "reference backend failure status %d\n", (int)f.status);
    return 2;
}
