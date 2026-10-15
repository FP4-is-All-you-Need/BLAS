#ifndef KSEG
#define KSEG 1
#endif
#ifndef ISSUE_WARP
#define ISSUE_WARP 1
#endif
#ifndef A_SMEM
#define A_SMEM 1
#endif
#ifndef DIRECT
#define DIRECT 1
#endif
#ifndef DSLOTS
#define DSLOTS 2
#endif
#ifndef A_FILL_ST
#define A_FILL_ST 0
#endif
#ifndef RELAYS
#define RELAYS 1
#endif
#ifndef PAYOUT_LDS
#define PAYOUT_LDS 2
#endif
#ifndef CUTS_PER_REQ
#define CUTS_PER_REQ 4
#endif
#ifndef TILE_N
#define TILE_N 224
#endif
#ifndef A_STAGES
#define A_STAGES 5
#endif
#ifndef A_TMEM_STAGES
#define A_TMEM_STAGES 5
#endif
#ifndef B_STAGES
#define B_STAGES 5
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda.h>
#include "packed_1526.h"
#include "duald40_load.cuh"
namespace d1 {
struct DimsT {
    int M, N, RowTiles, ColTiles, Tiles, TilesPerPair, TilesPerBand, TailS, FullWaves, FullTiles, TailTiles;
};
__constant__ DimsT gd;
#define M (gd.M)
#define N (gd.N)
#define RowTiles (gd.RowTiles)
#define ColTiles (gd.ColTiles)
#define Tiles (gd.Tiles)
#define TilesPerPair (gd.TilesPerPair)
#define TilesPerBand (gd.TilesPerBand)
#define TailS (gd.TailS)
#define FullWaves (gd.FullWaves)
#define FullTiles (gd.FullTiles)
#define TailTiles (gd.TailTiles)
#define R16(v)                                                                                                         \
    "+r"(v[0]), "+r"(v[1]), "+r"(v[2]), "+r"(v[3]), "+r"(v[4]), "+r"(v[5]), "+r"(v[6]), "+r"(v[7]), "+r"(v[8]),        \
        "+r"(v[9]), "+r"(v[10]), "+r"(v[11]), "+r"(v[12]), "+r"(v[13]), "+r"(v[14]), "+r"(v[15])
__device__ __forceinline__ void ld16nw(unsigned addr, unsigned (&v)[16]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
                 : "r"(addr)
                 : "memory");
}
__device__ __forceinline__ void waitld16(unsigned (&v)[16]) {
    asm volatile("tcgen05.wait::ld.sync.aligned;" : R16(v)::"memory");
}
__device__ __forceinline__ void tie16(unsigned (&v)[16]) { asm volatile("" : R16(v)::"memory"); }

#ifndef A_STAGES
#define A_STAGES 2
#endif
#ifndef B_STAGES
#define B_STAGES 3
#endif
#ifndef A_TMEM_STAGES
#define A_TMEM_STAGES 4
#endif
#ifndef CUTS_PER_REQ
#define CUTS_PER_REQ 4
#endif
constexpr int CutsPerReq = CUTS_PER_REQ;
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef DSLOTS
#define DSLOTS 2
#endif
constexpr int DSlots = DSLOTS;
#ifndef KSEG
#define KSEG 0
#endif
#ifndef KFIX
#define KFIX 0
#endif
#ifndef COMBINED_AB
#define COMBINED_AB 0
#endif
#ifndef A_SMEM
#define A_SMEM 0
#endif
#ifndef DIRECT
#define DIRECT 0
#endif
#ifndef ISSUE_WARP
#define ISSUE_WARP 0
#endif
#ifndef PAYOUT_WIDE
#define PAYOUT_WIDE 0
#endif
#ifndef PAYOUT_LDS
#define PAYOUT_LDS 4
#endif
#ifndef A_LIMBS
#define A_LIMBS 0
#endif
constexpr int ARowBytes = A_LIMBS ? 24 : 48;
#ifndef A_FILL_LDG
#define A_FILL_LDG 0
#endif
#ifndef A_ARRIVE_RELAY
#define A_ARRIVE_RELAY 1
#endif
constexpr bool Relay = A_FILL_ST && A_ARRIVE_RELAY;
#ifndef RELAYS
#define RELAYS 1
#endif
constexpr int Relays = RELAYS;
#ifndef A_RELEASE_LATE
#define A_RELEASE_LATE 1
#endif
#ifndef A_FILL_ST
#define A_FILL_ST 0
#endif
#ifndef WSPLIT
#define WSPLIT 1
#endif
constexpr int WSplit = WSPLIT;
static_assert(CUTS_PER_REQ % WSPLIT == 0, "cuts per request must split evenly over the writer groups");
constexpr int XR = 416 + 128 * WSplit;
#ifndef PAYOUT16
#define PAYOUT16 0
#endif
constexpr int XP = (A_FILL_ST ? XR + (RELAYS > 2 ? 64 : 0) : 416);
constexpr int Threads = (A_SMEM && DIRECT && !A_FILL_ST && !PAYOUT16) ? 352 : XP + (PAYOUT16 ? 256 : 0);
#ifndef MN
#define MN 8192
#endif
constexpr int TileM = 256, TileN = TILE_N, HalfN = TileN / 2, CtaRows = 128;
constexpr int Grid = 148, Pairs = Grid / 2;
#ifndef FASTCVT
#define FASTCVT 0
#endif
__device__ __forceinline__ int f2i_int24(unsigned u) {
    const unsigned e = (u >> 23) & 255u;
    const int m = int((u & 0x7FFFFFu) | 0x800000u);
    const int sh = 150 - int(e);
    int a = sh > 0 ? (m >> (sh > 31 ? 31 : sh)) : (m << (-sh));
    a = e == 0u ? 0 : a;
    return (u >> 31) ? -a : a;
}
#ifndef KSEG_STOREV4
#define KSEG_STOREV4 0
#endif
#ifndef KSEG_NOSTORE
#define KSEG_NOSTORE 0
#endif
#ifndef L2PROMO
#define L2PROMO 3
#endif
#ifndef BAND
#define BAND 8
#endif
constexpr int RowTilesPerBand = BAND;
constexpr int ATile = CtaRows * ARowBytes * CutsPerReq, BTile = HalfN * 48 * CutsPerReq, ACut = CtaRows * ARowBytes,
              BCut = HalfN * 48;
constexpr int AStages = A_STAGES, BStages = B_STAGES, ATStages = A_TMEM_STAGES, BOffset = AStages * ATile,
              SharedBytes = BOffset + BStages * BTile;
static_assert(SharedBytes <= 231424,
              "dynamic Shared Memory: the part reports opt-in 232,448 (7139 smem_optin) minus 1,024 static");
constexpr int SfbCols = TileN > 128 ? 16 : 8; // B scale form: 16 columns for N>128, 8 for N<=128 (665)
constexpr int TmemD = 0, TmemSfa = DSlots * TileN, TmemSfb = TmemSfa + 8, TmemA = TmemSfb + SfbCols,
              TmemAStage = 12 * CutsPerReq; // D slots, A scales 8 cols, B scales
static_assert(TmemA + (A_SMEM ? 0 : ATStages * TmemAStage) <= 512, "TMEM columns"); // A_SMEM: no TMEM A ring
static_assert(!A_FILL_LDG || (A_FILL_ST && A_ARRIVE_RELAY),
              "LDG fill needs the st writers with the local arrive + relay");
static_assert(!A_LIMBS || (A_FILL_ST && !A_FILL_LDG), "3-limb A needs the Shared st fill");
static_assert(!ISSUE_WARP || A_FILL_ST || A_SMEM,
              "whole-warp elect is only wired where no single-lane cp thread commits (st fill or A from Shared)");
static_assert(!DIRECT || (ATile % 256 == 0 && BTile % 256 == 0 && ATile / 256 <= 256 && BTile / 256 <= 256),
              "tensor-map boxes of 256-byte rows");
static_assert(!A_SMEM || (DIRECT && !A_FILL_ST && A_TMEM_STAGES == A_STAGES),
              "A_SMEM: direct cp-path build, the A ring is the Shared A ring");
static_assert(!COMBINED_AB || (A_SMEM && DIRECT && A_STAGES == B_STAGES),
              "combined A+B stage barriers need A_SMEM + DIRECT and equal stage counts");
static_assert(Relays == 1 || (ATStages % Relays == 0 && BStages % Relays == 0),
              "each stage must always have the same relay");
__host__ __device__ constexpr int tile_byte(int row, int w) {
    return (row / 8) * 384 + (w / 16) * 128 + (row % 8) * 16 + w % 16;
}
__device__ __forceinline__ unsigned tile_row(unsigned tile) {
    const unsigned band = tile / TilesPerBand, r0 = band * RowTilesPerBand,
                   h = min(unsigned(RowTilesPerBand), unsigned(RowTiles) - r0);
    return r0 + (tile - band * TilesPerBand) % h;
}
__device__ __forceinline__ bool item_of(int q, int j, unsigned &tile, int &part) {
    if (TailS == 1 || j < FullWaves) {
        tile = unsigned(j * Pairs + q);
        part = -1;
        return tile < unsigned(Tiles);
    }
    const int id = (j - FullWaves) * Pairs + q;
    if (id >= TailTiles * TailS)
        return false;
    tile = unsigned(FullTiles + id / TailS);
    part = id % TailS;
    return true;
}
__device__ __forceinline__ unsigned tile_col(unsigned tile) {
    const unsigned band = tile / TilesPerBand, r0 = band * RowTilesPerBand,
                   h = min(unsigned(RowTilesPerBand), unsigned(RowTiles) - r0);
    return (tile - band * TilesPerBand) / h;
}

__device__ __forceinline__ unsigned sa(const void *p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(unsigned b, unsigned c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(b), "r"(c) : "memory");
}
__device__ __forceinline__ void await(unsigned b, unsigned parity) {
    asm volatile(
        "{.reg .pred p;Lw: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra Lw;}" ::"r"(b),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void arrive_local(unsigned b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];" ::"r"(b) : "memory");
}
#ifndef ISSUE_WARP
#define ISSUE_WARP 0
#endif
#if ISSUE_WARP
#define ELECT_PFX "{.reg .pred e__;elect.sync _|e__,0xffffffff;@e__ "
#define ELECT_SFX "}"
#else
#define ELECT_PFX ""
#define ELECT_SFX ""
#endif
__device__ __forceinline__ void done_both(unsigned b) {
    asm volatile(
        ELECT_PFX
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;" ELECT_SFX::
            "r"(b),
        "h"((unsigned short)3)
        : "memory");
}
__device__ __forceinline__ void done_leader(unsigned b) {
    asm volatile(
        ELECT_PFX
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;" ELECT_SFX::
            "r"(b),
        "h"((unsigned short)1)
        : "memory");
}
__device__ __forceinline__ unsigned peer0(unsigned a) {
    unsigned b;
    asm volatile("mapa.shared::cluster.u32 %0,%1,0;" : "=r"(b) : "r"(a));
    return b;
}
#ifndef REMOTE_RELAXED
#define REMOTE_RELAXED 0
#endif
__device__ __forceinline__ void arrive_remote(unsigned b) {
#if REMOTE_RELAXED
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::"r"(b) : "memory");
#else
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(b) : "memory");
#endif
}
__device__ __forceinline__ void tst(unsigned a, unsigned v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long tile_desc(unsigned address) {
    return ((address >> 4) & 0x3fffull) | (8ull << 16) | (24ull << 32) | (1ull << 46);
}
constexpr unsigned IDESC = 0x480u | ((unsigned(TileN) >> 3) << 17) | (2u << 27) |
                           0x80000000u; // M=256 (pair), N=TileN, E2M1, UE4M3, K=96, SFA_ID=SFB_ID=0
__device__ __forceinline__ void mma_s(unsigned d, unsigned long long a_desc, unsigned long long b_desc, unsigned sfa,
                                      unsigned sfb, bool acc);
__device__ __forceinline__ void mma1(unsigned d, unsigned a_tmem, unsigned long long b_desc, unsigned sfa, unsigned sfb,
                                     bool acc) {
    asm volatile("{.reg .pred p;setp.ne.u32 p,%5,0;" ELECT_PFX
                 "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0],[%1],%2,%3,[%4],[%6],p;" ELECT_SFX
                 "}" ::"r"(d),
                 "r"(a_tmem), "l"(b_desc), "r"(IDESC), "r"(sfa), "r"(int(acc)), "r"(sfb)
                 : "memory");
}
__device__ __forceinline__ void mma_s(unsigned d, unsigned long long a_desc, unsigned long long b_desc, unsigned sfa,
                                      unsigned sfb, bool acc) {
    asm volatile("{.reg .pred p;setp.ne.u32 p,%5,0;" ELECT_PFX
                 "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0],%1,%2,%3,[%4],[%6],p;" ELECT_SFX
                 "}" ::"r"(d),
                 "l"(a_desc), "l"(b_desc), "r"(IDESC), "r"(sfa), "r"(int(acc)), "r"(sfb)
                 : "memory");
}
__device__ __forceinline__ unsigned smid_() {
    unsigned x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
    return x;
}
struct Metrics {
    unsigned long long cb, fi, le, dwait, seen, pend, ns0, ns1, cta_begin_ns, pay_end_ns, begin0, aw, bw, wt, wa, wst,
        warr, wt1, wa1, wst1, warr1;
    unsigned smid;
};
struct Scales {
    unsigned a0, a1, b0, b1;
}; // A: cols 0-3 = SF0..SF3, cols 4-7 = SF4,SF5; B: cols 0-7 / 8-15 the same bytes

// Threads: 0..255 payout (both CTAs), 256 issue (rank 0), 288 A supply, 320 B supply (both CTAs), 352/353 relay of rank
// 1's landings to rank 0 (531's form), 384 Shared -> TMEM copy of A with tcgen05.cp.cta_group::2 (rank 0; each CTA's
// own smem -> TMEM).
__global__ __launch_bounds__(Threads, 1) void p2_kernel(const __grid_constant__ CUtensorMap tmA,
                                                        const __grid_constant__ CUtensorMap tmB,
                                                        const unsigned char *__restrict__ A,
                                                        const unsigned char *__restrict__ B, float *C, Metrics *mt,
                                                        int cuts, Scales sc, const int *skip) {
    if (skip && *skip)
        return;
    const unsigned q = blockIdx.x / 2, t = threadIdx.x;
    auto cluster = cooperative_groups::this_cluster();
    const unsigned rank = cluster.block_rank();
    const auto begin0 = clock64();
    unsigned long long gbegin;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(gbegin));
    __shared__ unsigned base;
    __shared__ __align__(8) unsigned long long afull[AStages], alocal[AStages], aempty[AStages], aready[ATStages],
        aready_r[ATStages], tmem_empty[ATStages], bfull[BStages], blocal[BStages], bempty[BStages], drain[DSlots],
        read_done[DSlots];
    extern __shared__ __align__(1024) unsigned char sm[];
    if (t == 0) {
        for (int s = 0; s < AStages; ++s) {
            mbar_init(sa(&afull[s]), COMBINED_AB ? 2 : ((A_FILL_ST || DIRECT) ? 1 : 2));
            mbar_init(sa(&alocal[s]), 1);
            mbar_init(sa(&aempty[s]), A_FILL_ST ? 4 * WSplit : 1);
        }
        for (int s = 0; s < ATStages; ++s) {
            mbar_init(sa(&aready[s]), Relay ? 4 * WSplit : (A_FILL_ST ? 8 : 1));
            mbar_init(sa(&aready_r[s]), 1);
            mbar_init(sa(&tmem_empty[s]), 1);
        }
        for (int s = 0; s < BStages; ++s) {
            mbar_init(sa(&bfull[s]), DIRECT ? 1 : 2);
            mbar_init(sa(&blocal[s]), 1);
            mbar_init(sa(&bempty[s]), 1);
        }
        for (int s = 0; s < DSlots; ++s) {
            mbar_init(sa(&drain[s]), 1);
            mbar_init(sa(&read_done[s]), 2);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    cluster.sync();
    if (t < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],512;" ::"r"(sa(&base)) : "memory");
    __syncthreads();
    cluster.sync();
    const unsigned tm = base, local = tm + ((t / 32 % 4 * 32) << 16);
    if (t < 128) {
        for (int c = 0; c < 8; ++c)
            tst(local + TmemSfa + c, c < 4 ? sc.a0 : sc.a1);
        for (int c = 0; c < SfbCols; ++c)
            tst(local + TmemSfb + c, c < SfbCols / 2 ? sc.b0 : sc.b1);
        asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    __syncthreads();
    cluster.sync();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const unsigned sm_base = sa(sm);
    const int reqs = (cuts + CutsPerReq - 1) / CutsPerReq, padded = reqs * CutsPerReq;
    const size_t a_tile_bytes = size_t(padded) * ACut, b_tile_bytes = size_t(padded) * BCut;
    if ((t == 288 && !A_FILL_LDG) || t == 320) {
        const bool side_b = t == 320;
        const int depth = side_b ? BStages : AStages;
        const unsigned bytes = side_b ? BTile : ATile;
        int G = -1;
        for (int k = 0;; ++k) {
            unsigned tile;
            int part;
            if (!item_of(q, k, tile, part))
                break;
            const unsigned char *src = side_b ? (B + size_t(2 * tile_col(tile) + rank) * b_tile_bytes)
                                              : (A + size_t(2 * tile_row(tile) + rank) * a_tile_bytes);
            const int rb = part < 0 ? 0 : part * (reqs / TailS), re = part < 0 ? reqs : rb + reqs / TailS;
            for (int r = rb; r < re; ++r) {
                ++G;
                const int slot = G % depth;
                if (G >= depth)
                    await(sa((side_b && !COMBINED_AB) ? &bempty[slot] : &aempty[slot]), ((G / depth) + 1) & 1);
                const unsigned dest = sm_base + (side_b ? BOffset + slot * BTile : slot * ATile);
                if (DIRECT && (side_b || !A_FILL_ST)) {
                    const unsigned lbar_local = sa((side_b && !COMBINED_AB) ? &bfull[slot] : &afull[slot]),
                                   lbar = peer0(lbar_local);
                    const size_t off = size_t(src - (side_b ? B : A)) + size_t(r) * bytes;
                    if (rank == 0)
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ::"r"(lbar_local),
                            "r"(2u * bytes)
                            : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                            "r"(dest),
                        "l"(reinterpret_cast<unsigned long long>(side_b ? &tmB : &tmA)), "r"(0), "r"(int(off / 256)),
                        "r"(lbar)
                        : "memory");
                } else {
                    const unsigned bar =
                        sa((rank == 0 || (A_FILL_ST && !side_b)) ? (side_b ? &bfull[slot] : &afull[slot])
                                                                 : (side_b ? &blocal[slot] : &alocal[slot]));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%2],[%3],%1,[%0];" ::
                            "r"(bar),
                        "r"(bytes), "r"(dest), "l"(src + size_t(r) * bytes)
                        : "memory");
                }
            }
        }
    } else if (Relay && rank == 1 &&
               (t == 352 || t == 353 || (Relays > 1 && (t == 384 || t == 385)) ||
                (Relays > 2 && (t == XR || t == XR + 1 || t == XR + 32 || t == XR + 33)))) {
        const bool side_b = t & 1;
        const int ri = t < 384 ? 0 : (t < 416 ? 1 : (t < XR + 32 ? 2 : 3)), depth = side_b ? BStages : ATStages;
        if (!(DIRECT && side_b))
            for (int k = 0; k < TilesPerPair; ++k) {
                const unsigned tile = k * Pairs + q;
                if (tile >= Tiles)
                    break;
                for (int r = 0; r < reqs; ++r) {
                    const int G = k * reqs + r;
                    if (Relays > 1 && (G % Relays) != ri)
                        continue;
                    const int slot = G % depth;
                    const unsigned ph = (G / depth) & 1;
                    if (side_b) {
                        await(sa(&blocal[slot]), ph);
                        arrive_remote(peer0(sa(&bfull[slot])));
                    } else {
                        await(sa(&aready[slot]), ph);
                        arrive_remote(peer0(sa(&aready_r[slot])));
                    }
                }
            }
    } else if (!Relay && !DIRECT && rank == 1 && (t == 353 || (!A_FILL_ST && t == 352))) {
        const bool side_b = t == 353;
        const int depth = side_b ? BStages : AStages;
        for (int k = 0; k < TilesPerPair; ++k) {
            const unsigned tile = k * Pairs + q;
            if (tile >= Tiles)
                break;
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % depth;
                await(sa(side_b ? &blocal[slot] : &alocal[slot]), (G / depth) & 1);
                arrive_remote(peer0(sa(side_b ? &bfull[slot] : &afull[slot])));
            }
        }
    } else if (!A_FILL_ST && !A_SMEM && rank == 0 && t == 384) {
        for (int k = 0; k < TilesPerPair; ++k) {
            const unsigned tile = k * Pairs + q;
            if (tile >= Tiles)
                break;
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % AStages, tslot = G % ATStages;
                if (G >= ATStages)
                    await(sa(&tmem_empty[tslot]), ((G / ATStages) + 1) & 1);
                await(sa(&afull[slot]), (G / AStages) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                for (int c = 0; c < CutsPerReq; ++c) {
                    const unsigned at = tm + TmemA + tslot * TmemAStage + c * 12,
                                   src = sm_base + slot * ATile + c * ACut;
                    const auto head = tile_desc(src), tail = tile_desc(src + 256);
                    asm volatile(
                        "tcgen05.cp.cta_group::2.128x256b [%0],%1;tcgen05.cp.cta_group::2.128x128b [%2],%3;" ::"r"(at),
                        "l"(head), "r"(at + 8), "l"(tail)
                        : "memory");
                }
                done_leader(sa(&aready[tslot]));
                done_both(sa(&aempty[slot]));
            }
        }
    } else if (A_FILL_LDG && t >= 416 && t < XR) {
        const unsigned row = (t / 32 % 4) * 32 + t % 32;
        const unsigned lane_row = (t / 32 % 4 * 32) << 16;
        unsigned off[3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
            off[i] = tile_byte(row, 16 * i);
        const int nT = q < unsigned(Tiles) ? int((unsigned(Tiles) - 1 - q) / Pairs) + 1 : 0, total = nT * reqs;
        auto load = [&](uint4(&w)[CutsPerReq][3], int G) {
            const int k = G / reqs, r = G % reqs;
            const unsigned char *p = A + size_t(2 * tile_row(k * Pairs + q) + rank) * a_tile_bytes + size_t(r) * ATile;
#pragma unroll
            for (int c = 0; c < CutsPerReq; ++c) {
#pragma unroll
                for (int i = 0; i < 3; ++i)
                    w[c][i] = __ldg(reinterpret_cast<const uint4 *>(p + c * ACut + off[i]));
            }
        };
        auto put = [&](uint4(&w)[CutsPerReq][3], int G) {
            const int tslot = G % ATStages;
            if (G >= ATStages)
                await(sa(&tmem_empty[tslot]), ((G / ATStages) + 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
            for (int c = 0; c < CutsPerReq; ++c) {
                const unsigned at = tm + lane_row + TmemA + tslot * TmemAStage + c * 12;
                asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(at),
                             "r"(w[c][0].x), "r"(w[c][0].y), "r"(w[c][0].z), "r"(w[c][0].w), "r"(w[c][1].x),
                             "r"(w[c][1].y), "r"(w[c][1].z), "r"(w[c][1].w)
                             : "memory");
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};" ::"r"(at + 8), "r"(w[c][2].x),
                             "r"(w[c][2].y), "r"(w[c][2].z), "r"(w[c][2].w)
                             : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::: "memory");
            __syncwarp();
            if (t % 32 == 0)
                arrive_local(sa(&aready[tslot]));
        };
        uint4 wa[CutsPerReq][3], wb[CutsPerReq][3];
        int G = 0;
        if (G < total)
            load(wa, G);
        for (; G < total; G += 2) {
            if (G + 1 < total)
                load(wb, G + 1);
            put(wa, G);
            if (G + 1 >= total)
                break;
            if (G + 2 < total)
                load(wa, G + 2);
            put(wb, G + 1);
        }
    } else if (A_FILL_ST && t >= 416 && t < XR) {
        const int grp = int(t - 416) / 128;
        const unsigned row = (t / 32 % 4) * 32 + t % 32;
        const unsigned lane_row = (t / 32 % 4 * 32) << 16;
        const unsigned ar_remote = peer0(sa(&aready[0]));
        unsigned off[3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
            off[i] = tile_byte(row, 16 * i);
        for (int k = 0; k < TilesPerPair; ++k) {
            const unsigned tile = k * Pairs + q;
            if (tile >= Tiles)
                break;
            unsigned long long wt = 0, wa = 0, wst = 0, warr = 0;
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % AStages, tslot = G % ATStages;
                const auto x0 = clock64();
                if (G >= ATStages)
                    await(sa(&tmem_empty[tslot]), ((G / ATStages) + 1) & 1);
                const auto x1 = clock64();
                await(sa(&afull[slot]), (G / AStages) & 1);
                const auto x2 = clock64();
                wa += x2 - x1;
                wt += x1 - x0;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                uint4 w[CutsPerReq / WSplit][3];
#if A_LIMBS
#pragma unroll
                for (int j = 0; j < CutsPerReq / WSplit; ++j) {
                    const int c = grp + j * WSplit;
                    const unsigned char *b = sm + slot * ATile + c * ACut + row * 8;
                    const uint2 l0 = *reinterpret_cast<const uint2 *>(b),
                                l1 = *reinterpret_cast<const uint2 *>(b + CtaRows * 8),
                                l2 = *reinterpret_cast<const uint2 *>(b + 2 * CtaRows * 8);
                    const auto e = integer_planes::form(l2.x, l1.x, l0.x),
                               o = integer_planes::form(l2.y, l1.y, l0.y); // d01=a2-a1, d12=a1-a0, d02=a2-a0
                    w[j][0] = make_uint4(l0.x, l0.y, l1.x, l1.y);
                    w[j][1] = make_uint4(l2.x, l2.y, e.d12, o.d12);
                    w[j][2] = make_uint4(e.d01, o.d01, e.d02, o.d02);
                }
#else
#pragma unroll
                for (int j = 0; j < CutsPerReq / WSplit; ++j) {
                    const int c = grp + j * WSplit;
#pragma unroll
                    for (int i = 0; i < 3; ++i)
                        w[j][i] = *reinterpret_cast<const uint4 *>(sm + slot * ATile + c * ACut + off[i]);
                }
#endif
#if !A_RELEASE_LATE
                __syncwarp();
                if (t % 32 == 0)
                    arrive_local(sa(&aempty[slot]));
#endif
#pragma unroll
                for (int j = 0; j < CutsPerReq / WSplit; ++j) {
                    const int c = grp + j * WSplit;
                    const unsigned at = tm + lane_row + TmemA + tslot * TmemAStage + c * 12;
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(at),
                                 "r"(w[j][0].x), "r"(w[j][0].y), "r"(w[j][0].z), "r"(w[j][0].w), "r"(w[j][1].x),
                                 "r"(w[j][1].y), "r"(w[j][1].z), "r"(w[j][1].w)
                                 : "memory");
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};" ::"r"(at + 8),
                                 "r"(w[j][2].x), "r"(w[j][2].y), "r"(w[j][2].z), "r"(w[j][2].w)
                                 : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::: "memory");
                const auto x3 = clock64();
#if A_RELEASE_LATE
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (t % 32 == 0) {
                    arrive_local(sa(&aempty[slot]));
                    if (Relay)
                        arrive_local(sa(&aready[tslot]));
                    else
                        arrive_remote(ar_remote + 8 * tslot);
                }
#else
                __syncwarp();
                if (t % 32 == 0)
                    arrive_remote(ar_remote + 8 * tslot);
#endif
                const auto x4 = clock64();
                wst += x3 - x2;
                warr += x4 - x3;
            }
            if (t == 416) {
                if (rank == 0) {
                    mt[tile].wt = wt;
                    mt[tile].wa = wa;
                    mt[tile].wst = wst;
                    mt[tile].warr = warr;
                } else {
                    mt[tile].wt1 = wt;
                    mt[tile].wa1 = wa;
                    mt[tile].wst1 = wst;
                    mt[tile].warr1 = warr;
                }
            }
        }
    } else if (rank == 0 && (ISSUE_WARP ? (t >= 256 && t < 288) : t == 256)) {
        const unsigned sfa = tm + TmemSfa, sfb = tm + TmemSfb;
        const unsigned long long bdesc0 = tile_desc(sm_base + BOffset);
        const unsigned aBeg = tm + TmemA, rr0 = sa(&aready_r[0]), ar0 = A_SMEM ? sa(&afull[0]) : sa(&aready[0]),
                       te0 = A_SMEM ? sa(&aempty[0]) : sa(&tmem_empty[0]), bf0 = sa(&bfull[0]), be0 = sa(&bempty[0]),
                       ar_end = ar0 + 8 * ATStages, bf_end = bf0 + 8 * BStages;
        const unsigned long long adesc0 = tile_desc(sm_base);
        unsigned long long ad = adesc0;
        unsigned rr_bar = rr0, aa = aBeg, ar_bar = ar0, te_bar = te0, bf_bar = bf0, be_bar = be0, aph = 0, bph = 0;
        unsigned long long dd = bdesc0;
        const int segc = (KSEG && cuts > 64) ? 64 : cuts, nseg = cuts / segc, rps = segc / CutsPerReq;
        int U = -1;
        for (int k = 0;; ++k) {
            unsigned tile;
            int part;
            if (!item_of(q, k, tile, part))
                break;
            unsigned long long ns0, fi = 0;
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns0));
            unsigned long long aw = 0, bw = 0, dw = 0, cb = 0;
            const int nsgi = part < 0 ? nseg : nseg / TailS;
            for (int sg = 0; sg < (KSEG ? nsgi : 1); ++sg) {
                const int u = KSEG ? ++U : k;
                const int ds = u % DSlots, use = u / DSlots;
                const unsigned d = tm + TmemD + ds * TileN;
                {
                    const auto c0 = clock64();
                    if (sg == 0)
                        cb = c0;
                    if (u >= DSlots)
                        await(sa(&read_done[ds]), (use - 1) & 1);
                    dw += clock64() - c0;
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if KFIX
#pragma unroll
                for (int r = 0; r < KFIX / 16 / CutsPerReq; ++r) {
#else
                for (int r = 0; r < (KSEG ? rps : reqs);
                     ++r) { // add-and-wrap ring state (669: no division or modulo in the issue loop)
#endif
                    const auto w0 = clock64();
                    await(ar_bar, aph);
                    if (Relay)
                        await(rr_bar, aph);
                    const auto w1 = clock64();
                    if (!COMBINED_AB)
                        await(bf_bar, bph);
                    bw += clock64() - w1;
                    aw += w1 - w0;
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if A_SMEM
                    if (r == 0) {
                        if (sg == 0)
                            fi = clock64();
                        mma_s(d, ad, dd, sfa, sfb, false);
                    } else
                        mma_s(d, ad, dd, sfa, sfb, true);
#pragma unroll
                    for (int c = 1; c < CutsPerReq; ++c)
                        mma_s(d, ad + ((unsigned long long)(c * ACut) >> 4), dd + ((unsigned long long)(c * BCut) >> 4),
                              sfa, sfb, true);
#else
                    if (r == 0) {
                        fi = clock64();
                        mma1(d, aa, dd, sfa, sfb, false);
                    } else
                        mma1(d, aa, dd, sfa, sfb, true);
#pragma unroll
                    for (int c = 1; c < CutsPerReq; ++c)
                        mma1(d, aa + c * 12, dd + ((unsigned long long)(c * BCut) >> 4), sfa, sfb, true);
#endif
                    if (A_FILL_ST || A_SMEM)
                        done_both(te_bar);
                    else
                        done_leader(te_bar);
                    if (!COMBINED_AB)
                        done_both(be_bar);
                    aa += TmemAStage;
                    ad += (unsigned long long)(ATile >> 4);
                    ar_bar += 8;
                    rr_bar += 8;
                    te_bar += 8;
                    if (ar_bar == ar_end) {
                        aa = aBeg;
                        ad = adesc0;
                        ar_bar = ar0;
                        rr_bar = rr0;
                        te_bar = te0;
                        aph ^= 1u;
                    }
                    dd += (unsigned long long)(BTile >> 4);
                    bf_bar += 8;
                    be_bar += 8;
                    if (bf_bar == bf_end) {
                        dd = bdesc0;
                        bf_bar = bf0;
                        be_bar = be0;
                        bph ^= 1u;
                    }
                }
                done_both(sa(&drain[ds]));
            } // segments
            const auto le = clock64();
            if (t == 256) {
                auto &z = mt[tile];
                z.cb = cb;
                z.fi = fi;
                z.le = le;
                z.dwait = dw;
                z.ns0 = ns0;
                z.cta_begin_ns = gbegin;
                z.begin0 = begin0;
                z.aw = aw;
                z.bw = bw;
            }
        }
    } else if (KSEG && t < 256) {
        static_assert(!KSEG || (DSlots == 2 && (TileN / 2) % 16 == 0),
                      "KSEG needs D and D' and 16-column read-out groups");
        const int segc = cuts > 64 ? 64 : cuts, nseg = cuts / segc;
        const unsigned col0 = (t / 128) * (TileN / 2);
        int U = -1;
        for (int k = 0;; ++k) {
            unsigned tile;
            int part;
            if (!item_of(q, k, tile, part))
                break;
            const unsigned tr = tile_row(tile), tc = tile_col(tile);
            const int nsgi = part < 0 ? nseg : nseg / TailS;
            int acc[TileN / 2];
            unsigned long long seen = 0, ns1 = 0;
            for (int sg = 0; sg < nsgi; ++sg) {
                const int u = ++U, ds = u % DSlots, use = u / DSlots;
                await(sa(&drain[ds]), use & 1);
                if (sg == nsgi - 1) {
                    seen = clock64();
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1));
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                for (int i = 0; i < (TileN / 2) / 16; ++i) {
                    unsigned v[16];
                    load16(local + TmemD + ds * TileN + col0 + 16 * i, v);
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int x = FASTCVT ? f2i_int24(v[j]) : int(__uint_as_float(v[j]));
                        acc[16 * i + j] = sg == 0 ? x : acc[16 * i + j] + x;
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
                if (t == 0)
                    arrive_remote(peer0(sa(&read_done[ds])));
            }
            int *cp =
                reinterpret_cast<int *>(C) + size_t(tc * TileN + col0) * M + tr * TileM + rank * CtaRows + (t % 128);
            const unsigned gn0 = tc * TileN + col0;
#pragma unroll
#if KSEG_STOREV4
            {
                int4 *vp = reinterpret_cast<int4 *>(
                    reinterpret_cast<int *>(C) +
                    (size_t(tile) * TileM * TileN) % (size_t(M) * N - size_t(TileM) * TileN) +
                    size_t(rank * CtaRows + (t % 128)) * TileN +
                    col0); // wrap keeps the writes inside C (padded tiles); both terms are multiples of 4 ints
#pragma unroll
                for (int j = 0; j < TileN / 8; ++j)
                    vp[j] = make_int4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
            } // outputs land in the wrong places by design
#elif KSEG_NOSTORE
            {
                int x = 0;
#pragma unroll
                for (int j = 0; j < TileN / 2; ++j)
                    x ^= acc[j];
                if (x == 0x12345678)
                    cp[0] = x;
            }
#else
        for (int j = 0; j < TileN / 2; ++j) {
            if (N % TileN == 0 || gn0 + j < unsigned(N)) {
                if (TailS > 1 && part >= 0)
                    atomicAdd(cp + size_t(j) * M, acc[j]);
                else
                    cp[size_t(j) * M] = acc[j];
            }
        }
#endif
            asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0 && rank == 0) {
                const auto pe = clock64();
                unsigned long long g;
                asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                mt[tile].seen = seen;
                mt[tile].pend = pe;
                mt[tile].ns1 = ns1;
                mt[tile].pay_end_ns = g;
                mt[tile].smid = smid_();
            }
        }
    } else if (PAYOUT16 && (t < 256 || (t >= unsigned(XP) && t < unsigned(XP + 256)))) {
        static_assert(!PAYOUT16 || TileN == 256, "PAYOUT16 is written for N=256 (two 64-column groups per half)");
        const unsigned w = t / 32, row = (w % 4) * 32 + t % 32;
        const bool extra = t >= 256;
        const unsigned e = extra ? (t - XP) / 128 : t / 128; // half 0/1
        const unsigned colbase = e * 128 + (extra ? 64u : 0u);
        for (int k = 0; k < TilesPerPair; ++k) {
            const unsigned tile = k * Pairs + q;
            if (tile >= Tiles)
                break;
            const unsigned tr = tile_row(tile), tc = tile_col(tile);
            const int ds = k % DSlots, use = k / DSlots;
            await(sa(&drain[ds]), use & 1);
            const auto seen = clock64();
            unsigned long long ns1;
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1));
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            unsigned v[4][16];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ld16nw(local + TmemD + ds * TileN + colbase + 16 * i, v[i]);
            waitld16(v[0]);
            tie16(v[1]);
            tie16(v[2]);
            tie16(v[3]);
            asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,512;" ::: "memory");
            if (t == 0)
                arrive_remote(peer0(sa(&read_done[ds])));
            {
                float *cp = C + size_t(tc * TileN + colbase) * M + tr * TileM + rank * CtaRows + row;
                const unsigned gn0 = tc * TileN + colbase; // one base pointer
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        if (N % TileN == 0 || gn0 + 16 * i + j < unsigned(N))
                            cp[size_t(16 * i + j) * M] = __uint_as_float(v[i][j]);
                    }
                }
            }
            asm volatile("barrier.cta.sync 1,512;" ::: "memory");
            if (t == 0 && rank == 0) {
                const auto pe = clock64();
                unsigned long long g;
                asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                mt[tile].seen = seen;
                mt[tile].pend = pe;
                mt[tile].ns1 = ns1;
                mt[tile].pay_end_ns = g;
                mt[tile].smid = smid_();
            }
        }
    } else if (t < 256) {
        for (int k = 0; k < TilesPerPair; ++k) {
            const unsigned tile = k * Pairs + q;
            if (tile >= Tiles)
                break;
            const unsigned tr = tile_row(tile), tc = tile_col(tile);
            const int ds = k % DSlots, use = k / DSlots;
            await(sa(&drain[ds]), use & 1);
            const auto seen = clock64();
            unsigned long long ns1;
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1));
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if PAYOUT_WIDE == 3
            {
                constexpr int NL = (TileN / 2) / 16;
                unsigned v[NL][16];
                const unsigned col0 = (t / 128) * (TileN / 2);
#pragma unroll
                for (int i = 0; i < NL; ++i)
                    ld16nw(local + TmemD + ds * TileN + col0 + 16 * i, v[i]);
                waitld16(v[0]);
#pragma unroll
                for (int i = 1; i < NL; ++i)
                    tie16(v[i]);
                asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
                if (t == 0)
                    arrive_remote(peer0(sa(&read_done[ds])));
                float *cp = C + size_t(tc * TileN + col0) * M + tr * TileM + rank * CtaRows + (t % 128);
                const unsigned gn0 = tc * TileN + col0;
#pragma unroll
                for (int i = 0; i < NL; ++i) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        if (N % TileN == 0 || gn0 + 16 * i + j < unsigned(N))
                            cp[size_t(16 * i + j) * M] = __uint_as_float(v[i][j]);
                    }
                }
            }
            asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0) {
#elif PAYOUT_WIDE
            constexpr int L = PAYOUT_LDS, GW = 16 * L;
            static_assert((TileN / 2) % GW == 0 && (L == 2 || L == 4), "read-out groups");
            constexpr int Groups = (TileN / 2) / GW;
#pragma unroll 1
            for (int g = 0; g < Groups; ++g) {
                unsigned v[L][16];
                const unsigned col0 = (t / 128) * (TileN / 2) + g * GW;
#pragma unroll
                for (int i = 0; i < L; ++i)
                    ld16nw(local + TmemD + ds * TileN + col0 + 16 * i, v[i]);
                waitld16(v[0]);
#pragma unroll
                for (int i = 1; i < L; ++i)
                    tie16(v[i]);
                if (PAYOUT_WIDE == 2 && g == Groups - 1) {
                    asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
                    if (t == 0)
                        arrive_remote(peer0(sa(&read_done[ds])));
                }
#pragma unroll
                for (int i = 0; i < L; ++i) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const unsigned gn = tc * TileN + col0 + 16 * i + j;
                        if (gn < unsigned(N))
                            C[size_t(gn) * M + tr * TileM + rank * CtaRows + (t % 128)] = __uint_as_float(v[i][j]);
                    }
                }
            }
            if (PAYOUT_WIDE != 2)
                asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
            else
                asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0) {
                if (PAYOUT_WIDE != 2)
                    arrive_remote(peer0(sa(&read_done[ds])));
#else
#pragma unroll 1
        for (int chunk = 0; chunk < TileN / 32; ++chunk) {
            unsigned v[16];
            const unsigned col = (t / 128) * (TileN / 2) + chunk * 16;
            load16(local + TmemD + ds * TileN + col, v);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned gn = tc * TileN + col + j;
                if (gn < unsigned(N))
                    C[size_t(gn) * M + tr * TileM + rank * CtaRows + (t % 128)] = __uint_as_float(v[j]);
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
        if (t == 0) {
            arrive_remote(peer0(sa(&read_done[ds])));
#endif
                if (rank == 0) {
                    const auto pe = clock64();
                    unsigned long long g;
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                    mt[tile].seen = seen;
                    mt[tile].pend = pe;
                    mt[tile].ns1 = ns1;
                    mt[tile].pay_end_ns = g;
                    mt[tile].smid = smid_();
                }
            }
        }
    }
    __syncthreads();
    cluster.sync();
    if (t < 32)
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,512;tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                "r"(tm)
            : "memory");
    cluster.sync();
}

// (1,5,26) encoding: row r of a Rows-row tile, cut c (16 original k) -> 96 FP4 codes, block f = face f of those 16 k.
// Byte w of the 48 per row holds k' = 2w (low nibble) and 2w+1 (high), in the K-major 8-row core-matrix layout (534).
#undef M
#undef N
#undef RowTiles
#undef ColTiles
#undef Tiles
#undef TilesPerPair
#undef TilesPerBand
#undef TailS
#undef FullWaves
#undef FullTiles
#undef TailTiles
} // namespace d1
