#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include "duald40_load.cuh" // load16 (the original includes it before line 88)
#undef A_SMEM
#undef A_STAGES
#undef A_TMEM_STAGES
#undef B_STAGES
#undef CHECK_HASH
#undef COLLECTOR
#undef CUTS_PER_REQ
#undef DSLOTS
#undef EARLY_RELEASE
#undef FULL_N
#undef MBLK
#undef MN
#undef PAIR_DIRECT
#undef REF_ONLY
#undef REG_HIGH
#undef REG_LOW
#undef RELAXED_RD
#undef ROW_TILES_PER_BAND
#undef SPLIT33
#undef SPLIT_N
#undef STAMPS
#undef SUPPLY_SPLIT
#undef TILE_N
#undef SCHEME
#define CHECK_HASH 0
#define REF_ONLY 0
#define PAIR_DIRECT 1
#define TILE_N 224
#define EARLY_RELEASE 1
#define MBLK 1
#define A_SMEM 1
#define A_STAGES 5
#define B_STAGES 4
#define ROW_TILES_PER_BAND 8
#define FULL_N 1
#define CUTS_PER_REQ 4
#define DSLOTS 1
#define SPLIT33 1
#define SCHEME 1
#define MN 8192
namespace f5 {
struct DimsT {
    int M, N, RowTiles, ColTiles, Tiles, TilesPerCluster, TilesPerBand;
};
__constant__ DimsT gd;
#define M (gd.M)
#define N (gd.N)
#define RowTiles (gd.RowTiles)
#define ColTiles (gd.ColTiles)
#define Tiles (gd.Tiles)
#define TilesPerCluster (gd.TilesPerCluster)
#define TilesPerBand (gd.TilesPerBand)
__device__ __forceinline__ void load16_nowait(unsigned addr, unsigned (&v)[16]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},[%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
                 : "r"(addr)
                 : "memory");
}
#ifndef A_STAGES
#define A_STAGES 2
#endif
#ifndef B_STAGES
#define B_STAGES 3
#endif
#ifndef A_TMEM_STAGES
#define A_TMEM_STAGES 4
#endif
#ifndef PAIR_DIRECT
#define PAIR_DIRECT 1
#endif
#ifndef MN
#define MN 8192
#endif
#ifndef SUPPLY_SPLIT
#define SUPPLY_SPLIT 0
#endif
#if SUPPLY_SPLIT && (MBLK != 2 || !PAIR_DIRECT || !A_SMEM || !EARLY_RELEASE)
#error "SUPPLY_SPLIT is for MBLK=2 with PAIR_DIRECT, A_SMEM and an early release (thread 352 free)"
#endif
#ifndef CUTS_PER_REQ
#define CUTS_PER_REQ 4
#endif
constexpr int CutsPerReq = CUTS_PER_REQ;
static_assert(CUTS_PER_REQ == 2 || CUTS_PER_REQ == 4, "cuts per request: 2 or 4");
#ifndef EARLY_RELEASE
#define EARLY_RELEASE 0
#endif
#if EARLY_RELEASE && !PAIR_DIRECT
#error "EARLY_RELEASE uses thread 352 for the copy, free only in the direct form"
#endif
#ifndef TILE_N
#define TILE_N 256
#endif
#ifndef MBLK
#define MBLK 1
#endif
#ifndef FULL_N
#define FULL_N 0
#endif
constexpr int TileM = 256 * MBLK, TileN = TILE_N, Half = 128,
              BHalf = TILE_N / 2; // M, N: runtime (f5::gd)   // each CTA: 128 rows of A, TILE_N/2 rows of B
static_assert(TILE_N % 32 == 0 && TILE_N >= 128 && TILE_N <= 256,
              "pair N: multiple of 32 (payout chunks of 16 per half), 128..256");
constexpr int Grid = 148, Clusters = Grid / 2; // RowTiles, ColTiles, Tiles, TilesPerCluster: runtime (f5::gd)
#ifndef RELAXED_RD
#define RELAXED_RD 0
#endif
#ifndef CHECK_HASH
#define CHECK_HASH 0
#endif
#ifndef REF_ONLY
#define REF_ONLY 0
#endif
#ifndef STAMPS
#define STAMPS 0
#endif
#if STAMPS && (EARLY_RELEASE != 1 || MBLK != 1)
#error "STAMPS is for the one-block early-release form"
#endif
#ifndef DSLOTS
#define DSLOTS 1
#endif
#ifndef SPLIT33
#define SPLIT33 0
#endif
#if DSLOTS != 1 && (DSLOTS != 2 || MBLK != 1 || EARLY_RELEASE != 1 || !A_SMEM || !PAIR_DIRECT || 2 * TILE_N + 32 > 512)
#error "DSLOTS=2 is MBLK=1, EARLY_RELEASE=1, A_SMEM, PAIR_DIRECT with 2 x TILE_N + 32 <= 512 TMEM columns"
#endif
#ifndef ROW_TILES_PER_BAND
#define ROW_TILES_PER_BAND 16
#endif
constexpr int RowTilesPerBand = ROW_TILES_PER_BAND; // TilesPerBand: runtime (f5::gd)
constexpr int ACut = Half * 48, BCut = BHalf * 48, ATile = ACut * CutsPerReq, BTile = BCut * CutsPerReq;
constexpr int AStageBytes = MBLK * ATile, AStages = A_STAGES, BStages = B_STAGES, ATStages = A_TMEM_STAGES,
              BOffset = AStages * AStageBytes, SharedBytes = BOffset + BStages * BTile;
static_assert(SharedBytes <= 229376, "dynamic Shared Memory accepted on this part (440)");
constexpr int NAcc = SPLIT33 ? 2 : DSLOTS * MBLK, TmemSfa = NAcc * TileN, TmemSfb = NAcc * TileN + 16,
              TmemA = NAcc * TileN + 32, TmemAStage = 12 * CutsPerReq;
constexpr int TmemSfa2 = NAcc * TileN + 32, TmemSfb2 = NAcc * TileN + 48;
constexpr int Fold0 = SCHEME == 1 ? 37 : 32, Fold1 = SCHEME == 1 ? 7 : 1,
              FoldShift =
                  SCHEME == 1 ? 1 : 0; // C = (Fold0 D0 + Fold1 D1) >> FoldShift (SCHEME 1: 37 D0 + 7 D1 is even)   //
                                       // SPLIT33: the difference group's scale set (A in Shared Memory: no TMEM A)
static_assert(!SPLIT33 || TmemSfb2 + 16 <= 512, "SPLIT33 TMEM columns");
static_assert((MBLK == 1 && !A_SMEM) ? (TmemA + ATStages * TmemAStage <= 512) : (TmemA <= 512),
              "TMEM columns"); // A in Shared Memory: no A stages in TMEM
__host__ __device__ constexpr int tile_byte(int row, int w) {
    return (row / 8) * 384 + (w / 16) * 128 + (row % 8) * 16 + w % 16;
}
__device__ __forceinline__ unsigned tile_row(unsigned tile) {
    const unsigned band = tile / TilesPerBand, r0 = band * RowTilesPerBand,
                   h = min(unsigned(RowTilesPerBand), unsigned(RowTiles) - r0);
    return r0 + (tile - band * TilesPerBand) % h;
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
// cluster-scope acquire: some arrivals come from the other CTA (relay, multicast commit)
__device__ __forceinline__ void await(unsigned b, unsigned parity) {
    asm volatile(
        "{.reg .pred p;Lw: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra Lw;}" ::"r"(b),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void arrive_local(unsigned b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];" ::"r"(b) : "memory");
}
// read_done only orders the payout's finished tcgen05.ld (tcgen05.wait::ld blocks) before the next MMAs; no memory
// needs releasing
__device__ __forceinline__ void arrive_local_rd(unsigned b) {
    if (RELAXED_RD)
        asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _,[%0];" ::"r"(b) : "memory");
    else
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void arrive_leader_rd(unsigned b) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0,%1,0;" : "=r"(r) : "r"(b));
    if (RELAXED_RD)
        asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::"r"(r) : "memory");
    else
        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(r) : "memory");
}
__device__ __forceinline__ void arrive_leader(unsigned b) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0,%1,0;" : "=r"(r) : "r"(b));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(r) : "memory");
}
__device__ __forceinline__ void commit_both(unsigned b) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;" ::"r"(b),
        "h"((unsigned short)3)
        : "memory");
}
__device__ __forceinline__ void commit_leader(unsigned b) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;" ::"r"(b),
        "h"((unsigned short)1)
        : "memory");
}
__device__ __forceinline__ void tst(unsigned a, unsigned v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long tile_desc(unsigned address) {
    return ((address >> 4) & 0x3fffull) | (8ull << 16) | (24ull << 32) | (1ull << 46);
}
constexpr unsigned IDESC = 0x480u | ((unsigned(TileN) >> 3) << 17) | (2u << 27) |
                           0x80000000u; // M=256, N=256, E2M1, UE4M3, K=96
static_assert(TILE_N != 256 || IDESC == 0x90400480u, "instruction descriptor at N=256");
__device__ __forceinline__ void mma2(unsigned d, unsigned a_tmem, unsigned long long b_desc, unsigned sfa, unsigned sfb,
                                     bool acc) {
    asm volatile(
        "{.reg .pred p;setp.ne.u32 p,%5,0;tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0],[%1],%2,%3,[%4],[%6],p;}" ::
            "r"(d),
        "r"(a_tmem), "l"(b_desc), "r"(IDESC), "r"(sfa), "r"(int(acc)), "r"(sfb)
        : "memory");
}
#ifndef A_SMEM
#define A_SMEM 0
#endif
#ifndef SPLIT_N
#define SPLIT_N 1
#endif
#ifndef COLLECTOR
#define COLLECTOR 0
#endif
#ifndef REG_HIGH
#define REG_HIGH 232
#endif
#ifndef REG_LOW
#define REG_LOW 40
#endif
// The pool setmaxnreg moves registers within is what the CTA got at launch: 384 threads x 168
// (__launch_bounds__(384,1)) = 64,512, not 65,536.  Asking for more makes the second setmaxnreg.inc wait forever
// (bundle 20260927122500: 232 / 48 hung every e3 case).
#if EARLY_RELEASE == 3 && (MBLK != 2 || 2 * REG_HIGH + REG_LOW > 3 * 168)
#error "EARLY_RELEASE=3 is MBLK=2 with 2 x REG_HIGH + REG_LOW <= 3 x 168"
#endif
#if MBLK != 1 && (MBLK != 2 || !A_SMEM || EARLY_RELEASE == 1 || !PAIR_DIRECT)
#error "MBLK=2 needs A_SMEM, PAIR_DIRECT and EARLY_RELEASE 0 or 2"
#endif
#if (EARLY_RELEASE == 2 || EARLY_RELEASE == 3) && MBLK != 2
#error "EARLY_RELEASE=2 is the per-accumulator release of MBLK=2"
#endif
#if (SPLIT_N != 1 && (SPLIT_N != 2 || !A_SMEM || TILE_N != 256)) || (COLLECTOR && SPLIT_N != 2)
#error "SPLIT_N=2 needs A_SMEM and TILE_N=256; COLLECTOR needs SPLIT_N=2"
#endif
#if SPLIT33 && (MBLK != 1 || DSLOTS != 1 || EARLY_RELEASE != 1 || !A_SMEM || !PAIR_DIRECT || SPLIT_N != 1 ||           \
                2 * TILE_N + 64 > 512 || CUTS_PER_REQ % 2 || STAMPS || REF_ONLY || CHECK_HASH)
#error                                                                                                                 \
    "SPLIT33 is MBLK=1, DSLOTS=1, EARLY_RELEASE=1, A_SMEM, PAIR_DIRECT, SPLIT_N=1, 2 TILE_N + 64 <= 512, even CUTS_PER_REQ"
#endif
#if SCHEME == 1 && !SPLIT33
#error "SCHEME=1 ((1,7,28)) needs SPLIT33 (two accumulators)"
#endif
constexpr unsigned IDESC_H = 0x480u | ((unsigned(TILE_N / 2) >> 3) << 17) | (2u << 27) |
                             0x80000000u; // M=256, N=128 (one of the two MMAs of SPLIT_N=2)
// op 0 = no collector qualifier, 1 = fill, 2 = lastuse
template <int OP>
__device__ __forceinline__ void mma2h(unsigned d, unsigned long long a_desc, unsigned long long b_desc, unsigned sfa,
                                      unsigned sfb, bool acc) {
    if constexpr (OP == 1)
        asm volatile(
            "{.reg .pred p;setp.ne.u32 p,%5,0;tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [%0],%1,%2,%3,[%4],[%6],p;}" ::
                "r"(d),
            "l"(a_desc), "l"(b_desc), "r"(IDESC_H), "r"(sfa), "r"(int(acc)), "r"(sfb)
            : "memory");
    else if constexpr (OP == 2)
        asm volatile(
            "{.reg .pred p;setp.ne.u32 p,%5,0;tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0],%1,%2,%3,[%4],[%6],p;}" ::
                "r"(d),
            "l"(a_desc), "l"(b_desc), "r"(IDESC_H), "r"(sfa), "r"(int(acc)), "r"(sfb)
            : "memory");
    else
        asm volatile(
            "{.reg .pred p;setp.ne.u32 p,%5,0;tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0],%1,%2,%3,[%4],[%6],p;}" ::
                "r"(d),
            "l"(a_desc), "l"(b_desc), "r"(IDESC_H), "r"(sfa), "r"(int(acc)), "r"(sfb)
            : "memory");
}
// SPLIT_N=2: D column j -> column of the pair tile (MMA h = j/128, rank r = (j%128)/64, row i = j%64 of that rank's B
// half)
__device__ __forceinline__ unsigned dcol(unsigned j) {
    return SPLIT_N == 2 ? ((j % 128) / 64) * 128 + (j / 128) * 64 + j % 64 : j;
}
__device__ __forceinline__ void mma2s(unsigned d, unsigned long long a_desc, unsigned long long b_desc, unsigned sfa,
                                      unsigned sfb, bool acc) {
    asm volatile(
        "{.reg .pred p;setp.ne.u32 p,%5,0;tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0],%1,%2,%3,[%4],[%6],p;}" ::
            "r"(d),
        "l"(a_desc), "l"(b_desc), "r"(IDESC), "r"(sfa), "r"(int(acc)), "r"(sfb)
        : "memory");
}
__device__ __forceinline__ unsigned smid_() {
    unsigned x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
    return x;
}
__device__ __forceinline__ unsigned ctarank_() {
    unsigned x;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(x));
    return x;
}
struct Metrics {
    unsigned long long core, ns, a_wait, b_wait, commit_wait, issue, payout, total, ns0, ns1, first_issue, last_issue,
        cta_begin_ns, pay_end_ns, store, c0, c1, c2;
    unsigned smid;
};
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long g;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
    return g;
}
struct Scales {
    unsigned a0, a1, b0, b1;
};

#if !REF_ONLY
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(EARLY_RELEASE ? 384 : 416, 1)
    p2_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
              const unsigned char *__restrict__ A, const unsigned char *__restrict__ B, float *C, Metrics *mt, int cuts,
              Scales sc, Scales sc2, const unsigned *gate) {
    const unsigned gw = *gate;
    if (!(gw & 4u))
        return;
    const int cvt = int(gw & 3u);
    const unsigned q = blockIdx.x / 2, t = threadIdx.x, rank = ctarank_();
    const auto begin0 = clock64();
    unsigned long long gbegin;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(gbegin));
    __shared__ unsigned base;
    __shared__ __align__(8) unsigned long long afull[AStages], alocal[AStages], aempty[AStages], aready[ATStages],
        tmem_empty[ATStages], bfull[BStages], blocal[BStages], bempty[BStages], drain, drain1, read_done, read_done1;
    extern __shared__ __align__(1024) unsigned char sm[];
    if (t == 0) {
        for (int s = 0; s < AStages; ++s) {
            mbar_init(sa(&afull[s]), PAIR_DIRECT ? 1 : 2);
            mbar_init(sa(&alocal[s]), 1);
            mbar_init(sa(&aempty[s]), 1);
        }
        for (int s = 0; s < ATStages; ++s) {
            mbar_init(sa(&aready[s]), 1);
            mbar_init(sa(&tmem_empty[s]), 1);
        }
        for (int s = 0; s < BStages; ++s) {
            mbar_init(sa(&bfull[s]), PAIR_DIRECT ? 1 : 2);
            mbar_init(sa(&blocal[s]), 1);
            mbar_init(sa(&bempty[s]), 1);
        }
        mbar_init(sa(&drain), 1);
        mbar_init(sa(&drain1), 1);
        mbar_init(sa(&read_done), 2);
        mbar_init(sa(&read_done1), 2);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (t < 32)
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],512;tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                "r"(sa(&base))
            : "memory");
    __syncthreads();
    const unsigned tm = base, local = tm + ((t / 32 % 4 * 32) << 16);
    if (t < 128) {
        for (int c = 0; c < 8; ++c)
            tst(local + TmemSfa + c, c < 4 ? sc.a0 : sc.a1);
        for (int c = 0; c < 16; ++c)
            tst(local + TmemSfb + c, c < 8 ? sc.b0 : sc.b1);
        if (SPLIT_N == 2)
            for (int c = 0; c < 8; ++c)
                tst(local + TmemSfa + 32 + c, c < 4 ? sc.b0 : sc.b1);
        if (SPLIT33) {
            for (int c = 0; c < 8; ++c)
                tst(local + TmemSfa2 + c, c < 4 ? sc2.a0 : sc2.a1);
            for (int c = 0; c < 16; ++c)
                tst(local + TmemSfb2 + c, c < 8 ? sc2.b0 : sc2.b1);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const unsigned sm_base = sa(sm);
    const int reqs = (cuts + CutsPerReq - 1) / CutsPerReq, padded = reqs * CutsPerReq;
    const size_t a_half_bytes = size_t(padded) * ACut, b_half_bytes = size_t(padded) * BCut;
#if EARLY_RELEASE == 3
    if (t >= 256)
        asm volatile(
            "setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REG_LOW)); // the issue / supply warpgroup gives registers up
#endif
    if (t == 288 || t == 320 || (SUPPLY_SPLIT && t == 352)) {
        const bool side_b = t == 320;
        const int depth = side_b ? BStages : AStages;
        const unsigned bytes = side_b ? BTile : ATile;
        const int b_first = (SUPPLY_SPLIT && t == 352) ? 1 : 0,
                  b_count = side_b ? 1 : (SUPPLY_SPLIT ? 1 : MBLK); // SUPPLY_SPLIT: 288 = A block 0, 352 = A block 1
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = k * Clusters + q;
            if (tile >= Tiles)
                break;
            // this CTA's half: A rows tile_row*256 + rank*128 (packed 128-row tile 2*tile_row+rank), B columns likewise
            const unsigned char *src = side_b ? (B + size_t(2 * tile_col(tile) + rank) * b_half_bytes)
                                              : (A + size_t(2 * MBLK * tile_row(tile) + rank) * a_half_bytes);
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % depth;
                if (G >= depth)
                    await(sa(side_b ? &bempty[slot] : &aempty[slot]), ((G / depth) + 1) & 1);
                const unsigned dest = sm_base + (side_b ? BOffset + slot * BTile : slot * AStageBytes);
                const int nblk = side_b ? 1 : MBLK;
#if PAIR_DIRECT
                // both CTAs land in their own Shared Memory; the completion goes to the LEADER's full barrier, which
                // expects both halves (leader: arrive.expect_tx 2 x bytes; rank 1: no arrival, only complete_tx).  The
                // tensor is the packed buffer seen as 256-byte rows, one request = one box of bytes/256 rows.
                const unsigned lbar_local = sa(side_b ? &bfull[slot] : &afull[slot]);
                unsigned lbar;
                asm volatile("mapa.shared::cluster.u32 %0,%1,0;" : "=r"(lbar) : "r"(lbar_local));
                if (rank == 0 && b_first == 0)
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ::"r"(lbar_local),
                                 "r"(2u * bytes * unsigned(nblk))
                                 : "memory");
                for (int b = b_first; b < b_first + b_count; ++b) {
                    const size_t off = size_t(src - (side_b ? B : A)) +
                                       size_t(b) * 2 * (side_b ? b_half_bytes : a_half_bytes) + size_t(r) * bytes;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                            "r"(dest + unsigned(b) * ATile),
                        "l"(reinterpret_cast<unsigned long long>(side_b ? &tmB : &tmA)), "r"(0), "r"(int(off / 256)),
                        "r"(lbar)
                        : "memory");
                }
#else
                const unsigned bar =
                    sa(rank == 0 ? (side_b ? &bfull[slot] : &afull[slot]) : (side_b ? &blocal[slot] : &alocal[slot]));
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%2],[%3],%1,[%0];" ::
                        "r"(bar),
                    "r"(bytes), "r"(dest), "l"(src + size_t(r) * bytes)
                    : "memory");
#endif
            }
        }
    } else if (!PAIR_DIRECT && (t == 352 || t == 353) && rank == 1) {
        const bool side_b = t == 353;
        const int depth = side_b ? BStages : AStages;
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = k * Clusters + q;
            if (tile >= Tiles)
                break;
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % depth;
                await(sa(side_b ? &blocal[slot] : &alocal[slot]), (G / depth) & 1);
                arrive_leader(sa(side_b ? &bfull[slot] : &afull[slot]));
            }
        }
    } else if (!A_SMEM && t == (EARLY_RELEASE ? 352u : 384u) && rank == 0) {
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = k * Clusters + q;
            if (tile >= Tiles)
                break;
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % AStages, tslot = G % ATStages;
                if (G >= ATStages)
                    await(sa(&tmem_empty[tslot]), ((G / ATStages) + 1) & 1);
                await(sa(&afull[slot]), (G / AStages) & 1); // both CTAs' A halves landed
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
                commit_leader(sa(&aready[tslot]));
                commit_both(sa(&aempty[slot])); // the A slot is free in both CTAs once the copies landed
            }
        }
    } else if (t == 256 && rank == 0) {
        const unsigned long long bdesc0 = tile_desc(sm_base + BOffset), adesc0 = tile_desc(sm_base);
        (void)adesc0;
        const unsigned sfa = tm + TmemSfa, sfb = tm + TmemSfb, a0 = tm + TmemA, sfb_h = tm + TmemSfa + 32;
        (void)sfb_h;
        const unsigned sfa2 = tm + TmemSfa2, sfb2 = tm + TmemSfb2;
        (void)sfa2;
        (void)sfb2;
        unsigned aslot = 0, aph = 0, bslot = 0, bph = 0;
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = k * Clusters + q;
            if (tile >= Tiles)
                break;
            unsigned long long ns0, fi = 0, li = 0;
            const auto core_begin = clock64();
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns0));
            const unsigned slot = DSLOTS == 2 ? unsigned(k & 1) : 0u;
            if (DSLOTS == 2 && k >= 2) {
                await(sa(slot ? &read_done1 : &read_done), ((k >> 1) - 1) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            } // D' of tile k-2 read out in both CTAs
            for (int r = 0; r < reqs; ++r) {
#if A_SMEM
                await(sa(&afull[aslot]), aph);
                await(sa(&bfull[bslot]), bph); // both CTAs' A and B halves landed
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                if (r == 0)
                    fi = clock64() - core_begin;
                else if (r == reqs - 1)
                    li = clock64() - core_begin;
                const unsigned long long ad = adesc0 + (unsigned long long)((aslot * AStageBytes) >> 4),
                                         bd = bdesc0 + (unsigned long long)((bslot * BTile) >> 4);
#if SPLIT_N == 2
#pragma unroll
                for (int c = 0; c < CutsPerReq; ++c) {
                    const unsigned long long a_c = ad + (unsigned long long)((c * ACut) >> 4),
                                             b_c = bd + (unsigned long long)((c * BCut) >> 4);
                    const bool acc = (r != 0) || (c != 0);
                    mma2h<COLLECTOR ? 1 : 0>(tm, a_c, b_c, sfa, sfb_h, acc); // B rows 0-63 of each CTA -> D 0-127
                    mma2h<COLLECTOR ? 2 : 0>(tm + 128, a_c, b_c + (unsigned long long)((64 / 8 * 384) >> 4), sfa, sfb_h,
                                             acc);
                } // B rows 64-127 -> D 128-255, same A
#elif SPLIT33
#pragma unroll
                for (int c = 0; c < CutsPerReq; ++c) {
                    const unsigned db = unsigned(
                        c & 1); // even: diagonal faces -> D_diag (col 0); odd: differences -> D_diff (col TileN)
                    mma2s(tm + db * unsigned(TileN), ad + (unsigned long long)((c * ACut) >> 4),
                          bd + (unsigned long long)((c * BCut) >> 4), db ? sfa2 : sfa, db ? sfb2 : sfb,
                          (r != 0) || (c >= 2));
                } // first MMA of each D in the tile clears it
#else
#pragma unroll
                for (int c = 0; c < CutsPerReq; ++c) {
#pragma unroll
                    for (int b = 0; b < MBLK; ++b) {
                        if (EARLY_RELEASE == 2 && b == 1 && c == 0 && r == 0 && k > 0) {
                            await(sa(&read_done1), (k - 1) & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        }
                        mma2s(tm + unsigned((slot * MBLK + b) * TileN),
                              ad + (unsigned long long)((b * ATile + c * ACut) >> 4),
                              bd + (unsigned long long)((c * BCut) >> 4), sfa, sfb, (r != 0) || (c != 0));
                    }
                }
#endif
                commit_both(sa(&aempty[aslot]));
                commit_both(sa(&bempty[bslot])); // the MMAs have read A and B: both slots free in both CTAs
                if (++aslot == AStages) {
                    aslot = 0;
                    aph ^= 1;
                }
#else
                await(sa(&aready[aslot]), aph);
                await(sa(&bfull[bslot]), bph);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                if (r == 0)
                    fi = clock64() - core_begin;
                else if (r == reqs - 1)
                    li = clock64() - core_begin;
                const unsigned at = a0 + aslot * TmemAStage;
                const unsigned long long bd = bdesc0 + (unsigned long long)((bslot * BTile) >> 4);
                mma2(tm, at, bd, sfa, sfb, r != 0);
#pragma unroll
                for (int c = 1; c < CutsPerReq; ++c)
                    mma2(tm, at + c * 12, bd + (unsigned long long)((c * BCut) >> 4), sfa, sfb, true);
                commit_leader(sa(&tmem_empty[aslot]));
                commit_both(sa(&bempty[bslot]));
                if (++aslot == ATStages) {
                    aslot = 0;
                    aph ^= 1;
                }
#endif
                if (++bslot == BStages) {
                    bslot = 0;
                    bph ^= 1;
                }
            }
            if (reqs == 1)
                li = fi;
#if DSLOTS == 2
            commit_both(sa(slot ? &drain1 : &drain));
            const auto ca = clock64();
            const auto end = ca;
            unsigned long long ns1;
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1)); // no wait: the next tile goes to the other slot
#else
            commit_both(sa(&drain));
            const auto ca = clock64();
            await(sa(&drain), k & 1);
            const auto end = clock64();
            unsigned long long ns1;
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1));
#endif
            auto &z = mt[size_t(tile) * 2];
            z.core = end - core_begin;
            z.ns = ns1 - ns0;
            z.ns0 = ns0;
            z.ns1 = ns1;
            z.a_wait = 0;
            z.b_wait = 0;
            z.commit_wait = end - ca;
            z.issue = 0;
            z.first_issue = fi;
            z.last_issue = li;
            z.cta_begin_ns = gbegin;
            if (DSLOTS == 1) {
                await(sa(&read_done), k & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            } // both CTAs read their D out
            if (STAMPS)
                mt[size_t(tile) * 2].c1 = gtime();
        }
    } else if (t < 256) {
#if EARLY_RELEASE == 3
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(
            REG_HIGH)); // the two payout warpgroups take them (the payout code below is dominated by this)
#endif
        long long prev = begin0;
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = k * Clusters + q;
            if (tile >= Tiles)
                break;
            const unsigned tr = tile_row(tile), tc = tile_col(tile);
            float *const crow = C + size_t(tr) * TileM + rank * Half +
                                (t % 128); // hoisted: the indexed form spilled (128 regs) in the unrolled loop
            const unsigned slot = DSLOTS == 2 ? unsigned(k & 1) : 0u;
            if (DSLOTS == 2)
                await(sa(slot ? &drain1 : &drain), (k >> 1) & 1);
            else
                await(sa(&drain), k & 1);
            if (STAMPS && t == 0)
                mt[size_t(tile) * 2 + rank].c2 = gtime();
            const auto began = clock64();
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if EARLY_RELEASE == 3
            {
                unsigned v[2][TileN / 32][16];
#pragma unroll
                for (int b = 0; b < 2; ++b) {
#pragma unroll
                    for (int chunk = 0; chunk < TileN / 32; ++chunk)
                        load16_nowait(local + unsigned(b * TileN) + (t / 128) * (TileN / 2) + chunk * 16, v[b][chunk]);
                }
                asm volatile(
                    "tcgen05.wait::ld.sync.aligned;tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::
                        : "memory");
                const auto released = clock64();
                if (t == 0) {
                    if (rank == 0)
                        arrive_local_rd(sa(&read_done));
                    else
                        arrive_leader_rd(sa(&read_done));
                } // D0 and D1 both free
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    float *const crow_b = crow + b * 256;
#pragma unroll
                    for (int chunk = 0; chunk < TileN / 32; ++chunk) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const unsigned gn = tc * TileN + (t / 128) * (TileN / 2) + chunk * 16 + j;
                            if (!FULL_N || gn < unsigned(N))
                                crow_b[size_t(gn) * M] = __uint_as_float(v[b][chunk][j]);
                        }
                    }
                }
                asm volatile("barrier.cta.sync 1,256;" ::: "memory");
                if (t == 0) {
                    auto &z = mt[size_t(tile) * 2 + rank];
                    z.payout = released - began;
                    z.store = clock64() - released;
                    {
                        unsigned long long g;
                        asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                        z.pay_end_ns = g;
                    }
                    const auto now = clock64();
                    z.total = now - prev;
                    z.smid = smid_();
                    prev = now;
                }
            }
#elif EARLY_RELEASE == 2
            unsigned long long released = 0;
            for (int b = 0; b < 2; ++b) {
                unsigned v[TileN / 32][16];
#pragma unroll
                for (int chunk = 0; chunk < TileN / 32; ++chunk)
                    load16_nowait(local + unsigned(b * TileN) + (t / 128) * (TileN / 2) + chunk * 16, v[chunk]);
                asm volatile(
                    "tcgen05.wait::ld.sync.aligned;tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::
                        : "memory");
                if (b == 0)
                    released = clock64();
                if (t == 0) {
                    unsigned long long *rd = b == 0 ? &read_done : &read_done1;
                    if (rank == 0)
                        arrive_local_rd(sa(rd));
                    else
                        arrive_leader_rd(sa(rd));
                }
                float *const crow_b = crow + b * 256;
#pragma unroll
                for (int chunk = 0; chunk < TileN / 32; ++chunk) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const unsigned gn = tc * TileN + (t / 128) * (TileN / 2) + chunk * 16 + j;
                        if (!FULL_N || gn < unsigned(N))
                            crow_b[size_t(gn) * M] = __uint_as_float(v[chunk][j]);
                    }
                }
            }
            asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0) {
                auto &z = mt[size_t(tile) * 2 + rank];
                z.payout = released - began;
                z.store = clock64() - released;
                {
                    unsigned long long g;
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                    z.pay_end_ns = g;
                }
                const auto now = clock64();
                z.total = now - prev;
                z.smid = smid_();
                prev = now;
            }
#elif EARLY_RELEASE && SPLIT33
            unsigned v[TileN / 32][16]; // SPLIT33: INT32 D_diff + 32 D_diag in place (D_diag first in one go, then
                                        // D_diff 16 columns at a time)
#pragma unroll
            for (int chunk = 0; chunk < TileN / 32; ++chunk)
                load16_nowait(local + (t / 128) * (TileN / 2) + chunk * 16, v[chunk]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            // cvt bit 0 / 1 (uniform, from the host proof): |D_diag| / |D_diff| <= 2^22, so FADD + IADD gives the exact
            // integer (v + 1.5 * 2^23 has the integer in its low bits; checked for every integer |v| <= 2^22) instead
            // of F2I (16 / clock / SM)
            if (cvt & 1) {
#pragma unroll
                for (int chunk = 0; chunk < TileN / 32; ++chunk) {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                        v[chunk][j] =
                            unsigned(Fold0 * (__float_as_int(__uint_as_float(v[chunk][j]) + 12582912.0f) - 0x4B400000));
                }
            } else {
#pragma unroll
                for (int chunk = 0; chunk < TileN / 32; ++chunk) {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                        v[chunk][j] = unsigned(Fold0 * int(__uint_as_float(v[chunk][j])));
                }
            } // exact integers below 2^24 (host proof)
#pragma unroll
            for (int chunk = 0; chunk < TileN / 32; ++chunk) {
                unsigned d1[16];
                load16_nowait(local + unsigned(TileN) + (t / 128) * (TileN / 2) + chunk * 16, d1);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (cvt & 2) {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                        v[chunk][j] =
                            unsigned(int(v[chunk][j]) +
                                     Fold1 * (__float_as_int(__uint_as_float(d1[j]) + 12582912.0f) - 0x4B400000));
                } else {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                        v[chunk][j] = unsigned(int(v[chunk][j]) + Fold1 * int(__uint_as_float(d1[j])));
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
            const auto released = clock64();
            if (t == 0) {
                if (rank == 0)
                    arrive_local_rd(sa(&read_done));
                else
                    arrive_leader_rd(sa(&read_done));
            } // both D free
#pragma unroll
            for (int chunk = 0; chunk < TileN / 32; ++chunk) {
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const unsigned gn = tc * TileN + (t / 128) * (TileN / 2) + chunk * 16 + j;
                    if (!FULL_N || gn < unsigned(N))
                        crow[size_t(gn) * M] = __int_as_float(int(v[chunk][j]) >> FoldShift);
                }
            }
            asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0) {
                auto &z = mt[size_t(tile) * 2 + rank];
                z.payout = released - began;
                z.store = clock64() - released;
                {
                    unsigned long long g;
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                    z.pay_end_ns = g;
                }
                const auto now = clock64();
                z.total = now - prev;
                z.smid = smid_();
                prev = now;
            }
#elif EARLY_RELEASE
            unsigned v[TileN / 32][16];
#pragma unroll
            for (int chunk = 0; chunk < TileN / 32; ++chunk)
                load16_nowait(local + slot * unsigned(TileN) + (t / 128) * (TileN / 2) + chunk * 16, v[chunk]);
            asm volatile("tcgen05.wait::ld.sync.aligned;tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::
                             : "memory");
            const auto released = clock64();
            if (t == 0) {
                unsigned long long *rd = slot ? &read_done1 : &read_done;
                if (STAMPS)
                    mt[size_t(tile) * 2 + rank].c0 = gtime();
                if (rank == 0)
                    arrive_local_rd(sa(rd));
                else
                    arrive_leader_rd(sa(rd));
            }
#pragma unroll
            for (int chunk = 0; chunk < TileN / 32; ++chunk) {
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const unsigned gn = tc * TileN + dcol((t / 128) * (TileN / 2) + chunk * 16) + j;
                    if (!FULL_N || gn < unsigned(N))
                        crow[size_t(gn) * M] = __uint_as_float(v[chunk][j]);
                }
            }
            asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0) {
                auto &z = mt[size_t(tile) * 2 + rank];
                z.payout = released - began;
                z.store = clock64() - released;
                {
                    unsigned long long g;
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                    z.pay_end_ns = g;
                }
                const auto now = clock64();
                z.total = now - prev;
                z.smid = smid_();
                prev = now;
            }
#else
            for (int b = 0; b < MBLK; ++b) {
                float *const crow_b = crow + b * 256;
#pragma unroll
                for (int chunk = 0; chunk < TileN / 32; ++chunk) {
                    unsigned v[16];
                    const unsigned col = (t / 128) * (TileN / 2) + chunk * 16;
                    load16(local + unsigned(b * TileN) + col, v);
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const unsigned gn = tc * TileN + dcol(col) + j;
                        if (!FULL_N || gn < unsigned(N))
                            crow_b[size_t(gn) * M] = __uint_as_float(v[j]);
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0) {
                auto &z = mt[size_t(tile) * 2 + rank];
                z.payout = clock64() - began;
                {
                    unsigned long long g;
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(g));
                    z.pay_end_ns = g;
                }
                if (rank == 0)
                    arrive_local_rd(sa(&read_done));
                else
                    arrive_leader_rd(sa(&read_done));
                const auto now = clock64();
                z.total = now - prev;
                z.smid = smid_();
                prev = now;
            }
#endif
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (t < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,512;" ::"r"(tm) : "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
#endif
#undef M
#undef N
#undef RowTiles
#undef ColTiles
#undef Tiles
#undef TilesPerCluster
#undef TilesPerBand
} // namespace f5
#undef A_SMEM
#undef A_STAGES
#undef A_TMEM_STAGES
#undef B_STAGES
#undef CHECK_HASH
#undef COLLECTOR
#undef CUTS_PER_REQ
#undef DSLOTS
#undef EARLY_RELEASE
#undef FULL_N
#undef MBLK
#undef MN
#undef PAIR_DIRECT
#undef REF_ONLY
#undef REG_HIGH
#undef REG_LOW
#undef RELAXED_RD
#undef ROW_TILES_PER_BAND
#undef SPLIT33
#undef SPLIT_N
#undef STAMPS
#undef SUPPLY_SPLIT
#undef TILE_N
#undef SCHEME
