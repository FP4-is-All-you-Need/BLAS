#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cooperative_groups.h>
#include <cuda.h>
#include "packed_1526.h" // as the source (unconditional there; the API's copy is the same file, sha256 7e11c100)
#include "duald40_load.cuh"
#define KSEG 1
#define ISSUE_WARP 1
#define DIRECT 1
#define DSLOTS 2
#define PAYOUT_LDS 2
#define CUTS_PER_REQ 5
#define A_SMEM 1
#define A_FILL_ST 0
#define RELAYS 1
#define A_STAGES 4
#define A_TMEM_STAGES 4
#define B_STAGES 4
#define C0A_RR 1
#define TILE_N 224
#define FACES 5
#define F5_CONTRACT 1
#define SEGC 35
#define EXT103 0
#define F5_EXT 2
#define F5_AGRP 1
#define F5_AEXT 1
namespace v5k {
#ifndef LE4
#define LE4 0
#endif
#ifndef ENC3
#define ENC3 0
#endif
#if LE4
#else
#endif
#if ENC3
#endif
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
__device__ __forceinline__ void st16w(unsigned at, uint4 a, uint4 b, uint4 c, uint4 d) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
            at),
        "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w), "r"(c.x), "r"(c.y), "r"(c.z),
        "r"(c.w), "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w)
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
#ifndef CUTS_PER_REQ
#define CUTS_PER_REQ 4
#endif
constexpr int CutsPerReq = CUTS_PER_REQ;
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef FACES
#define FACES 6
#endif // unit i, block j holds face (6i + j) mod 5 = (i + j) mod 5 of the 16-k chunk (6i + j) div 5, so every (chunk,
       // face) appears once.
#ifndef F5_EXT
#define F5_EXT 0
#endif // carry sum_b t_b sum_{k in b} x (B shifted by t_b per 16-block); XSEGR requests per extension segment (the
       // extension's own bound)
#ifndef XSEGR
#define XSEGR (40 / CUTS_PER_REQ)
#endif
#ifndef F5_AEXT
#define F5_AEXT 0
#endif // the block sums, the same weights and scale set 0 as t_b x (block sums of x)) after the B-side units; the
       // read-out keeps only A's leftovers
#ifndef F5_AGRP
#define F5_AGRP 1
#endif
#ifndef F5_CONTRACT
#define F5_CONTRACT                                                                                                    \
    0 // FACES=5: 0 = (1,10,10) (lut5_1_10_10.h; A and B scales rotate: 5 + 5 sets; D + D' fits at N 192), 1 = (1,10,13)
      // (lut5_1_10_13.h;
#endif // lambda rides on A's scale alone, B has one common set: D + D' fits at N 224). TMEM = 2 TileN + 8 x 5 + SfbCols
       // x (5 or 1).
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
constexpr int NLimbs = A_LIMBS == 2 ? 5 : 3, ARowBytes = A_LIMBS ? 8 * NLimbs : 48;
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
#ifndef AREL
#define AREL 0
#endif //       with try_wait.acquire.cluster) instead of the 16 B bulk-copy token, whose completion does not order the
       //       tcgen05.st
#ifndef SEGC
#define SEGC 64
#endif
#ifndef ST16
#define ST16 0
#endif //       contiguous) using three 16-column stores instead of four pairs of 8- and 4-column stores.
#ifndef LIMB_EARLY
#define LIMB_EARLY 0
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
#ifndef MM
#define MM MN
#endif
#ifndef NN
#define NN MN
#endif
constexpr int TileM = 256, TileN = TILE_N, HalfN = TileN / 2, CtaRows = 128; // v5k: M, N at run time (gd)
constexpr int Grid = 148, Pairs = Grid / 2;
struct DimsT {
    int M, N, RowTiles, ColTiles, Tiles, TilesPerPair, TilesPerBand;
};
__constant__ DimsT gd;
#define M (gd.M)
#define N (gd.N)
#define RowTiles (gd.RowTiles)
#define ColTiles (gd.ColTiles)
#define Tiles (gd.Tiles)
#define TilesPerPair (gd.TilesPerPair)
#define TilesPerBand (gd.TilesPerBand)
#ifndef TAILSPLIT
#define TAILSPLIT 0
#endif //   segments), dealt to the pairs; their INT32 partial sums are added into C with red.global.add.s32 (C zeroed
       //   first)
#ifndef TAIL_ST
#define TAIL_ST 0
#endif
#ifndef TAIL_INZERO
#define TAIL_INZERO 1
#endif //   their share of the tail tiles' C at the kernel start, then count it (release); a part waits (acquire) before
       //   its adds
#ifndef PDL
#define PDL 0
#endif //   prologue (TMEM alloc, mbarrier init, scale stores: no global memory), and the host launches with
       //   programmatic stream serialization
#ifndef ALLOC_EARLY
#define ALLOC_EARLY 0
#endif //   host allocates pa / pb / C / mt right after the raw inputs, before the +-103 buffers (the pre-unify order),
       //   to test placement
#ifndef BWARP
#define BWARP 0
#endif //   (threads 320-351, otherwise idle) walks the supply loop; only lane 0 waits and issues, the warp meets
       //   (__syncwarp) after each request
#ifndef EXT103
#define EXT103 1
#endif //   adds; the host refuses any input that would need them (A outliers or an extension request)
constexpr int TailS = TAILSPLIT > 1 ? TAILSPLIT : 1; // v5k: the rest at run time
#define FullWaves (Tiles / Pairs)
#define FullTiles (FullWaves * Pairs)
#define TailTiles (Tiles - FullTiles)
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
#ifndef C0A_PREF
#define C0A_PREF 0
#endif
#ifndef EXT_STAGGER
#define EXT_STAGGER 0
#endif
#ifndef C0A_RR
#define C0A_RR 1
#endif
#ifndef C0A_PERSEG
#define C0A_PERSEG 0
#endif
static_assert(!(C0A_RR && (C0A_PERSEG || C0A_PREF)), "5929: one A-outlier form at a time");
static_assert(!(EXT_STAGGER && C0A_PERSEG),
              "5927: the per-segment A-outlier adds assume the extension is the last segment");
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
#ifndef ATOKEN
#define ATOKEN 0
#endif
#ifndef D3
#define D3 0
#endif //       holding the current segment's D (free once its read-out is done) (user's D3 idea, IDEA_COMBOS axis D)
#ifndef RSPLIT
#define RSPLIT 1
#endif
#ifndef D3W
#define D3W 4
#endif
#ifndef RFREE
#define RFREE 0
#endif //       columns (0..47 and 96..143 at N=192) and at the end; region stages in the order 0, 96, 48, 144 so the
       //       first two are the early half
#ifndef ONETOK
#define ONETOK 0
#endif
#ifndef WT
#define WT 0
#endif
#ifndef RDREL
#define RDREL 0
#endif // Use relaxed cluster arrivals, then tcgen05.fence::after_thread_sync on each read-out thread.
       // and before_thread_sync on the arriving thread
#ifndef PAYG
#define PAYG 1
#endif
#ifndef PAYFF
#define PAYFF 0
#endif
#ifndef PAYT
#define PAYT 0
#endif
#ifndef WPIPE
#define WPIPE 0
#endif
#ifndef ISSUE_SWAP
#define ISSUE_SWAP 0
#endif
constexpr unsigned TIssue = ISSUE_SWAP ? 288u : 256u, TALoad = ISSUE_SWAP ? 256u : 288u;
#ifndef KSEG_SACC
#define KSEG_SACC 0
#endif
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
constexpr int RowTilesPerBand = BAND; // v5k: TilesPerBand = RowTilesPerBand x ColTiles at run time (gd)
constexpr int ATile = CtaRows * ARowBytes * CutsPerReq, BTile = HalfN * 48 * CutsPerReq, ACut = CtaRows * ARowBytes,
              BCut = HalfN * 48;
constexpr int AStages = A_STAGES, BStages = B_STAGES, ATStages = A_TMEM_STAGES, BOffset = AStages * ATile,
              SaccOffset = BOffset + BStages * BTile, SaccBytes = KSEG_SACC * 256 * 4,
              SharedBytes = SaccOffset + SaccBytes;
static_assert(!KSEG_SACC || (KSEG && KSEG_SACC % 16 == 0 && KSEG_SACC <= TILE_N / 2 && !KSEG_STOREV4 && !KSEG_NOSTORE),
              "KSEG_SACC: KSEG only, 16-column groups, not with the store diagnostics");
static_assert(SharedBytes <= 231424,
              "dynamic Shared Memory: the part reports opt-in 232,448 (7139 smem_optin) minus 1,024 static");
constexpr int SfbCols = TileN > 128 ? 16 : 8; // B scale form: 16 columns for N>128, 8 for N<=128 (665)
constexpr int ASets = FACES == 5 ? 5 : 1,
              BSets = (FACES == 5 && F5_CONTRACT == 0)
                          ? 5
                          : 1; // FACES=5: scale sets = the 5 rotations (B: 1 set when lambda rides on A alone)
constexpr int TmemD = 0, TmemSfa = DSlots * TileN, TmemSfb = TmemSfa + 8 * ASets, TmemA = TmemSfb + SfbCols * BSets,
              TmemAStage = 12 * CutsPerReq; // D slots, A scales 8 cols, B scales
static_assert(TmemA + (A_SMEM ? 0 : (D3 ? (512 - TmemA) / TmemAStage : ATStages) * TmemAStage) <= 512,
              "TMEM columns"); // A_SMEM: no TMEM A ring
constexpr int NSpare = D3 ? (512 - TmemA) / TmemAStage : 0, NReg = D3 ? TileN / TmemAStage : 0,
              NST = D3 ? NSpare + NReg : ATStages;
static_assert(!D3 || (KSEG && DSlots == 2 && A_FILL_ST && !A_SMEM && !WPIPE && A_TMEM_STAGES == NSpare + NReg &&
                      NSpare >= 1 && TileN % TmemAStage == 0 &&
                      (D3W + NSpare < SEGC / CutsPerReq || D3W >= SEGC / CutsPerReq)),
              "D3: KSEG + D/D' + st fill; A_TMEM_STAGES = spare + region stages");
static_assert(PAYG >= 1 && PAYG <= 7 && (PAYG == 1 || (!PAYFF && !RFREE && !PAYT)),
              "PAYG: 1..7 chunks per wait, not with PAYFF / RFREE / the payout timers");
static_assert(!RFREE || (D3 && !PAYFF && NReg == 4 && TileN / 2 == 2 * TmemAStage && TmemAStage % 16 == 0 && D3W < 16),
              "RFREE: 4 region stages, each thread's first half of columns = one stage");
__device__ __forceinline__ int d3stage(int rin) {
    return rin < D3W ? rin % (NSpare > 0 ? NSpare : 1) : (rin - D3W) % NST;
}
__device__ __forceinline__ unsigned d3col(int st, int u) {
    const int g = st - NSpare;
    return st < NSpare ? unsigned(TmemA + st * TmemAStage)
                       : unsigned(TmemD + ((u + 1) & 1) * TileN +
                                  (RFREE ? (g & 1) * (TileN / 2) + (g >> 1) * TmemAStage : g * TmemAStage));
}
static_assert(!A_FILL_LDG || (A_FILL_ST && A_ARRIVE_RELAY),
              "LDG fill needs the st writers with the local arrive + relay");
static_assert(!WPIPE || (A_FILL_ST && A_LIMBS && A_RELEASE_LATE && !A_FILL_LDG),
              "WPIPE is written for the 3-limb st fill with the late slot release");
static_assert(!ATOKEN || (A_FILL_ST && A_ARRIVE_RELAY && DIRECT && !A_FILL_LDG),
              "ATOKEN replaces the relay of the st fill (direct-completion B)");
static_assert(!BWARP || (A_SMEM && DIRECT && !A_FILL_ST && !A_FILL_LDG && !COMBINED_AB && Threads == 352),
              "7202 BWARP: 7154's form, where threads 321-351 have no role");
static_assert(
    TailS == 1 || (KSEG && A_SMEM && !A_FILL_ST && DIRECT && !PAYFF && !KSEG_STOREV4 && !KSEG_NOSTORE && !EXT_STAGGER &&
                   !C0A_PERSEG),
    "TAILSPLIT: 7154's form; the extension at the end of the carrying part; A outliers round-robin (C0A_RR) or at the end");
static_assert(RSPLIT == 1 || (RSPLIT == 2 && WSPLIT == 2 && D3 && ONETOK && !WPIPE),
              "RSPLIT=2: two request groups of 4 writer warps (thread count from WSPLIT=2), with D3 and ONETOK");
static_assert(!ONETOK || (ATOKEN && A_RELEASE_LATE && !WPIPE), "ONETOK is written for the token form without WPIPE");
static_assert(!AREL || (ONETOK && ATOKEN && (AREL != 2 || (Relay && Relays <= 2))),
              "AREL replaces the token of the one-signal form (2: one or two relay threads)");
static_assert(
    AREL != 2 || Relays == 1 || !D3 || (D3W % 2 == 0 && NST % 2 == 0 && (SEGC / CutsPerReq) % 2 == 0),
    "AREL=2 with 2 relays: every D3 stage must always get requests of one parity (one relay per stage, or the parity waits alias)");
static_assert(FACES == 5 ? (KSEG && SEGC % 5 == 0 && SEGC % CutsPerReq == 0) : (SEGC == 64 || (KSEG && SEGC % 64 == 0)),
              "SEGC: a multiple of 64 MMAs (FACES=5: whole 96-k groups and whole requests)");
static_assert(
    FACES == 6 || (FACES == 5 && !EXT103 && !A_LIMBS && A_SMEM && !D3 && (CutsPerReq == 4 || CutsPerReq == 5) &&
                   TAILSPLIT <= 1 && F5_CONTRACT >= 0 && F5_CONTRACT <= 1),
    "FACES=5: no +-103 extension, no A limbs, A from Shared Memory (mma_s), no D3, 4 or 5 units per request, no tail split");
static_assert(
    F5_EXT != 2 || (KSEG_SACC == 0 && TailS == 1),
    "F5_EXT=2 (ENC5 stages 2 / 3): the read-out adds into the 112 register sums before the store (no Shared sums, no tail split)");
static_assert(
    !F5_EXT || (FACES == 5 && F5_CONTRACT == 1 && !EXT103 && XSEGR >= 1 && XSEGR * CutsPerReq <= 40),
    "F5_EXT: five faces, lambda on A alone (scale set 0 = |lambda| on A, one B set), no +-103 path, at most 40 extension MMAs per segment");
static_assert(!ST16 || (RSPLIT > 1 && CutsPerReq == 4),
              "ST16: one warp stores all 4 cuts of its request (48 contiguous columns)");
static_assert(!LIMB_EARLY || (ONETOK && A_LIMBS == 2 && ENC3),
              "LIMB_EARLY: the C2 limbs with the one-signal token form");
static_assert(A_LIMBS != 2 || (LE4 && ENC3), "CARRY 2 uses enc3::to_u / to_code (le4 LUT)");
static_assert(!WPIPE || A_LIMBS != 2, "WPIPE builds only the 3-limb faces");
static_assert(!ENC3 || (LE4 && A_LIMBS), "enc3 needs the le4 LUT and the 3-limb A");
static_assert(!A_LIMBS || (A_FILL_ST && !A_FILL_LDG), "3-limb A needs the Shared st fill");
static_assert(!ISSUE_WARP || A_FILL_ST || A_SMEM,
              "whole-warp elect is only wired where no single-lane cp thread commits (st fill or A from Shared)");
static_assert(!DIRECT || (ATile % 256 == 0 && BTile % 256 == 0 && ATile / 256 <= 256 && BTile / 256 <= 256),
              "tensor-map boxes of 256-byte rows");
static_assert(!A_SMEM || (DIRECT && !A_FILL_ST && A_TMEM_STAGES == A_STAGES),
              "A_SMEM: direct cp-path build, the A ring is the Shared A ring");
static_assert(!COMBINED_AB || (A_SMEM && DIRECT && A_STAGES == B_STAGES),
              "combined A+B stage barriers need A_SMEM + DIRECT and equal stage counts");
static_assert(Relays == 1 || (ATStages % Relays == 0 && (DIRECT || BStages % Relays == 0)),
              "each stage must always have the same relay");
__host__ __device__ constexpr int tile_byte(int row, int w) {
    return (row / 8) * 384 + (w / 16) * 128 + (row % 8) * 16 + w % 16;
}
__device__ __forceinline__ unsigned tile_row(unsigned tile) {
    return (tile / TilesPerBand) * RowTilesPerBand + tile % RowTilesPerBand;
}
__device__ __forceinline__ unsigned tile_col(unsigned tile) { return (tile % TilesPerBand) / RowTilesPerBand; }

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
static_assert(RDREL == 0 || (RDREL == 2 && KSEG && !D3 && !REMOTE_RELAXED && !PAYOUT16),
              "RDREL=2: the KSEG read-out of 7154's form (not D3, not REMOTE_RELAXED)");
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
        warr, wt1, wa1, wst1, warr1, pl0, pw0, pc0, pb0, pl1, pw1, pc1, pb1, xb0, xs0, xw0, xf0, xz0, xb1, xs1, xw1,
        xf1, xz1;
    unsigned smid;
};
struct Scales {
    unsigned a0[5], a1[5], b0[5], b1[5];
}; // per rotation (FACES=5: set i = blocks j of face (i + j) mod 5); FACES=6 uses set 0
struct C0A {
    const int *arow;
    const int *aent;
    const int8_t *bt;
    int on;
};
__constant__ C0A c0a;
struct F5C {
    const int *arow;
    const int *aent;
    const int8_t *y2;
    const short *y2s;
    const int *bcol;
    const int *bent;
    const int8_t *xa;
    const int *arow1;
    int np, k, m;
    const int *segc;
};
__constant__ F5C f5c;
__constant__ int c_segc;
#if F5_EXT == 2
// ENC5 stage 3, B leftovers (sparse): C[n][m] += sum over column n's entries (k, d) of d x x[m][k] (the original A).
// One thread per (row, listed column) holds the whole sum, so no atomics. Launched right after the GEMM kernel in the
// same stream (and timed with it).
__global__ void f5_fixb(int *C, int Mr, int K, const int *cols, const int *bcol, const int *bent, const int8_t *xa) {
    const int n = cols[blockIdx.y], m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= Mr)
        return;
    const int q0 = __ldg(bcol + n), q1 = __ldg(bcol + n + 1);
    int sum_ = 0;
    for (int e = q0; e < q1; ++e) {
        const int ent = __ldg(bent + e);
        sum_ += ((ent & 255) - 128) * int(__ldg(xa + size_t(m) * size_t(K) + (ent >> 8)));
    }
    C[size_t(n) * size_t(Mr) + m] += sum_;
}
#endif
__device__ __forceinline__ int sb8(unsigned w, unsigned sel) {
    int r;
    asm("prmt.b32 %0,%1,0,%2;" : "=r"(r) : "r"(w), "r"(sel));
    return r;
} // A: cols 0-3 = SF0..SF3, cols 4-7 = SF4,SF5; B: cols 0-7 / 8-15 the same bytes

// Threads: 0..255 payout (both CTAs), 256 issue (rank 0), 288 A supply, 320 B supply (both CTAs), 352/353 relay of rank
// 1's landings to rank 0 (531's form), 384 Shared -> TMEM copy of A with tcgen05.cp.cta_group::2 (rank 0; each CTA's
// own smem -> TMEM).
__device__ unsigned long long g_zdone;
__device__ __forceinline__ unsigned long long ld_acq_gpu(const unsigned long long *p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0,[%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__global__ __launch_bounds__(Threads, 1) void p2_kernel(
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ B, float *C, Metrics *mt, int cuts,
    Scales sc, const __grid_constant__ CUtensorMap tmEA, const __grid_constant__ CUtensorMap tmEB,
    const unsigned char *__restrict__ ETAB, int emax, unsigned long long zep) {
    const unsigned q = blockIdx.x / 2, t = threadIdx.x;
    auto cluster = cooperative_groups::this_cluster();
    const unsigned rank = cluster.block_rank();
    const auto begin0 = clock64();
    unsigned long long gbegin;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(gbegin));
    if (PDL)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __shared__ unsigned base;
    __shared__ __align__(16) unsigned char atok[16 * 4 * WSPLIT];
    __shared__ __align__(8) unsigned long long afull[AStages], alocal[AStages], aempty[AStages], aready[ATStages],
        aready_r[ATStages], tmem_empty[ATStages], bfull[BStages], blocal[BStages], bempty[BStages], drain[DSlots],
        read_done[DSlots], rdone_local[DSlots], rdone_half[DSlots];
    extern __shared__ __align__(1024) unsigned char sm[];
    if (t == 0) {
        for (int s = 0; s < AStages; ++s) {
            mbar_init(sa(&afull[s]), COMBINED_AB ? 2 : ((A_FILL_ST || DIRECT) ? 1 : 2));
            mbar_init(sa(&alocal[s]), 1);
            mbar_init(sa(&aempty[s]),
                      ONETOK ? (LIMB_EARLY ? (RSPLIT > 1 ? 4 : 4 * WSplit) : 1) : (A_FILL_ST ? 4 * WSplit : 1));
        }
        for (int s = 0; s < ATStages; ++s) {
            mbar_init(sa(&aready[s]), ONETOK ? 1 : (Relay ? 4 * WSplit : (A_FILL_ST ? 8 : 1)));
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
            mbar_init(sa(&rdone_local[s]), 1);
            mbar_init(sa(&rdone_half[s]), 1);
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
        for (int z = 0; z < ASets; ++z)
            for (int c = 0; c < 8; ++c)
                tst(local + TmemSfa + 8 * z + c, c < 4 ? sc.a0[z] : sc.a1[z]);
        for (int z = 0; z < BSets; ++z)
            for (int c = 0; c < SfbCols; ++c)
                tst(local + TmemSfb + SfbCols * z + c, c < SfbCols / 2 ? sc.b0[z] : sc.b1[z]);
        asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::: "memory");
    }
    if (PDL)
        asm volatile("griddepcontrol.wait;" ::: "memory");
    __shared__ unsigned char s_ereq[64]; /* v5k: the host checks TilesPerPair <= 64 (s_ereq) */
    for (int k = int(t); k < TilesPerPair; k += blockDim.x) {
        const unsigned tl = k * Pairs + q;
        s_ereq[k] = tl < unsigned(Tiles) ? ETAB[tl] : 0;
    }
    __syncthreads();
    cluster.sync();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const unsigned sm_base = sa(sm);
    const int reqs = (cuts + CutsPerReq - 1) / CutsPerReq, padded = reqs * CutsPerReq;
    const size_t a_tile_bytes = size_t(padded) * ACut, b_tile_bytes = size_t(padded) * BCut;
    if ((t == TALoad && !A_FILL_LDG) || (BWARP ? (t >= 320 && t < 352) : t == 320)) {
        const bool side_b = t >= 320;
        const int depth = side_b ? BStages : AStages;
        const unsigned bytes = side_b ? BTile : ATile;
        int gbase = 0;
        for (int k = 0;; ++k) {
            unsigned tile;
            int part;
            if (!item_of(q, k, tile, part))
                break;
            const unsigned char *src = side_b ? (B + size_t(2 * tile_col(tile) + rank) * b_tile_bytes)
                                              : (A + size_t(2 * tile_row(tile) + rank) * a_tile_bytes);
            const bool carry = part < 0 || part == TailS - 1;
            const int rb = part < 0 ? 0 : part * (reqs / TailS), rmain = part < 0 ? reqs : reqs / TailS;
            const int ne = F5_EXT ? emax : (EXT103 && carry ? int(TailS > 1 ? ETAB[tile] : s_ereq[k]) : 0),
                      nreq = rmain + ne;
            const int segc_s = (KSEG && cuts > SEGC) ? SEGC : cuts,
                      es = (EXT_STAGGER && ne > 0) ? (int(q) % (cuts / segc_s + 1)) * (segc_s / CutsPerReq) : rmain;
            const size_t ebase = side_b ? size_t(2 * tile_col(tile) + rank) * size_t(emax) * bytes
                                        : size_t(2 * (F5_EXT ? tile_row(tile) : tile) + rank) * size_t(emax) *
                                              bytes; // F5_EXT: the A extension is per row tile
            for (int rr = 0; rr < nreq; ++rr) {
                const int G = gbase + rr, slot = G % depth;
                const int r = rb + (rr < es ? rr : rr - ne);
                if (!BWARP || !side_b || (t & 31) == 0) {
                    if (G >= depth)
                        await(sa((side_b && !COMBINED_AB) ? &bempty[slot] : &aempty[slot]), ((G / depth) + 1) & 1);
                    const unsigned dest = sm_base + (side_b ? BOffset + slot * BTile : slot * ATile);
                    if (DIRECT && (side_b || !A_FILL_ST)) {
                        const unsigned lbar_local = sa((side_b && !COMBINED_AB) ? &bfull[slot] : &afull[slot]),
                                       lbar = peer0(lbar_local);
                        const bool ext = rr >= es && rr < es + ne;
                        const size_t off =
                            ext ? ebase + size_t(rr - es) * bytes : size_t(src - (side_b ? B : A)) + size_t(r) * bytes;
                        const CUtensorMap *map = ext ? (side_b ? &tmEB : &tmEA) : (side_b ? &tmB : &tmA);
                        if (rank == 0)
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ::"r"(lbar_local),
                                "r"(2u * bytes)
                                : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                                "r"(dest),
                            "l"(reinterpret_cast<unsigned long long>(map)), "r"(0), "r"(int(off / 256)), "r"(lbar)
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
                if (BWARP && side_b)
                    __syncwarp();
            }
            gbase += nreq;
        }
    } else if (Relay && (!ATOKEN || AREL == 2) && rank == 1 &&
               (t == 352 || t == 353 || (Relays > 1 && (t == 384 || t == 385)) ||
                (Relays > 2 && (t == XR || t == XR + 1 || t == XR + 32 || t == XR + 33)))) {
        const bool side_b = t & 1;
        const int ri = t < 384 ? 0 : (t < 416 ? 1 : (t < XR + 32 ? 2 : 3)), depth = side_b ? BStages : ATStages;
        unsigned d3rph = 0;
        if (!(DIRECT && side_b))
            for (int k = 0; k < TilesPerPair; ++k) {
                const unsigned tile = k * Pairs + q;
                if (tile >= Tiles)
                    break;
                for (int r = 0; r < reqs; ++r) {
                    const int G = k * reqs + r;
                    int slot = G % depth;
                    unsigned ph = (G / depth) & 1;
                    if (D3 && !side_b) {
                        const int rps3 = (cuts > SEGC ? SEGC : cuts) / CutsPerReq;
                        slot = d3stage(r % rps3);
                        ph = (d3rph >> slot) & 1u;
                        d3rph ^= 1u << slot;
                    }
                    if (Relays > 1 && (G % Relays) != ri)
                        continue;
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
        unsigned d3ph = 0, d3used = 0;
#if WPIPE
        {
            int ntile = 0;
            for (int k = 0; k < TilesPerPair; ++k) {
                if (k * Pairs + q >= Tiles)
                    break;
                ++ntile;
            }
            const int total = ntile * reqs;
            uint4 wA[CutsPerReq / WSplit][3], wB[CutsPerReq / WSplit][3];
            auto build = [&](int G, uint4(&w)[CutsPerReq / WSplit][3]) {
                const int slot = G % AStages;
                await(sa(&afull[slot]), (G / AStages) & 1);
#pragma unroll
                for (int j = 0; j < CutsPerReq / WSplit; ++j) {
                    const int c = grp + j * WSplit;
                    const unsigned char *b = sm + slot * ATile + c * ACut + row * 8;
                    const uint2 l0 = *reinterpret_cast<const uint2 *>(b),
                                l1 = *reinterpret_cast<const uint2 *>(b + CtaRows * 8),
                                l2 = *reinterpret_cast<const uint2 *>(b + 2 * CtaRows * 8);
#if ENC3
                    uint32_t e3[3], o3[3];
                    enc3::three(l0.x, l1.x, l2.x, true, e3);
                    enc3::three(l0.y, l1.y, l2.y, true, o3);
                    w[j][0] = make_uint4(l0.x, l0.y, l1.x, l1.y);
                    w[j][1] = make_uint4(l2.x, l2.y, e3[0], o3[0]);
                    w[j][2] = make_uint4(e3[1], o3[1], e3[2], o3[2]);
                }
            };
#else
                    const auto e = integer_planes::form(l2.x, l1.x, l0.x), o = integer_planes::form(l2.y, l1.y, l0.y);
                    w[j][0] = make_uint4(l0.x, l0.y, l1.x, l1.y);
                    w[j][1] = make_uint4(l2.x, l2.y, e.d12, o.d12);
                    w[j][2] = make_uint4(e.d01, o.d01, e.d02, o.d02);
                }
            };
#endif
            auto put = [&](int G, const uint4(&w)[CutsPerReq / WSplit][3]) {
                const int tslot = G % ATStages;
                if (G >= ATStages)
                    await(sa(&tmem_empty[tslot]), ((G / ATStages) + 1) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
            };
            auto finish = [&](int G) {
                const int slot = G % AStages, tslot = G % ATStages;
                asm volatile(
                    "tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;fence.proxy.async.shared::cta;" ::
                        : "memory");
                __syncwarp();
                if (t % 32 == 0) {
                    arrive_local(sa(&aempty[slot]));
                    if (ATOKEN && rank == 1) {
                        const unsigned tk = sa(atok) + 16 * ((t - 416) / 32);
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],16,[%2];" ::
                                "r"(peer0(tk)),
                            "r"(tk), "r"(peer0(sa(&aready_r[tslot])))
                            : "memory");
                    } else if (Relay)
                        arrive_local(sa(&aready[tslot]));
                    else
                        arrive_remote(ar_remote + 8 * tslot);
                }
            };
            if (total > 0)
                build(0, wA);
            for (int G = 0; G < total; G += 2) {
                put(G, wA);
                if (G + 1 < total)
                    build(G + 1, wB);
                finish(G);
                if (G + 1 < total) {
                    put(G + 1, wB);
                    if (G + 2 < total)
                        build(G + 2, wA);
                    finish(G + 1);
                }
            }
        }
#else
        constexpr int CS = RSPLIT > 1 ? 1 : WSplit;
        const int cg = RSPLIT > 1 ? 0 : grp;
        int d3wu = -1, d3wh = -1;
        for (int k = 0; k < TilesPerPair; ++k) {
            const unsigned tile = k * Pairs + q;
            if (tile >= Tiles)
                break;
            unsigned long long wt = 0, wa = 0, wst = 0, warr = 0, tb = 0, ts = 0, tw = 0, tf = 0, tz = 0, xb = 0,
                               xs = 0, xf = 0;
#if D3
            const int segcw = cuts > SEGC ? SEGC : cuts, nsegw = cuts / segcw, rpsw = segcw / CutsPerReq;
#endif
            for (int r = 0; r < reqs; ++r) {
                const int G = k * reqs + r, slot = G % AStages;
#if D3
                const int uw = k * nsegw + r / rpsw, rin = r % rpsw, tslot = d3stage(rin);
                const auto x0 = clock64();
                if (RSPLIT > 1 && (G % RSPLIT) != grp) {
                    d3ph ^= 1u << tslot;
                    d3used |= 1u << tslot;
                    continue;
                }
                if (tslot >= NSpare && uw >= 1) {
                    const bool h = RFREE && tslot - NSpare < 2;
                    if (h ? d3wh != uw : d3wu != uw) {
                        await(sa(h ? &rdone_half[(uw - 1) & 1] : &rdone_local[(uw - 1) & 1]), ((uw - 1) >> 1) & 1);
                        if (h)
                            d3wh = uw;
                        else
                            d3wu = uw;
                    }
                }
                if ((d3used >> tslot) & 1u)
                    await(sa(&tmem_empty[tslot]), ((d3ph >> tslot) & 1u) ^ 1u);
                const auto x1 = clock64();
#else
                const int tslot = G % ATStages, uw = 0;
                const auto x0 = clock64();
                if (G >= ATStages)
                    await(sa(&tmem_empty[tslot]), ((G / ATStages) + 1) & 1);
                const auto x1 = clock64();
#endif
                await(sa(&afull[slot]), (G / AStages) & 1);
                const auto x2 = clock64();
                wa += x2 - x1;
                wt += x1 - x0;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                uint4 w[CutsPerReq / CS][3];
#if A_LIMBS == 2 && LIMB_EARLY
                uint2 L[CutsPerReq / CS][5];
#pragma unroll
                for (int j = 0; j < CutsPerReq / CS; ++j) {
                    const unsigned char *b = sm + slot * ATile + (cg + j * CS) * ACut + row * 8;
#pragma unroll
                    for (int q = 0; q < 5; ++q)
                        L[j][q] = *reinterpret_cast<const uint2 *>(b + q * CtaRows * 8);
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (t % 32 == 0)
                    arrive_local(sa(&aempty[slot]));
#pragma unroll
                for (int j = 0; j < CutsPerReq / CS; ++j) {
                    const uint2 l0 = L[j][0], l1 = L[j][1], l2 = L[j][2], l3 = L[j][3], l4 = L[j][4];
                    const integer_planes::Diff e{l4.x, l3.x, enc3::to_code(enc3::to_u(l3.x) + enc3::to_u(l4.x) - enc3::H)},
                        o{l4.y, l3.y,
                          enc3::to_code(enc3::to_u(l3.y) + enc3::to_u(l4.y) - enc3::H)}; // as the A_LIMBS==2 form below
                    w[j][0] = make_uint4(l0.x, l0.y, l1.x, l1.y);
                    w[j][1] = make_uint4(l2.x, l2.y, e.d12, o.d12);
                    w[j][2] = make_uint4(e.d01, o.d01, e.d02, o.d02);
                }
#elif A_LIMBS
#pragma unroll
                for (int j = 0; j < CutsPerReq / CS; ++j) {
                    const int c = cg + j * CS;
                    const unsigned char *b = sm + slot * ATile + c * ACut + row * 8;
                    const uint2 l0 = *reinterpret_cast<const uint2 *>(b),
                                l1 = *reinterpret_cast<const uint2 *>(b + CtaRows * 8),
                                l2 = *reinterpret_cast<const uint2 *>(b + 2 * CtaRows * 8);
#if A_LIMBS == 2
                    const uint2 l3 = *reinterpret_cast<const uint2 *>(b + 3 * CtaRows * 8),
                                l4 = *reinterpret_cast<const uint2 *>(b + 4 * CtaRows * 8);
                    const integer_planes::Diff e{l4.x, l3.x, enc3::to_code(enc3::to_u(l3.x) + enc3::to_u(l4.x) - enc3::H)},
                        o{l4.y, l3.y,
                          enc3::to_code(enc3::to_u(l3.y) + enc3::to_u(l4.y) -
                                        enc3::H)}; // d01=a2-a1, d12=a1-a0, d02=a2-a0
#elif ENC3
                    uint32_t e3[3], o3[3];
                    enc3::three(l0.x, l1.x, l2.x, true, e3);
                    enc3::three(l0.y, l1.y, l2.y, true, o3); // out = a1-a0, a2-a1, a2-a0
                    const integer_planes::Diff e{e3[1], e3[0], e3[2]},
                        o{o3[1], o3[0], o3[2]}; // same fields as form(): d01=a2-a1, d12=a1-a0, d02=a2-a0
#else
                    const auto e = integer_planes::form(l2.x, l1.x, l0.x),
                               o = integer_planes::form(l2.y, l1.y, l0.y); // d01=a2-a1, d12=a1-a0, d02=a2-a0
#endif
                    w[j][0] = make_uint4(l0.x, l0.y, l1.x, l1.y);
                    w[j][1] = make_uint4(l2.x, l2.y, e.d12, o.d12);
                    w[j][2] = make_uint4(e.d01, o.d01, e.d02, o.d02);
                }
#else
#pragma unroll
                for (int j = 0; j < CutsPerReq / CS; ++j) {
                    const int c = cg + j * CS;
#pragma unroll
                    for (int i = 0; i < 3; ++i)
                        w[j][i] = *reinterpret_cast<const uint4 *>(sm + slot * ATile + c * ACut + off[i]);
                }
#endif
                if (WT) {
#pragma unroll
                    for (int j = 0; j < CutsPerReq / CS; ++j)
                        asm volatile(""
                                     : "+r"(w[j][0].x), "+r"(w[j][0].y), "+r"(w[j][0].z), "+r"(w[j][0].w),
                                       "+r"(w[j][1].x), "+r"(w[j][1].y), "+r"(w[j][1].z), "+r"(w[j][1].w),
                                       "+r"(w[j][2].x), "+r"(w[j][2].y), "+r"(w[j][2].z), "+r"(w[j][2].w));
                    asm volatile("mov.u64 %0,%%clock64;" : "=l"(xb)::"memory");
                }
#if !A_RELEASE_LATE
                __syncwarp();
                if (t % 32 == 0)
                    arrive_local(sa(&aempty[slot]));
#endif
#if ST16
                {
                    const unsigned at = tm + lane_row + (D3 ? d3col(tslot, uw) : unsigned(TmemA + tslot * TmemAStage));
                    st16w(at, w[0][0], w[0][1], w[0][2], w[1][0]);
                    st16w(at + 16, w[1][1], w[1][2], w[2][0], w[2][1]);
                    st16w(at + 32, w[2][2], w[3][0], w[3][1], w[3][2]);
                }
#else
#pragma unroll
                for (int j = 0; j < CutsPerReq / CS; ++j) {
                    const int c = cg + j * CS;
                    const unsigned at =
                        tm + lane_row + (D3 ? d3col(tslot, uw) : unsigned(TmemA + tslot * TmemAStage)) + c * 12;
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(at),
                                 "r"(w[j][0].x), "r"(w[j][0].y), "r"(w[j][0].z), "r"(w[j][0].w), "r"(w[j][1].x),
                                 "r"(w[j][1].y), "r"(w[j][1].z), "r"(w[j][1].w)
                                 : "memory");
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};" ::"r"(at + 8),
                                 "r"(w[j][2].x), "r"(w[j][2].y), "r"(w[j][2].z), "r"(w[j][2].w)
                                 : "memory");
                }
#endif
                if (WT)
                    asm volatile("mov.u64 %0,%%clock64;" : "=l"(xs)::"memory");
                asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::: "memory");
                const auto x3 = clock64();
#if A_RELEASE_LATE
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (WT)
                    asm volatile("mov.u64 %0,%%clock64;" : "=l"(xf)::"memory");
#if ONETOK
                asm volatile("barrier.cta.sync %1,%0;" ::"r"(RSPLIT > 1 ? 128 : 128 * WSPLIT),
                             "r"(RSPLIT > 1 ? 2 + grp : 2)
                             : "memory");
                if (t == unsigned(416 + (RSPLIT > 1 ? 128 * grp : 0))) {
                    if (!LIMB_EARLY)
                        arrive_local(sa(&aempty[slot]));
                    if (rank == 1) {
                        if (AREL == 3) {
                            const unsigned rb = peer0(sa(&aready_r[tslot]));
                            asm volatile(
                                "tcgen05.fence::after_thread_sync;tcgen05.fence::before_thread_sync;mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::
                                    "r"(rb)
                                : "memory");
                        } else if (AREL == 2)
                            arrive_local(sa(&aready[tslot]));
                        else if (AREL)
                            arrive_remote(peer0(sa(&aready_r[tslot])));
                        else {
                            const unsigned tk = sa(atok);
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],16,[%2];" ::
                                    "r"(peer0(tk)),
                                "r"(tk), "r"(peer0(sa(&aready_r[tslot])))
                                : "memory");
                        }
                    } else
                        arrive_local(sa(&aready[tslot]));
                }
                if (false) {
#else
                __syncwarp();
                if (t % 32 == 0) {
                    arrive_local(sa(&aempty[slot]));
#endif
                    if (ATOKEN && rank == 1) {
                        const unsigned tk = sa(atok) + 16 * ((t - 416) / 32);
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],16,[%2];" ::
                                "r"(peer0(tk)),
                            "r"(tk), "r"(peer0(sa(&aready_r[tslot])))
                            : "memory");
                    } else if (Relay)
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
                if (D3) {
                    d3ph ^= 1u << tslot;
                    d3used |= 1u << tslot;
                }
                if (WT) {
                    tb += xb - x2;
                    ts += xs - xb;
                    tw += x3 - xs;
                    tf += xf - x3;
                    tz += x4 - xf;
                }
            }
            if (WT && t == 416) {
                auto &z = mt[tile];
                if (rank == 0) {
                    z.xb0 = tb;
                    z.xs0 = ts;
                    z.xw0 = tw;
                    z.xf0 = tf;
                    z.xz0 = tz;
                } else {
                    z.xb1 = tb;
                    z.xs1 = ts;
                    z.xw1 = tw;
                    z.xf1 = tf;
                    z.xz1 = tz;
                }
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
#endif
    } else if (rank == 0 && (ISSUE_WARP ? (t >= TIssue && t < TIssue + 32) : t == TIssue)) {
        const unsigned sfa = tm + TmemSfa, sfb = tm + TmemSfb;
        const unsigned long long bdesc0 = tile_desc(sm_base + BOffset);
        const unsigned aBeg = tm + TmemA, rr0 = sa(&aready_r[0]), ar0 = A_SMEM ? sa(&afull[0]) : sa(&aready[0]),
                       te0 = A_SMEM ? sa(&aempty[0]) : sa(&tmem_empty[0]), bf0 = sa(&bfull[0]), be0 = sa(&bempty[0]),
                       ar_end = ar0 + 8 * ATStages, bf_end = bf0 + 8 * BStages;
        const unsigned long long adesc0 = tile_desc(sm_base);
        unsigned long long ad = adesc0;
        unsigned rr_bar = rr0, aa = aBeg, ar_bar = ar0, te_bar = te0, bf_bar = bf0, be_bar = be0, aph = 0, bph = 0,
                 d3iph = 0;
        unsigned long long dd = bdesc0;
        const int segc0 = FACES == 5 ? __ldg(f5c.segc) : SEGC, segc = (KSEG && cuts > segc0) ? segc0 : cuts,
                  nseg = FACES == 5 ? (cuts + segc - 1) / segc : cuts / segc, rps = segc / CutsPerReq;
        const int rlast =
            (cuts - (nseg - 1) * segc) / CutsPerReq; // FACES=5 (10:3x): the last segment may be shorter (K not a
                                                     // multiple of the segment); still whole groups and requests
        int ubase = 0;
        for (int k = 0;; ++k) {
            unsigned tile;
            int part;
            if (!item_of(q, k, tile, part))
                break;
            const bool carry = part < 0 || part == TailS - 1;
            const int nsgm = part < 0 ? nseg : nseg / TailS;
            const int ne = F5_EXT ? emax : (EXT103 && carry ? int(TailS > 1 ? ETAB[tile] : s_ereq[k]) : 0),
                      nsegx = nsgm + (F5_EXT ? (ne + XSEGR - 1) / XSEGR
                                             : (ne > 0 ? 1 : 0)); // F5_EXT: extension segments of XSEGR requests
            unsigned long long ns0, fi = 0;
            asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns0));
            unsigned long long aw = 0, bw = 0, dw = 0, cb = 0;
            const int epos = (EXT_STAGGER && ne > 0) ? int(q) % (nseg + 1) : nsgm;
            for (int sg = 0; sg < (KSEG ? nsegx : 1); ++sg) {
                const int u = KSEG ? ubase + sg : k;
                const int nr =
                    (F5_EXT && sg >= nsgm)
                        ? (ne - (sg - nsgm) * XSEGR < XSEGR ? ne - (sg - nsgm) * XSEGR : XSEGR)
                        : ((KSEG && ne > 0 && sg == epos) ? ne : ((FACES == 5 && sg == nsgm - 1) ? rlast : rps));
                const bool xseg = F5_EXT && sg >= nsgm; // F5_EXT: an extension segment: every MMA uses scale set 0
                                                        // (block j = digit j, weight |lambda_(j mod 5)|)
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
                unsigned rot5 = 0; // FACES=5: the rotation of this request's first unit; a segment is whole 96-k
                                   // groups, so it starts at 0
#if KFIX
#pragma unroll
                for (int r = 0; r < KFIX / 16 / CutsPerReq; ++r) {
#else
                for (int r = 0; r < (KSEG ? nr : reqs);
                     ++r) { // add-and-wrap ring state (669: no division or modulo in the issue loop)
#endif
#if D3
                    {
                        const int st = d3stage(r);
                        ar_bar = ar0 + 8u * st;
                        rr_bar = rr0 + 8u * st;
                        te_bar = te0 + 8u * st;
                        aa = tm + d3col(st, u);
                        aph = (d3iph >> st) & 1u;
                        d3iph ^= 1u << st;
                    }
#endif
                    const auto w0 = clock64();
                    await(ar_bar, aph);
                    if (ATOKEN && !AREL)
                        asm volatile(
                            ELECT_PFX
                            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ELECT_SFX::"r"(rr_bar),
                            "r"(ONETOK ? 16u : 16u * 4u * WSPLIT)
                            : "memory");
                    if (Relay)
                        await(rr_bar, aph);
                    const auto w1 = clock64();
                    if (!COMBINED_AB)
                        await(bf_bar, bph);
                    bw += clock64() - w1;
                    aw += w1 - w0;
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#if A_SMEM && FACES == 5
                    {
                        constexpr unsigned SB = BSets == 5 ? unsigned(SfbCols) : 0u;
                        unsigned z = rot5; // unit c of this request uses rotation (rot5 + c) mod 5, by add-and-wrap
                        if (r == 0) {
                            if (sg == 0)
                                fi = clock64();
                            mma_s(d, ad, dd, sfa + 8u * z, sfb + SB * z, false);
                        } else
                            mma_s(d, ad, dd, sfa + 8u * z, sfb + SB * z, true);
#pragma unroll
                        for (int c = 1; c < CutsPerReq; ++c) {
                            z = xseg ? 0u : (z == 4u ? 0u : z + 1u);
                            mma_s(d, ad + ((unsigned long long)(c * ACut) >> 4),
                                  dd + ((unsigned long long)(c * BCut) >> 4), sfa + 8u * z, sfb + SB * z, true);
                        }
                        rot5 = xseg ? 0u : (z == 4u ? 0u : z + 1u);
                    } // the next request starts CutsPerReq units on (an extension segment stays at set 0) (4 units: one
                      // rotation back; 5 units: rotation 0 again)
#elif A_SMEM
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
                    if (!D3) {
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
            ubase += nsegx;
            const auto le = clock64();
            if (t == TIssue) {
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
        if (TAIL_INZERO && TailS > 1 && zep) {
            const unsigned long long per = (unsigned long long)TileM * TileN,
                                     total = per * TailTiles; // with a release at gpu scope (CUTLASS's semaphore form:
                                                              // bar.sync, then red.release.gpu)
            for (unsigned long long e = (unsigned long long)blockIdx.x * 256 + t; e < total;
                 e += (unsigned long long)Grid * 256) {
                const unsigned tl = unsigned(FullTiles) + unsigned(e / per), r = unsigned(e % per), m = r % TileM,
                               n = r / TileM, gn = tile_col(tl) * TileN + n;
                if (gn < unsigned(N))
                    reinterpret_cast<int *>(C)[size_t(gn) * M + tile_row(tl) * TileM + m] = 0;
            }
            asm volatile("barrier.cta.sync 1,256;" ::: "memory");
            if (t == 0)
                asm volatile("red.release.gpu.global.add.u64 [%0],1;" ::"l"(&g_zdone) : "memory");
        }
        const int segc0 = FACES == 5 ? __ldg(f5c.segc) : SEGC, segc = cuts > segc0 ? segc0 : cuts,
                  nseg = FACES == 5 ? (cuts + segc - 1) / segc : cuts / segc;
        const unsigned col0 = (t / 128) * (TileN / 2);
        int ubase = 0; // FACES=5: the issuer's segment count (a shorter last one)
        for (int k = 0;; ++k) {
            unsigned tile;
            int part;
            if (!item_of(q, k, tile, part))
                break;
            const unsigned tr = tile_row(tile), tc = tile_col(tile);
            const bool carry = part < 0 || part == TailS - 1;
            const int nsgm =
                part < 0 ? nseg
                         : nseg / TailS; // unified: only the carrying item has the extension and the A-outlier adds
            const int nsegx =
                nsgm + (F5_EXT ? (emax + XSEGR - 1) / XSEGR
                               : ((EXT103 && carry && (TailS > 1 ? int(ETAB[tile]) : int(s_ereq[k])) > 0) ? 1 : 0));
            constexpr int RC = TileN / 2 - KSEG_SACC;
            int acc[RC > 0 ? RC : 1];
            unsigned long long seen = 0, ns1 = 0, pl = 0, pw = 0, pc = 0, pb = 0;
            int *const sacc = reinterpret_cast<int *>(sm + SaccOffset) +
                              t; // this thread's Shared sums: sacc[256*j], conflict-free across the warp
            const unsigned am = tr * TileM + rank * CtaRows + (t % 128), an0 = tc * TileN + col0;
            int ea = 0, eaend = 0;
            if ((C0A_PERSEG || C0A_RR) && EXT103 && c0a.on && carry) {
                ea = __ldg(c0a.arow + am);
                eaend = __ldg(c0a.arow + am + 1);
            }
            int entn = 0;
            if (C0A_RR && EXT103 && c0a.on && ea < eaend)
                entn = __ldg(c0a.aent + ea);
            for (int sg = 0; sg < nsegx; ++sg) {
                const int u = ubase + sg, ds = u % DSlots, use = u / DSlots;
                await(sa(&drain[ds]), use & 1);
                if (sg == nsegx - 1) {
                    seen = clock64();
                    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1));
                }
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                unsigned long long q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                unsigned sink = 0;
#define ACCUM(x, jj)                                                                                                   \
    {                                                                                                                  \
        if (PAYT)                                                                                                      \
            sink ^= unsigned(x);                                                                                       \
        if ((jj) < RC)                                                                                                 \
            acc[(jj) < RC ? (jj) : 0] = sg == 0 ? (x) : acc[(jj) < RC ? (jj) : 0] + (x);                               \
        else {                                                                                                         \
            int *const p = sacc + 256 * ((jj) - RC);                                                                   \
            *p = sg == 0 ? (x) : *p + (x);                                                                             \
        }                                                                                                              \
    }
#if PAYFF
                {
                    unsigned vv[(TileN / 2) / 16][16];
                    if (PAYT)
                        q0 = clock64();
#pragma unroll
                    for (int i = 0; i < (TileN / 2) / 16; ++i)
                        ld16nw(local + TmemD + ds * TileN + col0 + 16 * i, vv[i]);
                    if (PAYT)
                        q1 = clock64();
                    waitld16(vv[0]);
#pragma unroll
                    for (int i = 1; i < (TileN / 2) / 16; ++i)
                        tie16(vv[i]);
                    if (PAYT)
                        q2 = clock64();
#pragma unroll
                    for (int i = 0; i < (TileN / 2) / 16; ++i) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const int x = FASTCVT ? f2i_int24(vv[i][j]) : int(__uint_as_float(vv[i][j]));
                            ACCUM(x, 16 * i + j)
                        }
                    }
                    if (PAYT) {
                        asm volatile("" : "+r"(sink));
                        q3 = clock64();
                        pl += q1 - q0;
                        pw += q2 - q1;
                        pc += q3 - q2;
                    }
                }
#elif PAYG > 1
                {
                    constexpr int NCH = (TileN / 2) / 16;
#pragma unroll
                    for (int i0 = 0; i0 < NCH; i0 += PAYG) {
                        unsigned vv[PAYG][16];
#pragma unroll
                        for (int g = 0; g < PAYG; ++g)
                            if (i0 + g < NCH)
                                ld16nw(local + TmemD + ds * TileN + col0 + 16 * (i0 + g), vv[g]);
                        waitld16(vv[0]);
#pragma unroll
                        for (int g = 1; g < PAYG; ++g)
                            if (i0 + g < NCH)
                                tie16(vv[g]);
#pragma unroll
                        for (int g = 0; g < PAYG; ++g)
                            if (i0 + g < NCH) {
#pragma unroll
                                for (int j = 0; j < 16; ++j) {
                                    const int x = FASTCVT ? f2i_int24(vv[g][j]) : int(__uint_as_float(vv[g][j]));
                                    ACCUM(x, 16 * (i0 + g) + j)
                                }
                            }
                    }
                }
#else
#pragma unroll
            for (int i = 0; i < (TileN / 2) / 16; ++i) {
                unsigned v[16];
                if (PAYT)
                    q0 = clock64();
                ld16nw(local + TmemD + ds * TileN + col0 + 16 * i, v);
                if (PAYT)
                    q1 = clock64();
                waitld16(v);
                if (RFREE && i == (TileN / 2) / 32 - 1) {
                    asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
                    if (t == 0)
                        arrive_local(sa(&rdone_half[ds]));
                }
                if (PAYT)
                    q2 = clock64();
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int x = FASTCVT ? f2i_int24(v[j]) : int(__uint_as_float(v[j]));
                    ACCUM(x, 16 * i + j)
                }
                if (PAYT) {
                    asm volatile("" : "+r"(sink));
                    q3 = clock64();
                    pl += q1 - q0;
                    pw += q2 - q1;
                    pc += q3 - q2;
                }
            }
#endif
#undef ACCUM
                if (PAYT)
                    q0 = clock64();
                asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::: "memory");
                if (D3 && t == 0)
                    arrive_local(sa(&rdone_local[ds]));
#if RDREL == 2
                asm volatile("tcgen05.fence::after_thread_sync;" ::
                                 : "memory"); // every read-out thread: its finished tcgen05.ld (waited above) ordered
                                              // after the barrier
                if (t == 0) {
                    const unsigned rb = peer0(sa(&read_done[ds]));
                    asm volatile(
                        "tcgen05.fence::before_thread_sync;mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::
                            "r"(rb)
                        : "memory");
                }
#else
                if (t == 0)
                    arrive_remote(peer0(sa(&read_done[ds])));
#endif
                if (PAYT)
                    pb += clock64() - q0;
#if C0A_PREF
                if (EXT103 && c0a.on && sg == (nsegx > 3 ? nsegx - 3 : 0)) {
                    const int p0 = __ldg(c0a.arow + am), p1 = __ldg(c0a.arow + am + 1);
                    for (int e = p0; e < p1; e += 4) {
                        int kq[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                            kq[i] = e + i < p1 ? (__ldg(c0a.aent + e + i) >> 8) : -1;
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                            if (kq[i] >= 0) {
                                const int8_t *pp = c0a.bt + size_t(kq[i]) * size_t(ColTiles * TileN) + an0;
                                asm volatile("prefetch.global.L2 [%0];" ::"l"(pp));
                                asm volatile("prefetch.global.L2 [%0];" ::"l"(pp + (TileN / 2 - 1)));
                            }
                    }
                }
#endif
#if C0A_RR && KSEG_SACC == 0
                if (EXT103 && c0a.on) {
                    for (int e = ea + sg; e < eaend; e += nsegx) {
                        const int ent = e == ea + sg ? entn : __ldg(c0a.aent + e), kk = ent >> 8,
                                  dx = ((ent & 255) - 128) > 0 ? 1 : -1;
                        const uint4 *yr =
                            reinterpret_cast<const uint4 *>(c0a.bt + size_t(kk) * size_t(ColTiles * TileN) + an0);
#pragma unroll
                        for (int ch = 0; ch < (TileN / 2) / 16; ++ch) {
                            const uint4 q4 = __ldg(yr + ch);
                            const unsigned w[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                            for (int j = 0; j < 16; ++j)
                                acc[16 * ch + j] += dx * sb8(w[j / 4], 0x8880u + (j % 4) * 0x1111u);
                        }
                    }
                    entn = ea + sg + 1 < eaend ? __ldg(c0a.aent + ea + sg + 1) : 0;
                } // the next unit's first entry, one unit ahead
#endif
#if C0A_PERSEG && KSEG_SACC == 0
                if (sg < nseg) {
                    const int kend = (sg + 1) * segc * 16;
                    while (ea < eaend) {
                        const int ent = __ldg(c0a.aent + ea), kk = ent >> 8;
                        if (kk >= kend)
                            break;
                        const int dx = ((ent & 255) - 128) > 0 ? 1 : -1;
                        const uint4 *yr =
                            reinterpret_cast<const uint4 *>(c0a.bt + size_t(kk) * size_t(ColTiles * TileN) + an0);
#pragma unroll
                        for (int ch = 0; ch < (TileN / 2) / 16; ++ch) {
                            const uint4 q4 = __ldg(yr + ch);
                            const unsigned w[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                            for (int j = 0; j < 16; ++j)
                                acc[16 * ch + j] += dx * sb8(w[j / 4], 0x8880u + (j % 4) * 0x1111u);
                        }
                        ++ea;
                    }
                }
#endif
            }
            ubase += nsegx;
#if KSEG_SACC == 0
            if (!C0A_PERSEG && !C0A_RR && EXT103 && c0a.on && carry) {
                const unsigned m = tr * TileM + rank * CtaRows + (t % 128), n0 = tc * TileN + col0;
                const int ea0 = __ldg(c0a.arow + m), ea1 = __ldg(c0a.arow + m + 1);
#if C0A_PREF
                int entc = ea0 < ea1 ? __ldg(c0a.aent + ea0) : 0;
                for (int e = ea0; e < ea1; ++e) {
                    const int ent = entc, kk = ent >> 8, dx = ((ent & 255) - 128) > 0 ? 1 : -1;
                    entc = e + 1 < ea1 ? __ldg(c0a.aent + e + 1) : 0;
#else
                for (int e = ea0; e < ea1; ++e) {
                    const int ent = __ldg(c0a.aent + e), kk = ent >> 8, dx = ((ent & 255) - 128) > 0 ? 1 : -1;
#endif
                    const uint4 *yr =
                        reinterpret_cast<const uint4 *>(c0a.bt + size_t(kk) * size_t(ColTiles * TileN) + n0);
#pragma unroll
                    for (int ch = 0; ch < (TileN / 2) / 16; ++ch) {
                        const uint4 q4 = __ldg(yr + ch);
                        const unsigned w[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                        for (int j = 0; j < 16; ++j)
                            acc[16 * ch + j] += dx * sb8(w[j / 4], 0x8880u + (j % 4) * 0x1111u);
                    }
                }
            }
#endif
#if F5_EXT == 2
            if (am < unsigned(f5c.m)) { // ENC5 stage 2: this row's A-side shift terms and leftovers into its 112 sums
                                        // (stage 3, the B leftovers: f5_fixb)
                const int p0 = __ldg(f5c.arow + am), p1 = __ldg(f5c.arow + am + 1);
#if F5_AGRP > 1
                const int pm = __ldg(f5c.arow1 + am);
                for (int e0 = p0; e0 < pm; e0 += F5_AGRP) {
                    const uint4 *bp[F5_AGRP];
                    int vv[F5_AGRP];
#pragma unroll
                    for (int i = 0; i < F5_AGRP; ++i) {
                        const bool ok = e0 + i < pm;
                        const int ent = ok ? __ldg(f5c.aent + e0 + i) : 0;
                        vv[i] = ok ? (ent & 255) - 128 : 0;
                        bp[i] = reinterpret_cast<const uint4 *>(
                            f5c.y2s + size_t(ok ? (ent >> 8) & 0x3fffff : 0) * size_t(f5c.np) + an0);
                    }
#pragma unroll
                    for (int ch = 0; ch < (TileN / 2) / 8; ++ch) {
                        uint4 q[F5_AGRP];
#pragma unroll
                        for (int i = 0; i < F5_AGRP; ++i)
                            q[i] = __ldg(bp[i] + ch);
#pragma unroll
                        for (int i = 0; i < F5_AGRP; ++i) {
                            const unsigned w[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
#pragma unroll
                            for (int j = 0; j < 8; ++j)
                                acc[8 * ch + j] += vv[i] * int((short)(w[j / 2] >> (16 * (j % 2))));
                        }
                    }
                }
                if (pm < p1) {
#pragma unroll
                    for (int ch = 0; ch < (TileN / 2) / 8; ++ch) {
                        for (int e = pm; e < p1; ++e) {
                            const int ent = __ldg(f5c.aent + e), v = (ent & 255) - 128, idx = (ent >> 8) & 0x3fffff;
                            const uint2 q2 = __ldg(
                                reinterpret_cast<const uint2 *>(f5c.y2 + size_t(idx) * size_t(f5c.np) + an0) + ch);
                            const unsigned w[2] = {q2.x, q2.y}; // d_A x y''[k]
#pragma unroll
                            for (int j = 0; j < 8; ++j)
                                acc[8 * ch + j] += v * int((signed char)(w[j / 4] >> (8 * (j % 4))));
                        }
                    }
                }
#else
                if (p0 < p1) { // 13:5x: chunk-major (8 columns at a time, the row's entries inside) keeps 8 sums and
                               // one load live; entry-major spilled 832 B into the read-out
#pragma unroll
                    for (int ch = 0; ch < (TileN / 2) / 8; ++ch) {
                        for (int e = p0; e < p1; ++e) {
                            const int ent = __ldg(f5c.aent + e), v = (ent & 255) - 128, idx = (ent >> 8) & 0x3fffff;
                            if ((ent >> 30) & 1) {
                                const uint4 q4 = __ldg(
                                    reinterpret_cast<const uint4 *>(f5c.y2s + size_t(idx) * size_t(f5c.np) + an0) + ch);
                                const unsigned w[4] = {q4.x, q4.y, q4.z, q4.w}; // s_b x the block sums of y''
#pragma unroll
                                for (int j = 0; j < 8; ++j)
                                    acc[8 * ch + j] += v * int((short)(w[j / 2] >> (16 * (j % 2))));
                            } else {
                                const uint2 q2 = __ldg(
                                    reinterpret_cast<const uint2 *>(f5c.y2 + size_t(idx) * size_t(f5c.np) + an0) + ch);
                                const unsigned w[2] = {q2.x, q2.y}; // d_A x y''[k]
#pragma unroll
                                for (int j = 0; j < 8; ++j)
                                    acc[8 * ch + j] += v * int((signed char)(w[j / 4] >> (8 * (j % 4))));
                            }
                        }
                    }
                }
#endif
            } // the B leftovers are added by f5_fixb after the kernel (13:4x: in the read-out they unrolled 112 loops
              // and spilled 1,244 B)
#endif
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
        if (TAIL_INZERO && TailS > 1 && part >= 0 && zep)
            while (ld_acq_gpu(&g_zdone) < zep * (unsigned long long)Grid) {
            }
        for (int j = 0; j < TileN / 2; ++j) {
            if (N % TileN == 0 || gn0 + j < unsigned(N)) {
                const int v = j < RC ? acc[j < RC ? j : 0] : sacc[256 * (j - RC)];
                if (TailS > 1 && part >= 0 && !TAIL_ST)
                    atomicAdd(cp + size_t(j) * M, v);
                else
                    cp[size_t(j) * M] = v;
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
            if (PAYT && rank == 0 && (t == 0 || t == 32)) {
                auto &z = mt[tile];
                if (t == 0) {
                    z.pl0 = pl;
                    z.pw0 = pw;
                    z.pc0 = pc;
                    z.pb0 = pb;
                } else {
                    z.pl1 = pl;
                    z.pw1 = pw;
                    z.pc1 = pc;
                    z.pb1 = pb;
                }
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

} // namespace v5k
#undef M
#undef N
#undef RowTiles
#undef ColTiles
#undef Tiles
#undef TilesPerPair
#undef TilesPerBand
#undef FullWaves
#undef FullTiles
#undef TailTiles
