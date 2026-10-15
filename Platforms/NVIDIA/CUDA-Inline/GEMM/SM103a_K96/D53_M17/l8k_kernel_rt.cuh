// Table-driven GEMM over paired CTAs with per-part drain, read_done, and store_done handshakes.
// Each part names its planes and products, including accumulator, scale slot, and first write.
// The ring geometry uses the active FTS, Slots, and cut width; groups are numbered across
// parts, so the ring phase is k * gt + gbase + g.
// B scale slots encode E4M3 2w over the base 2 x 2. Payout warps rewrite weights after
// reading accumulators and before read_done; the issuer waits before the next part's MMA.
// Payout forms t = D0 + gamma D1 + bias and applies Barrett reduction. D1 is omitted for
// a single-accumulator part. The residue is lambda0 R; its inverse is folded into CRT weights.
// The host checks each accumulator's bound sum w max|plane A| max|plane B| K <= 2^24 - 1,
// 0 <= t + bias < 2^32, and that groups contain a multiple of BoxTiles plane tiles.

// Runtime kernel geometry and reconstruction.
#ifndef L8_PAY_N
#define L8_PAY_N 1
#endif
struct L8Shape {
    int m, n, cuts, tile_rows, tile_cols, tiles, clusters,
        nchunk; // nchunk (was pad) = modulus ranges per tile
    int whole; // the leading work items that are whole tiles (the rest split into nchunk ranges; unused when nchunk = 1)
    long long ldc; // doubles between columns of C (>= m)
    unsigned long long
        plane;  // residues per modulus plane = tiles * Cols * Side (tile-major, padding included)
    int pace_s; // < 0 off; S = the parts a cluster may run ahead of the slowest
    unsigned *pace;          // per cluster, parts started + 1 (0xffffffff when done)
    unsigned *pace_timeouts; // waits that gave up
    int item_order;          // 0 = the split items tile-major, 1 = chunk-major
    int pingpong;            // 0 = the payout before the next part; 1 = behind it
    int crt_pf, crt_prop,
        crt2; // CRT prefetch, proportional column assignment, and two-group reconstruction.
};
__device__ __forceinline__ int l8_ntiles(const L8Shape &sh, unsigned q) {
    return q < unsigned(sh.tiles)
               ? int((unsigned(sh.tiles) - q + unsigned(sh.clusters) - 1) / unsigned(sh.clusters))
               : 0;
}
__device__ __forceinline__ int l8_nwork(const L8Shape &sh, unsigned q,
                                        int nc) { // work items of cluster q
    const unsigned items = nc > 1
                               ? unsigned(sh.whole) + unsigned(sh.tiles - sh.whole) * unsigned(nc)
                               : unsigned(sh.tiles);
    return q < items ? int((items - q + unsigned(sh.clusters) - 1) / unsigned(sh.clusters)) : 0;
}
__device__ __forceinline__ unsigned l8_tile_row(const L8Shape &sh, unsigned tile) {
    const unsigned band = tile / (8u * sh.tile_cols), within = tile - band * 8u * sh.tile_cols;
    const unsigned rb = min(8u, unsigned(sh.tile_rows) - band * 8u);
    return band * 8u + within % rb;
}
__device__ __forceinline__ unsigned l8_tile_col(const L8Shape &sh, unsigned tile) {
    const unsigned band = tile / (8u * sh.tile_cols), within = tile - band * 8u * sh.tile_cols;
    const unsigned rb = min(8u, unsigned(sh.tile_rows) - band * 8u);
    return within / rb;
}
// Reconstruct one element from its residues: u = sum r_q W_q mod P.
// Use four-limb sums, or crt2_reduce when sh.crt2 is set; both use the same final rounding.
__device__ __forceinline__ double crt_elem_rt(const unsigned (&r)[kParts], f64_impl::U128 P,
                                              int exponent, const L8Shape &sh) {
    f64_impl::U128 u;
#if L8_CRT2
    if (CRT2_ON(sh)) {
        u = crt2_reduce(r);
    } else
#endif
    {
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
        u = lo - f64_impl::U128(__umul64hi(v3, crt_pt_magic)) * P;
        if (u >= P) {
            u -= P;
        }
        if (u >= P) {
            u -= P;
        }
    }
    const bool neg = u > P / 2;
    return round_scaled_int(neg ? P - u : u, neg, exponent);
}
// CRT for columns [c0, c1) of this thread's row of a tile.
// Columns at or past n are the tail of [c0, c1); sh.crt_pf prefetches the next
// column's residues before this column's arithmetic in crt_elem_rt.
__device__ __forceinline__ void crt_cols_rt(const unsigned short *resid, double *cout,
                                            const int *ea, const int *eb, f64_impl::U128 P,
                                            unsigned tm_row, unsigned tn, unsigned rank, unsigned t,
                                            int c0, int c1, const L8Shape &sh, unsigned tile_id) {
    const unsigned row = tm_row * Side + rank * (Side / 2) + (t % 128);
    if (row >= unsigned(sh.m)) {
        return;
    }
    const int er = -ea[row];
    const unsigned n0 = tn * Cols + (t / 128) * 112;
    int cend = c1;
    if (int(n0) + cend > sh.n) {
        cend = sh.n - int(n0);
    }
    if (c0 >= cend) {
        return;
    }
    const ResT *base = reinterpret_cast<const ResT *>(resid) +
                       (size_t(tile_id) * Cols + (t / 128) * 112) * Side + rank * (Side / 2) +
                       (t % 128);
    unsigned r[kParts];
    if (CRT_PF_ON(sh)) {
        {
            const ResT *rp = base + size_t(c0) * Side;
#pragma unroll
            for (int q = 0; q < kParts; ++q) {
                r[q] = __ldcg(rp);
                rp += sh.plane;
            }
        }
        for (int c = c0; c < cend; ++c) {
            unsigned rn[kParts];
#pragma unroll
            for (int q = 0; q < kParts; ++q) {
                rn[q] = 0u;
            }
            if (c + 1 < cend) {
                const ResT *rp = base + size_t(c + 1) * Side;
#pragma unroll
                for (int q = 0; q < kParts; ++q) {
                    rn[q] = __ldcg(rp);
                    rp += sh.plane;
                }
            }
            const unsigned global_n = n0 + unsigned(c);
            cout[size_t(global_n) * sh.ldc + row] = crt_elem_rt(r, P, er - eb[global_n], sh);
#pragma unroll
            for (int q = 0; q < kParts; ++q) {
                r[q] = rn[q];
            }
        }
    } else {
        for (int c = c0; c < cend; ++c) {
            const ResT *rp = base + size_t(c) * Side;
#pragma unroll
            for (int q = 0; q < kParts; ++q) {
                r[q] = __ldcg(rp);
                rp += sh.plane;
            }
            const unsigned global_n = n0 + unsigned(c);
            cout[size_t(global_n) * sh.ldc + row] = crt_elem_rt(r, P, er - eb[global_n], sh);
        }
    }
}

template <bool Unused>
__global__ __launch_bounds__(384, 1) void l8k_gemm_kernel_rt(
    const unsigned char *A, const unsigned char *B, unsigned short *resid, const L8kMod *mods,
    unsigned block_tiles, KernelMetrics *km, Metrics *measures, f64_impl::U128 *crt_state,
    const f64_impl::U128 *crtw, double *cout, const int *ea, const int *eb, f64_impl::U128 crt_p,
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ L8Shape sh) {
    (void)crt_state;
    (void)crtw;
    const unsigned q = blockIdx.x / 2;
    const int nc = sh.nchunk > 1 ? sh.nchunk : 1;
    const int ntiles = l8_nwork(sh, q, nc), cuts = sh.cuts; // work items (= tiles when nc = 1)
    const unsigned F = nc > 1 ? unsigned(sh.whole) : 0xffffffffu; // items w < F are whole tiles
    const unsigned NS =
        unsigned(sh.tiles) - (F < unsigned(sh.tiles) ? F : unsigned(sh.tiles)); // the split tiles
    auto chunk_of = [&](unsigned w) {
        return sh.item_order ? (w - F) / NS : (w - F) % unsigned(nc);
    };
    auto tile_of = [&](unsigned w) {
        return w < F ? w : (sh.item_order ? F + (w - F) % NS : F + (w - F) / unsigned(nc));
    };
    auto p0_of = [&](unsigned w) { return w < F ? 0 : int(chunk_of(w)) * kParts / nc; };
    auto p1_of = [&](unsigned w) { return w < F ? kParts : (int(chunk_of(w)) + 1) * kParts / nc; };
    const auto tile_begin = clock64();
    auto cluster = cooperative_groups::this_cluster();
    unsigned rank = cluster.block_rank(), t = threadIdx.x;
    __shared__ unsigned tmbase;
    __shared__ int crt_cum[kParts + 1]; // the prefix of the products per part (sh.crt_prop)
    __shared__ __align__(8) unsigned long long afull[kSlots], bfull[kSlots], empty[kSlots], drain,
        read_done, store_done, slots_ready, place_free0,
        place_free1; // the next modulus's scale slots written; TMEM place 0 / 1 read
    extern __shared__ __align__(1024) unsigned char sm[];
    if (t == 0) {
#pragma unroll
        for (int s = 0; s < kSlots; ++s) {
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0],1;mbarrier.init.shared::cta.b64 [%1],1;"
                "mbarrier.init.shared::cta.b64 [%2],1;" ::"r"(sa(&afull[s])),
                "r"(sa(&bfull[s])), "r"(sa(&empty[s]))
                : "memory");
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0],1;mbarrier.init.shared::cta.b64 [%1],2;"
                     "mbarrier.init.shared::cta.b64 [%2],2;mbarrier.init.shared::cta.b64 [%3],2;"
                     "mbarrier.init.shared::cta.b64 [%4],2;mbarrier.init.shared::cta.b64 [%5],2;"
                     "fence.mbarrier_init.release.cluster;" ::"r"(sa(&drain)),
                     "r"(sa(&read_done)), "r"(sa(&store_done)), "r"(sa(&slots_ready)),
                     "r"(sa(&place_free0)), "r"(sa(&place_free1))
                     : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    cluster.sync();
    if (t < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],512;" ::"r"(sa(&tmbase))
            : "memory");
    }
    __syncthreads();
    cluster.sync();
    unsigned tm = tmbase, local = tm + ((t / 32 % 4 * 32) << 16);
    if (t == 128) { // counted once; the __syncthreads below publishes it
        int a = 0;
        for (int p = 0; p < kParts; ++p) {
            crt_cum[p] = a;
            a += mods[p].nprod;
        }
        crt_cum[kParts] = a;
    }
    // Initialize four 16-column scale slots starting at TMEM column 448.
    // Slot 0 holds weight 1; slots 1..3 hold the first assigned part's weights.
    if (t < 128) {
        const int f0 = ntiles > 0 ? p0_of(q) : 0; // the first modulus of this cluster's first item
        const unsigned code[4] = {0x40u, mods[f0].code[0], mods[f0].code[1], mods[f0].code[2]};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned bits = code[s] * 0x01010101u;
            for (int c = 0; c < 16; ++c) {
                st(local + 448 + 16 * s + c, bits);
            }
        }
        asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::
                         : "memory");
    }
    __syncthreads();
    cluster.sync();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (t == 288 ||
        t == 320) { // supply (A side / B side); both CTAs, completion on rank 0's barrier
        const bool side_b = t == 320;
        int gcnt = 0; // the running group count (= k * gt + gbase with whole tiles)
        unsigned pcnt = 0,
                 pace_seen =
                     0; // the parts this supply has started; the smallest pace entry it has seen
        for (int k = 0; k < ntiles; ++k) {
            const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);
            const size_t block = size_t(side_b ? l8_tile_col(sh, tile) : l8_tile_row(sh, tile)) *
                                 block_tiles; // in plane tiles
            for (int part = p0_of(w); part < p1_of(w); ++part) {
                const int F = mods[part].planes, cpg = mods[part].cpg, groups = mods[part].groups;
                const int gb = gcnt, pbase = mods[part].pbase;
                gcnt += groups;
                if (!side_b && sh.pace_s >= 0) { // the pace (see L8Shape)
                    if (rank == 0) {
                        *(volatile unsigned *)(sh.pace + q) = pcnt + 1u;
                    }
                    if (pcnt >= unsigned(sh.pace_s)) {
                        const unsigned need = pcnt - unsigned(sh.pace_s) + 1u,
                                       nclu = unsigned(sh.clusters);
                        if (pace_seen < need) {
                            int it = 0;
                            for (; it < 20000; ++it) {
                                unsigned mn = 0xffffffffu;
                                for (unsigned c = 0; c < nclu; ++c) {
                                    mn = min(mn, *(const volatile unsigned *)(sh.pace + c));
                                }
                                pace_seen = mn;
                                if (mn >= need) {
                                    break;
                                }
                                __nanosleep(256);
                            }
                            if (it == 20000) {
                                atomicAdd(sh.pace_timeouts, 1u);
                            }
                        }
                    }
                }
                ++pcnt;
                for (int g = 0; g < groups; ++g) {
                    const int G = gb + g, slot = G % kSlots;
                    if (G >= kSlots) {
                        await(sa(&empty[slot]), ((G / kSlots) + 1) & 1);
                    }
                    const int cut0 = g * cpg, nc = min(cpg, cuts - cut0);
                    const unsigned ntile = unsigned(nc * F), bytes = ntile * TileBytes;
                    // l8e::plane_offset: tile block, then 2 x cuts x pbase tiles, then ((rank x cuts + cut) x F + plane)
#if L8_FOLD_TILES
                    // Diagnostic loads fold into the first L8_FOLD_TILES planes of each operand.
                    // The output is not a matrix product.
                    const size_t first =
                        (block + size_t(2) * cuts * pbase + (size_t(rank) * cuts + cut0) * F) %
                        L8_FOLD_TILES;
#else
                    const size_t first =
                        block + size_t(2) * cuts * pbase + (size_t(rank) * cuts + cut0) * F;
#endif
                    const unsigned dest = sa(sm + slot * kGroup + (side_b ? kOperand : 0));
                    const unsigned lbar_local = sa(side_b ? &bfull[slot] : &afull[slot]);
                    unsigned lbar;
                    asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                 : "=r"(lbar)
                                 : "r"(lbar_local));
                    if (rank == 0) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ::
                                "r"(lbar_local),
                            "r"(2u * bytes)
                            : "memory");
                    }
                    for (unsigned b = 0; b < ntile; b += kBoxTiles) {
                        asm volatile(
                            "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                            "[%0],[%1,{%2,%3}],[%4];" ::"r"(dest + b * TileBytes),
                            "l"(reinterpret_cast<unsigned long long>(side_b ? &tmB : &tmA)), "r"(0),
                            "r"(int((first + b) * (TileBytes / 256))), "r"(lbar)
                            : "memory");
                    }
                }
            }
        }
        if (!side_b && sh.pace_s >= 0 && rank == 0) {
            *(volatile unsigned *)(sh.pace + q) = 0xffffffffu; // done, never wait on me
        }
    } else if (t == 256 && rank == 0) { // issue
        unsigned long long abase[kSlots], bbase[kSlots];
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const unsigned addr = sa(sm + slot * kGroup);
            abase[slot] = tile_desc(addr);
            bbase[slot] = tile_desc(addr + kOperand);
        }
        int gcnt = 0, scnt = 0; // the running group and step counts
        int sel = 0, u0 = 0, u1 = 0, w0 = 0,
            w1 = 0; // the next place; uses of place 0 / 1 issued / freed seen
        for (int k = 0; k < ntiles; ++k) {
            const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);
            for (int part = p0_of(w); part < p1_of(w); ++part) {
                const int step = scnt++;
                const int F = mods[part].planes, cpg = mods[part].cpg, np = mods[part].nprod;
                const int first = gcnt, last = first + mods[part].groups;
                gcnt = last;
                unsigned pr[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pr[i] = mods[part].prod[i];
                }
                unsigned prx[6]; // products 5..10
#pragma unroll
                for (int i = 0; i < 6; ++i) {
                    prx[i] = np > 4 + i ? mods[part].prod[4 + i] : 0u;
                }
                const unsigned sf0 = tm + 448;
                int accb = 0; // the TMEM place of a one-accumulator part (0 unless pingpong)
                if (sh.pingpong) { // Wait for the previous payout before reusing either selected TMEM place.
                    const int na = mods[part].nacc;
                    accb = na > 1 ? 0 : sel;
                    const bool p0 = na > 1 || sel == 0, p1 = na > 1 || sel == 1;
                    if (p0 && u0 >= 1 && w0 < u0) {
                        await(sa(&place_free0), unsigned(u0 - 1) & 1u);
                        w0 = u0;
                    }
                    if (p1 && u1 >= 1 && w1 < u1) {
                        await(sa(&place_free1), unsigned(u1 - 1) & 1u);
                        w1 = u1;
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    u0 += p0 ? 1 : 0;
                    u1 += p1 ? 1 : 0;
                    if (na == 1) {
                        sel ^= 1;
                    }
                }
                unsigned long long aw = 0, bw = 0, issue = 0, ns0;
                auto core_begin = clock64();
                asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns0));
                measures[(size_t(tile) * kParts + part) * 2 + 1].ns =
                    ns0;                 // the part's issue start in rank 1's unused ns
                if (np == 4 && F == 2) { // 4 products on 2 faces
                    for (int round = first / kSlots; round <= (last - 1) / kSlots; ++round) {
#pragma unroll
                        for (int slot = 0; slot < kSlots; ++slot) {
                            const int G = round * kSlots + slot;
                            if (G >= first && G < last) {
                                const int g = G - first, nc = min(cpg, cuts - g * cpg);
                                auto w = clock64();
                                await(sa(&afull[slot]), round & 1);
                                aw += clock64() - w;
                                w = clock64();
                                await(sa(&bfull[slot]), round & 1);
                                bw += clock64() - w;
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                                auto i0 = clock64();
                                if (nc == 3) { // a full group, the cut loop unrolled
#pragma unroll
                                    for (int c = 0; c < 3; ++c) {
#pragma unroll
                                        for (int i = 0; i < 4; ++i) {
                                            if (i < 4) {
                                                const unsigned e = pr[i];
                                                const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                               acc = (e >> 4) & 1u,
                                                               bs = ((e >> 5) & 3u) |
                                                                    (((e >> 8) & 1u) << 2),
                                                               as = (e >> 9) & 7u;
                                                const bool overwrite =
                                                    ((e >> 7) & 1u) && g == 0 && c == 0;
                                                const unsigned long long a =
                                                    abase[slot] +
                                                    (unsigned long long)((c * 2 + int(fa)) *
                                                                         (TileBytes >> 4));
                                                const unsigned long long b =
                                                    bbase[slot] +
                                                    (unsigned long long)((c * 2 + int(fb)) *
                                                                         (TileBytes >> 4));
                                                face_mma<false, 0>(
                                                    tm + (unsigned(accb) + acc) * Cols, a, b,
                                                    sf0 + 16 * as, sf0 + 16 * bs, !overwrite);
                                                /* 16-column slots for A and B. */
                                            }
                                        }
                                    }
                                } else {
                                    for (int c = 0; c < nc; ++c) {
#pragma unroll
                                        for (int i = 0; i < 4; ++i) {
                                            if (i < 4) {
                                                const unsigned e = pr[i];
                                                const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                               acc = (e >> 4) & 1u,
                                                               bs = ((e >> 5) & 3u) |
                                                                    (((e >> 8) & 1u) << 2),
                                                               as = (e >> 9) & 7u;
                                                const bool overwrite =
                                                    ((e >> 7) & 1u) && g == 0 && c == 0;
                                                const unsigned long long a =
                                                    abase[slot] +
                                                    (unsigned long long)((c * 2 + int(fa)) *
                                                                         (TileBytes >> 4));
                                                const unsigned long long b =
                                                    bbase[slot] +
                                                    (unsigned long long)((c * 2 + int(fb)) *
                                                                         (TileBytes >> 4));
                                                face_mma<false, 0>(
                                                    tm + (unsigned(accb) + acc) * Cols, a, b,
                                                    sf0 + 16 * as, sf0 + 16 * bs, !overwrite);
                                                /* 16-column slots for A and B. */
                                            }
                                        }
                                    }
                                }
                                issue += clock64() - i0;
                                done(sa(&empty[slot]));
                            }
                        }
                    }
                } else if (np == 3 && F == 2) { // 3 products on 2 faces
                    for (int round = first / kSlots; round <= (last - 1) / kSlots; ++round) {
#pragma unroll
                        for (int slot = 0; slot < kSlots; ++slot) {
                            const int G = round * kSlots + slot;
                            if (G >= first && G < last) {
                                const int g = G - first, nc = min(cpg, cuts - g * cpg);
                                auto w = clock64();
                                await(sa(&afull[slot]), round & 1);
                                aw += clock64() - w;
                                w = clock64();
                                await(sa(&bfull[slot]), round & 1);
                                bw += clock64() - w;
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                                auto i0 = clock64();
                                if (nc == 3) { // a full group, the cut loop unrolled
#pragma unroll
                                    for (int c = 0; c < 3; ++c) {
#pragma unroll
                                        for (int i = 0; i < 4; ++i) {
                                            if (i < 3) {
                                                const unsigned e = pr[i];
                                                const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                               acc = (e >> 4) & 1u,
                                                               bs = ((e >> 5) & 3u) |
                                                                    (((e >> 8) & 1u) << 2),
                                                               as = (e >> 9) & 7u;
                                                const bool overwrite =
                                                    ((e >> 7) & 1u) && g == 0 && c == 0;
                                                const unsigned long long a =
                                                    abase[slot] +
                                                    (unsigned long long)((c * 2 + int(fa)) *
                                                                         (TileBytes >> 4));
                                                const unsigned long long b =
                                                    bbase[slot] +
                                                    (unsigned long long)((c * 2 + int(fb)) *
                                                                         (TileBytes >> 4));
                                                face_mma<false, 0>(
                                                    tm + (unsigned(accb) + acc) * Cols, a, b,
                                                    sf0 + 16 * as, sf0 + 16 * bs, !overwrite);
                                                /* 16-column slots for A and B. */
                                            }
                                        }
                                    }
                                } else {
                                    for (int c = 0; c < nc; ++c) {
#pragma unroll
                                        for (int i = 0; i < 4; ++i) {
                                            if (i < 3) {
                                                const unsigned e = pr[i];
                                                const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                               acc = (e >> 4) & 1u,
                                                               bs = ((e >> 5) & 3u) |
                                                                    (((e >> 8) & 1u) << 2),
                                                               as = (e >> 9) & 7u;
                                                const bool overwrite =
                                                    ((e >> 7) & 1u) && g == 0 && c == 0;
                                                const unsigned long long a =
                                                    abase[slot] +
                                                    (unsigned long long)((c * 2 + int(fa)) *
                                                                         (TileBytes >> 4));
                                                const unsigned long long b =
                                                    bbase[slot] +
                                                    (unsigned long long)((c * 2 + int(fb)) *
                                                                         (TileBytes >> 4));
                                                face_mma<false, 0>(
                                                    tm + (unsigned(accb) + acc) * Cols, a, b,
                                                    sf0 + 16 * as, sf0 + 16 * bs, !overwrite);
                                                /* 16-column slots for A and B. */
                                            }
                                        }
                                    }
                                }
                                issue += clock64() - i0;
                                done(sa(&empty[slot]));
                            }
                        }
                    }
                } else if (np <= 4) { // the original round loop for <= 4 products
                    for (int round = first / kSlots; round <= (last - 1) / kSlots; ++round) {
#pragma unroll
                        for (int slot = 0; slot < kSlots; ++slot) {
                            const int G = round * kSlots + slot;
                            if (G >= first && G < last) {
                                const int g = G - first, nc = min(cpg, cuts - g * cpg);
                                auto w = clock64();
                                await(sa(&afull[slot]), round & 1);
                                aw += clock64() - w;
                                w = clock64();
                                await(sa(&bfull[slot]), round & 1);
                                bw += clock64() - w;
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                                auto i0 = clock64();
                                for (int c = 0; c < nc; ++c) {
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        if (i < np) {
                                            const unsigned e = pr[i];
                                            const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                           acc = (e >> 4) & 1u,
                                                           bs = ((e >> 5) & 3u) |
                                                                (((e >> 8) & 1u) << 2),
                                                           as = (e >> 9) & 7u;
                                            const bool overwrite =
                                                ((e >> 7) & 1u) && g == 0 && c == 0;
                                            const unsigned long long a =
                                                abase[slot] +
                                                (unsigned long long)((c * F + int(fa)) *
                                                                     (TileBytes >> 4));
                                            const unsigned long long b =
                                                bbase[slot] +
                                                (unsigned long long)((c * F + int(fb)) *
                                                                     (TileBytes >> 4));
                                            face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols,
                                                               a, b, sf0 + 16 * as, sf0 + 16 * bs,
                                                               !overwrite);
                                            /* 16-column slots for A and B. */
                                        }
                                    }
                                }
                                issue += clock64() - i0;
                                done(sa(&empty[slot]));
                            }
                        }
                    }
                } else { // 5..10 products
                    for (int round = first / kSlots; round <= (last - 1) / kSlots; ++round) {
#pragma unroll
                        for (int slot = 0; slot < kSlots; ++slot) {
                            const int G = round * kSlots + slot;
                            if (G >= first && G < last) {
                                const int g = G - first, nc = min(cpg, cuts - g * cpg);
                                auto w = clock64();
                                await(sa(&afull[slot]), round & 1);
                                aw += clock64() - w;
                                w = clock64();
                                await(sa(&bfull[slot]), round & 1);
                                bw += clock64() - w;
                                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                                auto i0 = clock64();
                                for (int c = 0; c < nc; ++c) {
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        if (i < np) {
                                            const unsigned e = pr[i];
                                            const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                           acc = (e >> 4) & 1u,
                                                           bs = ((e >> 5) & 3u) |
                                                                (((e >> 8) & 1u) << 2),
                                                           as = (e >> 9) & 7u;
                                            const bool overwrite =
                                                ((e >> 7) & 1u) && g == 0 && c == 0;
                                            const unsigned long long a =
                                                abase[slot] +
                                                (unsigned long long)((c * F + int(fa)) *
                                                                     (TileBytes >> 4));
                                            const unsigned long long b =
                                                bbase[slot] +
                                                (unsigned long long)((c * F + int(fb)) *
                                                                     (TileBytes >> 4));
                                            face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols,
                                                               a, b, sf0 + 16 * as, sf0 + 16 * bs,
                                                               !overwrite);
                                            /* 16-column slots for A and B. */
                                        }
                                    }
                                    if (np > 4) { // product 5

                                        const unsigned e = prx[0];
                                        const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                       acc = (e >> 4) & 1u,
                                                       bs =
                                                           ((e >> 5) & 3u) | (((e >> 8) & 1u) << 2),
                                                       as = (e >> 9) & 7u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0 + 16 * as, sf0 + 16 * bs,
                                                           !overwrite);
                                        /* 16-column slots for A and B. */
                                    }
                                    if (np > 5) { // product 6

                                        const unsigned e = prx[1];
                                        const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                       acc = (e >> 4) & 1u,
                                                       bs =
                                                           ((e >> 5) & 3u) | (((e >> 8) & 1u) << 2),
                                                       as = (e >> 9) & 7u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0 + 16 * as, sf0 + 16 * bs,
                                                           !overwrite);
                                        /* 16-column slots for A and B. */
                                    }
                                    if (np > 6) { // product 7

                                        const unsigned e = prx[2];
                                        const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                       acc = (e >> 4) & 1u,
                                                       bs =
                                                           ((e >> 5) & 3u) | (((e >> 8) & 1u) << 2),
                                                       as = (e >> 9) & 7u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0 + 16 * as, sf0 + 16 * bs,
                                                           !overwrite);
                                        /* 16-column slots for A and B. */
                                    }
                                    if (np > 7) { // product 8

                                        const unsigned e = prx[3];
                                        const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                       acc = (e >> 4) & 1u,
                                                       bs =
                                                           ((e >> 5) & 3u) | (((e >> 8) & 1u) << 2),
                                                       as = (e >> 9) & 7u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0 + 16 * as, sf0 + 16 * bs,
                                                           !overwrite);
                                        /* 16-column slots for A and B. */
                                    }
                                    if (np > 8) { // product 9

                                        const unsigned e = prx[4];
                                        const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                       acc = (e >> 4) & 1u,
                                                       bs =
                                                           ((e >> 5) & 3u) | (((e >> 8) & 1u) << 2),
                                                       as = (e >> 9) & 7u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0 + 16 * as, sf0 + 16 * bs,
                                                           !overwrite);
                                        /* 16-column slots for A and B. */
                                    }
                                    if (np > 9) { // product 10

                                        const unsigned e = prx[5];
                                        const unsigned fa = e & 3u, fb = (e >> 2) & 3u,
                                                       acc = (e >> 4) & 1u,
                                                       bs =
                                                           ((e >> 5) & 3u) | (((e >> 8) & 1u) << 2),
                                                       as = (e >> 9) & 7u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0 + 16 * as, sf0 + 16 * bs,
                                                           !overwrite);
                                        /* 16-column slots for A and B. */
                                    }
                                }
                                issue += clock64() - i0;
                                done(sa(&empty[slot]));
                            }
                        }
                    }
                }
                done(sa(&drain));
                auto commit_after = clock64();
                await(sa(&drain), step & 1);
                auto end = clock64();
                unsigned long long ns1;
                asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns1));
                auto &z = measures[(size_t(tile) * kParts + part) * 2];
                z.core = end - core_begin;
                z.ns = ns1 - ns0;
                z.a_wait = aw;
                z.b_wait = bw;
                z.commit_wait = end - commit_after;
                z.issue = issue;
                if (sh.pingpong) { // only the scale slots; read_done is waited per place before a part
                    await(sa(&slots_ready), step & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                } else {
                    await(sa(&read_done), step & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    await(sa(&store_done), step & 1);
                }
            }
        }
    } else if (t < 256) { // payout
        long long prev = tile_begin;
        int scnt = 0; // the running step count
        int psel = 0; // the payout's copy of the issuer's sel
        for (int k = 0; k < ntiles; ++k) {
            const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);
            const unsigned tm_row = l8_tile_row(sh, tile), tn = l8_tile_col(sh, tile);
            const int pend = p1_of(w), pnext = k + 1 < ntiles ? p0_of(w + sh.clusters) : 0;
            if (k > 0 && w >= F &&
                w - sh.clusters <
                    F) { // the first split item after whole tiles: the last whole tile's CRT first
                const unsigned pt = w - sh.clusters;
                crt_cols_rt(resid, cout, ea, eb, crt_p, l8_tile_row(sh, pt), l8_tile_col(sh, pt),
                            rank, t, 0, 112, sh, pt);
            }
            for (int part = p0_of(w); part < pend; ++part) {
                const int step = scnt++;
                await(sa(&drain), step & 1);
                auto began = clock64();
                const int p = mods[part].p, gamma = mods[part].gamma, bias = mods[part].bias,
                          nacc = mods[part].nacc;
                const unsigned mmagic = mods[part].mmagic;
                ResT *const out = reinterpret_cast<ResT *>(resid) + size_t(part) * sh.plane;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                int paccb = 0; // this part's TMEM place (the issuer's accb)
                if (sh.pingpong) {
                    if (nacc == 1) {
                        paccb = psel;
                        psel ^= 1;
                    }
                    // the next modulus's weights first: this part's MMAs are complete (drain), so the slots are free; then the issuer may
                    // start the next part on the other place while this payout reads this one
                    if (t < 128) {
                        const int nx = part + 1 < pend ? part + 1 : pnext;
                        const unsigned c1 = mods[nx].code[0], c2 = mods[nx].code[1],
                                       c3 = mods[nx].code[2];
                        const unsigned b1 = c1 * 0x01010101u, b2 = c2 * 0x01010101u,
                                       b3 = c3 * 0x01010101u;
                        for (int c = 0; c < 16; ++c) {
                            st(local + 448 + 16 + c, b1);
                            st(local + 448 + 32 + c, b2);
                            st(local + 448 + 48 + c, b3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::
                                     : "memory");
                    if (t ==
                        0) { // the relaxed representative arrive (RREL 2) in place of arrive.release.cluster
                        unsigned remote;
                        asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                     : "=r"(remote)
                                     : "r"(sa(&slots_ready)));
                        asm volatile(
                            "tcgen05.fence::after_thread_sync;tcgen05.fence::before_thread_sync;" ::
                                : "memory");
                        asm volatile(
                            "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::"r"(
                                remote)
                            : "memory");
                    }
                }
#if L8_PAY_N // one-accumulator moduli neither load nor convert D1
                if (nacc > 1) {
#pragma unroll
                    for (int chunk = 0; chunk < 7; ++chunk) {
                        unsigned v0[16], v1[16];
                        const unsigned col = (t / 128) * 112 + chunk * 16;
                        load16(local + unsigned(paccb) * Cols + col,
                               v0); // paccb (0 for two accumulators)
                        if (nacc > 1) {
                            load16(local + Cols + col, v1);
                        } else {
#pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                v1[j] = 0u;
                            }
                        }
                        // the whole tile, padding included: residues without a condition, one register, immediates j * Side
                        ResT *const tp = out + (size_t(tile) * Cols + col) * Side +
                                         rank * (Side / 2) + (t % 128);
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const int x0 = __float2int_rn(__uint_as_float(v0[j]));
                            const int x1 = __float2int_rn(__uint_as_float(v1[j]));
                            // D0 + gamma D1 made positive by bias (a multiple of p), then Barrett; the host checked 0 <= tt < 2^32.
                            const unsigned tt =
                                (unsigned)((long long)x0 + (long long)gamma * x1 + (long long)bias);
                            const unsigned qd = (unsigned)(((unsigned long long)tt * mmagic) >> 32);
                            unsigned r = tt - qd * (unsigned)p;
                            if (r >= (unsigned)p) {
                                r -= (unsigned)p;
                            }
                            tp[j * Side] = (ResT)r;
                        }
                    }
                } else {
#pragma unroll
                    for (int chunk = 0; chunk < 7; ++chunk) {
                        unsigned v0[16];
                        const unsigned col = (t / 128) * 112 + chunk * 16;
                        load16(local + unsigned(paccb) * Cols + col,
                               v0); // paccb (0 for two accumulators)
                        // the whole tile, padding included: residues without a condition, one register, immediates j * Side
                        ResT *const tp = out + (size_t(tile) * Cols + col) * Side +
                                         rank * (Side / 2) + (t % 128);
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const int x0 = __float2int_rn(__uint_as_float(v0[j]));
                            // D0 + gamma D1 made positive by bias (a multiple of p), then Barrett; the host checked 0 <= tt < 2^32.
                            const unsigned tt =
                                (unsigned)((long long)x0 +
                                           (long long)bias); // one accumulator: gamma = 0, no D1
                            const unsigned qd = (unsigned)(((unsigned long long)tt * mmagic) >> 32);
                            unsigned r = tt - qd * (unsigned)p;
                            if (r >= (unsigned)p) {
                                r -= (unsigned)p;
                            }
                            tp[j * Side] = (ResT)r;
                        }
                    }
                }
#else
#pragma unroll
                for (int chunk = 0; chunk < 7; ++chunk) {
                    unsigned v0[16], v1[16];
                    const unsigned col = (t / 128) * 112 + chunk * 16;
                    load16(local + unsigned(paccb) * Cols + col,
                           v0); // paccb (0 for two accumulators)
                    if (nacc > 1) {
                        load16(local + Cols + col, v1);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            v1[j] = 0u;
                        }
                    }
                    // the whole tile, padding included: residues without a condition, one register, immediates j * Side
                    ResT *const tp =
                        out + (size_t(tile) * Cols + col) * Side + rank * (Side / 2) + (t % 128);
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int x0 = __float2int_rn(__uint_as_float(v0[j]));
                        const int x1 = __float2int_rn(__uint_as_float(v1[j]));
                        // D0 + gamma D1 made positive by bias (a multiple of p), then Barrett; the host checked 0 <= tt < 2^32.
                        const unsigned tt =
                            (unsigned)((long long)x0 + (long long)gamma * x1 + (long long)bias);
                        const unsigned qd = (unsigned)(((unsigned long long)tt * mmagic) >> 32);
                        unsigned r = tt - qd * (unsigned)p;
                        if (r >= (unsigned)p) {
                            r -= (unsigned)p;
                        }
                        tp[j * Side] = (ResT)r;
                    }
                }
#endif
                // the next modulus's weights into scale slots 1..3 (this CTA's TMEM), before the accumulators are handed back
                if (t < 128 && !sh.pingpong) { // pingpong 1 wrote them right after drain
                    const int nx =
                        part + 1 < pend ? part + 1 : pnext; // the next modulus this cluster runs
                    const unsigned c1 = mods[nx].code[0], c2 = mods[nx].code[1],
                                   c3 = mods[nx].code[2];
                    const unsigned b1 = c1 * 0x01010101u, b2 = c2 * 0x01010101u,
                                   b3 = c3 * 0x01010101u;
                    for (int c = 0; c < 16; ++c) {
                        st(local + 448 + 16 + c, b1);
                        st(local + 448 + 32 + c, b2);
                        st(local + 448 + 48 + c, b3);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;barrier.cta.sync 1,256;" ::
                                 : "memory");
                if (t ==
                    0) { // the relaxed representative arrives (RREL 2) for the TMEM-read signals
                    unsigned remote;
                    asm volatile(
                        "tcgen05.fence::after_thread_sync;tcgen05.fence::before_thread_sync;" ::
                            : "memory");
                    if (sh.pingpong) { // Release the TMEM places after all payout loads and the CTA barrier.
                        if (nacc > 1 || paccb == 0) {
                            asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                         : "=r"(remote)
                                         : "r"(sa(&place_free0)));
                            asm volatile(
                                "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::"r"(
                                    remote)
                                : "memory");
                        }
                        if (nacc > 1 || paccb == 1) {
                            asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                         : "=r"(remote)
                                         : "r"(sa(&place_free1)));
                            asm volatile(
                                "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::"r"(
                                    remote)
                                : "memory");
                        }
                    }
                    asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                 : "=r"(remote)
                                 : "r"(sa(&read_done)));
                    asm volatile(
                        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,[%0];" ::"r"(remote)
                        : "memory");
                }
                asm volatile("barrier.cta.sync 1,256;" ::: "memory");
                if (t == 0) {
                    measures[(size_t(tile) * kParts + part) * 2 + rank].payout = clock64() - began;
                    unsigned remote;
                    asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                 : "=r"(remote)
                                 : "r"(sa(&store_done)));
                    asm volatile(
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(remote)
                        : "memory");
                }
                if (k > 0 &&
                    w < F) { // the CRT of one slice of the previous tile's columns, behind the next MMA
                    const auto s0 = clock64();
                    const unsigned pt = unsigned(k - 1) * sh.clusters + q;
                    const int cc0 =
                        CRT_PROP_ON(sh)
                            ? crt_cum[part] * 112 / crt_cum[kParts]
                            : part * 112 / kParts; // Proportional CRT column assignment.
                    const int cc1 = CRT_PROP_ON(sh) ? crt_cum[part + 1] * 112 / crt_cum[kParts]
                                                    : (part + 1) * 112 / kParts;
                    crt_cols_rt(resid, cout, ea, eb, crt_p, l8_tile_row(sh, pt),
                                l8_tile_col(sh, pt), rank, t, cc0, cc1, sh, pt);
                    if (t == 0) {
                        measures[(size_t(tile) * kParts + part) * 2 + rank].payout_read =
                            clock64() - s0;
                    }
                }
            }
            if (t == 0 && w < F) { // the per-tile timing of whole items only
                auto now = clock64();
                km[size_t(tile) * 2 + rank].total = now - prev;
                km[size_t(tile) * 2 + rank].smid = smid();
                prev = now;
            }
        }
        if (ntiles > 0 && unsigned(ntiles - 1) * sh.clusters + q <
                              F) { // the last item whole = the kernel's CRT; split = l8k_crt_after
            const unsigned pt = unsigned(ntiles - 1) * sh.clusters + q;
            crt_cols_rt(resid, cout, ea, eb, crt_p, l8_tile_row(sh, pt), l8_tile_col(sh, pt), rank,
                        t, 0, 112, sh, pt);
        }
    }
    __syncthreads();
    cluster.sync();
    if (t < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,512;tcgen05.relinquish_alloc_"
            "permit.cta_group::2.sync.aligned;" ::"r"(tm)
            : "memory");
    }
    cluster.sync();
}

// the CRT after the GEMM when the moduli of a tile are split over clusters (sh.nchunk > 1): the same crt_cols_rt as the
// The follow-up kernel partitions each (tile, rank) into column ranges.
// the split's GEMM stage at 214 us for about 12 us of MMA work per cluster, the CRT on 12 blocks being the suspect).
__global__ void __launch_bounds__(256)
    l8k_crt_after(const unsigned short *resid, double *cout, const int *ea, const int *eb,
                  f64_impl::U128 crt_p, const __grid_constant__ L8Shape sh, int nch) {
    const unsigned n = unsigned(nch), tile = unsigned(sh.whole) + blockIdx.x / (2u * n),
                   rank = (blockIdx.x / n) % 2u, ch = blockIdx.x % n; // the split tiles
    crt_cols_rt(resid, cout, ea, eb, crt_p, l8_tile_row(sh, tile), l8_tile_col(sh, tile), rank,
                threadIdx.x, int(ch) * 112 / nch, (int(ch) + 1) * 112 / nch, sh, tile);
}
