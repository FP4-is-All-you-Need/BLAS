

// ---- 2026-09-28 (make_l8k_kernel_rt.py): the run-time form. See make_l8k_kernel_rt.py for what changed. ----
#ifndef L8_PAY_N
#define L8_PAY_N 1
#endif
struct L8Shape {
    const unsigned *s3_tiles = nullptr;
    int s3_lower = 0, s3_triangular_a = 0, s3_defer_crt = 0;
    int m, n, cuts, tile_rows, tile_cols, tiles, clusters, nchunk;
    int whole;
    long long ldc; // doubles between columns of C (>= m)
    unsigned long long
        plane; // residues per modulus plane = tiles * Cols * Side (tile-major, padding included)
    int pace_s;
    unsigned *pace;
    unsigned *pace_timeouts;
    int item_order;
    int pingpong;
};
__device__ __forceinline__ int l8_ntiles(const L8Shape &sh, unsigned q) {
    return q < unsigned(sh.tiles)
               ? int((unsigned(sh.tiles) - q + unsigned(sh.clusters) - 1) / unsigned(sh.clusters))
               : 0;
}
__device__ __forceinline__ int l8_nwork(const L8Shape &sh, unsigned q, int nc) {
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
// the CRT of columns [c0, c1) of this thread's row of one tile (fp64k_6251.cu crt_cols, RESID_TILE with RT_CRT)
__device__ __forceinline__ void crt_cols_rt(const unsigned short *resid, L8OutT *cout,
                                            const int *ea, const int *eb, f64_impl::U128 P,
                                            unsigned tm_row, unsigned tn, unsigned rank, unsigned t,
                                            int c0, int c1, const L8Shape &sh, unsigned tile_id) {
    const unsigned row = tm_row * Side + rank * (Side / 2) + (t % 128);
    if (row >= unsigned(sh.m)) {
        return;
    }
    const int er = -ea[row];
    for (int c = c0; c < c1; ++c) {
        const unsigned global_n = tn * Cols + (t / 128) * 112 + c;
        if (global_n >= unsigned(sh.n) || (sh.s3_lower && row < global_n)) {
            continue;
        }
        const size_t cindex = size_t(global_n) * sh.ldc + row;
        unsigned r[kParts];
        {
            const ResT *rp = reinterpret_cast<const ResT *>(resid) +
                             (size_t(tile_id) * Cols + (t / 128) * 112 + c) * Side +
                             rank * (Side / 2) + (t % 128);
#pragma unroll
            for (int q = 0; q < kParts; ++q) {
                r[q] = __ldcg(rp);
                rp += sh.plane;
            }
        }
#if L8_F32 // l8f: crt_f32.cuh (s mod P for 2^64 < P < 2^66, one rounding to binary32)
        cout[cindex] = __uint_as_float(crtf_element_bits(r, crtf_c, er - eb[global_n]));
#else
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
        cout[cindex] = round_scaled_int(neg ? P - u : u, neg, er - eb[global_n]);
#endif
    }
}

template <bool Unused>
__global__ __launch_bounds__(384, 1) void l8k_gemm_kernel_rt(
    const unsigned char *A, const unsigned char *B, unsigned short *resid, const L8kMod *mods,
    unsigned block_tiles, KernelMetrics *km, Metrics *measures, f64_impl::U128 *crt_state,
    const f64_impl::U128 *crtw, L8OutT *cout, const int *ea, const int *eb, f64_impl::U128 crt_p,
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
    const __grid_constant__ L8Shape sh) {
    (void)crt_state;
    (void)crtw;
    const unsigned q = blockIdx.x / 2;
    const int nc = sh.nchunk > 1 ? sh.nchunk : 1;
    const int ntiles = l8_nwork(sh, q, nc), cuts = sh.cuts; // work items (= tiles when nc = 1)
    const unsigned F = nc > 1 ? unsigned(sh.whole) : 0xffffffffu;
    const unsigned NS = unsigned(sh.tiles) - (F < unsigned(sh.tiles) ? F : unsigned(sh.tiles));
    auto chunk_of = [&](unsigned w) {
        return sh.item_order ? (w - F) / NS : (w - F) % unsigned(nc);
    };
    auto tile_of = [&](unsigned w) {
        unsigned tile = w < F ? w : (sh.item_order ? F + (w - F) % NS : F + (w - F) / unsigned(nc));
        return sh.s3_tiles ? sh.s3_tiles[tile] : tile;
    };
    auto p0_of = [&](unsigned w) { return w < F ? 0 : int(chunk_of(w)) * kParts / nc; };
    auto p1_of = [&](unsigned w) { return w < F ? kParts : (int(chunk_of(w)) + 1) * kParts / nc; };
    const auto tile_begin = clock64();
    auto cluster = cooperative_groups::this_cluster();
    unsigned rank = cluster.block_rank(), t = threadIdx.x;
    __shared__ unsigned tmbase;
    __shared__ __align__(8) unsigned long long afull[kSlots], bfull[kSlots], empty[kSlots], drain,
        read_done, store_done, slots_ready, place_free0, place_free1;
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
    // scale slots: 4 x 16 columns after the two accumulators (2 x 224 + 64 = 512). Slot 0 = weight 1 (E4M3 2.0); slots 1..3 =
    // modulus 0's weights here, then the payout writes the next modulus's.
    if (t < 128) {
        const int f0 = ntiles > 0 ? p0_of(q) : 0;
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
        int gcnt = 0;
        unsigned pcnt = 0, pace_seen = 0;
        for (int k = 0; k < ntiles; ++k) {
            const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);
            const int s3_cuts =
                sh.s3_triangular_a
                    ? min(cuts, l8e::cuts_of(
                                    int(min(unsigned(sh.m), (l8_tile_row(sh, tile) + 1) * Side))))
                    : cuts;
            const size_t block = size_t(side_b ? l8_tile_col(sh, tile) : l8_tile_row(sh, tile)) *
                                 block_tiles; // in plane tiles
            for (int part = p0_of(w); part < p1_of(w); ++part) {
                const int F = mods[part].planes, cpg = mods[part].cpg,
                          groups = (s3_cuts + cpg - 1) / cpg;
                const int gb = gcnt, pbase = mods[part].pbase;
                gcnt += groups;
                if (!side_b && sh.pace_s >= 0) {
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
                    const int cut0 = g * cpg, nc = min(cpg, s3_cuts - cut0);
                    const unsigned ntile = unsigned(nc * F), bytes = ntile * TileBytes;

#if L8_FOLD_TILES

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
            *(volatile unsigned *)(sh.pace + q) = 0xffffffffu;
        }
    } else if (t == 256 && rank == 0) { // issue
        unsigned long long abase[kSlots], bbase[kSlots];
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const unsigned addr = sa(sm + slot * kGroup);
            abase[slot] = tile_desc(addr);
            bbase[slot] = tile_desc(addr + kOperand);
        }
        int gcnt = 0, scnt = 0;
        int sel = 0, u0 = 0, u1 = 0, w0 = 0, w1 = 0;
        for (int k = 0; k < ntiles; ++k) {
            const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);
            const int s3_cuts =
                sh.s3_triangular_a
                    ? min(cuts, l8e::cuts_of(
                                    int(min(unsigned(sh.m), (l8_tile_row(sh, tile) + 1) * Side))))
                    : cuts;
            for (int part = p0_of(w); part < p1_of(w); ++part) {
                const int step = scnt++;
                const int F = mods[part].planes, cpg = mods[part].cpg, np = mods[part].nprod;
                const int first = gcnt, last = first + (s3_cuts + cpg - 1) / cpg;
                gcnt = last;
                unsigned pr[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pr[i] = mods[part].prod[i];
                }
                const unsigned sf0 = tm + 448;
                int accb = 0;
                if (sh.pingpong) {
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
                measures[(size_t(tile) * kParts + part) * 2 + 1].ns = ns0;
                for (int round = first / kSlots; round <= (last - 1) / kSlots; ++round) {
#pragma unroll
                    for (int slot = 0; slot < kSlots; ++slot) {
                        const int G = round * kSlots + slot;
                        if (G >= first && G < last) {
                            const int g = G - first, nc = min(cpg, s3_cuts - g * cpg);
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
                                                       acc = (e >> 4) & 1u, bs = (e >> 5) & 3u;
                                        const bool overwrite = ((e >> 7) & 1u) && g == 0 && c == 0;
                                        const unsigned long long a =
                                            abase[slot] + (unsigned long long)((c * F + int(fa)) *
                                                                               (TileBytes >> 4));
                                        const unsigned long long b =
                                            bbase[slot] + (unsigned long long)((c * F + int(fb)) *
                                                                               (TileBytes >> 4));
                                        face_mma<false, 0>(tm + (unsigned(accb) + acc) * Cols, a, b,
                                                           sf0, sf0 + 16 * bs, !overwrite);
                                    }
                                }
                            }
                            issue += clock64() - i0;
                            done(sa(&empty[slot]));
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
                if (sh.pingpong) {
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
        int scnt = 0;
        int psel = 0;
        for (int k = 0; k < ntiles; ++k) {
            const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);
            const int s3_cuts =
                sh.s3_triangular_a
                    ? min(cuts, l8e::cuts_of(
                                    int(min(unsigned(sh.m), (l8_tile_row(sh, tile) + 1) * Side))))
                    : cuts;
            const unsigned tm_row = l8_tile_row(sh, tile), tn = l8_tile_col(sh, tile);
            const int pend = p1_of(w), pnext = k + 1 < ntiles ? p0_of(w + sh.clusters) : 0;
            if (!sh.s3_defer_crt && k > 0 && w >= F && w - sh.clusters < F) {
                const unsigned pt = tile_of(w - sh.clusters);
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
                int paccb = 0;
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
                    if (t == 0) {
                        unsigned remote;
                        asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                     : "=r"(remote)
                                     : "r"(sa(&slots_ready)));
                        asm volatile(
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(
                                remote)
                            : "memory");
                    }
                }
#if L8_PAY_N
                if (nacc > 1) {
#pragma unroll
                    for (int chunk = 0; chunk < 7; ++chunk) {
                        unsigned v0[16], v1[16];
                        const unsigned col = (t / 128) * 112 + chunk * 16;
                        load16(local + unsigned(paccb) * Cols + col, v0);
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
                        load16(local + unsigned(paccb) * Cols + col, v0);
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
                    load16(local + unsigned(paccb) * Cols + col, v0);
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
                if (t < 128 && !sh.pingpong) {
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
                if (t == 0) {
                    unsigned remote;
                    asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                 : "=r"(remote)
                                 : "r"(sa(&read_done)));
                    asm volatile(
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(remote)
                        : "memory");
                    if (sh.pingpong) {
                        if (nacc > 1 || paccb == 0) {
                            asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                         : "=r"(remote)
                                         : "r"(sa(&place_free0)));
                            asm volatile(
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(
                                    remote)
                                : "memory");
                        }
                        if (nacc > 1 || paccb == 1) {
                            asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                         : "=r"(remote)
                                         : "r"(sa(&place_free1)));
                            asm volatile(
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];" ::"r"(
                                    remote)
                                : "memory");
                        }
                    }
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
                if (!sh.s3_defer_crt && k > 0 && w < F) {
                    const auto s0 = clock64();
                    const unsigned pt = tile_of(unsigned(k - 1) * sh.clusters + q);
                    crt_cols_rt(resid, cout, ea, eb, crt_p, l8_tile_row(sh, pt),
                                l8_tile_col(sh, pt), rank, t, part * 112 / kParts,
                                (part + 1) * 112 / kParts, sh, pt);
                    if (t == 0) {
                        measures[(size_t(tile) * kParts + part) * 2 + rank].payout_read =
                            clock64() - s0;
                    }
                }
            }
            if (t == 0 && w < F) {
                auto now = clock64();
                km[size_t(tile) * 2 + rank].total = now - prev;
                km[size_t(tile) * 2 + rank].smid = smid();
                prev = now;
            }
        }
        if (!sh.s3_defer_crt && ntiles > 0 && unsigned(ntiles - 1) * sh.clusters + q < F) {
            const unsigned pt = tile_of(unsigned(ntiles - 1) * sh.clusters + q);
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

__global__ void __launch_bounds__(256)
    l8k_crt_after(const unsigned short *resid, L8OutT *cout, const int *ea, const int *eb,
                  f64_impl::U128 crt_p, const __grid_constant__ L8Shape sh, int nch) {
    const unsigned n = unsigned(nch), tile = unsigned(sh.whole) + blockIdx.x / (2u * n),
                   rank = (blockIdx.x / n) % 2u, ch = blockIdx.x % n;
    crt_cols_rt(resid, cout, ea, eb, crt_p, l8_tile_row(sh, tile), l8_tile_col(sh, tile), rank,
                threadIdx.x, int(ch) * 112 / nch, (int(ch) + 1) * 112 / nch, sh, tile);
}
