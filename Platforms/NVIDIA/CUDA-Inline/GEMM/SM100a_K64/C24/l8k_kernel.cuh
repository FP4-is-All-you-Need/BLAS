template <bool Unused>
__global__ __launch_bounds__(384, 1) void l8k_gemm_kernel(
    const unsigned char *A, const unsigned char *B, unsigned short *resid, const L8kMod *mods,
    unsigned block_tiles, KernelMetrics *km, Metrics *measures, f64_impl::U128 *crt_state,
    const f64_impl::U128 *crtw, double *cout, const int *ea, const int *eb, f64_impl::U128 crt_p,
    const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB) {
    (void)crt_state;
    (void)crtw;
    const unsigned q = blockIdx.x / 2;
    const auto tile_begin = clock64();
    auto cluster = cooperative_groups::this_cluster();
    unsigned rank = cluster.block_rank(), t = threadIdx.x;
    __shared__ unsigned tmbase;
    __shared__ __align__(8) unsigned long long afull[kSlots], bfull[kSlots], empty[kSlots], drain,
        read_done, store_done;
    extern __shared__ __align__(1024) unsigned char sm[];
    if (t == 0) {
#pragma unroll
        for (int s = 0; s < kSlots; ++s)
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0],1;mbarrier.init.shared::cta.b64 [%1],1;"
                "mbarrier.init.shared::cta.b64 [%2],1;" ::"r"(sa(&afull[s])),
                "r"(sa(&bfull[s])), "r"(sa(&empty[s]))
                : "memory");
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0],1;mbarrier.init.shared::cta.b64 [%1],2;"
            "mbarrier.init.shared::cta.b64 [%2],2;fence.mbarrier_init.release.cluster;" ::"r"(
                sa(&drain)),
            "r"(sa(&read_done)), "r"(sa(&store_done))
            : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    cluster.sync();
    if (t < 32)
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],512;" ::"r"(sa(&tmbase))
            : "memory");
    __syncthreads();
    cluster.sync();
    unsigned tm = tmbase, local = tm + ((t / 32 % 4 * 32) << 16);

    if (t < 128) {
        const unsigned code[4] = {0x40u, mods[0].code[0], mods[0].code[1], mods[0].code[2]};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned bits = code[s] * 0x01010101u;
            for (int c = 0; c < 16; ++c)
                st(local + 448 + 16 * s + c, bits);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;tcgen05.fence::before_thread_sync;" ::
                         : "memory");
    }
    __syncthreads();
    cluster.sync();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (t == 288 || t == 320) {
        const bool side_b = t == 320;
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = stream_tile(q, k);
            const size_t block = size_t(side_b ? tile_col(tile) : tile_row(tile)) * block_tiles;
            for (int part = 0; part < kParts; ++part) {
                const int F = mods[part].planes, cpg = mods[part].cpg, groups = mods[part].groups;
                const int gb = k * mods[part].gt + mods[part].gbase, pbase = mods[part].pbase;
                for (int g = 0; g < groups; ++g) {
                    const int G = gb + g, slot = G % kSlots;
                    if (G >= kSlots)
                        await(sa(&empty[slot]), ((G / kSlots) + 1) & 1);
                    const int cut0 = g * cpg, nc = min(cpg, Cuts - cut0);
                    const unsigned ntile = unsigned(nc * F), bytes = ntile * TileBytes;

#if L8_FOLD_TILES

                    const size_t first =
                        (block + size_t(2) * Cuts * pbase + (size_t(rank) * Cuts + cut0) * F) %
                        L8_FOLD_TILES;
#else
                    const size_t first =
                        block + size_t(2) * Cuts * pbase + (size_t(rank) * Cuts + cut0) * F;
#endif
                    const unsigned dest = sa(sm + slot * kGroup + (side_b ? kOperand : 0));
                    const unsigned lbar_local = sa(side_b ? &bfull[slot] : &afull[slot]);
                    unsigned lbar;
                    asm volatile("mapa.shared::cluster.u32 %0,%1,0;"
                                 : "=r"(lbar)
                                 : "r"(lbar_local));
                    if (rank == 0)
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;" ::
                                "r"(lbar_local),
                            "r"(2u * bytes)
                            : "memory");
                    for (unsigned b = 0; b < ntile; b += kBoxTiles)
                        asm volatile(
                            "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                            "[%0],[%1,{%2,%3}],[%4];" ::"r"(dest + b * TileBytes),
                            "l"(reinterpret_cast<unsigned long long>(side_b ? &tmB : &tmA)), "r"(0),
                            "r"(int((first + b) * (TileBytes / 256))), "r"(lbar)
                            : "memory");
                }
            }
        }
    } else if (t == 256 && rank == 0) {
        unsigned long long abase[kSlots], bbase[kSlots];
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            const unsigned addr = sa(sm + slot * kGroup);
            abase[slot] = tile_desc(addr);
            bbase[slot] = tile_desc(addr + kOperand);
        }
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = stream_tile(q, k);
            for (int part = 0; part < kParts; ++part) {
                const int step = k * kParts + part;
                const int F = mods[part].planes, cpg = mods[part].cpg, np = mods[part].nprod;
                const int first = k * mods[part].gt + mods[part].gbase,
                          last = first + mods[part].groups;
                unsigned pr[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    pr[i] = mods[part].prod[i];
                const unsigned sf0 = tm + 448;
                unsigned long long aw = 0, bw = 0, issue = 0, ns0;
                auto core_begin = clock64();
                asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ns0));
                for (int round = first / kSlots; round <= (last - 1) / kSlots; ++round) {
#pragma unroll
                    for (int slot = 0; slot < kSlots; ++slot) {
                        const int G = round * kSlots + slot;
                        if (G >= first && G < last) {
                            const int g = G - first, nc = min(cpg, Cuts - g * cpg);
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
                                        face_mma<false, 0>(tm + acc * Cols, a, b, sf0,
                                                           sf0 + 16 * bs, !overwrite);
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
                await(sa(&read_done), step & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                await(sa(&store_done), step & 1);
            }
        }
    } else if (t < 256) {
        long long prev = tile_begin;
        for (int k = 0; k < TilesPerCluster; ++k) {
            const unsigned tile = stream_tile(q, k), tm_row = tile_row(tile), tn = tile_col(tile);
            for (int part = 0; part < kParts; ++part) {
                const int step = k * kParts + part;
                await(sa(&drain), step & 1);
                auto began = clock64();
                const int p = mods[part].p, gamma = mods[part].gamma, bias = mods[part].bias,
                          nacc = mods[part].nacc;
                const unsigned mmagic = mods[part].mmagic;
                ResT *out = reinterpret_cast<ResT *>(resid) + size_t(part) * Count;
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
                for (int chunk = 0; chunk < 7; ++chunk) {
                    unsigned v0[16], v1[16];
                    const unsigned col = (t / 128) * 112 + chunk * 16;
                    load16(local + col, v0);
                    if (nacc > 1) {
                        load16(local + Cols + col, v1);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; ++j)
                            v1[j] = 0u;
                    }
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const unsigned global_n = tn * Cols + col + j;
                        if (global_n >= unsigned(N))
                            continue;
                        const size_t index =
                            size_t(global_n) * M + tm_row * Side + rank * (Side / 2) + (t % 128);
                        const int x0 = __float2int_rn(__uint_as_float(v0[j]));
                        const int x1 = __float2int_rn(__uint_as_float(v1[j]));

                        const unsigned tt =
                            (unsigned)((long long)x0 + (long long)gamma * x1 + (long long)bias);
                        const unsigned qd = (unsigned)(((unsigned long long)tt * mmagic) >> 32);
                        unsigned r = tt - qd * (unsigned)p;
                        if (r >= (unsigned)p)
                            r -= (unsigned)p;
                        out[index] = (ResT)r;
                    }
                }

                if (t < 128) {
                    const int nx = part + 1 == kParts ? 0 : part + 1;
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
                if (k > 0) {
                    const auto s0 = clock64();
                    const unsigned pt = stream_tile(q, k - 1);
                    crt_cols(resid, cout, ea, eb, crt_p, tile_row(pt), tile_col(pt), rank, t,
                             part * 112 / kParts, (part + 1) * 112 / kParts);
                    if (t == 0)
                        measures[(size_t(tile) * kParts + part) * 2 + rank].payout_read =
                            clock64() - s0;
                }
            }
            if (t == 0) {
                auto now = clock64();
                km[size_t(tile) * 2 + rank].total = now - prev;
                km[size_t(tile) * 2 + rank].smid = smid();
                prev = now;
            }
        }
        {
            const unsigned pt = stream_tile(q, TilesPerCluster - 1);
            crt_cols(resid, cout, ea, eb, crt_p, tile_row(pt), tile_col(pt), rank, t, 0, 112);
        }
    }
    __syncthreads();
    cluster.sync();
    if (t < 32)
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,512;tcgen05.relinquish_alloc_"
            "permit.cta_group::2.sync.aligned;" ::"r"(tm)
            : "memory");
    cluster.sync();
}
