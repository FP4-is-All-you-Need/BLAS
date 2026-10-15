#pragma once
#include <atomic>
#include <map>

__global__ void pad_changed_d(const unsigned long long *c, int m, int n, long long ldc,
                              unsigned long long *out) {
    const size_t per_col = size_t(ldc - m), total = per_col * size_t(n);
    unsigned long long bad = 0;
    for (size_t e = size_t(blockIdx.x) * blockDim.x + threadIdx.x; e < total;
         e += size_t(blockDim.x) * gridDim.x) {
        const size_t j = e / per_col, i = size_t(m) + e % per_col;
        bad += c[j * size_t(ldc) + i] != 0xa5a5a5a5a5a5a5a5ull;
    }
    if (bad)
        atomicAdd(out, bad);
}

__global__ void xor_hash_d(const double *a, size_t n, unsigned long long *out) {
    unsigned long long h = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x)
        h ^= (unsigned long long)__double_as_longlong(a[i]) * 0x9E3779B97F4A7C15ull +
             i * 0xD1342543DE82EF95ull;
    for (int o = 16; o; o >>= 1)
        h ^= __shfl_down_sync(0xffffffffu, h, o);
    if ((threadIdx.x & 31) == 0)
        atomicXor(out, h);
}
static unsigned long long xor_hash_of(const double *a, size_t n) {
    unsigned long long *d;
    ck(cudaMalloc(&d, 8));
    ck(cudaMemset(d, 0, 8));
    xor_hash_d<<<4096, 256>>>(a, n, d);
    ck(cudaGetLastError());
    unsigned long long h = 0;
    ck(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
    cudaFree(d);
    return h;
}

int main(int argc, char **argv) try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const int runs = argc > 1 ? atoi(argv[1]) : 4;
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || (prop.minor != 3 && prop.minor != 0))
        throw std::runtime_error("requires B200 sm_100a or B300 sm_103a (dense K64)");
    int runtime = 0;
    ck(cudaRuntimeGetVersion(&runtime));
    int m = 8192, n = 8192, k = 8192;
    long long lda = 0, ldb = 0, ldc = 0;
    if (const char *zs = getenv("L8SHAPE")) {
        const int got = sscanf(zs, "%d,%d,%d,%lld,%lld,%lld", &m, &n, &k, &lda, &ldb, &ldc);
        if (got != 3 && got != 6)
            throw std::runtime_error("L8SHAPE must be m,n,k or m,n,k,lda,ldb,ldc");
    }
    if (!lda) {
        lda = m;
        ldb = k;
        ldc = m;
    }
    const std::string sus = getenv("L8RT_SUSTAIN") ? getenv("L8RT_SUSTAIN") : "none";

    const unsigned reuse = getenv("L8RT_REUSE") ? unsigned(atoi(getenv("L8RT_REUSE"))) : 0u;
    if (sus != "full" && sus != "short" && sus != "none")
        throw std::runtime_error("L8RT_SUSTAIN must be full, short or none");

    l8rt::Plan plan;
    plan.init(m, n, k, lda, ldb, ldc);
    const size_t ca = size_t(m) * k, cb = size_t(k) * n, cc = size_t(m) * n;
    const size_t la = size_t(lda) * k, lb = size_t(ldb) * n, lc = size_t(ldc) * n;
    printf(
        "{\"kind\":\"shape\",\"m\":%d,\"n\":%d,\"k\":%d,\"k_plan\":%d,\"cuts\":%d,\"tile_rows\":%d,\"tile_cols\":%d,"
        "\"tiles\":%d,\"clusters\":%d,\"nchunk\":%d,\"whole\":%d,\"crt_chunks\":%d,\"crt_after_gemm\":%s}\n",
        m, n, k, plan.kplan, plan.sh.cuts, plan.sh.tile_rows, plan.sh.tile_cols, plan.sh.tiles,
        plan.sh.clusters, plan.sh.nchunk, plan.sh.whole, plan.crt_chunks,
        plan.sh.nchunk > 1 ? "true" : "false");

#if L8_FAST_ENC
    const int enc_minb = plan.enc.minb;
    const bool enc_skip = plan.enc.skip_b_pad;
    const bool pdl = plan.enc.pdl, pdl_ok = plan.enc.pdl_supported;
    const int shift_form = plan.enc.shift_form;
#if L8_EXTRA_FX_KNOB
    const int extra_fx = plan.enc.extra_fx;
#else
    const int extra_fx = -1;
#endif
#else
    const int enc_minb = -1;
    const bool enc_skip = false;
    const bool pdl = false, pdl_ok = false;
    const int shift_form = -1;
    const int extra_fx = -1;
#endif
    printf(
        "{\"kind\":\"environment\",\"engine\":\"l8rt\",\"variant\":59,\"gpu\":\"%s\",\"cuda_runtime\":%d,"
        "\"m\":%d,\"n\":%d,\"k\":%d,\"bits\":%d,\"moduli\":%d,\"planes_per_side\":%d,\"mma_products\":%d,"
        "\"accumulators_per_tile\":%d,\"k_plan\":%d,\"tile_m\":%d,\"tile_n\":%d,\"cuts\":%d,\"face_tiles_per_slot\":%d,"
        "\"slots\":%d,\"cta_group\":2,\"lambda\":\"per accumulator\",\"encoder\":\"%s\",\"sustain\":\"%s\",\"reuse\":%u,"
        "\"enc_minb\":%d,\"enc_skip_b_pad\":%s,\"pace_s\":%d,\"pdl\":%s,\"pdl_supported\":%s,\"enc_shift_form\":%d,\"enc_extra_fx\":%d,\"item_order\":%d,\"pingpong\":%d}\n",
        prop.name, runtime, m, n, k, plan.bits, L8K_NMOD, L8K_PLANES59, L8K_PRODUCTS59, plan.accs,
        plan.kplan, Side, Cols, plan.sh.cuts, L8K_FTS, kSlots, L8_FAST_ENC ? "fast" : "base",
        sus.c_str(), reuse, enc_minb, enc_skip ? "true" : "false", plan.sh.pace_s,
        pdl ? "true" : "false", pdl_ok ? "true" : "false", shift_form, extra_fx, plan.sh.item_order,
        plan.sh.pingpong);
    {
        cudaFuncAttributes ka{};
        ck(cudaFuncGetAttributes(&ka, l8k_gemm_kernel_rt<true>));
        printf(
            "{\"kind\":\"kernel\",\"registers\":%d,\"static_shared\":%zu,\"dynamic_shared\":%d}\n",
            ka.numRegs, ka.sharedSizeBytes, kShared);
    }

    double *A, *B, *C, *Ap, *Bp, *Cp;
    ck(cudaMalloc(&Ap, ca * 8));
    ck(cudaMalloc(&Bp, cb * 8));
    ck(cudaMalloc(&Cp, cc * 8));
    ck(cudaMalloc(&A, la * 8));
    ck(cudaMalloc(&B, lb * 8));
    ck(cudaMalloc(&C, lc * 8));
    fill_normal<<<4096, 256>>>(Ap, ca, 20260921ull);
    fill_normal<<<4096, 256>>>(Bp, cb, 76543210ull);
    ck(cudaGetLastError());
    printf("{\"kind\":\"inputs\",\"a_hash\":\"%016llx\",\"b_hash\":\"%016llx\"}\n",
           xor_hash_of(Ap, ca), xor_hash_of(Bp, cb));
    ck(cudaMemset(A, 0xff, la * 8));
    ck(cudaMemset(B, 0xff, lb * 8));
    ck(cudaMemcpy2D(A, size_t(lda) * 8, Ap, size_t(m) * 8, size_t(m) * 8, k,
                    cudaMemcpyDeviceToDevice));
    ck(cudaMemcpy2D(B, size_t(ldb) * 8, Bp, size_t(k) * 8, size_t(k) * 8, n,
                    cudaMemcpyDeviceToDevice));
    cudaFree(Ap);
    cudaFree(Bp);
    printf("{\"kind\":\"ld\",\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,\"strided\":%s}\n", lda, ldb,
           ldc, (lda != m || ldb != k || ldc != m) ? "true" : "false");
    auto out_hash = [&] {
        ck(cudaMemcpy2D(Cp, size_t(m) * 8, C, size_t(ldc) * 8, size_t(m) * 8, n,
                        cudaMemcpyDeviceToDevice));
        return xor_hash_of(Cp, cc);
    };

    double best = 1e30;
    for (int r = 0; r < runs; ++r) {
        ck(cudaMemsetAsync(C, 0xa5, lc * 8, 0));
        if (!(r > 0 && (reuse & 1u)))
            ck(cudaMemsetAsync(plan.pa, 0xa5, plan.bytesA, 0));
        if (!(r > 0 && (reuse & 2u)))
            ck(cudaMemsetAsync(plan.pb, 0xa5, plan.bytesB, 0));
        ck(cudaMemsetAsync(plan.resid, 0xa5, plan.resid_bytes, 0));
        ck(cudaMemsetAsync(plan.km, 0, size_t(plan.sh.tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(plan.mm, 0, size_t(plan.sh.tiles) * kParts * 2 * sizeof(Metrics), 0));
        float st[3] = {0, 0, 0};
        if (plan.run(A, B, C, 0, st, r > 0 ? reuse : 0u) == 2)
            throw std::runtime_error("input not finite");
        const float tot = st[0] + st[1] + st[2];
        printf(
            "{\"kind\":\"ours\",\"run\":%d,\"warmup\":%s,\"total_ms\":%.6f,\"quantize_ms\":%.6f,\"encode_ms\":%.6f,"
            "\"gemm_ms\":%.6f,\"crt_ms\":%.6f}\n",
            r, r == 0 ? "true" : "false", tot, st[0], st[1], st[2], plan.last_crt_ms);
        if (r)
            best = std::min<double>(best, tot);
    }
    {
        unsigned long long *dcnt;
        ck(cudaMalloc(&dcnt, 8));
        ck(cudaMemset(dcnt, 0, 8));
        if (ldc > m) {
            pad_changed_d<<<1024, 256>>>(reinterpret_cast<const unsigned long long *>(C), m, n, ldc,
                                         dcnt);
            ck(cudaGetLastError());
        }
        unsigned long long hc = 0;
        ck(cudaMemcpy(&hc, dcnt, 8, cudaMemcpyDeviceToHost));
        cudaFree(dcnt);
        printf(
            "{\"kind\":\"ld_check\",\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,\"c_padding_changed\":%llu}\n",
            lda, ldb, ldc, hc);
    }
    const unsigned long long h1 = out_hash();
    printf("{\"kind\":\"output\",\"ours_hash\":\"%016llx\",\"ours_best_ms\":%.6f}\n", h1, best);
    {
        const int Tiles = plan.sh.tiles;
        std::vector<KernelMetrics> hkm(size_t(Tiles) * 2);
        std::vector<Metrics> hmm(size_t(Tiles) * kParts * 2);
        ck(cudaMemcpy(hkm.data(), plan.km, hkm.size() * sizeof(KernelMetrics),
                      cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(hmm.data(), plan.mm, hmm.size() * sizeof(Metrics), cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> tot;
        tot.reserve(Tiles);
        for (int i = 0; i < Tiles; ++i)
            tot.push_back(hkm[size_t(i) * 2].total);
        std::sort(tot.begin(), tot.end());
        unsigned long long core = 0, aw = 0, bw = 0, iss = 0, pay = 0, pread = 0, ns = 0;
        for (size_t i = 0; i < hmm.size(); i += 2) {
            core += hmm[i].core;
            aw += hmm[i].a_wait;
            bw += hmm[i].b_wait;
            iss += hmm[i].issue;
            pay += hmm[i].payout;
            pread += hmm[i].payout_read;
            ns += hmm[i].ns;
        }
        const double nn = double(Tiles) * kParts;
        printf(
            "{\"kind\":\"cycles\",\"tile_total_median\":%llu,\"per_modulus_core_mean\":%.1f,"
            "\"a_wait_mean\":%.1f,\"b_wait_mean\":%.1f,\"issue_mean\":%.1f,\"payout_mean\":%.1f,\"payout_read_mean\":%.1f,"
            "\"mma_bound_per_modulus\":%d,\"core_ghz\":%.4f}\n",
            tot[Tiles / 2], core / nn, aw / nn, bw / nn, iss / nn, pay / nn, pread / nn,
            plan.sh.cuts * L8K_PRODUCTS59 * 112 / L8K_NMOD, ns ? double(core) / double(ns) : 0.0);
        {

            const int C = plan.sh.clusters, W = plan.sh.nchunk > 1 ? plan.sh.whole : Tiles,
                      TR = plan.sh.tile_rows, TC = plan.sh.tile_cols;
            auto row_of = [&](int t) {
                const int band = t / (8 * TC), w = t - band * 8 * TC,
                          rb = std::min(8, TR - band * 8);
                return band * 8 + w % rb;
            };
            auto col_of = [&](int t) {
                const int band = t / (8 * TC), w = t - band * 8 * TC,
                          rb = std::min(8, TR - band * 8);
                return w / rb;
            };
            auto st = [&](int t, int j) {
                return (long long)hmm[(size_t(t) * kParts + j) * 2 + 1].ns;
            };
            std::vector<double> all_sp, first_sp, last_sp, arow_sp, bcol_sp, dur;
            int rounds = 0;
            for (int t0 = 0; t0 + C <= W; t0 += C) {
                ++rounds;
                for (int j = 0; j < kParts; ++j) {
                    long long lo = st(t0, j), hi = lo;
                    std::map<int, std::pair<long long, long long>> byr, byc;
                    for (int t = t0; t < t0 + C; ++t) {
                        const long long s = st(t, j);
                        lo = std::min(lo, s);
                        hi = std::max(hi, s);
                        auto &a = byr.emplace(row_of(t), std::make_pair(s, s)).first->second;
                        a.first = std::min(a.first, s);
                        a.second = std::max(a.second, s);
                        auto &b = byc.emplace(col_of(t), std::make_pair(s, s)).first->second;
                        b.first = std::min(b.first, s);
                        b.second = std::max(b.second, s);
                    }
                    all_sp.push_back((hi - lo) / 1e3);
                    if (j == 0)
                        first_sp.push_back((hi - lo) / 1e3);
                    if (j == kParts - 1)
                        last_sp.push_back((hi - lo) / 1e3);
                    for (auto &kv : byr)
                        arow_sp.push_back((kv.second.second - kv.second.first) / 1e3);
                    for (auto &kv : byc)
                        bcol_sp.push_back((kv.second.second - kv.second.first) / 1e3);
                }
            }
            for (int t = 0; t < W; ++t)
                for (int j = 0; j < kParts; ++j)
                    dur.push_back(hmm[(size_t(t) * kParts + j) * 2].ns / 1e3);
            auto q = [](std::vector<double> v, double f) {
                if (v.empty())
                    return -1.0;
                std::sort(v.begin(), v.end());
                return v[size_t(f * (v.size() - 1) + 0.5)];
            };
            printf(
                "{\"kind\":\"part_skew\",\"rounds\":%d,\"clusters\":%d,\"part_ns_median_us\":%.2f,\"spread_median_us\":%.2f,\"spread_p90_us\":%.2f,"
                "\"spread_first_part_median_us\":%.2f,\"spread_last_part_median_us\":%.2f,\"same_a_row_spread_median_us\":%.2f,"
                "\"same_a_row_spread_p90_us\":%.2f,\"same_b_col_spread_median_us\":%.2f,\"same_b_col_spread_p90_us\":%.2f,"
                "\"pace_s\":%d,\"pace_timeouts_so_far\":%u}\n",
                rounds, C, q(dur, 0.5), q(all_sp, 0.5), q(all_sp, 0.9), q(first_sp, 0.5),
                q(last_sp, 0.5), q(arow_sp, 0.5), q(arow_sp, 0.9), q(bcol_sp, 0.5), q(bcol_sp, 0.9),
                plan.sh.pace_s, plan.pace_timeouts());
        }
        {

            const int C = plan.sh.clusters, nc = plan.sh.nchunk > 1 ? plan.sh.nchunk : 1;
            const long F = plan.sh.nchunk > 1 ? plan.sh.whole : Tiles,
                       NS = Tiles - (F < Tiles ? F : Tiles);
            const long items = plan.sh.nchunk > 1 ? F + (long)(Tiles - F) * nc : Tiles;
            auto tile_of = [&](long w) -> long {
                return w < F ? w : (plan.sh.item_order ? F + (w - F) % NS : F + (w - F) / nc);
            };
            auto chunk_of = [&](long w) -> long {
                return plan.sh.item_order ? (w - F) / NS : (w - F) % nc;
            };
            std::vector<double> gin, gbd, span, first_off, last_off, busy;
            long long t0 = -1;
            for (long t = 0; t < Tiles; ++t)
                for (int j = 0; j < kParts; ++j) {
                    const long long s = (long long)hmm[(size_t(t) * kParts + j) * 2 + 1].ns;
                    if (s > 0 && (t0 < 0 || s < t0))
                        t0 = s;
                }
            for (int qc = 0; qc < C; ++qc) {
                long long prev_end = -1, first = -1, sum = 0;
                for (long w = qc; w < items; w += C) {
                    const long t = tile_of(w);
                    long p0 = 0, p1 = kParts;
                    if (w >= F) {
                        const long c = chunk_of(w);
                        p0 = c * kParts / nc;
                        p1 = (c + 1) * kParts / nc;
                    }
                    for (long j = p0; j < p1; ++j) {
                        const long long s = (long long)hmm[(size_t(t) * kParts + j) * 2 + 1].ns,
                                        sp = (long long)hmm[(size_t(t) * kParts + j) * 2].ns;
                        if (first < 0)
                            first = s;
                        if (prev_end >= 0)
                            (j == p0 ? gbd : gin).push_back((s - prev_end) / 1e3);
                        span.push_back(sp / 1e3);
                        sum += sp;
                        prev_end = s + sp;
                    }
                }
                if (first >= 0 && t0 >= 0) {
                    first_off.push_back((first - t0) / 1e3);
                    last_off.push_back((prev_end - t0) / 1e3);
                    busy.push_back(sum / 1e3);
                }
            }
            auto q = [](std::vector<double> v, double f) {
                if (v.empty())
                    return -1.0;
                std::sort(v.begin(), v.end());
                return v[size_t(f * (v.size() - 1) + 0.5)];
            };
            auto s_of = [](const std::vector<double> &v) {
                double a = 0;
                for (double x : v)
                    a += x;
                return a;
            };
            printf(
                "{\"kind\":\"item_gaps\",\"nchunk\":%d,\"item_order\":%d,\"clusters\":%d,\"items\":%ld,\"gap_in_item_n\":%zu,"
                "\"gap_in_item_median_us\":%.3f,\"gap_in_item_p90_us\":%.3f,\"gap_item_boundary_n\":%zu,\"gap_item_boundary_median_us\":%.3f,"
                "\"gap_item_boundary_p90_us\":%.3f,\"span_median_us\":%.3f,\"gap_sum_per_cluster_median_us\":%.1f,\"busy_per_cluster_median_us\":%.1f,"
                "\"first_start_max_us\":%.2f,\"last_end_min_us\":%.1f,\"last_end_max_us\":%.1f}\n",
                nc, plan.sh.item_order, C, items, gin.size(), q(gin, 0.5), q(gin, 0.9), gbd.size(),
                q(gbd, 0.5), q(gbd, 0.9), q(span, 0.5), C ? (s_of(gin) + s_of(gbd)) / C : -1.0,
                q(busy, 0.5), q(first_off, 1.0), q(last_off, 0.0), q(last_off, 1.0));
        }
    }

    if (sus != "none") {
        nvmlDevice_t nd{};
        bool nv_ok = false;
        {
            char bus[32] = {0};
            if (cudaDeviceGetPCIBusId(bus, sizeof(bus), 0) == cudaSuccess &&
                nvmlInit_v2() == NVML_SUCCESS &&
                nvmlDeviceGetHandleByPciBusId_v2(bus, &nd) == NVML_SUCCESS)
                nv_ok = true;
        }
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            size_t z = v.size();
            return z ? (z % 2 ? v[z / 2] : 0.5 * (v[z / 2 - 1] + v[z / 2])) : 0.0;
        };
        struct Sample {
            double t_ms, read_us, e_read_us;
            unsigned clk, mw, temp;
            unsigned long long rs, e_mj;
            int ok;
        };
        struct ESample {
            double t_ms, read_us;
            unsigned long long e_mj;
        };
        struct Res {
            std::vector<double> ms, launch_ms, gs, ge;
            std::vector<Sample> smp;
            std::vector<ESample> esm;
            double energy_j = -1, wall_s = 0;
        };
        const double sus_sec = getenv("L8RT_SUSTAIN_SEC") ? atof(getenv("L8RT_SUSTAIN_SEC")) : 0.0;
        const int cold_sec = getenv("L8RT_COLD_SEC") ? atoi(getenv("L8RT_COLD_SEC")) : 20;
        const double sample_ms = getenv("L8RT_SAMPLE_MS") ? atof(getenv("L8RT_SAMPLE_MS")) : 10.0;
        using clk_t = std::chrono::steady_clock;

        auto sampler = [&](std::vector<Sample> *out, std::atomic<bool> *stop,
                           clk_t::time_point w0) {
            for (long i = 0; nv_ok && !stop->load(std::memory_order_relaxed); ++i) {
                Sample x{};
                unsigned long long e = 0;
                const auto a = clk_t::now();
                x.ok = (nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &x.clk) == NVML_SUCCESS) |
                       (nvmlDeviceGetPowerUsage(nd, &x.mw) == NVML_SUCCESS) << 1 |
                       (nvmlDeviceGetCurrentClocksEventReasons(nd, &x.rs) == NVML_SUCCESS) << 2 |
                       (nvmlDeviceGetTemperature(nd, NVML_TEMPERATURE_GPU, &x.temp) == NVML_SUCCESS)
                           << 4;
                const auto b = clk_t::now();
                (void)e;
                x.t_ms = std::chrono::duration<double, std::milli>(a - w0).count();
                x.read_us = std::chrono::duration<double, std::micro>(b - a).count();
                out->push_back(x);
                std::this_thread::sleep_until(
                    w0 + std::chrono::microseconds((long long)((i + 1) * sample_ms * 1000.0)));
            }
        };

        auto esampler = [&](std::vector<ESample> *out, std::atomic<bool> *stop,
                            clk_t::time_point w0) {
            while (nv_ok && !stop->load(std::memory_order_relaxed)) {
                ESample x{};
                unsigned long long e = 0;
                const auto a = clk_t::now();
                const bool ok = nvmlDeviceGetTotalEnergyConsumption(nd, &e) == NVML_SUCCESS;
                const auto b = clk_t::now();
                if (!ok) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                x.e_mj = e;
                x.t_ms = std::chrono::duration<double, std::milli>(a - w0).count();
                x.read_us = std::chrono::duration<double, std::micro>(b - a).count();
                out->push_back(x);
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
            const auto w0 = clk_t::now();
            std::atomic<bool> stop{false};
            std::thread th(sampler, &r.smp, &stop, w0);
            std::thread te(esampler, &r.esm, &stop, w0);
            r.launch_ms.reserve(R);
            for (int i = 0; i < R; ++i) {
                r.launch_ms.push_back(
                    std::chrono::duration<double, std::milli>(clk_t::now() - w0).count());
                ck(cudaEventRecord(st[i], 0));
                launch();
                ck(cudaEventRecord(en[i], 0));
            }
            ck(cudaEventSynchronize(en[R - 1]));
            r.wall_s = std::chrono::duration<double>(clk_t::now() - w0).count();
            stop = true;
            th.join();
            te.join();
            if (eok && nvmlDeviceGetTotalEnergyConsumption(nd, &e1) == NVML_SUCCESS)
                r.energy_j = (e1 - e0) / 1000.0;
            r.ms.resize(R);
            r.gs.resize(R);
            r.ge.resize(R);
            for (int i = 0; i < R; ++i) {
                float x, a, b;
                ck(cudaEventElapsedTime(&x, st[i], en[i]));
                r.ms[i] = x;
                ck(cudaEventElapsedTime(&a, st[0], st[i]));
                ck(cudaEventElapsedTime(&b, st[0], en[i]));
                r.gs[i] = r.launch_ms[0] + a;
                r.ge[i] = r.launch_ms[0] + b;
            }
            for (int i = 0; i < R; ++i) {
                cudaEventDestroy(st[i]);
                cudaEventDestroy(en[i]);
            }
            return r;
        };

        auto run_for = [&](double secs, auto &&launch) {
            Res r;
            std::vector<cudaEvent_t> st, en;
            unsigned long long e0 = 0, e1 = 0;
            bool eok = nv_ok && nvmlDeviceGetTotalEnergyConsumption(nd, &e0) == NVML_SUCCESS;
            const auto w0 = clk_t::now();
            std::atomic<bool> stop{false};
            std::thread th(sampler, &r.smp, &stop, w0);
            std::thread te(esampler, &r.esm, &stop, w0);
            for (size_t i = 0;; ++i) {
                if (i >= 4)
                    ck(cudaEventSynchronize(en[i - 4]));
                const double now_ms =
                    std::chrono::duration<double, std::milli>(clk_t::now() - w0).count();
                if (i >= 30 && now_ms >= secs * 1000.0)
                    break;
                cudaEvent_t a, b;
                ck(cudaEventCreate(&a));
                ck(cudaEventCreate(&b));
                st.push_back(a);
                en.push_back(b);
                r.launch_ms.push_back(now_ms);
                ck(cudaEventRecord(a, 0));
                launch();
                ck(cudaEventRecord(b, 0));
            }
            ck(cudaEventSynchronize(en.back()));
            r.wall_s = std::chrono::duration<double>(clk_t::now() - w0).count();
            stop = true;
            th.join();
            te.join();
            if (eok && nvmlDeviceGetTotalEnergyConsumption(nd, &e1) == NVML_SUCCESS)
                r.energy_j = (e1 - e0) / 1000.0;
            r.ms.resize(st.size());
            r.gs.resize(st.size());
            r.ge.resize(st.size());
            for (size_t i = 0; i < st.size(); ++i) {
                float x, a, b;
                ck(cudaEventElapsedTime(&x, st[i], en[i]));
                r.ms[i] = x;
                ck(cudaEventElapsedTime(&a, st[0], st[i]));
                ck(cudaEventElapsedTime(&b, st[0], en[i]));
                r.gs[i] = r.launch_ms[0] + a;
                r.ge[i] = r.launch_ms[0] + b;
            }
            for (size_t i = 0; i < st.size(); ++i) {
                cudaEventDestroy(st[i]);
                cudaEventDestroy(en[i]);
            }
            return r;
        };
        auto block = [&](int calls, auto &&launch) {
            return sus_sec > 0 ? run_for(sus_sec, launch) : run_block(calls, launch);
        };
        auto report = [&](const char *who, const char *phase, const Res &r) {
            const size_t R = r.ms.size();
            const double f10 =
                median(std::vector<double>(r.ms.begin(), r.ms.begin() + std::min<size_t>(10, R)));
            const double l30 =
                median(std::vector<double>(r.ms.end() - std::min<size_t>(30, R), r.ms.end()));
            const double wall_ms = r.wall_s * 1000.0, t_tail = wall_ms * 2.0 / 3.0;
            std::vector<double> clk, pw, clk_t2, pw_t2, rd, erd, tail_ms, tmp, tmp_t2;
            size_t rn = 0, cap = 0, rn_t = 0, cap_t = 0;
            const ESample *e_a = nullptr;
            const ESample *e_b = nullptr;
            for (const Sample &x : r.smp) {
                const bool tail = x.t_ms >= t_tail;
                if (x.ok & 1) {
                    clk.push_back(x.clk);
                    if (tail)
                        clk_t2.push_back(x.clk);
                }
                if (x.ok & 2) {
                    pw.push_back(x.mw / 1000.0);
                    if (tail)
                        pw_t2.push_back(x.mw / 1000.0);
                }
                if (x.ok & 4) {
                    ++rn;
                    cap += (x.rs & 0x4ull) != 0;
                    if (tail) {
                        ++rn_t;
                        cap_t += (x.rs & 0x4ull) != 0;
                    }
                }
                if (x.ok & 16) {
                    tmp.push_back(x.temp);
                    if (tail)
                        tmp_t2.push_back(x.temp);
                }
                rd.push_back(x.read_us);
            }
            for (const ESample &x : r.esm) {
                if (x.t_ms >= t_tail) {
                    if (!e_a)
                        e_a = &x;
                    e_b = &x;
                }
                erd.push_back(x.read_us);
            }
            for (size_t i = 0; i < R; ++i)
                if (r.launch_ms[i] >= t_tail)
                    tail_ms.push_back(r.ms[i]);
            double tail_e = -1, tail_win = 0;
            size_t tail_calls_win = 0;
            if (e_a && e_b && e_b != e_a) {
                tail_e = double(e_b->e_mj - e_a->e_mj) / 1000.0;
                tail_win = e_b->t_ms - e_a->t_ms;
                for (size_t i = 0; i < R; ++i)
                    tail_calls_win += r.launch_ms[i] >= e_a->t_ms && r.launch_ms[i] < e_b->t_ms;
            }

            struct Step {
                double t_ms, width_ms;
                unsigned long long e_mj;
            };
            std::vector<Step> cp;
            for (size_t i = 1; i < r.esm.size(); ++i)
                if (r.esm[i].e_mj != r.esm[i - 1].e_mj) {
                    const double a = r.esm[i - 1].t_ms + r.esm[i - 1].read_us / 2000.0,
                                 b = r.esm[i].t_ms + r.esm[i].read_us / 2000.0;
                    cp.push_back({0.5 * (a + b), b - a, r.esm[i].e_mj});
                }
            auto calls_in = [&](double a, double b) {
                double c = 0;
                for (size_t i = 0; i < R; ++i) {
                    const double lo = std::max(a, r.gs[i]), hi = std::min(b, r.ge[i]);
                    if (hi > lo && r.ge[i] > r.gs[i])
                        c += (hi - lo) / (r.ge[i] - r.gs[i]);
                }
                return c;
            };
            struct CpWin {
                double e_j = -1, win_ms = 0, calls = 0, edge_a = -1, edge_b = -1;
                size_t steps = 0;
            };
            auto cp_window = [&](double from) {
                CpWin w;
                size_t a = cp.size();
                for (size_t i = 0; i < cp.size(); ++i)
                    if (cp[i].t_ms >= from) {
                        a = i;
                        break;
                    }
                if (a + 1 >= cp.size())
                    return w;
                const size_t b = cp.size() - 1;
                w.e_j = double(cp[b].e_mj - cp[a].e_mj) / 1000.0;
                w.win_ms = cp[b].t_ms - cp[a].t_ms;
                w.steps = b - a;
                w.calls = calls_in(cp[a].t_ms, cp[b].t_ms);
                w.edge_a = cp[a].width_ms;
                w.edge_b = cp[b].width_ms;
                return w;
            };
            const CpWin cw = cp_window(0.0), ct = cp_window(t_tail);
            auto mx = [](const std::vector<double> &v) {
                return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
            };
            auto mean = [](const std::vector<double> &v) {
                double t = 0;
                for (double x : v)
                    t += x;
                return v.empty() ? 0.0 : t / v.size();
            };
            const double span = r.smp.size() > 1
                                    ? (r.smp.back().t_ms - r.smp.front().t_ms) / (r.smp.size() - 1)
                                    : 0.0;
            printf(
                "{\"kind\":\"sustained\",\"who\":\"%s\",\"phase\":\"%s\",\"reps\":%zu,\"first10_median_ms\":%.6f,"
                "\"last30_median_ms\":%.6f,\"all_median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"last30_over_first10\":%.6f,"
                "\"sm_clock_median_mhz\":%.1f,\"power_median_w\":%.1f,\"nvml_samples\":%zu,\"energy_j\":%.3f,"
                "\"energy_per_call_j\":%.5f,\"wall_s\":%.4f,\"mean_power_w\":%.1f,\"mode\":\"%s\",\"target_s\":%.2f,"
                "\"reason_samples\":%zu,\"sw_power_cap_frac\":%.4f,\"sampling\":\"host thread every %.1f ms\","
                "\"sample_interval_mean_ms\":%.3f,\"sample_read_us_mean\":%.1f,\"sample_read_us_max\":%.1f,"
                "\"energy_samples\":%zu,\"energy_read_us_mean\":%.1f,\"energy_read_us_max\":%.1f,\"tail_from_ms\":%.1f,\"tail_calls\":%zu,"
                "\"tail_median_ms\":%.6f,\"tail_sm_clock_median_mhz\":%.1f,\"tail_power_median_w\":%.1f,\"tail_reason_samples\":%zu,"
                "\"tail_sw_power_cap_frac\":%.4f,\"tail_energy_j\":%.3f,\"tail_energy_window_ms\":%.1f,\"tail_calls_in_window\":%zu,"
                "\"tail_energy_per_call_j\":%.5f,\"tail_mean_power_w\":%.1f,\"temp_first_c\":%.0f,\"temp_median_c\":%.1f,"
                "\"tail_temp_median_c\":%.1f,\"energy_steps\":%zu,\"energy_cp_j\":%.3f,\"energy_cp_window_ms\":%.1f,"
                "\"energy_cp_steps\":%zu,\"energy_cp_calls\":%.2f,\"energy_cp_per_call_j\":%.6f,\"energy_cp_power_w\":%.1f,"
                "\"tail_cp_j\":%.3f,\"tail_cp_window_ms\":%.1f,\"tail_cp_steps\":%zu,\"tail_cp_calls\":%.2f,"
                "\"tail_cp_per_call_j\":%.6f,\"tail_cp_power_w\":%.1f,\"energy_cp_edge_a_ms\":%.2f,\"energy_cp_edge_b_ms\":%.2f,"
                "\"tail_cp_edge_a_ms\":%.2f,\"tail_cp_edge_b_ms\":%.2f}\n",
                who, phase, R, f10, l30, median(r.ms), *std::min_element(r.ms.begin(), r.ms.end()),
                *std::max_element(r.ms.begin(), r.ms.end()), l30 / f10, median(clk), median(pw),
                r.smp.size(), r.energy_j, r.energy_j >= 0 ? r.energy_j / R : -1.0, r.wall_s,
                r.energy_j >= 0 ? r.energy_j / r.wall_s : -1.0, sus_sec > 0 ? "seconds" : "calls",
                sus_sec, rn, rn ? double(cap) / rn : -1.0, sample_ms, span, mean(rd), mx(rd),
                r.esm.size(), mean(erd), mx(erd), t_tail, tail_ms.size(), median(tail_ms),
                median(clk_t2), median(pw_t2), rn_t, rn_t ? double(cap_t) / rn_t : -1.0, tail_e,
                tail_win, tail_calls_win,
                tail_e >= 0 && tail_calls_win ? tail_e / tail_calls_win : -1.0,
                tail_e >= 0 && tail_win > 0 ? tail_e / (tail_win / 1000.0) : -1.0,
                tmp.empty() ? -1.0 : tmp.front(), median(tmp), median(tmp_t2), cp.size(), cw.e_j,
                cw.win_ms, cw.steps, cw.calls,
                cw.e_j >= 0 && cw.calls > 0 ? cw.e_j / cw.calls : -1.0,
                cw.e_j >= 0 && cw.win_ms > 0 ? cw.e_j / (cw.win_ms / 1000.0) : -1.0, ct.e_j,
                ct.win_ms, ct.steps, ct.calls,
                ct.e_j >= 0 && ct.calls > 0 ? ct.e_j / ct.calls : -1.0,
                ct.e_j >= 0 && ct.win_ms > 0 ? ct.e_j / (ct.win_ms / 1000.0) : -1.0, cw.edge_a,
                cw.edge_b, ct.edge_a, ct.edge_b);
            printf(
                "{\"kind\":\"nvml_series\",\"who\":\"%s\",\"phase\":\"%s\",\"fields\":\"t_ms,sm_mhz,power_mw,reasons,read_us,ok_bits,temp_c\","
                "\"samples\":[",
                who, phase);
            for (size_t i = 0; i < r.smp.size(); ++i) {
                const Sample &x = r.smp[i];
                printf("%s[%.2f,%u,%u,%llu,%.0f,%d,%u]", i ? "," : "", x.t_ms, x.clk, x.mw, x.rs,
                       x.read_us, x.ok, x.temp);
            }
            printf("]}\n");
            printf(
                "{\"kind\":\"energy_series\",\"who\":\"%s\",\"phase\":\"%s\",\"fields\":\"t_ms,energy_mj_from_first,read_us\",\"samples\":[",
                who, phase);
            for (size_t i = 0; i < r.esm.size(); ++i)
                printf("%s[%.2f,%llu,%.0f]", i ? "," : "", r.esm[i].t_ms,
                       r.esm[i].e_mj - r.esm.front().e_mj, r.esm[i].read_us);
            printf("]}\n");
        };
        auto ours = [&] {
            if (plan.run(A, B, C, 0, nullptr, reuse))
                throw std::runtime_error("sustained call failed");
        };
        if (sus == "full") {
            cublasHandle_t h2;
            if (cublasCreate(&h2) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cublasCreate");
            int ver = 0;
            cublasGetVersion(h2, &ver);
            if (ver < 130800) {
                printf(
                    "{\"kind\":\"sustained\",\"status\":\"skipped_version\",\"cublas_version\":%d}\n",
                    ver);
            } else {
                double *Ce;
                ck(cudaMalloc(&Ce, cc * 8));
                const size_t ewsb = size_t(4) << 30;
                void *ews;
                ck(cudaMalloc(&ews, ewsb));
                const double one = 1.0, zero = 0.0;
                cublasSetMathMode(h2, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
                cublasSetEmulationStrategy(h2, CUBLAS_EMULATION_STRATEGY_EAGER);
                cublasSetFixedPointEmulationMantissaControl(h2,
                                                            CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
                cublasSetFixedPointEmulationMaxMantissaBitCount(h2, 53);
                cublasSetWorkspace(h2, ews, ewsb);
                auto cub = [&] {
                    if (cublasDgemm(h2, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, int(lda), B,
                                    int(ldb), &zero, Ce, m) != CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("cublasDgemm");
                };
                cub();
                ck(cudaDeviceSynchronize());
                const std::string order =
                    getenv("L8RT_ORDER") ? getenv("L8RT_ORDER") : "cublas_first";
                const double gap_sec = getenv("L8RT_GAP_SEC") ? atof(getenv("L8RT_GAP_SEC")) : 5.0;
                if (order != "cublas_first" && order != "ours_first")
                    throw std::runtime_error("L8RT_ORDER: cublas_first or ours_first");
                printf(
                    "{\"kind\":\"sustained_order\",\"order\":\"%s\",\"cold_s\":%d,\"gap_s\":%.2f}\n",
                    order.c_str(), cold_sec, gap_sec);
                std::this_thread::sleep_for(std::chrono::seconds(cold_sec));
                const auto gap = std::chrono::microseconds((long long)(gap_sec * 1e6));
                if (order == "cublas_first") {
                    const Res rc = block(300, cub);
                    report("cublas_fixed53", "first_from_cold", rc);
                    std::this_thread::sleep_for(gap);
                    const Res ro = block(300, ours);
                    report("ours", "second_after_cublas", ro);
                } else {
                    const Res ro = block(300, ours);
                    report("ours", "first_from_cold", ro);
                    std::this_thread::sleep_for(gap);
                    const Res rc = block(300, cub);
                    report("cublas_fixed53", "second_after_ours", rc);
                }
                printf("{\"kind\":\"cublas_version\",\"cublas_version\":%d}\n", ver);
                cudaFree(ews);
                cudaFree(Ce);
            }
            cublasDestroy(h2);
        } else {
            const Res ro = block(50, ours);
            report("ours", sus_sec > 0 ? "short_seconds" : "short_50", ro);
        }
        printf("{\"kind\":\"sustained_verdict\",\"ours_hash\":\"%016llx\"}\n", out_hash());
        if (nv_ok)
            nvmlShutdown();
    }
    printf("{\"kind\":\"reuse\",\"mask\":%u,\"reused_a\":%llu,\"reused_b\":%llu}\n", reuse,
           plan.reused_a, plan.reused_b);
    printf("{\"kind\":\"pace\",\"pace_s\":%d,\"timeouts_total\":%u}\n", plan.sh.pace_s,
           plan.pace_timeouts());
    plan.destroy();
    return 0;
} catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
}
