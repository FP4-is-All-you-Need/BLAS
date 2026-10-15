

#include "gen_mc.h"
#include <chrono>
static long long l8f_wall_ms() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
__global__ void fill_mc(float *x, size_t n, unsigned long long stream) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        x[i] = gen_val(stream, i);
    }
}
__global__ void gather_bits(const float *c, const int *pos, int count, long long ldc,
                            unsigned *out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) {
        out[t] = __float_as_uint(c[size_t(pos[2 * t + 1]) * ldc + pos[2 * t]]);
    }
}
__global__ void to_f32(const double *x, float *y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        y[i] = __double2float_rn(x[i]);
    }
}
__global__ void xor_hash_f(const float *a, size_t n, unsigned long long *out) {
    unsigned long long h = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        h ^= (unsigned long long)__float_as_uint(a[i]) * 0x9E3779B97F4A7C15ull +
             i * 0xD1342543DE82EF95ull;
    }
    for (int o = 16; o; o >>= 1) {
        h ^= __shfl_down_sync(0xffffffffu, h, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicXor(out, h);
    }
}
static unsigned long long xor_hash_of_f(const float *a, size_t n) {
    unsigned long long *d;
    ck(cudaMalloc(&d, 8));
    ck(cudaMemset(d, 0, 8));
    xor_hash_f<<<4096, 256>>>(a, n, d);
    ck(cudaGetLastError());
    unsigned long long h = 0;
    ck(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
    cudaFree(d);
    return h;
}

int main(int argc, char **argv) try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const int runs = argc > 1 ? atoi(argv[1]) : 4; // the first run is the warm-up
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || (prop.minor != 3 && prop.minor != 0)) {
        throw std::runtime_error("requires B200 sm_100a or B300 sm_103a (dense K64)");
    }
    int runtime = 0;
    ck(cudaRuntimeGetVersion(&runtime));
    int m = 8192, n = 8192, k = 8192;
    if (const char *zs = getenv("L8SHAPE")) {
        if (sscanf(zs, "%d,%d,%d", &m, &n, &k) != 3) {
            throw std::runtime_error("L8SHAPE must be m,n,k (packed)");
        }
    }
    const long long lda = m, ldb = k, ldc = m;
    const std::string input = getenv("L8F_INPUT") ? getenv("L8F_INPUT") : "b";
    if (input != "a" && input != "b") {
        throw std::runtime_error("L8F_INPUT must be a (gen.h) or b (fill_normal float)");
    }
    std::vector<int> pos; // L8F_SAMPLES_POS: (i, j) pairs, read and range-checked before any timing
    if (const char *sp = getenv("L8F_SAMPLES_POS")) {
        FILE *f = fopen(sp, "rb");
        if (!f) {
            throw std::runtime_error("L8F_SAMPLES_POS: cannot open");
        }
        int v[2];
        while (fread(v, 4, 2, f) == 2) {
            pos.push_back(v[0]);
            pos.push_back(v[1]);
        }
        fclose(f);
        for (size_t t = 0; t < pos.size(); t += 2) {
            if (pos[t] < 0 || pos[t] >= m || pos[t + 1] < 0 || pos[t + 1] >= n) {
                throw std::runtime_error("L8F_SAMPLES_POS: position outside C");
            }
        }
        if (pos.empty()) {
            throw std::runtime_error("L8F_SAMPLES_POS: no positions");
        }
    }
    l8rt::Plan plan;
    plan.init(m, n, k, lda, ldb, ldc); // throws before any timing when a host check fails
    const size_t ca = size_t(m) * k, cb = size_t(k) * n, cc = size_t(m) * n;
    printf(
        "{\"kind\":\"shape\",\"m\":%d,\"n\":%d,\"k\":%d,\"k_plan\":%d,\"cuts\":%d,\"tile_rows\":%d,\"tile_cols\":%d,"
        "\"tiles\":%d,\"clusters\":%d,\"nchunk\":%d,\"whole\":%d,\"crt_chunks\":%d,\"crt_after_gemm\":%s}\n",
        m, n, k, plan.kplan, plan.sh.cuts, plan.sh.tile_rows, plan.sh.tile_cols, plan.sh.tiles,
        plan.sh.clusters, plan.sh.nchunk, plan.sh.whole, plan.crt_chunks,
        plan.sh.nchunk > 1 ? "true" : "false");
    printf(
        "{\"kind\":\"environment\",\"engine\":\"l8f\",\"precision\":\"fp32\",\"gpu\":\"%s\",\"cuda_runtime\":%d,\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"bits\":%d,\"moduli\":%d,\"planes_per_side\":%d,\"mma_products\":%d,\"accumulators_per_tile\":%d,\"k_plan\":%d,"
        "\"P_bits\":%d,\"encoder\":\"%s\",\"enc_minb\":%d,\"pace_s\":%d,\"item_order\":%d,\"pingpong\":%d,\"input\":\"%s\"}\n",
        prop.name, runtime, m, n, k, plan.bits, L8K_NMOD, L8K_PLANES59, L8K_PRODUCTS59, plan.accs,
        plan.kplan,
        ((plan.P >> 64) ? 128 - __builtin_clzll((unsigned long long)(plan.P >> 64))
                        : 64 - __builtin_clzll((unsigned long long)plan.P)),
        L8_FAST_ENC ? "fast" : "base",
#if L8_FAST_ENC
        plan.enc.minb,
#else
        -1,
#endif
        plan.sh.pace_s, plan.sh.item_order, plan.sh.pingpong, input.c_str());
    {
        cudaFuncAttributes ka{};
        ck(cudaFuncGetAttributes(&ka, l8k_gemm_kernel_rt<true>));
        printf(
            "{\"kind\":\"kernel\",\"registers\":%d,\"static_shared\":%zu,\"dynamic_shared\":%d}\n",
            ka.numRegs, ka.sharedSizeBytes, kShared);
    }
    float *A, *B, *C;
    double *tmp;
    ck(cudaMalloc(&A, ca * 4));
    ck(cudaMalloc(&B, cb * 4));
    ck(cudaMalloc(&C, cc * 4));
    if (input == "b") {
        ck(cudaMalloc(&tmp, std::max(ca, cb) * 8));
        fill_normal<<<4096, 256>>>(tmp, ca, 20260921ull);
        to_f32<<<4096, 256>>>(tmp, A, ca);
        fill_normal<<<4096, 256>>>(tmp, cb, 76543210ull);
        to_f32<<<4096, 256>>>(tmp, B, cb);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        cudaFree(tmp);
    } else {
        fill_mc<<<4096, 256>>>(A, ca, 1ull);
        fill_mc<<<4096, 256>>>(B, cb, 2ull);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
    }
    printf("{\"kind\":\"inputs\",\"a_hash\":\"%016llx\",\"b_hash\":\"%016llx\"}\n",
           xor_hash_of_f(A, ca), xor_hash_of_f(B, cb));
    double best = 1e30;
    for (int r = 0; r < runs; ++r) {
        ck(cudaMemsetAsync(C, 0xa5, cc * 4,
                           0)); // poison (not timed): every element must be written
        ck(cudaMemsetAsync(plan.pa, 0xa5, plan.bytesA, 0));
        ck(cudaMemsetAsync(plan.pb, 0xa5, plan.bytesB, 0));
        ck(cudaMemsetAsync(plan.resid, 0xa5, plan.resid_bytes, 0));
        ck(cudaMemsetAsync(plan.km, 0, size_t(plan.sh.tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(plan.mm, 0, size_t(plan.sh.tiles) * kParts * 2 * sizeof(Metrics), 0));
        float st[3] = {0, 0, 0};
        if (plan.run(A, B, C, 0, st, 0u) == 2) {
            throw std::runtime_error("input not finite");
        }
        const float tot = st[0] + st[1] + st[2];
        printf(
            "{\"kind\":\"ours\",\"run\":%d,\"warmup\":%s,\"total_ms\":%.6f,\"quantize_ms\":%.6f,\"encode_ms\":%.6f,\"gemm_ms\":%.6f,"
            "\"crt_ms\":%.6f}\n",
            r, r == 0 ? "true" : "false", tot, st[0], st[1], st[2], plan.last_crt_ms);
        if (r) {
            best = std::min<double>(best, tot);
        }
    }
    printf("{\"kind\":\"output\",\"ours_hash\":\"%016llx\",\"ours_best_ms\":%.6f}\n",
           xor_hash_of_f(C, cc), best);
    {
        // payout apart from the others (the last part of a whole tile also runs the CRT and the rounding to float)
        const int Tiles = plan.sh.tiles;
        std::vector<KernelMetrics> hkm(size_t(Tiles) * 2);
        std::vector<Metrics> hmm(size_t(Tiles) * kParts * 2);
        ck(cudaMemcpy(hkm.data(), plan.km, hkm.size() * sizeof(KernelMetrics),
                      cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(hmm.data(), plan.mm, hmm.size() * sizeof(Metrics), cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> tot;
        tot.reserve(Tiles);
        for (int i = 0; i < Tiles; ++i) {
            tot.push_back(hkm[size_t(i) * 2].total);
        }
        std::sort(tot.begin(), tot.end());
        unsigned long long core = 0, aw = 0, bw = 0, iss = 0, pay = 0, pread = 0, ns = 0,
                           pay_last = 0, pay_other = 0;
        for (size_t i = 0; i < hmm.size(); i += 2) {
            core += hmm[i].core;
            aw += hmm[i].a_wait;
            bw += hmm[i].b_wait;
            iss += hmm[i].issue;
            pay += hmm[i].payout;
            pread += hmm[i].payout_read;
            ns += hmm[i].ns;
            ((i / 2) % kParts == size_t(kParts - 1) ? pay_last : pay_other) += hmm[i].payout;
        }
        const double nn = double(Tiles) * kParts;
        printf(
            "{\"kind\":\"cycles\",\"tiles\":%d,\"parts\":%d,\"tile_total_median\":%llu,\"per_modulus_core_mean\":%.1f,"
            "\"a_wait_mean\":%.1f,\"b_wait_mean\":%.1f,\"issue_mean\":%.1f,\"payout_mean\":%.1f,\"payout_read_mean\":%.1f,"
            "\"payout_last_part_mean\":%.1f,\"payout_other_parts_mean\":%.1f,\"mma_bound_per_modulus\":%d,\"core_ghz\":%.4f}\n",
            Tiles, kParts, tot[Tiles / 2], core / nn, aw / nn, bw / nn, iss / nn, pay / nn,
            pread / nn, pay_last / double(Tiles),
            kParts > 1 ? pay_other / (double(Tiles) * (kParts - 1)) : 0.0,
            plan.sh.cuts * L8K_PRODUCTS59 * 112 / L8K_NMOD, ns ? double(core) / double(ns) : 0.0);
    }
    if (!pos.empty()) {
        const int count = int(pos.size() / 2);
        int *dpos;
        unsigned *dout;
        ck(cudaMalloc(&dpos, pos.size() * 4));
        ck(cudaMalloc(&dout, size_t(count) * 4));
        ck(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
        gather_bits<<<(count + 255) / 256, 256>>>(C, dpos, count, ldc, dout);
        ck(cudaGetLastError());
        std::vector<unsigned> hb(count);
        ck(cudaMemcpy(hb.data(), dout, size_t(count) * 4, cudaMemcpyDeviceToHost));
        FILE *f = fopen("samples_c.bin", "wb");
        if (!f || fwrite(hb.data(), 4, count, f) != size_t(count)) {
            throw std::runtime_error("cannot write samples_c.bin");
        }
        fclose(f);
        unsigned long long h = 0;
        for (int t = 0; t < count; ++t) {
            h ^= (unsigned long long)hb[t] * 0x9E3779B97F4A7C15ull +
                 (unsigned long long)t * 0xD1342543DE82EF95ull;
        }
        printf("{\"kind\":\"samples\",\"count\":%d,\"samples_hash\":\"%016llx\"}\n", count, h);
        cudaFree(dpos);
        cudaFree(dout);
    }
    if (const char *su = getenv("L8F_SUSTAIN_MS")) {
        const int sustain = atoi(su),
                  batch_ms = getenv("L8F_BATCH_MS") ? atoi(getenv("L8F_BATCH_MS")) : 20;
        if (sustain < 1 || batch_ms < 1 || !(best > 0 && best < 1e29)) {
            throw std::runtime_error("L8F_SUSTAIN_MS / L8F_BATCH_MS >= 1 after timed runs");
        }
        const int batch = std::max(1, int(batch_ms / best + 0.5));
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0));
        ck(cudaEventCreate(&e1));
        long long launches = 0;
        const long long w0 = l8f_wall_ms();
        ck(cudaEventRecord(e0, 0));
        while (l8f_wall_ms() - w0 < sustain) {
            for (int r = 0; r < batch; ++r) {
                if (plan.run(A, B, C, 0, nullptr, 0u) != 0) {
                    throw std::runtime_error("run failed in the window");
                }
            }
            ck(cudaDeviceSynchronize());
            launches += batch;
        }
        ck(cudaEventRecord(e1, 0));
        ck(cudaEventSynchronize(e1));
        float gms = 0;
        ck(cudaEventElapsedTime(&gms, e0, e1));
        const long long w1 = l8f_wall_ms();
        printf(
            "{\"kind\":\"sustain\",\"config\":\"ours_%s\",\"launches\":%lld,\"batch\":%d,\"wall_start_ms\":%lld,\"wall_end_ms\":%lld,"
            "\"gpu_ms\":%.3f,\"gpu_ms_kind\":\"one_event_pair_around_window\",\"wall_per_call_ms\":%.6f}\n",
            input.c_str(), launches, batch, w0, w1, gms, gms / double(launches));
        printf("{\"kind\":\"output_after_sustain\",\"ours_hash\":\"%016llx\"}\n",
               xor_hash_of_f(C, cc));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    plan.destroy();
    return 0;
} catch (const std::exception &e) {
    printf("{\"kind\":\"error\",\"what\":\"%s\"}\n", e.what());
    return 2;
}
