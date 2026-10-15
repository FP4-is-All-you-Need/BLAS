

#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include "zverify.cuh"

namespace zl8rt {
static std::string last_error;
struct Ctx {
    l8rt::Plan plan;
    std::string info;
    int poison = 1;
    long long *q[4] = {nullptr, nullptr, nullptr, nullptr};
};
} // namespace zl8rt

extern "C" const char *z_last_error() {
    return zl8rt::last_error.c_str();
}

extern "C" void *z_create_ld(int m, int n, int k, long long lda, long long ldb, long long ldc) try {
    zl8rt::last_error.clear();
    cudaDeviceProp prop{};
    int device = 0;
    ck(cudaGetDevice(&device));
    ck(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 10 || prop.minor != 3) {
        throw std::runtime_error("requires B300 sm_103a");
    }
    int runtime = 0;
    ck(cudaRuntimeGetVersion(&runtime));
    if (const char *s = awe_additional_env("AWE_FP64_BITS")) {
        if (atoi(s) != 53) {
            throw std::runtime_error("z_create: this library is built for 53 bits");
        }
    }
    auto *c = new zl8rt::Ctx;
    try {
        c->plan.init(m, n, k, lda, ldb, ldc);
    } catch (...) {
        delete c;
        throw;
    }
    const l8rt::Plan &p = c->plan;
    int pbits = 0;
    {
        U128 z = p.P;
        while (z) {
            ++pbits;
            z >>= 1;
        }
    }
    char buf[2048];
    snprintf(
        buf, sizeof buf,
        "{\"gpu\":\"%s\",\"cuda_runtime\":%d,\"complex\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,"
        "\"quantize_bits\":53,\"engine\":\"zl8 (the complex GEMM on the L8 DGEMM kernel)\",\"moduli\":%d,\"parts\":%d,"
        "\"mma_products\":%d,\"planes_per_side\":%d,\"k_plan\":%d,\"accumulators_per_tile\":%d,\"product_bits\":%d,\"cuts\":%d,"
        "\"tiles\":%d,\"clusters\":%d,\"nchunk\":%d,\"herk\":%d,\"pingpong\":%d,\"pace_s\":%d,\"tile_m\":%d,\"tile_n\":%d,\"cta_group\":2,"
        "\"exactness_scope\":\"all_inputs_at_bits\",\"k_max\":16384,\"set\":\"CRT engine with the selected residue tables.\","
        "\"encoder_kernel\":\"%s\","
        "\"source\":\"CRT engine with the selected residue tables.\"}",
        prop.name, runtime, m, n, k, lda, ldb, ldc, ZL8_NMODC, kParts, L8K_PRODUCTS59, L8K_PLANES59,
        p.kplan, p.accs, pbits, p.sh.cuts, p.sh.tiles, p.sh.clusters, p.sh.nchunk, p.sh.herk,
        p.sh.pingpong, p.sh.pace_s, Side, Cols, p.enc.fast ? "encode_zl8f" : "encode_zl8");
    c->info = buf;
    return c;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return nullptr;
}

extern "C" void *z_create() {
    return z_create_ld(8192, 8192, 8192, 8192, 8192, 8192);
}

extern "C" const char *z_info(void *h) {
    return h ? static_cast<zl8rt::Ctx *>(h)->info.c_str() : "";
}

extern "C" int z_fill_normal(void *x, long long n_doubles, unsigned long long seed) try {
    zl8rt::last_error.clear();
    fill_normal<<<4096, 256>>>(reinterpret_cast<double *>(x), size_t(n_doubles), seed);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    return 0;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return 1;
}

extern "C" int z_run_reuse(void *h, const void *A, const void *B, void *C, int runs, double *stage,
                           unsigned reuse) try {
    zl8rt::last_error.clear();
    auto *c = static_cast<zl8rt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("z_run: no context");
    }
    if (reuse) {
        throw std::runtime_error("z_run_reuse: reuse is not built in the zl8 library");
    }
    l8rt::Plan &p = c->plan;
    for (int r = 0; r < runs; ++r) {
        if (c->poison) {
            ck(cudaMemset2DAsync(C, size_t(p.ldc) * 16, 0xa5, size_t(p.m) * 16, p.n, 0));
            ck(cudaMemsetAsync(p.pa, 0xa5, p.bytesA, 0));
            ck(cudaMemsetAsync(p.pb, 0xa5, p.bytesB, 0));
            ck(cudaMemsetAsync(p.resid, 0xa5, p.resid_bytes, 0));
        }
        ck(cudaMemsetAsync(p.km, 0, size_t(p.sh.tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(p.mm, 0, size_t(p.sh.tiles) * kParts * 2 * sizeof(Metrics), 0));
        float st[3] = {0, 0, 0};
        if (p.run(reinterpret_cast<const double *>(A), reinterpret_cast<const double *>(B),
                  reinterpret_cast<double *>(C), 0, st, 0u) == 2) {
            throw std::runtime_error("input out of range (Inf / NaN or beyond the shift)");
        }
        stage[6 * r + 0] = double(st[0]) + st[1] + st[2];
        stage[6 * r + 1] = st[0];
        stage[6 * r + 2] = 0;
        stage[6 * r + 3] = 0;
        stage[6 * r + 4] = st[1];
        stage[6 * r + 5] = st[2];
    }
    return 0;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return 1;
}

extern "C" int z_run(void *h, const void *A, const void *B, void *C, int runs, double *stage) {
    return z_run_reuse(h, A, B, C, runs, stage, 0u);
}

extern "C" int z_hash(const void *x, long long n_doubles, unsigned long long *total,
                      unsigned long long *blocks) try {
    zl8rt::last_error.clear();
    unsigned long long hb[zver::NB];
    *total = zver::hash_array_any(reinterpret_cast<const double *>(x), size_t(n_doubles), hb);
    if (blocks) {
        memcpy(blocks, hb, sizeof(hb));
    }
    return 0;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return 1;
}

extern "C" int z_verify(void *h, const void *A, const void *B, const void *C, int npts,
                        unsigned long long *total, unsigned long long *blocks, int *spot_bad) try {
    zl8rt::last_error.clear();
    auto *c = static_cast<zl8rt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("z_verify: no context");
    }
    const l8rt::Plan &p = c->plan;
    zver::quantize_both_rt(reinterpret_cast<const double2 *>(A),
                           reinterpret_cast<const double2 *>(B), p.ea, p.eb, c->q, p.m, p.n, p.k,
                           p.lda, p.ldb);
    *spot_bad = zver::spot_check_rt(c->q, p.ea, p.eb, reinterpret_cast<const double *>(C),
                                    npts <= 0 ? 16 : npts, p.m, p.n, p.k, p.ldc);
    const double *hsrc = reinterpret_cast<const double *>(C);
    double *tmp = nullptr;
    if (p.ldc != p.m) {
        ck(cudaMalloc(&tmp, size_t(p.m) * p.n * 16));
        ck(cudaMemcpy2D(tmp, size_t(p.m) * 16, C, size_t(p.ldc) * 16, size_t(p.m) * 16, p.n,
                        cudaMemcpyDeviceToDevice));
        hsrc = tmp;
    }
    unsigned long long hb[zver::NB];
    *total = zver::hash_array_any(hsrc, size_t(p.m) * p.n * 2, hb);
    if (blocks) {
        memcpy(blocks, hb, sizeof(hb));
    }
    if (tmp) {
        cudaFree(tmp);
    }
    return 0;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return 1;
}

extern "C" int z_cycles(void *h, double *out) try {
    zl8rt::last_error.clear();
    auto *c = static_cast<zl8rt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("z_cycles: no context");
    }
    const l8rt::Plan &p = c->plan;
    const int Tl = p.sh.tiles;
    std::vector<KernelMetrics> hkm(size_t(Tl) * 2);
    std::vector<Metrics> hmm(size_t(Tl) * kParts * 2);
    ck(cudaMemcpy(hkm.data(), p.km, hkm.size() * sizeof(KernelMetrics), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(hmm.data(), p.mm, hmm.size() * sizeof(Metrics), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> tot;
    tot.reserve(Tl);
    for (int i = 0; i < Tl; ++i) {
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
    const double nn = double(Tl) * kParts;
    out[0] = double(tot[Tl / 2]);
    out[1] = core / nn;
    out[2] = aw / nn;
    out[3] = bw / nn;
    out[4] = iss / nn;
    out[5] = pay / nn;
    out[6] = pread / nn;
    out[7] = double(p.sh.cuts) * L8K_PRODUCTS59 * 112.0 / kParts;
    return 0;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return 1;
}

extern "C" int z_poison(void *h, int on) {
    auto *c = static_cast<zl8rt::Ctx *>(h);
    if (!c) {
        return -1;
    }
    const int was = c->poison;
    c->poison = on ? 1 : 0;
    return was;
}

extern "C" int z_pace(void *h, int *pace_s, unsigned *timeouts) try {
    zl8rt::last_error.clear();
    auto *c = static_cast<zl8rt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("z_pace: no context");
    }
    *pace_s = c->plan.sh.pace_s;
    *timeouts = c->plan.pace_timeouts();
    return 0;
} catch (const std::exception &e) {
    zl8rt::last_error = e.what();
    return 1;
}

extern "C" void z_destroy(void *h) {
    auto *c = static_cast<zl8rt::Ctx *>(h);
    if (!c) {
        return;
    }
    for (auto &q : c->q) {
        if (q) {
            cudaFree(q);
            q = nullptr;
        }
    }
    c->plan.destroy();
    delete c;
}
