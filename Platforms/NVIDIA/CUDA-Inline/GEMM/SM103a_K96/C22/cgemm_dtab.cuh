// FP32 C ABI around the copied ZGEMM Plan, using its modular 2M / 3M encoder.
#pragma once
#include <memory>
#ifndef CGEMM_MMA_K
#define CGEMM_MMA_K 64
#endif
namespace cl8rt {
thread_local std::string error;
struct Ctx {
    l8rt::Plan plan;
    std::string info;
    int poison = 1;
    ~Ctx() {
        plan.destroy();
    }
};
} // namespace cl8rt
extern "C" const char *c_last_error() {
    return cl8rt::error.c_str();
}
extern "C" void *c_create_ld(int m, int n, int k, long long lda, long long ldb, long long ldc) try {
    cl8rt::error.clear();
    cudaDeviceProp prop{};
    int dev = 0;
    ck(cudaGetDevice(&dev));
    ck(cudaGetDeviceProperties(&prop, dev));
    if (prop.major != 10 || (prop.minor != 0 && prop.minor != 3)) {
        throw std::runtime_error("B200/B300 required");
    }
    auto c = std::make_unique<cl8rt::Ctx>();
    c->plan.init(m, n, k, lda, ldb, ldc);
    const auto &p = c->plan;
    char text[1600];
    snprintf(
        text, sizeof text,
        "{\"engine\":\"cgemm_l8\",\"precision\":\"complex-fp32\",\"sm\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,\"bits\":%d,\"moduli\":%d,\"parts\":%d,\"mma_products\":%d,\"planes_per_side\":%d,\"cuts\":%d,\"k_plan\":%d,\"mma_k\":%d,\"reference_checked\":false}",
        prop.major * 10 + prop.minor, m, n, k, lda, ldb, ldc, p.bits, ZL8_NMODC, L8K_NMOD,
        L8K_PRODUCTS59, L8K_PLANES59, p.sh.cuts, p.kplan, CGEMM_MMA_K);
    c->info = text;
    return c.release();
} catch (const std::exception &e) {
    cl8rt::error = e.what();
    return nullptr;
}
extern "C" const char *c_info(void *h) {
    return h ? static_cast<cl8rt::Ctx *>(h)->info.c_str() : "";
}
extern "C" void *c_create() {
    return c_create_ld(8192, 8192, 8192, 8192, 8192, 8192);
}
extern "C" int c_run(void *h, const void *A, const void *B, void *C, int runs, double *stage) try {
    cl8rt::error.clear();
    if (!h || !A || !B || !C || !stage || runs < 1) {
        throw std::runtime_error("invalid run arguments");
    }
    auto &c = *static_cast<cl8rt::Ctx *>(h);
    auto &p = c.plan;
    for (int r = 0; r < runs; ++r) {
        if (c.poison) {
            ck(cudaMemset2DAsync(C, size_t(p.ldc) * 8, 0xa5, size_t(p.m) * 8, p.n, 0));
            ck(cudaMemsetAsync(p.pa, 0xa5, p.bytesA, 0));
            ck(cudaMemsetAsync(p.pb, 0xa5, p.bytesB, 0));
            ck(cudaMemsetAsync(p.resid, 0xa5, p.resid_bytes, 0));
        }
        ck(cudaMemsetAsync(p.km, 0, size_t(p.sh.tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(p.mm, 0, size_t(p.sh.tiles) * kParts * 2 * sizeof(Metrics), 0));
        float st[3] = {};
        if (p.run(static_cast<const float *>(A), static_cast<const float *>(B),
                  static_cast<float *>(C), 0, st, 0u) == 2) {
            throw std::runtime_error("nonfinite or unrepresentable input");
        }
        stage[6 * r] = double(st[0]) + st[1] + st[2];
        stage[6 * r + 1] = st[0];
        stage[6 * r + 2] = stage[6 * r + 3] = 0;
        stage[6 * r + 4] = st[1];
        stage[6 * r + 5] = st[2];
    }
    return 0;
} catch (const std::exception &e) {
    cl8rt::error = e.what();
    return 1;
}
extern "C" int c_poison(void *h, int on) {
    if (!h) {
        return -1;
    }
    auto &c = *static_cast<cl8rt::Ctx *>(h);
    int prev = c.poison;
    c.poison = on ? 1 : 0;
    return prev;
}
extern "C" void c_destroy(void *h) {
    delete static_cast<cl8rt::Ctx *>(h);
}
