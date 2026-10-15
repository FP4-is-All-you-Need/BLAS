#include <climits>
#include "k64_trsm.cuh"
#include "l8k_tables_rt.h"
namespace {
thread_local std::string last_error;
struct Context {
    Engine engine;
    BLAS blas;
    K64TrsmPlan plan;
    int ldb, ldc;
    Context(int m, int n, int lda, int ldb, int ldc)
        : engine(), plan(engine, blas, m, n, 2048, lda, ldc), ldb(ldb), ldc(ldc) {}
};
} // namespace
extern "C" void *t_create_configured(int m, int n, int k, long long lda, long long ldb,
                                     long long ldc, int operation,
                                     const awe_engine_configuration *config) try {
    last_error.clear();
    if (!awe_configuration_supported(config) || operation != 10 || k != m || m < 1 || n < 1 ||
        m > 4096 || n > 2048 || lda < m || ldb < m || ldc < m || lda > INT_MAX || ldb > INT_MAX ||
        ldc > INT_MAX) {
        throw std::invalid_argument("unsupported triangular solve shape, strides or options");
    }
    return new Context(m, n, int(lda), int(ldb), int(ldc));
} catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
}
extern "C" int t_run(void *h, const void *ap, const void *bp, void *cp, int runs,
                     double *stage) try {
    if (!h || !ap || !bp || !cp || !stage || runs < 1) {
        throw std::invalid_argument("invalid triangular solve arguments");
    }
    auto &c = *static_cast<Context *>(h);
    for (int r = 0; r < runs; ++r) {
        const auto start = Clock::now();
        ck(cudaMemcpy2DAsync(cp, size_t(c.ldc) * sizeof(double), bp, size_t(c.ldb) * sizeof(double),
                             size_t(c.plan.m) * sizeof(double), c.plan.n,
                             cudaMemcpyDeviceToDevice));
        c.plan.run(static_cast<const double *>(ap), static_cast<double *>(cp));
        for (int q = 0; q < 6; ++q) {
            stage[r * 6 + q] = 0;
        }
        stage[r * 6] = ms(start);
    }
    return 0;
} catch (const std::exception &e) {
    last_error = e.what();
    return 1;
}
extern "C" void t_destroy(void *h) {
    delete static_cast<Context *>(h);
}
extern "C" const char *t_last_error() {
    return last_error.c_str();
}
extern "C" int t_poison(void *h, int on) {
    if (!h) {
        return -1;
    }
    for (auto &p : static_cast<Context *>(h)->plan.plans) {
        if (p->e.poison(p->h, on) < 0) {
            return -1;
        }
    }
    return 0;
}
extern "C" int t_effective_options(void *h, awe_engine_effective *out) {
    if (!h || !out) {
        return 1;
    }
    auto &plans = static_cast<Context *>(h)->plan.plans;
    if (!plans.empty()) {
        return d_effective_options(plans.front()->h, out);
    }
    *out = {};
    out->moduli = L8K_NMOD;
    out->encoder_minb = -1;
    out->encoder_fast = -1;
    return 0;
}
