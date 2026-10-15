#include "hayashi_i8gemm.h"
#include "../../../../CUDA-Common/Common/host_options.h"
#include <climits>
#include <memory>
#include <string>

namespace {
thread_local std::string last_error;
struct Plan {
    hayashiHandle_t handle = nullptr;
    int m, n, k, lda, ldb, ldc;
    ~Plan() {
        if (handle) {
            hayashiDestroy(handle);
        }
    }
};
bool ok(hayashiStatus_t status) {
    if (status == HAYASHI_STATUS_SUCCESS) {
        return true;
    }
    last_error = "INT8 engine status " + std::to_string(int(status));
    return false;
}
} // namespace
extern "C" const char *i_last_error() {
    return last_error.c_str();
}
extern "C" void i_destroy(void *p) {
    delete static_cast<Plan *>(p);
}
extern "C" void *i_create_configured(int m, int n, int k, long long lda, long long ldb,
                                     long long ldc, int operation,
                                     const awe_engine_configuration *options) {
    last_error.clear();
    if (!awe_configuration_supported(options) || operation != 1 || m <= 0 || n <= 0 || k <= 0 ||
        m > 65536 || n > 65536 || k > 16384 || lda < m || ldb < k || ldc < m || lda > INT_MAX ||
        ldb > INT_MAX || ldc > INT_MAX) {
        last_error = "unsupported INT8 shape or options";
        return nullptr;
    }
    // The direct INT8 path has no CRT. This bound keeps every exact sum inside INT32.
    if (int64_t(k) * 128 * 128 > INT_MAX) {
        last_error = "INT8 sum exceeds INT32 coverage";
        return nullptr;
    }
    try {
        auto p = std::make_unique<Plan>();
        p->m = m;
        p->n = n;
        p->k = k;
        p->lda = int(lda);
        p->ldb = int(ldb);
        p->ldc = int(ldc);
        if (!ok(hayashiCreate(&p->handle)) || !ok(hayashiSetBackend(p->handle, 6)) ||
            !ok(hayashiSetGraphMode(p->handle, 0)) || !ok(hayashiSetReuseB(p->handle, 0))) {
            return nullptr;
        }
        return p.release();
    } catch (...) {
        last_error = "INT8 plan allocation failed";
        return nullptr;
    }
}
extern "C" int i_poison(void *p, int on) {
    return p && !on ? 0 : -1;
}
extern "C" int i_effective_options(void *p, awe_engine_effective *out) {
    if (!p || !out) {
        return -1;
    }
    *out = {};
    out->chunks = 1;
    out->encoder_minb = 4;
    return 0;
}
extern "C" int i_run(void *context, const void *a, const void *b, void *c, int repeat,
                     double *times) {
    auto *p = static_cast<Plan *>(context);
    if (!p || !a || !b || !c || repeat != 1) {
        last_error = "invalid INT8 execution argument";
        return -1;
    }
    const int32_t one = 1, zero = 0;
    // Each call checks FP32 accumulation bounds; the integer fallback handles inputs outside them.
    if (!ok(hayashiGemmI8(p->handle, HAYASHI_OP_N, HAYASHI_OP_N, p->m, p->n, p->k, &one,
                          static_cast<const int8_t *>(a), p->lda, 0, static_cast<const int8_t *>(b),
                          p->ldb, 0, &zero, static_cast<int32_t *>(c), p->ldc, 0, 1))) {
        return -1;
    }
    const auto status = cudaStreamSynchronize(nullptr);
    if (status != cudaSuccess) {
        last_error = cudaGetErrorString(status);
        return -1;
    }
    if (times) {
        for (int j = 0; j < 4; ++j) {
            times[j] = 0;
        }
    }
    return 0;
}
