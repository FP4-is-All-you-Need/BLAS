

#pragma once
#include "gen_mc.h"
#include <memory>

namespace fvrt {
static thread_local std::string error;
struct Ctx {
    l8rt::Plan plan;
    std::string info;
    int poison = 1;
};
// Same input conversion and hash formula as l8f_main.cuh.
__global__ void to_f32(const double *x, float *y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        y[i] = __double2float_rn(x[i]);
    }
}
__global__ void fill_mc(float *x, size_t n, unsigned long long stream) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        x[i] = gen_val(stream, i);
    }
}
__global__ void hash_float(const float *x, size_t n, unsigned long long *out) {
    unsigned long long h = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        h ^= (unsigned long long)__float_as_uint(x[i]) * 0x9E3779B97F4A7C15ull +
             i * 0xD1342543DE82EF95ull;
    }
    for (int o = 16; o; o >>= 1) {
        h ^= __shfl_down_sync(0xffffffffu, h, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicXor(out, h);
    }
}
} // namespace fvrt

extern "C" const char *f_last_error() {
    return fvrt::error.c_str();
}
extern "C" void *f_create_ld(int m, int n, int k, long long lda, long long ldb, long long ldc) try {
    fvrt::error.clear();
    cudaDeviceProp prop{};
    int device = 0;
    ck(cudaGetDevice(&device));
    ck(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 10 || prop.minor != 3) {
        throw std::runtime_error("requires B200 sm_100a or B300 sm_103a (dense K96)");
    }
    auto c = std::make_unique<fvrt::Ctx>();
    c->plan.init(m, n, k, lda, ldb, ldc);
    const auto &p = c->plan;
    char info[1536];
    snprintf(info, sizeof info,
             "{\"engine\":\"l8f_k96\",\"precision\":\"fp32\",\"gpu\":\"%s\","
             "\"sm\":%d,\"m\":%d,\"n\":%d,\"k\":%d,\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,"
             "\"bits\":%d,\"moduli\":%d,\"planes_per_side\":%d,\"mma_products\":%d,"
             "\"accumulators_per_tile\":%d,\"k_plan\":%d,\"cuts\":%d,\"mma_k\":96,"
             "\"output\":\"column-major float\",\"reference_checked\":false}",
             prop.name, prop.major * 10 + prop.minor, m, n, k, lda, ldb, ldc, p.bits, L8K_NMOD,
             L8K_PLANES59, L8K_PRODUCTS59, p.accs, p.kplan, p.sh.cuts);
    c->info = info;
    return c.release();
} catch (const std::exception &e) {
    fvrt::error = e.what();
    return nullptr;
}
extern "C" void *f_create() {
    return f_create_ld(8192, 8192, 8192, 8192, 8192, 8192);
}
extern "C" const char *f_info(void *h) {
    return h ? static_cast<fvrt::Ctx *>(h)->info.c_str() : "";
}

extern "C" int f_fill_normal(void *x, long long n, unsigned long long seed) {
    fvrt::error.clear();
    double *tmp = nullptr;
    try {
        if (!x || n <= 0) {
            throw std::runtime_error("f_fill_normal: invalid arguments");
        }
        ck(cudaMalloc(&tmp, size_t(n) * sizeof(double)));
        fill_normal<<<4096, 256>>>(tmp, size_t(n), seed);
        ck(cudaGetLastError());
        fvrt::to_f32<<<4096, 256>>>(tmp, static_cast<float *>(x), size_t(n));
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        ck(cudaFree(tmp));
        return 0;
    } catch (const std::exception &e) {
        if (tmp) {
            cudaFree(tmp);
        }
        fvrt::error = e.what();
        return 1;
    }
}
extern "C" int f_fill_mc(void *x, long long n, unsigned long long stream) try {
    fvrt::error.clear();
    if (!x || n <= 0) {
        throw std::runtime_error("f_fill_mc: invalid arguments");
    }
    fvrt::fill_mc<<<4096, 256>>>(static_cast<float *>(x), size_t(n), stream);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    return 0;
} catch (const std::exception &e) {
    fvrt::error = e.what();
    return 1;
}
extern "C" int f_run(void *h, const void *A, const void *B, void *C, int runs, double *stage) try {
    fvrt::error.clear();
    if (!h || !A || !B || !C || !stage || runs <= 0) {
        throw std::runtime_error("f_run: invalid arguments");
    }
    auto &c = *static_cast<fvrt::Ctx *>(h);
    auto &p = c.plan;
    for (int r = 0; r < runs; ++r) {
        if (c.poison) {
            ck(cudaMemset2DAsync(C, size_t(p.ldc) * 4, 0xa5, size_t(p.m) * 4, p.n, 0));
            ck(cudaMemsetAsync(p.pa, 0xa5, p.bytesA, 0));
            ck(cudaMemsetAsync(p.pb, 0xa5, p.bytesB, 0));
            ck(cudaMemsetAsync(p.resid, 0xa5, p.resid_bytes, 0));
        }
        ck(cudaMemsetAsync(p.km, 0, size_t(p.sh.tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(p.mm, 0, size_t(p.sh.tiles) * kParts * 2 * sizeof(Metrics), 0));
        float st[3] = {};
        if (p.run(static_cast<const float *>(A), static_cast<const float *>(B),
                  static_cast<float *>(C), 0, st, 0u) == 2) {
            throw std::runtime_error("input out of range (Inf / NaN or beyond the shift)");
        }
        stage[6 * r] = double(st[0]) + st[1] + st[2];
        stage[6 * r + 1] = st[0];
        stage[6 * r + 2] = 0;
        stage[6 * r + 3] = 0;
        stage[6 * r + 4] = st[1];
        stage[6 * r + 5] = st[2];
    }
    return 0;
} catch (const std::exception &e) {
    fvrt::error = e.what();
    return 1;
}
// Hash n contiguous floats. For ldc>m callers compact valid C before hashing.
extern "C" int f_hash(const void *x, long long n, unsigned long long *total) {
    fvrt::error.clear();
    unsigned long long *tmp = nullptr;
    try {
        if (!x || n <= 0 || !total) {
            throw std::runtime_error("f_hash: invalid arguments");
        }
        ck(cudaMalloc(&tmp, 8));
        ck(cudaMemset(tmp, 0, 8));
        fvrt::hash_float<<<4096, 256>>>(static_cast<const float *>(x), size_t(n), tmp);
        ck(cudaGetLastError());
        ck(cudaMemcpy(total, tmp, 8, cudaMemcpyDeviceToHost));
        ck(cudaFree(tmp));
        return 0;
    } catch (const std::exception &e) {
        if (tmp) {
            cudaFree(tmp);
        }
        fvrt::error = e.what();
        return 1;
    }
}
extern "C" int f_poison(void *h, int on) {
    if (!h) {
        return -1;
    }
    auto &c = *static_cast<fvrt::Ctx *>(h);
    int prev = c.poison;
    c.poison = on ? 1 : 0;
    return prev;
}
extern "C" void f_destroy(void *h) {
    if (!h) {
        return;
    }
    auto *c = static_cast<fvrt::Ctx *>(h);
    c->plan.destroy();
    delete c;
}
