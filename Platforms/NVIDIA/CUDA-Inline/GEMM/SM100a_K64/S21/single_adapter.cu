#include <cstddef>
#include "../../../../CUDA-Common/Common/host_options.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>

extern "C" void *f_create_ld(int, int, int, long long, long long, long long);
extern "C" int f_run(void *, const void *, const void *, void *, int, double *);
extern "C" int f_poison(void *, int);
extern "C" void f_destroy(void *);
extern "C" const char *f_last_error();
extern "C" const char *f_info(void *);
extern "C" int f_effective_options(void *, awe_engine_effective *);

namespace {
thread_local std::string error;
void ck(cudaError_t s) {
    if (s != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(s));
    }
}
struct Api {
    void *so = nullptr;
    void *(*create)(int, int, int, long long, long long, long long) = nullptr;
    int (*run)(void *, const void *, const void *, void *, int, double *) = nullptr;
    int (*poison)(void *, int) = nullptr;
    void (*destroy)(void *) = nullptr;
    const char *(*err)() = nullptr;
    const char *(*info)(void *) = nullptr;
    ~Api() {
        if (so) {
            dlclose(so);
        }
    }
    void load() {
        create = f_create_ld;
        run = f_run;
        poison = f_poison;
        destroy = f_destroy;
        err = f_last_error;
        info = f_info;
    }
};
// Same tiled transpose as the D adapter, generalized destination ld and offset.
__global__ void transpose_into(const float *A, size_t n, size_t k, float *Y, size_t ldo,
                               size_t row0) {
    __shared__ float t[32][33];
    const size_t i0 = size_t(blockIdx.x) * 32, p0 = size_t(blockIdx.y) * 32;
    for (int y = threadIdx.y; y < 32; y += 8) {
        size_t i = i0 + threadIdx.x, p = p0 + y;
        if (i < n && p < k) {
            t[y][threadIdx.x] = A[i + p * n];
        }
    }
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) {
        size_t p = p0 + threadIdx.x, i = i0 + y;
        if (i < n && p < k) {
            Y[row0 + p + i * ldo] = t[threadIdx.x][y];
        }
    }
}
__global__ void matrix_copy(const float *A, size_t n, float *X, int symmetric) {
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n * n;
         t += size_t(gridDim.x) * blockDim.x) {
        const size_t i = t % n, j = t / n;
        X[t] = i >= j ? A[t] : (symmetric ? A[j + i * n] : 0.0f);
    }
}
__global__ void commit(const float *P, size_t n, float *C, int lower, int xxt) {
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n * n;
         t += size_t(gridDim.x) * blockDim.x) {
        const size_t i = t % n, j = t / n;
        if (!lower || i >= j) {
            C[t] = xxt ? __fadd_rn(P[t], P[j + i * n]) : P[t];
        }
    }
}
struct Ctx {
    Api api;
    int routine, n, k, param, strips = 1;
    float *X = nullptr, *Y = nullptr, *P = nullptr;
    void *plan[2] = {};
    cudaEvent_t event[3] = {};
    std::string info;
    ~Ctx() {
        for (auto p : plan) {
            if (p) {
                api.destroy(p);
            }
        }
        if (X) {
            cudaFree(X);
        }
        if (Y) {
            cudaFree(Y);
        }
        if (P) {
            cudaFree(P);
        }
        for (auto e : event) {
            if (e) {
                cudaEventDestroy(e);
            }
        }
    }
    void make_plan(int j, int m, int nn, int kk, long long lda, long long ldb, long long ldc) {
        plan[j] = api.create(m, nn, kk, lda, ldb, ldc);
        if (!plan[j]) {
            throw std::runtime_error(std::string("SGEMM create: ") + api.err());
        }
        api.poison(plan[j], 0);
    }
    void mul(int j, const float *A, const float *B, float *C, double *stages) {
        if (api.run(plan[j], A, B, C, 1, stages)) {
            throw std::runtime_error(std::string("SGEMM run: ") + api.err());
        }
    }
};
} // namespace
extern "C" void *s3_create(int routine, int n, int k, int param) try {
    error.clear();
    if (routine < 0 || routine > 3 || n < 1 || k < 1 || k > 16384) {
        throw std::runtime_error("invalid routine or size");
    }
    if ((routine == 0 && (param != 1 && param != 2)) ||
        (routine == 2 && (param != 1 && param != 3)) ||
        ((routine == 1 || routine == 3) && (param != 1 || k != n))) {
        throw std::runtime_error("invalid param or square operand size");
    }
    if (routine == 0 && param == 2 && n % 2) {
        throw std::runtime_error("two strips require even n");
    }
    if (routine == 2 && param == 1 && k > 8192) {
        throw std::runtime_error("concatenated SYR2K requires 2*k<=16384");
    }
    auto c = std::make_unique<Ctx>();
    c->routine = routine;
    c->n = n;
    c->k = k;
    c->param = param;
    c->api.load();
    const size_t nn = size_t(n) * n, nk = size_t(n) * k;
    ck(cudaMalloc(&c->P, nn * sizeof(float)));
    if (routine == 0) {
        ck(cudaMalloc(&c->X, nk * sizeof(float)));
        c->strips = param;
        const int w = n / c->strips;
        for (int j = 0; j < c->strips; ++j) {
            c->make_plan(j, n - j * w, w, k, n, k, n);
        }
    } else if (routine == 2) {
        if (param == 3) {
            ck(cudaMalloc(&c->Y, nk * sizeof(float)));
            c->make_plan(0, n, n, k, n, k, n);
        } else {
            ck(cudaMalloc(&c->X, 2 * nk * sizeof(float)));
            ck(cudaMalloc(&c->Y, 2 * nk * sizeof(float)));
            c->make_plan(0, n, n, 2 * k, n, 2 * k, n);
        }
    } else {
        ck(cudaMalloc(&c->X, nn * sizeof(float)));
        c->make_plan(0, n, n, n, n, n, n);
    }
    for (auto &e : c->event) {
        ck(cudaEventCreate(&e));
    }
    c->info = "{\"routine\":" + std::to_string(routine) + ",\"n\":" + std::to_string(n) +
              ",\"k\":" + std::to_string(k) + ",\"param\":" + std::to_string(param) +
              ",\"strips\":" + std::to_string(c->strips) +
              ",\"engine\":" + c->api.info(c->plan[0]) + "}";
    return c.release();
} catch (const std::exception &e) {
    error = e.what();
    return nullptr;
}
extern "C" int s3_run(void *h, const void *ap, const void *bp, void *cp, int runs,
                      double *stage) try {
    error.clear();
    if (!h || !ap || !cp || !stage || runs < 1) {
        throw std::runtime_error("invalid run arguments");
    }
    auto &c = *static_cast<Ctx *>(h);
    if (c.routine != 0 && !bp) {
        throw std::runtime_error("B required");
    }
    if (cp == ap || cp == bp) {
        throw std::runtime_error("output alias is unsupported");
    }
    const auto *A = static_cast<const float *>(ap);
    const auto *B = static_cast<const float *>(bp);
    auto *C = static_cast<float *>(cp);
    const size_t n = c.n, k = c.k;
    const dim3 tg(unsigned((n + 31) / 32), unsigned((k + 31) / 32)), tb(32, 8);
    for (int r = 0; r < runs; ++r) {
        const auto start = std::chrono::steady_clock::now();
        ck(cudaEventRecord(c.event[0]));
        if (c.routine == 0) {
            transpose_into<<<tg, tb>>>(A, n, k, c.X, k, 0);
        } else if (c.routine == 2) {
            if (c.param == 3) {
                transpose_into<<<tg, tb>>>(B, n, k, c.Y, k, 0);
            } else {
                ck(cudaMemcpyAsync(c.X, A, n * k * sizeof(float), cudaMemcpyDeviceToDevice));
                ck(cudaMemcpyAsync(c.X + n * k, B, n * k * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
                transpose_into<<<tg, tb>>>(B, n, k, c.Y, 2 * k, 0);
                transpose_into<<<tg, tb>>>(A, n, k, c.Y, 2 * k, k);
            }
        } else {
            matrix_copy<<<4096, 256>>>(A, n, c.X, c.routine == 3);
        }
        ck(cudaGetLastError());
        ck(cudaEventRecord(c.event[1]));
        ck(cudaEventSynchronize(c.event[1]));
        float copies = 0;
        ck(cudaEventElapsedTime(&copies, c.event[0], c.event[1]));
        double sum[6] = {}, ss[6] = {};
        if (c.routine == 0) {
            const size_t w = n / c.strips;
            for (int j = 0; j < c.strips; ++j) {
                c.mul(j, A + j * w, c.X + j * w * k, c.P + j * w + j * w * n, ss);
                for (int q = 0; q < 6; ++q) {
                    sum[q] += ss[q];
                }
            }
        } else {
            c.mul(0, c.routine == 2 ? (c.param == 3 ? A : c.X) : c.X, c.routine == 2 ? c.Y : B, c.P,
                  sum);
        }
        ck(cudaEventRecord(c.event[1]));
        commit<<<4096, 256>>>(c.P, n, C, c.routine == 0 || c.routine == 2,
                              c.routine == 2 && c.param == 3);
        ck(cudaGetLastError());
        ck(cudaEventRecord(c.event[2]));
        ck(cudaEventSynchronize(c.event[2]));
        float post = 0;
        ck(cudaEventElapsedTime(&post, c.event[1], c.event[2]));
        stage[6 * r] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        stage[6 * r + 1] = copies;
        stage[6 * r + 2] = sum[0];
        stage[6 * r + 3] = post;
        stage[6 * r + 4] = sum[4];
        stage[6 * r + 5] = sum[5];
    }
    return 0;
} catch (const std::exception &e) {
    error = e.what();
    return 1;
}
extern "C" const char *s3_info(void *h) {
    return h ? static_cast<Ctx *>(h)->info.c_str() : "";
}
extern "C" const char *s3_last_error() {
    return error.c_str();
}
extern "C" void s3_destroy(void *h) {
    delete static_cast<Ctx *>(h);
}

// Keep the measured adapter's two-strip SYRK and product-plus-transpose SYR2K paths.
extern "C" void *u_create_configured(int m, int n, int k, long long lda, long long ldb,
                                     long long ldc, int op, const awe_engine_configuration *cfg) {
    if (!awe_configuration_supported(cfg) || m != n || lda != m || ldc != m ||
        ldb != (op == 7 ? m : k)) {
        error = "unsupported single-precision layout or tuning";
        return nullptr;
    }
    const int routine = op == 2 ? 0 : op == 3 ? 1 : op == 7 ? 2 : op == 5 ? 3 : -1;
    return s3_create(routine, n, k, routine == 0 && n % 2 == 0 ? 2 : routine == 2 ? 3 : 1);
}
extern "C" int u_run(void *p, const void *a, const void *b, void *c, int r, double *t) {
    return s3_run(p, a, b, c, r, t);
}
extern "C" void u_destroy(void *p) {
    s3_destroy(p);
}
extern "C" const char *u_last_error() {
    return s3_last_error();
}
extern "C" int u_poison(void *h, int on) {
    if (!h) {
        return -1;
    }
    auto p = static_cast<Ctx *>(h);
    for (auto plan : p->plan) {
        if (plan && p->api.poison(plan, on) < 0) {
            return -1;
        }
    }
    return 0;
}
extern "C" int u_effective_options(void *h, awe_engine_effective *out) {
    if (!h || !out) {
        return 1;
    }
    return f_effective_options(static_cast<Ctx *>(h)->plan[0], out);
}
