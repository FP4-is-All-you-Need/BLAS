#include "../../../../CUDA-Common/Common/host_options.h"
extern "C" void *d_create_configured(int, int, int, long long, long long, long long, int,
                                     const awe_engine_configuration *);
extern "C" int d_run(void *, const void *, const void *, void *, int, double *);
extern "C" void d_destroy(void *);
extern "C" int d_poison(void *, int);
extern "C" const char *d_info(void *);
extern "C" const char *d_last_error();
extern "C" int d_effective_options(void *, awe_engine_effective *);
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static void ck(cudaError_t x) {
    if (x != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(x));
    }
}
static void cb(cublasStatus_t x) {
    if (x != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS status " + std::to_string(x));
    }
}
struct Mem {
    double *p = nullptr;
    size_t n;
    Mem(size_t n) : n(n) {
        ck(cudaMalloc(&p, n * 8));
    }
    ~Mem() {
        cudaFree(p);
    }
    void put(const std::vector<double> &v) {
        if (v.size() != n) {
            throw std::runtime_error("size");
        }
        ck(cudaMemcpy(p, v.data(), n * 8, cudaMemcpyHostToDevice));
    }
    std::vector<double> get() {
        std::vector<double> v(n);
        ck(cudaMemcpy(v.data(), p, n * 8, cudaMemcpyDeviceToHost));
        return v;
    }
};
struct BLAS {
    cublasHandle_t h{};
    BLAS() {
        cb(cublasCreate(&h));
        cb(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    }
    ~BLAS() {
        cublasDestroy(h);
    }
};
struct Engine {
    void *so = nullptr;
    using Create = void *(*)(int, int, int, long long, long long, long long);
    using Run = int (*)(void *, const void *, const void *, void *, int, double *);
    using Destroy = void (*)(void *);
    using Poison = int (*)(void *, int);
    using Info = const char *(*)(void *);
    using Err = const char *(*)();
    Create create;
    Run run;
    Destroy destroy;
    Poison poison;
    Info info;
    Err error;
    Engine(const char * = nullptr) {
        create = [](int m, int n, int k, long long lda, long long ldb, long long ldc) -> void * {
            awe_engine_configuration c{};
            c.struct_size = sizeof(c);
            return d_create_configured(m, n, k, lda, ldb, ldc, 1, &c);
        };
        run = d_run;
        destroy = d_destroy;
        poison = d_poison;
        info = d_info;
        error = d_last_error;
    }
    void *sym(const char *n) {
        void *p = dlsym(so, n);
        if (!p) {
            throw std::runtime_error(n);
        }
        return p;
    }
    ~Engine() {
        if (so) {
            dlclose(so);
        }
    }
};
struct Plan {
    Engine &e;
    void *h;
    Plan(Engine &e, int m, int n, int k, long long lda, long long ldb, long long ldc)
        : e(e), h(e.create(m, n, k, lda, ldb, ldc)) {
        if (!h) {
            throw std::runtime_error(e.error());
        }
        if (e.poison(h, 0) < 0) {
            throw std::runtime_error("d_poison invalid handle");
        }
    }
    ~Plan() {
        e.destroy(h);
    }
    void run(const double *a, const double *b, double *c) {
        double stages[6];
        if (e.run(h, a, b, c, 1, stages)) {
            throw std::runtime_error(e.error());
        }
    }
};
__global__ void subtract(double *b, const double *p, int rows, int cols, int ldb) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < size_t(rows) * cols) {
        int i = t % rows, j = t / rows;
        b[i + size_t(j) * ldb] = __dsub_rn(b[i + size_t(j) * ldb], p[t]);
    }
}
