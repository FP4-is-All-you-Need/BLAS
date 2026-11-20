// Timing of the 8192^3 engine against rocBLAS DGEMM and, with --hipblaslt, the fastest hipBLASLt
// heuristic. Whole-call host time (stream synchronized before and after) with the method order
// rotated every round; 3 warm-up rounds, then 11 timed rounds. Each method's sampled accuracy is
// recorded once, with the qualification flag against rocBLAS.
//   --module PATH   --distribution uniform|normal|wide   --hipblaslt   --warmup 3   --repeats 11
#include "../plan.h"
#include "../Tools/common.h"
#include <rocblas/rocblas.h>
#ifdef AWE_MXFP6_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace awe_mxfp6;
using namespace awe_mxfp6::host;

#ifndef AWE_MXFP6_MODULE_PATH
#define AWE_MXFP6_MODULE_PATH "awe_mxfp6_modular.co"
#endif

static void rb(rocblas_status s) {
    if (s != rocblas_status_success) {
        throw std::runtime_error("rocBLAS failure");
    }
}

struct Roc {
    rocblas_handle h{};
    explicit Roc(hipStream_t s) {
        rb(rocblas_create_handle(&h));
        rb(rocblas_set_stream(h, s));
        rb(rocblas_set_pointer_mode(h, rocblas_pointer_mode_host));
    }
    ~Roc() {
        rocblas_destroy_handle(h);
    }
    void dgemm(int m, int n, int k, const double *a, const double *b, double *c) {
        const double one = 1, zero = 0;
        rb(rocblas_dgemm(h, rocblas_operation_none, rocblas_operation_none, m, n, k, &one, a, m, b,
                         k, &zero, c, m));
    }
};

#ifdef AWE_MXFP6_HIPBLASLT
static void lb(hipblasStatus_t s) {
    if (s != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("hipBLASLt failure");
    }
}

struct LtDgemm {
    hipblasLtHandle_t h{};
    hipblasLtMatmulDesc_t desc{};
    hipblasLtMatrixLayout_t a{}, b{}, c{};
    hipblasLtMatmulPreference_t pref{};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    std::unique_ptr<Memory> work;
    size_t cap = 64ull << 20;
    LtDgemm(int m, int n, int k) {
        lb(hipblasLtCreate(&h));
        lb(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
        lb(hipblasLtMatrixLayoutCreate(&a, HIP_R_64F, m, k, m));
        lb(hipblasLtMatrixLayoutCreate(&b, HIP_R_64F, k, n, k));
        lb(hipblasLtMatrixLayoutCreate(&c, HIP_R_64F, m, n, m));
        lb(hipblasLtMatmulPreferenceCreate(&pref));
        lb(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &cap, sizeof(cap)));
        candidates.resize(8);
        int returned = 0;
        lb(hipblasLtMatmulAlgoGetHeuristic(h, desc, a, b, c, c, pref, 8, candidates.data(),
                                           &returned));
        candidates.resize(returned);
        work = std::make_unique<Memory>(cap);
    }
    ~LtDgemm() {
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(a);
        hipblasLtMatrixLayoutDestroy(b);
        hipblasLtMatrixLayoutDestroy(c);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(h);
    }
    void run(int i, const double *x, const double *y, double *z, hipStream_t s) {
        const double one = 1, zero = 0;
        lb(hipblasLtMatmul(h, desc, &one, x, a, y, b, &zero, z, c, z, c, &candidates.at(i).algo,
                           work->p, cap, s));
    }
};
#endif

__global__ void gather(const double *c, const uint64_t *indices, double *got, unsigned samples) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < samples) {
        got[i] = c[indices[i]];
    }
}

static double timed(const std::function<void()> &run, hipStream_t s) {
    hip_check(hipStreamSynchronize(s));
    const auto t = std::chrono::steady_clock::now();
    run();
    hip_check(hipGetLastError());
    hip_check(hipStreamSynchronize(s));
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static void list(const std::vector<double> &x) {
    std::cout << '[';
    for (size_t i = 0; i < x.size(); ++i) {
        if (i) {
            std::cout << ',';
        }
        number(x[i]);
    }
    std::cout << ']';
}

int main(int argc, char **argv) try {
    std::cout << std::setprecision(17);
    const std::string module = option(argc, argv, "--module", AWE_MXFP6_MODULE_PATH);
    const std::string distribution = option(argc, argv, "--distribution", "uniform");
    const unsigned warmup = unsigned(std::stoul(option(argc, argv, "--warmup", "3")));
    const unsigned repeats = unsigned(std::stoul(option(argc, argv, "--repeats", "11")));
    const bool use_lt = has_flag(argc, argv, "--hipblaslt");
    const int m = Dgemm8192::m(), n = Dgemm8192::n(), k = Dgemm8192::k();
    const size_t count = size_t(m) * n;

    const Fixture f = make_fixture(m, n, k, distribution, 20261021, table::max_bits);
    const auto refs = sampled_references(f, 32, 32);
    std::vector<uint64_t> indices;
    for (const auto &s : refs) {
        indices.push_back(s.index);
    }
    hipStream_t stream;
    hip_check(hipStreamCreate(&stream));
    Memory da(f.a.size() * 8), db(f.b.size() * 8), dc(count * 8), di(indices.size() * 8),
        dg(refs.size() * 8);
    hip_check(hipMemcpy(da.p, f.a.data(), f.a.size() * 8, hipMemcpyHostToDevice));
    hip_check(hipMemcpy(db.p, f.b.data(), f.b.size() * 8, hipMemcpyHostToDevice));
    hip_check(hipMemcpy(di.p, indices.data(), indices.size() * 8, hipMemcpyHostToDevice));
    Roc roc(stream);
    Dgemm8192 plan(stream, module);
    Timings last{};

    auto samples = [&]() {
        hipLaunchKernelGGL(gather, dim3((refs.size() + 255) / 256), dim3(256), 0, stream,
                           dc.as<double>(), di.as<uint64_t>(), dg.as<double>(),
                           unsigned(refs.size()));
        hip_check(hipGetLastError());
        hip_check(hipStreamSynchronize(stream));
        std::vector<double> got(refs.size());
        hip_check(hipMemcpy(got.data(), dg.p, refs.size() * 8, hipMemcpyDeviceToHost));
        return metrics(got, refs);
    };
    auto poison = [&]() { hip_check(hipMemsetAsync(dc.p, 0xff, count * 8, stream)); };

    std::vector<std::string> names = {"awe_mxfp6", "rocblas_dgemm"};
    std::vector<std::function<void()>> engines = {
        [&]() { last = plan.run(da.as<double>(), db.as<double>(), dc.as<double>(), m, k, m); },
        [&]() { roc.dgemm(m, n, k, da.as<double>(), db.as<double>(), dc.as<double>()); }};
#ifdef AWE_MXFP6_HIPBLASLT
    std::unique_ptr<LtDgemm> lt;
    int lt_best = -1;
    if (use_lt) {
        lt = std::make_unique<LtDgemm>(m, n, k);
        double best = INFINITY;
        for (unsigned i = 0; i < lt->candidates.size(); ++i) {
            if (lt->candidates[i].state != HIPBLAS_STATUS_SUCCESS ||
                lt->candidates[i].workspaceSize > lt->cap) {
                continue;
            }
            auto run = [&]() {
                lt->run(int(i), da.as<double>(), db.as<double>(), dc.as<double>(), stream);
            };
            std::vector<double> times;
            for (unsigned w = 0; w < 3; ++w) {
                timed(run, stream);
            }
            for (unsigned t = 0; t < 5; ++t) {
                times.push_back(timed(run, stream));
            }
            if (median(times) < best) {
                best = median(times);
                lt_best = int(i);
            }
        }
        if (lt_best >= 0) {
            names.push_back("hipblaslt_dgemm_ordinal" + std::to_string(lt_best));
            engines.push_back([&]() {
                lt->run(lt_best, da.as<double>(), db.as<double>(), dc.as<double>(), stream);
            });
        }
    }
#else
    if (use_lt) {
        throw std::runtime_error("built without hipBLASLt (AWE_MXFP6_HIPBLASLT)");
    }
#endif

    // Sampled accuracy of every method, with rocBLAS as the reference of the criterion.
    poison();
    engines[1]();
    const Metrics native = samples();
    std::vector<Metrics> errors(engines.size());
    std::vector<bool> eligible(engines.size());
    for (size_t e = 0; e < engines.size(); ++e) {
        poison();
        engines[e]();
        errors[e] = samples();
        eligible[e] = qualified(errors[e], native);
    }

    std::vector<std::vector<double>> times(engines.size());
    std::vector<std::vector<double>> phases(5);
    for (unsigned round = 0; round < warmup + repeats; ++round) {
        for (unsigned offset = 0; offset < engines.size(); ++offset) {
            const unsigned e = (round + offset) % unsigned(engines.size());
            poison();
            const double ms = timed(engines[e], stream);
            if (round >= warmup) {
                times[e].push_back(ms);
                if (e == 0) {
                    phases[0].push_back(last.shift);
                    phases[1].push_back(last.encode);
                    phases[2].push_back(last.products);
                    phases[3].push_back(last.crt);
                    phases[4].push_back(last.total);
                }
            }
        }
    }

    std::cout
        << "{\"kind\":\"COMPARISON\",\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k
        << ",\"distribution\":\"" << distribution << "\",\"bits\":" << table::max_bits
        << ",\"samples\":" << refs.size() << ",\"warmup\":" << warmup << ",\"repeats\":" << repeats
        << ",\"scope\":\"FP64 inputs resident on the device to FP64 output, stream synchronized before and after each call\""
        << ",\"accuracy_criterion\":\"sampled ULP p95 and max no worse than rocBLAS on the same inputs\",\"engines\":[";
    const char *labels[] = {"shift", "encode", "products", "crt", "event_total"};
    for (size_t e = 0; e < engines.size(); ++e) {
        if (e) {
            std::cout << ',';
        }
        std::cout << "{\"name\":\"" << names[e]
                  << "\",\"qualified\":" << (eligible[e] ? "true" : "false")
                  << ",\"median_ms\":" << median(times[e]) << ",\"ms\":";
        list(times[e]);
        std::cout << ",\"ulp_p95\":" << errors[e].ulp_p95 << ",\"ulp_max\":" << errors[e].ulp_max;
        if (e == 0) {
            std::cout << ",\"phase_ms\":{";
            for (unsigned p = 0; p < 5; ++p) {
                if (p) {
                    std::cout << ',';
                }
                std::cout << '"' << labels[p] << "\":";
                list(phases[p]);
            }
            std::cout << '}';
        }
        std::cout << '}';
    }
    std::vector<double> ratios;
    for (size_t t = 0; t < times[0].size(); ++t) {
        ratios.push_back(times[0][t] / times[1][t]);
    }
    std::cout << "],\"paired_awe_over_rocblas_median\":" << median(ratios) << "}\n";
    hip_check(hipStreamDestroy(stream));
    return 0;
} catch (const std::exception &e) {
    std::cerr << "bench failed: " << e.what() << '\n';
    return 2;
}
