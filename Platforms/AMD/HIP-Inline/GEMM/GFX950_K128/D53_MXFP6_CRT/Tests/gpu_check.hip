// GPU correctness check of the 8192^3 engine against exact references and rocBLAS DGEMM.
//   --module PATH         code object (default: the build's awe_mxfp6_modular.co)
//   --distribution NAME   uniform (default), normal or wide
//   --full                also check every output against rocBLAS with an interval bound (slower)
// Checks: (1) the 1,024 sampled outputs equal the exactly rounded dot product of the quantized inputs
// bit for bit; (2) their ULP errors against the exact dot of the original inputs satisfy the
// accuracy criterion (p95 and max no worse than rocBLAS on the same samples); (3) inputs unchanged;
// (4) with --full, |C - C_rocBLAS| is within the quantization plus DGEMM rounding bound everywhere.
#include "../plan.h"
#include "../Tools/common.h"
#include <rocblas/rocblas.h>
#include <iomanip>
#include <iostream>
#include <limits>

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

__global__ void gather(const double *c, const uint64_t *indices, double *got, unsigned samples) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < samples) {
        got[i] = c[indices[i]];
    }
}

static std::vector<double> download(const Memory &mem, size_t count) {
    std::vector<double> out(count);
    hip_check(hipMemcpy(out.data(), mem.p, count * 8, hipMemcpyDeviceToHost));
    return out;
}

// Full-output consistency with rocBLAS: |C_awe - C_roc| <= quantization error bound + DGEMM rounding bound,
// where the bounds use |A| |B| (one more rocBLAS call on the absolute inputs) and the per-line shifts.
static size_t full_interval_check(const Fixture &f, const std::vector<double> &awe,
                                  const std::vector<double> &roc,
                                  const std::vector<double> &abs_product) {
    const int m = f.m, n = f.n, k = f.k;
    auto upward = [](double x) { return std::nextafter(x, INFINITY); };
    std::vector<double> sum_a(m), sum_b(n);
    for (int r = 0; r < m; ++r) {
        for (int t = 0; t < k; ++t) {
            sum_a[r] = upward(sum_a[r] + std::abs(f.a[size_t(t) * m + r]));
        }
    }
    for (int c = 0; c < n; ++c) {
        for (int t = 0; t < k; ++t) {
            sum_b[c] = upward(sum_b[c] + std::abs(f.b[size_t(c) * k + t]));
        }
    }
    const double unit = 0x1p-53, ku = k * unit, gamma = upward(ku / (1 - ku));
    size_t bad = 0;
#pragma omp parallel for reduction(+ : bad)
    for (size_t index = 0; index < size_t(m) * n; ++index) {
        const double x = awe[index], y = roc[index];
        if (!std::isfinite(x) || !std::isfinite(y)) {
            ++bad;
            continue;
        }
        const int row = int(index % m), col = int(index / m);
        const double sh = abs_product[index], sup = upward(sh / (1 - gamma));
        const double tau = std::ldexp(1., -f.sa[row] - 1), ups = std::ldexp(1., -f.sb[col] - 1);
        const double eq = upward(upward(upward(tau * sum_b[col]) + upward(ups * sum_a[row])) +
                                 upward(upward(tau * ups) * k));
        const double tolerance = upward(
            upward(upward(std::nextafter(1., INFINITY) * eq) + upward(upward(unit + gamma) * sup)) +
            std::numeric_limits<double>::denorm_min());
        if (std::abs((long double)x - y) > tolerance) {
            ++bad;
        }
    }
    return bad;
}

int main(int argc, char **argv) try {
    std::cout << std::setprecision(17);
    const std::string module = option(argc, argv, "--module", AWE_MXFP6_MODULE_PATH);
    const std::string distribution = option(argc, argv, "--distribution", "uniform");
    const bool full = has_flag(argc, argv, "--full");
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
    std::cout << "{\"kind\":\"ENV\",\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k
              << ",\"distribution\":\"" << distribution << "\",\"bits\":" << table::max_bits
              << ",\"launches\":" << plan.launches() << ",\"device_bytes\":" << plan.device_bytes()
              << "}\n";

    auto samples = [&]() {
        hipLaunchKernelGGL(gather, dim3((refs.size() + 255) / 256), dim3(256), 0, stream,
                           dc.as<double>(), di.as<uint64_t>(), dg.as<double>(),
                           unsigned(refs.size()));
        hip_check(hipGetLastError());
        hip_check(hipStreamSynchronize(stream));
        return download(dg, refs.size());
    };
    auto poison = [&]() { hip_check(hipMemsetAsync(dc.p, 0xff, count * 8, stream)); };

    poison();
    roc.dgemm(m, n, k, da.as<double>(), db.as<double>(), dc.as<double>());
    const Metrics native = metrics(samples(), refs);
    std::vector<double> roc_full;
    if (full) {
        roc_full = download(dc, count);
    }

    poison();
    const Timings t = plan.run(da.as<double>(), db.as<double>(), dc.as<double>(), m, k, m);
    const auto got = samples();
    size_t quantized_mismatch = 0;
    for (size_t i = 0; i < refs.size(); ++i) {
        quantized_mismatch += bits(got[i]) != bits(refs[i].quantized);
    }
    const Metrics own = metrics(got, refs);
    const bool ok_samples = quantized_mismatch == 0 && qualified(own, native);
    std::cout << "{\"kind\":\"SAMPLES\",\"samples\":" << refs.size()
              << ",\"quantized_bit_mismatches\":" << quantized_mismatch
              << ",\"awe_ulp_p95\":" << own.ulp_p95 << ",\"awe_ulp_max\":" << own.ulp_max
              << ",\"rocblas_ulp_p95\":" << native.ulp_p95
              << ",\"rocblas_ulp_max\":" << native.ulp_max
              << ",\"qualified\":" << (ok_samples ? "true" : "false")
              << ",\"event_ms\":{\"shift\":" << t.shift << ",\"encode\":" << t.encode
              << ",\"products\":" << t.products << ",\"crt\":" << t.crt << ",\"total\":" << t.total
              << "}}\n";

    // Inputs must be byte-identical after the call.
    const auto a_after = download(da, f.a.size()), b_after = download(db, f.b.size());
    const bool ok_inputs = std::memcmp(a_after.data(), f.a.data(), f.a.size() * 8) == 0 &&
                           std::memcmp(b_after.data(), f.b.data(), f.b.size() * 8) == 0;
    std::cout << "{\"kind\":\"INPUTS\",\"unchanged\":" << (ok_inputs ? "true" : "false") << "}\n";

    bool ok_full = true;
    if (full) {
        const auto awe_full = download(dc, count);
        std::vector<double> aa = f.a, bb = f.b;
        for (auto &x : aa) {
            x = std::abs(x);
        }
        for (auto &x : bb) {
            x = std::abs(x);
        }
        hip_check(hipMemcpy(da.p, aa.data(), aa.size() * 8, hipMemcpyHostToDevice));
        hip_check(hipMemcpy(db.p, bb.data(), bb.size() * 8, hipMemcpyHostToDevice));
        roc.dgemm(m, n, k, da.as<double>(), db.as<double>(), dc.as<double>());
        hip_check(hipStreamSynchronize(stream));
        const auto abs_product = download(dc, count);
        const size_t bad = full_interval_check(f, awe_full, roc_full, abs_product);
        size_t different = 0;
        for (size_t i = 0; i < count; ++i) {
            different += awe_full[i] != roc_full[i];
        }
        ok_full = bad == 0;
        std::cout << "{\"kind\":\"FULL\",\"checked\":" << count << ",\"outside_bound\":" << bad
                  << ",\"different_from_rocblas\":" << different << "}\n";
    }

    const bool passed = ok_samples && ok_inputs && ok_full;
    std::cout << "{\"kind\":\"END\",\"passed\":" << (passed ? "true" : "false") << "}\n";
    hip_check(hipStreamDestroy(stream));
    return passed ? 0 : 1;
} catch (const std::exception &e) {
    std::cerr << "gpu_check failed: " << e.what() << '\n';
    return 2;
}
