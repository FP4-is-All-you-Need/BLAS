// C entry points for the real runtime-sized GEMM plan; the plan owns device workspace.
// d_create uses the compile-time default dimensions and contiguous leading dimensions.
// d_create_ld(m,n,k,lda,ldb,ldc) accepts column-major A (lda x k), B (ldb x n), C (ldc x n),
// with lda >= m, ldb >= k, ldc >= m. The selected plan bounds k. Padding is not read or written.
// d_info returns plan metadata; d_last_error returns the last error string.
// d_fill_normal fills an array with the fixed-seed input generator.
// d_run writes stage[6*r] = (total, shifts, norm=0, gate_gap=0, encode, gemm).
// When poisoning is enabled, valid C, packed operands, and residues are poisoned before timing.
// d_verify recomputes npts output elements exactly from the quantized inputs on the host,
// rounds once, and reports spot_bad plus the hash of valid m x n output, excluding padding.
// d_hash hashes contiguous output: total is XOR over i of
// bits(x[i]) * 0x9E3779B97F4A7C15 + i * 0xD1342543DE82EF95.
// For d_hash and d_verify, blocks is NULL or an array of 256 unsigned long long elements
// (2,048 bytes). A shorter buffer corrupts caller memory. Each block covers floor(n/256)
// consecutive elements, with the remainder in the last block.
// d_cycles returns cycle counters; d_poison controls poisoning; d_destroy releases the plan.
// d_run_reuse uses bit 1 for A and bit 2 for B: the caller promises the same values at the
// same pointer. Reuse skips shifts and encoding; a 4,096-element sample hash is checked.
// A changed sample under the same pointer causes re-encoding and counts as a broken promise.
// Changes outside the sample are not detected. Reused packed planes are not poisoned.
// d_reuse_stats returns skipped-A, skipped-B, and detected broken-promise counts.
#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace dvrt {
constexpr int NB = 256;
__global__ void __launch_bounds__(256)
    hash_blocks_any(const double *__restrict__ x, size_t n, unsigned long long *out) {
    const int g_per = gridDim.x / NB;
    const int b = blockIdx.x / g_per, g = blockIdx.x % g_per;
    const size_t per = n / NB, lo = size_t(b) * per, hi = b == NB - 1 ? n : lo + per;
    unsigned long long h = 0;
    for (size_t i = lo + size_t(g) * blockDim.x + threadIdx.x; i < hi;
         i += size_t(g_per) * blockDim.x) {
        h ^= (unsigned long long)__double_as_longlong(x[i]) * 0x9E3779B97F4A7C15ull +
             i * 0xD1342543DE82EF95ull;
    }
    for (int o = 16; o; o >>= 1) {
        h ^= __shfl_xor_sync(0xffffffffu, h, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicXor(out + b, h);
    }
}
static unsigned long long hash_any(const double *x, size_t n, unsigned long long *hb) {
    unsigned long long *d;
    ck(cudaMalloc(&d, NB * 8));
    ck(cudaMemset(d, 0, NB * 8));
    hash_blocks_any<<<NB * 16, 256>>>(x, n, d);
    ck(cudaGetLastError());
    ck(cudaMemcpy(hb, d, NB * 8, cudaMemcpyDeviceToHost));
    ck(cudaFree(d));
    unsigned long long t = 0;
    for (int b = 0; b < NB; ++b) {
        t ^= hb[b];
    }
    return t;
}
// our definition on the host (dref_rt.cu / realref_1024.cu)
static int ilogb_bits(unsigned long long a) {
    const int e = int(a >> 52);
    return e ? e - 1023 : (63 - __builtin_clzll(a)) - 1074;
}
static int shift_of(unsigned long long mx) {
    return mx == 0 ? 0 : 52 - ilogb_bits(mx);
}
static long long scale_round(double z, int shift) {
    unsigned long long b;
    memcpy(&b, &z, 8);
    const unsigned long long a = b & 0x7fffffffffffffffull;
    if (!a) {
        return 0;
    }
    const int e = int(a >> 52);
    const unsigned long long m = (a & ((1ull << 52) - 1)) | (e ? (1ull << 52) : 0ull);
    const int s = (e ? e : 1) - 1075 + shift;
    unsigned long long v;
    if (s >= 0) {
        v = m << s;
    } else if (s < -54) {
        v = 0;
    } else {
        const int d = -s;
        v = m >> d;
        const unsigned long long rem = m & ((1ull << d) - 1), half = 1ull << (d - 1);
        if (rem > half || (rem == half && (v & 1))) {
            ++v;
        }
    }
    return (b >> 63) ? -(long long)v : (long long)v;
}
static double round_scaled_host(unsigned __int128 x, bool neg, int exponent) {
    if (!x) {
        return 0.;
    }
    const unsigned long long hi = (unsigned long long)(x >> 64), lo = (unsigned long long)x;
    const int top = hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(lo);
    const int drop = std::max(0, top - 52);
    unsigned long long mant = (unsigned long long)(x >> drop);
    if (drop) {
        const unsigned __int128 rem = x - ((unsigned __int128)mant << drop),
                                half = (unsigned __int128)1 << (drop - 1);
        if (rem > half || (rem == half && (mant & 1))) {
            ++mant;
        }
    }
    const double v = std::ldexp(double(mant), drop + exponent);
    return neg ? -v : v;
}
static std::string last_error;

struct Ctx {
    l8rt::Plan plan;
    std::string info;
    int poison = 1;
};
} // namespace dvrt

extern "C" const char *d_last_error() {
    return dvrt::last_error.c_str();
}

static void *d_create_impl(int m, int n, int k, long long lda, long long ldb, long long ldc, const awe_engine_options *options) try {
    dvrt::last_error.clear();
    cudaDeviceProp prop{};
    int device = 0;
    ck(cudaGetDevice(&device));
    ck(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 10 || prop.minor != 3) {
        throw std::runtime_error("requires B300 sm_103a");
    }
    int runtime = 0;
    ck(cudaRuntimeGetVersion(&runtime));
    if (const char *s = awe_option_env(options, "AWE_FP64_BITS")) {
        if (atoi(s) != 53) {
            throw std::runtime_error("d_create: the library is built and judged for 53 bits");
        }
    }
    auto *c = new dvrt::Ctx;
    try {
        c->plan.init(m, n, k, lda, ldb, ldc, options);
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
        "{\"gpu\":\"%s\",\"cuda_runtime\":%d,\"complex\":false,\"m\":%d,\"n\":%d,\"k\":%d,\"lda\":%lld,\"ldb\":%lld,\"ldc\":%lld,"
        "\"quantize_bits\":53,\"engine\":\"l8rt\",\"variant\":59,\"moduli\":%d,\"parts\":%d,\"mma_products\":%d,"
        "\"planes_per_side\":%d,\"k_plan\":%d,\"accumulators_per_tile\":%d,\"product_bits\":%d,\"cuts\":%d,\"tiles\":%d,"
        "\"clusters\":%d,\"nchunk\":%d,\"syrk\":%d,\"trmm\":%d,\"pace_on\":%d,\"crt_chunks\":%d,\"crt_after_gemm\":%s,\"face_tiles_per_slot\":%d,\"slots\":%d,"
        "\"tile_m\":%d,\"tile_n\":%d,\"cta_group\":2,"
        "\"lambda\":\"per accumulator\",\"encoder\":\"%s\",\"norm_gate\":false,\"exactness_scope\":\"all_inputs_53bit\","
        "\"k_max\":16384,\"crt\":\"%s\",\"verify\":\"exact points on the host + hash\","
        "\"source\":\"Real matrix engine; exact host point checks and output hashes apply only to the recorded test shapes.\","
        "\"output\":\"column-major double, one exponent per row of A and column of B\"}",
        prop.name, runtime, m, n, k, lda, ldb, ldc, L8K_NMOD, kParts, L8K_PRODUCTS59, L8K_PLANES59,
        p.kplan, p.accs, pbits, p.sh.cuts, p.sh.tiles, p.sh.clusters, p.sh.nchunk, p.sh.syrk,
        p.sh.trmm, p.sh.pace_s >= 0 ? 1 : 0, p.crt_chunks, p.sh.nchunk > 1 ? "true" : "false",
        L8K_FTS, kSlots, Side, Cols,
        L8_FAST_ENC ? "l8e::FastEncoder (FX 25: padded staging, byte_perm packing, byte LUTs)" : "l8e::Encoder",
        p.sh.nchunk > 1
            ? "after the GEMM in crt_chunks column ranges (split below 2 tiles per cluster for CRT)"
            : "in the payout (PAYOUT_CRT=5)");
    c->info = buf;
    return c;
} catch (const std::exception &e) {
    dvrt::last_error = e.what();
    return nullptr;
}

extern "C" void *d_create_ld(int m, int n, int k, long long lda, long long ldb, long long ldc) {
    return d_create_impl(m, n, k, lda, ldb, ldc, nullptr);
}

extern "C" void *d_create_operation(int m, int n, int k, long long lda, long long ldb, long long ldc, int operation) {
    if (operation != 1 && operation != 2 && operation != 3) {
        dvrt::last_error = "operation is unsupported by this engine";
        return nullptr;
    }
    const awe_engine_options options{operation};
    return d_create_impl(m, n, k, lda, ldb, ldc, &options);
}

extern "C" void *d_create_configured(int m, int n, int k, long long lda, long long ldb, long long ldc, int operation, const awe_engine_configuration *configuration) {
    if (!awe_configuration_supported(configuration) || (operation != 1 && operation != 2 && operation != 3)) {
        dvrt::last_error = "unsupported operation or tuning configuration";
        return nullptr;
    }
    const awe_engine_options options{operation, true};
    return d_create_impl(m, n, k, lda, ldb, ldc, &options);
}

extern "C" int d_effective_options(void *h, awe_engine_effective *out) {
    if (!h || !out) return 1;
    const auto &p = static_cast<dvrt::Ctx *>(h)->plan;
    *out = {};
    out->pace = p.sh.pace_s;
    out->chunks = p.sh.nchunk;
    out->crt_chunks = p.crt_chunks;
    out->item_order = p.sh.item_order;
    out->pingpong = p.sh.pingpong;
    out->encoder_minb = -1;
    out->encoder_fast = -1;
#if ZL8
    out->encoder_fast = p.enc.fast;
    out->moduli = ZL8_NMODC;
#else
    out->moduli = L8K_NMOD;
#if L8_FAST_ENC
    out->encoder_minb = p.enc.minb;
    out->skip_b_padding = p.enc.skip_b_pad;
    out->pdl = p.enc.pdl;
    out->shift_form = p.enc.shift_form;
#if L8_EXTRA_FX_KNOB
    out->extra_fx = p.enc.extra_fx;
#endif
#endif
#endif
    return 0;
}

extern "C" void *d_create() {
    return d_create_ld(8192, 8192, 8192, 8192, 8192, 8192);
}

extern "C" const char *d_info(void *h) {
    return h ? static_cast<dvrt::Ctx *>(h)->info.c_str() : "";
}

extern "C" int d_fill_normal(void *x, long long n_doubles, unsigned long long seed) try {
    dvrt::last_error.clear();
    fill_normal<<<4096, 256>>>(reinterpret_cast<double *>(x), size_t(n_doubles), seed);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    return 0;
} catch (const std::exception &e) {
    dvrt::last_error = e.what();
    return 1;
}

extern "C" int d_run_reuse(void *h, const void *A, const void *B, void *C, int runs, double *stage,
                           unsigned reuse) try {
    dvrt::last_error.clear();
    auto *c = static_cast<dvrt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("d_run: no context");
    }
    l8rt::Plan &p = c->plan;
    for (int r = 0; r < runs; ++r) {
        if (c->poison) {
            ck(cudaMemset2DAsync(C, size_t(p.ldc) * 8, 0xa5, size_t(p.m) * 8, p.n, 0));
            if (!(reuse & 1u)) {
                ck(cudaMemsetAsync(p.pa, 0xa5, p.bytesA,
                                   0)); // a side asked to be reused keeps its planes
            }
            if (!(reuse & 2u)) {
                ck(cudaMemsetAsync(p.pb, 0xa5, p.bytesB, 0));
            }
            ck(cudaMemsetAsync(p.resid, 0xa5, p.resid_bytes, 0));
        }
        ck(cudaMemsetAsync(p.km, 0, size_t(p.sh.tiles) * 2 * sizeof(KernelMetrics), 0));
        ck(cudaMemsetAsync(p.mm, 0, size_t(p.sh.tiles) * kParts * 2 * sizeof(Metrics), 0));
        float st[3] = {0, 0, 0};
        if (p.run(reinterpret_cast<const double *>(A), reinterpret_cast<const double *>(B),
                  reinterpret_cast<double *>(C), 0, st, reuse) == 2) {
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
    dvrt::last_error = e.what();
    return 1;
}

extern "C" int d_run(void *h, const void *A, const void *B, void *C, int runs, double *stage) {
    return d_run_reuse(h, A, B, C, runs, stage,
                       0u); // one body; reuse 0 poisons both sides' planes as before
}

extern "C" int d_reuse_stats(void *h, unsigned long long *out3) {
    auto *c = static_cast<dvrt::Ctx *>(h);
    if (!c || !out3) {
        return 1;
    }
    out3[0] = c->plan.reused_a;
    out3[1] = c->plan.reused_b;
    out3[2] = c->plan.violations;
    return 0;
}

extern "C" int d_hash(const void *x, long long n_doubles, unsigned long long *total,
                      unsigned long long *blocks) try {
    dvrt::last_error.clear();
    unsigned long long hb[dvrt::NB];
    *total = dvrt::hash_any(reinterpret_cast<const double *>(x), size_t(n_doubles), hb);
    if (blocks) {
        memcpy(blocks, hb, sizeof(hb));
    }
    return 0;
} catch (const std::exception &e) {
    dvrt::last_error = e.what();
    return 1;
}

extern "C" int d_verify(void *h, const void *A, const void *B, const void *C, int npts,
                        unsigned long long *total, unsigned long long *blocks, int *spot_bad) try {
    dvrt::last_error.clear();
    auto *c = static_cast<dvrt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("d_verify: no context");
    }
    const l8rt::Plan &p = c->plan;
    const int pts = npts <= 0 ? 16 : std::min(npts, 1024);
    const double *dA = reinterpret_cast<const double *>(A);
    const double *dB = reinterpret_cast<const double *>(B);
    const double *dC = reinterpret_cast<const double *>(C);
    int bad = 0;
    std::vector<double> row(p.k), col(p.k);
    for (int s = 0; s < pts; ++s) {
        const int i = int((1ull + 2654435761ull * s) % unsigned(p.m)),
                  j = int((7ull + 40503ull * s) % unsigned(p.n));
        ck(cudaMemcpy2D(row.data(), 8, dA + i, size_t(p.lda) * 8, 8, p.k, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(col.data(), dB + size_t(j) * p.ldb, size_t(p.k) * 8, cudaMemcpyDeviceToHost));
        unsigned long long ma = 0, mb = 0, bits;
        for (int kk = 0; kk < p.k; ++kk) {
            memcpy(&bits, &row[kk], 8);
            ma = std::max(ma, bits & 0x7fffffffffffffffull);
            memcpy(&bits, &col[kk], 8);
            mb = std::max(mb, bits & 0x7fffffffffffffffull);
        }
        const int sa = dvrt::shift_of(ma), sb = dvrt::shift_of(mb);
        __int128 acc = 0;
        for (int kk = 0; kk < p.k; ++kk) {
            acc += (__int128)dvrt::scale_round(row[kk], sa) * dvrt::scale_round(col[kk], sb);
        }
        const bool neg = acc < 0;
        const double want = dvrt::round_scaled_host(
            neg ? (unsigned __int128)(-acc) : (unsigned __int128)acc, neg, -sa - sb);
        double got = 0;
        ck(cudaMemcpy(&got, dC + size_t(j) * p.ldc + i, 8, cudaMemcpyDeviceToHost));
        if (memcmp(&want, &got, 8)) {
            ++bad;
        }
    }
    *spot_bad = bad;
    // the hash of the valid m x n (compacted from ldc when ldc > m)
    const double *hsrc = dC;
    double *tmp = nullptr;
    if (p.ldc != p.m) {
        ck(cudaMalloc(&tmp, size_t(p.m) * p.n * 8));
        ck(cudaMemcpy2D(tmp, size_t(p.m) * 8, dC, size_t(p.ldc) * 8, size_t(p.m) * 8, p.n,
                        cudaMemcpyDeviceToDevice));
        hsrc = tmp;
    }
    unsigned long long hb[dvrt::NB];
    *total = dvrt::hash_any(hsrc, size_t(p.m) * p.n, hb);
    if (blocks) {
        memcpy(blocks, hb, sizeof(hb));
    }
    if (tmp) {
        cudaFree(tmp);
    }
    return 0;
} catch (const std::exception &e) {
    dvrt::last_error = e.what();
    return 1;
}

extern "C" int d_cycles(void *h, double *out) try {
    dvrt::last_error.clear();
    auto *c = static_cast<dvrt::Ctx *>(h);
    if (!c) {
        throw std::runtime_error("d_cycles: no context");
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
    out[7] = double(p.sh.cuts) * L8K_PRODUCTS59 * 112.0 / L8K_NMOD;
    return 0;
} catch (const std::exception &e) {
    dvrt::last_error = e.what();
    return 1;
}

extern "C" int d_poison(void *h, int on) {
    auto *c = static_cast<dvrt::Ctx *>(h);
    if (!c) {
        return -1;
    }
    const int was = c->poison;
    c->poison = on ? 1 : 0;
    return was;
}

extern "C" void d_destroy(void *h) {
    auto *c = static_cast<dvrt::Ctx *>(h);
    if (!c) {
        return;
    }
    c->plan.destroy();
    delete c;
}
