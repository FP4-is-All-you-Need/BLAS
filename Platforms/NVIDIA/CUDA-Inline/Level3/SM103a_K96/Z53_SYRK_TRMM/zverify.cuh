// Shared output checks for the B300 result and an independently run complex reference.
// Reference values are computed off B300; matching hash definitions are not by themselves
// evidence that a new engine or a new shape produces the same output.
//   hash: XOR over i of (bits(x[i]) * C1 + i * C2), matching compare_out, kept
// per 1/256 of the array so that a mismatch is located; the XOR of the 256 is the whole-array hash.
// spot: the quantized integers of both sides (the shifts of the run) and the exact Re / Im at a few points on the CPU,
// rounded once, compared with the output bytes.
// Needs before inclusion: ck, fp64k::M / N / K, qint:: (quant_int.cuh).
#pragma once
#include <vector>
#include <cstring>
#include <cmath>

namespace zver {
constexpr int NB = 256;

__global__ void __launch_bounds__(256)
    hash_blocks(const double *__restrict__ x, size_t n, unsigned long long *out) {
    const int g_per = gridDim.x / NB;
    const int b = blockIdx.x / g_per, g = blockIdx.x % g_per;
    const size_t per = n / NB, lo = size_t(b) * per, hi = lo + per;
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

// hb[NB] = block hashes of x[0, n); returns their XOR (= compare_out's hash of the whole array)
static unsigned long long hash_array(const double *x, size_t n, unsigned long long *hb) {
    if (n % NB) {
        throw std::runtime_error("hash_array: n not a multiple of 256");
    }
    unsigned long long *d;
    ck(cudaMalloc(&d, NB * 8));
    ck(cudaMemset(d, 0, NB * 8));
    hash_blocks<<<NB * 16, 256>>>(x, n, d);
    ck(cudaGetLastError());
    ck(cudaMemcpy(hb, d, NB * 8, cudaMemcpyDeviceToHost));
    ck(cudaFree(d));
    unsigned long long t = 0;
    for (int b = 0; b < NB; ++b) {
        t ^= hb[b];
    }
    return t;
}

// the quantized integers of one side: out[o * k + q] (A: element (o, q) at q * ld + o; B: at o * ld + q)
__global__ void quant_c(const double2 *__restrict__ x, const int *__restrict__ shifts,
                        long long *__restrict__ qr, long long *__restrict__ qi, int outer, int k,
                        int64_t ld, int side_b) {
    const size_t n = size_t(outer) * k;
    for (size_t e = size_t(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += size_t(gridDim.x) * blockDim.x) {
        const int o = int(e / k), q = int(e % k);
        const double2 z = side_b ? x[size_t(o) * ld + q] : x[size_t(q) * ld + o];
        const int sh = shifts[o];
        qr[e] = qint::finite_bits(qint::absbits(z.x)) ? qint::scale_round(z.x, sh) : 0;
        qi[e] = qint::finite_bits(qint::absbits(z.y)) ? qint::scale_round(z.y, sh) : 0;
    }
}

// the kernel's rounding of an exact integer times 2^exponent, on the host
static double host_round_i128(__int128 v, int exponent) {
    if (v == 0) {
        return 0.0;
    }
    const bool neg = v < 0;
    unsigned __int128 x = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
    int top = 127;
    while (!((x >> top) & 1)) {
        --top;
    }
    const int drop = top > 52 ? top - 52 : 0;
    unsigned long long mant = (unsigned long long)(x >> drop);
    if (drop) {
        const unsigned __int128 rem = x - ((unsigned __int128)mant << drop),
                                half = (unsigned __int128)1 << (drop - 1);
        if (rem > half || (rem == half && (mant & 1))) {
            ++mant;
        }
    }
    const double v2 = std::ldexp(double(mant), drop + exponent);
    return neg ? -v2 : v2;
}

// quantizes both sides into q[4] (Count long longs each; allocated here when null) with the run's shifts
static void quantize_both(const double2 *A, const double2 *B, const int *ea, const int *eb,
                          long long *q[4]) {
    using namespace fp64k;
    for (int s = 0; s < 4; ++s) {
        if (!q[s]) {
            ck(cudaMalloc(&q[s], size_t(M) * K * 8));
        }
    }
    quant_c<<<4096, 256>>>(A, ea, q[0], q[1], M, K, M, 0);
    quant_c<<<4096, 256>>>(B, eb, q[2], q[3], N, K, K, 1);
    ck(cudaGetLastError());
}

// exact Re / Im at npts points on the CPU from q[4], compared with out (column-major double2); returns the mismatches
static int spot_check(long long *const q[4], const int *ea, const int *eb, const double *out,
                      int npts) {
    using namespace fp64k;
    std::vector<int> hea(M), heb(N);
    ck(cudaMemcpy(hea.data(), ea, M * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(heb.data(), eb, N * 4, cudaMemcpyDeviceToHost));
    std::vector<long long> xr(K), xi(K), yr(K), yi(K);
    int bad = 0;
    for (int s = 0; s < npts; ++s) {
        const int i = (s * 2654435761u + 17) % M, j = (s * 40503u + 91) % N;
        ck(cudaMemcpy(xr.data(), q[0] + size_t(i) * K, K * 8, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(xi.data(), q[1] + size_t(i) * K, K * 8, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(yr.data(), q[2] + size_t(j) * K, K * 8, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(yi.data(), q[3] + size_t(j) * K, K * 8, cudaMemcpyDeviceToHost));
        __int128 re = 0, im = 0;
        for (int k = 0; k < K; ++k) {
            re += (__int128)xr[k] * yr[k] - (__int128)xi[k] * yi[k];
            im += (__int128)xr[k] * yi[k] + (__int128)xi[k] * yr[k];
        }
        const int ex = -hea[i] - heb[j];
        double g[2];
        ck(cudaMemcpy(g, out + 2 * (size_t(j) * M + i), 16, cudaMemcpyDeviceToHost));
        const double h[2] = {host_round_i128(re, ex), host_round_i128(im, ex)};
        if (memcmp(g, h, 16) != 0) {
            ++bad;
        }
    }
    return bad;
}

// Runtime checks accept m, n, k and leading dimensions. hash_array_any splits x into
// 256 blocks of ceil(n / 256) (the last ones shorter or empty); for n a multiple of 256 it gives hash_array's values exactly.
__global__ void __launch_bounds__(256)
    hash_blocks_any(const double *__restrict__ x, size_t n, unsigned long long *out) {
    const int g_per = gridDim.x / NB;
    const int b = blockIdx.x / g_per, g = blockIdx.x % g_per;
    const size_t per = (n + NB - 1) / NB, lo = size_t(b) * per, hi = lo + per < n ? lo + per : n;
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
static unsigned long long hash_array_any(const double *x, size_t n, unsigned long long *hb) {
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
// A is m x k column-major with lda >= m (element (i, q) at q * lda + i); B is k x n with ldb >= k (element (q, j) at j * ldb + q)
static void quantize_both_rt(const double2 *A, const double2 *B, const int *ea, const int *eb,
                             long long *q[4], int m, int n, int k, int64_t lda, int64_t ldb) {
    for (int s = 0; s < 2; ++s) {
        if (!q[s]) {
            ck(cudaMalloc(&q[s], size_t(m) * k * 8));
        }
    }
    for (int s = 2; s < 4; ++s) {
        if (!q[s]) {
            ck(cudaMalloc(&q[s], size_t(n) * k * 8));
        }
    }
    quant_c<<<4096, 256>>>(A, ea, q[0], q[1], m, k, lda, 0);
    quant_c<<<4096, 256>>>(B, eb, q[2], q[3], n, k, ldb, 1);
    ck(cudaGetLastError());
}
static int spot_check_rt(long long *const q[4], const int *ea, const int *eb, const double *out,
                         int npts, int m, int n, int k, int64_t ldc) {
    std::vector<int> hea(m), heb(n);
    ck(cudaMemcpy(hea.data(), ea, size_t(m) * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(heb.data(), eb, size_t(n) * 4, cudaMemcpyDeviceToHost));
    std::vector<long long> xr(k), xi(k), yr(k), yi(k);
    int bad = 0;
    for (int s = 0; s < npts; ++s) {
        const int i = int((s * 2654435761u + 17) % unsigned(m)),
                  j = int((s * 40503u + 91) % unsigned(n));
        ck(cudaMemcpy(xr.data(), q[0] + size_t(i) * k, size_t(k) * 8, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(xi.data(), q[1] + size_t(i) * k, size_t(k) * 8, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(yr.data(), q[2] + size_t(j) * k, size_t(k) * 8, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(yi.data(), q[3] + size_t(j) * k, size_t(k) * 8, cudaMemcpyDeviceToHost));
        __int128 re = 0, im = 0;
        for (int t = 0; t < k; ++t) {
            re += (__int128)xr[t] * yr[t] - (__int128)xi[t] * yi[t];
            im += (__int128)xr[t] * yi[t] + (__int128)xi[t] * yr[t];
        }
        const int ex = -hea[i] - heb[j];
        double g[2];
        ck(cudaMemcpy(g, out + 2 * (size_t(j) * ldc + i), 16, cudaMemcpyDeviceToHost));
        const double h[2] = {host_round_i128(re, ex), host_round_i128(im, ex)};
        if (memcmp(g, h, 16) != 0) {
            ++bad;
        }
    }
    return bad;
}

static void print_hashes(const char *kind, unsigned long long total, const unsigned long long *hb) {
    printf("{\"kind\":\"%s\",\"total\":\"%016llx\",\"blocks\":[", kind, total);
    for (int b = 0; b < NB; ++b) {
        printf("%s\"%016llx\"", b ? "," : "", hb[b]);
    }
    printf("]}\n");
}
} // namespace zver
