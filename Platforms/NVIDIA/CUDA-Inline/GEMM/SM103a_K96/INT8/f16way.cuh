#pragma once
#include <cublasLt.h>
#ifdef F16WAY_CUBLAS
#include <cublas_v2.h>
#endif
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace f16way {
inline void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("f16way: ") + cudaGetErrorString(e));
}
inline void cl(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("f16way: cublasLt status " + std::to_string(int(s)));
}
__global__ void i8_to_h(const int8_t *__restrict__ src, long long ld, int rows, int k, int kp,
                        __half *__restrict__ dst) {
    const long long tot = (long long)rows * kp;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < tot;
         e += (long long)gridDim.x * blockDim.x) {
        const long long r = e / kp;
        const int q = int(e % kp);
        dst[e] = __int2half_rn(q < k ? int(src[r * ld + q]) : 0);
    }
}
__global__ void sum_planes(const float *__restrict__ pl, int nch, int m, int n, int32_t *__restrict__ out, int ldc) {
    const long long tot = (long long)m * n, pstride = tot;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < tot;
         e += (long long)gridDim.x * blockDim.x) {
        int acc = 0;
        for (int c = 0; c < nch; ++c)
            acc += __float2int_rn(__ldcs(pl + c * pstride + e));
        out[(e / m) * ldc + (e % m)] = acc;
    }
}
// SPLIT: B rows (k of each, stride ld) -> yh rows 0..rows-1 and yl rows rows..2 rows-1, each of kp (zeros after k)
__global__ void split_to_h(const int8_t *__restrict__ src, long long ld, int rows, int k, int kp,
                           __half *__restrict__ dst) {
    const long long tot = (long long)rows * kp;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < tot;
         e += (long long)gridDim.x * blockDim.x) {
        const long long r = e / kp;
        const int q = int(e % kp);
        const int y = q < k ? int(src[r * ld + q]) : 0, yh = (y + 8) >> 4, yl = y - 16 * yh;
        dst[e] = __int2half_rn(yh);
        dst[tot + e] = __int2half_rn(yl);
    }
}
// SPLIT: out (INT32, m x n, ld ldc) = 16 rint(C_h) + rint(C_l); c2 is m x 2n FP32 with ld m (C_h = columns 0..n-1, C_l
// = n..2n-1)
__global__ void combine_hl(const float *__restrict__ c2, int m, int n, int32_t *__restrict__ out, int ldc) {
    const long long tot = (long long)m * n;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < tot;
         e += (long long)gridDim.x * blockDim.x)
        out[(e / m) * ldc + (e % m)] = 16 * __float2int_rn(__ldcs(c2 + e)) + __float2int_rn(__ldcs(c2 + tot + e));
}
__global__ void count_ne(const int32_t *__restrict__ a, int lda_, const int32_t *__restrict__ b, int ldb_, int m, int n,
                         unsigned long long *out) {
    unsigned long long c = 0;
    const long long tot = (long long)m * n;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < tot;
         e += (long long)gridDim.x * blockDim.x) {
        const long long j = e / m, i = e % m;
        c += a[j * lda_ + i] != b[j * ldb_ + i];
    }
    atomicAdd(out, c);
}
__global__ void i8_to_h_v16(const int8_t *__restrict__ src, long long ld, int rows, int k16, int kp,
                            __half *__restrict__ dst) {
    for (int r = blockIdx.y; r < rows; r += gridDim.y)
        for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < k16; q += gridDim.x * blockDim.x) {
            const int4 v = __ldg(reinterpret_cast<const int4 *>(src + (long long)r * ld) + q);
            const int8_t *b = reinterpret_cast<const int8_t *>(&v);
            __align__(16) __half h[16];
#pragma unroll
            for (int i = 0; i < 16; ++i)
                h[i] = __int2half_rn(int(b[i]));
            uint4 *d = reinterpret_cast<uint4 *>(dst + (long long)r * kp + (long long)q * 16);
            d[0] = reinterpret_cast<const uint4 *>(h)[0];
            d[1] = reinterpret_cast<const uint4 *>(h)[1];
        }
}
__global__ void split_to_h_v16(const int8_t *__restrict__ src, long long ld, int rows, int k16, int kp,
                               __half *__restrict__ dst) {
    const long long tot = (long long)rows * kp;
    for (int r = blockIdx.y; r < rows; r += gridDim.y)
        for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < k16; q += gridDim.x * blockDim.x) {
            const int4 v = __ldg(reinterpret_cast<const int4 *>(src + (long long)r * ld) + q);
            const int8_t *b = reinterpret_cast<const int8_t *>(&v);
            __align__(16) __half hh[16], hl[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int y = b[i], yh = (y + 8) >> 4;
                hh[i] = __int2half_rn(yh);
                hl[i] = __int2half_rn(y - 16 * yh);
            }
            uint4 *d = reinterpret_cast<uint4 *>(dst + (long long)r * kp + (long long)q * 16);
            d[0] = reinterpret_cast<const uint4 *>(hh)[0];
            d[1] = reinterpret_cast<const uint4 *>(hh)[1];
            uint4 *e = reinterpret_cast<uint4 *>(dst + tot + (long long)r * kp + (long long)q * 16);
            e[0] = reinterpret_cast<const uint4 *>(hl)[0];
            e[1] = reinterpret_cast<const uint4 *>(hl)[1];
        }
}
__global__ void sum_planes_v4(const float *__restrict__ pl, int nch, long long mn4, int32_t *__restrict__ out) {
    const long long pstride4 = mn4; // planes are m n floats apart = mn4 float4s
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < mn4;
         e += (long long)gridDim.x * blockDim.x) {
        int4 acc = make_int4(0, 0, 0, 0);
        for (int c = 0; c < nch; ++c) {
            const float4 f = __ldcs(reinterpret_cast<const float4 *>(pl) + c * pstride4 + e);
            acc.x += __float2int_rn(f.x);
            acc.y += __float2int_rn(f.y);
            acc.z += __float2int_rn(f.z);
            acc.w += __float2int_rn(f.w);
        }
        reinterpret_cast<int4 *>(out)[e] = acc;
    }
}
__global__ void combine_hl_v4(const float *__restrict__ c2, long long mn4, int32_t *__restrict__ out) {
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < mn4;
         e += (long long)gridDim.x * blockDim.x) {
        const float4 h = __ldcs(reinterpret_cast<const float4 *>(c2) + e),
                     l = __ldcs(reinterpret_cast<const float4 *>(c2) + mn4 + e);
        reinterpret_cast<int4 *>(out)[e] =
            make_int4(16 * __float2int_rn(h.x) + __float2int_rn(l.x), 16 * __float2int_rn(h.y) + __float2int_rn(l.y),
                      16 * __float2int_rn(h.z) + __float2int_rn(l.z), 16 * __float2int_rn(h.w) + __float2int_rn(l.w));
    }
}
enum Form { CHUNK = 0, SPLIT = 1 };
inline const char *form_name(int f) { return f == CHUNK ? "chunk1024" : "split16"; }
struct Way {
    cublasLtHandle_t lt = nullptr;
    cudaStream_t s = nullptr;
    int M = 0, N = 0, K = 0, Kp = 0, nch = 0;
    bool use[2] = {true, true};
    size_t ws = 0;
    void *dws = nullptr;
    __half *ha = nullptr, *hb = nullptr;
    float *pl = nullptr;
    unsigned long long *dmm = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t L[2][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};
    cublasLtMatmulHeuristicResult_t hr[2][8];
    int nres[2] = {0, 0};
    int best_form = -1, best_index = -1;
    double best_ms = -1;
#ifdef F16WAY_CUBLAS
    cublasHandle_t cb = nullptr;
    void set_cublas(cublasHandle_t h) {
        cb = h;
        if (cb) {
            cl(cublasSetStream(cb, s));
            cl(cublasSetWorkspace(cb, dws, ws));
        }
    }
    bool has_cb() const { return cb != nullptr; }
#else
    bool has_cb() const { return false; }
#endif
    int ncand(int f) const { return nres[f] + (has_cb() ? 1 : 0); }
    bool is_cb(int f, int q) const { return has_cb() && q == nres[f]; }
    const char *api_name(int f, int q) const { return is_cb(f, q) ? "cublas" : "lt"; }
    void init(cublasLtHandle_t lt_, cudaStream_t s_, int M_, int N_, int K_, int forms = 3,
              size_t ws_bytes = size_t(256) << 20) {
        lt = lt_;
        s = s_;
        M = M_;
        N = N_;
        K = K_;
        Kp = (K + 1023) / 1024 * 1024;
        nch = Kp / 1024;
        ws = ws_bytes;
        use[CHUNK] = forms & 1;
        use[SPLIT] = (forms & 2) && Kp <= 16384;
        if (!use[CHUNK] && !use[SPLIT])
            throw std::runtime_error("f16way: no form left (SPLIT needs Kp <= 16,384)");
        // the exactness bounds this file relies on, checked where they are used
        static_assert(128LL * 128 * 1024 == (1LL << 24), "CHUNK: 2^14 * 1,024 = 2^24");
        static_assert(128LL * 8 * 16384 == (1LL << 24), "SPLIT: 1,024 * 16,384 = 2^24");
        ck(cudaMalloc(&dws, ws));
        ck(cudaMalloc(&ha, size_t(M) * Kp * 2));
        ck(cudaMalloc(&hb, size_t(N) * Kp * 2 * (use[SPLIT] ? 2 : 1)));
        size_t plb = 0;
        if (use[CHUNK])
            plb = size_t(nch) * M * N * 4;
        if (use[SPLIT] && size_t(2) * M * N * 4 > plb)
            plb = size_t(2) * M * N * 4;
        ck(cudaMalloc(&pl, plb));
        ck(cudaMalloc(&dmm, 8));
        ck(cudaMemset(ha, 0, size_t(M) * Kp * 2));
        ck(cudaMemset(hb, 0, size_t(N) * Kp * 2 * (use[SPLIT] ? 2 : 1)));
        cl(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cl(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
        cl(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
        cublasLtMatmulPreference_t pref;
        cl(cublasLtMatmulPreferenceCreate(&pref));
        cl(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
        if (use[CHUNK]) {
            cl(cublasLtMatrixLayoutCreate(&L[0][0], CUDA_R_16F, 1024, M, Kp));
            cl(cublasLtMatrixLayoutCreate(&L[0][1], CUDA_R_16F, 1024, N, Kp));
            cl(cublasLtMatrixLayoutCreate(&L[0][2], CUDA_R_32F, M, N, M));
            const long long st3[3] = {1024, 1024, (long long)M * N};
            for (int q = 0; q < 3; ++q) {
                cl(cublasLtMatrixLayoutSetAttribute(L[0][q], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &nch, sizeof nch));
                cl(cublasLtMatrixLayoutSetAttribute(L[0][q], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st3[q],
                                                    sizeof(long long)));
            }
            cl(cublasLtMatmulAlgoGetHeuristic(lt, op, L[0][0], L[0][1], L[0][2], L[0][2], pref, 8, hr[0], &nres[0]));
        }
        if (use[SPLIT]) {
            cl(cublasLtMatrixLayoutCreate(&L[1][0], CUDA_R_16F, Kp, M, Kp));
            cl(cublasLtMatrixLayoutCreate(&L[1][1], CUDA_R_16F, Kp, 2 * N, Kp));
            cl(cublasLtMatrixLayoutCreate(&L[1][2], CUDA_R_32F, M, 2 * N, M));
            cl(cublasLtMatmulAlgoGetHeuristic(lt, op, L[1][0], L[1][1], L[1][2], L[1][2], pref, 8, hr[1], &nres[1]));
        }
        cublasLtMatmulPreferenceDestroy(pref);
        if (nres[0] + nres[1] < 1)
            throw std::runtime_error("f16way: no cuBLASLt FP16 algorithm");
    }
    static bool v16_ok(const void *p, int k, int ld) {
        return k % 16 == 0 && ld % 16 == 0 && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
    }
    int last_copy_v16 = -1, last_int32_v4 = -1;
    bool prof = false;
    cudaEvent_t pev[4] = {nullptr, nullptr, nullptr, nullptr};
    void set_profiling(bool on) {
        if (on && !pev[0])
            for (auto &e : pev)
                ck(cudaEventCreate(&e));
        prof = on;
    }
    void last_parts(float *ms3) {
        if (!prof)
            throw std::runtime_error("f16way: last_parts() without set_profiling(1)");
        ck(cudaStreamSynchronize(s));
        for (int i = 0; i < 3; ++i)
            ck(cudaEventElapsedTime(ms3 + i, pev[i], pev[i + 1]));
    }
    void part_copy(int form, const int8_t *A, int lda, const int8_t *B, int ldb) {
        const bool va = v16_ok(A, K, lda), vb = v16_ok(B, K, ldb);
        const int k16 = K / 16, bx = (k16 + 255) / 256;
        if (va)
            i8_to_h_v16<<<dim3(bx, M < 65535 ? M : 65535), 256, 0, s>>>(A, lda, M, k16, Kp, ha);
        else
            i8_to_h<<<4096, 256, 0, s>>>(A, lda, M, K, Kp, ha);
        if (form == CHUNK) {
            if (vb)
                i8_to_h_v16<<<dim3(bx, N < 65535 ? N : 65535), 256, 0, s>>>(B, ldb, N, k16, Kp, hb);
            else
                i8_to_h<<<4096, 256, 0, s>>>(B, ldb, N, K, Kp, hb);
        } else {
            if (vb)
                split_to_h_v16<<<dim3(bx, N < 65535 ? N : 65535), 256, 0, s>>>(B, ldb, N, k16, Kp, hb);
            else
                split_to_h<<<4096, 256, 0, s>>>(B, ldb, N, K, Kp, hb);
        }
        last_copy_v16 = (va && vb) ? 1 : (va || vb) ? 2 : 0;
        ck(cudaGetLastError());
    }
    cublasStatus_t part_gemm(int form, int q) {
        const float one = 1.f, zero = 0.f;
#ifdef F16WAY_CUBLAS
        if (is_cb(form, q)) {
            if (form == CHUNK)
                return cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, 1024, &one, ha, CUDA_R_16F, Kp,
                                                  1024, hb, CUDA_R_16F, Kp, 1024, &zero, pl, CUDA_R_32F, M,
                                                  (long long)M * N, nch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            return cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, 2 * N, Kp, &one, ha, CUDA_R_16F, Kp, hb, CUDA_R_16F,
                                Kp, &zero, pl, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
#endif
        if (q < 0 || q >= nres[form])
            return CUBLAS_STATUS_INVALID_VALUE;
        return cublasLtMatmul(lt, op, &one, ha, L[form][0], hb, L[form][1], &zero, pl, L[form][2], pl, L[form][2],
                              &hr[form][q].algo, dws, ws, s);
    }
    void part_int32(int form, int32_t *C, int ldc) {
        const long long mn = (long long)M * N;
        const bool v4 = ldc == M && mn % 4 == 0 && (reinterpret_cast<uintptr_t>(C) & 15) == 0;
        if (v4) {
            if (form == CHUNK)
                sum_planes_v4<<<4096, 256, 0, s>>>(pl, nch, mn / 4, C);
            else
                combine_hl_v4<<<4096, 256, 0, s>>>(pl, mn / 4, C);
        } else {
            if (form == CHUNK)
                sum_planes<<<4096, 256, 0, s>>>(pl, nch, M, N, C, ldc);
            else
                combine_hl<<<4096, 256, 0, s>>>(pl, M, N, C, ldc);
        }
        last_int32_v4 = v4 ? 1 : 0;
        ck(cudaGetLastError());
    }
    // the whole way with candidate (form, q): copies + GEMM + INT32 into C (ldc >= M); returns the cuBLASLt status
    // (launch errors are thrown)
    cublasStatus_t run_q(int form, int q, const int8_t *A, int lda, const int8_t *B, int ldb, int32_t *C, int ldc) {
        if (prof)
            ck(cudaEventRecord(pev[0], s));
        part_copy(form, A, lda, B, ldb);
        if (prof)
            ck(cudaEventRecord(pev[1], s));
        const cublasStatus_t st = part_gemm(form, q);
        if (prof)
            ck(cudaEventRecord(pev[2], s));
        part_int32(form, C, ldc);
        if (prof)
            ck(cudaEventRecord(pev[3], s));
        return st;
    }
    void stages(const int8_t *A, int lda, const int8_t *B, int ldb, int32_t *C, int ldc, int reps, const char *tag,
                FILE *out = stdout) {
        if (best_form < 0)
            throw std::runtime_error("f16way: stages() before sweep()");
        const int f = best_form, q = best_index;
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0));
        ck(cudaEventCreate(&e1));
        cl(run_q(f, q, A, lda, B, ldb, C, ldc));
        ck(cudaStreamSynchronize(s));
        double med[3];
        for (int part = 0; part < 3; ++part) {
            float v[64];
            const int R = reps < 1 ? 1 : (reps > 64 ? 64 : reps);
            for (int r = 0; r < R; ++r) {
                ck(cudaEventRecord(e0, s));
                if (part == 0)
                    part_copy(f, A, lda, B, ldb);
                else if (part == 1)
                    cl(part_gemm(f, q));
                else
                    part_int32(f, C, ldc);
                ck(cudaEventRecord(e1, s));
                ck(cudaEventSynchronize(e1));
                ck(cudaEventElapsedTime(&v[r], e0, e1));
            }
            for (int a = 1; a < R; ++a)
                for (int b = a; b > 0 && v[b - 1] > v[b]; --b) {
                    const float t = v[b];
                    v[b] = v[b - 1];
                    v[b - 1] = t;
                } // no <algorithm> (nvcc 11.5 + gcc 11)
            med[part] = R % 2 ? v[R / 2] : 0.5 * (v[R / 2 - 1] + v[R / 2]);
        }
        ck(cudaEventDestroy(e0));
        ck(cudaEventDestroy(e1));
        const double mn = double(M) * N, planes = f == CHUNK ? double(nch) : 2.0;
        const double copy_b = double(M + N) * K + double(M + (f == CHUNK ? N : 2 * N)) * Kp * 2,
                     gemm_b = planes * mn * 4, int32_b = planes * mn * 4 + mn * 4;
        fprintf(
            out,
            "{\"kind\":\"f16stages\",\"tag\":\"%s\",\"form\":\"%s\",\"index\":%d,\"reps\":%d,\"copy_ms\":%.6f,\"gemm_ms\":%.6f,\"int32_ms\":%.6f,\"parts_sum_ms\":%.6f,"
            "\"planes\":%d,\"copy_bytes\":%.0f,\"planes_bytes\":%.0f,\"int32_bytes\":%.0f,\"c_bytes\":%.0f,\"m\":%d,\"n\":%d,\"k\":%d,\"kp\":%d,\"copy_path\":\"%s\",\"int32_path\":\"%s\",\"api\":\"%s\",\"cublas_candidates\":%d}\n",
            tag, form_name(f), q, reps < 1 ? 1 : (reps > 64 ? 64 : reps), med[0], med[1], med[2],
            med[0] + med[1] + med[2], int(planes), copy_b, gemm_b, int32_b, mn * 4, M, N, K, Kp,
            last_copy_v16 == 1   ? "v16"
            : last_copy_v16 == 2 ? "mixed"
                                 : "generic",
            last_int32_v4 == 1 ? "v4" : "generic", api_name(f, q), has_cb() ? int(use[CHUNK]) + int(use[SPLIT]) : 0);
        fflush(out);
    }
    void run(const int8_t *A, int lda, const int8_t *B, int ldb, int32_t *C,
             int ldc) { // the chosen candidate (after sweep)
        if (best_form < 0)
            throw std::runtime_error("f16way: run() before sweep()");
        cl(run_q(best_form, best_index, A, lda, B, ldb, C, ldc));
    }
    unsigned long long mismatches(const int32_t *C, int ldc, const int32_t *R, int ldr) {
        ck(cudaStreamSynchronize(s));
        ck(cudaMemset(dmm, 0, 8));
        count_ne<<<1184, 256>>>(C, ldc, R, ldr, M, N, dmm);
        ck(cudaGetLastError());
        unsigned long long mm = 0;
        ck(cudaMemcpy(&mm, dmm, 8, cudaMemcpyDeviceToHost));
        return mm;
    }
    int algo_attr(int form, int q, cublasLtMatmulAlgoConfigAttributes_t w) {
        int v = -1;
        size_t got = 0;
        if (q < 0 || q >= nres[form])
            return -1;
        cublasLtMatmulAlgoConfigGetAttribute(&hr[form][q].algo, w, &v, sizeof v, &got);
        return v;
    }
    // every candidate of every enabled form: poison, one whole way, compare with R; sweep_ms back to back; compare
    // again. Rows go to `out` with `tag` as their "tag" (not "case": harnesses read a "case" key as the start of a
    // case). Throws if no candidate is exact. C is a scratch INT32 output (ldc >= M) the sweep may overwrite.
    void sweep(const int8_t *A, int lda, const int8_t *B, int ldb, int32_t *C, int ldc, const int32_t *R, int ldr,
               int sweep_ms, const char *tag, FILE *out = stdout) {
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0));
        ck(cudaEventCreate(&e1));
        best_form = best_index = -1;
        best_ms = 1e30;
        for (int f = 0; f < 2; ++f) {
            if (!use[f])
                continue;
            for (int q = 0; q < ncand(f); ++q) {
                ck(cudaMemsetAsync(C, 0xFF, size_t(ldc) * N * 4, s));
                const bool ok = run_q(f, q, A, lda, B, ldb, C, ldc) == CUBLAS_STATUS_SUCCESS;
                const long long mm = ok ? (long long)mismatches(C, ldc, R, ldr) : -1;
                double ms = -1;
                long long calls = 0, mm2 = -1;
                if (ok) {
                    const double t0 =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
                            .count();
                    ck(cudaEventRecord(e0, s));
                    for (;;) {
                        for (int r = 0; r < 16; ++r)
                            cl(run_q(f, q, A, lda, B, ldb, C, ldc));
                        calls += 16;
                        ck(cudaStreamSynchronize(s));
                        if (std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                    .count() -
                                t0 >=
                            double(sweep_ms))
                            break;
                    }
                    ck(cudaEventRecord(e1, s));
                    ck(cudaEventSynchronize(e1));
                    float ev = 0;
                    ck(cudaEventElapsedTime(&ev, e0, e1));
                    ms = double(ev) / calls;
                    mm2 = (long long)mismatches(C, ldc, R, ldr);
                    if (mm == 0 && mm2 == 0 && ms < best_ms) {
                        best_ms = ms;
                        best_form = f;
                        best_index = q;
                    }
                }
                fprintf(
                    out,
                    "{\"kind\":\"f16cand\",\"tag\":\"%s\",\"form\":\"%s\",\"index\":%d,\"api\":\"%s\",\"ok\":%s,\"algo_id\":%d,\"tile\":%d,\"stages\":%d,\"splitk\":%d,\"reduction\":%d,\"swizzle\":%d,"
                    "\"workspace\":%zu,\"calls\":%lld,\"ms_per_call\":%.6f,\"mismatches_first\":%lld,\"mismatches_after\":%lld}\n",
                    tag, form_name(f), q, api_name(f, q), ok ? "true" : "false",
                    algo_attr(f, q, CUBLASLT_ALGO_CONFIG_ID), algo_attr(f, q, CUBLASLT_ALGO_CONFIG_TILE_ID),
                    algo_attr(f, q, CUBLASLT_ALGO_CONFIG_STAGES_ID), algo_attr(f, q, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
                    algo_attr(f, q, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
                    algo_attr(f, q, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING), is_cb(f, q) ? ws : hr[f][q].workspaceSize,
                    calls, ms, mm, mm2);
            }
        }
        ck(cudaEventDestroy(e0));
        ck(cudaEventDestroy(e1));
        if (best_form < 0)
            throw std::runtime_error("f16way: no exact FP16 candidate");
        fprintf(
            out,
            "{\"kind\":\"f16best\",\"tag\":\"%s\",\"form\":\"%s\",\"index\":%d,\"ms_per_call\":%.6f,\"m\":%d,\"n\":%d,\"k\":%d,\"kp\":%d,\"chunks\":%d,\"candidates_chunk\":%d,\"candidates_split\":%d,\"sweep_ms\":%d,\"api\":\"%s\",\"cublas_candidates\":%d}\n",
            tag, form_name(best_form), best_index, best_ms, M, N, K, Kp, use[CHUNK] ? nch : 0, use[CHUNK] ? nres[0] : 0,
            use[SPLIT] ? nres[1] : 0, sweep_ms, api_name(best_form, best_index),
            has_cb() ? int(use[CHUNK]) + int(use[SPLIT]) : 0);
        fflush(out);
    }
    void destroy() {
        for (auto &f : L)
            for (auto &l : f)
                if (l) {
                    cublasLtMatrixLayoutDestroy(l);
                    l = nullptr;
                }
        if (op)
            cublasLtMatmulDescDestroy(op);
        op = nullptr;
        for (auto &e : pev)
            if (e) {
                cudaEventDestroy(e);
                e = nullptr;
            }
        cudaFree(dws);
        cudaFree(ha);
        cudaFree(hb);
        cudaFree(pl);
        cudaFree(dmm);
        dws = ha = hb = nullptr;
        pl = nullptr;
        dmm = nullptr;
    }
};
} // namespace f16way
