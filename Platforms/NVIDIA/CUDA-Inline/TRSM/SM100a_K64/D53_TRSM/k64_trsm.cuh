#pragma once
#include "k64_support.cuh"
#include <memory>
// Lower, left, no transpose, non-unit diagonal, alpha=1, default stream.
struct K64TrsmPlan {
    Engine &engine;
    BLAS &blas;
    int m, n, block, lda, ldb;
    std::vector<std::unique_ptr<Plan>> plans;
    std::unique_ptr<Mem> scratch;
    size_t workspace_delta = 0;
    K64TrsmPlan(Engine &e, BLAS &b, int m, int n, int block, int lda, int ldb)
        : engine(e), blas(b), m(m), n(n), block(block), lda(lda), ldb(ldb) {
        if (m < 0 || n < 0 || m > 4096 || n > 2048 || block < 1 || block > 16384 ||
            lda < std::max(1, m) || ldb < std::max(1, m)) {
            throw std::invalid_argument("shape/leading dimension outside experimental contract");
        }
        if (!m || !n) {
            return;
        }
        if (m > 128 && block < 128) {
            throw std::invalid_argument("block below 128 outside tested allocation contract");
        }
        size_t before, total;
        ck(cudaMemGetInfo(&before, &total));
        size_t estimate = size_t(m) * n * 8;
        for (int p = 0; p < m; p += block) {
            int w = std::min(block, m - p), rem = m - p - w;
            if (!rem) {
                continue;
            }
            size_t mp = (rem + 255) / 256 * 256, np = (n + 255) / 256 * 256,
                   kp = (w + 63) / 64 * 64;
            estimate += 128 * (mp + np) * kp + 256 * mp * np;
        }
        if (estimate > size_t(16) * 1024 * 1024 * 1024 || estimate > before * 3 / 4) {
            throw std::runtime_error("workspace estimate exceeds memory budget");
        }
        scratch = std::make_unique<Mem>(size_t(m) * n);
        for (int p = 0; p < m; p += block) {
            int w = std::min(block, m - p), rem = m - p - w;
            if (rem) {
                plans.push_back(std::make_unique<Plan>(e, rem, n, w, lda, ldb, rem));
            }
        }
        size_t after;
        ck(cudaMemGetInfo(&after, &total));
        workspace_delta = before >= after ? before - after : 0;
    }
    void run(const double *A, double *B, bool use_k64 = true) {
        if (!m || !n) {
            return;
        }
        if (!A || !B || A == B) {
            throw std::invalid_argument("nonnull distinct A/B required");
        }
        double one = 1;
        size_t pi = 0;
        for (int p = 0; p < m; p += block) {
            int w = std::min(block, m - p), rem = m - p - w;
            cb(cublasDtrsm(blas.h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           CUBLAS_DIAG_NON_UNIT, w, n, &one, A + p + size_t(p) * lda, lda, B + p,
                           ldb));
            if (rem) {
                ck(cudaDeviceSynchronize());
                if (use_k64) {
                    plans[pi++]->run(A + p + w + size_t(p) * lda, B + p, scratch->p);
                } else {
                    double zero = 0;
                    cb(cublasDgemm(blas.h, CUBLAS_OP_N, CUBLAS_OP_N, rem, n, w, &one,
                                   A + p + w + size_t(p) * lda, lda, B + p, ldb, &zero, scratch->p,
                                   rem));
                }
                subtract<<<(size_t(rem) * n + 255) / 256, 256>>>(B + p + w, scratch->p, rem, n,
                                                                 ldb);
                ck(cudaGetLastError());
            }
        }
        ck(cudaDeviceSynchronize());
    }
};
