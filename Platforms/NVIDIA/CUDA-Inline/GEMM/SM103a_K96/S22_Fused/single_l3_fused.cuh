#pragma once
#include "s3_encode_view.cuh"
#include <chrono>
#include <memory>
namespace s3f {
static thread_local std::string error;
__global__ void shift_view(const float *x, int64_t ld, int *exponents, int *invalid, int outer,
                           int k, int bits, int side, S3InputView view) {
    __shared__ unsigned long long values[256];
    const int row = blockIdx.x, t = threadIdx.x;
    unsigned long long mx = 0;
    for (int p = t; p < k; p += 256) {
        const size_t index = side ? size_t(row) * ld + p : size_t(p) * ld + row;
        const auto a = l8e::abs64(view.load(x, index, ld, side));
        if (!l8e::rfinite(a)) {
            atomicExch(invalid, 1);
        } else {
            mx = max(mx, a);
        }
    }
    values[t] = mx;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) {
            values[t] = max(values[t], values[t + s]);
        }
        __syncthreads();
    }
    if (t == 0 && row < outer) {
        exponents[row] = values[0] ? bits - 1 - qint::ilogb_bits(values[0]) : 0;
    }
}
// Decode the same tile-major residues as the inherited CRT, rounding each
// product before the final SSYR2K addition. All residues are complete here.
__device__ __forceinline__ float element(const ResT *residues, const int *ea, const int *eb,
                                         unsigned i, unsigned j, L8Shape sh) {
    const unsigned tr = i / Side, tc = j / Cols, band = tr / 8;
    const unsigned rb = min(8u, unsigned(sh.tile_rows) - band * 8);
    const unsigned tile = band * (8u * sh.tile_cols) + tc * rb + tr - band * 8;
    const size_t offset = (size_t(tile) * Cols + j % Cols) * Side + i % Side;
    unsigned r[kParts];
#pragma unroll
    for (int q = 0; q < kParts; ++q) {
        r[q] = __ldcg(residues + offset + size_t(q) * sh.plane);
    }
    return __uint_as_float(crtf_element_bits(r, crtf_c, -ea[i] - eb[j]));
}
__global__ void syr2k_crt(const unsigned short *residues, const int *ea, const int *eb, float *C,
                          L8Shape sh) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < size_t(sh.m) * sh.n;
         index += size_t(gridDim.x) * blockDim.x) {
        const unsigned i = index % sh.m, j = index / sh.m;
        if (i >= j) {
            const auto *r = reinterpret_cast<const ResT *>(residues);
            C[size_t(j) * sh.ldc + i] =
                __fadd_rn(element(r, ea, eb, i, j, sh), element(r, ea, eb, j, i, sh));
        }
    }
}
struct Context {
    void *f = nullptr;
    int routine, n, k, param;
    unsigned *maps[2] = {};
    int counts[2] = {}, gemms = 1;
    std::string info;
    ~Context() {
        for (auto p : maps) {
            if (p) {
                cudaFree(p);
            }
        }
        if (f) {
            f_destroy(f);
        }
    }
    l8rt::Plan &plan() {
        return static_cast<fvrt::Ctx *>(f)->plan;
    }
    void set_maps() {
        auto &p = plan();
        std::vector<unsigned> host[2];
        const int half = (p.sh.tile_cols + 1) / 2;
        for (unsigned tile = 0; tile < unsigned(p.sh.tiles); ++tile) {
            const unsigned band = tile / (8u * p.sh.tile_cols),
                           within = tile - band * 8u * p.sh.tile_cols;
            const unsigned rb = std::min(8u, unsigned(p.sh.tile_rows) - band * 8);
            const unsigned tr = band * 8 + within % rb, tc = within / rb;
            if (routine == 0 && std::min(unsigned(n), (tr + 1) * Side) <= tc * Cols) {
                continue;
            }
            host[routine == 0 && param == 2 && int(tc) >= half ? 1 : 0].push_back(tile);
        }
        gemms = routine == 0 && param == 2 ? 2 : 1;
        for (int j = 0; j < gemms; ++j) {
            counts[j] = int(host[j].size());
            if (!counts[j]) {
                throw std::runtime_error("empty tile partition");
            }
            ck(cudaMalloc(&maps[j], host[j].size() * sizeof(unsigned)));
            ck(cudaMemcpy(maps[j], host[j].data(), host[j].size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
        }
    }
};
// Use inherited fast encoding arithmetic/LUTs; the only operand change is
// view.load at the original reads. No PDL dependencies for these plain grids.
inline void encode(l8rt::Plan &p, const float *x, int64_t ld, int side, S3InputView view) {
    l8e::launch_s3_x<0, true, l8e::kFastFx, 3, float>(
        p.enc, x, ld, side ? p.eb : p.ea, side ? p.pb : p.pa, side, 0, p.enc.dlutx[side], view);
}
inline void run(Context &c, const float *A, const float *B, float *C, double *stage) {
    auto &p = c.plan();
    const auto start = std::chrono::steady_clock::now();
    ck(cudaEventRecord(p.ev[0]));
    if (l8rt::crt_owner() != &p) {
        ck(cudaMemcpyToSymbolAsync(crtf_c, &p.hcf, sizeof(p.hcf), 0, cudaMemcpyHostToDevice, 0));
        l8rt::crt_owner() = &p;
    }
    ck(cudaMemsetAsync(p.bad, 0, 4));
    S3InputView av{c.routine == 1 ? 2 : c.routine == 3 ? 3 : 0, c.n};
    S3InputView bv{c.routine == 0 || c.routine == 2 ? 1 : 0, c.n};
    const float *right = c.routine == 0 ? A : B;
    shift_view<<<c.n, 256>>>(A, c.n, p.ea, p.bad, c.n, c.k, p.bits, 0, av);
    if (c.routine == 0) {
        ck(cudaMemcpyAsync(p.eb, p.ea, size_t(c.n) * sizeof(int), cudaMemcpyDeviceToDevice));
    } else {
        shift_view<<<c.n, 256>>>(right, p.ldb, p.eb, p.bad, c.n, c.k, p.bits, 1, bv);
    }
    ck(cudaEventRecord(p.ev[1]));
    encode(p, A, c.n, 0, av);
    encode(p, right, p.ldb, 1, bv);
    ck(cudaEventRecord(p.ev[2]));
    for (int j = 0; j < c.gemms; ++j) {
        auto sh = p.sh;
        auto cfg = p.cfg;
        sh.s3_tiles = c.maps[j];
        sh.tiles = c.counts[j];
        sh.whole = sh.tiles;
        sh.nchunk = 1;
        sh.clusters = std::min(GridClusters, sh.tiles);
        sh.pace_s = -1;
        sh.s3_lower = c.routine == 0;
        sh.s3_triangular_a = c.routine == 1;
        sh.s3_defer_crt = c.routine == 2;
        cfg.gridDim = dim3(2 * sh.clusters);
        cfg.stream = 0;
        ck(cudaLaunchKernelEx(&cfg, l8k_gemm_kernel_rt<true>, (const unsigned char *)p.pa,
                              (const unsigned char *)p.pb, p.resid, (const L8kMod *)p.dmods,
                              p.block_tiles, p.km, p.mm, p.crt_state,
                              (const f64_impl::U128 *)nullptr, C, (const int *)p.ea,
                              (const int *)p.eb, p.P, p.tmA, p.tmB, sh));
    }
    if (c.routine == 2) {
        syr2k_crt<<<4096, 256>>>(p.resid, p.ea, p.eb, C, p.sh);
    }
    ck(cudaGetLastError());
    ck(cudaEventRecord(p.ev[3]));
    ck(cudaEventSynchronize(p.ev[3]));
    float st[3];
    for (int i = 0; i < 3; ++i) {
        ck(cudaEventElapsedTime(st + i, p.ev[i], p.ev[i + 1]));
    }
    ck(cudaMemcpy(p.hbad, p.bad, 4, cudaMemcpyDeviceToHost));
    if (*p.hbad) {
        throw std::runtime_error("nonfinite input");
    }
    stage[0] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    stage[1] = 0;
    stage[2] = double(st[0]) + st[1] + st[2];
    stage[3] = 0;
    stage[4] = st[1];
    stage[5] = st[2];
}
}
extern "C" void *s3_create(int routine, int n, int k, int param) try {
    s3f::error.clear();
    if (routine < 0 || routine > 3 || n < 1 || k < 1 || k > 16384) {
        throw std::runtime_error("invalid routine/shape");
    }
    if ((routine == 0 && (param != 1 && param != 2)) || (routine == 2 && param != 3) ||
        ((routine == 1 || routine == 3) && (param != 1 || k != n)) ||
        (routine == 0 && param == 2 && (n % 2 || n <= 224))) {
        throw std::runtime_error("unsupported param/shape");
    }
    auto c = std::make_unique<s3f::Context>();
    c->routine = routine;
    c->n = n;
    c->k = k;
    c->param = param;
    c->f = f_create_ld(n, n, k, n, k, n);
    if (!c->f) {
        throw std::runtime_error(f_last_error());
    }
    c->set_maps();
    c->info =
        "{\"routine\":" + std::to_string(routine) + ",\"n\":" + std::to_string(n) +
        ",\"k\":" + std::to_string(k) + ",\"param\":" + std::to_string(param) +
        ",\"strips\":" + std::to_string(c->gemms) +
        ",\"fused\":true,\"encoded_once_per_call\":true,\"temporary_float_matrices\":0,\"engine\":" +
        f_info(c->f) + "}";
    return c.release();
} catch (const std::exception &e) {
    s3f::error = e.what();
    return nullptr;
}
extern "C" int s3_run(void *h, const void *A, const void *B, void *C, int runs, double *stage) try {
    s3f::error.clear();
    if (!h || !A || !C || !stage || runs < 1 || C == A || C == B) {
        throw std::runtime_error("invalid run arguments");
    }
    auto &c = *static_cast<s3f::Context *>(h);
    if (c.routine != 0 && !B) {
        throw std::runtime_error("B required");
    }
    for (int i = 0; i < runs; ++i) {
        s3f::run(c, static_cast<const float *>(A), static_cast<const float *>(B),
                 static_cast<float *>(C), stage + 6 * i);
    }
    return 0;
} catch (const std::exception &e) {
    s3f::error = e.what();
    return 1;
}
extern "C" const char *s3_info(void *h) {
    return h ? static_cast<s3f::Context *>(h)->info.c_str() : "";
}
extern "C" const char *s3_last_error() {
    return s3f::error.c_str();
}
extern "C" void s3_destroy(void *h) {
    delete static_cast<s3f::Context *>(h);
}
