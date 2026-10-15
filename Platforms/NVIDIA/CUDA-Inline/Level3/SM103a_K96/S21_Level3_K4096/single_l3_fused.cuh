#pragma once
#include "s3_encode_view.cuh"
#include "s3_encode_y.cuh"
#include <chrono>
#include <memory>
namespace s3f {
static thread_local std::string error;

__device__ __forceinline__ unsigned abs32(float z) {
    return __float_as_uint(z) & 0x7fffffffu;
}
__device__ __forceinline__ int ilogb32(unsigned a) {
    const int e = int(a >> 23);
    return e ? e - 127 : (31 - __clz(int(a))) - 149;
}
__global__ void shift_view32(const float *x, int64_t ld, int *exponents, int *invalid, int outer,
                             int k, int bits) { // side 1, mode 0
    __shared__ unsigned values[256];
    const int row = blockIdx.x, t = threadIdx.x;
    unsigned mx = 0;
    const float *r = x + size_t(row) * ld;
    for (int p = t; p < k; p += 256) {
        const unsigned a = abs32(r[p]);
        if (a >= 0x7f800000u) {
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
        exponents[row] = values[0] ? bits - 1 - ilogb32(values[0]) : 0;
    }
}
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

constexpr int kShiftKs = 8; // blocks along p per 32 rows
__global__ void __launch_bounds__(256)
    shift_rows_cm(const float *__restrict__ x, long long ld, unsigned long long *part, int *invalid,
                  int rows, int k, int lower) {
    __shared__ unsigned long long red[8][32];
    const int tx = threadIdx.x, ty = threadIdx.y, row = blockIdx.x * 32 + tx;
    const int span = (k + kShiftKs - 1) / kShiftKs, p0 = blockIdx.y * span;
    unsigned long long mx = 0;
    bool bad = false;
    if (row < rows) {
        int p1 = min(k, p0 + span);
        if (lower) {
            p1 = min(p1, row + 1);
        }
        for (int q = p0 + ty; q < p1; q += 8) {
            const auto a = l8e::abs64(x[size_t(q) * ld + row]);
            if (!l8e::rfinite(a)) {
                bad = true;
            } else {
                mx = mx > a ? mx : a;
            }
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[ty][tx] = mx;
    __syncthreads();
    if (ty == 0 && row < rows) {
        for (int i = 1; i < 8; ++i) {
            mx = red[i][tx] > mx ? red[i][tx] : mx;
        }
        if (mx) {
            atomicMax(part + row, mx);
        }
    }
}
__global__ void __launch_bounds__(256)
    shift_tail_cm(const float *__restrict__ x, long long n, unsigned long long *part, int *invalid,
                  int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) {
        return;
    }
    unsigned long long mx = 0;
    bool bad = false;
    for (int q = row + 1 + lane; q < k; q += 32) { // (row, q), q > row, reflected = x[q + row * n]
        const auto a = l8e::abs64(x[size_t(row) * n + q]);
        if (!l8e::rfinite(a)) {
            bad = true;
        } else {
            mx = mx > a ? mx : a;
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    for (int o = 16; o; o /= 2) {
        const unsigned long long y = __shfl_down_sync(0xffffffffu, mx, o);
        mx = y > mx ? y : mx;
    }
    if (lane == 0 && mx) {
        atomicMax(part + row, mx);
    }
}

__global__ void __launch_bounds__(256)
    shift_rows_cm32(const float *__restrict__ x, long long ld, unsigned *part, int *invalid,
                    int rows, int k, int lower) {
    __shared__ unsigned red[8][32];
    const int tx = threadIdx.x, ty = threadIdx.y, row = blockIdx.x * 32 + tx;
    const int span = (k + kShiftKs - 1) / kShiftKs, p0 = blockIdx.y * span;
    unsigned mx = 0;
    bool bad = false;
    if (row < rows) {
        int p1 = min(k, p0 + span);
        if (lower) {
            p1 = min(p1, row + 1);
        }
        for (int q = p0 + ty; q < p1; q += 8) {
            const unsigned a = abs32(x[size_t(q) * ld + row]);
            if (a >= 0x7f800000u) {
                bad = true;
            } else {
                mx = mx > a ? mx : a;
            }
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    red[ty][tx] = mx;
    __syncthreads();
    if (ty == 0 && row < rows) {
        for (int i = 1; i < 8; ++i) {
            mx = red[i][tx] > mx ? red[i][tx] : mx;
        }
        if (mx) {
            atomicMax(part + row, mx);
        }
    }
}
__global__ void __launch_bounds__(256)
    shift_tail_cm32(const float *__restrict__ x, long long n, unsigned *part, int *invalid,
                    int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) {
        return;
    }
    unsigned mx = 0;
    bool bad = false;
    for (int q = row + 1 + lane; q < k; q += 32) {
        const unsigned a = abs32(x[size_t(row) * n + q]);
        if (a >= 0x7f800000u) {
            bad = true;
        } else {
            mx = mx > a ? mx : a;
        }
    }
    if (bad) {
        atomicExch(invalid, 1);
    }
    for (int o = 16; o; o /= 2) {
        const unsigned y = __shfl_down_sync(0xffffffffu, mx, o);
        mx = y > mx ? y : mx;
    }
    if (lane == 0 && mx) {
        atomicMax(part + row, mx);
    }
}
__global__ void __launch_bounds__(256)
    shift_finish32(const unsigned *part, int *exponents, int rows, int bits) {
    const int row = blockIdx.x * 256 + threadIdx.x;
    if (row < rows) {
        exponents[row] = part[row] ? bits - 1 - ilogb32(part[row]) : 0;
    }
}
__global__ void __launch_bounds__(256)
    shift_finish(const unsigned long long *part, int *exponents, int rows, int bits) {
    const int row = blockIdx.x * 256 + threadIdx.x;
    if (row < rows) {
        exponents[row] = part[row] ? bits - 1 - qint::ilogb_bits(part[row]) : 0;
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

__global__ void __launch_bounds__(256)
    syr2k_add(const float *__restrict__ T, float *__restrict__ C, int n, long long ldc) {
    __shared__ float mirror[32][33];
    const int bi = blockIdx.x, bj = blockIdx.y;
    if (bi < bj) {
        return;
    }
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int r = ty; r < 32;
         r +=
         8) { // mirror[r][tx] = T(j = bj*32 + tx, i = bi*32 + r) = T[(bi*32 + r) * n + bj*32 + tx]
        const int i = bi * 32 + r, j = bj * 32 + tx;
        mirror[r][tx] = (i < n && j < n) ? T[size_t(i) * n + j] : 0.f;
    }
    __syncthreads();
    for (int c = ty; c < 32; c += 8) { // column j = bj*32 + c, rows i = bi*32 + tx
        const int i = bi * 32 + tx, j = bj * 32 + c;
        if (i < n && j < n && i >= j) {
            C[size_t(j) * ldc + i] = __fadd_rn(T[size_t(j) * n + i], mirror[tx][c]);
        }
    }
}
struct Context {
    void *f = nullptr;
    int routine, n, k, param;
    unsigned *maps[2] = {};
    float *t32 = nullptr;
    int enc_form = 1;
    unsigned long long *rowmax = nullptr;
    int counts[2] = {}, gemms = 1;
    std::string info;
    ~Context() {
        for (auto p : maps) {
            if (p) {
                cudaFree(p);
            }
        }
        if (t32) {
            cudaFree(t32);
        }
        if (rowmax) {
            cudaFree(rowmax);
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
            (void)half;
            host[0].push_back(tile);
        }
        gemms = 1;
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

inline void encode_y(l8rt::Plan &p, const float *x, int64_t ld, int side, S3InputView view, int n) {
    l8e::YArgs a{x,
                 ld,
                 side ? p.eb : p.ea,
                 side ? p.pb : p.pa,
                 nullptr,
                 p.enc.dlutx[side],
                 nullptr,
                 side,
                 view.mode,
                 n,
                 0};
    if (side) {
        l8e::launch_s3_y<false, true>(p.enc, a);
    } else {
        l8e::launch_s3_y<false, false>(p.enc, a);
    }
}
inline void encode_y_both(l8rt::Plan &p, const float *x, int n) {
    l8e::YArgs a{x, n, p.ea, p.pa, p.pb, p.enc.dlutx[0], p.enc.dlutx[1], 0, 0, n, p.enc.tiles(1)};
    l8e::launch_s3_y<true, false>(p.enc, a);
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

    auto shift_cm = [&](const float *x, int *e, int mode) {
        ck(cudaMemsetAsync(c.rowmax, 0, size_t(c.n) * sizeof(unsigned long long)));
        shift_rows_cm<<<dim3((c.n + 31) / 32, kShiftKs), dim3(32, 8)>>>(
            x, c.n, c.rowmax, p.bad, c.n, c.k, mode == 2 || mode == 3);
        if (mode == 3) {
            shift_tail_cm<<<(c.n + 7) / 8, 256>>>(x, c.n, c.rowmax, p.bad, c.n, c.k);
        }
        shift_finish<<<(c.n + 255) / 256, 256>>>(c.rowmax, e, c.n, p.bits);
    };
    auto shift_cm32 = [&](const float *x, int *e, int mode) {
        unsigned *const part = reinterpret_cast<unsigned *>(c.rowmax);
        ck(cudaMemsetAsync(part, 0, size_t(c.n) * sizeof(unsigned)));
        shift_rows_cm32<<<dim3((c.n + 31) / 32, kShiftKs), dim3(32, 8)>>>(
            x, c.n, part, p.bad, c.n, c.k, mode == 2 || mode == 3);
        if (mode == 3) {
            shift_tail_cm32<<<(c.n + 7) / 8, 256>>>(x, c.n, part, p.bad, c.n, c.k);
        }
        shift_finish32<<<(c.n + 255) / 256, 256>>>(part, e, c.n, p.bits);
    };
    if (c.enc_form) {
        shift_cm32(A, p.ea, av.mode);
    } else {
        shift_cm(A, p.ea, av.mode);
    }
    if (c.routine == 0) {
        ck(cudaMemcpyAsync(p.eb, p.ea, size_t(c.n) * sizeof(int), cudaMemcpyDeviceToDevice));
    } else if (bv.mode == 1) {
        if (c.enc_form) {
            shift_cm32(right, p.eb, 1);
        } else {
            shift_cm(right, p.eb, 1);
        }
    } else if (c.enc_form) {
        shift_view32<<<c.n, 256>>>(right, p.ldb, p.eb, p.bad, c.n, c.k, p.bits);
    } else {
        shift_view<<<c.n, 256>>>(right, p.ldb, p.eb, p.bad, c.n, c.k, p.bits, 1, bv);
    }
    ck(cudaEventRecord(p.ev[1]));
    if (!c.enc_form) {
        encode(p, A, c.n, 0, av);
        encode(p, right, p.ldb, 1, bv);
    } else if (c.routine == 0) {
        encode_y_both(p, A, c.n);
    } else {
        encode_y(p, A, c.n, 0, av, c.n);
        encode_y(p, right, p.ldb, 1, bv, c.n);
    }
    ck(cudaEventRecord(p.ev[2]));
    for (int j = 0; j < c.gemms; ++j) {
        auto sh = p.sh;
        auto cfg = p.cfg;
        sh.s3_tiles = c.maps[j];
        sh.tiles = c.counts[j];
        sh.whole = sh.tiles;
        sh.nchunk = 1;
        {
            const int rem = sh.tiles % GridClusters,
                      nct = rem > 0 ? std::min(kParts, GridClusters / rem) : 1;
            if (sh.tiles >= 2 * GridClusters && nct >= 2) {
                sh.nchunk = nct;
                sh.whole = sh.tiles - rem;
            }
        }
        const int items = sh.nchunk > 1 ? sh.whole + (sh.tiles - sh.whole) * sh.nchunk : sh.tiles;
        sh.clusters = std::min(GridClusters, items);
        sh.pace_s = -1;
        sh.s3_lower = c.routine == 0;
        sh.s3_triangular_a = c.routine == 1;
        sh.s3_defer_crt = 0;
        cfg.gridDim = dim3(2 * sh.clusters);
        cfg.stream = 0;
        float *const out = c.routine == 2 ? c.t32 : C;
        ck(cudaLaunchKernelEx(&cfg, l8k_gemm_kernel_rt<true>, (const unsigned char *)p.pa,
                              (const unsigned char *)p.pb, p.resid, (const L8kMod *)p.dmods,
                              p.block_tiles, p.km, p.mm, p.crt_state,
                              (const f64_impl::U128 *)nullptr, out, (const int *)p.ea,
                              (const int *)p.eb, p.P, p.tmA, p.tmB, sh));
        if (sh.nchunk > 1) {
            l8k_crt_after<<<(sh.tiles - sh.whole) * 2 * p.crt_chunks, 256>>>(
                p.resid, out, (const int *)p.ea, (const int *)p.eb, p.P, sh, p.crt_chunks);
        }
    }
    if (c.routine == 2) {
        syr2k_add<<<dim3((c.n + 31) / 32, (c.n + 31) / 32), dim3(32, 8)>>>(c.t32, C, c.n, p.sh.ldc);
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
    if (routine == 2) {
        ck(cudaMalloc(&c->t32, size_t(n) * n * sizeof(float)));
    }
    {
        const char *e = getenv("S3_ENC_FORM");
        c->enc_form = e && *e == '0' ? 0 : 1;
        auto &p = c->plan();
        if (c->enc_form && (p.enc.bits > 25 || !l8e::lut_period_ok(l8e::lut_x_host(p.enc, 0)) ||
                            !l8e::lut_period_ok(l8e::lut_x_host(p.enc, 1)))) {
            throw std::runtime_error(
                "s3: the new encoder's conditions (bits <= 25, LUT period p) do not hold; set S3_ENC_FORM=0");
        }
        if (c->enc_form) {
            ck(cudaMemset(p.pb, 0, p.bytesB));
        }
    }
    ck(cudaMalloc(&c->rowmax, size_t(n) * sizeof(unsigned long long)));
    c->info = "{\"routine\":" + std::to_string(routine) + ",\"n\":" + std::to_string(n) +
              ",\"k\":" + std::to_string(k) + ",\"param\":" + std::to_string(param) +
              ",\"strips\":" + std::to_string(c->gemms) +
              ",\"fused\":true,\"encoded_once_per_call\":true,\"enc_form\":" +
              std::to_string(c->enc_form) +
              ",\"temporary_float_matrices\":" + std::to_string(routine == 2 ? 1 : 0) +
              ",\"engine\":" + f_info(c->f) + "}";
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
