#include "v5_kernel.cuh" // first: its build flags
#include "v5fenc.cuh"
#include "v5_backend.h"
#include <cmath>
#include <cstdlib>
namespace v5b {
static unsigned char ue4m3(double v) {
    for (int e = -6; e <= 8; ++e)
        for (int m = 0; m < 8; ++m)
            if ((1.0 + m / 8.0) * std::ldexp(1.0, e) == v)
                return (unsigned char)(((e + 7) << 3) | m);
    return 0;
}
struct Ctx {
    int m = -1, n = -1, k = -1, Mpad = 0, np = 0, Kp = 0, nb = 0, ng = 0, gx = 0, ya = 0, yb = 0, cuts = 0, padded = 0,
        xub = 0, xu = 0, xreq = 0, tiles = 0;
    int8_t *y2t = nullptr;
    v5f::u64 *s8a = nullptr, *s8b = nullptr;
    unsigned *part = nullptr;
    unsigned *kmask = nullptr;
    int *kbase = nullptr, *rtot = nullptr;
    int8_t *xk = nullptr;
    int nw = 0;
    unsigned *kmaskA = nullptr;
    int *alist = nullptr, *acnt = nullptr;
    const int8_t *b_ptr = nullptr;
    long long b_ld = -1;
    bool b_valid = false;
    int *wca = nullptr, *wcb = nullptr, *rpa = nullptr, *rpb = nullptr, *enta = nullptr, *entb = nullptr,
        *segc = nullptr;
    unsigned *stat = nullptr;
    unsigned long long *proof = nullptr, *flg = nullptr;
    long long capa = 0, capb = 0;
    unsigned char *pa = nullptr, *pb = nullptr, *xa = nullptr, *xb = nullptr, *etab = nullptr;
    int32_t *c = nullptr;
    v5k::Metrics *mt = nullptr;
    short *y2s = nullptr;
    CUtensorMap tmA, tmB, tmEA, tmEB;
    v5k::Scales sc{};
    bool tables = false;
};
Ctx *create() { return new Ctx(); }
static void free_bufs(Ctx *c) {
    void *p[] = {c->y2t,  c->s8a, c->s8b, c->part, c->kmask, c->kbase, c->rtot, c->xk,   c->kmaskA, c->alist,
                 c->acnt, c->wca, c->wcb, c->rpa,  c->rpb,   c->enta,  c->entb, c->segc, c->stat,   c->proof,
                 c->flg,  c->pa,  c->pb,  c->xa,   c->xb,    c->etab,  c->c,    c->mt,   c->y2s};
    for (void *q : p)
        if (q)
            cudaFree(q);
    Ctx keep = *c;
    *c = Ctx();
    c->tables = keep.tables;
}
void destroy(Ctx *c) {
    if (!c)
        return;
    free_bufs(c);
    delete c;
}
int32_t *out(Ctx *c) { return c->c; }
int mpad(Ctx *c) { return c->Mpad; }
const unsigned long long *flag(Ctx *c) { return c->flg; }
const unsigned *stat_dev(Ctx *c) { return c->stat; }
#define V5CK(x)                                                                                                        \
    do {                                                                                                               \
        if ((x) != cudaSuccess)                                                                                        \
            return 2;                                                                                                  \
    } while (0)
int set_shape(Ctx *c, int m, int n, int k) {
    if (c->m == m && c->n == n && c->k == k)
        return 0;
    free_bufs(c);
    if (!c->tables) {
        V5CK(v5f::init_tables());
        V5CK(v5f::init_kernels());
        c->tables = true;
    }
    const int band_rows = v5k::TileM * v5k::RowTilesPerBand; // BAND 8: the rows are walked in bands of 8 row tiles
    c->m = m;
    c->n = n;
    c->k = k;
    c->Mpad = (m + band_rows - 1) / band_rows * band_rows;
    c->np = (n + v5k::TileN - 1) / v5k::TileN * v5k::TileN;
    c->Kp = (k + 95) / 96 * 96;
    c->nb = c->Kp / 16;
    c->ng = c->Kp / 96;
    c->gx = (c->ng + v5f::GroupsPerWin - 1) / v5f::GroupsPerWin;
    c->ya = (c->Mpad + v5f::Threads - 1) / v5f::Threads;
    c->yb = (c->np + v5f::Threads - 1) / v5f::Threads;
    c->cuts = 5 * c->Kp / 96;
    c->padded = (c->cuts + v5k::CutsPerReq - 1) / v5k::CutsPerReq * v5k::CutsPerReq;
    c->xub = (c->nb + 15) / 16;
    c->xreq = (2 * c->xub + v5k::CutsPerReq - 1) / v5k::CutsPerReq;
    c->xu = c->xreq * v5k::CutsPerReq;
    const int rowtiles = c->Mpad / v5k::TileM, coltiles = c->np / v5k::TileN;
    c->tiles = rowtiles * coltiles;
    const int tpp = (c->tiles + v5k::Pairs - 1) / v5k::Pairs;
    if (tpp > 64)
        return 1; // s_ereq[64] in the kernel (its static_assert moved here)
    c->capa = std::max<long long>(65536, (long long)c->Mpad * c->Kp / 32);
    c->capb = std::max<long long>(65536, (long long)c->np * c->Kp / 32);
    V5CK(cudaMalloc(&c->y2t, size_t(c->Kp) * c->np));
    V5CK(cudaMalloc(&c->s8a, size_t(c->Mpad) * c->ng * 8));
    V5CK(cudaMalloc(&c->s8b, size_t(c->np) * c->ng * 8));
    V5CK(cudaMalloc(&c->part, size_t(c->gx) * (c->ya + c->yb) * v5f::PartWords * 4));
    V5CK(cudaMalloc(&c->wca, size_t(c->Mpad) * c->gx * 4));
    V5CK(cudaMalloc(&c->wcb, size_t(c->np) * c->gx * 4));
    V5CK(cudaMalloc(&c->rpa, size_t(c->Mpad + 1) * 4));
    V5CK(cudaMalloc(&c->rpb, size_t(c->np + 1) * 4));
    V5CK(cudaMalloc(&c->alist, v5f::GatherMax * 4));
    V5CK(cudaMalloc(&c->acnt, 4));
    c->nw = c->Kp / 32;
    V5CK(cudaMalloc(&c->kmaskA, size_t(c->nw) * 4));
    V5CK(cudaMalloc(&c->kmask, size_t(c->nw) * 4));
    V5CK(cudaMalloc(&c->kbase, size_t(c->nw + 1) * 4));
    V5CK(cudaMalloc(&c->rtot, size_t(c->Mpad + c->np) * 4));
    V5CK(cudaMalloc(&c->xk, size_t(c->Kp) * c->Mpad)); // xk: at most every column
    V5CK(cudaMalloc(&c->enta, size_t(c->capa) * 4));
    V5CK(cudaMalloc(&c->entb, size_t(c->capb) * 4));
    V5CK(cudaMalloc(&c->segc, 4));
    V5CK(cudaMalloc(&c->stat, 64));
    V5CK(cudaMalloc(&c->proof, 16));
    V5CK(cudaMalloc(&c->flg, 16));
    V5CK(cudaMemset(c->flg, 0, 16));
    const size_t a_bytes = size_t(c->Mpad / v5k::CtaRows) * c->padded * v5k::ACut,
                 b_bytes = size_t(c->np / v5k::HalfN) * c->padded * v5k::BCut;
    const size_t ea_bytes = size_t(c->Mpad / v5k::CtaRows) * c->xu * v5k::ACut,
                 eb_bytes = size_t(c->np / v5k::HalfN) * c->xu * v5k::BCut;
    V5CK(cudaMalloc(&c->pa, a_bytes));
    V5CK(cudaMalloc(&c->pb, b_bytes));
    V5CK(cudaMalloc(&c->xa, ea_bytes));
    V5CK(cudaMalloc(&c->xb, eb_bytes));
    V5CK(cudaMemset(c->pa, 0, a_bytes));
    V5CK(cudaMemset(c->pb, 0, b_bytes)); // the padded cuts stay zero
    V5CK(cudaMemset(c->xa, 0, ea_bytes));
    V5CK(cudaMemset(c->xb, 0, eb_bytes));
    V5CK(cudaMalloc(&c->etab, c->tiles));
    V5CK(cudaMemset(c->etab, 0, c->tiles));
    V5CK(cudaMalloc(&c->mt, size_t(c->tiles) * sizeof(v5k::Metrics)));
    V5CK(cudaMalloc(&c->c, size_t(c->Mpad) * n * 4));
    V5CK(cudaMalloc(&c->y2s, 64));
    V5CK(cudaMemset(c->y2s, 0, 64));
    // tensor maps (d1p_1526.cu 1400-1406): 256-byte rows, boxes of one request
    typedef CUresult (*EncodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult qr;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr) != cudaSuccess ||
        !fn || qr != cudaDriverEntryPointSuccess)
        return 3;
    auto make = [&](CUtensorMap *tm, void *b, size_t bytes, unsigned box_rows) {
        const cuuint64_t gdim[2] = {256, cuuint64_t(bytes / 256)};
        const cuuint64_t gs[1] = {256};
        const cuuint32_t box[2] = {256, box_rows};
        const cuuint32_t es[2] = {1, 1};
        return ((EncodeTiled)fn)(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, b, gdim, gs, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 (CUtensorMapL2promotion)L2PROMO, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    };
    if (!make(&c->tmA, c->pa, a_bytes, v5k::ATile / 256) || !make(&c->tmB, c->pb, b_bytes, v5k::BTile / 256) ||
        !make(&c->tmEA, c->xa, ea_bytes, v5k::ATile / 256) || !make(&c->tmEB, c->xb, eb_bytes, v5k::BTile / 256))
        return 2;
    // scales (1136-1150, F5_CONTRACT 1): per face A x B = 4 |lambda_f|, A = |lambda|, B = 4; set z = block j of face (z
    // + j) mod 5
    {
        const double la5[5] = {120, 120, 24, 40, 15}, lb5[5] = {4, 4, 4, 4, 4};
        unsigned char A6[5], B6[5];
        for (int f = 0; f < 5; ++f) {
            A6[f] = ue4m3(la5[f]);
            B6[f] = ue4m3(lb5[f]);
        }
        for (int z = 0; z < 5; ++z) {
            unsigned a[6], b[6];
            for (int j = 0; j < 6; ++j) {
                a[j] = A6[(z + j) % 5];
                b[j] = B6[(z + j) % 5];
            }
            c->sc.a0[z] = a[0] | a[1] << 8 | a[2] << 16 | a[3] << 24;
            c->sc.a1[z] = a[4] | a[5] << 8;
            c->sc.b0[z] = b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24;
            c->sc.b1[z] = b[4] | b[5] << 8;
        }
    }
    // the kernel's run-time dimensions and read-out tables
    const v5k::DimsT d{c->Mpad, n, rowtiles, coltiles, c->tiles, tpp, v5k::RowTilesPerBand * coltiles};
    V5CK(cudaMemcpyToSymbol(v5k::gd, &d, sizeof d));
    v5k::F5C f{};
    f.arow = c->rpa;
    f.aent = c->enta;
    f.y2 = c->y2t;
    f.y2s = c->y2s;
    f.bcol = c->rpb;
    f.bent = c->entb;
    f.xa = nullptr;
    f.arow1 = c->rpa;
    f.np = c->np;
    f.k = c->Kp;
    f.m = c->Mpad;
    f.segc = c->segc;
    V5CK(cudaMemcpyToSymbol(v5k::f5c, &f, sizeof f));
    const v5k::C0A off{nullptr, nullptr, nullptr, 0};
    V5CK(cudaMemcpyToSymbol(v5k::c0a, &off, sizeof off));
    V5CK(cudaFuncSetAttribute(v5k::p2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, v5k::SharedBytes));
    return 0;
}
// the gate failed: no leftover entries for the kernel to read (the exact fallback writes C)
__global__ void sanitize(const unsigned long long *flag, int *rpa, int na, int *rpb, int nb_) {
    if (!flag[0])
        return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <= na || i <= nb_; i += gridDim.x * blockDim.x) {
        if (i <= na)
            rpa[i] = 0;
        if (i <= nb_)
            rpb[i] = 0;
    }
}
int issue(Ctx *c, cudaStream_t s, const int8_t *A, long long lda, const int8_t *B, long long ldb, bool prof,
          cudaEvent_t *ev, int32_t *Cd, int form, bool reuse_b) {
    const bool skip_b = reuse_b && form != 0 && c->b_valid && c->b_ptr == B && c->b_ld == ldb;
    const int K = c->Kp;
    int32_t *const Cw = Cd ? Cd : c->c;
    v5f::EncArgs a{};
    a.raw[0] = A;
    a.raw[1] = B;
    a.ld[0] = lda;
    a.ld[1] = ldb;
    a.rows[0] = c->Mpad;
    a.rows[1] = c->np;
    a.rows_true[0] = c->m;
    a.rows_true[1] = c->n;
    for (int i = 0; i < 2; ++i) {
        const uintptr_t p = uintptr_t(a.raw[i]);
        a.vec[i] = (p % 16 == 0 && a.ld[i] % 16 == 0) ? 2 : (p % 4 == 0 && a.ld[i] % 4 == 0) ? 1 : 0;
    }
    a.kr = c->k;
    a.ng = c->ng;
    a.gx = c->gx;
    a.ya = c->ya;
    a.faces[0] = c->pa;
    a.faces[1] = c->pb;
    a.stride_cuts = c->padded;
    a.ext[0] = c->xa;
    a.ext[1] = c->xb;
    a.xub = c->xub;
    a.xu = c->xu;
    a.sh8[0] = c->s8a;
    a.sh8[1] = c->s8b;
    a.wcnt[0] = c->wca;
    a.wcnt[1] = c->wcb;
    a.part = c->part;
    a.rowtot[0] = c->rtot;
    a.rowtot[1] = c->rtot + c->Mpad;
    a.kmaskA = c->kmaskA;
    if (prof)
        V5CK(cudaEventRecord(ev[0], s));
    V5CK(cudaMemsetAsync(c->rtot, 0, size_t(skip_b ? c->Mpad : c->Mpad + c->np) * 4, s));
    V5CK(cudaMemsetAsync(c->kmaskA, 0, size_t(c->nw) * 4, s));
    if (form == 0) {
        V5CK(cudaMemsetAsync(c->kmask, 0, size_t(c->nw) * 4, s));
        a.kmask = c->kmask;
        a.kbase = c->kbase;
        a.xk = c->xk;
        v5f::EncArgs b = a;
        b.ya = 0;
        b.part_base = c->gx * c->ya;
        v5f::encode_fused<<<dim3(c->gx, c->yb), v5f::Threads, v5f::DynSmem, s>>>(b);
        v5f::kprefix<<<1, 1024, 0, s>>>(c->kmask, c->nw, c->kbase);
        a.part_base = 0;
        v5f::encode_fused<<<dim3(c->gx, c->ya), v5f::Threads, v5f::DynSmem, s>>>(a);
    } else {
        a.xt = form == 2 ? c->xk : nullptr;
        v5f::encode_fused<<<dim3(c->gx, skip_b ? c->ya : c->ya + c->yb), v5f::Threads, v5f::DynSmem, s>>>(a);
    }
    if (reuse_b && form != 0) {
        c->b_valid = true;
        c->b_ptr = B;
        c->b_ld = ldb;
    } else
        c->b_valid = false;
    if (prof)
        V5CK(cudaEventRecord(ev[1], s));
    v5f::post1<<<1, 1024, 0, s>>>(c->rtot, c->Mpad, c->rpa, c->rtot + c->Mpad, c->np, c->rpb, c->part, c->gx * c->ya,
                                  c->gx * c->yb, c->capa, c->capb, c->stat, c->proof, c->flg, c->segc, c->kmaskA, c->nw,
                                  c->alist, c->acnt);
    const v5f::LeftSide la{A, lda, c->Mpad, c->m, c->s8a, c->wca, c->rpa, c->enta, c->capa},
        lb{B, ldb, c->np, c->n, c->s8b, c->wcb, c->rpb, c->entb, c->capb};
    {
        const long long items = (long long)(c->Mpad + c->np) * c->ng;
        const int nl = int(std::min<long long>((items + 255) / 256, 65535));
        v5f::post2<<<nl + 296, 256, 0, s>>>(la, lb, c->ng, c->gx, c->k, nl, c->alist, c->acnt, B, ldb, c->n, c->np,
                                            c->s8b, K, c->y2t);
    }
    if (prof)
        V5CK(cudaEventRecord(ev[2], s));
    {
        cudaLaunchAttribute at[1]{};
        at[0].id = cudaLaunchAttributeClusterDimension;
        at[0].val.clusterDim = {2, 1, 1};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(v5k::Grid);
        cfg.blockDim = dim3(v5k::Threads);
        cfg.dynamicSmemBytes = v5k::SharedBytes;
        cfg.stream = s;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        V5CK(cudaLaunchKernelEx(&cfg, v5k::p2_kernel, c->tmA, c->tmB, (const unsigned char *)c->pa,
                                (const unsigned char *)c->pb, reinterpret_cast<float *>(Cw), c->mt, c->cuts, c->sc,
                                c->tmEA, c->tmEB, (const unsigned char *)c->etab, c->xreq, 0ull));
    }
    if (prof)
        V5CK(cudaEventRecord(ev[3], s));
    if (form == 0)
        v5f::fixb_cols2<<<1184, 256, 0, s>>>(Cw, c->Mpad, c->m, c->n, c->xk, c->Mpad, c->kmask, c->kbase, c->rpb,
                                             c->entb, c->flg);
    else if (form == 2)
        v5f::fixb_xt<<<1184, 256, 0, s>>>(Cw, c->Mpad, c->m, c->n, c->xk, c->Mpad, c->rpb, c->entb, c->flg);
    else
        v5f::fixb_cols<<<1184, 256, 0, s>>>(Cw, c->Mpad, c->m, c->n, A, lda, c->rpb, c->entb, c->flg);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
int last_stats(Ctx *c, long long *o10) {
    unsigned st[16];
    int sg = 0;
    unsigned long long fl = 0;
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(st, c->stat, 64, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&sg, c->segc, 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&fl, c->flg, 8, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 2;
    o10[0] = st[1];
    o10[1] = st[2];
    o10[2] = st[0];
    o10[3] = st[9];
    o10[4] = st[10];
    o10[5] = st[8];
    o10[6] = sg;
    o10[7] = (long long)fl;
    o10[8] = st[4];
    o10[9] = st[12];
    return 0;
}
} // namespace v5b
