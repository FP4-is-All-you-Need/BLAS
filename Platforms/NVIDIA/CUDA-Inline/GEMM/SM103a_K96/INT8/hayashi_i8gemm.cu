#include "v5_backend.h"
#include "f16way.cuh"
__global__ void add_i32(int32_t *__restrict__ C, int ldc, const int32_t *__restrict__ T, int m, int n) {
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < (long long)m * n;
         e += (long long)gridDim.x * blockDim.x) {
        const long long j = e / m, i = e - j * m;
        C[j * ldc + i] += T[e];
    }
}
__global__ void f5_out(int32_t *__restrict__ C, int ldc, const int32_t *__restrict__ T, int ldt, int m, int n,
                       int beta1, const unsigned long long *fl) {
    if (fl[0])
        return;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < (long long)m * n;
         e += (long long)gridDim.x * blockDim.x) {
        const long long j = e / m, i = e - j * m;
        const int32_t v = T[j * ldt + i];
        int32_t *d = C + j * ldc + i;
        *d = beta1 ? *d + v : v;
    }
}
#include "hayashi_i8gemm.h"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "lut1526_le4.h"
#include "lut1526_alt.h"
#include "enc1526.h"
#include "d1_kernel.cuh"
#include "e2_kernel.cuh"
#include "f5_kernel.cuh"
#include "f5enc.cuh"
#include "e1_kernel.cuh"
#include "api_layout.cuh"

#ifndef HAYASHI_I8_OUTLIER_CAP_DIV
#define HAYASHI_I8_OUTLIER_CAP_DIV 32
#endif

namespace {
constexpr int KC = 1024;      // K of one piece
constexpr int UC = 4;         // cuts (16 inputs) per thread in the encoder
constexpr int MAXPIECES = 16; // FP32 piece buffers added in one pass

struct EncSide {
    const int8_t *raw;
    uint4 *out;
    unsigned long long units;
    unsigned cpr;
    int ld;
    unsigned long long *cnt;
    int2 *list;
    unsigned long long cap;
    int *linecnt;
    unsigned long long rpb, bstride, orows;
    const float *inv;
}; // inv: QUANT only, per-row 1/scale   // rows per batch, bytes between batches, rows per batch in the faces
// output cut of input cut x: batch b = row / rpb lands at row b * orows + row % rpb of the faces
__device__ __forceinline__ unsigned long long dunit(unsigned long long x, unsigned cpr, unsigned long long rpb,
                                                    unsigned long long orows) {
    if (rpb == orows)
        return x;
    const unsigned long long r = x / cpr;
    return ((r / rpb) * orows + r % rpb) * cpr + x % cpr;
}

// Table entry of value x (index (uint8)x): the six E2M1 codes, face f in nibble f.  +-103 carry the entry of +-102.
// The warp handles 32*UC consecutive cuts of one side; lane l takes cuts base + j*32 + l.  Cut u = row u / cpr, cut u %
// cpr. 4 BF16 (two words) -> 4 bytes of round(x * s) clamped to -102..102
__device__ __forceinline__ uint32_t q4(uint32_t w0, uint32_t w1, float s) {
    const float f[4] = {__uint_as_float(w0 << 16), __uint_as_float(w0 & 0xFFFF0000u), __uint_as_float(w1 << 16),
                        __uint_as_float(w1 & 0xFFFF0000u)};
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = min(102, max(-102, __float2int_rn(f[i] * s)));
        r |= uint32_t(q & 0xFF) << (8 * i);
    }
    return r;
}
template <int QUANT>
__global__ void __launch_bounds__(256, 4) encode_ab(EncSide a, EncSide b, const uint32_t *__restrict__ lut_g) {
    __shared__ uint32_t lut[512];
    __shared__ uint4 stg[8][96];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    lut[threadIdx.x + 256] = lut_g[threadIdx.x + 256];
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    const unsigned long long per = 32ull * UC, ca = (a.units + per - 1) / per, cb = (b.units + per - 1) / per;
    const unsigned long long nw = (unsigned long long)gridDim.x * 8;
    for (unsigned long long w = (unsigned long long)blockIdx.x * 8 + wib; w < ca + cb; w += nw) {
        const bool isb = w >= ca;
        const int8_t *raw = isb ? b.raw : a.raw;
        uint4 *out = isb ? b.out : a.out;
        const unsigned long long units = isb ? b.units : a.units;
        const unsigned cpr = isb ? b.cpr : a.cpr;
        const int ld = isb ? b.ld : a.ld;
        const unsigned long long rpb = isb ? b.rpb : a.rpb, bstride = isb ? b.bstride : a.bstride,
                                 orows = isb ? b.orows : a.orows;
        const unsigned long long b0 = (isb ? w - ca : w) * per;
        const uint32_t *L = lut + (isb ? 256 : 0);
        const bool dense = (unsigned)ld == cpr * 16u && bstride == rpb * (unsigned long long)ld;
        uint4 x[UC];
#pragma unroll
        for (int j = 0; j < UC; ++j) {
            const unsigned long long u = b0 + j * 32 + lane;
            x[j] = make_uint4(0, 0, 0, 0);
            if (u < units) {
                const unsigned long long r = u / cpr;
                if (QUANT) { // raw = BF16: the cut's 16 values are 32 bytes
                    const unsigned long long e =
                        (r / rpb) * bstride + (r % rpb) * (unsigned long long)ld + (u % cpr) * 16;
                    const uint4 *p = reinterpret_cast<const uint4 *>(reinterpret_cast<const uint16_t *>(raw) + e);
                    const uint4 h0 = __ldcs(p), h1 = __ldcs(p + 1);
                    const float sc = (isb ? b.inv : a.inv)[r];
                    x[j] = make_uint4(q4(h0.x, h0.y, sc), q4(h0.z, h0.w, sc), q4(h1.x, h1.y, sc), q4(h1.z, h1.w, sc));
                } else {
                    const unsigned long long src =
                        dense ? u : ((r / rpb) * bstride + (r % rpb) * (unsigned long long)ld) / 16 + u % cpr;
                    x[j] = __ldcs(reinterpret_cast<const uint4 *>(raw) + src);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < UC; ++j) {
            if (b0 + j * 32 >= units)
                break; // warp-uniform: nothing of this side left
            const unsigned long long u = b0 + j * 32 + lane;
            const uint32_t w4[4] = {x[j].x, x[j].y, x[j].z, x[j].w};
            uint32_t hit = 0;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                hit |= QUANT ? 0u : (__vcmpeq4(w4[q], 0x67676767u) | __vcmpeq4(w4[q], 0x99999999u));
            if (hit &&
                u < units) { // rare: record every +-103 of this cut as (row, k<<1 | negative) and count it for its row
                unsigned long long *cnt = isb ? b.cnt : a.cnt;
                int2 *list = isb ? b.list : a.list;
                const unsigned long long cap = isb ? b.cap : a.cap;
                int *linecnt = isb ? b.linecnt : a.linecnt;
                const unsigned long long row = u / cpr, cut = u % cpr;
                for (int e = 0; e < 16; ++e) {
                    const uint32_t v = (w4[e >> 2] >> (8 * (e & 3))) & 0xFFu;
                    if (v == 0x67u || v == 0x99u) {
                        const unsigned long long idx = atomicAdd(cnt, 1ull);
                        if (idx < cap) {
                            list[idx] = make_int2(int(row), int(((cut * 16 + e) << 1) | (v == 0x99u ? 1u : 0u)));
                            atomicAdd(linecnt + row, 1);
                        }
                    }
                }
            }
            uint32_t ev[16];
#pragma unroll
            for (int e = 0; e < 16; ++e)
                ev[e] = L[(w4[e >> 2] >> (8 * (e & 3))) & 0xFFu];
            uint32_t o0[6], o1[6];
            enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[0]), o0);
            enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[8]), o1);
            stg[wib][lane * 3] = make_uint4(o0[0], o1[0], o0[1], o1[1]);
            stg[wib][lane * 3 + 1] = make_uint4(o0[2], o1[2], o0[3], o1[3]);
            stg[wib][lane * 3 + 2] = make_uint4(o0[4], o1[4], o0[5], o1[5]);
            __syncwarp();
            const unsigned long long f0 = b0 + j * 32, f1 = min(f0 + 31, units - 1);
            if (rpb == orows ||
                (f0 / cpr) / rpb == (f1 / cpr) / rpb) { // warp-uniform: the 32 cuts' 1,536 bytes are contiguous
                uint4 *d = out + dunit(f0, cpr, rpb, orows) * 3;
                const unsigned long long lim = units * 3 - f0 * 3;
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const unsigned e = i * 32 + lane;
                    if (e < lim)
                        __stcs(d + e, stg[wib][e]);
                }
            } else if (u < units) { // the 32 cuts cross a batch boundary with padding rows between: each lane its own
                                    // 48 bytes
                uint4 *d = out + dunit(u, cpr, rpb, orows) * 3;
#pragma unroll
                for (int i = 0; i < 3; ++i)
                    __stcs(d + i, stg[wib][lane * 3 + i]);
            }
            __syncwarp();
        }
    }
}

// Only when a side has +-103: exclusive scan of its per-line counts into ptr[0..lines], the counts reset (they become
// the cursors of the scatter).  Block 0 = A (rows), block 1 = B (columns); 1,024 threads each.
struct Lines {
    int *cnt;
    int *ptr;
    int lines;
    const unsigned long long *total;
};
__global__ void __launch_bounds__(1024) scan_lines(Lines la, Lines lb) {
    const Lines L = blockIdx.x ? lb : la;
    if (*L.total == 0)
        return;
    __shared__ int ws[32];
    __shared__ int carry;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    for (int base = 0; base < L.lines; base += 1024) {
        const int i = base + threadIdx.x, v = i < L.lines ? L.cnt[i] : 0;
        int x = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFFu, x, o);
            if ((threadIdx.x & 31) >= o)
                x += y;
        }
        if ((threadIdx.x & 31) == 31)
            ws[threadIdx.x >> 5] = x;
        __syncthreads();
        if (threadIdx.x < 32) {
            int t = ws[threadIdx.x];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xFFFFFFFFu, t, o);
                if (threadIdx.x >= o)
                    t += y;
            }
            ws[threadIdx.x] = t;
        }
        __syncthreads();
        const int incl = x + ((threadIdx.x >> 5) ? ws[(threadIdx.x >> 5) - 1] : 0) + carry;
        if (i < L.lines) {
            L.ptr[i] = incl - v;
            L.cnt[i] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 1023)
            carry = incl;
        __syncthreads();
    }
    if (threadIdx.x == 0)
        L.ptr[L.lines] = carry;
}
// One launch for everything after the scan (returns at once without +-103):
//   the lists (line, k<<1|neg) -> csr[ptr[line] + cursor] = k<<1|neg (A and B);
//   Bt (k x n, B as given) when A has +-103; At (k x m, +-103 -> +-102 = A') when B has.  32 x 32 tiles, 256 threads.
struct Scat {
    const int2 *list;
    const unsigned long long *total;
    const int *ptr;
    int *cur;
    int *csr;
};
struct Tr {
    const int8_t *src;
    int ld;
    int rows;
    int8_t *dst;
    int map103;
    int bc;
    unsigned long long bstride;
}; // rows per batch
__global__ void __launch_bounds__(256) scatter_transpose(Scat a, Scat b, unsigned long long cap, Tr tb, Tr ta, int k) {
    const unsigned long long ca = *a.total, cb = *b.total;
    if (ca == 0 && cb == 0)
        return;
    const unsigned long long na = min(ca, cap), nb = min(cb, cap);
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < na + nb;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const bool isa = i < na;
        const int2 e = isa ? a.list[i] : b.list[i - na];
        const int *ptr = isa ? a.ptr : b.ptr;
        int *cur = isa ? a.cur : b.cur;
        int *csr = isa ? a.csr : b.csr;
        const int pos = ptr[e.x] + atomicAdd(cur + e.x, 1);
        csr[pos] = e.y;
    }
    __shared__ int8_t t[32][33];
    const int kt = (k + 31) / 32, rtb = (tb.rows + 31) / 32, rta = (ta.rows + 31) / 32;
    const int tbn = ca ? tb.bc * kt * rtb : 0, tan = cb ? ta.bc * kt * rta : 0;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    for (int q = blockIdx.x; q < tbn + tan; q += gridDim.x) {
        const Tr &T = q < tbn ? tb : ta;
        const int qq = q < tbn ? q : q - tbn;
        const int rt = q < tbn ? rtb : rta, bb = qq / (kt * rt), rem = qq % (kt * rt), bk = (rem % kt) * 32,
                  br = (rem / kt) * 32;
        const int8_t *src = T.src + bb * T.bstride;
        int8_t *dst = T.dst + size_t(bb) * k * T.rows;
        for (int r = ty; r < 32; r += 8) {
            const int rr = br + r, kk = bk + tx;
            int8_t v = 0;
            if (rr < T.rows && kk < k) {
                v = src[size_t(rr) * T.ld + kk];
                if (T.map103) {
                    if (v == 103)
                        v = 102;
                    else if (v == -103)
                        v = -102;
                }
            }
            t[r][tx] = v;
        }
        __syncthreads();
        for (int r = ty; r < 32; r += 8) {
            const int kk = bk + r, rr = br + tx;
            if (kk < k && rr < T.rows)
                dst[size_t(kk) * T.rows + rr] = t[tx][r];
        }
        __syncthreads();
    }
}

struct TSide {
    const int8_t *raw;
    unsigned char *out;
    int rows, rows_pad, rt, ld, cuts, padded;
    unsigned long long *cnt;
    int2 *list;
    unsigned long long cap;
    int *linecnt;
    unsigned *rowE;
    __half *lr = nullptr;
    unsigned long long *shc = nullptr;
    int lrstep = 1;
};
struct ZeroSpec {
    int32_t *c;
    int ld, n, ntiles;
    unsigned short tr[d1::Pairs], tc[d1::Pairs];
};
__device__ __forceinline__ void zero_tail_dev(const ZeroSpec &z) {
    if (z.ntiles <= 0)
        return;
    constexpr int TM = d1::TileM, TN = d1::TileN;
    const long long per = (long long)TM * TN, total = per * z.ntiles;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (long long)gridDim.x * blockDim.x) {
        const int t = int(e / per), r = int(e % per), row = r % TM, col = r / TM, gn = int(z.tc[t]) * TN + col;
        if (gn < z.n)
            z.c[size_t(gn) * z.ld + size_t(z.tr[t]) * TM + row] = 0;
    }
}
__global__ void zero_tail_k(const ZeroSpec z) { zero_tail_dev(z); }
__global__ void __launch_bounds__(256)
    encode_tiled(TSide a, TSide b, const uint32_t *__restrict__ lut_g, const ZeroSpec z) {
    zero_tail_dev(z);
    __shared__ uint32_t lut[512];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    lut[threadIdx.x + 256] = lut_g[threadIdx.x + 256];
    __syncthreads();
    const unsigned long long ua = (unsigned long long)a.rows_pad * a.padded,
                             ub = (unsigned long long)b.rows_pad * b.padded;
    for (unsigned long long u0 = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; u0 < ua + ub;
         u0 += (unsigned long long)gridDim.x * blockDim.x) {
        const bool isb = u0 >= ua;
        const TSide &S = isb ? b : a;
        const unsigned long long u = isb ? u0 - ua : u0;
        const unsigned long long per_group = 8ull * S.padded, g = u / per_group, rem = u % per_group;
        const int c = int(rem / 8), r = int(g * 8 + rem % 8);
        uint4 x = make_uint4(0, 0, 0, 0);
        if (r < S.rows && c < S.cuts)
            x = __ldcs(reinterpret_cast<const uint4 *>(S.raw + size_t(r) * S.ld + size_t(c) * 16));
        const uint32_t w4[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int e = 0; e < 16; ++e) { // rare: +-103
            const uint32_t v = (w4[e >> 2] >> (8 * (e & 3))) & 0xFFu;
            if (v == 0x67u || v == 0x99u) {
                const unsigned long long idx = atomicAdd(S.cnt, 1ull);
                if (idx < S.cap) {
                    S.list[idx] = make_int2(r, ((c * 16 + e) << 1) | (v == 0x99u ? 1 : 0));
                    atomicAdd(S.linecnt + r, 1);
                }
            }
        }
        const uint32_t *L = lut + (isb ? 256 : 0);
        uint32_t ev[16];
#pragma unroll
        for (int e = 0; e < 16; ++e)
            ev[e] = L[(w4[e >> 2] >> (8 * (e & 3))) & 0xFFu];
        uint32_t o0[6], o1[6];
        enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[0]), o0);
        enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[8]), o1);
        unsigned char *d = S.out + size_t(r / S.rt) * S.padded * S.rt * 48 + size_t(c) * S.rt * 48 +
                           size_t((r % S.rt) / 8) * 384 + size_t(r % 8) * 16;
        __stcs(reinterpret_cast<uint4 *>(d), make_uint4(o0[0], o1[0], o0[1], o1[1]));
        __stcs(reinterpret_cast<uint4 *>(d + 128), make_uint4(o0[2], o1[2], o0[3], o1[3]));
        __stcs(reinterpret_cast<uint4 *>(d + 256), make_uint4(o0[4], o1[4], o0[5], o1[5]));
    }
}
__global__ void __launch_bounds__(256)
    encode_tiled3(TSide a, TSide b, const uint32_t *__restrict__ lut_g, const ZeroSpec z) {
    zero_tail_dev(z);
    __shared__ uint32_t lut[512];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    lut[threadIdx.x + 256] = lut_g[threadIdx.x + 256];
    __syncthreads();
    const int cga = (a.padded + 7) / 8, cgb = (b.padded + 7) / 8;
    const long long ta = (long long)(a.rows_pad / a.rt) * cga, tb = (long long)(b.rows_pad / b.rt) * cgb;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (long long q = blockIdx.x; q < ta + tb; q += gridDim.x) {
        const bool isb = q >= ta;
        const TSide &S = isb ? b : a;
        const long long qq = isb ? q - ta : q;
        const int cg = isb ? cgb : cga;
        const int band = int(qq / cg), c0 = int(qq % cg) * 8, ntask = (S.rt / 8) * 2;
        const uint32_t *L = lut + (isb ? 256 : 0);
        for (int tau = w; tau < ntask; tau += 8) {
            const int rg = tau >> 1, ch = tau & 1;
            const int r = band * S.rt + rg * 8 + (lane & 7), c = c0 + ch * 4 + (lane >> 3);
            const bool live = c < S.padded;
            unsigned esum = 0;
            if (live) {
                uint4 x = make_uint4(0, 0, 0, 0);
                if (r < S.rows && c < S.cuts)
                    x = __ldcs(reinterpret_cast<const uint4 *>(S.raw + size_t(r) * S.ld + size_t(c) * 16));
                const uint32_t w4[4] = {x.x, x.y, x.z, x.w};
                uint32_t hit = 0;
#pragma unroll
                for (int q2 = 0; q2 < 4; ++q2)
                    hit |= __vcmpeq4(w4[q2], 0x67676767u) | __vcmpeq4(w4[q2], 0x99999999u);
                if (hit) {
#pragma unroll
                    for (int e = 0; e < 16; ++e) { // rare: +-103
                        const uint32_t v = (w4[e >> 2] >> (8 * (e & 3))) & 0xFFu;
                        if (v == 0x67u || v == 0x99u) {
                            const unsigned long long idx = atomicAdd(S.cnt, 1ull);
                            if (idx < S.cap) {
                                S.list[idx] = make_int2(r, ((c * 16 + e) << 1) | (v == 0x99u ? 1 : 0));
                                atomicAdd(S.linecnt + r, 1);
                            }
                        }
                    }
                }
                uint32_t ev[16];
#pragma unroll
                for (int e = 0; e < 16; ++e)
                    ev[e] = L[(w4[e >> 2] >> (8 * (e & 3))) & 0xFFu];
                uint32_t o0[6], o1[6];
                enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[0]), o0);
                enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[8]), o1);
                unsigned char *d = S.out + size_t(band) * S.padded * S.rt * 48 + size_t(c) * S.rt * 48 +
                                   size_t(rg) * 384 + size_t(lane & 7) * 16;
                __stcs(reinterpret_cast<uint4 *>(d), make_uint4(o0[0], o1[0], o0[1], o1[1]));
                __stcs(reinterpret_cast<uint4 *>(d + 128), make_uint4(o0[2], o1[2], o0[3], o1[3]));
                __stcs(reinterpret_cast<uint4 *>(d + 256), make_uint4(o0[4], o1[4], o0[5], o1[5]));
                if (S.rowE) {
#pragma unroll
                    for (int q4 = 0; q4 < 4; ++q4) {
                        const uint32_t p = __byte_perm(__byte_perm(ev[4 * q4], ev[4 * q4 + 1], 0x0073u),
                                                       __byte_perm(ev[4 * q4 + 2], ev[4 * q4 + 3], 0x0073u), 0x5410u);
                        esum = __dp4a(p, 0x01010101u, esum);
                    }
                }
            }
            if (S.rowE) { // the 4 lanes of a row (lane >> 3 = cut) add up; one atomic per row per warp task
                esum += __shfl_xor_sync(0xFFFFFFFFu, esum, 8);
                esum += __shfl_xor_sync(0xFFFFFFFFu, esum, 16);
                if ((lane >> 3) == 0 && esum)
                    atomicAdd(S.rowE + r, esum);
            }
        }
    }
}

__device__ __forceinline__ uint32_t has103(uint32_t w) {
    const uint32_t a = w ^ 0x67676767u, b = w ^ 0x99999999u;
    return ((a - 0x01010101u) & ~a & 0x80808080u) | ((b - 0x01010101u) & ~b & 0x80808080u);
}
template <bool ROWE, int MINB, int ILP, bool STREAM, bool SHIFT = false, bool DET = false>
__global__ void __launch_bounds__(256, MINB)
    encode_t3(TSide a, TSide b, const uint32_t *__restrict__ lut_g, const ZeroSpec z) {
    zero_tail_dev(z);
    __shared__ uint32_t lut[512];
    __shared__ unsigned long long shcnt[4];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    lut[threadIdx.x + 256] = lut_g[threadIdx.x + 256];
    if (SHIFT && threadIdx.x < 4)
        shcnt[threadIdx.x] = 0;
    __syncthreads();
    const int cga = (a.padded + 7) / 8, cgb = (b.padded + 7) / 8;
    const long long ta = (long long)(a.rows_pad / a.rt) * cga, tb = (long long)(b.rows_pad / b.rt) * cgb;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (long long q = blockIdx.x; q < ta + tb; q += gridDim.x) {
        const bool isb = q >= ta;
        const TSide &S = isb ? b : a;
        const long long qq = isb ? q - ta : q;
        const int cg = isb ? cgb : cga;
        const int band = int(qq / cg), c0 = int(qq % cg) * 8, ntask = (S.rt / 8) * 2;
        const uint32_t *L = lut + (isb ? 256 : 0);
        for (int tau0 = w; tau0 < ntask; tau0 += 8 * ILP) {
            uint4 xs[ILP];
            int rs[ILP], cs[ILP];
#pragma unroll
            for (int i = 0; i < ILP; ++i) { // all loads of this round first
                const int tau = tau0 + 8 * i, rg = tau >> 1, ch = tau & 1;
                rs[i] = band * S.rt + rg * 8 + (lane & 7);
                cs[i] = c0 + ch * 4 + (lane >> 3);
                xs[i] = make_uint4(0, 0, 0, 0);
                if (tau < ntask && cs[i] < S.padded && rs[i] < S.rows && cs[i] < S.cuts)
                    xs[i] = __ldcs(reinterpret_cast<const uint4 *>(S.raw + size_t(rs[i]) * S.ld + size_t(cs[i]) * 16));
            }
#pragma unroll
            for (int i = 0; i < ILP; ++i) {
                const int tau = tau0 + 8 * i;
                if (tau >= ntask)
                    break; // the same for the whole warp
                const int rg = tau >> 1, r = rs[i], c = cs[i];
                const bool live = c < S.padded;
                unsigned esum = 0;
                if (live) {
                    uint32_t w4[4] = {xs[i].x, xs[i].y, xs[i].z, xs[i].w};
                    uint32_t hit = 0;
                    bool hit_known = false;
                    if (SHIFT && S.lr && r < S.rows && c < S.cuts) {
                        // the first of 1, -1, 2, -2, 3, -3 with every x - s in [-128, 127] and not +-103 (s = 0 without
                        // +-103 or when none fits: the sparse correction then pays that block).  The thin GEMM's row: A
                        // [sum(x - s) | s], B [t | sum(x)] (FP16, exact: |sum| <= 2,048).
                        uint32_t bad = 0;
#pragma unroll
                        for (int q2 = 0; q2 < 4; ++q2)
                            bad |= DET ? has103(w4[q2])
                                       : (__vcmpeq4(w4[q2], 0x67676767u) | __vcmpeq4(w4[q2], 0x99999999u));
                        int sh = 0, nbad = 0;
                        if (bad) {
#pragma unroll
                            for (int q2 = 0; q2 < 4; ++q2)
                                nbad += __popc(__vcmpeq4(w4[q2], 0x67676767u) | __vcmpeq4(w4[q2], 0x99999999u)) >> 3;
                            const uint32_t mn = __vmins4(__vmins4(w4[0], w4[1]), __vmins4(w4[2], w4[3])),
                                           mx = __vmaxs4(__vmaxs4(w4[0], w4[1]), __vmaxs4(w4[2], w4[3]));
                            int lo = 127, hi = -128;
#pragma unroll
                            for (int q2 = 0; q2 < 4; ++q2) {
                                lo = min(lo, int(int8_t(mn >> (8 * q2))));
                                hi = max(hi, int(int8_t(mx >> (8 * q2))));
                            }
#pragma unroll
                            for (int cc = 1; cc <= 6; ++cc) {
                                const int t = (cc & 1) ? (cc + 1) / 2 : -(cc / 2);
                                if (lo - t < -128 || hi - t > 127)
                                    continue;
                                const uint32_t pp = 0x01010101u * uint32_t(uint8_t(103 + t)),
                                               pn = 0x01010101u * uint32_t(uint8_t(t - 103));
                                uint32_t h2 = 0;
#pragma unroll
                                for (int q2 = 0; q2 < 4; ++q2)
                                    h2 |= __vcmpeq4(w4[q2], pp) | __vcmpeq4(w4[q2], pn);
                                if (!h2) {
                                    sh = t;
                                    break;
                                }
                            }
                            if (sh) {
                                const uint32_t sr = 0x01010101u * uint32_t(uint8_t(sh));
#pragma unroll
                                for (int q2 = 0; q2 < 4; ++q2)
                                    w4[q2] = __vsub4(w4[q2], sr);
                            }
                        }
                        int sum = 0;
#pragma unroll
                        for (int q2 = 0; q2 < 4; ++q2)
                            sum = __dp4a(int(w4[q2]), 0x01010101, sum);
                        __half *lr = S.lr + size_t(r) * (2 * S.cuts);
                        if (S.lrstep == 2)
                            *reinterpret_cast<__half2 *>(lr + 2 * c) =
                                !isb ? __halves2half2(__int2half_rn(sum), __int2half_rn(sh))
                                     : __halves2half2(__int2half_rn(sh), __int2half_rn(sum + 16 * sh));
                        else if (!isb) {
                            lr[c] = __int2half_rn(sum);
                            lr[S.cuts + c] = __int2half_rn(sh);
                        } else {
                            lr[c] = __int2half_rn(sh);
                            lr[S.cuts + c] = __int2half_rn(sum + 16 * sh);
                        }
                        const unsigned am = __activemask();
                        const unsigned nsh = __reduce_add_sync(am, sh != 0 ? 1u : 0u),
                                       nb3 = __reduce_add_sync(am, unsigned(nbad));
                        if (lane == __ffs(am) - 1 && (nsh | nb3)) {
                            atomicAdd(&shcnt[isb ? 1 : 0], (unsigned long long)nsh);
                            atomicAdd(&shcnt[isb ? 3 : 2], (unsigned long long)nb3);
                        }
                        if (DET) {
                            hit = (bad && !sh) ? 1u : 0u;
                            hit_known = true;
                        }
                    }
                    if (!hit_known) {
#pragma unroll
                        for (int q2 = 0; q2 < 4; ++q2)
                            hit |= DET ? has103(w4[q2])
                                       : (__vcmpeq4(w4[q2], 0x67676767u) | __vcmpeq4(w4[q2], 0x99999999u));
                    }
                    if (hit) {
#pragma unroll
                        for (int e = 0; e < 16; ++e) {
                            const uint32_t v = (w4[e >> 2] >> (8 * (e & 3))) & 0xFFu;
                            if (v == 0x67u || v == 0x99u) {
                                const unsigned long long idx = atomicAdd(S.cnt, 1ull);
                                if (idx < S.cap) {
                                    S.list[idx] = make_int2(r, ((c * 16 + e) << 1) | (v == 0x99u ? 1 : 0));
                                    atomicAdd(S.linecnt + r, 1);
                                }
                            }
                        }
                    }
                    uint32_t ev[16];
#pragma unroll
                    for (int e = 0; e < 16; ++e)
                        ev[e] = L[(w4[e >> 2] >> (8 * (e & 3))) & 0xFFu];
                    uint32_t o0[6], o1[6];
                    enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[0]), o0);
                    enc1526::transpose_pack(*reinterpret_cast<const uint32_t (*)[8]>(&ev[8]), o1);
                    uint4 *d =
                        reinterpret_cast<uint4 *>(S.out + size_t(band) * S.padded * S.rt * 48 + size_t(c) * S.rt * 48 +
                                                  size_t(rg) * 384 + size_t(lane & 7) * 16);
                    const uint4 v0 = make_uint4(o0[0], o1[0], o0[1], o1[1]),
                                v1 = make_uint4(o0[2], o1[2], o0[3], o1[3]),
                                v2 = make_uint4(o0[4], o1[4], o0[5], o1[5]);
                    if (STREAM) {
                        __stcs(d, v0);
                        __stcs(d + 8, v1);
                        __stcs(d + 16, v2);
                    } else {
                        d[0] = v0;
                        d[8] = v1;
                        d[16] = v2;
                    }
                    if (ROWE && S.rowE) {
#pragma unroll
                        for (int q4 = 0; q4 < 4; ++q4) {
                            const uint32_t p =
                                __byte_perm(__byte_perm(ev[4 * q4], ev[4 * q4 + 1], 0x0073u),
                                            __byte_perm(ev[4 * q4 + 2], ev[4 * q4 + 3], 0x0073u), 0x5410u);
                            esum = __dp4a(p, 0x01010101u, esum);
                        }
                    }
                }
                if (ROWE && S.rowE) { // the 4 lanes of a row (lane >> 3 = cut) add up; one atomic per row per warp task
                    esum += __shfl_xor_sync(0xFFFFFFFFu, esum, 8);
                    esum += __shfl_xor_sync(0xFFFFFFFFu, esum, 16);
                    if ((lane >> 3) == 0 && esum)
                        atomicAdd(S.rowE + r, esum);
                }
            }
        }
    }
    if (SHIFT) {
        __syncthreads();
        if (threadIdx.x < 4 && shcnt[threadIdx.x]) {
            unsigned long long *g = (threadIdx.x & 1) ? b.shc : a.shc;
            if (g)
                atomicAdd(g + ((threadIdx.x & 2) ? 2 : 0), shcnt[threadIdx.x]);
        }
    }
}
static void launch_encode(int v, int grid, cudaStream_t s, const TSide &ta, const TSide &tb, const uint32_t *lut,
                          const ZeroSpec &z) {
    switch (v) {
    case 0:
        encode_tiled<<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 2:
        encode_t3<false, 1, 1, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 3:
        encode_t3<false, 4, 1, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 4:
        encode_t3<false, 1, 2, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 5:
        encode_t3<false, 4, 2, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 6:
        encode_t3<false, 4, 2, false><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 7:
        encode_t3<false, 6, 1, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 8:
        encode_t3<false, 4, 2, true, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 9:
    case 10:
        encode_t3<false, 4, 2, true, false, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    case 11:
        encode_t3<false, 4, 2, true, true, true><<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    default:
        encode_tiled3<<<grid, 256, 0, s>>>(ta, tb, lut, z);
        break;
    }
}

__global__ void __launch_bounds__(1024)
    e2_gate(const unsigned *__restrict__ ea, int na, const unsigned *__restrict__ eb, int nb, int k, int *use) {
    __shared__ unsigned sa_[32], sb_[32];
    unsigned a = 0, b = 0;
    for (int i = threadIdx.x; i < na; i += 1024)
        a = max(a, ea[i]);
    for (int i = threadIdx.x; i < nb; i += 1024)
        b = max(b, eb[i]);
    for (int o = 16; o; o >>= 1) {
        a = max(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
        b = max(b, __shfl_xor_sync(0xFFFFFFFFu, b, o));
    }
    if ((threadIdx.x & 31) == 0) {
        sa_[threadIdx.x >> 5] = a;
        sb_[threadIdx.x >> 5] = b;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        a = sa_[threadIdx.x];
        b = sb_[threadIdx.x];
        for (int o = 16; o; o >>= 1) {
            a = max(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
            b = max(b, __shfl_xor_sync(0xFFFFFFFFu, b, o));
        }
        if (threadIdx.x == 0) {
            const unsigned long long pa = 65ull * a, pb = 65ull * b;
            use[0] = (k <= 8192 && (k <= 1024 || pa * pb <= (1ull << 48))) ? 1 : 0;
        }
    }
}

// 64 x 64 transpose with 4-byte loads and stores (rows % 4 == 0, ld % 4 == 0 and k % 4 == 0 for the word path; bytes
// otherwise)
struct Tr2 {
    const int8_t *src;
    int ld;
    int rows;
    int8_t *dst;
    int map103;
    const __half *lr = nullptr;
    int lrld = 0, lroff = 0, lrstep = 1;
};
__device__ __forceinline__ uint32_t map4(uint32_t w) { // +-103 -> +-102 in each byte (A')
    const uint32_t e67 = __vcmpeq4(w, 0x67676767u), e99 = __vcmpeq4(w, 0x99999999u);
    return (w & ~(e67 | e99)) | (e67 & 0x66666666u) | (e99 & 0x9A9A9A9Au);
}
__global__ void __launch_bounds__(256)
    transpose64(Tr2 t0, Tr2 t1, int k, const unsigned long long *g0, const unsigned long long *g1) {
    const bool on0 = *g0 > 0, on1 = *g1 > 0; // Bt when A has +-103 (g0 = count A), At when B has (g1 = count B)
    if (!on0 && !on1)
        return;
    __shared__ uint32_t sm[64][17]; // 64 rows x 64 bytes (16 words + 1 pad)
    const int kt = (k + 63) / 64, rt0 = (t0.rows + 63) / 64, rt1 = (t1.rows + 63) / 64;
    const int n0 = on0 ? kt * rt0 : 0, n1 = on1 ? kt * rt1 : 0;
    const int t = threadIdx.x, c4 = t & 15, r16 = t >> 4;
    for (int q = blockIdx.x; q < n0 + n1; q += gridDim.x) {
        const Tr2 &T = q < n0 ? t0 : t1;
        const int qq = q < n0 ? q : q - n0;
        const int kt_ = kt;
        const int bk = (qq % kt_) * 64, br = (qq / kt_) * 64;
#pragma unroll
        for (int p = 0; p < 4; ++p) { // load 64 rows x 64 bytes
            const int r = br + r16 + 16 * p, kk = bk + 4 * c4;
            uint32_t w = 0;
            if (r < T.rows) {
                if (kk + 3 < k)
                    w = *reinterpret_cast<const uint32_t *>(T.src + size_t(r) * T.ld + kk);
                else
                    for (int b = 0; b < 4; ++b)
                        if (kk + b < k)
                            w |= uint32_t(uint8_t(T.src[size_t(r) * T.ld + kk + b])) << (8 * b);
            }
            if (T.lr && r < T.rows && kk < k)
                w = __vsub4(w, 0x01010101u * uint32_t(uint8_t(__half2int_rn(
                                                 T.lr[size_t(r) * T.lrld + T.lroff + T.lrstep * (kk / 16)]))));
            sm[r16 + 16 * p][c4] = T.map103 ? map4(w) : w;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < 4; ++p) { // write 64 k-rows x 64 bytes (rows br..br+63)
            const int kl = r16 + 16 * p, kk = bk + kl, rl = 4 * c4, r = br + rl;
            if (kk < k) {
                const int wi = kl >> 2, sh = 8 * (kl & 3);
                uint32_t o = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    o |= ((sm[rl + b][wi] >> sh) & 0xFFu) << (8 * b);
                int8_t *d = T.dst + size_t(kk) * T.rows + r;
                if (r + 3 < T.rows && (T.rows & 3) == 0)
                    *reinterpret_cast<uint32_t *>(d) = o;
                else
                    for (int b = 0; b < 4; ++b)
                        if (r + b < T.rows)
                            d[b] = int8_t((o >> (8 * b)) & 0xFF);
            }
        }
        __syncthreads();
    }
}
__device__ __forceinline__ void tr4x4(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t &y0, uint32_t &y1,
                                      uint32_t &y2, uint32_t &y3) {
    const uint32_t t0 = __byte_perm(x0, x1, 0x5140), t1 = __byte_perm(x0, x1, 0x7362), t2 = __byte_perm(x2, x3, 0x5140),
                   t3 = __byte_perm(x2, x3, 0x7362);
    y0 = __byte_perm(t0, t2, 0x5410);
    y1 = __byte_perm(t0, t2, 0x7632);
    y2 = __byte_perm(t1, t3, 0x5410);
    y3 = __byte_perm(t1, t3, 0x7632);
}
__global__ void __launch_bounds__(256)
    transpose128(Tr2 t0, Tr2 t1, int k, const unsigned long long *g0, const unsigned long long *g1) {
    const bool on0 = *g0 > 0, on1 = *g1 > 0; // Bt when A has +-103 (g0 = count A), At when B has (g1 = count B)
    if (!on0 && !on1)
        return;
    __shared__ __align__(16) uint32_t sm[128 * 32];
    const int kt = (k + 127) / 128, rt0 = (t0.rows + 127) / 128, rt1 = (t1.rows + 127) / 128;
    const int n0 = on0 ? kt * rt0 : 0, n1 = on1 ? kt * rt1 : 0;
    const int t = threadIdx.x, c8 = t & 7, r32 = t >> 3, rg = t & 7, kg = t >> 3;
    for (int q = blockIdx.x; q < n0 + n1; q += gridDim.x) {
        const Tr2 &T = q < n0 ? t0 : t1;
        const int qq = q < n0 ? q : q - n0;
        const int bk = (qq % kt) * 128, br = (qq / kt) * 128;
        uint4 v[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) { // row r32 + 32p, bytes 16 c8 .. 16 c8 + 15 of the tile
            const int r = br + r32 + 32 * p, kk = bk + 16 * c8;
            v[p] = (r < T.rows && kk < k) ? *reinterpret_cast<const uint4 *>(T.src + size_t(r) * T.ld + kk)
                                          : make_uint4(0, 0, 0, 0);
            if (T.lr && r < T.rows && kk < k) {
                const uint32_t sr =
                    0x01010101u *
                    uint32_t(uint8_t(__half2int_rn(T.lr[size_t(r) * T.lrld + T.lroff + T.lrstep * (kk / 16)])));
                v[p].x = __vsub4(v[p].x, sr);
                v[p].y = __vsub4(v[p].y, sr);
                v[p].z = __vsub4(v[p].z, sr);
                v[p].w = __vsub4(v[p].w, sr);
            }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int rl = r32 + 32 * p;
            uint4 w = v[p];
            if (T.map103) {
                w.x = map4(w.x);
                w.y = map4(w.y);
                w.z = map4(w.z);
                w.w = map4(w.w);
            }
            *reinterpret_cast<uint4 *>(&sm[rl * 32 + ((4 * c8) ^ (4 * ((rl >> 4) & 7)))]) = w;
        }
        __syncthreads();
        // thread: tile rows 16 rg .. 16 rg + 15, tile bytes 4 kg .. 4 kg + 3 (one word column): 4 groups of 4 rows -> 4
        // x 4 transposes
        uint32_t y[4][4]; // y[b][g] = rows 16 rg + 4g .. +3 at byte 4 kg + b
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t x[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int rl = 16 * rg + 4 * g + u;
                x[u] = sm[rl * 32 + (kg ^ (4 * rg))];
            }
            tr4x4(x[0], x[1], x[2], x[3], y[0][g], y[1][g], y[2][g], y[3][g]);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int kk = bk + 4 * kg + b, r = br + 16 * rg;
            if (kk < k && r < T.rows)
                *reinterpret_cast<uint4 *>(T.dst + size_t(kk) * T.rows + r) =
                    make_uint4(y[b][0], y[b][1], y[b][2], y[b][3]);
        }
        __syncthreads();
    }
}
static bool tr128_ok(int trvar, const void *A, int lda, int m, const void *B, int ldb, int n, int k, const void *at,
                     const void *bt) {
    auto al = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    return trvar == 2 && m % 16 == 0 && n % 16 == 0 && lda % 16 == 0 && ldb % 16 == 0 && k % 16 == 0 && al(A) &&
           al(B) && al(at) && al(bt);
}
__global__ void scatter_only(Scat a, Scat b, unsigned long long cap) {
    const unsigned long long ca = *a.total, cb = *b.total;
    if (ca == 0 && cb == 0)
        return;
    const unsigned long long na = min(ca, cap), nb = min(cb, cap);
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < na + nb;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const bool isa = i < na;
        const int2 e = isa ? a.list[i] : b.list[i - na];
        const int *ptr = isa ? a.ptr : b.ptr;
        int *cur = isa ? a.cur : b.cur;
        int *csr = isa ? a.csr : b.csr;
        const int pos = ptr[e.x] + atomicAdd(cur + e.x, 1);
        csr[pos] = e.y;
    }
}

struct PiecePtrs {
    const float *p[MAXPIECES];
};
struct Corr {
    int k;
    const int *rowptrA;
    const int *csrA;
    const unsigned long long *cntA;
    const int8_t *Bt;
    const int *colptrB;
    const int *csrB;
    const unsigned long long *cntB;
    const int8_t *At;
    unsigned long long cap;
};

struct FB {
    const int8_t *A;
    int lda;
    long long sA;
    const int8_t *B;
    int ldb;
    long long sB;
    int32_t *C;
    int ldc;
    long long sC;
    int m, n, k, bc, beta1;
    const unsigned long long *cnt;
    unsigned long long cap;
};
__global__ void __launch_bounds__(256) exact_fallback(FB f) {
    if (f.cnt[0] <= f.cap && f.cnt[1] <= f.cap)
        return;                            // the records fit: the fast path's result stands
    __shared__ int sa[64][17], sb[64][17]; // 64 rows x 16 words (64 bytes of k) + 1 pad word
    const int t = threadIdx.x, ty = t >> 4, tx = t & 15, lr = t >> 2, lw = (t & 3) * 4;
    const int tm = (f.m + 63) / 64, tn = (f.n + 63) / 64;
    for (long long q = blockIdx.x; q < (long long)tm * tn * f.bc; q += gridDim.x) {
        const int bb = int(q / ((long long)tm * tn)), tt = int(q % ((long long)tm * tn));
        const int i0 = (tt % tm) * 64, j0 = (tt / tm) * 64;
        const int8_t *A = f.A + bb * f.sA;
        const int8_t *B = f.B + bb * f.sB;
        int32_t *C = f.C + bb * f.sC;
        int acc[4][4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            acc[u][0] = acc[u][1] = acc[u][2] = acc[u][3] = 0;
        }
        for (int k0 = 0; k0 < f.k; k0 += 64) {
            const int kk = k0 + lw * 4; // this thread's 16 bytes (k % 16 == 0, so a 16-byte piece is all in or all out)
            uint4 va = make_uint4(0, 0, 0, 0), vb = make_uint4(0, 0, 0, 0);
            if (i0 + lr < f.m && kk < f.k)
                va = *reinterpret_cast<const uint4 *>(A + size_t(i0 + lr) * f.lda + kk);
            if (j0 + lr < f.n && kk < f.k)
                vb = *reinterpret_cast<const uint4 *>(B + size_t(j0 + lr) * f.ldb + kk);
            sa[lr][lw] = int(va.x);
            sa[lr][lw + 1] = int(va.y);
            sa[lr][lw + 2] = int(va.z);
            sa[lr][lw + 3] = int(va.w);
            sb[lr][lw] = int(vb.x);
            sb[lr][lw + 1] = int(vb.y);
            sb[lr][lw + 2] = int(vb.z);
            sb[lr][lw + 3] = int(vb.w);
            __syncthreads();
#pragma unroll
            for (int w = 0; w < 16; ++w) {
                int a[4], b[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    a[u] = sa[ty * 4 + u][w];
                    b[u] = sb[tx * 4 + u][w];
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
#pragma unroll
                    for (int v = 0; v < 4; ++v)
                        acc[u][v] = __dp4a(a[u], b[v], acc[u][v]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = i0 + ty * 4 + u;
            if (i >= f.m)
                continue;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int j = j0 + tx * 4 + v;
                if (j >= f.n)
                    continue;
                int32_t *d = C + size_t(j) * f.ldc + i;
                *d = f.beta1 ? *d + acc[u][v] : acc[u][v];
            }
        }
    }
}

// C (+)= sum of the FP32 pieces (exact integers, |.| <= 2^24) (+ the +-103 correction when CORR), tile 128 x 128 per
// step. mode: 0 = always (no correction), 1 = only when there is no +-103, 2 (CORR) = only when there is.
//   A side (row i, k, s): C[i, j] += s * B[j, k] (Bt row k)            -> E_A * B
//   B side (col j, k, s): C[i, j] += s * A'[i, k] (At row k, A')       -> A' * E_B
// Lane l owns rows 4l..4l+3 of the tile (float4 of the pieces); warp w owns columns w, w+8, ... (and rows w, w+8, ...
// for the A side).  Shared Memory (CORR): sc[ri(r)][ci(c)], ri(r) = (r%4)*32 + r/4, ci(c) = (c%4)*32 + c/4, row stride
// 129, so the A side (lanes over 4 columns each), the B side and the add (lanes over 4 rows each) are all free of bank
// conflicts.
#ifndef DENSE // a warp with at least DENSE records per line on average walks its lines one at a time (RTX 4070 Ti: 8
              // best of 4 / 8 / 16)
#define DENSE 8
#endif
constexpr int T2 = 128, SCS = 129, CORR_SMEM = T2 * SCS * 4;
#ifndef INLU
#define INLU 4
#endif
#ifndef INLB
#define INLB 2
#endif
template <bool CORR, bool INTIN = false, int CV = 1, bool INL = false>
__global__ void __launch_bounds__(256, CORR  ? 3
                                       : INL ? INLB
                                             : 4)
    add_tile(PiecePtrs pp, int np, int m, int n, int ldp, int32_t *__restrict__ C, int ldc, int addc, int mode, Corr cr,
             int bc, unsigned long long pstride, unsigned long long cstride, const float *__restrict__ thin = nullptr,
             unsigned long long scap = 0) {
    extern __shared__ int sc[];
    __shared__ const float *sp[MAXPIECES];
    __shared__ int cps[INL ? 8 : 1][INL ? 32 : 1];
    if (*cr.cntA > cr.cap || *cr.cntB > cr.cap)
        return;
    const bool has = *cr.cntA > 0 || *cr.cntB > 0;
    const bool few = scap > 0 && has && *cr.cntA + *cr.cntB <= scap;
    if (INL ? !few : ((mode == 1 && has) || (mode == 2 && (!has || few))))
        return;
    const bool hA = CORR && *cr.cntA > 0, hB = CORR && *cr.cntB > 0;
    const bool iA = INL && *cr.cntA > 0, iB = INL && *cr.cntB > 0;
#pragma unroll
    for (int j = 0; j < MAXPIECES; ++j)
        if (threadIdx.x == j)
            sp[j] = pp.p[j];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, tm = (m + T2 - 1) / T2, tn = (n + T2 - 1) / T2;
    const bool vecC = (ldc % 4 == 0) && ((reinterpret_cast<uintptr_t>(C) & 15) == 0);
    for (long long t = blockIdx.x; t < (long long)tm * tn * bc; t += gridDim.x) {
        const int bb = int(t / ((long long)tm * tn)), tt = int(t % ((long long)tm * tn));
        const int i0 = (tt % tm) * T2, j0 = (tt / tm) * T2, wi = min(T2, m - i0), wj = min(T2, n - j0);
        const int ra = bb * m, cbb = bb * n; // first CSR line of this batch (A rows, B columns)
        if (CORR) {
            for (int e = threadIdx.x; e < T2 * SCS; e += 256)
                sc[e] = 0;
            __syncthreads();
            if (hA && 16 * w < wi) {
                const int l0 = 16 * w, nl = min(16, wi - l0);
                const int p = lane <= nl ? cr.rowptrA[ra + i0 + l0 + lane] : 0x7FFFFFFF;
                const int q0 = __shfl_sync(0xFFFFFFFFu, p, 0), q1 = __shfl_sync(0xFFFFFFFFu, p, nl);
                const bool v4 = (n & 3) == 0 && lane * 4 + 3 < wj;
                const int8_t *base = cr.Bt + size_t(bb) * cr.k * n + j0 + lane * 4;
                if (CV == 2) {
                    int cur = -1, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    for (int qc = q0; qc < q1; qc += 32) {
                        const int my_e = qc + lane < q1 ? cr.csrA[qc + lane] : -1;
                        int my_l = 0;
                        for (int t2 = 1; t2 < nl; ++t2)
                            my_l += __shfl_sync(0xFFFFFFFFu, p, t2) <= qc + lane;
#pragma unroll
                        for (int g = 0; g < 32; g += 16) {
                            if (qc + g >= q1)
                                break;      // the same for the whole warp
                            uint32_t v[16]; // the entries are fetched again by shuffle below (keeping 16 of them live
                                            // spilled)
#pragma unroll
                            for (int u = 0; u < 16; ++u) {
                                const int eu = __shfl_sync(0xFFFFFFFFu, my_e, g + u);
                                v[u] = 0;
                                if (eu >= 0) {
                                    const int8_t *xr = base + size_t(eu >> 1) * n;
                                    if (v4)
                                        v[u] = *reinterpret_cast<const uint32_t *>(xr);
                                    else
                                        for (int b = 0; b < 4; ++b)
                                            if (lane * 4 + b < wj)
                                                v[u] |= uint32_t(uint8_t(xr[b])) << (8 * b);
                                }
                            }
#pragma unroll
                            for (int u = 0; u < 16; ++u) {
                                const int r = l0 + __shfl_sync(0xFFFFFFFFu, my_l, g + u);
                                const int eu = __shfl_sync(0xFFFFFFFFu, my_e, g + u);
                                if (eu < 0)
                                    continue; // the same for the whole warp
                                if (r != cur) {
                                    if (cur >= 0) {
                                        const int rb = ((cur & 3) * 32 + (cur >> 2)) * SCS;
                                        sc[rb + lane] += a0;
                                        sc[rb + 32 + lane] += a1;
                                        sc[rb + 64 + lane] += a2;
                                        sc[rb + 96 + lane] += a3;
                                    }
                                    cur = r;
                                    a0 = a1 = a2 = a3 = 0;
                                }
                                const int s = (eu & 1) ? -1 : 1, x = int(v[u]); // signed bytes by shifts
                                a0 += s * ((x << 24) >> 24);
                                a1 += s * ((x << 16) >> 24);
                                a2 += s * ((x << 8) >> 24);
                                a3 += s * (x >> 24);
                            }
                        }
                    }
                    if (cur >= 0) {
                        const int rb = ((cur & 3) * 32 + (cur >> 2)) * SCS;
                        sc[rb + lane] += a0;
                        sc[rb + 32 + lane] += a1;
                        sc[rb + 64 + lane] += a2;
                        sc[rb + 96 + lane] += a3;
                    }
                } else if (q1 - q0 >= DENSE * nl) {
                    for (int li = 0; li < nl; ++li) {
                        const int r = l0 + li, qa = __shfl_sync(0xFFFFFFFFu, p, li),
                                  qb = __shfl_sync(0xFFFFFFFFu, p, li + 1), rb = ((r & 3) * 32 + (r >> 2)) * SCS;
                        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        for (int q = qa; q < qb; q += 8) {
                            int e[8];
                            uint32_t v[8];
#pragma unroll
                            for (int u = 0; u < 8; ++u)
                                e[u] = q + u < qb ? cr.csrA[q + u] : -1;
#pragma unroll
                            for (int u = 0; u < 8; ++u) {
                                v[u] = 0;
                                if (e[u] >= 0) {
                                    const int8_t *bt = base + size_t(e[u] >> 1) * n;
                                    if (v4)
                                        v[u] = *reinterpret_cast<const uint32_t *>(bt);
                                    else
                                        for (int b = 0; b < 4; ++b)
                                            if (lane * 4 + b < wj)
                                                v[u] |= uint32_t(uint8_t(bt[b])) << (8 * b);
                                }
                            }
#pragma unroll
                            for (int u = 0; u < 8; ++u) {
                                const int s = (e[u] & 1) ? -1 : 1, x = int(v[u]);
                                a0 += s * ((x << 24) >> 24);
                                a1 += s * ((x << 16) >> 24);
                                a2 += s * ((x << 8) >> 24);
                                a3 += s * (x >> 24);
                            }
                        }
                        if (qb > qa) {
                            sc[rb + lane] += a0;
                            sc[rb + 32 + lane] += a1;
                            sc[rb + 64 + lane] += a2;
                            sc[rb + 96 + lane] += a3;
                        }
                    }
                } else {
                    int cur = -1, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    for (int q = q0; q < q1; q += 8) {
                        int e[8];
                        uint32_t v[8];
#pragma unroll
                        for (int u = 0; u < 8; ++u)
                            e[u] = q + u < q1 ? cr.csrA[q + u] : -1;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            v[u] = 0;
                            if (e[u] >= 0) {
                                const int8_t *bt = base + size_t(e[u] >> 1) * n;
                                if (v4)
                                    v[u] = *reinterpret_cast<const uint32_t *>(bt);
                                else
                                    for (int b = 0; b < 4; ++b)
                                        if (lane * 4 + b < wj)
                                            v[u] |= uint32_t(uint8_t(bt[b])) << (8 * b);
                            }
                        }
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            const int r = l0 + __popc(__ballot_sync(0xFFFFFFFFu, lane >= 1 && lane < nl && p <= q + u));
                            if (e[u] < 0)
                                continue; // the same for the whole warp
                            if (r != cur) {
                                if (cur >= 0) {
                                    const int rb = ((cur & 3) * 32 + (cur >> 2)) * SCS;
                                    sc[rb + lane] += a0;
                                    sc[rb + 32 + lane] += a1;
                                    sc[rb + 64 + lane] += a2;
                                    sc[rb + 96 + lane] += a3;
                                }
                                cur = r;
                                a0 = a1 = a2 = a3 = 0;
                            }
                            const int s = (e[u] & 1) ? -1 : 1, x = int(v[u]); // signed bytes by shifts
                            a0 += s * ((x << 24) >> 24);
                            a1 += s * ((x << 16) >> 24);
                            a2 += s * ((x << 8) >> 24);
                            a3 += s * (x >> 24);
                        }
                    }
                    if (cur >= 0) {
                        const int rb = ((cur & 3) * 32 + (cur >> 2)) * SCS;
                        sc[rb + lane] += a0;
                        sc[rb + 32 + lane] += a1;
                        sc[rb + 64 + lane] += a2;
                        sc[rb + 96 + lane] += a3;
                    }
                }
            }
            __syncthreads();
            if (hB && 16 * w < wj) {
                const int l0 = 16 * w, nl = min(16, wj - l0);
                const int p = lane <= nl ? cr.colptrB[cbb + j0 + l0 + lane] : 0x7FFFFFFF;
                const int q0 = __shfl_sync(0xFFFFFFFFu, p, 0), q1 = __shfl_sync(0xFFFFFFFFu, p, nl);
                const bool v4 = (m & 3) == 0 && lane * 4 + 3 < wi;
                const int8_t *base = cr.At + size_t(bb) * cr.k * m + i0 + lane * 4;
                if (CV == 2) {
                    int cur = -1, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    for (int qc = q0; qc < q1; qc += 32) {
                        const int my_e = qc + lane < q1 ? cr.csrB[qc + lane] : -1;
                        int my_l = 0;
                        for (int t2 = 1; t2 < nl; ++t2)
                            my_l += __shfl_sync(0xFFFFFFFFu, p, t2) <= qc + lane;
#pragma unroll
                        for (int g = 0; g < 32; g += 16) {
                            if (qc + g >= q1)
                                break;      // the same for the whole warp
                            uint32_t v[16]; // the entries are fetched again by shuffle below (keeping 16 of them live
                                            // spilled)
#pragma unroll
                            for (int u = 0; u < 16; ++u) {
                                const int eu = __shfl_sync(0xFFFFFFFFu, my_e, g + u);
                                v[u] = 0;
                                if (eu >= 0) {
                                    const int8_t *xr = base + size_t(eu >> 1) * m;
                                    if (v4)
                                        v[u] = *reinterpret_cast<const uint32_t *>(xr);
                                    else
                                        for (int b = 0; b < 4; ++b)
                                            if (lane * 4 + b < wi)
                                                v[u] |= uint32_t(uint8_t(xr[b])) << (8 * b);
                                }
                            }
#pragma unroll
                            for (int u = 0; u < 16; ++u) {
                                const int c = l0 + __shfl_sync(0xFFFFFFFFu, my_l, g + u);
                                const int eu = __shfl_sync(0xFFFFFFFFu, my_e, g + u);
                                if (eu < 0)
                                    continue; // the same for the whole warp
                                if (c != cur) {
                                    if (cur >= 0) {
                                        const int cb = (cur & 3) * 32 + (cur >> 2);
                                        sc[lane * SCS + cb] += a0;
                                        sc[(32 + lane) * SCS + cb] += a1;
                                        sc[(64 + lane) * SCS + cb] += a2;
                                        sc[(96 + lane) * SCS + cb] += a3;
                                    }
                                    cur = c;
                                    a0 = a1 = a2 = a3 = 0;
                                }
                                const int s = (eu & 1) ? -1 : 1, x = int(v[u]); // signed bytes by shifts
                                a0 += s * ((x << 24) >> 24);
                                a1 += s * ((x << 16) >> 24);
                                a2 += s * ((x << 8) >> 24);
                                a3 += s * (x >> 24);
                            }
                        }
                    }
                    if (cur >= 0) {
                        const int cb = (cur & 3) * 32 + (cur >> 2);
                        sc[lane * SCS + cb] += a0;
                        sc[(32 + lane) * SCS + cb] += a1;
                        sc[(64 + lane) * SCS + cb] += a2;
                        sc[(96 + lane) * SCS + cb] += a3;
                    }
                } else if (q1 - q0 >= DENSE * nl) {
                    for (int li = 0; li < nl; ++li) {
                        const int c = l0 + li, qa = __shfl_sync(0xFFFFFFFFu, p, li),
                                  qb = __shfl_sync(0xFFFFFFFFu, p, li + 1), cb = (c & 3) * 32 + (c >> 2);
                        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                        for (int q = qa; q < qb; q += 8) {
                            int e[8];
                            uint32_t v[8];
#pragma unroll
                            for (int u = 0; u < 8; ++u)
                                e[u] = q + u < qb ? cr.csrB[q + u] : -1;
#pragma unroll
                            for (int u = 0; u < 8; ++u) {
                                v[u] = 0;
                                if (e[u] >= 0) {
                                    const int8_t *at = base + size_t(e[u] >> 1) * m;
                                    if (v4)
                                        v[u] = *reinterpret_cast<const uint32_t *>(at);
                                    else
                                        for (int b = 0; b < 4; ++b)
                                            if (lane * 4 + b < wi)
                                                v[u] |= uint32_t(uint8_t(at[b])) << (8 * b);
                                }
                            }
#pragma unroll
                            for (int u = 0; u < 8; ++u) {
                                const int s = (e[u] & 1) ? -1 : 1, x = int(v[u]);
                                a0 += s * ((x << 24) >> 24);
                                a1 += s * ((x << 16) >> 24);
                                a2 += s * ((x << 8) >> 24);
                                a3 += s * (x >> 24);
                            }
                        }
                        if (qb > qa) {
                            sc[lane * SCS + cb] += a0;
                            sc[(32 + lane) * SCS + cb] += a1;
                            sc[(64 + lane) * SCS + cb] += a2;
                            sc[(96 + lane) * SCS + cb] += a3;
                        }
                    }
                } else {
                    int cur = -1, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    for (int q = q0; q < q1; q += 8) {
                        int e[8];
                        uint32_t v[8];
#pragma unroll
                        for (int u = 0; u < 8; ++u)
                            e[u] = q + u < q1 ? cr.csrB[q + u] : -1;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            v[u] = 0;
                            if (e[u] >= 0) {
                                const int8_t *at = base + size_t(e[u] >> 1) * m;
                                if (v4)
                                    v[u] = *reinterpret_cast<const uint32_t *>(at);
                                else
                                    for (int b = 0; b < 4; ++b)
                                        if (lane * 4 + b < wi)
                                            v[u] |= uint32_t(uint8_t(at[b])) << (8 * b);
                            }
                        }
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            const int c = l0 + __popc(__ballot_sync(0xFFFFFFFFu, lane >= 1 && lane < nl && p <= q + u));
                            if (e[u] < 0)
                                continue; // the same for the whole warp
                            if (c != cur) {
                                if (cur >= 0) {
                                    const int cb = (cur & 3) * 32 + (cur >> 2);
                                    sc[lane * SCS + cb] += a0;
                                    sc[(32 + lane) * SCS + cb] += a1;
                                    sc[(64 + lane) * SCS + cb] += a2;
                                    sc[(96 + lane) * SCS + cb] += a3;
                                }
                                cur = c;
                                a0 = a1 = a2 = a3 = 0;
                            }
                            const int s = (e[u] & 1) ? -1 : 1, x = int(v[u]); // signed bytes by shifts
                            a0 += s * ((x << 24) >> 24);
                            a1 += s * ((x << 16) >> 24);
                            a2 += s * ((x << 8) >> 24);
                            a3 += s * (x >> 24);
                        }
                    }
                    if (cur >= 0) {
                        const int cb = (cur & 3) * 32 + (cur >> 2);
                        sc[lane * SCS + cb] += a0;
                        sc[(32 + lane) * SCS + cb] += a1;
                        sc[(64 + lane) * SCS + cb] += a2;
                        sc[(96 + lane) * SCS + cb] += a3;
                    }
                }
            }
            __syncthreads();
        }
        const int r0 = lane * 4;
        if (INL) {
            // B: warp w's columns in this tile are w + 8 t (t < 16); their line starts go to Shared Memory at the tile
            // start (one lane each)
            if (iB) {
                if (lane < 16 && w + 8 * lane < wj) {
                    const int *cp = cr.colptrB + cbb + j0 + w + 8 * lane;
                    cps[w][2 * lane] = cp[0];
                    cps[w][2 * lane + 1] = cp[1];
                }
                __syncwarp();
            }
            if (r0 < wi) {
                int pa0 = 0, pa4 = 0, na = 0, rc[4] = {0, 0, 0, 0}, ro[4] = {0, 0, 0, 0};
                const int8_t *btb = cr.Bt + size_t(bb) * cr.k * n + j0;
                const int8_t *atb = cr.At + size_t(bb) * cr.k * m + i0 + r0;
                const int *rp = cr.rowptrA + ra + i0 + r0;
                const int nr = min(4, wi - r0);
                if (iA) {
                    pa0 = rp[0];
                    pa4 = rp[nr];
                    na = min(4, pa4 - pa0);
                    const int pa1 = rp[min(1, nr)], pa2 = rp[min(2, nr)], pa3 = rp[min(3, nr)];
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        if (t < na) {
                            const int q = pa0 + t, e = cr.csrA[q];
                            rc[t] = ((q >= pa1) + (q >= pa2) + (q >= pa3)) | ((e & 1) << 2);
                            ro[t] = (e >> 1) * n;
                        }
                }
                const bool v4 = vecC && r0 + 3 < wi, atv = (m & 3) == 0 && r0 + 3 < wi;
                for (int cg = w; cg < wj;
                     cg += 8 * INLU) { // INLU columns per step: the records' values issued with the pieces
                    int4 acc[INLU];
                    int eb[INLU], xa[4];
#pragma unroll
                    for (int u = 0; u < INLU; ++u) {
                        acc[u] = make_int4(0, 0, 0, 0);
                        eb[u] = -1;
                    }
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        xa[t] = 0;
                    bool more = false; // this step has B records past a column's first
#pragma unroll
                    for (int u = 0; u < INLU; ++u) {
                        const int c = cg + 8 * u;
                        if (c >= wj)
                            continue;
                        if (iB) {
                            const int t = (c - w) >> 3, q0 = cps[w][2 * t], q1 = cps[w][2 * t + 1];
                            if (q1 > q0)
                                eb[u] = cr.csrB[q0];
                            more |= q1 > q0 + 1;
                        }
#pragma unroll
                        for (int t = 0; t < 4; ++t)
                            if (t < na)
                                xa[t] |= int(uint8_t(btb[ro[t] + c])) << (8 * u);
                    }
                    for (int p = 0; p <= np; ++p) { // p = np: the thin product
                        if (p == np && !thin)
                            break;
                        float4 f[INLU];
#pragma unroll
                        for (int u = 0; u < INLU; ++u) {
                            const int c = cg + 8 * u;
                            if (c < wj)
                                f[u] = __ldcs(reinterpret_cast<const float4 *>((p < np ? sp[p] : thin) + bb * pstride +
                                                                               size_t(j0 + c) * ldp + i0 + r0));
                        }
#pragma unroll
                        for (int u = 0; u < INLU; ++u)
                            if (cg + 8 * u < wj) {
                                if (INTIN && p < np) {
                                    acc[u].x += __float_as_int(f[u].x);
                                    acc[u].y += __float_as_int(f[u].y);
                                    acc[u].z += __float_as_int(f[u].z);
                                    acc[u].w += __float_as_int(f[u].w);
                                } else {
                                    acc[u].x += __float2int_rn(f[u].x);
                                    acc[u].y += __float2int_rn(f[u].y);
                                    acc[u].z += __float2int_rn(f[u].z);
                                    acc[u].w += __float2int_rn(f[u].w);
                                }
                            }
                    }
#pragma unroll
                    for (int u = 0; u < INLU; ++u) {
                        const int c = cg + 8 * u;
                        if (c >= wj)
                            continue;
                        int4 v = acc[u];
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const int x0 = (xa[t] << (24 - 8 * u)) >> 24, x = (rc[t] & 4) ? -x0 : x0,
                                      sb = rc[t] & 3; // A: + sign * Bt[q][j0 + c] (0 past na)
                            v.x += sb == 0 ? x : 0;
                            v.y += sb == 1 ? x : 0;
                            v.z += sb == 2 ? x : 0;
                            v.w += sb == 3 ? x : 0;
                        }
                        if (eb[u] >= 0) {
                            const int sg = (eb[u] & 1) ? -1 : 1;
                            const int8_t *ar = atb + size_t(eb[u] >> 1) * m;
                            if (atv) {
                                const int x = *reinterpret_cast<const int *>(ar);
                                v.x += sg * ((x << 24) >> 24);
                                v.y += sg * ((x << 16) >> 24);
                                v.z += sg * ((x << 8) >> 24);
                                v.w += sg * (x >> 24);
                            } else {
                                v.x += sg * int(ar[0]);
                                if (r0 + 1 < wi)
                                    v.y += sg * int(ar[1]);
                                if (r0 + 2 < wi)
                                    v.z += sg * int(ar[2]);
                                if (r0 + 3 < wi)
                                    v.w += sg * int(ar[3]);
                            }
                        }
                        int32_t *dst = C + bb * cstride + size_t(j0 + c) * ldc + i0 + r0;
                        if (v4) {
                            if (addc) {
                                const int4 o = *reinterpret_cast<const int4 *>(dst);
                                v.x += o.x;
                                v.y += o.y;
                                v.z += o.z;
                                v.w += o.w;
                            }
                            __stcs(reinterpret_cast<int4 *>(dst), v);
                        } else {
                            const int vv[4] = {v.x, v.y, v.z, v.w};
                            for (int u2 = 0; u2 < 4; ++u2)
                                if (r0 + u2 < wi)
                                    dst[u2] = addc ? dst[u2] + vv[u2] : vv[u2];
                        }
                    }
                    if (pa4 - pa0 > 4 || more) { // the rest, after the stores (same thread, same addresses: in order)
#pragma unroll 1
                        for (int u = 0; u < INLU; ++u) {
                            const int c = cg + 8 * u;
                            if (c >= wj)
                                break;
                            int32_t *dst = C + bb * cstride + size_t(j0 + c) * ldc + i0 + r0;
                            for (int q = pa0 + 4; q < pa4; ++q) {
                                const int e = cr.csrA[q], x0 = int(btb[size_t(e >> 1) * n + c]),
                                          sb = (q >= rp[min(1, nr)]) + (q >= rp[min(2, nr)]) + (q >= rp[min(3, nr)]);
                                dst[sb] += (e & 1) ? -x0 : x0;
                            }
                            if (iB) {
                                const int t = (c - w) >> 3;
                                for (int q = cps[w][2 * t] + 1; q < cps[w][2 * t + 1]; ++q) {
                                    const int e = cr.csrB[q], sg = (e & 1) ? -1 : 1;
                                    const int8_t *ar = atb + size_t(e >> 1) * m;
                                    for (int u2 = 0; u2 < 4; ++u2)
                                        if (r0 + u2 < wi)
                                            dst[u2] += sg * int(ar[u2]);
                                }
                            }
                        }
                    }
                }
            }
            if (iB)
                __syncwarp(); // the next tile's line starts are written after every lane has read these
        } else if (r0 < wi) {
            for (int cg = w; cg < wj; cg += 32) { // 4 columns per step, loads first
                int4 acc[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    acc[u] = make_int4(0, 0, 0, 0);
                for (int p = 0; p < np; ++p) {
                    float4 f[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int c = cg + 8 * u;
                        if (c < wj)
                            f[u] = __ldcs(reinterpret_cast<const float4 *>(sp[p] + bb * pstride + size_t(j0 + c) * ldp +
                                                                           i0 + r0));
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (cg + 8 * u < wj) {
                            if (INTIN) {
                                acc[u].x += __float_as_int(f[u].x);
                                acc[u].y += __float_as_int(f[u].y);
                                acc[u].z += __float_as_int(f[u].z);
                                acc[u].w += __float_as_int(f[u].w);
                            } else {
                                acc[u].x += __float2int_rn(f[u].x);
                                acc[u].y += __float2int_rn(f[u].y);
                                acc[u].z += __float2int_rn(f[u].z);
                                acc[u].w += __float2int_rn(f[u].w);
                            }
                        }
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int c = cg + 8 * u;
                    if (c >= wj)
                        continue;
                    int4 v = acc[u];
                    if (CORR) {
                        const int cb = (c & 3) * 32 + (c >> 2);
                        v.x += sc[lane * SCS + cb];
                        v.y += sc[(32 + lane) * SCS + cb];
                        v.z += sc[(64 + lane) * SCS + cb];
                        v.w += sc[(96 + lane) * SCS + cb];
                    }
                    if (thin) {
                        const float4 tf = __ldcs(
                            reinterpret_cast<const float4 *>(thin + bb * pstride + size_t(j0 + c) * ldp + i0 + r0));
                        v.x += __float2int_rn(tf.x);
                        v.y += __float2int_rn(tf.y);
                        v.z += __float2int_rn(tf.z);
                        v.w += __float2int_rn(tf.w);
                    }
                    int32_t *dst = C + bb * cstride + size_t(j0 + c) * ldc + i0 + r0;
                    if (vecC && r0 + 3 < wi) {
                        if (addc) {
                            const int4 o = *reinterpret_cast<const int4 *>(dst);
                            v.x += o.x;
                            v.y += o.y;
                            v.z += o.z;
                            v.w += o.w;
                        }
                        __stcs(reinterpret_cast<int4 *>(dst), v);
                    } else {
                        const int vv[4] = {v.x, v.y, v.z, v.w};
                        for (int u2 = 0; u2 < 4; ++u2)
                            if (r0 + u2 < wi)
                                dst[u2] = addc ? dst[u2] + vv[u2] : vv[u2];
                    }
                }
            }
        }
        if (CORR)
            __syncthreads();
    }
}

unsigned char ue4m3(double v) {
    for (int e = -6; e <= 8; ++e)
        for (int mm = 0; mm < 8; ++mm)
            if ((1.0 + mm / 8.0) * std::ldexp(1.0, e) == v)
                return (unsigned char)(((e + 7) << 3) | mm);
    return 0;
}
std::vector<uint8_t> build_scales(int rows, int kc, const unsigned char *v6) {
    const int Cc = 6 * kc / 16, Cpad = (Cc + 3) / 4 * 4, Rpad = (rows + 127) / 128 * 128;
    std::vector<uint8_t> s(size_t(Rpad) * Cpad, 0);
    for (int r = 0; r < Rpad; ++r)
        for (int c = 0; c < Cpad; ++c)
            s[(size_t(r / 128) * (Cpad / 4) + size_t(c / 4)) * 512 + size_t(r % 32) * 16 + size_t((r % 128) / 32) * 4 +
              size_t(c % 4)] = c < Cc ? v6[c % 6] : 0;
    return s;
}
} // namespace

struct PieceShape { // one K-length of a piece (the full 1,024 and the last one)
    int kc = 0;
    uint8_t *sa = nullptr, *sb = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool ok = false;
};

struct hayashiContext {
    cudaStream_t stream = 0;
    cublasLtHandle_t lt = nullptr;
    int sms = 0, enc_grid = 0, enc_grid_q = 0, add_grid0 = 0, add_grid1 = 0, add_grid2 = 0;
    size_t piece_limit = size_t(4) << 30;
    uint32_t *lut = nullptr;
    unsigned long long *cnt = nullptr; // cnt[0] = A, cnt[1] = B
    // per shape
    int m = -1, n = -1, k = -1, bc = 0, ldp = 0, mp = 0, np_ = 0, m_next = 0, n_next = 0, k_next = 0, bc_next = 0;
    bool loopb = false;
    cublasLtMatrixLayout_t lc = nullptr;
    PieceShape ps[2];
    int npieces = 0, group = 0;
    // workspace
    void *ws = nullptr;
    size_t ws_bytes = 0;
    uint8_t *qa = nullptr, *qb = nullptr;
    float *piece[MAXPIECES] = {};
    int2 *listA = nullptr, *listB = nullptr;
    unsigned long long cap = 0;
    int8_t *at = nullptr, *bt = nullptr;
    int *lineA = nullptr, *lineB = nullptr, *ptrA = nullptr, *ptrB = nullptr, *csrA = nullptr, *csrB = nullptr;
    void *ltws = nullptr;
    size_t ltws_bytes = size_t(64) << 20;
    size_t zbytes = 0; // cnt (16 B) + lineA (m) + lineB (n): one memset per call
    // CUDA Graph replay of the last call (graph mode)
    int graph_on = 0;
    cudaGraphExec_t gexec = nullptr;
    v5b::Ctx *v5 = nullptr;
    bool v5_active = false;
    unsigned *v5obs = nullptr;
    cudaEvent_t v5ev = nullptr;
    int v5_route = -1;
    bool v5_pend = false, v5_last = false;
    bool reuse_b = false;
    f16way::Way *f16 = nullptr;
    bool f16_active = false, f16_last = false;
    int32_t *f16tmp = nullptr;
    bool f5_active = false, f5_last = false, f5_pend = false, f5_b_ready = false, f5_table_ok = false;
    int f5_route = -1;
    unsigned *f5obs = nullptr;
    cudaEvent_t f5ev = nullptr;
    unsigned long long *f5lut = nullptr;
    unsigned f5tm[2] = {0, 0};
    int f5Mpad = 0, f5Npad = 0, f5cuts = 0;
    unsigned char *f5pa = nullptr, *f5pb = nullptr;
    size_t f5pa_bytes = 0, f5pb_bytes = 0;
    int32_t *f5c = nullptr;
    f5::Metrics *f5mt = nullptr;
    unsigned *f5E =
        nullptr; // [E0 A (Mpad) | E1 A | E0 B (Npad) | E1 B | gate (4 words) | flag (2 x u64) | stats (8 words)]
    CUtensorMap f5tmA, f5tmB;
    f5::Scales f5sc{}, f5sc2{};
    f5::DimsT f5dims{};
    struct GraphLast {
        bool dense_now = false;
        int last_direct = 0, last_tr = 0;
        unsigned long long last_scap = 0;
    } glast;
    struct Key {
        const void *A, *B, *C;
        long long sA, sB, sC;
        int m, n, k, lda, ldb, ldc, beta, bc;
        cudaStream_t s;
    } key;
    int backend = 2;
    bool d1_active = false;
    bool is_b300 = false;
    int trvar = 2;
    int direct = 1;
    int encvar = 5;
    int e2on = 0;
    bool e2_table_ok = false;
    bool e2_active = false;
    int prof = 0;
    cudaEvent_t ev[5] = {};
    int Mpad = 0, Npad2 = 0, cuts = 0, rowtiles = 0, coltiles = 0;
    unsigned char *pa = nullptr, *pb = nullptr;
    int *cint = nullptr;
    d1::Metrics *mt = nullptr;
    size_t pa_bytes = 0, pb_bytes = 0;
    CUtensorMap tmA, tmB;
    d1::Scales sc{};
    d1::DimsT dims{};
    int8_t *lw = nullptr;
    size_t lw_bytes = 0;
    bool e1_active = false;
    e1::DimsT e1dims{};
    e1::Metrics *e1mt = nullptr;
    e1::Scales e1sc{};
    CUtensorMap tmB3;
    unsigned char *pb3 = nullptr;
    size_t pb3_bytes = 0;
    int Npad3 = 0;
    e2::Scales e2sc{};
    e2::DimsT e2dims{};
    e2::Metrics *e2mt = nullptr;
    unsigned *rowE = nullptr;
    int *use = nullptr;
    struct GdHost {
        d1::DimsT d1;
        e2::DimsT e2;
        e1::DimsT e1;
        f5::DimsT f5;
    } *gdh = nullptr;
    int last_direct = 0, last_tr = 0;
    bool loop1 = false;
    unsigned long long *tot = nullptr;
    int tailsplit = 1;
    ZeroSpec zs{};
    int p103 = 0;
    bool dense_auto = false, dense_now = false, obs_pending = false, obs_dense = false, shift_ready = false;
    double obs_elems = 0, obs_lines = 0;
    unsigned long long last_scap = 0;
    int few_on = 0;
    long long n_mat = 0, n_dense = 0;
    static constexpr int kObsPos = 256;
    int cur_b = 0, obs_n = 0;
    unsigned char dense_pos[kObsPos] = {}, obs_dense_pos[kObsPos] = {};
    __half *lrA = nullptr, *lrB = nullptr;
    float *thin = nullptr;
    unsigned long long *shc = nullptr;
    unsigned long long *obs = nullptr;
    cudaEvent_t obs_ev = nullptr;
    cublasLtMatmulDesc_t thop = nullptr;
    cublasLtMatrixLayout_t thA = nullptr, thB = nullptr, thC = nullptr;
    cublasLtMatmulAlgo_t thalgo{};
};

#define HCK(x)                                                                                                         \
    do {                                                                                                               \
        if ((x) != cudaSuccess)                                                                                        \
            return HAYASHI_STATUS_CUDA_ERROR;                                                                          \
    } while (0)
#define HBL(x)                                                                                                         \
    do {                                                                                                               \
        if ((x) != CUBLAS_STATUS_SUCCESS)                                                                              \
            return HAYASHI_STATUS_CUBLAS_ERROR;                                                                        \
    } while (0)

constexpr double kV5RoutePpm = 100.0;
static void drop_graph(hayashiContext *h) {
    if (h->gexec) {
        cudaGraphExecDestroy(h->gexec);
        h->gexec = nullptr;
    }
}
static void free_shape(hayashiContext *h) {
    drop_graph(h);
    h->v5_active = false;
    h->v5_route = -1;
    h->v5_pend = false;
    h->v5_last = false;
    if (h->f16) {
        h->f16->destroy();
        delete h->f16;
        h->f16 = nullptr;
    }
    cudaFree(h->f16tmp);
    h->f16tmp = nullptr;
    h->f16_active = false;
    h->f16_last = false;
    cudaFree(h->f5pa);
    cudaFree(h->f5pb);
    cudaFree(h->f5c);
    cudaFree(h->f5mt);
    cudaFree(h->f5E);
    h->f5pa = h->f5pb = nullptr;
    h->f5c = nullptr;
    h->f5mt = nullptr;
    h->f5E = nullptr;
    h->f5_active = false;
    h->f5_last = false;
    h->f5_pend = false;
    h->f5_b_ready = false;
    h->f5_route = -1;
    for (auto &p : h->ps) {
        if (p.op)
            cublasLtMatmulDescDestroy(p.op);
        if (p.la)
            cublasLtMatrixLayoutDestroy(p.la);
        if (p.lb)
            cublasLtMatrixLayoutDestroy(p.lb);
        cudaFree(p.sa);
        cudaFree(p.sb);
        p = PieceShape{};
    }
    if (h->lc)
        cublasLtMatrixLayoutDestroy(h->lc);
    h->lc = nullptr;
    h->m = h->n = h->k = -1;
    h->bc = 0;
    cudaFree(h->pa);
    cudaFree(h->pb);
    cudaFree(h->cint);
    cudaFree(h->mt);
    h->pa = h->pb = nullptr;
    h->cint = nullptr;
    h->mt = nullptr;
    h->d1_active = false;
    cudaFree(h->e2mt);
    cudaFree(h->rowE);
    h->e2mt = nullptr;
    h->rowE = nullptr;
    h->use = nullptr;
    h->e2_active = false;
    cudaFree(h->e1mt);
    cudaFree(h->pb3);
    h->e1mt = nullptr;
    h->pb3 = nullptr;
    h->e1_active = false;
    cudaFree(h->lrA);
    cudaFree(h->lrB);
    cudaFree(h->thin);
    cudaFree(h->shc);
    h->lrA = h->lrB = nullptr;
    h->thin = nullptr;
    h->shc = nullptr;
    h->shift_ready = false;
    if (h->thA)
        cublasLtMatrixLayoutDestroy(h->thA);
    if (h->thB)
        cublasLtMatrixLayoutDestroy(h->thB);
    if (h->thC)
        cublasLtMatrixLayoutDestroy(h->thC);
    if (h->thop)
        cublasLtMatmulDescDestroy(h->thop);
    h->thA = h->thB = h->thC = nullptr;
    h->thop = nullptr;
}

// backend 1: sizes, buffers, tensor maps and the runtime dimensions of d1::p2_kernel for (m, n, k), batch 1, k <=
// 16,384. K is padded to a multiple of 64 (k <= 1,024) or of 1,024 (the kernel's 1,024-long segments); M to 256 (one
// pair = 256 rows); N to 224 (one tile).  Padding rows / cuts are zero faces written by encode_tiled.
static hayashiStatus_t set_shift(hayashiContext *h, int m, int n, int k);
static hayashiStatus_t set_d1(hayashiContext *h, int m, int n, int k) {
    const int kpad = k <= 1024 ? (k + 63) / 64 * 64 : (k + 1023) / 1024 * 1024;
    const bool e2 = h->e2on && h->e2_table_ok;
    h->cuts = kpad / 16;
    h->Mpad = e2 ? (m + 511) / 512 * 512 : (m + 255) / 256 * 256;
    h->coltiles = (n + d1::TileN - 1) / d1::TileN;
    h->Npad2 = h->coltiles * d1::TileN;
    h->rowtiles = h->Mpad / d1::TileM;
    h->pa_bytes = size_t(h->Mpad) * h->cuts * 48;
    h->pb_bytes = size_t(h->Npad2) * h->cuts * 48;
    const int tiles = h->rowtiles * h->coltiles;
    HCK(cudaMalloc(&h->pa, h->pa_bytes));
    HCK(cudaMalloc(&h->pb, h->pb_bytes));
    HCK(cudaMalloc(&h->cint, size_t(h->Mpad) * n * 4));
    HCK(cudaMalloc(&h->mt, size_t(tiles) * sizeof(d1::Metrics)));
    {
        const int fw = tiles / d1::Pairs, ft = fw * d1::Pairs, tt = tiles - ft, nsg = h->cuts > 64 ? h->cuts / 64 : 1,
                  rq = h->cuts / 4;
        int S = 1;
        if (h->tailsplit && tt > 0 && nsg >= 2)
            for (const int c : {8, 4, 2})
                if (nsg % c == 0 && rq % c == 0 && tt * c <= d1::Pairs) {
                    S = c;
                    break;
                }
        h->dims = d1::DimsT{h->Mpad,
                            n,
                            h->rowtiles,
                            h->coltiles,
                            tiles,
                            (tiles + d1::Pairs - 1) / d1::Pairs,
                            d1::RowTilesPerBand * h->coltiles,
                            S,
                            fw,
                            ft,
                            tt};
        h->zs = ZeroSpec{};
        h->zs.ntiles = S > 1 ? tt : 0;
        const unsigned tpb = unsigned(d1::RowTilesPerBand * h->coltiles);
        for (int i = 0; i < h->zs.ntiles; ++i) { // tile_row / tile_col of d1_kernel.cuh on the host
            const unsigned tile = unsigned(ft + i), band = tile / tpb, r0 = band * d1::RowTilesPerBand,
                           hh = std::min(unsigned(d1::RowTilesPerBand), unsigned(h->rowtiles) - r0);
            h->zs.tr[i] = (unsigned short)(r0 + (tile - band * tpb) % hh);
            h->zs.tc[i] = (unsigned short)((tile - band * tpb) / hh);
        }
    }
    typedef CUresult (*EncodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult qr;
    HCK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr));
    if (!fn || qr != cudaDriverEntryPointSuccess)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    auto make = [&](CUtensorMap *tm, void *b, size_t bytes, unsigned box_rows) {
        const cuuint64_t gdim[2] = {256, cuuint64_t(bytes / 256)};
        const cuuint64_t gs[1] = {256};
        const cuuint32_t box[2] = {256, box_rows};
        const cuuint32_t es[2] = {1, 1};
        return ((EncodeTiled)fn)(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, b, gdim, gs, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    memset(&h->tmA, 0, sizeof h->tmA);
    memset(&h->tmB, 0, sizeof h->tmB);
    if (make(&h->tmA, h->pa, h->pa_bytes, d1::ATile / 256) != CUDA_SUCCESS ||
        make(&h->tmB, h->pb, h->pb_bytes, d1::BTile / 256) != CUDA_SUCCESS)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    const double la6[6] = {8, 20, 52, 2, 20, 4}, lb6[6] = {16, 32, 64, 10, 26, 26};
    unsigned char A6[6], B6[6];
    for (int f = 0; f < 6; ++f) {
        A6[f] = ue4m3(la6[f]);
        B6[f] = ue4m3(lb6[f]);
    }
    h->sc = d1::Scales{unsigned(A6[0]) | unsigned(A6[1]) << 8 | unsigned(A6[2]) << 16 | unsigned(A6[3]) << 24,
                       unsigned(A6[4]) | unsigned(A6[5]) << 8,
                       unsigned(B6[0]) | unsigned(B6[1]) << 8 | unsigned(B6[2]) << 16 | unsigned(B6[3]) << 24,
                       unsigned(B6[4]) | unsigned(B6[5]) << 8};
    if (e2) {
        static_assert(e2::ATile == d1::ATile && e2::BTile == d1::BTile && e2::TileN == d1::TileN,
                      "e2 and 7154 share the packed layout");
        const int rt2 = h->Mpad / e2::TileM, tiles2 = rt2 * h->coltiles;
        h->e2dims = e2::DimsT{h->Mpad,
                              n,
                              rt2,
                              h->coltiles,
                              tiles2,
                              (tiles2 + e2::Clusters - 1) / e2::Clusters,
                              e2::RowTilesPerBand * h->coltiles};
        HCK(cudaMalloc(&h->e2mt, size_t(tiles2) * 2 * sizeof(e2::Metrics)));
        HCK(cudaMalloc(&h->rowE, size_t(h->Mpad + h->Npad2) * 4 + 16));
        h->use = reinterpret_cast<int *>(h->rowE + h->Mpad + h->Npad2);
        HCK(cudaMemset(h->use, 0, 16));
        h->e2sc = e2::Scales{h->sc.a0, h->sc.a1, h->sc.b0, h->sc.b1};
        h->e2_active = true;
    }
    HCK(cudaStreamSynchronize(h->stream));
    h->gdh->d1 = h->dims;
    h->gdh->e2 = h->e2dims;
    h->d1_active = true;
    return set_shift(h, m, n, k);
}

static hayashiStatus_t set_f5(hayashiContext *h, int m, int n, int k) {
    if (!h->f5_table_ok || k > 16384)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    const int kpad = (k + 63) / 64 * 64, ct = (n + f5::TileN - 1) / f5::TileN;
    h->f5cuts = kpad / 16;
    h->f5Mpad = (m + 255) / 256 * 256;
    h->f5Npad = ct * f5::TileN;
    const int rt = h->f5Mpad / f5::TileM, tiles = rt * ct;
    h->f5pa_bytes = size_t(h->f5Mpad) * h->f5cuts * 48;
    h->f5pb_bytes = size_t(h->f5Npad) * h->f5cuts * 48;
    const size_t eb = (size_t(2) * (h->f5Mpad + h->f5Npad) + 16) * 4;
    HCK(cudaMalloc(&h->f5pa, h->f5pa_bytes));
    HCK(cudaMalloc(&h->f5pb, h->f5pb_bytes));
    HCK(cudaMalloc(&h->f5c, size_t(h->f5Mpad) * n * 4));
    HCK(cudaMalloc(&h->f5mt, size_t(tiles) * 2 * sizeof(f5::Metrics)));
    HCK(cudaMalloc(&h->f5E, eb));
    HCK(cudaMemset(h->f5E, 0, eb));
    h->f5dims =
        f5::DimsT{h->f5Mpad, n, rt, ct, tiles, (tiles + f5::Clusters - 1) / f5::Clusters, f5::RowTilesPerBand * ct};
    typedef CUresult (*EncodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult qr;
    HCK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr));
    if (!fn || qr != cudaDriverEntryPointSuccess)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    auto make = [&](CUtensorMap *tm, void *b, size_t bytes,
                    unsigned box_rows) { // Tensor maps use two dimensions with 256-byte rows.
        const cuuint64_t gdim[2] = {256, cuuint64_t(bytes / 256)};
        const cuuint64_t gs[1] = {256};
        const cuuint32_t box[2] = {256, box_rows};
        const cuuint32_t es[2] = {1, 1};
        return ((EncodeTiled)fn)(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, b, gdim, gs, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    memset(&h->f5tmA, 0, sizeof h->f5tmA);
    memset(&h->f5tmB, 0, sizeof h->f5tmB);
    if (make(&h->f5tmA, h->f5pa, h->f5pa_bytes, f5::ATile / 256) != CUDA_SUCCESS ||
        make(&h->f5tmB, h->f5pb, h->f5pb_bytes, f5::BTile / 256) != CUDA_SUCCESS)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    const double fa0[3] = {2, 7, 7}, fb0[3] = {2, 8, 32}, fa1[3] = {2, 4, 7}, fb1[3] = {2, 4, 32};
    auto pack = [&](const double *sa3, const double *sb3) {
        unsigned char A6[6], B6[6];
        for (int j = 0; j < 6; ++j) {
            A6[j] = ue4m3(sa3[j % 3]);
            B6[j] = ue4m3(sb3[j % 3]);
        }
        return f5::Scales{unsigned(A6[0]) | unsigned(A6[1]) << 8 | unsigned(A6[2]) << 16 | unsigned(A6[3]) << 24,
                          unsigned(A6[4]) | unsigned(A6[5]) << 8,
                          unsigned(B6[0]) | unsigned(B6[1]) << 8 | unsigned(B6[2]) << 16 | unsigned(B6[3]) << 24,
                          unsigned(B6[4]) | unsigned(B6[5]) << 8};
    };
    h->f5sc = pack(fa0, fb0);
    h->f5sc2 = pack(fa1, fb1);
    if (!h->f5obs) {
        HCK(cudaMallocHost(&h->f5obs, 64));
        HCK(cudaEventCreateWithFlags(&h->f5ev, cudaEventDisableTiming));
    }
    HCK(cudaStreamSynchronize(h->stream));
    h->gdh->f5 = h->f5dims;
    h->f5_active = true;
    h->f5_b_ready = false;
    h->f5_route = -1;
    h->f5_pend = false;
    return HAYASHI_STATUS_SUCCESS;
}

static hayashiStatus_t set_shift(hayashiContext *h, int m, int n, int k) {
    if (h->p103 == 1 || k > 16384 || k % 16)
        return HAYASHI_STATUS_SUCCESS;
    const int k2 = 2 * (k / 16);
    HCK(cudaMalloc(&h->lrA, size_t(m) * k2 * 2));
    HCK(cudaMalloc(&h->lrB, size_t(n) * k2 * 2));
    HCK(cudaMalloc(&h->thin, size_t(h->Mpad) * n * 4));
    HCK(cudaMalloc(&h->shc, 32));
    HBL(cublasLtMatmulDescCreate(&h->thop, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t tt = CUBLAS_OP_T, tn = CUBLAS_OP_N;
    HBL(cublasLtMatmulDescSetAttribute(h->thop, CUBLASLT_MATMUL_DESC_TRANSA, &tt, sizeof(tt)));
    HBL(cublasLtMatmulDescSetAttribute(h->thop, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
    HBL(cublasLtMatrixLayoutCreate(&h->thA, CUDA_R_16F, k2, m, k2));
    HBL(cublasLtMatrixLayoutCreate(&h->thB, CUDA_R_16F, k2, n, k2));
    HBL(cublasLtMatrixLayoutCreate(&h->thC, CUDA_R_32F, m, n, h->Mpad));
    cublasLtMatmulPreference_t pref;
    HBL(cublasLtMatmulPreferenceCreate(&pref));
    HBL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->ltws_bytes,
                                             sizeof(size_t)));
    cublasLtMatmulHeuristicResult_t r[1];
    int got = 0;
    const cublasStatus_t hs =
        cublasLtMatmulAlgoGetHeuristic(h->lt, h->thop, h->thA, h->thB, h->thC, h->thC, pref, 1, r, &got);
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs == CUBLAS_STATUS_SUCCESS && got > 0) {
        h->thalgo = r[0].algo;
        h->shift_ready = true;
    }
    return HAYASHI_STATUS_SUCCESS;
}
static hayashiStatus_t run_thin(hayashiContext *h, int m, int n, int32_t *out, cudaStream_t s) {
    const float one = 1.f, zero = 0.f;
    HBL(cublasLtMatmul(h->lt, h->thop, &one, h->lrA, h->thA, h->lrB, h->thB, &zero, h->thin, h->thC, h->thin, h->thC,
                       &h->thalgo, h->ltws, h->ltws_bytes, s));
    HCK(cudaGetLastError());
    return HAYASHI_STATUS_SUCCESS;
}
static hayashiStatus_t set_e1(hayashiContext *h, int m, int n, int k) {
    const int kpad = (k + 63) / 64 * 64, ct = (n + e1::TileN - 1) / e1::TileN;
    h->cuts = kpad / 16;
    h->Mpad = (m + 255) / 256 * 256;
    h->Npad3 = ct * e1::TileN;
    h->pa_bytes = size_t(h->Mpad) * h->cuts * 48;
    h->pb3_bytes = size_t(h->Npad3) * h->cuts * 48;
    const int rt = h->Mpad / e1::TileM, tiles = rt * ct;
    HCK(cudaMalloc(&h->pa, h->pa_bytes));
    HCK(cudaMalloc(&h->pb3, h->pb3_bytes));
    HCK(cudaMalloc(&h->cint, size_t(h->Mpad) * n * 4));
    HCK(cudaMalloc(&h->e1mt, size_t(tiles) * 2 * sizeof(e1::Metrics)));
    const int rtpb = h->Mpad >= 16384 ? 8 : e1::RowTilesPerBand;
    h->e1dims = e1::DimsT{h->Mpad, n, rt, ct, tiles, (tiles + e1::Clusters - 1) / e1::Clusters, rtpb * ct, rtpb};
    typedef CUresult (*EncodeTiled)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                    const cuuint64_t *, const cuuint32_t *, const cuuint32_t *, CUtensorMapInterleave,
                                    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult qr;
    HCK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr));
    if (!fn || qr != cudaDriverEntryPointSuccess)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    auto make = [&](CUtensorMap *tm, void *b, size_t bytes, unsigned box_rows) {
        const cuuint64_t gdim[2] = {256, cuuint64_t(bytes / 256)};
        const cuuint64_t gs[1] = {256};
        const cuuint32_t box[2] = {256, box_rows};
        const cuuint32_t es[2] = {1, 1};
        return ((EncodeTiled)fn)(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, b, gdim, gs, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    memset(&h->tmA, 0, sizeof h->tmA);
    memset(&h->tmB3, 0, sizeof h->tmB3);
    if (make(&h->tmA, h->pa, h->pa_bytes, e1::ATile / 256) != CUDA_SUCCESS ||
        make(&h->tmB3, h->pb3, h->pb3_bytes, e1::BTile / 256) != CUDA_SUCCESS)
        return HAYASHI_STATUS_NOT_SUPPORTED;
    const double la6[6] = {8, 20, 52, 2, 20, 4}, lb6[6] = {16, 32, 64, 10, 26, 26};
    unsigned char A6[6], B6[6];
    for (int f = 0; f < 6; ++f) {
        A6[f] = ue4m3(la6[f]);
        B6[f] = ue4m3(lb6[f]);
    }
    h->sc = d1::Scales{unsigned(A6[0]) | unsigned(A6[1]) << 8 | unsigned(A6[2]) << 16 | unsigned(A6[3]) << 24,
                       unsigned(A6[4]) | unsigned(A6[5]) << 8,
                       unsigned(B6[0]) | unsigned(B6[1]) << 8 | unsigned(B6[2]) << 16 | unsigned(B6[3]) << 24,
                       unsigned(B6[4]) | unsigned(B6[5]) << 8};
    h->e1sc = e1::Scales{h->sc.a0, h->sc.a1, h->sc.b0, h->sc.b1};
    HCK(cudaStreamSynchronize(h->stream));
    h->gdh->e1 = h->e1dims;
    h->e1_active = true;
    return set_shift(h, m, n, k);
}

// cuBLASLt descriptions per piece length; batched = batch count bc with strided offsets (scales repeated per batch)
static hayashiStatus_t make_descs(hayashiContext *h, bool batched) {
    const int m = h->m_next, n = h->n_next, k = h->k_next, bc = h->bc_next, mp = h->mp, np_ = h->np_, ldp = h->ldp;
    const int npieces = h->npieces, kl = k - (npieces - 1) * KC;
    const double la6[6] = {8, 20, 52, 2, 20, 4}, lb6[6] = {16, 32, 64, 10, 26, 26};
    unsigned char A6[6], B6[6]; // lambda = sA * sB / 4
    for (int f = 0; f < 6; ++f) {
        A6[f] = ue4m3(la6[f]);
        B6[f] = ue4m3(lb6[f]);
    }
    auto setb = [&](cublasLtMatrixLayout_t l, long long stride) -> hayashiStatus_t {
        if (!batched)
            return HAYASHI_STATUS_SUCCESS;
        HBL(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
        HBL(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
        return HAYASHI_STATUS_SUCCESS;
    };
    HBL(cublasLtMatrixLayoutCreate(&h->lc, CUDA_R_32F, mp, np_, ldp));
    hayashiStatus_t st = setb(h->lc, (long long)mp * np_);
    if (st)
        return st;
    const int lens[2] = {KC, kl};
    for (int t = 0; t < 2; ++t) {
        PieceShape &s = h->ps[t];
        if (t == 1 && kl == KC)
            break; // every piece is full
        if (t == 0 && npieces == 1 && kl != KC)
            continue; // only the short piece exists
        s.kc = lens[t];
        auto one_a = build_scales(mp, s.kc, A6), one_b = build_scales(np_, s.kc, B6);
        std::vector<uint8_t> hsa, hsb;
        for (int b = 0; b < bc; ++b) {
            hsa.insert(hsa.end(), one_a.begin(), one_a.end());
            hsb.insert(hsb.end(), one_b.begin(), one_b.end());
        }
        HCK(cudaMalloc(&s.sa, hsa.size()));
        HCK(cudaMalloc(&s.sb, hsb.size()));
        HCK(cudaMemcpy(s.sa, hsa.data(), hsa.size(), cudaMemcpyHostToDevice));
        HCK(cudaMemcpy(s.sb, hsb.data(), hsb.size(), cudaMemcpyHostToDevice));
        HBL(cublasLtMatmulDescCreate(&s.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        HBL(cublasLtMatmulDescSetAttribute(s.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        HBL(cublasLtMatmulDescSetAttribute(s.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        auto sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        HBL(cublasLtMatmulDescSetAttribute(s.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
        HBL(cublasLtMatmulDescSetAttribute(s.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
        HBL(cublasLtMatmulDescSetAttribute(s.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &s.sa, sizeof(void *)));
        HBL(cublasLtMatmulDescSetAttribute(s.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &s.sb, sizeof(void *)));
        HBL(cublasLtMatrixLayoutCreate(&s.la, CUDA_R_4F_E2M1, 6 * s.kc, mp,
                                       6 * k)); // a piece = 6*kc columns of the 6k-long rows
        HBL(cublasLtMatrixLayoutCreate(&s.lb, CUDA_R_4F_E2M1, 6 * s.kc, np_, 6 * k));
        st = setb(s.la, (long long)mp * 6 * k);
        if (st)
            return st;
        st = setb(s.lb, (long long)np_ * 6 * k);
        if (st)
            return st;
        cublasLtMatmulPreference_t pref;
        HBL(cublasLtMatmulPreferenceCreate(&pref));
        HBL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->ltws_bytes,
                                                 sizeof(size_t)));
        cublasLtMatmulHeuristicResult_t r[1];
        int got = 0;
        const cublasStatus_t cs =
            cublasLtMatmulAlgoGetHeuristic(h->lt, s.op, s.la, s.lb, h->lc, h->lc, pref, 1, r, &got);
        cublasLtMatmulPreferenceDestroy(pref);
        if (cs != CUBLAS_STATUS_SUCCESS || got < 1)
            return HAYASHI_STATUS_NOT_SUPPORTED;
        s.algo = r[0].algo;
        s.ok = true;
    }
    return HAYASHI_STATUS_SUCCESS;
}

static hayashiStatus_t set_shape(hayashiContext *h, int m, int n, int k, int bc) {
    if (m == h->m && n == h->n && k == h->k && bc == h->bc)
        return HAYASHI_STATUS_SUCCESS;
    cudaStreamSynchronize(h->stream);
    free_shape(h);
    const int npieces = (k + KC - 1) / KC;
    const int mp = (m + 15) / 16 * 16, np_ = (n + 15) / 16 * 16,
              ldp = mp;                              // the GEMM runs on m, n padded to 16 (zero rows)
    const size_t fbytes = size_t(mp) * np_ * 4 * bc; // one piece, every batch
    const int group = int(std::min<size_t>(std::min(npieces, MAXPIECES), std::max<size_t>(1, h->piece_limit / fbytes)));
    const unsigned long long cap = std::max<unsigned long long>(65536ull, (unsigned long long)std::max(m, n) * k * bc /
                                                                              HAYASHI_I8_OUTLIER_CAP_DIV);
    if (cap > 0x7FFFFFFFull)
        return HAYASHI_STATUS_NOT_SUPPORTED; // CSR offsets are int
    // workspace: faces A, faces B, pieces, the +-103 records (lists, line counts, offsets, CSR), At, Bt, cuBLASLt
    const size_t fa = size_t(mp) * k * 3 * bc, fb = size_t(np_) * k * 3 * bc, al = 256;
    const size_t la_ = size_t(m) * bc, lb_ = size_t(n) * bc; // CSR lines: A rows, B columns of every batch
    auto up = [&](size_t x) { return (x + al - 1) / al * al; };
    const size_t zbytes = 16 + (la_ + lb_) * 4;
    const size_t need = up(fa) + up(fb) + size_t(group) * up(fbytes) + 2 * up(cap * 8) + up(zbytes) +
                        up((la_ + 1) * 4) + up((lb_ + 1) * 4) + 2 * up(cap * 4) + up(size_t(k) * la_) +
                        up(size_t(k) * lb_) + up(h->ltws_bytes);
    if (need > h->ws_bytes) {
        cudaFree(h->ws);
        h->ws = nullptr;
        h->ws_bytes = 0;
        if (cudaMalloc(&h->ws, need) != cudaSuccess)
            return HAYASHI_STATUS_ALLOC_FAILED;
        h->ws_bytes = need;
    }
    uint8_t *p = static_cast<uint8_t *>(h->ws);
    h->qa = p;
    p += up(fa);
    h->qb = p;
    p += up(fb);
    for (int g = 0; g < MAXPIECES; ++g)
        h->piece[g] = nullptr;
    for (int g = 0; g < group; ++g) {
        h->piece[g] = reinterpret_cast<float *>(p);
        p += up(fbytes);
    }
    h->listA = reinterpret_cast<int2 *>(p);
    p += up(cap * 8);
    h->listB = reinterpret_cast<int2 *>(p);
    p += up(cap * 8);
    h->cnt = reinterpret_cast<unsigned long long *>(p);
    h->lineA = reinterpret_cast<int *>(p + 16);
    h->lineB = h->lineA + la_;
    h->zbytes = zbytes;
    HCK(cudaMemset(p, 0, zbytes));
    p += up(zbytes);
    h->ptrA = reinterpret_cast<int *>(p);
    p += up((la_ + 1) * 4);
    h->ptrB = reinterpret_cast<int *>(p);
    p += up((lb_ + 1) * 4);
    h->csrA = reinterpret_cast<int *>(p);
    p += up(cap * 4);
    h->csrB = reinterpret_cast<int *>(p);
    p += up(cap * 4);
    h->at = reinterpret_cast<int8_t *>(p);
    p += up(size_t(k) * la_);
    h->bt = reinterpret_cast<int8_t *>(p);
    p += up(size_t(k) * lb_);
    h->ltws = p;
    h->cap = cap;
    h->group = group;
    h->npieces = npieces;
    h->ldp = ldp;
    h->mp = mp;
    h->np_ = np_;
    for (int b = 0; b < bc; ++b) { // padding rows of every batch: zero faces, never written
        if (mp > m)
            HCK(cudaMemset(h->qa + (size_t(b) * mp + m) * k * 3, 0, size_t(mp - m) * k * 3));
        if (np_ > n)
            HCK(cudaMemset(h->qb + (size_t(b) * np_ + n) * k * 3, 0, size_t(np_ - n) * k * 3));
    }
    h->m_next = m;
    h->n_next = n;
    h->k_next = k;
    h->bc_next = bc;
    h->loopb = false;
    hayashiStatus_t st = make_descs(h, bc > 1);
    if (st == HAYASHI_STATUS_NOT_SUPPORTED && bc > 1) { // cuBLASLt refused the batched layouts: one call per batch
        free_shape(h);
        h->loopb = true;
        st = make_descs(h, false);
    }
    if (st != HAYASHI_STATUS_SUCCESS) {
        free_shape(h);
        return st;
    }
    const long long v5mp = ((long long)m + 2047) / 2048 * 2048;
    const bool f5_auto = h->backend == 2 && bc == 1 && h->is_b300 && h->f5_table_ok && k <= 16384 &&
                         (double)m * n >= 7168.0 * 7168.0 && (double)m * n * k >= 1.0e11;
    const bool v5_auto = h->backend == 2 && !f5_auto && (double)m * n >= 8192.0 * 8192.0 &&
                         (double)m * n * k >= (h->reuse_b ? 5.0e11 : 1.0e12) && v5mp * 8 <= (long long)m * 9;
    if ((h->backend == 4 || v5_auto) && bc == 1 && h->is_b300) {
        if (!h->v5)
            h->v5 = v5b::create();
        const int r = v5b::set_shape(h->v5, m, n, k);
        if (r && !(v5_auto && r == 1)) {
            free_shape(h);
            return r == 1 ? HAYASHI_STATUS_NOT_SUPPORTED : HAYASHI_STATUS_CUDA_ERROR;
        }
        if (r == 0) {
            h->v5_active = true;
            if (!h->v5obs) {
                HCK(cudaMallocHost(&h->v5obs, 128));
                HCK(cudaEventCreateWithFlags(&h->v5ev, cudaEventDisableTiming));
            }
        }
    }
    if ((h->backend == 6 || h->v5_active || f5_auto) && bc == 1 && h->is_b300 && k <= 16384) {
        st = set_f5(h, m, n, k);
        if (st != HAYASHI_STATUS_SUCCESS) {
            if (h->backend == 6 || st != HAYASHI_STATUS_NOT_SUPPORTED) {
                free_shape(h);
                return st;
            }
            st = HAYASHI_STATUS_SUCCESS;
        }
    }
    if (h->backend == 6 && !h->f5_active) {
        free_shape(h);
        return HAYASHI_STATUS_NOT_SUPPORTED;
    } // not B300 / batched / k > 16,384
    if (bc == 1 && h->is_b300 && (h->backend == 5 || h->backend == 2 || h->v5_active || h->f5_active)) {
        try {
            h->f16 = new f16way::Way();
            h->f16->init(h->lt, h->stream, m, n, k, 1, size_t(256) << 20);
            h->f16->best_form = f16way::CHUNK;
            h->f16->best_index = 0;
            h->f16_active = true;
        } catch (...) {
            if (h->f16) {
                h->f16->destroy();
                delete h->f16;
                h->f16 = nullptr;
            }
            h->f16_active = false;
            cudaGetLastError();
            if (h->backend == 5) {
                free_shape(h);
                return HAYASHI_STATUS_CUDA_ERROR;
            }
        }
    }
    const bool as_auto = h->backend == 2 || h->v5_active;
    const bool auto_ok = as_auto && double(m) * n >= 1048576.0 && k > 1024,
               auto_ok3 = as_auto && double(m) * n >= 4194304.0;
    if ((h->backend == 3 || auto_ok3) && bc == 1 && k <= 1024 && h->is_b300) {
        st = set_e1(h, m, n, k);
        if (st != HAYASHI_STATUS_SUCCESS) {
            free_shape(h);
            return st;
        }
    } else if ((h->backend == 1 || auto_ok) && bc == 1 && k <= 16384 && h->is_b300) {
        st = set_d1(h, m, n, k);
        if (st != HAYASHI_STATUS_SUCCESS) {
            free_shape(h);
            return st;
        }
    }
    h->m = m;
    h->n = n;
    h->k = k;
    h->bc = bc;
    return HAYASHI_STATUS_SUCCESS;
}

// value -> six codes, A (flipped differences) and B; +-103 get the entry of +-102
static hayashiStatus_t load_table(hayashiContext *h, int which) {
    const signed char (*TA)[6] = which == 1 ? Z4_A : which == 2 ? ZG_A : LUT_A;
    const signed char (*TB)[6] = which == 1 ? Z4_B : which == 2 ? ZG_B : LUT_B;
    std::vector<uint32_t> t(512);
    bool ok3 = true;
    for (int side = 0; side < 2; ++side)
        for (int u = 0; u < 256; ++u) {
            int x = int(int8_t(uint8_t(u)));
            if (x == 103)
                x = 102;
            if (x == -103)
                x = -102;
            const signed char *row = side ? TB[x + 128] : TA[x + 128];
            uint32_t v = 0;
            for (int f = 0; f < 6; ++f) {
                const int c = row[f], a = c < 0 ? -c : c;
                const unsigned mg = a <= 4 ? unsigned(a) : (a == 6 ? 5u : (a == 8 ? 6u : 7u));
                v |= (c == 0 ? 0u : (mg | (c < 0 ? 8u : 0u))) << (4 * f);
            }
            {
                static const int lam[6] = {32, 160, 832, 5, 130, 26};
                long long E = 0;
                for (int f = 0; f < 6; ++f)
                    E += (long long)lam[f] * row[f] * row[f];
                const long long b3 = (E + 64) / 65;
                if (b3 > 255)
                    ok3 = false;
                v |= uint32_t(b3 > 255 ? 255 : b3) << 24;
            }
            t[side * 256 + u] = v;
        }
    h->e2_table_ok = ok3;
    HCK(cudaMemcpy(h->lut, t.data(), 2048, cudaMemcpyHostToDevice));
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiCreate(hayashiHandle_t *out) {
    if (!out)
        return HAYASHI_STATUS_INVALID_VALUE;
    hayashiContext *h = new hayashiContext();
    if (cublasLtCreate(&h->lt) != CUBLAS_STATUS_SUCCESS) {
        delete h;
        return HAYASHI_STATUS_CUBLAS_ERROR;
    }
    int dev = 0;
    HCK(cudaGetDevice(&dev));
    HCK(cudaDeviceGetAttribute(&h->sms, cudaDevAttrMultiProcessorCount, dev));
    int o = 0;
    HCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, encode_ab<0>, 256, 0));
    h->enc_grid = h->sms * std::max(o, 1);
    HCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, encode_ab<1>, 256, 0));
    h->enc_grid_q = h->sms * std::max(o, 1);
    HCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, add_tile<false>, 256, 0));
    h->add_grid0 = h->sms * std::max(o, 1);
    HCK(cudaFuncSetAttribute(add_tile<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, T2 * SCS * 4));
    HCK(cudaFuncSetAttribute(add_tile<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, T2 * SCS * 4));
    HCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, add_tile<true>, 256, T2 * SCS * 4));
    h->add_grid1 = h->sms * std::max(o, 1);
    HCK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, add_tile<false, true, 1, true>, 256, 0));
    h->add_grid2 = h->sms * std::max(o, 1);
    HCK(cudaMalloc(&h->lut, 2048));
    HCK(cudaMalloc(&h->tot, 32));
    HCK(cudaMemset(h->tot, 0, 32));
    HCK(cudaMallocHost(&h->obs, 48 * hayashiContext::kObsPos));
    memset(h->obs, 0, 48 * hayashiContext::kObsPos);
    HCK(cudaEventCreateWithFlags(&h->obs_ev, cudaEventDisableTiming));
    HCK(cudaMallocHost(&h->gdh, sizeof(*h->gdh)));
    memset(h->gdh, 0, sizeof(*h->gdh));
    {
        const hayashiStatus_t st = load_table(h, 0);
        if (st != HAYASHI_STATUS_SUCCESS)
            return st;
    }
    {
        cudaDeviceProp pr{};
        HCK(cudaGetDeviceProperties(&pr, dev));
        h->is_b300 = pr.major == 10 && (pr.minor == 3 || pr.minor == 0) && h->sms >= d1::Grid;
    }
    if (h->is_b300)
        HCK(cudaFuncSetAttribute(d1::p2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, d1::SharedBytes));
    if (h->is_b300)
        HCK(cudaFuncSetAttribute(e2::p2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, e2::SharedBytes));
    if (h->is_b300)
        HCK(cudaFuncSetAttribute(e1::p2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, e1::SharedBytes));
    if (h->is_b300)
        HCK(cudaFuncSetAttribute(f5::p2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, f5::SharedBytes));
    {
        unsigned long long t[512];
        h->f5_table_ok = f5e::build_table(t, h->f5tm);
        if (h->f5_table_ok) {
            HCK(cudaMalloc(&h->f5lut, sizeof t));
            HCK(cudaMemcpy(h->f5lut, t, sizeof t, cudaMemcpyHostToDevice));
        }
    }
    *out = h;
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiDestroy(hayashiHandle_t h) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    free_shape(h);
    for (auto &e : h->ev)
        if (e)
            cudaEventDestroy(e);
    cudaFree(h->ws);
    cudaFree(h->lw);
    cudaFree(h->lut);
    cudaFree(h->tot);
    cudaFreeHost(h->obs);
    if (h->obs_ev)
        cudaEventDestroy(h->obs_ev);
    cudaFreeHost(h->gdh);
    cublasLtDestroy(h->lt);
    if (h->v5)
        v5b::destroy(h->v5);
    if (h->v5obs)
        cudaFreeHost(h->v5obs);
    if (h->v5ev)
        cudaEventDestroy(h->v5ev);
    cudaFree(h->f5lut);
    if (h->f5obs)
        cudaFreeHost(h->f5obs);
    if (h->f5ev)
        cudaEventDestroy(h->f5ev);
    delete h;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetStream(hayashiHandle_t h, cudaStream_t s) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    h->stream = s;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetPieceBufferLimit(hayashiHandle_t h, size_t b) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    h->piece_limit = b;
    free_shape(h);
    return HAYASHI_STATUS_SUCCESS;
}

// Enqueue one strided-batched product on h->stream (the shape is already set).  Strides in elements.
static hayashiStatus_t issue_fast(hayashiContext *h, int m, int n, int k, const int8_t *A, int lda, long long sA,
                                  const int8_t *B, int ldb, long long sB, int beta1, int32_t *C, int ldc, long long sC,
                                  int bc);
__global__ void add_counts(const unsigned long long *cnt, unsigned long long *tot, unsigned long long cap,
                           const unsigned long long *shc) {
    tot[0] += shc ? shc[2] : cnt[0];
    tot[1] += shc ? shc[3] : cnt[1];
    if (cnt[0] > cap)
        tot[2] = 1;
    if (cnt[1] > cap)
        tot[3] = 1;
}
static void poll_obs(hayashiContext *h) {
    if (h->p103 != 0 || !h->obs_pending)
        return;
    if (cudaEventQuery(h->obs_ev) == cudaSuccess) {
        for (int b = 0; b < h->obs_n; ++b) {
            const unsigned long long *o = h->obs + 6 * b;
            const unsigned long long raw = h->obs_dense_pos[b] ? o[4] + o[5] : o[0] + o[1];
            h->dense_pos[b] = double(raw) > 2.0 * h->obs_lines ? 1 : 0;
        }
        h->dense_auto = h->obs_n > 0 && h->dense_pos[0];
        h->obs_pending = false;
    }
    cudaGetLastError();
}
static hayashiStatus_t issue(hayashiContext *h, int m, int n, int k, const int8_t *A, int lda, long long sA,
                             const int8_t *B, int ldb, long long sB, int beta1, int32_t *C, int ldc, long long sC,
                             int bc) {
    if (h->loop1 && bc > 1) {
        HCK(cudaMemsetAsync(h->tot, 0, 32, h->stream));
        {
            cudaStreamCaptureStatus cs0 = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(h->stream, &cs0) == cudaSuccess && cs0 == cudaStreamCaptureStatusNone)
                poll_obs(h);
            cudaGetLastError();
        }
        h->obs_n = 0;
        for (int b = 0; b < bc; ++b) {
            h->cur_b = b;
            const hayashiStatus_t sb =
                issue(h, m, n, k, A + b * sA, lda, sA, B + b * sB, ldb, sB, beta1, C + b * sC, ldc, sC, 1);
            if (sb != HAYASHI_STATUS_SUCCESS) {
                h->cur_b = 0;
                return sb;
            }
            add_counts<<<1, 1, 0, h->stream>>>(h->cnt, h->tot, h->cap, h->dense_now ? h->shc : nullptr);
        }
        h->cur_b = 0;
        HCK(cudaGetLastError());
        return HAYASHI_STATUS_SUCCESS;
    }
    const hayashiStatus_t st = issue_fast(h, m, n, k, A, lda, sA, B, ldb, sB, beta1, C, ldc, sC, bc);
    if (st != HAYASHI_STATUS_SUCCESS)
        return st;
    h->n_mat += bc;
    if (h->dense_now)
        h->n_dense += bc;
    if (h->v5_last || h->f16_last || h->f5_last)
        return HAYASHI_STATUS_SUCCESS;
    exact_fallback<<<h->sms * 4, 256, 0, h->stream>>>(
        FB{A, lda, sA, B, ldb, sB, C, ldc, sC, m, n, k, bc, beta1 ? 1 : 0, h->cnt, h->cap});
    HCK(cudaGetLastError());
    if (h->p103 == 0 && h->shift_ready) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(h->stream, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone) {
            const int b = h->cur_b;
            if (b < hayashiContext::kObsPos) {
                HCK(cudaMemcpyAsync(h->obs + 6 * b, h->cnt, 16, cudaMemcpyDeviceToHost, h->stream));
                if (h->dense_now)
                    HCK(cudaMemcpyAsync(h->obs + 6 * b + 2, h->shc, 32, cudaMemcpyDeviceToHost, h->stream));
                h->obs_dense_pos[b] = h->dense_now ? 1 : 0;
                h->obs_n = std::max(h->obs_n, b + 1);
            }
            HCK(cudaEventRecord(h->obs_ev, h->stream));
            h->obs_pending = true;
            h->obs_dense = h->dense_now;
            h->obs_elems = double(m + n) * k;
            h->obs_lines = double(m + n);
        }
        cudaGetLastError();
    }
    return HAYASHI_STATUS_SUCCESS;
}
static hayashiStatus_t issue_fast(hayashiContext *h, int m, int n, int k, const int8_t *A, int lda, long long sA,
                                  const int8_t *B, int ldb, long long sB, int beta1, int32_t *C, int ldc, long long sC,
                                  int bc) {
    cudaStream_t s = h->stream;
    const int mp = h->mp, np_ = h->np_;
    h->last_direct = 0;
    h->last_tr = 0;
    h->last_scap = 0;
    cudaStreamCaptureStatus capst = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &capst) != cudaSuccess) {
        cudaGetLastError();
        capst = cudaStreamCaptureStatusActive;
    }
    if (h->cur_b == 0 && capst == cudaStreamCaptureStatusNone)
        poll_obs(h);
    const bool auto_dense = h->cur_b < hayashiContext::kObsPos && h->dense_pos[h->cur_b];
    bool dense = h->shift_ready && (h->p103 == 2 || (h->p103 == 0 && auto_dense));
    h->dense_now = false; // backend 0 never shifts; the paths below set it
    if (h->v5_active && h->v5_pend) {
        if (cudaEventQuery(h->v5ev) == cudaSuccess) {
            const double ppm = 1e6 * (double(h->v5obs[2]) + double(h->v5obs[10])) / ((double(m) + n) * k);
            const bool fb = (reinterpret_cast<const unsigned long long *>(h->v5obs + 16))[0] != 0;
            h->v5_route = (fb || ppm > kV5RoutePpm) ? 1 : 0;
            h->v5_pend = false;
        } else
            cudaGetLastError();
    } // cudaErrorNotReady is not an error
    h->v5_last = h->v5_active && h->v5_route != 1;
    h->f16_last = false;
    h->f5_last = false;
    if (h->v5_last) {
        HCK(cudaMemsetAsync(
            h->cnt, 0, 16,
            s)); // the six-face record counters: 0 (issue's exact_fallback and hayashiGetLastRecords read them)
        const bool direct = h->direct && !beta1 && m == v5b::mpad(h->v5) && ldc == m &&
                            (reinterpret_cast<uintptr_t>(C) & 15) == 0; // hayashiSetDirectOutput (default 1)
        h->last_direct = direct ? 1 : 0;
        if (v5b::issue(h->v5, s, A, lda, B, ldb, h->prof != 0, h->ev, direct ? C : nullptr,
                       h->encvar == 12   ? 1
                       : h->encvar == 14 ? 0
                                         : 2,
                       h->reuse_b))
            return HAYASHI_STATUS_CUDA_ERROR;
        const unsigned long long *fl = v5b::flag(
            h->v5); // [0] nonzero: a gate failed -> the add pass leaves C alone and exact_fallback computes it
        if (!direct) {
            PiecePtrs pp{};
            pp.p[0] = reinterpret_cast<const float *>(v5b::out(h->v5));
            const Corr cr{k, nullptr, nullptr, fl, nullptr, nullptr, nullptr, fl + 1, nullptr, 0ull};
            add_tile<false, true><<<h->add_grid0, 256, 0, s>>>(pp, 1, m, n, v5b::mpad(h->v5), C, ldc, beta1 ? 1 : 0, 1,
                                                               cr, 1, 0, (unsigned long long)sC, nullptr, 0ull);
        }
        exact_fallback<<<h->sms * 4, 256, 0, s>>>(
            FB{A, lda, sA, B, ldb, sB, C, ldc, sC, m, n, k, 1, beta1 ? 1 : 0, fl, 0ull});
        if (h->prof)
            HCK(cudaEventRecord(h->ev[4], s));
        if (h->v5_route < 0 && !h->v5_pend && capst == cudaStreamCaptureStatusNone) {
            HCK(cudaMemcpyAsync(h->v5obs, v5b::stat_dev(h->v5), 64, cudaMemcpyDeviceToHost, s));
            HCK(cudaMemcpyAsync(h->v5obs + 16, v5b::flag(h->v5), 8, cudaMemcpyDeviceToHost, s));
            HCK(cudaEventRecord(h->v5ev, s));
            h->v5_pend = true;
        }
        HCK(cudaGetLastError());
        return HAYASHI_STATUS_SUCCESS;
    }
    if (h->f5_active && h->f5_pend) {
        if (cudaEventQuery(h->f5ev) == cudaSuccess) {
            h->f5_route = (h->f5obs[0] & 4u) ? 0 : 1;
            h->f5_pend = false;
        } else
            cudaGetLastError();
    } // cudaErrorNotReady is not an error
    h->f5_last = h->f5_active && h->f5_route != 1;
    if (h->f5_last) { // encode (+ the rows' E sums), gate, the kernel (returns at once when the gate failed), the
                      // output pass, exact_fallback
        const int Mp = h->f5Mpad, Np = h->f5Npad;
        unsigned *E = h->f5E;
        unsigned *gate = E + 2 * (Mp + Np);
        unsigned long long *fl = reinterpret_cast<unsigned long long *>(gate + 4);
        unsigned *st5 = gate + 8;
        const bool skipb =
            h->reuse_b &&
            h->f5_b_ready; // hayashiSetReuseB: B's faces and E sums from the last backend-6 call of this shape
        HCK(cudaMemsetAsync(h->cnt, 0, 16, s)); // the six-face record counters: 0 (hayashiGetLastRecords)
        HCK(cudaMemsetAsync(E, 0, size_t(skipb ? 2 * Mp : 2 * (Mp + Np)) * 4, s));
        const bool direct = h->direct && !beta1 && m == Mp && ldc == Mp && (reinterpret_cast<uintptr_t>(C) & 15) == 0;
        h->last_direct = direct ? 1 : 0;
        int32_t *out = direct ? C : h->f5c;
        const f5e::F5Side fa{A, h->f5pa, m, Mp, f5::Half, lda, k / 16, h->f5cuts, E, E + Mp};
        const f5e::F5Side fb{B,          h->f5pb,        n, Np, f5::BHalf, ldb, k / 16, skipb ? 0 : h->f5cuts,
                             E + 2 * Mp, E + 2 * Mp + Np};
        if (h->prof)
            HCK(cudaEventRecord(h->ev[0], s));
        f5e::encode_f5<4, 2, true><<<h->sms * 8, 256, 0, s>>>(fa, fb, h->f5lut);
        if (h->prof)
            HCK(cudaEventRecord(h->ev[1], s));
        f5e::f5_gate<<<1, 1024, 0, s>>>(E, E + Mp, Mp, E + 2 * Mp, E + 2 * Mp + Np, Np, k, h->f5tm[0], h->f5tm[1], gate,
                                        fl, st5);
        if (h->prof)
            HCK(cudaEventRecord(h->ev[2], s));
        HCK(cudaMemcpyToSymbolAsync(f5::gd, &h->gdh->f5, sizeof(f5::DimsT), 0, cudaMemcpyHostToDevice, s));
        f5::p2_kernel<<<f5::Grid, 384, f5::SharedBytes, s>>>(
            h->f5tmA, h->f5tmB, (const unsigned char *)h->f5pa, (const unsigned char *)h->f5pb,
            reinterpret_cast<float *>(out), h->f5mt, h->f5cuts, h->f5sc, h->f5sc2, gate);
        if (h->prof)
            HCK(cudaEventRecord(h->ev[3], s));
        if (!direct)
            f5_out<<<h->sms * 4, 256, 0, s>>>(C, ldc, out, Mp, m, n, beta1 ? 1 : 0, fl);
        exact_fallback<<<h->sms * 4, 256, 0, s>>>(
            FB{A, lda, sA, B, ldb, sB, C, ldc, sC, m, n, k, 1, beta1 ? 1 : 0, fl, 0ull});
        if (h->prof)
            HCK(cudaEventRecord(h->ev[4], s));
        h->f5_b_ready = true;
        if (h->f5_route < 0 && !h->f5_pend &&
            capst == cudaStreamCaptureStatusNone) { // the gate word read back without a sync, decided at a later call
            HCK(cudaMemcpyAsync(h->f5obs, gate, 4, cudaMemcpyDeviceToHost, s));
            HCK(cudaEventRecord(h->f5ev, s));
            h->f5_pend = true;
        }
        HCK(cudaGetLastError());
        return HAYASHI_STATUS_SUCCESS;
    }
    h->f16_last = false;
    if (h->f16_active) {
        HCK(cudaMemsetAsync(h->cnt, 0, 16,
                            s)); // the six-face record counters: 0 (issue's exact_fallback returns at once)
        try {
            f16way::Way &w = *h->f16;
            int32_t *out = C;
            int ldo = ldc;
            if (beta1) {
                if (!h->f16tmp)
                    HCK(cudaMalloc(&h->f16tmp, size_t(m) * n * 4));
                out = h->f16tmp;
                ldo = m;
            }
            if (h->prof)
                HCK(cudaEventRecord(h->ev[0], s));
            w.part_copy(f16way::CHUNK, A, lda, B, ldb);
            if (h->prof) {
                HCK(cudaEventRecord(h->ev[1], s));
                HCK(cudaEventRecord(h->ev[2], s));
            } // stages: encode = the FP16 copies, order = 0
            if (w.part_gemm(f16way::CHUNK, 0) != CUBLAS_STATUS_SUCCESS)
                return HAYASHI_STATUS_CUBLAS_ERROR;
            if (h->prof)
                HCK(cudaEventRecord(h->ev[3], s));
            w.part_int32(f16way::CHUNK, out, ldo);
            if (beta1)
                add_i32<<<h->sms * 4, 256, 0, s>>>(C, ldc, h->f16tmp, m, n); // C += the product (beta 1)
            if (h->prof)
                HCK(cudaEventRecord(h->ev[4], s));
        } catch (...) {
            cudaGetLastError();
            return HAYASHI_STATUS_CUDA_ERROR;
        }
        h->f16_last = true;
        HCK(cudaGetLastError());
        return HAYASHI_STATUS_SUCCESS;
    }
    if (h->encvar == 12 || h->encvar == 14)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (h->e1_active) {
        HCK(cudaMemsetAsync(h->cnt, 0, h->zbytes, s));
        const bool e2 = false;
        if (e2)
            HCK(cudaMemsetAsync(h->rowE, 0, size_t(h->Mpad + h->Npad2) * 4, s));
        TSide ta{A,      h->pa,    m,      h->Mpad,  d1::CtaRows,           lda, k / 16, h->cuts,
                 h->cnt, h->listA, h->cap, h->lineA, e2 ? h->rowE : nullptr};
        TSide tb{B,       h->pb3,     n,        h->Npad3, e1::BHalf, ldb,    k / 16,
                 h->cuts, h->cnt + 1, h->listB, h->cap,   h->lineB,  nullptr};
        h->dense_now = dense;
        if (dense) {
            HCK(cudaMemsetAsync(h->shc, 0, 32, s));
            ta.lr = h->lrA;
            ta.shc = h->shc;
            tb.lr = h->lrB;
            tb.shc = h->shc + 1;
            ta.lrstep = tb.lrstep = h->encvar == 10 ? 2 : 1;
        }
        if (h->prof)
            HCK(cudaEventRecord(h->ev[0], s));
        launch_encode(dense ? (h->encvar >= 9 ? 11 : 8) : h->encvar, h->sms * 8, s, ta, tb, h->lut, ZeroSpec{});
        if (h->prof)
            HCK(cudaEventRecord(h->ev[1], s));
        scan_lines<<<2, 1024, 0, s>>>(Lines{h->lineA, h->ptrA, m, h->cnt}, Lines{h->lineB, h->ptrB, n, h->cnt + 1});
        if (h->trvar == 0 && !dense) {
            scatter_transpose<<<h->sms * 4, 256, 0, s>>>(Scat{h->listA, h->cnt, h->ptrA, h->lineA, h->csrA},
                                                         Scat{h->listB, h->cnt + 1, h->ptrB, h->lineB, h->csrB}, h->cap,
                                                         Tr{B, ldb, n, h->bt, 0, 1, (unsigned long long)sB},
                                                         Tr{A, lda, m, h->at, 1, 1, (unsigned long long)sA}, k);
        } else {
            scatter_only<<<h->sms * 4, 256, 0, s>>>(Scat{h->listA, h->cnt, h->ptrA, h->lineA, h->csrA},
                                                    Scat{h->listB, h->cnt + 1, h->ptrB, h->lineB, h->csrB}, h->cap);
            Tr2 tB2{B, ldb, n, h->bt, 0}, tA2{A, lda, m, h->at, 1};
            if (dense) {
                tB2.lr = h->lrB;
                tB2.lrld = 2 * (k / 16);
                tB2.lroff = 0;
                tA2.lr = h->lrA;
                tA2.lrld = 2 * (k / 16);
                tA2.lroff = k / 16;
                if (h->encvar == 10) {
                    tB2.lrstep = tA2.lrstep = 2;
                    tA2.lroff = 1;
                }
            }
            if ((h->last_tr = tr128_ok(h->trvar == 0 ? 1 : h->trvar, A, lda, m, B, ldb, n, k, h->at, h->bt) ? 2 : 1) ==
                2)
                transpose128<<<h->sms * 8, 256, 0, s>>>(tB2, tA2, k, h->cnt, h->cnt + 1);
            else
                transpose64<<<h->sms * 8, 256, 0, s>>>(tB2, tA2, k, h->cnt, h->cnt + 1);
        }
        if (h->prof)
            HCK(cudaEventRecord(h->ev[2], s));
        HCK(cudaGetLastError());
        const bool direct = h->direct && !beta1 && m == h->Mpad && ldc == h->Mpad;
        h->last_direct = direct;
        int32_t *out = direct ? C : h->cint;
        HCK(cudaMemcpyToSymbolAsync(e1::gd, &h->gdh->e1, sizeof(e1::DimsT), 0, cudaMemcpyHostToDevice, s));
        e1::p2_kernel<<<e1::Grid, 384, e1::SharedBytes, s>>>(h->tmA, h->tmB3, (const unsigned char *)h->pa,
                                                             (const unsigned char *)h->pb3,
                                                             reinterpret_cast<float *>(out), h->e1mt, h->cuts, h->e1sc);
        HCK(cudaGetLastError());
        if (dense) {
            const hayashiStatus_t st2 = run_thin(h, m, n, out, s);
            if (st2 != HAYASHI_STATUS_SUCCESS)
                return st2;
        }
        if (h->prof)
            HCK(cudaEventRecord(h->ev[3], s));
        PiecePtrs pp{};
        pp.p[0] = reinterpret_cast<const float *>(out);
        Corr cr{k, h->ptrA, h->csrA, h->cnt, h->bt, h->ptrB, h->csrB, h->cnt + 1, h->at, h->cap};
        const float *th = dense ? h->thin : nullptr;
        const unsigned long long scap = (long long)k * std::max(m, n) < (1ll << 31)
                                            ? (unsigned long long)h->few_on * ((unsigned long long)m + n) / 8
                                            : 0;
        h->last_scap = scap;
        if (!direct || dense)
            add_tile<false, true><<<h->add_grid0, 256, 0, s>>>(pp, 1, m, n, h->Mpad, C, ldc, beta1 ? 1 : 0, 1, cr, 1, 0,
                                                               (unsigned long long)sC, th, scap);
        if (scap)
            add_tile<false, true, 1, true><<<h->add_grid2, 256, 0, s>>>(pp, 1, m, n, h->Mpad, C, ldc, beta1 ? 1 : 0, 1,
                                                                        cr, 1, 0, (unsigned long long)sC, th, scap);
        add_tile<true, true><<<h->add_grid1, 256, T2 * SCS * 4, s>>>(pp, 1, m, n, h->Mpad, C, ldc, beta1 ? 1 : 0, 2, cr,
                                                                     1, 0, (unsigned long long)sC, th, scap);
        HCK(cudaGetLastError());
        if (h->prof)
            HCK(cudaEventRecord(h->ev[4], s));
        return HAYASHI_STATUS_SUCCESS;
    }
    if (h->d1_active) { // backend 1: encode into the tiled layout, d1::p2_kernel, then the add pass on its INT32 output
        HCK(cudaMemsetAsync(h->cnt, 0, h->zbytes, s));
        const bool e2 = h->e2_active;
        if (e2)
            HCK(cudaMemsetAsync(h->rowE, 0, size_t(h->Mpad + h->Npad2) * 4, s));
        TSide ta{A,      h->pa,    m,      h->Mpad,  d1::CtaRows,           lda, k / 16, h->cuts,
                 h->cnt, h->listA, h->cap, h->lineA, e2 ? h->rowE : nullptr};
        TSide tb{B,
                 h->pb,
                 n,
                 h->Npad2,
                 d1::HalfN,
                 ldb,
                 k / 16,
                 h->cuts,
                 h->cnt + 1,
                 h->listB,
                 h->cap,
                 h->lineB,
                 e2 ? h->rowE + h->Mpad : nullptr};
        dense = dense && !e2;
        h->dense_now = dense;
        if (dense) {
            HCK(cudaMemsetAsync(h->shc, 0, 32, s));
            ta.lr = h->lrA;
            ta.shc = h->shc;
            tb.lr = h->lrB;
            tb.shc = h->shc + 1;
            ta.lrstep = tb.lrstep = h->encvar == 10 ? 2 : 1;
        }
        const bool direct = h->direct && !beta1 && m == h->Mpad && ldc == h->Mpad;
        h->last_direct = direct;
        int32_t *out = direct ? C : h->cint;
        ZeroSpec z = h->zs;
        z.c = out;
        z.ld = h->Mpad;
        z.n = n;
        if (h->tailsplit == 2 && z.ntiles > 0) {
            zero_tail_k<<<z.ntiles, 256, 0, s>>>(z);
            z.ntiles = 0;
        }
        if (h->prof)
            HCK(cudaEventRecord(h->ev[0], s));
        launch_encode(e2 ? 1 : (dense ? (h->encvar >= 9 ? 11 : 8) : h->encvar), h->sms * 8, s, ta, tb, h->lut, z);
        if (h->prof)
            HCK(cudaEventRecord(h->ev[1], s));
        scan_lines<<<2, 1024, 0, s>>>(Lines{h->lineA, h->ptrA, m, h->cnt}, Lines{h->lineB, h->ptrB, n, h->cnt + 1});
        if (h->trvar == 0 && !dense) {
            scatter_transpose<<<h->sms * 4, 256, 0, s>>>(Scat{h->listA, h->cnt, h->ptrA, h->lineA, h->csrA},
                                                         Scat{h->listB, h->cnt + 1, h->ptrB, h->lineB, h->csrB}, h->cap,
                                                         Tr{B, ldb, n, h->bt, 0, 1, (unsigned long long)sB},
                                                         Tr{A, lda, m, h->at, 1, 1, (unsigned long long)sA}, k);
        } else {
            scatter_only<<<h->sms * 4, 256, 0, s>>>(Scat{h->listA, h->cnt, h->ptrA, h->lineA, h->csrA},
                                                    Scat{h->listB, h->cnt + 1, h->ptrB, h->lineB, h->csrB}, h->cap);
            Tr2 tB2{B, ldb, n, h->bt, 0}, tA2{A, lda, m, h->at, 1};
            if (dense) {
                tB2.lr = h->lrB;
                tB2.lrld = 2 * (k / 16);
                tB2.lroff = 0;
                tA2.lr = h->lrA;
                tA2.lrld = 2 * (k / 16);
                tA2.lroff = k / 16;
                if (h->encvar == 10) {
                    tB2.lrstep = tA2.lrstep = 2;
                    tA2.lroff = 1;
                }
            }
            if ((h->last_tr = tr128_ok(h->trvar == 0 ? 1 : h->trvar, A, lda, m, B, ldb, n, k, h->at, h->bt) ? 2 : 1) ==
                2)
                transpose128<<<h->sms * 8, 256, 0, s>>>(tB2, tA2, k, h->cnt, h->cnt + 1);
            else
                transpose64<<<h->sms * 8, 256, 0, s>>>(tB2, tA2, k, h->cnt, h->cnt + 1);
        }
        if (h->prof)
            HCK(cudaEventRecord(h->ev[2], s));
        HCK(cudaGetLastError());
        HCK(cudaMemcpyToSymbolAsync(d1::gd, &h->gdh->d1, sizeof(d1::DimsT), 0, cudaMemcpyHostToDevice, s));
        cudaLaunchAttribute at{};
        at.id = cudaLaunchAttributeClusterDimension;
        at.val.clusterDim.x = 2;
        at.val.clusterDim.y = 1;
        at.val.clusterDim.z = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(d1::Grid);
        cfg.blockDim = dim3(d1::Threads);
        cfg.dynamicSmemBytes = d1::SharedBytes;
        cfg.stream = s;
        cfg.attrs = &at;
        cfg.numAttrs = 1;
        if (e2) {
            e2_gate<<<1, 1024, 0, s>>>(h->rowE, h->Mpad, h->rowE + h->Mpad, h->Npad2, k, h->use);
            HCK(cudaMemcpyToSymbolAsync(e2::gd, &h->gdh->e2, sizeof(e2::DimsT), 0, cudaMemcpyHostToDevice, s));
            e2::p2_kernel<<<e2::Grid, 384, e2::SharedBytes, s>>>(
                h->tmA, h->tmB, (const unsigned char *)h->pa, (const unsigned char *)h->pb,
                reinterpret_cast<float *>(out), h->e2mt, h->cuts, h->e2sc, h->use);
            HCK(cudaGetLastError());
        }
        HCK(cudaLaunchKernelEx(&cfg, d1::p2_kernel, h->tmA, h->tmB, (const unsigned char *)h->pa,
                               (const unsigned char *)h->pb, reinterpret_cast<float *>(out), h->mt, h->cuts, h->sc,
                               e2 ? (const int *)h->use : (const int *)nullptr));
        if (dense) {
            const hayashiStatus_t st2 = run_thin(h, m, n, out, s);
            if (st2 != HAYASHI_STATUS_SUCCESS)
                return st2;
        }
        if (h->prof)
            HCK(cudaEventRecord(h->ev[3], s));
        PiecePtrs pp{};
        pp.p[0] = reinterpret_cast<const float *>(out);
        Corr cr{k, h->ptrA, h->csrA, h->cnt, h->bt, h->ptrB, h->csrB, h->cnt + 1, h->at, h->cap};
        const float *th = dense ? h->thin : nullptr;
        const unsigned long long scap = (long long)k * std::max(m, n) < (1ll << 31)
                                            ? (unsigned long long)h->few_on * ((unsigned long long)m + n) / 8
                                            : 0;
        h->last_scap = scap;
        if (!direct || dense)
            add_tile<false, true><<<h->add_grid0, 256, 0, s>>>(pp, 1, m, n, h->Mpad, C, ldc, beta1 ? 1 : 0, 1, cr, 1, 0,
                                                               (unsigned long long)sC, th, scap);
        if (scap)
            add_tile<false, true, 1, true><<<h->add_grid2, 256, 0, s>>>(pp, 1, m, n, h->Mpad, C, ldc, beta1 ? 1 : 0, 1,
                                                                        cr, 1, 0, (unsigned long long)sC, th, scap);
        add_tile<true, true><<<h->add_grid1, 256, T2 * SCS * 4, s>>>(pp, 1, m, n, h->Mpad, C, ldc, beta1 ? 1 : 0, 2, cr,
                                                                     1, 0, (unsigned long long)sC, th, scap);
        HCK(cudaGetLastError());
        if (h->prof)
            HCK(cudaEventRecord(h->ev[4], s));
        return HAYASHI_STATUS_SUCCESS;
    }
    // 1. encode A and B of every batch; record +-103 with per-row (A) / per-column (B) counts (line = batch * rows +
    // row)
    HCK(cudaMemsetAsync(h->cnt, 0, h->zbytes, s)); // counters + per-line counts, contiguous
    EncSide ea{A,
               reinterpret_cast<uint4 *>(h->qa),
               (unsigned long long)m * bc * (k / 16),
               unsigned(k / 16),
               lda,
               h->cnt,
               h->listA,
               h->cap,
               h->lineA,
               (unsigned long long)m,
               (unsigned long long)sA,
               (unsigned long long)mp,
               nullptr};
    EncSide eb{B,
               reinterpret_cast<uint4 *>(h->qb),
               (unsigned long long)n * bc * (k / 16),
               unsigned(k / 16),
               ldb,
               h->cnt + 1,
               h->listB,
               h->cap,
               h->lineB,
               (unsigned long long)n,
               (unsigned long long)sB,
               (unsigned long long)np_,
               nullptr};
    if (h->prof)
        HCK(cudaEventRecord(h->ev[0], s));
    encode_ab<0><<<h->enc_grid, 256, 0, s>>>(ea, eb, h->lut);
    if (h->prof)
        HCK(cudaEventRecord(h->ev[1], s));
    HCK(cudaGetLastError());
    // 2. +-103 into row order (A) / column order (B), and the transposes the correction reads (each returns at once
    // without +-103)
    scan_lines<<<2, 1024, 0, s>>>(Lines{h->lineA, h->ptrA, m * bc, h->cnt},
                                  Lines{h->lineB, h->ptrB, n * bc, h->cnt + 1});
    if (h->trvar == 0 || bc > 1) { // transpose64 has no batch form: batched calls keep the byte tiles
        scatter_transpose<<<h->sms * 4, 256, 0, s>>>(Scat{h->listA, h->cnt, h->ptrA, h->lineA, h->csrA},
                                                     Scat{h->listB, h->cnt + 1, h->ptrB, h->lineB, h->csrB}, h->cap,
                                                     Tr{B, ldb, n, h->bt, 0, bc, (unsigned long long)sB},
                                                     Tr{A, lda, m, h->at, 1, bc, (unsigned long long)sA}, k);
    } else {
        scatter_only<<<h->sms * 4, 256, 0, s>>>(Scat{h->listA, h->cnt, h->ptrA, h->lineA, h->csrA},
                                                Scat{h->listB, h->cnt + 1, h->ptrB, h->lineB, h->csrB}, h->cap);
        if ((h->last_tr = tr128_ok(h->trvar, A, lda, m, B, ldb, n, k, h->at, h->bt) ? 2 : 1) == 2)
            transpose128<<<h->sms * 8, 256, 0, s>>>(Tr2{B, ldb, n, h->bt, 0}, Tr2{A, lda, m, h->at, 1}, k, h->cnt,
                                                    h->cnt + 1);
        else
            transpose64<<<h->sms * 8, 256, 0, s>>>(Tr2{B, ldb, n, h->bt, 0}, Tr2{A, lda, m, h->at, 1}, k, h->cnt,
                                                   h->cnt + 1);
    }
    if (h->prof)
        HCK(cudaEventRecord(h->ev[2], s));
    HCK(cudaGetLastError());
    // 3. the pieces (every batch in one cuBLASLt call, or one call per batch), added into C in passes of `group`
    bool prof3 = false;
    const float one = 1.f, zero = 0.f;
    const size_t fa_b = size_t(mp) * k * 3, fb_b = size_t(np_) * k * 3,
                 pst = size_t(mp) * np_; // per-batch strides of faces (bytes) and pieces
    for (int g0 = 0; g0 < h->npieces; g0 += h->group) {
        const int np = std::min(h->group, h->npieces - g0);
        PiecePtrs pp{};
        for (int j = 0; j < np; ++j) {
            const int c = g0 + j;
            const PieceShape &ps = (c == h->npieces - 1 && h->ps[1].ok) ? h->ps[1] : h->ps[0];
            const uint8_t *ap = h->qa + size_t(c) * KC * 3;
            const uint8_t *bp = h->qb + size_t(c) * KC * 3; // 6*KC FP4 = 3*KC bytes
            if (!h->loopb)
                HBL(cublasLtMatmul(h->lt, ps.op, &one, ap, ps.la, bp, ps.lb, &zero, h->piece[j], h->lc, h->piece[j],
                                   h->lc, &ps.algo, h->ltws, h->ltws_bytes, s));
            else
                for (int b = 0; b < bc; ++b)
                    HBL(cublasLtMatmul(h->lt, ps.op, &one, ap + b * fa_b, ps.la, bp + b * fb_b, ps.lb, &zero,
                                       h->piece[j] + b * pst, h->lc, h->piece[j] + b * pst, h->lc, &ps.algo, h->ltws,
                                       h->ltws_bytes, s));
            pp.p[j] = h->piece[j];
        }
        Corr cr{k, h->ptrA, h->csrA, h->cnt, h->bt, h->ptrB, h->csrB, h->cnt + 1, h->at, h->cap};
        const int addc = (g0 > 0 || beta1) ? 1 : 0;
        if (h->prof && !prof3) {
            HCK(cudaEventRecord(h->ev[3], s));
            prof3 = true;
        }
        if (g0 ==
            0) { // the correction rides on the first pass: exactly one of the two runs (the other returns at once)
            add_tile<false><<<h->add_grid0, 256, 0, s>>>(pp, np, m, n, h->ldp, C, ldc, addc, 1, cr, bc, pst,
                                                         (unsigned long long)sC);
            add_tile<true><<<h->add_grid1, 256, T2 * SCS * 4, s>>>(pp, np, m, n, h->ldp, C, ldc, addc, 2, cr, bc, pst,
                                                                   (unsigned long long)sC);
        } else
            add_tile<false><<<h->add_grid0, 256, 0, s>>>(pp, np, m, n, h->ldp, C, ldc, addc, 0, cr, bc, pst,
                                                         (unsigned long long)sC);
        HCK(cudaGetLastError());
    }
    if (h->prof)
        HCK(cudaEventRecord(h->ev[4], s));
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiGemmI8TNStridedBatched(hayashiHandle_t h, int m, int n, int k, const int32_t *alpha,
                                                         const int8_t *A, int lda, long long strideA, const int8_t *B,
                                                         int ldb, long long strideB, const int32_t *beta, int32_t *C,
                                                         int ldc, long long strideC, int batchCount) {
    if (!h || !alpha || !beta || !A || !B || !C || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < k || ldc < m ||
        batchCount <= 0)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (batchCount > 1 && strideB == 0 && strideA == (long long)m * lda && strideC == m &&
        (long long)ldc >= (long long)m * batchCount) // shared B, stacked A and C
        return hayashiGemmI8TNStridedBatched(h, m * batchCount, n, k, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
    if (batchCount > 1 && (strideA < (long long)m * lda || (strideB != 0 && strideB < (long long)n * ldb) ||
                           strideC < (long long)n * ldc))
        return HAYASHI_STATUS_INVALID_VALUE;
    if (*alpha != 1 || (*beta != 0 && *beta != 1) || k % 16 || lda % 16 || ldb % 16 ||
        (batchCount > 1 && (strideA % 16 || strideB % 16)) || (reinterpret_cast<uintptr_t>(A) & 15) ||
        (reinterpret_cast<uintptr_t>(B) & 15))
        return HAYASHI_STATUS_NOT_SUPPORTED;
    if (batchCount == 1) {
        strideA = (long long)m * lda;
        strideB = (long long)n * ldb;
        strideC = (long long)n * ldc;
    }
    h->loop1 = batchCount > 1 && h->is_b300 && k <= 16384 && double(m) * n * k >= 34359738368.0 &&
               ((h->backend == 2 && double(m) * n >= 1048576.0) || h->backend == 1 || (h->backend == 3 && k <= 1024));
    hayashiStatus_t st = set_shape(h, m, n, k, h->loop1 ? 1 : batchCount);
    if (st != HAYASHI_STATUS_SUCCESS)
        return st;
    if (!h->graph_on || h->stream == 0)
        return issue(h, m, n, k, A, lda, strideA, B, ldb, strideB, *beta, C, ldc, strideC, batchCount);
    hayashiContext::Key key;
    memset(&key, 0, sizeof(key)); // padding zeroed: the key is compared with memcmp
    key.A = A;
    key.B = B;
    key.C = C;
    key.sA = strideA;
    key.sB = strideB;
    key.sC = strideC;
    key.m = m;
    key.n = n;
    key.k = k;
    key.lda = lda;
    key.ldb = ldb;
    key.ldc = ldc;
    key.beta = *beta;
    key.bc = batchCount;
    key.s = h->stream;
    if (!h->gexec || memcmp(&key, &h->key, sizeof(key)) != 0) { // capture this call once, then replay it
        drop_graph(h);
        const bool first_uncaptured = h->p103 == 0;
        struct LastCall {
            bool dense_now, obs_pending, obs_dense;
            int last_direct, last_tr, cur_b, obs_n;
            unsigned long long last_scap;
            long long n_mat, n_dense;
            double obs_elems, obs_lines;
            unsigned char obs_dense_pos[hayashiContext::kObsPos];
        } lc{};
        if (first_uncaptured) {
            st = issue(h, m, n, k, A, lda, strideA, B, ldb, strideB, *beta, C, ldc, strideC, batchCount);
            if (st != HAYASHI_STATUS_SUCCESS)
                return st;
            HCK(cudaStreamSynchronize(h->stream));
            poll_obs(h);
            lc.dense_now = h->dense_now;
            lc.obs_pending = h->obs_pending;
            lc.obs_dense = h->obs_dense;
            lc.last_direct = h->last_direct;
            lc.last_tr = h->last_tr;
            lc.cur_b = h->cur_b;
            lc.obs_n = h->obs_n;
            lc.last_scap = h->last_scap;
            lc.n_mat = h->n_mat;
            lc.n_dense = h->n_dense;
            lc.obs_elems = h->obs_elems;
            lc.obs_lines = h->obs_lines;
            memcpy(lc.obs_dense_pos, h->obs_dense_pos, sizeof lc.obs_dense_pos);
        }
        HCK(cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeThreadLocal));
        st = issue(h, m, n, k, A, lda, strideA, B, ldb, strideB, *beta, C, ldc, strideC, batchCount);
        cudaGraph_t g = nullptr;
        const cudaError_t ce = cudaStreamEndCapture(h->stream, &g);
        if (st != HAYASHI_STATUS_SUCCESS || ce != cudaSuccess) {
            if (g)
                cudaGraphDestroy(g);
            return st != HAYASHI_STATUS_SUCCESS ? st : HAYASHI_STATUS_CUDA_ERROR;
        }
        const cudaError_t ie = cudaGraphInstantiate(&h->gexec, g, 0);
        cudaGraphDestroy(g);
        if (ie != cudaSuccess) {
            h->gexec = nullptr;
            return HAYASHI_STATUS_CUDA_ERROR;
        }
        h->key = key;
        h->glast.dense_now = h->dense_now;
        h->glast.last_direct = h->last_direct;
        h->glast.last_tr = h->last_tr;
        h->glast.last_scap = h->last_scap;
        if (first_uncaptured) {
            h->dense_now = lc.dense_now;
            h->obs_pending = lc.obs_pending;
            h->obs_dense = lc.obs_dense;
            h->last_direct = lc.last_direct;
            h->last_tr = lc.last_tr;
            h->cur_b = lc.cur_b;
            h->obs_n = lc.obs_n;
            h->last_scap = lc.last_scap;
            h->n_mat = lc.n_mat;
            h->n_dense = lc.n_dense;
            h->obs_elems = lc.obs_elems;
            h->obs_lines = lc.obs_lines;
            memcpy(h->obs_dense_pos, lc.obs_dense_pos, sizeof lc.obs_dense_pos);
            return HAYASHI_STATUS_SUCCESS;
        }
    } else {
        h->dense_now = h->glast.dense_now;
        h->last_direct = h->glast.last_direct;
        h->last_tr = h->glast.last_tr;
        h->last_scap = h->glast.last_scap;
    }
    HCK(cudaGraphLaunch(h->gexec, h->stream));
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiGemmI8TN(hayashiHandle_t h, int m, int n, int k, const int32_t *alpha,
                                           const int8_t *A, int lda, const int8_t *B, int ldb, const int32_t *beta,
                                           int32_t *C, int ldc) {
    return hayashiGemmI8TNStridedBatched(h, m, n, k, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}

extern "C" hayashiStatus_t hayashiGemmBF16QuantTN(hayashiHandle_t h, int m, int n, int k, const void *A, int lda,
                                                  const float *invScaleA, const void *B, int ldb,
                                                  const float *invScaleB, int32_t *C, int ldc) {
    if (!h || !A || !B || !C || !invScaleA || !invScaleB || m <= 0 || n <= 0 || k <= 0 || lda < k || ldb < k || ldc < m)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (k % 16 || lda % 16 || ldb % 16 || (reinterpret_cast<uintptr_t>(A) & 15) ||
        (reinterpret_cast<uintptr_t>(B) & 15))
        return HAYASHI_STATUS_NOT_SUPPORTED;
    hayashiStatus_t st = set_shape(h, m, n, k, 1);
    if (st != HAYASHI_STATUS_SUCCESS)
        return st;
    cudaStream_t s = h->stream;
    EncSide ea{static_cast<const int8_t *>(A),
               reinterpret_cast<uint4 *>(h->qa),
               (unsigned long long)m * (k / 16),
               unsigned(k / 16),
               lda,
               h->cnt,
               h->listA,
               h->cap,
               h->lineA,
               (unsigned long long)m,
               (unsigned long long)m * lda,
               (unsigned long long)h->mp,
               invScaleA};
    EncSide eb{static_cast<const int8_t *>(B),
               reinterpret_cast<uint4 *>(h->qb),
               (unsigned long long)n * (k / 16),
               unsigned(k / 16),
               ldb,
               h->cnt + 1,
               h->listB,
               h->cap,
               h->lineB,
               (unsigned long long)n,
               (unsigned long long)n * ldb,
               (unsigned long long)h->np_,
               invScaleB};
    encode_ab<1><<<h->enc_grid_q, 256, 0, s>>>(ea, eb, h->lut); // values in -102..102: no +-103, no counters
    HCK(cudaGetLastError());
    const float one = 1.f, zero = 0.f;
    for (int g0 = 0; g0 < h->npieces; g0 += h->group) {
        const int np = std::min(h->group, h->npieces - g0);
        PiecePtrs pp{};
        for (int j = 0; j < np; ++j) {
            const int c = g0 + j;
            const PieceShape &ps = (c == h->npieces - 1 && h->ps[1].ok) ? h->ps[1] : h->ps[0];
            HBL(cublasLtMatmul(h->lt, ps.op, &one, h->qa + size_t(c) * KC * 3, ps.la, h->qb + size_t(c) * KC * 3, ps.lb,
                               &zero, h->piece[j], h->lc, h->piece[j], h->lc, &ps.algo, h->ltws, h->ltws_bytes, s));
            pp.p[j] = h->piece[j];
        }
        Corr cr{k, h->ptrA, h->csrA, h->cnt, h->bt, h->ptrB, h->csrB, h->cnt + 1, h->at, h->cap};
        add_tile<false><<<h->add_grid0, 256, 0, s>>>(pp, np, m, n, h->ldp, C, ldc, g0 > 0 ? 1 : 0, 0, cr, 1,
                                                     size_t(h->mp) * h->np_, 0);
        HCK(cudaGetLastError());
    }
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiSetE2(hayashiHandle_t h, int on) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    if ((on ? 1 : 0) != h->e2on) {
        cudaStreamSynchronize(h->stream);
        free_shape(h);
        h->e2on = on ? 1 : 0;
    }
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiGetLastKernel(hayashiHandle_t h, int *e2used) {
    if (!h || !e2used)
        return HAYASHI_STATUS_INVALID_VALUE;
    *e2used = 0;
    if (!h->d1_active || !h->e2_active)
        return HAYASHI_STATUS_SUCCESS;
    HCK(cudaStreamSynchronize(h->stream));
    int u = 0;
    HCK(cudaMemcpy(&u, h->use, 4, cudaMemcpyDeviceToHost));
    *e2used = u;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetBackend(hayashiHandle_t h, int backend) {
    if (!h || backend < 0 || backend > 6)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    free_shape(h);
    h->backend = backend;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" int hayashiLastBackend(hayashiHandle_t h) {
    return h ? (h->v5_last ? 4 : h->f5_last ? 6 : h->f16_last ? 5 : h->e1_active ? 3 : h->d1_active ? 1 : 0) : -1;
}
extern "C" hayashiStatus_t hayashiSetTailSplit(hayashiHandle_t h, int on) {
    if (!h || on < 0 || on > 2)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    h->tailsplit = on;
    free_shape(h);
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetPlus103Mode(hayashiHandle_t h, int mode) {
    if (!h || mode < 0 || mode > 2)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    h->p103 = mode;
    h->dense_auto = false;
    h->obs_pending = false;
    h->obs_n = 0;
    memset(h->dense_pos, 0, sizeof h->dense_pos);
    free_shape(h);
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" int hayashiLastPlus103Dense(hayashiHandle_t h) { return h ? int(h->dense_now) : -1; }
extern "C" hayashiStatus_t hayashiGetLastRecords(hayashiHandle_t h, long long *out4) {
    if (!h || !out4)
        return HAYASHI_STATUS_INVALID_VALUE;
    unsigned long long c2[2] = {0, 0};
    if (cudaStreamSynchronize(h->stream) != cudaSuccess ||
        cudaMemcpy(c2, h->cnt, 16, cudaMemcpyDeviceToHost) != cudaSuccess)
        return HAYASHI_STATUS_CUDA_ERROR;
    out4[0] = (long long)c2[0];
    out4[1] = (long long)c2[1];
    out4[2] = (long long)h->last_scap;
    out4[3] = (h->last_scap && c2[0] <= h->cap && c2[1] <= h->cap && c2[0] + c2[1] > 0 && c2[0] + c2[1] <= h->last_scap)
                  ? 1
                  : 0;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiGetLastV5(hayashiHandle_t h, long long *out10) {
    if (!h || !out10)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (!h->v5_active) {
        for (int i = 0; i < 10; ++i)
            out10[i] = -1;
        return HAYASHI_STATUS_SUCCESS;
    }
    return v5b::last_stats(h->v5, out10) ? HAYASHI_STATUS_CUDA_ERROR : HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiGetLastF5(hayashiHandle_t h, long long *out8) {
    // max over A rows of sum E0, of sum E1, max over B columns of sum E0, of sum E1, the gate word (bit 2 ran, bits 0 /
    // 1 fast conversion), tmax0, tmax1, the route (-1 not known yet, 0 (1,7,28), 1 FP16 way); all -1 when backend 6 is
    // not set up
    if (!h || !out8)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (!h->f5_active) {
        for (int i = 0; i < 8; ++i)
            out8[i] = -1;
        return HAYASHI_STATUS_SUCCESS;
    }
    unsigned st[5] = {0, 0, 0, 0, 0};
    if (cudaStreamSynchronize(h->stream) != cudaSuccess ||
        cudaMemcpy(st, h->f5E + 2 * (h->f5Mpad + h->f5Npad) + 8, 20, cudaMemcpyDeviceToHost) != cudaSuccess)
        return HAYASHI_STATUS_CUDA_ERROR;
    for (int i = 0; i < 5; ++i)
        out8[i] = (long long)st[i];
    out8[5] = h->f5tm[0];
    out8[6] = h->f5tm[1];
    out8[7] = h->f5_route;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetFewCorr(hayashiHandle_t h, int on) {
    if (!h || on < 0 || on > 64)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    h->few_on = on;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiGetDenseCounts(hayashiHandle_t h, long long *out2) {
    if (!h || !out2)
        return HAYASHI_STATUS_INVALID_VALUE;
    out2[0] = h->n_dense;
    out2[1] = h->n_mat;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" int hayashiLastTailSplit(hayashiHandle_t h) { return h ? (h->d1_active ? h->dims.TailS : 1) : -1; }
extern "C" hayashiStatus_t hayashiGetLastSettings(hayashiHandle_t h, int *s6) {
    if (!h || !s6)
        return HAYASHI_STATUS_INVALID_VALUE;
    const int be = h->v5_last ? 4 : h->f5_last ? 6 : h->f16_last ? 5 : h->e1_active ? 3 : h->d1_active ? 1 : 0;
    s6[0] = be;
    s6[1] = be ? h->encvar : -1;
    s6[2] = h->last_direct;
    s6[3] = h->last_tr;
    s6[4] = h->graph_on;
    s6[5] = h->e2_active ? 1 : 0;
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" int hayashiLastBatchedByLoop(hayashiHandle_t h) { return h ? (h->loop1 ? 2 : int(h->loopb)) : -1; }
extern "C" hayashiStatus_t hayashiGemmI8BatchedHost(hayashiHandle_t h, int transa, int transb, int m, int n, int k,
                                                    const int32_t *alpha, const int8_t *const *A, int lda,
                                                    const int8_t *const *B, int ldb, const int32_t *beta,
                                                    int32_t *const *C, int ldc, int batchCount) {
    if (!h || batchCount < 0 || (batchCount > 0 && (!A || !B || !C)))
        return HAYASHI_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        h->cur_b = b;
        const hayashiStatus_t st =
            hayashiGemmI8(h, transa, transb, m, n, k, alpha, A[b], lda, 0, B[b], ldb, 0, beta, C[b], ldc, 0, 1);
        if (st != HAYASHI_STATUS_SUCCESS) {
            h->cur_b = 0;
            return st;
        }
    }
    h->cur_b = 0;
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiSetGraphMode(hayashiHandle_t h, int on) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    drop_graph(h);
    h->graph_on = on;
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiSetProfiling(hayashiHandle_t h, int on) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (on && !h->ev[0])
        for (auto &e : h->ev)
            HCK(cudaEventCreate(&e));
    h->prof = on;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiGetLastStageTimes(hayashiHandle_t h, float *ms4) {
    if (!h || !ms4 || !h->prof)
        return HAYASHI_STATUS_INVALID_VALUE;
    HCK(cudaStreamSynchronize(h->stream));
    for (int i = 0; i < 4; ++i)
        HCK(cudaEventElapsedTime(ms4 + i, h->ev[i], h->ev[i + 1]));
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetTable(hayashiHandle_t h, int which) {
    if (!h || which < 0 || which > 2)
        return HAYASHI_STATUS_INVALID_VALUE;
    cudaStreamSynchronize(h->stream);
    return load_table(h, which);
}
extern "C" hayashiStatus_t hayashiSetEncodeVariant(hayashiHandle_t h, int v) {
    if (!h || v < 0 || (v > 10 && v != 12 && v != 14) || v == 8)
        return HAYASHI_STATUS_INVALID_VALUE;
    h->encvar = v;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetReuseB(hayashiHandle_t h, int on) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    if ((on != 0) != h->reuse_b) {
        cudaStreamSynchronize(h->stream);
        free_shape(h);
        h->reuse_b = on != 0;
    }
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetDirectOutput(hayashiHandle_t h, int on) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    h->direct = on ? 1 : 0;
    return HAYASHI_STATUS_SUCCESS;
}
extern "C" hayashiStatus_t hayashiSetTransposeVariant(hayashiHandle_t h, int v) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    h->trvar = v;
    return HAYASHI_STATUS_SUCCESS;
}

extern "C" hayashiStatus_t hayashiGetLastInfo(hayashiHandle_t h, long long *ca, long long *cb, int *overflow) {
    if (!h)
        return HAYASHI_STATUS_INVALID_VALUE;
    HCK(cudaStreamSynchronize(h->stream));
    if (h->f16_last) {
        if (ca)
            *ca = -1;
        if (cb)
            *cb = -1;
        if (overflow)
            *overflow = 0;
        return HAYASHI_STATUS_SUCCESS;
    }
    if (h->f5_last) {
        unsigned long long f[2] = {0, 0};
        HCK(cudaMemcpy(f, h->f5E + 2 * (h->f5Mpad + h->f5Npad) + 4, 16, cudaMemcpyDeviceToHost));
        if (ca)
            *ca = -1;
        if (cb)
            *cb = -1;
        if (overflow)
            *overflow = f[0] ? 1 : 0;
        return HAYASHI_STATUS_SUCCESS;
    }
    if (h->v5_last) {
        long long o[10];
        if (v5b::last_stats(h->v5, o))
            return HAYASHI_STATUS_CUDA_ERROR;
        if (ca)
            *ca = o[8];
        if (cb)
            *cb = o[9];
        if (overflow)
            *overflow = o[7] ? 1 : 0;
        return HAYASHI_STATUS_SUCCESS;
    }
    if (h->loop1 && h->tot) {
        unsigned long long t[4] = {0, 0, 0, 0};
        HCK(cudaMemcpy(t, h->tot, 32, cudaMemcpyDeviceToHost));
        if (ca)
            *ca = (long long)t[0];
        if (cb)
            *cb = (long long)t[1];
        if (overflow)
            *overflow = (t[2] || t[3]) ? 1 : 0;
        return HAYASHI_STATUS_SUCCESS;
    }
    unsigned long long c[2] = {0, 0};
    if (h->cnt)
        HCK(cudaMemcpy(c, h->cnt, 16, cudaMemcpyDeviceToHost));
    if (h->dense_now && h->shc) {
        unsigned long long g[4] = {0, 0, 0, 0};
        HCK(cudaMemcpy(g, h->shc, 32, cudaMemcpyDeviceToHost));
        if (ca)
            *ca = (long long)g[2];
        if (cb)
            *cb = (long long)g[3];
        if (overflow)
            *overflow = (c[0] > h->cap || c[1] > h->cap) ? 1 : 0;
        return HAYASHI_STATUS_SUCCESS;
    }
    if (ca)
        *ca = (long long)c[0];
    if (cb)
        *cb = (long long)c[1];
    if (overflow)
        *overflow = (c[0] > h->cap || c[1] > h->cap) ? 1 : 0;
    return HAYASHI_STATUS_SUCCESS;
}

__device__ __forceinline__ int32_t mulw_(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}
__device__ __forceinline__ int32_t addw_(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}
__global__ void scale_c(int32_t *C, long long ldc, long long strideC, int m, int n, int batch,
                        int32_t sv) { // C = s C (s = 0 writes 0 without reading C)
    const long long mn = (long long)m * n;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < mn * batch;
         g += (long long)gridDim.x * blockDim.x) {
        const long long b = g / mn, r = g % mn;
        int32_t *p = C + b * strideC + (r / m) * ldc + r % m;
        *p = sv == 0 ? 0 : mulw_(sv, *p);
    }
}
__global__ void axpby_c(int32_t *C, long long ldc, long long strideC, const int32_t *T, int m, int n, int batch,
                        int32_t al, int32_t be) { // C = al T + be C
    const long long mn = (long long)m * n;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < mn * batch;
         g += (long long)gridDim.x * blockDim.x) {
        const long long b = g / mn, r = g % mn;
        int32_t *p = C + b * strideC + (r / m) * ldc + r % m;
        *p = addw_(mulw_(al, T[g]), mulw_(be, *p));
    }
}
extern "C" hayashiStatus_t hayashiGemmI8(hayashiHandle_t h, int transa, int transb, int m, int n, int k,
                                         const int32_t *alpha, const int8_t *A, int lda, long long strideA,
                                         const int8_t *B, int ldb, long long strideB, const int32_t *beta, int32_t *C,
                                         int ldc, long long strideC, int batchCount) {
    if (!h || !alpha || !beta || m < 0 || n < 0 || k < 0 || batchCount < 0)
        return HAYASHI_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0 || batchCount == 0)
        return HAYASHI_STATUS_SUCCESS;
    if (!C || ldc < m)
        return HAYASHI_STATUS_INVALID_VALUE;
    if ((transa != HAYASHI_OP_N && transa != HAYASHI_OP_T) || (transb != HAYASHI_OP_N && transb != HAYASHI_OP_T))
        return HAYASHI_STATUS_INVALID_VALUE;
    const int32_t av = *alpha, bv = *beta;
    auto blocks = [&](long long cnt) { return int(std::min<long long>((cnt + 255) / 256, (long long)h->sms * 16)); };
    const long long mnb = (long long)m * n * batchCount;
    if (k == 0 || av == 0) { // no product: C = beta C
        if (bv != 1) {
            scale_c<<<blocks(mnb), 256, 0, h->stream>>>(C, ldc, strideC, m, n, batchCount, bv);
            HCK(cudaGetLastError());
        }
        return HAYASHI_STATUS_SUCCESS;
    }
    if (!A || !B)
        return HAYASHI_STATUS_INVALID_VALUE;
    const bool akm = transa == HAYASHI_OP_T, bkm = transb == HAYASHI_OP_N; // K contiguous in the caller's layout
    if (lda < (akm ? k : m) || ldb < (bkm ? k : n))
        return HAYASHI_STATUS_INVALID_VALUE;
    if (batchCount > 65535) { // the pre-pass puts the batch in grid z
        for (int b0 = 0; b0 < batchCount; b0 += 65535) {
            const int bc = std::min(65535, batchCount - b0);
            const hayashiStatus_t st = hayashiGemmI8(h, transa, transb, m, n, k, alpha, A + (long long)b0 * strideA,
                                                     lda, strideA, B + (long long)b0 * strideB, ldb, strideB, beta,
                                                     C + (long long)b0 * strideC, ldc, strideC, bc);
            if (st != HAYASHI_STATUS_SUCCESS)
                return st;
        }
        return HAYASHI_STATUS_SUCCESS;
    }
    auto al16 = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    const bool kok = k % 16 == 0, many = batchCount > 1;
    const bool natA = akm && kok && lda % 16 == 0 && al16(A) && (!many || strideA % 16 == 0);
    const bool natB = bkm && kok && ldb % 16 == 0 && al16(B) && (!many || strideB % 16 == 0);
    const bool fast_ab = av == 1 && (bv == 0 || bv == 1), tbuf = !fast_ab && av != 1 && bv != 0;
    const int kp = api_layout::kpad(k);
    const bool cpA = !natA, cpB = !natB;
    const int bA = (many && strideA == 0) ? 1 : batchCount, bB = (many && strideB == 0) ? 1 : batchCount;
    auto up = [](size_t x) { return (x + 255) / 256 * 256; };
    const size_t na = cpA ? up(size_t(bA) * m * kp) : 0, nb = cpB ? up(size_t(bB) * n * kp) : 0,
                 nt = tbuf ? up(size_t(mnb) * 4) : 0;
    if (na + nb + nt > h->lw_bytes) {
        HCK(cudaStreamSynchronize(h->stream));
        cudaFree(h->lw);
        h->lw = nullptr;
        h->lw_bytes = 0;
        if (cudaMalloc(&h->lw, na + nb + nt) != cudaSuccess)
            return HAYASHI_STATUS_ALLOC_FAILED;
        h->lw_bytes = na + nb + nt;
    }
    int8_t *Aw = h->lw;
    int8_t *Bw = h->lw + na;
    int32_t *T = reinterpret_cast<int32_t *>(h->lw + na + nb);
    if (cpA && api_layout::to_kmajor(A, lda, strideA, bA, m, k, akm, Aw, h->sms, h->stream) != cudaSuccess)
        return HAYASHI_STATUS_CUDA_ERROR;
    if (cpB && api_layout::to_kmajor(B, ldb, strideB, bB, n, k, bkm, Bw, h->sms, h->stream) != cudaSuccess)
        return HAYASHI_STATUS_CUDA_ERROR;
    const int8_t *pA = cpA ? Aw : A;
    const int8_t *pB = cpB ? Bw : B;
    const int ldA = cpA ? kp : lda, ldB = cpB ? kp : ldb, kk = (cpA || cpB) ? kp : k;
    const long long sA2 = cpA ? (bA == 1 && many ? 0 : (long long)m * kp) : strideA,
                    sB2 = cpB ? (bB == 1 && many ? 0 : (long long)n * kp) : strideB;
    static const int32_t one = 1, zero = 0;
    auto tn = [&](const int32_t *be, int32_t *Cx, int ldx, long long sx) {
        return hayashiGemmI8TNStridedBatched(h, m, n, kk, &one, pA, ldA, sA2, pB, ldB, sB2, be, Cx, ldx, sx,
                                             batchCount);
    };
    if (fast_ab)
        return tn(beta, C, ldc, strideC);
    if (av == 1) { // C = beta C, then C += A B
        scale_c<<<blocks(mnb), 256, 0, h->stream>>>(C, ldc, strideC, m, n, batchCount, bv);
        HCK(cudaGetLastError());
        return tn(&one, C, ldc, strideC);
    }
    if (bv == 0) { // C = A B, then C = alpha C
        const hayashiStatus_t st = tn(&zero, C, ldc, strideC);
        if (st != HAYASHI_STATUS_SUCCESS)
            return st;
        scale_c<<<blocks(mnb), 256, 0, h->stream>>>(C, ldc, strideC, m, n, batchCount, av);
        HCK(cudaGetLastError());
        return HAYASHI_STATUS_SUCCESS;
    }
    {
        const hayashiStatus_t st = tn(&zero, T, m, (long long)m * n);
        if (st != HAYASHI_STATUS_SUCCESS)
            return st;
    } // T = A B
    axpby_c<<<blocks(mnb), 256, 0, h->stream>>>(C, ldc, strideC, T, m, n, batchCount, av, bv);
    HCK(cudaGetLastError());
    return HAYASHI_STATUS_SUCCESS;
}

// Host-only public adapter shares this translation unit to keep the cubin inventory unchanged.
#include "awe_adapter.h"
