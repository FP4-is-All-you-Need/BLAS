#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "lut5_1_10_13.h"
#include "ext5_digits.h"
namespace v5 {
constexpr int CtaRows = 128, HalfN = 112, TileN = 224, CutsPerReq = 5, ACut = CtaRows * 48, BCut = HalfN * 48,
              Window = 1536;
__host__ __device__ constexpr int tile_byte(int row, int w) {
    return (row / 8) * 384 + (w / 16) * 128 + (row % 8) * 16 + w % 16;
}
__constant__ signed char
    c_face[2][256][5]; // A: the faces with lambda's sign (faces 2 and 4 flipped), B: the faces (d1p_1526.cu 1211-1213)
__constant__ signed char c_digits[EXT5_HI - EXT5_LO + 1][6]; // EXT5_DIGITS
__constant__ long long c_e5[256];      // E(v) = sum_f |lambda_f| face_f(v)^2, -1 for a hole (d4_e5_table)
__constant__ unsigned char c_cov[256]; // LUT5_1_10_13_COVERED
__constant__ signed char c_near[256];  // the nearest covered value (ties: the smaller), for the leftovers
__constant__ int c_s15[15];

__device__ __forceinline__ unsigned code4(int c) { // the FP4 code of c / 2, c in S15 (= e2m1_half)
    const int a = c < 0 ? -c : c;
    const unsigned m = a <= 4 ? unsigned(a) : (a == 6 ? 5u : (a == 8 ? 6u : 7u));
    return c == 0 ? 0u : (m | (c < 0 ? 8u : 0u));
}

// host: the constant tables (once per process / device)
inline cudaError_t init_tables() {
    signed char face[2][256][5];
    long long e5[256];
    unsigned char cov[256];
    signed char nearv[256];
    const int lam_abs[5] = {120, 120, 24, 40, 15};
    const int S15[15] = {0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12};
    for (int x = 0; x < 256; ++x) {
        for (int f = 0; f < 5; ++f) {
            face[0][x][f] = (signed char)(LUT5_1_10_13_LAMBDA[f] < 0 ? -LUT5_1_10_13[x][f] : LUT5_1_10_13[x][f]);
            face[1][x][f] = LUT5_1_10_13[x][f];
        }
        cov[x] = LUT5_1_10_13_COVERED[x];
        if (!cov[x])
            e5[x] = -1;
        else {
            long long s = 0;
            for (int f = 0; f < 5; ++f)
                s += (long long)lam_abs[f] * LUT5_1_10_13[x][f] * LUT5_1_10_13[x][f];
            e5[x] = s;
        }
        int best = 0, bd = 1 << 30;
        for (int v = -128; v < 128; ++v)
            if (LUT5_1_10_13_COVERED[v + 128]) {
                const int dd = v > x - 128 ? v - (x - 128) : (x - 128) - v;
                if (dd < bd) {
                    bd = dd;
                    best = v;
                }
            }
        nearv[x] = (signed char)best;
    }
    cudaError_t e;
    if ((e = cudaMemcpyToSymbol(c_face, face, sizeof face)) != cudaSuccess)
        return e;
    if ((e = cudaMemcpyToSymbol(c_digits, EXT5_DIGITS, sizeof EXT5_DIGITS)) != cudaSuccess)
        return e;
    if ((e = cudaMemcpyToSymbol(c_e5, e5, sizeof e5)) != cudaSuccess)
        return e;
    if ((e = cudaMemcpyToSymbol(c_cov, cov, sizeof cov)) != cudaSuccess)
        return e;
    if ((e = cudaMemcpyToSymbol(c_near, nearv, sizeof nearv)) != cudaSuccess)
        return e;
    return cudaMemcpyToSymbol(c_s15, S15, sizeof S15);
}

struct Side { // one operand, row-major rows x K (B = the rows of B^T); rows padded (zero rows), K a multiple of 96
    const int8_t *raw;
    long long ld;
    int rows, rows_true, K, nb, kr; // kr: the raw k (K = kr padded up to 96; the pad reads as 0)
    int8_t *z2;     // rows x K: after the shift, uncovered leftovers moved to their nearest covered value
    int8_t *sh;     // rows x nb: the block's shift
    short *sum_raw; // rows x nb: block sums of the raw values
    short *sum_z2;  // rows x nb: block sums of z2
    int *lcnt;      // rows x nb: leftovers per block
    unsigned *stat; // [0] max |shift| (atomicMax), [1] blocks shifted, [2] leftovers, [3] block sums outside the digit
                    // table, [4] raw +-103 (8 words a side)
};
// one thread per (row, block)
__global__ void __launch_bounds__(256) shift_blocks(Side s) {
    __shared__ unsigned char cov[256];
    __shared__ signed char nearv[256];
    __shared__ unsigned bst[5];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        cov[i] = c_cov[i];
        nearv[i] = c_near[i];
    }
    if (threadIdx.x < 5)
        bst[threadIdx.x] = 0;
    __syncthreads();
    unsigned mx = 0, nsh = 0, nl_all = 0, nout = 0, n103 = 0;
    const long long nblk = (long long)s.rows * s.nb;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < nblk;
         q += (long long)gridDim.x * blockDim.x) {
        const int r = int(q / s.nb), b = int(q - (long long)r * s.nb);
        int v[16], sr = 0;
        if (r < s.rows_true) {
            const int8_t *z = s.raw + (size_t)r * s.ld + 16 * b;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                v[i] = (16 * b + i < s.kr) ? int(z[i]) : 0;
                sr += v[i];
                n103 += (v[i] == 103 || v[i] == -103) ? 1u : 0u;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                v[i] = 0;
        }
        int pick = 0;
        bool found = false;
#pragma unroll 1
        for (int tr = 0; tr < 15 && !found; ++tr) {
            const int tt = c_s15[tr];
            bool ok = true;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int y = v[i] - tt;
                ok = ok && y >= -128 && y <= 127 && cov[y + 128];
            }
            if (ok) {
                pick = tt;
                found = true;
            }
        }
        int nl = 0, sz = 0;
        int8_t *o = s.z2 + (size_t)r * s.K + 16 * b;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int y;
            if (found)
                y = v[i] - pick;
            else if (cov[v[i] + 128])
                y = v[i];
            else {
                y = nearv[v[i] + 128];
                ++nl;
            }
            o[i] = (int8_t)y;
            sz += y;
        }
        const size_t qi = (size_t)r * s.nb + b;
        s.sh[qi] = (int8_t)pick;
        s.sum_raw[qi] = (short)sr;
        s.sum_z2[qi] = (short)sz;
        s.lcnt[qi] = nl;
        const unsigned ap = unsigned(pick < 0 ? -pick : pick);
        mx = ap > mx ? ap : mx;
        nsh += pick != 0;
        nl_all += nl;
        nout += (sr < EXT5_LO || sr > EXT5_HI || sz < EXT5_LO || sz > EXT5_HI) ? 1u : 0u;
    }
    atomicMax(&bst[0], mx);
    atomicAdd(&bst[1], nsh);
    atomicAdd(&bst[2], nl_all);
    atomicAdd(&bst[3], nout);
    atomicAdd(&bst[4], n103);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(s.stat, bst[0]);
        if (bst[1])
            atomicAdd(s.stat + 1, bst[1]);
        if (bst[2])
            atomicAdd(s.stat + 2, bst[2]);
        if (bst[3])
            atomicAdd(s.stat + 3, bst[3]);
        if (bst[4])
            atomicAdd(s.stat + 4, bst[4]);
    }
}
// the leftover offsets, in two passes without cub (the local nvcc 11.5 / GCC 11 cannot include it): per row the
// exclusive scan of its blocks' counts (off = row-relative) and the row total, then the exclusive scan of the row
// totals (rowptr, rows + 1 entries)
__global__ void __launch_bounds__(256) row_scan(const int *lcnt, int rows, int nb, int *off, int *rowtot) {
    __shared__ int buf[256];
    __shared__ int carry;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        if (threadIdx.x == 0)
            carry = 0;
        __syncthreads();
        for (int b0 = 0; b0 < nb; b0 += 256) {
            const int b = b0 + threadIdx.x;
            const int v = b < nb ? lcnt[(size_t)r * nb + b] : 0;
            buf[threadIdx.x] = v;
            __syncthreads();
            for (int d = 1; d < 256; d <<= 1) {
                const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
                __syncthreads();
                buf[threadIdx.x] += t;
                __syncthreads();
            }
            if (b < nb)
                off[(size_t)r * nb + b] = carry + buf[threadIdx.x] - v;
            __syncthreads();
            if (threadIdx.x == 255)
                carry += buf[255];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            rowtot[r] = carry;
        __syncthreads();
    }
}
__global__ void __launch_bounds__(1024) rows_scan(const int *rowtot, int rows, int *rowptr) { // one block of 1,024
    __shared__ int buf[1024];
    __shared__ int carry;
    if (threadIdx.x == 0)
        carry = 0;
    __syncthreads();
    for (int r0 = 0; r0 < rows; r0 += 1024) {
        const int r = r0 + threadIdx.x;
        const int v = r < rows ? rowtot[r] : 0;
        buf[threadIdx.x] = v;
        __syncthreads();
        for (int d = 1; d < 1024; d <<= 1) {
            const int t = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
            __syncthreads();
            buf[threadIdx.x] += t;
            __syncthreads();
        }
        if (r < rows)
            rowptr[r] = carry + buf[threadIdx.x] - v;
        __syncthreads();
        if (threadIdx.x == 1023)
            carry += buf[1023];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        rowptr[rows] = carry;
}
// the leftovers of block q go to entries[rowptr[r] + off[q] ..] in k order: (k << 8) | (d + 128), d = raw - its nearest
// covered value
__global__ void __launch_bounds__(256)
    write_leftovers(Side s, const int *off, const int *rowptr, int *entries, long long cap) {
    const long long nblk = (long long)s.rows * s.nb;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < nblk;
         q += (long long)gridDim.x * blockDim.x) {
        if (!s.lcnt[q])
            continue;
        const int r = int(q / s.nb), b = int(q - (long long)r * s.nb);
        const int8_t *z = s.raw + (size_t)r * s.ld + 16 * b;
        int j = rowptr[r] + off[q];
        for (int i = 0; i < 16; ++i) {
            const int x = 16 * b + i < s.kr ? int(z[i]) : 0;
            if (!c_cov[x + 128]) {
                if (j < cap)
                    entries[j] = ((16 * b + i) << 8) | (x - c_near[x + 128] + 128);
                ++j;
            }
        }
    } // beyond cap: the gate sends the call to the exact fallback
}

template <int Rows>
__global__ void encode_faces(const int8_t *z2, unsigned char *packed, int side, int cuts, int stride_cuts, int K) {
    const unsigned cut = blockIdx.x % cuts, tile = blockIdx.x / cuts;
    const unsigned linear = blockIdx.y * blockDim.x + threadIdx.x;
    if (linear >= unsigned(Rows) * 48)
        return;
    const unsigned row = linear / 48, w = linear % 48, kp0 = 2 * w;
    unsigned byte = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned kp = kp0 + h, kk = kp % 16;
        const unsigned z = cut % 5u, flat = 6u * z + kp / 16u, f = flat % 5u;
        const int k = int((cut / 5u) * 96u + (flat / 5u) * 16u + kk);
        int c = 0;
        if (k < K) {
            const int x = z2[size_t(tile * Rows + row) * K + k];
            c = c_face[side][x + 128][f];
        }
        byte |= code4(c) << (4 * h);
    }
    packed[size_t(tile) * stride_cuts * (Rows * 48) + size_t(cut) * (Rows * 48) + tile_byte(row, w)] =
        (unsigned char)byte;
}

// the extension operands (d1p_1526.cu 1283-1296): xub = ceil(nb / 16) units a side, B-side units first; xu = the padded
// unit count. A (side 0): unit u < xub: digit j of the block sum of the RAW x (block b = 16 u + kp % 16); xub <= u < 2
// xub: the code of s of block 16 (u - xub) + kp % 16 in every digit position.  B (side 1): u < xub: the code of t of
// block b in every digit; xub <= u < 2 xub: digit j of the block sum of y''.  One thread per byte:
// [half][u][tile_byte(row, w)], halves of 128 (A) / 112 (B) rows.
__global__ void __launch_bounds__(256)
    ext_pack(const Side s, int side, int xub, int xu, unsigned char *out, int rows_per_half) {
    const long long per_half = (long long)xu * rows_per_half * 48,
                    total = (long long)(s.rows / rows_per_half) * per_half;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < total;
         e += (long long)gridDim.x * blockDim.x) {
        const int half = int(e / per_half);
        const long long e2 = e - (long long)half * per_half;
        const int u = int(e2 / (rows_per_half * 48)), rw = int(e2 - (long long)u * rows_per_half * 48), row = rw / 48,
                  w = rw % 48, r = half * rows_per_half + row;
        unsigned byte = 0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kp = 2 * w + h, j = kp / 16;
            if (u < xub) {
                const int b = 16 * u + kp % 16;
                if (b < s.nb) { // as the host: A's digits on every row (padded rows: the digits of 0), B's t codes on
                                // the true rows only
                    if (side == 0) {
                        const int sv = min(max(int(s.sum_raw[(size_t)r * s.nb + b]), EXT5_LO), EXT5_HI);
                        byte |= code4(c_digits[sv - EXT5_LO][j]) << (4 * h);
                    } else if (r < s.rows_true)
                        byte |= code4(s.sh[(size_t)r * s.nb + b]) << (4 * h);
                }
            } else if (u < 2 * xub) {
                const int ba = 16 * (u - xub) + kp % 16;
                if (ba < s.nb && r < s.rows_true) {
                    if (side == 0)
                        byte |= code4(s.sh[(size_t)r * s.nb + ba]) << (4 * h);
                    else {
                        const int sv = min(max(int(s.sum_z2[(size_t)r * s.nb + ba]), EXT5_LO), EXT5_HI);
                        byte |= code4(c_digits[sv - EXT5_LO][j]) << (4 * h);
                    }
                }
            }
        }
        out[((size_t)half * xu + u) * (rows_per_half * 48) + tile_byte(row, w)] = (unsigned char)byte;
    }
}
// the interval proof (d4_row_window_max): max over rows and aligned windows of Window k of the sum of E(z2); one warp
// per (row, window)
__global__ void __launch_bounds__(256) window_max(const int8_t *z2, int rows, int K, unsigned long long *out) {
    __shared__ unsigned long long bmax;
    if (threadIdx.x == 0)
        bmax = 0;
    __syncthreads();
    const int nw = (K + Window - 1) / Window, lane = threadIdx.x & 31;
    const long long tasks = (long long)rows * nw;
    unsigned long long wmax = 0;
    for (long long t = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 32; t < tasks;
         t += (long long)gridDim.x * blockDim.x / 32) {
        const int r = int(t / nw), wi = int(t - (long long)r * nw), k0 = wi * Window, k1 = min(K, k0 + Window);
        long long sm = 0;
        for (int k = k0 + lane; k < k1; k += 32) {
            const long long e = c_e5[z2[(size_t)r * K + k] + 128];
            sm += e < 0 ? (1ll << 50) : e;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1)
            sm += __shfl_xor_sync(0xFFFFFFFFu, sm, o);
        if (lane == 0 && (unsigned long long)sm > wmax)
            wmax = (unsigned long long)sm;
    }
    if (lane == 0)
        atomicMax(&bmax, wmax);
    __syncthreads();
    if (threadIdx.x == 0 && bmax)
        atomicMax(out, bmax);
}
// y'' transposed [K][np] (the read-out of A's leftovers): 32 x 32 tiles
__global__ void transpose_kn(const int8_t *z2, int n_rows, int K, int np, int8_t *out,
                             const int *a_total) { // a_total: A's leftover count (0: nothing reads it)
    if (a_total && *a_total == 0)
        return;
    __shared__ int8_t t[32][33];
    const int k0 = blockIdx.x * 32, n0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int n = n0 + i, k = k0 + threadIdx.x;
        t[i][threadIdx.x] = (n < n_rows && k < K) ? z2[(size_t)n * K + k] : 0;
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int k = k0 + i, n = n0 + threadIdx.x;
        if (k < K && n < np)
            out[(size_t)k * np + n] = t[threadIdx.x][i];
    }
}
// the gate (one thread): the call goes to the exact fallback (flag[0] = 1) when a shift is over 12, a block sum is
// outside the digit table, or the leftovers exceed their cap; the segment is 1,536 k (80 units) when the Cauchy-Schwarz
// proof holds (max_rows S_A x max_cols S_B
// <= 2^48, d4_interval_ok), else 672 k (35 units, the table bound 672 x 22,311 < 2^24).  The extension bound 2,133 x 12
// x 16 x 8 x 5 < 2^24 holds whenever the shifts are within 12.  flag[1] = the segment units (80 / 35), copied into
// v5k::c_segc by the host (device to device).
__global__ void gate(const unsigned *statA, const unsigned *statB, const int *totA, const int *totB, long long capA,
                     long long capB, const unsigned long long *proof, unsigned long long *flag, int *segc) {
    const bool bad = statA[0] > 12 || statB[0] > 12 || statA[3] || statB[3] || *totA > capA || *totB > capB;
    const unsigned long long sa = proof[0], sb = proof[1];
    const bool ok1536 = sa == 0 || sb == 0 || (sa <= 3000000000ull && sb <= 3000000000ull && sa * sb <= (1ull << 48));
    flag[0] = bad ? 1ull : 0ull;
    flag[1] = 0;
    *segc = ok1536 ? 80 : 35;
}
// B's leftovers (f5_fixb of the source, reading the original A with its lda): C[n][m] += sum over column n's entries
// (k, d) of d x A[m][k]; one thread per (row, column), the columns without entries return at once
__global__ void __launch_bounds__(256) fixb(int *C, long long ldc, int mt, const int8_t *A, long long lda,
                                            const int *bcol, const int *bent, const unsigned long long *flag) {
    if (flag[0])
        return;
    const int n = blockIdx.y, m = blockIdx.x * blockDim.x + threadIdx.x;
    const int q0 = bcol[n], q1 = bcol[n + 1];
    if (q0 == q1 || m >= mt)
        return;
    int sum_ = 0;
    for (int e = q0; e < q1; ++e) {
        const int ent = bent[e];
        sum_ += ((ent & 255) - 128) * int(A[(size_t)m * lda + (ent >> 8)]);
    }
    C[(size_t)n * ldc + m] += sum_;
}
} // namespace v5
