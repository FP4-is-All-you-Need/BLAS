// Convert column-major operands to K-contiguous rows. Pad K with zero bytes
// to a multiple of 16, preserving the integer product for arbitrary leading dimensions.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace api_layout {

__host__ __device__ inline int kpad(int k) { return (k + 15) & ~15; }

// ---- KMAJ: copy K-contiguous rows into rows of kp bytes (16 bytes per thread; uint4 load when the source is aligned
// and full)
__global__ void __launch_bounds__(256) repack_kmaj(const int8_t *__restrict__ src, long long lds, long long sstride,
                                                   int rows, int k, int8_t *__restrict__ dst, int kp) {
    const int8_t *s = src + (long long)blockIdx.z * sstride;
    int8_t *d = dst + (long long)blockIdx.z * (long long)rows * kp;
    const int per_row = kp / 16;
    const long long total = (long long)rows * per_row;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const int r = int(i / per_row), c = int(i % per_row), kk = 16 * c;
        const int8_t *p = s + (long long)r * lds + kk;
        uint4 v;
        if (kk + 16 <= k && (reinterpret_cast<uintptr_t>(p) & 15) == 0)
            v = *reinterpret_cast<const uint4 *>(p);
        else {
            uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
            for (int b = 0; b < 16; ++b)
                if (kk + b < k)
                    w[b >> 2] |= uint32_t(uint8_t(p[b])) << (8 * (b & 3));
            v = make_uint4(w[0], w[1], w[2], w[3]);
        }
        *reinterpret_cast<uint4 *>(d + (long long)r * kp + kk) = v;
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

// ---- RMAJ: transpose rows-contiguous columns into K-major rows.  Tile = 64 rows x 64 k.  Load: each thread 16
// row-bytes of one k line (uint4 when aligned and full).  Store: each thread a 4-row x 4-k block through tr4x4, 4-byte
// stores (rows of kp bytes).
__global__ void __launch_bounds__(256) transpose_rmaj(const int8_t *__restrict__ src, long long lds, long long sstride,
                                                      int rows, int k, int8_t *__restrict__ dst, int kp) {
    __shared__ uint32_t tile[64][17]; // [k line][row word] (+1 pad)
    const int8_t *s = src + (long long)blockIdx.z * sstride;
    int8_t *d = dst + (long long)blockIdx.z * (long long)rows * kp;
    const int rt = (rows + 63) / 64, kt = (kp + 63) / 64, t = threadIdx.x;
    for (int q = blockIdx.x; q < rt * kt; q += gridDim.x) {
        const int r0 = (q % rt) * 64, k0 = (q / rt) * 64;
        { // load: k line kl = t / 4, row bytes 16 * (t % 4) .. +15
            const int kl = t >> 2, rb = 16 * (t & 3), kk = k0 + kl, r = r0 + rb;
            uint32_t w[4] = {0, 0, 0, 0};
            if (kk < k) {
                const int8_t *p = s + (long long)kk * lds + r;
                if (r + 16 <= rows && (reinterpret_cast<uintptr_t>(p) & 15) == 0) {
                    const uint4 v = *reinterpret_cast<const uint4 *>(p);
                    w[0] = v.x;
                    w[1] = v.y;
                    w[2] = v.z;
                    w[3] = v.w;
                } else {
#pragma unroll
                    for (int b = 0; b < 16; ++b)
                        if (r + b < rows)
                            w[b >> 2] |= uint32_t(uint8_t(p[b])) << (8 * (b & 3));
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                tile[kl][(rb >> 2) + i] = w[i];
        }
        __syncthreads();
        { // store: rows 4 * (t / 16) .. +3, k 4 * (t % 16) .. +3
            const int rg = t >> 4, kb = t & 15, kk = k0 + 4 * kb;
            uint32_t y[4];
            tr4x4(tile[4 * kb][rg], tile[4 * kb + 1][rg], tile[4 * kb + 2][rg], tile[4 * kb + 3][rg], y[0], y[1], y[2],
                  y[3]);
            if (kk < kp) { // kp is a multiple of 16 and kk of 4: the 4 bytes are all inside the row (zeros past k come
                           // from the load)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + 4 * rg + j;
                    if (r < rows)
                        *reinterpret_cast<uint32_t *>(d + (long long)r * kp + kk) = y[j];
                }
            }
        }
        __syncthreads();
    }
}

// ---- RMAJ v2 (20:5x): tile = 128 rows x 128 k (16 KB).  Load: thread t takes k line t / 2, row bytes 64 * (t % 2) ..
// +63 (four 16-byte pieces).  Store: thread t takes rows 4 * (t / 8) .. +3 and k 16 * (t % 8) .. +15: 16 word loads,
// four tr4x4, four 16-byte stores (one per row).  Shared Memory rows of 32 words, word w of k line L at column w ^
// f(L), f(L) = 4 * (((L >> 4) ^ L) & 7): the 16-byte stores of a quarter warp (4 lines x 2 halves) and the word loads
// of the gather (8 line groups x 4 columns) hit 32 banks.
__device__ __forceinline__ int swz(int L) { return 4 * (((L >> 4) ^ L) & 7); }
__global__ void __launch_bounds__(256) transpose_rmaj2(const int8_t *__restrict__ src, long long lds, long long sstride,
                                                       int rows, int k, int8_t *__restrict__ dst, int kp) {
    __shared__ __align__(16) uint32_t tile[128][32];
    const int8_t *s = src + (long long)blockIdx.z * sstride;
    int8_t *d = dst + (long long)blockIdx.z * (long long)rows * kp;
    const int rt = (rows + 127) / 128, kt = (kp + 127) / 128, t = threadIdx.x;
    for (int q = blockIdx.x; q < rt * kt; q += gridDim.x) {
        const int r0 = (q % rt) * 128, k0 = (q / rt) * 128;
        {
            const int kl = t >> 1, h = t & 1, kk = k0 + kl, f = swz(kl);
#pragma unroll
            for (int pc = 0; pc < 4; ++pc) {
                const int rb = 64 * h + 16 * pc, r = r0 + rb;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (kk < k) {
                    const int8_t *p = s + (long long)kk * lds + r;
                    if (r + 16 <= rows && (reinterpret_cast<uintptr_t>(p) & 15) == 0)
                        v = *reinterpret_cast<const uint4 *>(p);
                    else if (r < rows) {
                        uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
                        for (int b = 0; b < 16; ++b)
                            if (r + b < rows)
                                w[b >> 2] |= uint32_t(uint8_t(p[b])) << (8 * (b & 3));
                        v = make_uint4(w[0], w[1], w[2], w[3]);
                    }
                }
                *reinterpret_cast<uint4 *>(&tile[kl][(rb >> 2) ^ f]) =
                    v; // rb >> 2 is a multiple of 4 and f too: the 4 words stay together
            }
        }
        __syncthreads();
        {
            const int rg = t >> 3, kq = t & 7, kk = k0 + 16 * kq;
            uint32_t y[4][4]; // [block][row j]
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int L = 16 * kq + 4 * b;
                tr4x4(tile[L][rg ^ swz(L)], tile[L + 1][rg ^ swz(L + 1)], tile[L + 2][rg ^ swz(L + 2)],
                      tile[L + 3][rg ^ swz(L + 3)], y[b][0], y[b][1], y[b][2], y[b][3]);
            }
            if (kk < kp) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + 4 * rg + j;
                    if (r < rows)
                        *reinterpret_cast<uint4 *>(d + (long long)r * kp + kk) =
                            make_uint4(y[0][j], y[1][j], y[2][j], y[3][j]);
                }
            }
        }
        __syncthreads();
    }
}

// host: write op(X) as K-major rows of kp bytes into dst (dst holds batch * rows * kp bytes, 16-byte aligned).
// trans_is_kmaj: true when the operand is already K-contiguous (opA = T, opB = N).
// tvar: 2 = transpose_rmaj2 (default), 1 = the 64 x 64 form (kept for comparison).
inline cudaError_t to_kmajor(const int8_t *src, long long lds, long long sstride, int batch, int rows, int k,
                             bool trans_is_kmaj, int8_t *dst, int sms, cudaStream_t st, int tvar = 2) {
    const int kp = kpad(k);
    if (trans_is_kmaj)
        repack_kmaj<<<dim3(sms * 8, 1, batch), 256, 0, st>>>(src, lds, sstride, rows, k, dst, kp);
    else if (tvar == 1)
        transpose_rmaj<<<dim3(sms * 8, 1, batch), 256, 0, st>>>(src, lds, sstride, rows, k, dst, kp);
    else
        transpose_rmaj2<<<dim3(sms * 4, 1, batch), 256, 0, st>>>(src, lds, sstride, rows, k, dst, kp);
    return cudaGetLastError();
}

} // namespace api_layout
