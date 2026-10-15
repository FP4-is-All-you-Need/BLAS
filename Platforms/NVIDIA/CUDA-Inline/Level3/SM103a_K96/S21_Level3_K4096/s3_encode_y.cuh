#pragma once

#include "s3_encode_view.cuh"

namespace l8e {

constexpr int kYFx =
    25; // the fused path's form: FX 1 (staging offset) | 8 (byte_perm 3-plane packing) | 16 (u8 LUT copy)
static_assert((kFastFx & 25) == 25, "s3_encode_y follows kFastFx's form 25");

__device__ __forceinline__ int q1f(float z, int sh) {
    const unsigned b = __float_as_uint(z), a = b & 0x7fffffffu;
    if (a >= 0x7f800000u || a == 0u) {
        return 0; // non-finite and zero -> 0 (q1i's)
    }
    const int e = int(a >> 23);
    const unsigned m = (a & 0x7fffffu) | (e ? 0x800000u : 0u);
    const int s = (e ? e : 1) - 150 + sh; // |z| 2^sh = m 2^s
    unsigned v;
    if (s >= 0) {
        v = s < 8 ? m << s
                  : 0u; // s <= 1 for bits <= 25 (m < 2^24); a larger s never comes from the shifts
    } else {
        const int d = -s;
        if (d > 25) {
            v = 0u; // m < 2^24 <= 2^(d-2): below 1/2
        } else {
            v = m >> d;
            const unsigned rem = m & ((1u << d) - 1u), half = 1u << (d - 1);
            v += (rem > half || (rem == half && (v & 1u))) ? 1u : 0u;
        }
    }
    return int(b) < 0 ? -int(v) : int(v);
}
template <int Q>
__host__ __device__ constexpr unsigned
yoff() { // (2^53 mod p) + p ceil(2^26 / p): v + yoff > 0 and < 2^32 for |v| <= 2^25
    return unsigned((1ull << 53) % unsigned(ct_mod(Q))) +
           unsigned(ct_mod(Q)) * (((1u << 26) + unsigned(ct_mod(Q)) - 1u) / unsigned(ct_mod(Q)));
}
// the staging buffer: floats, [96][AStride] with FX 1's one-word offset per 8 k (the read pattern of the old kernel)
constexpr int YStage = (96 * AStride + 12) * 4;
__host__ __device__ constexpr int y_smem(bool both, bool stage) {
    return LutSmemX * (both ? 2 : 1) + (stage ? YStage : 0);
}

struct YArgs {
    const float *x;
    int64_t ld;                 // the operand; ld = n for the view modes 1..3
    const int *shifts;          // per logical row (pos for A, the source row for B)
    unsigned char *packed;      // this side's planes
    unsigned char *packed_b;    // BOTH: B's planes
    const unsigned char *lut;   // this side's lut_x (u16 LUT then the u8 copy)
    const unsigned char *lut_b; // BOTH: B's lut_x
    int side_b, mode, n;        // n: the view's matrix order (modes 1..3) and B's wrap length
    int tiles_b;                // BOTH: B's tile count
};

template <int MINB, bool BOTH, bool SKIPPAD>
__global__ __launch_bounds__(256, MINB) void encode_s3_l8y(const YArgs a, const Geom g) {
    constexpr int FX = kYFx;
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *const slut = reinterpret_cast<unsigned short *>(dsm);
    unsigned short *const slut_b = reinterpret_cast<unsigned short *>(dsm + LutSmemX);
    float *const sa = reinterpret_cast<float *>(dsm + LutSmemX * (BOTH ? 2 : 1));
    const unsigned part = blockIdx.x, cut = blockIdx.y, rank = blockIdx.z & 1u,
                   tile = blockIdx.z >> 1;
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 12, chunk = linear % 12,
                   k0 = cut * 96 + chunk * 8;
    const int side_b = BOTH ? 0 : a.side_b;
    const unsigned pos =
        side_b ? tile * Cols + rank * (Cols / 2) + row : tile * Side + rank * (Side / 2) + row;
    const bool live = k0 < unsigned(g.k) && (side_b ? row < 112 : pos < unsigned(g.m));
    const unsigned rlo = (part * 256) / 12; // the block's first row (22 rows, staging_rows_are_22)
    for (int i = threadIdx.x; i < LutXPad / 4; i += 256) {
        reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(a.lut)[i];
        if constexpr (BOTH) {
            reinterpret_cast<unsigned *>(slut_b)[i] =
                reinterpret_cast<const unsigned *>(a.lut_b)[i];
        }
    }
    // staging (A side every mode; B side mode 1): 22 rows x 96 k of the logical operand into sa[kk][r]
    const bool staged = !side_b || a.mode == 1;
    const unsigned sbase =
        side_b ? tile * Cols + rank * (Cols / 2) + rlo : tile * Side + rank * (Side / 2) + rlo;
    const unsigned kb = cut * 96;
    if (staged) {
        const unsigned rows_ok =
            unsigned(g.m); // A rows >= m are padding (B: every staged row is a wrapped source row)
        const float *const x = a.x;
        const int mode = a.mode;
        // above-diagonal rectangle of mode 3: every (i, p) with i < p, i = source row, p = k
        const bool upper3 =
            mode == 3 && sbase + 21u < kb; // the last row of the rectangle is below its first k
#pragma unroll 3
        for (unsigned e = threadIdx.x; e < 22u * 96u; e += 256) {
            unsigned kk, r;
            if (upper3) {
                r = e / 96u;
                kk = e % 96u;
            } else {
                kk = e / 22u;
                r = e % 22u;
            }
            const unsigned kg = kb + kk;
            const unsigned prow =
                sbase +
                r; // the position of the staged row (B: rows 112..127 of a half are staged, never used)
            const bool ok = kg < unsigned(g.k) && (side_b || prow < rows_ok);
            float val = 0.f;
            if (ok) {
                if (side_b) { // mode 1: B(i, p) = x[i + p n], i = the wrapped source row
                    unsigned i =
                        prow; // the wrapped source row: prow < tiles_b * Cols < n + Cols, so one subtraction unless n < Cols
                    if (i >= unsigned(a.n)) {
                        i -= unsigned(a.n);
                    }
                    if (i >= unsigned(a.n)) {
                        i %= unsigned(a.n);
                    }
                    val = x[size_t(kg) * size_t(a.n) + i];
                } else if (mode == 0) {
                    val = x[size_t(kg) * size_t(a.ld) + prow];
                } else if (mode == 2) {
                    val = prow >= kg ? x[size_t(kg) * size_t(a.n) + prow] : 0.f;
                } else { // mode 3
                    val = prow >= kg ? x[size_t(kg) * size_t(a.n) + prow]
                                     : x[size_t(prow) * size_t(a.n) + kg];
                }
            }
            sa[kk * AStride + kk / 8 + r] = val;
        }
    }
    __syncthreads();
    int v[8];
    if (staged) {
        const unsigned srow = side_b ? pos % unsigned(a.n) : pos;
        const int sh = live ? a.shifts[srow] : 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = live ? q1f(sa[(chunk * 8 + j) * AStride + chunk + (row - rlo)], sh) : 0;
        }
    } else if (live) { // B side, mode 0: row src_row of B, 8 consecutive k
        const unsigned src_row = pos % unsigned(g.n);
        const int sh = a.shifts[src_row];
        const float *src = a.x + size_t(src_row) * size_t(a.ld) + k0;
        if (k0 + 8 <= unsigned(g.k) && ((reinterpret_cast<size_t>(src) & 15u) == 0)) {
            const float4 f0 = __ldg(reinterpret_cast<const float4 *>(src)),
                         f1 = __ldg(reinterpret_cast<const float4 *>(src) + 1);
            v[0] = q1f(f0.x, sh);
            v[1] = q1f(f0.y, sh);
            v[2] = q1f(f0.z, sh);
            v[3] = q1f(f0.w, sh);
            v[4] = q1f(f1.x, sh);
            v[5] = q1f(f1.y, sh);
            v[6] = q1f(f1.z, sh);
            v[7] = q1f(f1.w, sh);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = k0 + j < unsigned(g.k) ? q1f(src[j], sh) : 0;
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = 0;
        }
    }
    const unsigned cs = 2u * unsigned(g.cuts) * unsigned(TileBytes),
                   rc = (rank * unsigned(g.cuts) + cut) * unsigned(TileBytes);
    unsigned char *const tbase = a.packed + size_t(tile) * tile_block_bytes(g.cuts, g.planes) +
                                 unsigned(tile_byte(row, chunk * 4));
    // BOTH: B's positions of this row (pos < n): the row itself and, for pos < tiles_b * Cols - n, its wrapped copy at pos + n
    unsigned char *tb1 = nullptr;
    unsigned rc1 = 0;
    unsigned char *tb2 = nullptr;
    unsigned rc2 = 0;
    if constexpr (BOTH) {
        if (pos < unsigned(g.n)) {
            const unsigned P = pos, tb = P / Cols, w = P % Cols, rb = w / (Cols / 2),
                           rr = w % (Cols / 2);
            tb1 = a.packed_b + size_t(tb) * tile_block_bytes(g.cuts, g.planes) +
                  unsigned(tile_byte(rr, chunk * 4));
            rc1 = (rb * unsigned(g.cuts) + cut) * unsigned(TileBytes);
            const unsigned P2 = pos + unsigned(g.n);
            if (P2 < unsigned(a.tiles_b) * Cols) {
                const unsigned tb_ = P2 / Cols, w_ = P2 % Cols, rb_ = w_ / (Cols / 2),
                               rr_ = w_ % (Cols / 2);
                tb2 = a.packed_b + size_t(tb_) * tile_block_bytes(g.cuts, g.planes) +
                      unsigned(tile_byte(rr_, chunk * 4));
                rc2 = (rb_ * unsigned(g.cuts) + cut) * unsigned(TileBytes);
            }
        }
    }
    const bool kpad = k0 >= unsigned(g.k); // the k padding of a live row: code 0 on both sides
    if (!live) {
        if (SKIPPAD && side_b && row >= 112) {
            return; // never read by the MMA; zeroed once by the caller
        }
        static_for<0, L8_NMOD>([&](auto qc) {
            constexpr int q = decltype(qc)::value;
#pragma unroll
            for (int p = 0; p < ct_planes(q); ++p) {
                *reinterpret_cast<unsigned *>(
                    tbase + (cs * unsigned(ct_pbase(q)) + rc * unsigned(ct_planes(q))) +
                    p * TileBytes) = 0u;
                if constexpr (BOTH) {
                    if (kpad && tb1)
                        *reinterpret_cast<unsigned *>(
                            tb1 + (cs * unsigned(ct_pbase(q)) + rc1 * unsigned(ct_planes(q))) +
                            p * TileBytes) = 0u;
                    if (kpad && tb2)
                        *reinterpret_cast<unsigned *>(
                            tb2 + (cs * unsigned(ct_pbase(q)) + rc2 * unsigned(ct_planes(q))) +
                            p * TileBytes) = 0u;
                }
            }
        });
        return;
    }
    auto emit = [&](const unsigned (&cd)[8], int planes, unsigned char *qb) {
        if (planes == 3) { // FX 8
            const unsigned xe = __byte_perm(cd[0], cd[2], 0x5140),
                           ye = __byte_perm(cd[4], cd[6], 0x5140);
            const unsigned xo = __byte_perm(cd[1], cd[3], 0x5140),
                           yo = __byte_perm(cd[5], cd[7], 0x5140);
            const unsigned ev = __byte_perm(xe, ye, 0x5410), eh = __byte_perm(xe, ye, 0x7632);
            const unsigned od = __byte_perm(xo, yo, 0x5410), oh = __byte_perm(xo, yo, 0x7632);
            *reinterpret_cast<unsigned *>(qb) = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
            *reinterpret_cast<unsigned *>(qb + TileBytes) =
                ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
            *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) = eh | (oh << 4);
            return;
        }
        const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                       od = gather4(cd[1], cd[3], cd[5], cd[7]);
        *reinterpret_cast<unsigned *>(qb) = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
        if (planes >= 2) {
            *reinterpret_cast<unsigned *>(qb + TileBytes) =
                ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
        }
    };
    static_for<0, L8_NMOD>([&](auto qc) {
        constexpr int q = decltype(qc)::value;
        constexpr unsigned p = unsigned(ct_mod(q));
        constexpr unsigned mm = unsigned((1ull << 32) / p), negp = 0u - p, off = yoff<q>();
        constexpr int planes = ct_planes(q), pbase = ct_pbase(q);
        unsigned t[8], cd[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            t[j] = bqn_asm(unsigned(v[j]) + off, negp, mm);
        if constexpr (planes < 3) { // FX 16: the u8 copy
            constexpr int lut8off = ct_lut8off(q);
            const unsigned char *tab8 =
                reinterpret_cast<const unsigned char *>(slut) + LutWords * 2 + lut8off;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                cd[j] = tab8[t[j]];
        } else {
            const unsigned short *tab = slut + ct_lutoff(q);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                cd[j] = tab[t[j]];
        }
        emit(cd, planes, tbase + (cs * unsigned(pbase) + rc * unsigned(planes)));
        if constexpr (BOTH) {
            if (tb1) {
                if constexpr (planes < 3) {
                    const unsigned char *tab8 = reinterpret_cast<const unsigned char *>(slut_b) +
                                                LutWords * 2 + ct_lut8off(q);
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        cd[j] = tab8[t[j]];
                } else {
                    const unsigned short *tab = slut_b + ct_lutoff(q);
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        cd[j] = tab[t[j]];
                }
                emit(cd, planes, tb1 + (cs * unsigned(pbase) + rc1 * unsigned(planes)));
                if (tb2)
                    emit(cd, planes, tb2 + (cs * unsigned(pbase) + rc2 * unsigned(planes)));
            }
        }
    });
}

// host: every LUT entry repeats with period p (the residue may land on t or t + p), both sides, u16 and u8 copies
inline bool lut_period_ok(const std::vector<unsigned char> &h) {
    for (int q = 0; q < L8_NMOD; ++q) {
        const int p = ct_mod(q), o16 = ct_lutoff(q), o8 = LutWords * 2 + ct_lut8off(q);
        for (int t = 0; t < p; ++t) {
            const unsigned a16 = h[2 * (o16 + t)] | (h[2 * (o16 + t) + 1] << 8),
                           b16 = h[2 * (o16 + t + p)] | (h[2 * (o16 + t + p) + 1] << 8);
            if (a16 != b16) {
                return false;
            }
            if (ct_planes(q) < 3 && (h[o8 + t] != h[o8 + t + p] || h[o8 + t] != (a16 & 0xFFu))) {
                return false;
            }
        }
    }
    return true;
}
template <bool BOTH, bool SKIPPAD, int MINB = 3>
inline void launch_s3_y(const Encoder &e, const YArgs &a, cudaStream_t st = 0) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(encode_s3_l8y<MINB, BOTH, SKIPPAD>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 y_smem(BOTH, true)) != cudaSuccess) {
            throw std::runtime_error("l8e: encode_s3_l8y attribute");
        }
        attr = true;
    }
    if (e.a_direct || !ct_tables_match()) {
        throw std::runtime_error("l8e: encode_s3_l8y is the compile-time form of l8_tables.h");
    }
    if (e.bits > 25) {
        throw std::runtime_error("l8e: encode_s3_l8y quantizes FP32 inputs with bits <= 25");
    }
    const int side_b = BOTH ? 0 : a.side_b;
    const bool staged = !side_b || a.mode == 1;
    const dim3 grid(6, unsigned(e.geo.cuts), unsigned(e.tiles(side_b) * 2));
    encode_s3_l8y<MINB, BOTH, SKIPPAD><<<grid, 256, y_smem(BOTH, staged), st>>>(a, e.geo);
}

} // namespace l8e
