#pragma once
#include "s3_input_view.cuh"
namespace l8e {
template <int PR, bool CT, int FX = 0, int MINB = 3, class T = double>
__global__ __launch_bounds__(256, MINB) void encode_s3_l8x(
    const T *__restrict__ x, int64_t ld, const int *__restrict__ shifts,
    unsigned char *__restrict__ packed, const ModC *__restrict__ mods,
    const unsigned short *__restrict__ lut, int side_b, const Geom g, S3InputView view = {}) {
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *slut = reinterpret_cast<unsigned short *>(dsm);
    T *sa = reinterpret_cast<T *>(
        dsm +
        lut_smem_x(
            FX)); // [96][AStride] (a float input uses the first half of the double-sized space)
    static_assert(!(FX & 24) || (CT && PR == 0),
                  "FX 8 / 16 are for the compile-time form without probes");
    static_assert(!(FX & 256) || ((FX & 16) && ct_pairs_ok()),
                  "FX 256 is on FX 16's LUT, with four 1-plane moduli first");
    static_assert(!(FX & 1024) || ((FX & 16) && (FX & 8) && ct_pairs_ok() && cs_tables_ok()),
                  "FX 1024 is on FX 24's LUT");

    static_assert(!(FX & (2048 | 4096 | 8192)) ||
                      ((FX & 1024) && __builtin_popcount(FX & (2048 | 4096 | 8192)) == 1),
                  "one FX 1024 probe at a time");
    static_assert(!(FX & 16384) || (CT && PR == 0),
                  "FX 16384 is for the compile-time form without probes");
    unsigned *const cs_s =
        reinterpret_cast<unsigned *>(dsm + (LutX3Bytes + 15) / 16 * 16); // FX 1024: [22][L8CS_N]
    static_assert(((FX >> 5) & 7) == 0 || ((FX >> 5) & 7) == 1 || ((FX >> 5) & 7) == 2 ||
                      ((FX >> 5) & 7) == 4,
                  "one PDL form");
    if constexpr ((FX & 128) != 0) {
        pdl_wait();
        pdl_trigger();
    }
    if constexpr ((FX & 64) != 0) {
        pdl_trigger();
    }
    struct EndWait {
        __device__ ~EndWait() {
            if constexpr ((FX & 64) != 0) {
                pdl_wait();
            }
        }
    } end_wait; // every exit of the block
    const unsigned part = blockIdx.x, cut = blockIdx.y, rank = blockIdx.z & 1u,
                   tile = blockIdx.z >> 1;
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 12, chunk = linear % 12,
                   k0 = cut * 96 + chunk * 8;
    const unsigned pos =
        side_b ? tile * Cols + rank * (Cols / 2) + row : tile * Side + rank * (Side / 2) + row;
    const bool live = k0 < unsigned(g.k) && (side_b ? row < 112 : pos < unsigned(g.m));
    const int inner = int(tile_byte(row, chunk * 4));
    unsigned char *const tbase = packed + size_t(tile) * tile_block_bytes(g.cuts, g.planes) + inner;
    if constexpr ((PR & 8) != 0) { // stores only
        const unsigned seed = (linear * 0x9E3779B9u) ^ (blockIdx.y * 0x85EBCA6Bu) ^ blockIdx.z;
#pragma unroll 1
        for (int q = 0; q < L8_NMOD; ++q) {
            const ModC c = mods[q];
            for (int p = 0; p < c.planes; ++p) {
                *reinterpret_cast<unsigned *>(
                    tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, p)) =
                    live ? seed ^ unsigned(q * 3 + p) : 0u;
            }
        }
        return;
    }
    for (int i = threadIdx.x; i < ((FX & 1024)  ? LutX3Bytes / 4
                                   : (FX & 256) ? LutX2Bytes / 4
                                   : (FX & 16)  ? LutXPad / 4
                                                : LutWords / 2);
         i += 256) {
        reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
    }
    if constexpr ((FX & 1024) != 0) {
        for (int i = threadIdx.x; i < CsRowWords; i += 256) {
            cs_s[i] = 0; // before the block's first barrier below
        }
    }
    const unsigned rlo_part = (part * 256) / 12; // the block's first row (22 rows from here)
    long long v[8];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 12, rhi = (part * 256 + 255) / 12,
                       nr = (FX & 512) ? 22u : rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        if constexpr ((FX & 512) != 0) {

#pragma unroll 4
            for (unsigned e = threadIdx.x; e < 22u * 96u; e += 256) {
                const unsigned kk = e / 22u, r = e % 22u, kg = cut * 96 + kk;
                sa[kk * AStride + ((FX & 1) ? kk / 8 : 0) + r] =
                    kg < unsigned(g.k) && abase + r < unsigned(g.m)
                        ? view.load(x, size_t(kg) * ld + abase + r, ld, 0)
                        : T{};
            }
        } else {
            for (unsigned e = threadIdx.x; e < nr * 96; e += 256) {
                const unsigned kk = e / nr, r = e % nr, kg = cut * 96 + kk;
                sa[kk * AStride + ((FX & 1) ? kk / 8 : 0) + r] =
                    kg < unsigned(g.k) && abase + r < unsigned(g.m)
                        ? view.load(x, size_t(kg) * ld + abase + r, ld, 0)
                        : T{};
            }
        }
        __syncthreads();
        if constexpr ((FX & 32) != 0) {
            pdl_wait();
            pdl_trigger();
        } // the shifts come from the grids before
        const int sh = live ? shifts[pos] : 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] =
                live ? q1t(sa[(chunk * 8 + j) * AStride + ((FX & 1) ? chunk : 0) + (row - rlo)], sh)
                     : 0;
        }
    } else {
        __syncthreads(); // the LUT copy
        if constexpr ((FX & 32) != 0) {
            pdl_wait();
            pdl_trigger();
        }
        if (live) {
            const unsigned src_row = pos % unsigned(g.n);
            const int sh = shifts[src_row];
            const T *src = x + size_t(src_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = k0 + j < unsigned(g.k)
                           ? q1t(view.load(x, size_t(src_row) * ld + k0 + j, ld, 1), sh)
                           : 0;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = 0;
            }
        }
    }
    if (!live) { // padding: code 0 in every plane (FX 1024: no return, the block's row sums need every thread at the barrier)
        if ((FX & 4) != 0 && side_b &&
            row >= 112) { // never read by the MMA; zeroed once by the caller
            if constexpr ((FX & 1024) == 0) {
                return;
            }
        } else if constexpr ((FX & 16384) != 0) { // the 38-plane layout
            static_for<0, L8_NMOD>([&](auto qc) {
                constexpr int q = decltype(qc)::value;
#pragma unroll
                for (int p = 0; p < ct_planes_ns(q); ++p)
                    *reinterpret_cast<unsigned *>(tbase + plane_offset(g.cuts, ct_pbase_ns(q),
                                                                       ct_planes_ns(q), rank, cut,
                                                                       p)) = 0u;
            });
        } else {
#pragma unroll 1
            for (int q = 0; q < L8_NMOD; ++q) {
                const ModC c = mods[q];
                for (int p = 0; p < c.planes; ++p) {
                    *reinterpret_cast<unsigned *>(
                        tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, p)) = 0u;
                }
            }
        }
        if constexpr ((FX & 1024) == 0) {
            return;
        }
    }
    auto body = [&]() {
        unsigned lo[8], ch[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            lo[j] = unsigned(v[j]) & 0x3FFFFFFFu;
            ch[j] = unsigned(int(v[j] >> 30) + fzqe2::P2Offset);
        }
        // the codes of one modulus, packed and stored (planes 1 to 3; PR & 4 folds them into one store)
        auto emit = [&](const unsigned (&cd)[8], int planes, unsigned char *qb) {
            if constexpr ((FX & 8) != 0) {
                if (planes ==
                    3) { // byte 0 (planes 0, 1) and byte 1 (plane 2) of two codes per byte_perm
                    const unsigned xe = __byte_perm(cd[0], cd[2], 0x5140),
                                   ye = __byte_perm(cd[4], cd[6], 0x5140);
                    const unsigned xo = __byte_perm(cd[1], cd[3], 0x5140),
                                   yo = __byte_perm(cd[5], cd[7], 0x5140);
                    const unsigned ev = __byte_perm(xe, ye, 0x5410),
                                   eh = __byte_perm(xe, ye, 0x7632);
                    const unsigned od = __byte_perm(xo, yo, 0x5410),
                                   oh = __byte_perm(xo, yo, 0x7632);
                    *reinterpret_cast<unsigned *>(qb) =
                        (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
                    *reinterpret_cast<unsigned *>(qb + TileBytes) =
                        ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
                    *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) =
                        eh | (oh << 4); // plane 2's codes are < 16
                    return;
                }
            }
            const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                           od = gather4(cd[1], cd[3], cd[5], cd[7]);
            unsigned w0 = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4), w1 = 0, w2 = 0;
            if (planes >= 2)
                w1 = ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
            if (planes == 3) {
                const unsigned eh = gather4(cd[0] >> 8, cd[2] >> 8, cd[4] >> 8, cd[6] >> 8),
                               oh = gather4(cd[1] >> 8, cd[3] >> 8, cd[5] >> 8, cd[7] >> 8);
                w2 = (eh & 0x0F0F0F0Fu) | ((oh & 0x0F0F0F0Fu) << 4);
            }
            if constexpr ((PR & 4) != 0) {
                *reinterpret_cast<unsigned *>(qb) = w0 ^ w1 ^ w2;
            } else {
                *reinterpret_cast<unsigned *>(qb) = w0;
                if (planes >= 2)
                    *reinterpret_cast<unsigned *>(qb + TileBytes) = w1;
                if (planes == 3)
                    *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) = w2;
            }
        };
        auto codes = [&](unsigned (&cd)[8], const unsigned short *tab, unsigned rs, unsigned negp,
                         unsigned mm) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned t = (PR & 1) ? ((lo[j] ^ ch[j]) & 7u)
                                      : bqn_asm(mad_asm(ch[j], rs, lo[j]), negp, mm);
                if constexpr ((FX & 2) != 0 && (PR & 1) == 0)
                    t = min(t, t + negp); // [0, 2p) -> [0, p)
                cd[j] = (PR & 2) ? t : unsigned(tab[t]);
            }
        };
        unsigned sum_all = 0; // FX 4096 only
        if constexpr (CT) {
            const unsigned cs = 2u * unsigned(g.cuts) * unsigned(TileBytes),
                           rc = (rank * unsigned(g.cuts) + cut) * unsigned(TileBytes);
            static_for<0, L8_NMOD>([&](auto qc) {
                constexpr int q = decltype(qc)::value;
                constexpr unsigned p = unsigned(ct_mod(q));
                constexpr unsigned mm = unsigned((1ull << 32) / p), negp = 0u - p,
                                   rs = unsigned((1ull << 30) % p);
                constexpr int planes = ct_planes(q), lutoff = ct_lutoff(q);
                // the planes written and their base: FX 16384 writes 2 of a 3-plane modulus's planes (the LUT's codes stay 3-plane)
                constexpr int wplanes = (FX & 16384) ? ct_planes_ns(q) : planes,
                              pbase = (FX & 16384) ? ct_pbase_ns(q) : ct_pbase(q);
                unsigned cd[8];
                if constexpr (
                    (FX & 256) != 0 &&
                    q < 4) { // q 0 / 1 store the pairs (0, 2) / (1, 3); q 2 / 3 were stored with them
                    if constexpr (q < 2) {
                        constexpr unsigned P = unsigned(ct_pairmod(q)),
                                           mmP = unsigned((1ull << 32) / P), negP = 0u - P,
                                           rsP = unsigned((1ull << 30) % P);
                        const unsigned char *tabp =
                            reinterpret_cast<const unsigned char *>(slut) + LutXPad + ct_pairoff(q);
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            cd[j] = tabp[bqn_asm(mad_asm(ch[j], rsP, lo[j]), negP, mmP)];
                        const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                                       od = gather4(cd[1], cd[3], cd[5], cd[7]);
                        constexpr unsigned pb0 = (FX & 16384) ? ct_pbase_ns(q) : ct_pbase(q),
                                           pb2 =
                                               (FX & 16384) ? ct_pbase_ns(q + 2) : ct_pbase(q + 2);
                        *reinterpret_cast<unsigned *>(tbase + (cs * pb0 + rc)) =
                            (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
                        *reinterpret_cast<unsigned *>(tbase + (cs * pb2 + rc)) =
                            ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
                    }
                } else if constexpr ((FX & 1024) != 0 &&
                                     cs_index(q) >= 0) { // counted: u32 entries = code | E << 16
                    constexpr int c = cs_index(q), off = l8cs_off[c];
                    const unsigned *tab32 =
                        reinterpret_cast<const unsigned *>(
                            reinterpret_cast<const unsigned char *>(slut) + LutX2Bytes) +
                        off;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        cd[j] = tab32[bqn_asm(mad_asm(ch[j], rs, lo[j]), negp, mm)];
                    emit(cd, wplanes, tbase + (cs * unsigned(pbase) + rc * unsigned(wplanes)));
                    if constexpr ((FX & 8192) == 0) {
                        const unsigned sum = ((cd[0] + cd[1]) + (cd[2] + cd[3])) +
                                             ((cd[4] + cd[5]) + (cd[6] + cd[7]));
                        if constexpr ((FX & 4096) != 0)
                            sum_all += sum >> 16;
                        else
                            atomicAdd(&cs_s[(row - rlo_part) * L8CS_N + c], sum >> 16);
                    }
                } else if constexpr ((FX & 16) != 0 &&
                                     planes < 3) { // the u8 copy: the residue is the byte offset
                    constexpr int lut8off = ct_lut8off(q);
                    const unsigned char *tab8 =
                        reinterpret_cast<const unsigned char *>(slut) + LutWords * 2 + lut8off;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        cd[j] = tab8[bqn_asm(mad_asm(ch[j], rs, lo[j]), negp, mm)];
                    emit(cd, wplanes, tbase + (cs * unsigned(pbase) + rc * unsigned(wplanes)));
                } else {
                    codes(cd, slut + lutoff, rs, negp, mm);
                    emit(cd, wplanes, tbase + (cs * unsigned(pbase) + rc * unsigned(wplanes)));
                }
            });
            if constexpr ((FX & 4096) != 0)
                atomicAdd(&cs_s[(row - rlo_part) * L8CS_N], sum_all);
        } else {
#pragma unroll 1
            for (int q = 0; q < L8_NMOD; ++q) {
                const ModC c = mods[q];
                unsigned cd[8];
                codes(cd, slut + c.lutoff, c.rs, c.negp, c.mmagic);
                emit(cd, c.planes, tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, 0));
            }
        }
    };
    if constexpr ((FX & 8192) != 0) {
        if (live) {
            body();
        }
    } else if constexpr ((FX & 1024) != 0) {
        if (live) {
            body();
        }
        __syncthreads();
        // once per (row, accumulator): A's rows below m; B's rows below 112 and below n (the wrapped copies of the last tile add
        // nothing); a row shared with the neighbouring part's block gets both blocks' parts
        const unsigned base_row =
            side_b ? tile * Cols + rank * (Cols / 2) : tile * Side + rank * (Side / 2);
        unsigned *const out = g.cs + (side_b ? size_t(g.m) * L8CS_N : size_t(0));
        for (int i = threadIdx.x; i < CsRowWords; i += 256) {
            const unsigned rr = rlo_part + unsigned(i) / L8CS_N, gr = base_row + rr, val = cs_s[i];
            const bool ok = side_b ? (rr < 112 && gr < unsigned(g.n)) : gr < unsigned(g.m);
            if (val && ok && rr < 128 && ((FX & 2048) == 0 || cut == 0)) {
                atomicAdd(out + size_t(gr) * L8CS_N + unsigned(i) % L8CS_N, val);
            }
        }
    } else {
        body();
    }
}

template <int PR, bool CT, int FX = 0, int MINB = 3, class T = double>
inline void launch_s3_x(const Encoder &e, const T *x, int64_t ld, const int *shifts,
                        unsigned char *packed, int side_b, cudaStream_t stream = 0,
                        const unsigned char *lutx = nullptr, S3InputView view = {}) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(encode_s3_l8x<PR, CT, FX, MINB, T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_smem_x(0, FX)) != cudaSuccess) {
            throw std::runtime_error("l8e: encode_l8x attribute");
        }
        attr = true;
    }
    if (CT && (e.a_direct || !ct_tables_match())) {
        throw std::runtime_error("l8e: encode_s3_l8x<CT> is the 59-product form of l8_tables.h");
    }
    if ((FX & 16) && !lutx) {
        throw std::runtime_error("l8e: FX 16 needs the lut_x of this side");
    }
    const unsigned short *tab =
        (FX & 16) ? reinterpret_cast<const unsigned short *>(lutx) : e.dlut[side_b];
    const dim3 grid(6, unsigned(e.geo.cuts), unsigned(e.tiles(side_b) * 2));
    encode_s3_l8x<PR, CT, FX, MINB, T><<<grid, 256, dyn_smem_x(side_b, FX), stream>>>(
        x, ld, shifts, packed, e.dmods, tab, side_b, e.geo, view);
}

} // namespace l8e
