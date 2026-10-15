#pragma once
// Encoder stage probes share the reference encoder's grid and packed layout.
// CT uses compile-time moduli and unrolls the modulus loop.
// PR changes output for timing only: 1 skips reduction, 2 skips the LUT, 4 XORs plane
// words into one store per modulus, and 8 performs stores without input or quantization.
// All probe arithmetic remains connected to a store.
// FX options are intended to preserve values except for unread padding below.
// Historical device/shape equality checks for the selected variants are listed in l8_encode_fast.cuh:
//   1 pads A staging by one double per eight k values to reduce shared-memory bank conflicts.
//   2 reduces a lazy residue to [0,p) with unsigned min(t,t-p) before looking up the LUT.
//   4 skips B rows 112..127; the paired-CTA N=224 MMA reads only 112 rows per CTA.
//     Zero the packed B buffer once if byte equality for unread padding is required.
//     k padding is still written as code 0.
//   8 packs three-plane codes with byte permutations, using that the third code is below 16.
//   16 uses byte LUT entries for one- and two-plane moduli, after the word LUT.
// For programmatic dependent launch on sm_90 and later: 32 stages inputs before waiting
// and then triggers dependents; 64 triggers first and waits before return; 128 waits first
// and then triggers. Shift values must be read after the preceding kernel completes.
#include <type_traits>
#include "l8_encode.cuh"

namespace l8e {

// Compile-time constants from l8_tables.h: moduli, offsets, and plane counts.
// Device code uses them in constant expressions.
__host__ __device__ constexpr int ct_mod(int q) {
    return l8_mod_ct(q);
}
__host__ __device__ constexpr int ct_planes(int q) {
    return l8_planes_ct(q);
}
__host__ __device__ constexpr int ct_lutoff(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += 2 * ct_mod(i);
    }
    return s;
}
__host__ __device__ constexpr int ct_pbase(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += ct_planes(i);
    }
    return s;
}
// FX 16: the u8 copy of the 1- and 2-plane moduli's entries, after the u16 LUT (byte offsets)
__host__ __device__ constexpr int ct_lut8off(int q) {
    int s = 0;
    for (int i = 0; i < q; ++i) {
        s += ct_planes(i) < 3 ? 2 * ct_mod(i) : 0;
    }
    return s;
}
constexpr int Lut8Bytes = ct_lut8off(L8_NMOD);      // 1,600
constexpr int LutXBytes = LutWords * 2 + Lut8Bytes; // 8,040
constexpr int LutSmemX = (LutXBytes + 15) / 16 * 16;
static_assert(ct_lutoff(L8_NMOD) == LutWords && ct_pbase(L8_NMOD) == L8_NPLANES, "l8e: ct tables");
inline bool
ct_tables_match() { // l8_tables.h's device-side copies against its arrays (both written by gen_l8_tables.py)
    for (int q = 0; q < L8_NMOD; ++q) {
        if (ct_mod(q) != l8_mod[q] || ct_planes(q) != l8_planes[q]) {
            return false;
        }
    }
    return true;
}

// programmatic dependent launch (PTX griddepcontrol, sm_90+): no-ops when the grid was not launched as a dependent
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <int I, int N, class F> __device__ __forceinline__ void static_for(F &&f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        static_for<I + 1, N>(f);
    }
}

constexpr int AStageX = (64 * AStride + 8) * 8; // FX & 1: + one double per chunk of 8 k
__host__ __device__ constexpr int lut_smem_x(int fx) {
    return (fx & 16) ? LutSmemX : LutSmem;
}
__host__ __device__ constexpr int dyn_smem_x(int side_b, int fx) {
    return lut_smem_x(fx) + (side_b ? 0 : ((fx & 1) ? AStageX : AStage));
}

template <int PR, bool CT, int FX = 0,
          int MINB = 3> // MINB 4: 64 registers, 4 blocks per SM
__global__ __launch_bounds__(256, MINB) void encode_l8x(const double *__restrict__ x, int64_t ld,
                                                        const int *__restrict__ shifts,
                                                        unsigned char *__restrict__ packed,
                                                        const ModC *__restrict__ mods,
                                                        const unsigned short *__restrict__ lut,
                                                        int side_b, const Geom g) {
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned short *slut = reinterpret_cast<unsigned short *>(dsm);
    double *sa = reinterpret_cast<double *>(dsm + lut_smem_x(FX)); // [96][AStride]
    static_assert(!(FX & 24) || (CT && PR == 0),
                  "FX 8 / 16 are for the compile-time form without probes");
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
    const unsigned linear = part * 256 + threadIdx.x, row = linear / 8, chunk = linear % 8,
                   k0 = cut * 64 + chunk * 8;
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
    for (int i = threadIdx.x; i < ((FX & 16) ? LutXBytes / 4 : LutWords / 2); i += 256) {
        reinterpret_cast<unsigned *>(slut)[i] = reinterpret_cast<const unsigned *>(lut)[i];
    }
    long long v[8];
    if (!side_b) {
        const unsigned rlo = (part * 256) / 8, rhi = (part * 256 + 255) / 8, nr = rhi - rlo + 1;
        const unsigned abase = tile * Side + rank * (Side / 2) + rlo;
        for (unsigned e = threadIdx.x; e < nr * 64; e += 256) {
            const unsigned kk = e / nr, r = e % nr, kg = cut * 64 + kk;
            sa[kk * AStride + ((FX & 1) ? kk / 8 : 0) + r] =
                kg < unsigned(g.k) && abase + r < unsigned(g.m) ? x[size_t(kg) * ld + abase + r]
                                                                : 0.0;
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
                live ? q1i(sa[(chunk * 8 + j) * AStride + ((FX & 1) ? chunk : 0) + (row - rlo)], sh)
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
            const double *src = x + size_t(src_row) * ld + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = k0 + j < unsigned(g.k) ? q1i(src[j], sh) : 0;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = 0;
            }
        }
    }
    if (!live) { // padding: code 0 in every plane
        if ((FX & 4) != 0 && side_b && row >= 112) {
            return; // never read by the MMA; zeroed once by the caller
        }
#pragma unroll 1
        for (int q = 0; q < L8_NMOD; ++q) {
            const ModC c = mods[q];
            for (int p = 0; p < c.planes; ++p) {
                *reinterpret_cast<unsigned *>(
                    tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, p)) = 0u;
            }
        }
        return;
    }
    unsigned lo[8], ch[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        lo[j] = unsigned(v[j]) & 0x3FFFFFFFu;
        ch[j] = unsigned(int(v[j] >> 30) + fzqe2::P2Offset);
    }
    // the codes of one modulus, packed and stored (planes 1 to 3; PR & 4 folds them into one store)
    auto emit = [&](const unsigned (&cd)[8], int planes, unsigned char *qb) {
        if constexpr ((FX & 8) != 0) {
            if (planes >= 3) { // byte 0 (planes 0, 1) and byte 1 of two codes per byte_perm
                const unsigned xe = __byte_perm(cd[0], cd[2], 0x5140),
                               ye = __byte_perm(cd[4], cd[6], 0x5140);
                const unsigned xo = __byte_perm(cd[1], cd[3], 0x5140),
                               yo = __byte_perm(cd[5], cd[7], 0x5140);
                const unsigned ev = __byte_perm(xe, ye, 0x5410), eh = __byte_perm(xe, ye, 0x7632);
                const unsigned od = __byte_perm(xo, yo, 0x5410), oh = __byte_perm(xo, yo, 0x7632);
                *reinterpret_cast<unsigned *>(qb) = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4);
                *reinterpret_cast<unsigned *>(qb + TileBytes) =
                    ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
                if (planes == 3) {
                    *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) =
                        eh | (oh << 4); // plane 2's codes are < 16
                } else {                // planes 2 and 3 share byte 1
                    *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) =
                        (eh & 0x0F0F0F0Fu) | ((oh & 0x0F0F0F0Fu) << 4);
                    *reinterpret_cast<unsigned *>(qb + 3 * TileBytes) =
                        ((eh >> 4) & 0x0F0F0F0Fu) | (oh & 0xF0F0F0F0u);
                }
                return;
            }
        }
        const unsigned ev = gather4(cd[0], cd[2], cd[4], cd[6]),
                       od = gather4(cd[1], cd[3], cd[5], cd[7]);
        unsigned w0 = (ev & 0x0F0F0F0Fu) | ((od & 0x0F0F0F0Fu) << 4), w1 = 0, w2 = 0,
                 w3 = 0; // w3
        if (planes >= 2) {
            w1 = ((ev >> 4) & 0x0F0F0F0Fu) | (od & 0xF0F0F0F0u);
        }
        if (planes >= 3) {
            const unsigned eh = gather4(cd[0] >> 8, cd[2] >> 8, cd[4] >> 8, cd[6] >> 8),
                           oh = gather4(cd[1] >> 8, cd[3] >> 8, cd[5] >> 8, cd[7] >> 8);
            w2 = (eh & 0x0F0F0F0Fu) | ((oh & 0x0F0F0F0Fu) << 4);
            if (planes == 4) {
                w3 = ((eh >> 4) & 0x0F0F0F0Fu) | (oh & 0xF0F0F0F0u);
            }
        }
        if constexpr ((PR & 4) != 0) {
            *reinterpret_cast<unsigned *>(qb) = w0 ^ w1 ^ w2 ^ w3;
        } else {
            *reinterpret_cast<unsigned *>(qb) = w0;
            if (planes >= 2) {
                *reinterpret_cast<unsigned *>(qb + TileBytes) = w1;
            }
            if (planes >= 3) {
                *reinterpret_cast<unsigned *>(qb + 2 * TileBytes) = w2;
            }
            if (planes == 4) {
                *reinterpret_cast<unsigned *>(qb + 3 * TileBytes) = w3;
            }
        }
    };
    auto codes = [&](unsigned (&cd)[8], const unsigned short *tab, unsigned rs, unsigned negp,
                     unsigned mm) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned t = (PR & 1)
                             ? ((lo[j] ^ ch[j]) & 7u)
                             : bqn_asm(mad_asm(l8e_chr(ch[j], rs, negp, mm), rs, lo[j]), negp, mm);
            if constexpr ((FX & 2) != 0 && (PR & 1) == 0)
                t = min(t, t + negp); // [0, 2p) -> [0, p)
            cd[j] = (PR & 2) ? t : unsigned(tab[t]);
        }
    };
    if constexpr (CT) {
        const unsigned cs = 2u * unsigned(g.cuts) * unsigned(TileBytes),
                       rc = (rank * unsigned(g.cuts) + cut) * unsigned(TileBytes);
        static_for<0, L8_NMOD>([&](auto qc) {
            constexpr int q = decltype(qc)::value;
            constexpr unsigned p = unsigned(ct_mod(q));
            constexpr unsigned mm = unsigned((1ull << 32) / p), negp = 0u - p,
                               rs = unsigned((1ull << 30) % p);
            constexpr int planes = ct_planes(q), lutoff = ct_lutoff(q), pbase = ct_pbase(q);
            unsigned cd[8];
            if constexpr ((FX & 16) != 0 &&
                          planes < 3) { // the u8 copy: the residue is the byte offset
                constexpr int lut8off = ct_lut8off(q);
                const unsigned char *tab8 =
                    reinterpret_cast<const unsigned char *>(slut) + LutWords * 2 + lut8off;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    cd[j] =
                        tab8[bqn_asm(mad_asm(l8e_chr(ch[j], rs, negp, mm), rs, lo[j]), negp, mm)];
            } else {
                codes(cd, slut + lutoff, rs, negp, mm);
            }
            emit(cd, planes, tbase + (cs * unsigned(pbase) + rc * unsigned(planes)));
        });
    } else {
#pragma unroll 1
        for (int q = 0; q < L8_NMOD; ++q) {
            const ModC c = mods[q];
            unsigned cd[8];
            codes(cd, slut + c.lutoff, c.rs, c.negp, c.mmagic);
            emit(cd, c.planes, tbase + plane_offset(g.cuts, c.pbase, c.planes, rank, cut, 0));
        }
    }
}

// FX 16's LUT of one side: the Encoder's u16 LUT (the same entries, rebuilt from Encoder::codes as init does) followed by the
// u8 copy of the 1- and 2-plane moduli's entries (LutXBytes bytes)
inline std::vector<unsigned char> lut_x_host(const Encoder &e, int side_b) {
    if (e.a_direct || !ct_tables_match()) {
        throw std::runtime_error("l8e: lut_x is the 59-product form of l8_tables.h");
    }
    std::vector<unsigned char> h(LutXBytes, 0);
    for (int q = 0; q < L8_NMOD; ++q) {
        const unsigned p = unsigned(ct_mod(q));
        const long long rot = (long long)((1ull << 53) % p);
        for (unsigned xx = 0; xx < 2 * p; ++xx) {
            const int r = int((((long long)xx - rot) % (long long)p + p) % p);
            const unsigned c = e.codes(side_b, q, r);
            h[2 * (ct_lutoff(q) + xx)] = (unsigned char)(c & 255u);
            h[2 * (ct_lutoff(q) + xx) + 1] = (unsigned char)(c >> 8);
            if (ct_planes(q) < 3) {
                if (c > 255u) {
                    throw std::runtime_error("l8e: a 1- or 2-plane code above 8 bits");
                }
                h[LutWords * 2 + ct_lut8off(q) + xx] = (unsigned char)c;
            }
        }
    }
    return h;
}

// launch on an Encoder's tables (the 59-product form for CT); FX 16 needs lutx = that side's lut_x_host on the device
template <int PR, bool CT, int FX = 0, int MINB = 3>
inline void launch_x(const Encoder &e, const double *x, int64_t ld, const int *shifts,
                     unsigned char *packed, int side_b, cudaStream_t stream = 0,
                     const unsigned char *lutx = nullptr) {
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(encode_l8x<PR, CT, FX, MINB>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dyn_smem_x(0, FX)) != cudaSuccess) {
            throw std::runtime_error("l8e: encode_l8x attribute");
        }
        attr = true;
    }
    if (CT && (e.a_direct || !ct_tables_match())) {
        throw std::runtime_error("l8e: encode_l8x<CT> is the 59-product form of l8_tables.h");
    }
    if ((FX & 16) && !lutx) {
        throw std::runtime_error("l8e: FX 16 needs the lut_x of this side");
    }
    const unsigned short *tab =
        (FX & 16) ? reinterpret_cast<const unsigned short *>(lutx) : e.dlut[side_b];
    const dim3 grid(4, unsigned(e.geo.cuts), unsigned(e.tiles(side_b) * 2));
    encode_l8x<PR, CT, FX, MINB><<<grid, 256, dyn_smem_x(side_b, FX), stream>>>(
        x, ld, shifts, packed, e.dmods, tab, side_b, e.geo);
}

} // namespace l8e
