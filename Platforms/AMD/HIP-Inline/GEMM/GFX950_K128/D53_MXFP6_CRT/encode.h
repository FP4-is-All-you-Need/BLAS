#pragma once
// Encoder: FP64 input -> residues modulo every form -> FP6 codes of every product plane, written
// directly in the AITER operand layout. One block handles a 32 (rows) x 128 (K) tile; each thread
// quantizes 16 inputs once, then the 19 forms are processed in pairs through LDS so the packet writers
// read 16 consecutive K residues of one row while the quantization stays coalesced along the input.
#include "common.h"
#include "layout.h"
#include "quantize.h"

namespace awe_mxfp6 {
namespace encode {

// Residues of the pair of forms (Fi, Fi+1) for one input; 1 << 16 marks an inactive or invalid input.
template <unsigned Fi, unsigned Group, unsigned J = 0>
__device__ __forceinline__ void tiled_residues_flag(uint32_t (&tile)[Group][32][129], unsigned row,
                                                    unsigned kk, const Prepared &q, bool active,
                                                    uint32_t &bad) {
    if constexpr (J < Group && Fi + J < table::form_count) {
        constexpr auto f = table::host_forms[Fi + J];
        const auto v = residue<f.p>(q);
        const bool ok = active && v.valid;
        bad |= active && !v.valid;
        tile[J][row][kk] = ok ? v.residue : (1u << 16);
        tiled_residues_flag<Fi, Group, J + 1>(tile, row, kk, q, active, bad);
    }
}

// At most one atomic OR per input and pair of forms.
template <unsigned Fi, unsigned Group>
__device__ __forceinline__ void tiled_residues(uint32_t (&tile)[Group][32][129], unsigned row,
                                               unsigned kk, const Prepared &q, bool active,
                                               int *invalid) {
    uint32_t bad = 0;
    tiled_residues_flag<Fi, Group>(tile, row, kk, q, active, bad);
    if (bad) {
        atomicOr(invalid, 1);
    }
}

// Codes of a 2- or 3-product form come from one packed word; byte j belongs to product j.
template <unsigned Fi, unsigned J = 0>
__device__ __forceinline__ void pack_products(uint32_t (&words)[table::host_forms[Fi].count][8],
                                              unsigned t, uint32_t codes, uint32_t &bad) {
    static_assert(table::host_forms[Fi].count >= 2 && table::host_forms[Fi].count <= 3);
    if constexpr (J < table::host_forms[Fi].count) {
        constexpr auto f = table::host_forms[Fi];
        constexpr auto p = table::host_products[f.first + J];
        static_assert(p.p == f.p && p.bits == 6, "E2M3 product binding");
        uint32_t code = (codes >> (8 * J)) & 255u;
        const bool over = code > 63;
        bad |= over;
        code = over ? 0u : code;
        insert(words[J], t, code);
        pack_products<Fi, J + 1>(words, t, codes, bad);
    }
}

// Write the packets of the pair of forms for 16 consecutive K values of one row.
template <bool SideB, unsigned Fi, unsigned Group, unsigned J = 0>
__device__ __forceinline__ void tiled_packets(const uint32_t (&tile)[Group][32][129], unsigned row,
                                              unsigned kk, uint64_t outer, uint64_t global_k,
                                              const ProductOperands *outputs, int *invalid,
                                              const uint32_t *packed, const uint8_t *single) {
    if constexpr (J < Group && Fi + J < table::form_count) {
        constexpr auto f = table::host_forms[Fi + J];
        // 16 codes of 6 bits fill 3 words. The array has 8 words: its size sets the register
        // allocation of this kernel.
        uint32_t words[f.count][8]{};
        uint32_t bad = 0;
#pragma unroll 16
        for (unsigned t = 0; t < 16; ++t) {
            const uint32_t r = tile[J][row][kk + t];
            const bool valid = r < (1u << 16);
            if constexpr (f.count >= 2) {
                constexpr size_t base = packed_lut::offset<SideB, Fi + J>();
                const uint32_t looked = packed[base + (r & 65535u)];
                pack_products<Fi + J>(words, t, valid ? looked : 0u, bad);
            } else {
                const uint32_t looked = single[single_lut::offset<Fi + J>() + (r & 65535u)];
                uint32_t code = valid ? looked : 0u;
                const bool over = code > 63;
                bad |= over;
                code = over ? 0u : code;
                insert(words[0], t, code);
            }
        }
        if (bad) {
            atomicOr(invalid, 4);
        }
#pragma unroll
        for (unsigned p = 0; p < f.count; ++p) {
            store_packet(outputs[f.first + p], unsigned(outer), global_k, words[p]);
        }
        tiled_packets<SideB, Fi, Group, J + 1>(tile, row, kk, outer, global_k, outputs, invalid,
                                               packed, single);
    }
}

template <bool SideB, unsigned Fi = 0>
__device__ __forceinline__ void
allform_pairs(const Prepared (&prepared)[16], const bool (&active)[16],
              uint32_t (&tile)[2][32][129], uint64_t o0, uint64_t k0,
              const ProductOperands *outputs, int *invalid, const uint32_t *packed,
              const uint8_t *single) {
    if constexpr (Fi < table::form_count) {
#pragma unroll 16
        for (unsigned q = 0; q < 16; ++q) {
            const unsigned linear = threadIdx.x + 256 * q;
            const unsigned row = SideB ? linear / 128 : linear % 32,
                           kk = SideB ? linear % 128 : linear / 32;
            tiled_residues<Fi, 2>(tile, row, kk, prepared[q], active[q], invalid);
        }
        __syncthreads();
        // Packet reader ownership: lanes 0..127 take rows 0..15, 128..255 rows 16..31; within a half,
        // consecutive lane pairs share a row and take the two 16-wide halves of a 32-wide K group,
        // so the stores of one group are contiguous.
        const unsigned lane = threadIdx.x;
        const unsigned row = (lane / 128) * 16 + (lane % 32) / 2;
        const unsigned kk = ((lane % 128) / 32) * 32 + (lane % 2) * 16;
        tiled_packets<SideB, Fi, 2>(tile, row, kk, o0 + row, k0 + kk, outputs, invalid, packed,
                                    single);
        // Every packet reader finishes before the next pair overwrites the LDS tile.
        __syncthreads();
        allform_pairs<SideB, Fi + 2>(prepared, active, tile, o0, k0, outputs, invalid, packed,
                                     single);
    }
}

// input: column-major A (M x K, element (o, g) at g * ld + o) or B (K x N, element (g, o) at o * ld + g).
// shifts: per row of A or column of B. outputs: the 29 product descriptors of this side.
// lut: host_lut on the device; packed: the packed LUT; invalid: OR of error flags (1 quantization, 4 LUT).
// shape: only outer and k are read. caps is not read; the argument keeps the kernel argument
// layout.
// Launch bounds (256, 4): 4 blocks per CU keep the encoder's many stores in flight.
template <bool SideB>
__global__ __launch_bounds__(256, 4) void encode_kernel(
    const double *input, const int *shifts, const ProductOperands *outputs, const uint8_t *lut,
    int *invalid, EncodeShape shape, int ld, const unsigned *caps, const uint32_t *packed) {
    __shared__ uint32_t tile[2][32][129];
    alignas(16) __shared__ uint8_t single[single_lut::allocation_bytes];
    alignas(16) __shared__ ProductOperands cached_outputs[table::product_count];
    for (unsigned i = threadIdx.x; i < single_lut::bytes; i += 256) {
        single[i] = lut[single_lut::source_index<SideB>(i)];
    }
    if (threadIdx.x < table::product_count) {
        cached_outputs[threadIdx.x] = outputs[threadIdx.x];
    }
    __syncthreads();
    const uint64_t o0 = uint64_t(blockIdx.x) * 32, k0 = uint64_t(blockIdx.y) * 128;
    Prepared prepared[16]{};
    bool active[16]{};
#pragma unroll 16
    for (unsigned q = 0; q < 16; ++q) {
        const unsigned linear = threadIdx.x + 256 * q;
        const unsigned row = SideB ? linear / 128 : linear % 32,
                       kk = SideB ? linear % 128 : linear / 32;
        const uint64_t o = o0 + row, g = k0 + kk;
        active[q] = o < shape.outer && g < shape.k;
        if (active[q]) {
            prepared[q] =
                prepare(raw_binary64(input[SideB ? o * uint64_t(ld) + g : g * uint64_t(ld) + o]),
                        shifts[o]);
        }
    }
    allform_pairs<SideB>(prepared, active, tile, o0, k0, cached_outputs, invalid, packed, single);
}

} // namespace encode
} // namespace awe_mxfp6
