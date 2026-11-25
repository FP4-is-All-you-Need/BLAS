#pragma once
#include "layout.h"
#include "quantize.h"

namespace awe_mxfp6::encode {

inline void prepare_prefetch(const ProductOperands &out, hipStream_t stream) {
    // Each 256-row tile has two zero K stages for the three-stage prefetch.
    const size_t stages = out.cap / 128;
    const size_t stage_bytes = 6144 * sizeof(uint32_t);
    for (unsigned s = 0; s < out.segments; ++s) {
        auto *data = reinterpret_cast<uint8_t *>(out.data[s]);
        hip_check(hipMemset2DAsync(data + stages * stage_bytes, (stages + 2) * stage_bytes,
                                   0, 2 * stage_bytes, row_tiles, stream));
    }
}

struct Digits {
    uint32_t lo, mid, hi;
};

template <unsigned P>
__device__ __forceinline__ unsigned reduce(Digits d) {
    // Reverse the 2^53 bias after reducing the three base-2^18 digits.
    constexpr unsigned r = (1u << 18) % P, rr = r * r % P;
    constexpr unsigned bias_correction = (P - (uint64_t(1) << 53) % P) % P;
    static_assert(uint64_t(0x3ffff) * (1 + r + rr) + bias_correction <= UINT32_MAX);
    return (d.lo + d.mid * r + d.hi * rr + bias_correction) % P;
}

template <bool SideB, unsigned Fi>
__device__ __forceinline__ void write_forms(const Digits (&digits)[4], unsigned row,
                                            unsigned packet_k, const ProductOperands *outputs,
                                            const uint8_t *lut, const uint32_t *packed) {
    if constexpr (Fi < table::form_count) {
        constexpr auto f = table::host_forms[Fi];
        unsigned chunks[f.count]{};
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            unsigned v = reduce<f.p>(digits[j]);
            unsigned codes;
            if constexpr (f.count == 1) {
                constexpr auto p = table::host_products[f.first];
                codes = lut[(SideB ? p.lut_b : p.lut_a) + v];
            } else {
                codes = packed[packed_lut::offset<SideB, Fi>() + v];
            }
#pragma unroll
            for (unsigned p = 0; p < f.count; ++p)
                chunks[p] |= ((codes >> (8 * p)) & 63u) << (6 * j);
        }
        // For each row the four lanes 16 apart hold 24 bits each. The first three
        // in each four-lane set concatenate adjacent chunks into three dwords.
        const unsigned w = (threadIdx.x >> 4) & 3;
#pragma unroll
        for (unsigned p = 0; p < f.count; ++p) {
            const unsigned a = chunks[p], b = __shfl_down(a, 16, 64);
            if (w < 3) {
                const auto &out = outputs[f.first + p];
                const unsigned k = packet_k & (out.cap - 1);
                out.data[packet_k / out.cap][word_index(row, k, w, out.cap)] =
                    (a >> (8 * w)) | (b << (24 - 8 * w));
            }
        }
        write_forms<SideB, Fi + 1>(digits, row, packet_k, outputs, lut, packed);
    }
}

template <bool SideB>
__global__ __launch_bounds__(256) void encode_tile(const double *input, const int *shifts,
                                                   const ProductOperands *outputs, int ld,
                                                   int *invalid, const uint8_t *lut,
                                                   const uint32_t *packed) {
    __shared__ uint64_t tile[16][65];
    const unsigned row0 = blockIdx.x * 16, k0 = blockIdx.y * 64;
    const unsigned lane = threadIdx.x;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned at = lane + 256 * i;
        const unsigned r = SideB ? at / 64 : at % 16, k = SideB ? at % 64 : at / 16;
        const unsigned row = row0 + r, g = k0 + k;
        const auto q = prepare(raw_binary64(input[SideB ? size_t(row) * ld + g : size_t(g) * ld + row]),
                               shifts[row]);
        // The row or column shift bounds |q| by 2^53 - 1.
        const uint64_t mag = (uint64_t(q.words[1]) << 32) | q.words[0];
        if (!q.valid || q.words[2] || (mag >> 53)) atomicOr(invalid, 1);
        tile[r][k] = (uint64_t(1) << 53) + (q.negative ? -mag : mag);
    }
    __syncthreads();
    const unsigned r = lane % 16, k = (lane / 16) * 4;
    Digits digits[4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        uint64_t x = tile[r][k + i];
        digits[i] = {unsigned(x) & 0x3ffff, unsigned(x >> 18) & 0x3ffff, unsigned(x >> 36)};
    }
    write_forms<SideB, 0>(digits, row0 + r, k0 + (k & ~15u), outputs, lut, packed);
}

} // namespace awe_mxfp6::encode
