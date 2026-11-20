#pragma once
// Operand layout of the AITER MXFP6 GEMM kernel, the K segmentation of each product, and the two
// lookup tables the encoder reads (one packed word per residue for 2- and 3-product forms, one byte
// per residue for single-product forms).
#include "common.h"
#include "table_d53_k8192.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace awe_mxfp6 {
namespace table = fp6_catalog_table;

static_assert(table::form_count == 19 && table::product_count == 29 && table::max_bits == 53 &&
                  table::max_k == 8192,
              "this engine is written for the 19-modulus, 29-product table");

// ---- AITER operand layout -------------------------------------------------------------------
// The kernel reads A and B as 256-row tiles of 128-element K stages. Each stage holds 1024 groups of
// 32 FP6 values (24 bytes) stored as 16 + 8 bytes in two halves, plus one E8M0 scale byte per group.
// Two zero-filled stages follow the last real stage because the kernel's prefetch reads past K.
// A segment of K columns (cap = 4096 or 8192) of one product is one such operand plane per side.
struct ProductOperands {
    uint32_t *data[2]; // plane data per K segment (at most 2 segments at K = 8192)
    uint8_t *scale[2]; // plane scale bytes per K segment
    uint32_t cap;      // K columns per segment
    uint32_t segments; // 8192 / cap
};
static_assert(sizeof(ProductOperands) == 40 && offsetof(ProductOperands, cap) == 32,
              "device copy of the table");

// Encoder shape argument, passed by value. The encoder reads only outer and k. The other fields are
// not read; the 80-byte layout fixes the argument offsets, which the compiled kernel depends on.
struct EncodeShape {
    uint64_t outer = 0, k = 0, products = 0, padded_outer = 0, padded_full_k = 0;
    uint64_t segments = 0, product_words = 0, total_words = 0, grid_blocks = 0;
    bool side_b = false;
};
static_assert(sizeof(EncodeShape) == 80 && offsetof(EncodeShape, k) == 8,
              "encoder argument layout");

constexpr unsigned rows = 8192; // M = N = K = 8192 is the only shape this engine handles
constexpr unsigned row_tiles = rows / 256;

AWE_MXFP6_HD constexpr uint64_t plane_words(unsigned cap) {
    return uint64_t(row_tiles) * (cap / 128 + 2) * 6144;
}
AWE_MXFP6_HD constexpr uint64_t plane_scale_bytes(unsigned cap) {
    return uint64_t(row_tiles) * (cap / 128 + 2) * 1024;
}
AWE_MXFP6_HD constexpr unsigned group(unsigned row, unsigned k) {
    return ((row % 256) / 16) * 64 + ((k % 128) / 32) * 16 + row % 16;
}
AWE_MXFP6_HD constexpr uint64_t word_index(unsigned row, unsigned k, unsigned w, unsigned cap) {
    const uint64_t base = (uint64_t(row / 256) * (cap / 128 + 2) + k / 128) * 6144;
    const unsigned g = group(row, k), j = 3 * ((k % 32) / 16) + w;
    return j < 4 ? base + g * 4 + j : base + 4096 + g * 2 + j - 4;
}
AWE_MXFP6_HD constexpr uint64_t scale_index(unsigned row, unsigned k, unsigned cap) {
    return (uint64_t(row / 256) * (cap / 128 + 2) + k / 128) * 1024 + group(row, k);
}

// Store one packet of 16 FP6 codes (3 words) for (row, global_k). The scale byte 130 is the E8M0
// value 2^3. E2M3 values are integers times 2^-3, so with this scale every operand value is an
// integer and every product term of the FP32 accumulation is an integer. The thread that owns
// k < 128 also zeroes the two trailing stages of its packet position so no stale data reaches the
// prefetch.
AWE_MXFP6_HD inline void store_packet(const ProductOperands &out, unsigned row, uint64_t global_k,
                                      const uint32_t *words) {
    const unsigned seg = unsigned(global_k >> (out.cap == 4096 ? 12 : 13)),
                   k = unsigned(global_k & (out.cap - 1u));
#pragma unroll
    for (unsigned w = 0; w < 3; ++w) {
        out.data[seg][word_index(row, k, w, out.cap)] = words[w];
    }
    if ((k & 31u) == 0) {
        out.scale[seg][scale_index(row, k, out.cap)] = 130;
    }
    if (k < 128) {
#pragma unroll
        for (unsigned pad = 0; pad < 2; ++pad) {
            const unsigned pk = out.cap + 128 * pad + k;
#pragma unroll
            for (unsigned w = 0; w < 3; ++w) {
                out.data[seg][word_index(row, pk, w, out.cap)] = 0;
            }
            if ((k & 31u) == 0) {
                out.scale[seg][scale_index(row, pk, out.cap)] = 127;
            }
        }
    }
}

// Pack a 6-bit code into the packet words at position t (0..15).
AWE_MXFP6_HD inline void insert(uint32_t *words, unsigned t, uint32_t code) {
    const unsigned bit = t * 6, w = bit / 32, s = bit % 32;
    words[w] |= (code & 63u) << s;
    if (s > 26) {
        words[w + 1] |= (code & 63u) >> (32 - s);
    }
}

// ---- Product bounds and K segmentation ------------------------------------------------------
// The FP32 accumulator of the kernel must stay an exact integer, so a product whose largest |code|
// values are max_a and max_b may sum at most (2^24 - 1) / (max_a * max_b) terms per segment.
inline unsigned magnitude(unsigned code) {
    if (code >= 64) {
        throw std::invalid_argument("E2M3 code range");
    }
    const unsigned f = code & 7, e = (code >> 3) & 3;
    return e ? (8 + f) << (e - 1) : f;
}

struct ProductBound {
    unsigned max_a = 0, max_b = 0;
    uint64_t terms = 0;
};

inline ProductBound product_bound(unsigned pi) {
    if (pi >= table::product_count) {
        throw std::invalid_argument("product index");
    }
    const auto p = table::host_products[pi];
    ProductBound b;
    if (p.bits != 6 || p.p < 2 || p.p > 65535 ||
        uint64_t(p.lut_a) + p.p > sizeof(table::host_lut) ||
        uint64_t(p.lut_b) + p.p > sizeof(table::host_lut)) {
        throw std::invalid_argument("product LUT range");
    }
    for (unsigned r = 0; r < p.p; ++r) {
        const unsigned a = magnitude(table::host_lut[p.lut_a + r]),
                       c = magnitude(table::host_lut[p.lut_b + r]);
        if (a > b.max_a) {
            b.max_a = a;
        }
        if (c > b.max_b) {
            b.max_b = c;
        }
    }
    const uint64_t term = uint64_t(b.max_a) * b.max_b;
    b.terms = term ? ((uint64_t(1) << 24) - 1) / term : UINT64_MAX;
    return b;
}

// Largest segment the kernel supports that keeps the accumulator exact; 0 when even 4096 is unsafe.
inline unsigned select_cap(uint64_t safe_terms) {
    return safe_terms >= 16384 ? 16384 : safe_terms >= 8192 ? 8192 : safe_terms >= 4096 ? 4096 : 0;
}

// Segment caps at K = 8192 (the kernel accepts K = 4096 or 8192 per launch).
inline std::vector<unsigned> segment_caps() {
    std::vector<unsigned> caps(table::product_count);
    for (unsigned pi = 0; pi < table::product_count; ++pi) {
        const unsigned cap = select_cap(product_bound(pi).terms);
        if (!cap) {
            throw std::invalid_argument("a product has no exact 4096 segment");
        }
        caps[pi] = cap < 8192 ? cap : 8192;
    }
    return caps;
}

// ---- Single-product LUT (420 bytes, copied to LDS per side) ---------------------------------
namespace single_lut {
template <unsigned Fi, unsigned J = 0> AWE_MXFP6_HD constexpr unsigned offset() {
    if constexpr (J < Fi) {
        constexpr auto f = table::host_forms[J];
        return (f.count == 1 ? f.p : 0) + offset<Fi, J + 1>();
    } else {
        return 0;
    }
}
inline constexpr unsigned bytes = offset<table::form_count>(),
                          allocation_bytes = (bytes + 15u) & ~15u;
static_assert(bytes == 420 && allocation_bytes == 432, "single-product forms of the public table");
template <bool SideB, unsigned Fi = 0> AWE_MXFP6_HD constexpr unsigned source_index(unsigned i) {
    if constexpr (Fi < table::form_count) {
        constexpr auto f = table::host_forms[Fi];
        if constexpr (f.count == 1) {
            constexpr auto p = table::host_products[f.first];
            if (i < f.p) {
                return (SideB ? p.lut_b : p.lut_a) + i;
            }
            i -= f.p;
        }
        return source_index<SideB, Fi + 1>(i);
    } else {
        return 0;
    }
}
} // namespace single_lut

// ---- Packed LUT for 2- and 3-product forms (one 32-bit word per residue and side) -------------
namespace packed_lut {
template <unsigned Fi> struct Prefix {
    static_assert(Fi <= table::form_count);
    static constexpr size_t value =
        Prefix<Fi - 1>::value +
        (table::host_forms[Fi - 1].count >= 2 ? table::host_forms[Fi - 1].p : 0);
};
template <> struct Prefix<0> {
    static constexpr size_t value = 0;
};
AWE_MXFP6_HD constexpr size_t side_words() {
    return Prefix<table::form_count>::value;
}
AWE_MXFP6_HD constexpr size_t word_count() {
    return 2 * side_words();
}
inline size_t offset(bool side_b, unsigned fi) {
    size_t out = side_b ? side_words() : 0;
    for (unsigned f = 0; f < fi; ++f) {
        if (table::host_forms[f].count >= 2) {
            out += table::host_forms[f].p;
        }
    }
    return out;
}
template <bool SideB, unsigned Fi> AWE_MXFP6_HD constexpr size_t offset() {
    static_assert(table::host_forms[Fi].count >= 2);
    return (SideB ? side_words() : 0) + Prefix<Fi>::value;
}
static_assert(word_count() == 11368, "packed LUT footprint of the public table");

// Host construction of the packed table from host_lut; byte j of a word is product j's code.
inline std::vector<uint32_t> build() {
    std::vector<uint32_t> words(word_count(), 0);
    for (unsigned side = 0; side < 2; ++side) {
        for (unsigned fi = 0; fi < table::form_count; ++fi) {
            const auto f = table::host_forms[fi];
            if (f.count < 1 || f.count > 3) {
                throw std::runtime_error("unsupported form product count");
            }
            if (f.count < 2) {
                continue;
            }
            for (unsigned r = 0; r < f.p; ++r) {
                for (unsigned j = 0; j < f.count; ++j) {
                    const auto p = table::host_products[f.first + j];
                    if (p.p != f.p || p.bits != 6) {
                        throw std::runtime_error("form/product binding");
                    }
                    const size_t at = (side ? p.lut_b : p.lut_a) + size_t(r);
                    if (at >= sizeof(table::host_lut)) {
                        throw std::runtime_error("raw LUT bounds");
                    }
                    words[offset(side, fi) + r] |= uint32_t(table::host_lut[at]) << (8 * j);
                }
            }
        }
    }
    return words;
}
} // namespace packed_lut

} // namespace awe_mxfp6
