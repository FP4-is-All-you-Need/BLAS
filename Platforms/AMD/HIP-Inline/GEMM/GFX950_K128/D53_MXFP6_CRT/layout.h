#pragma once
// Operand layout of the AITER MXFP6 GEMM kernel, the K segmentation of each product, and the two
// lookup tables the encoder reads (one packed word per residue for 2- and 3-product forms, one byte
// per residue for single-product forms).
#include "common.h"
#include "table_select.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace awe_mxfp6 {
namespace table = fp6_catalog_table;

static_assert(table::form_count == 19 && table::product_count == AWE_MXFP6_TABLE_PRODUCTS &&
                  table::max_bits == 53 && table::max_k == 8192,
              "this engine is written for the 19-modulus tables of table_select.h");

// ---- AITER operand layout -------------------------------------------------------------------
// The kernel reads A and B as 256-row tiles of 128-element K stages. Each stage holds 1024 groups of
// 32 FP6 values (24 bytes) stored as 16 + 8 bytes in two halves. Matrix fused multiply-add uses constant E8M0 scale 130.
// Two zero-filled stages follow the last real stage because the kernel's prefetch reads past K.
// A segment of K columns (cap = 4096 or 8192) of one product is one such operand plane per side.
struct ProductOperands {
    uint32_t *data[2]; // plane data per K segment (at most 2 segments at K = 8192)
    uint32_t cap;      // K columns per segment
    uint32_t segments; // 8192 / cap
};
static_assert(sizeof(ProductOperands) == 24 && offsetof(ProductOperands, cap) == 16,
              "device copy of the table");

constexpr unsigned rows = 8192; // M = N = K = 8192 is the only shape this engine handles
constexpr unsigned row_tiles = rows / 256;

AWE_MXFP6_HD constexpr uint64_t plane_words(unsigned cap) {
    return uint64_t(row_tiles) * (cap / 128 + 2) * 6144;
}
AWE_MXFP6_HD constexpr unsigned group(unsigned row, unsigned k) {
    return ((row % 256) / 16) * 64 + ((k % 128) / 32) * 16 + row % 16;
}
AWE_MXFP6_HD constexpr uint64_t word_index(unsigned row, unsigned k, unsigned w, unsigned cap) {
    const uint64_t base = (uint64_t(row / 256) * (cap / 128 + 2) + k / 128) * 6144;
    const unsigned g = group(row, k), j = 3 * ((k % 32) / 16) + w;
    return j < 4 ? base + g * 4 + j : base + 4096 + g * 2 + j - 4;
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

// Segment caps at K = 8192 (each record covers a K segment of 4096 or 8192).
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
static_assert(word_count() == AWE_MXFP6_TABLE_WORDS, "packed LUT footprint of the selected table");

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
