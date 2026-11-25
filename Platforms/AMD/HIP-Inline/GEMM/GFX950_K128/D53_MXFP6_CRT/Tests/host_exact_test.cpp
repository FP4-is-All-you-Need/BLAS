// CPU test of the integer arithmetic behind the engine, without a GPU:
//   1. the table: product bounds, segment caps and the E2M3 code range;
//   2. the plane identity of every form: sum_j w_j a_j(r) b_j(s) = r s (mod p) for all residues;
//   3. the line shift (crt::shift_of) for normal, subnormal and zero maxima;
//   4. small shapes end to end on the host: quantize -> residues -> integer products per segment ->
//      residue update -> CRT -> binary64, compared bit for bit with the exact dot of the quantized
//      inputs. The device kernels use the same quantization, CRT and rounding (quantize.h, crt.h).
#include "../crt.h"
#include "../layout.h"
#include "../quantize.h"
#include "../Tools/common.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace awe_mxfp6;
namespace table = fp6_catalog_table;

static int failures = 0;
static void check(bool ok, const char *what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

// E2M3 code -> signed integer value in units of 1/8 (the scale 2^3 per operand makes every value an integer).
static int decode(unsigned code) {
    const int v = int(magnitude(code & 63u));
    return code & 32u ? -v : v;
}

static void test_table() {
    // Expected segment caps of the selected table (table_select.h).
#if AWE_MXFP6_TABLE_K > 4096
    // 29 products at K 8192, 1 at 4096 = 31 segments.
    static const unsigned expected_cap[30] = {
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};
#else
    // 14 products at K 8192, 15 at 4096 = 44 segments.
    static const unsigned expected_cap[29] = {
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096, 4096,
        4096, 8192, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096};
#endif
    const auto caps = segment_caps();
    unsigned segments = 0;
    for (unsigned pi = 0; pi < table::product_count; ++pi) {
        const auto bound = product_bound(pi);
        check(bound.max_a <= 60 && bound.max_b <= 60, "E2M3 magnitude bound");
        check(uint64_t(caps[pi]) <= bound.terms, "segment cap within the exact accumulator bound");
        check(caps[pi] == expected_cap[pi], "segment cap of this table");
        segments += 8192 / caps[pi];
    }
    check(segments == AWE_MXFP6_TABLE_SEGMENTS_AT_8192, "product segments at K = 8192");
    for (unsigned char code : table::host_lut) {
        check(code <= 63, "LUT entries are E2M3 codes");
    }
    unsigned expected_first = 0;
    for (unsigned fi = 0; fi < table::form_count; ++fi) {
        const auto f = table::host_forms[fi];
        check(f.first == expected_first && f.count >= 1 && f.count <= 3 && f.format == 2,
              "form layout");
        expected_first += f.count;
    }
    check(expected_first == table::product_count, "forms cover the products");
    check(packed_lut::build().size() == packed_lut::word_count(), "packed LUT size");
}

static void test_plane_identity() {
    for (unsigned fi = 0; fi < table::form_count; ++fi) {
        const auto f = table::host_forms[fi];
        bool ok = true;
        for (unsigned r = 0; r < f.p && ok; ++r) {
            for (unsigned s = 0; s < f.p && ok; ++s) {
                int64_t sum = 0;
                for (unsigned j = 0; j < f.count; ++j) {
                    const auto p = table::host_products[f.first + j];
                    sum += int64_t(p.weight) * decode(table::host_lut[p.lut_a + r]) *
                           decode(table::host_lut[p.lut_b + s]);
                }
                sum %= int64_t(f.p);
                if (sum < 0) {
                    sum += f.p;
                }
                ok = sum == int64_t(uint64_t(r) * s % f.p);
            }
        }
        check(ok, "plane identity of a form");
    }
}

// Host model of the device pipeline for one output element.
static double host_output(const host::Fixture &f, int r, int c, const std::vector<unsigned> &caps) {
    std::vector<uint16_t> residues(table::form_count);
    for (unsigned fi = 0; fi < table::form_count; ++fi) {
        const auto form = table::host_forms[fi];
        uint64_t residue = 0;
        for (unsigned pi = form.first; pi < form.first + form.count; ++pi) {
            const auto p = table::host_products[pi];
            const unsigned cap = caps[pi];
            for (int k0 = 0; k0 < f.k; k0 += int(cap)) {
                int64_t acc = 0;
                for (int t = k0; t < std::min(f.k, k0 + int(cap)); ++t) {
                    const Prepared qa = prepare(raw_binary64(f.a[size_t(t) * f.m + r]), f.sa[r]);
                    const Prepared qb = prepare(raw_binary64(f.b[size_t(c) * f.k + t]), f.sb[c]);
                    check(qa.valid && qb.valid, "quantization of a finite input");
                    // The kernels use residue<P> with P fixed per form; the runtime form is the same arithmetic.
                    const uint32_t ra = residue_runtime(qa, form.p).residue,
                                   rb = residue_runtime(qb, form.p).residue;
                    acc += int64_t(decode(table::host_lut[p.lut_a + ra])) *
                           decode(table::host_lut[p.lut_b + rb]);
                }
                check(acc < (int64_t(1) << 24) && acc > -(int64_t(1) << 24),
                      "segment accumulator below 2^24");
                int64_t rem = acc % int64_t(form.p);
                if (rem < 0) {
                    rem += form.p;
                }
                residue = (residue + uint64_t(p.weight % form.p) * uint64_t(rem)) % form.p;
            }
        }
        residues[fi] = uint16_t(residue);
    }
    return crt::reconstruct_element(residues.data(), 0, 1, -f.sa[r] - f.sb[c]);
}

static void test_small_shapes() {
    const auto caps = segment_caps();
    struct Case {
        int m, n, k;
        const char *distribution;
    };
    const Case cases[] = {{7, 5, 300, "uniform"},
                          {3, 4, 4097, "normal"},
                          {5, 3, 1000, "wide"},
                          {2, 2, 8192, "uniform"}};
    for (const auto &cs : cases) {
        host::Fixture f =
            host::make_fixture(cs.m, cs.n, cs.k, cs.distribution, 20261021, table::max_bits);
        // Exact cancellation and a zero row are part of every case.
        for (int t = 0; t < f.k; ++t) {
            f.a[size_t(t) * f.m] = 0;
        }
        f.sa[0] = 0;
        const auto refs = host::sampled_references(f, f.m, f.n);
        size_t mismatches = 0;
        for (const auto &s : refs) {
            const int r = int(s.index % f.m), c = int(s.index / f.m);
            const double got = host_output(f, r, c, caps);
            mismatches += host::bits(got) != host::bits(s.quantized);
        }
        std::printf("%s %dx%dx%d: %zu outputs, %zu mismatches against the exact quantized dot\n",
                    cs.distribution, cs.m, cs.n, cs.k, refs.size(), mismatches);
        check(mismatches == 0, "small-shape exact reconstruction");
    }
}

// The device residue<P> and the runtime reduction above must agree; sample a few moduli and inputs.
static void test_residue_forms() {
    std::mt19937_64 rng(7);
    for (int i = 0; i < 2000; ++i) {
        double x;
        const uint64_t raw = rng();
        std::memcpy(&x, &raw, 8);
        if (!std::isfinite(x)) {
            continue;
        }
        const int shift = int(rng() % 200) - 100;
        const Prepared q = prepare(raw, shift);
        if (!q.valid) {
            continue;
        }
        const Residue a = residue<1381>(q), b = residue_runtime(q, 1381), c = residue<32>(q),
                      d = residue_runtime(q, 32);
        check(a.valid && b.valid && a.residue == b.residue,
              "32-bit residue equals the 64-bit reduction (p = 1381)");
        check(c.valid && d.valid && c.residue == d.residue,
              "32-bit residue equals the 64-bit reduction (p = 32)");
    }
}

// The shift of a line comes from the raw bits of its largest magnitude; subnormal maxima must give
// the same shift as their true exponent, and an all-zero line gives 0.
static void test_shifts() {
    const double values[] = {1.0, 3.0, 0x1p-1022, 0x1p-1023, 1e-310, 0x1p-1074, 0x1.8p-1070, 1e300};
    for (const double v : values) {
        uint64_t raw;
        std::memcpy(&raw, &v, 8);
        check(crt::shift_of(raw, 53) == 52 - std::ilogb(v),
              "shift of a normal or subnormal maximum");
    }
    check(crt::shift_of(0, 53) == 0, "shift of an all-zero line");
}

int main() {
    test_table();
    test_plane_identity();
    test_residue_forms();
    test_shifts();
    test_small_shapes();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("host exact test passed\n");
    return 0;
}
