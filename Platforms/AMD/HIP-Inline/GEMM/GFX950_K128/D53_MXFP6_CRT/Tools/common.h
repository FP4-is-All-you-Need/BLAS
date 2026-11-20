#pragma once
// Host helpers shared by the test and the benchmark: input generation, exact references, ULP metrics.
#include "../quantize.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace awe_mxfp6 {
namespace host {

// Exact accumulator of binary64 products on a fixed 4352-bit grid (independent of the GPU path).
struct ExactDot {
    static constexpr int Words = 68, Base = -2148;
    using Big = std::array<uint64_t, Words>;
    Big positive{}, negative{};
    static uint64_t bits(double x) {
        uint64_t b;
        std::memcpy(&b, &x, 8);
        return b;
    }
    static void add_word(Big &a, int index, uint64_t v) {
        while (v) {
            if (index >= Words) {
                throw std::runtime_error("reference overflow");
            }
            const uint64_t before = a[index];
            a[index] += v;
            v = a[index] < before;
            ++index;
        }
    }
    void add(double a, double b) {
        const uint64_t aa = bits(a), bb = bits(b);
        const unsigned ea = (aa >> 52) & 2047, eb = (bb >> 52) & 2047;
        if (ea == 2047 || eb == 2047) {
            throw std::runtime_error("nonfinite exact-dot input");
        }
        const uint64_t ma = (aa & 0xfffffffffffffull) | (ea ? 1ull << 52 : 0);
        const uint64_t mb = (bb & 0xfffffffffffffull) | (eb ? 1ull << 52 : 0);
        const unsigned __int128 product = static_cast<unsigned __int128>(ma) * mb;
        const int exponent = (ea ? int(ea) - 1075 : -1074) + (eb ? int(eb) - 1075 : -1074);
        const int shift = exponent - Base, index = shift / 64, bit = shift % 64;
        Big &sum = ((aa ^ bb) >> 63) ? negative : positive;
        for (int j = 0; j < 2; ++j) {
            const uint64_t word = uint64_t(product >> (64 * j));
            add_word(sum, index + j, word << bit);
            if (bit) {
                add_word(sum, index + j + 1, word >> (64 - bit));
            }
        }
    }
    static bool get(const Big &a, int bit) {
        return bit >= 0 && bit < Words * 64 && ((a[bit / 64] >> (bit % 64)) & 1);
    }
    uint64_t rounded_bits() const {
        int cmp = 0;
        for (int i = Words - 1; i >= 0; --i) {
            if (positive[i] != negative[i]) {
                cmp = positive[i] > negative[i] ? 1 : -1;
                break;
            }
        }
        if (!cmp) {
            return 0;
        }
        const Big &large = cmp > 0 ? positive : negative, &small = cmp > 0 ? negative : positive;
        Big a{};
        uint64_t borrow = 0;
        for (int i = 0; i < Words; ++i) {
            const unsigned __int128 rhs = static_cast<unsigned __int128>(small[i]) + borrow;
            a[i] = uint64_t(static_cast<unsigned __int128>(large[i]) - rhs);
            borrow = static_cast<unsigned __int128>(large[i]) < rhs;
        }
        int top = Words * 64 - 1;
        while (top >= 0 && !get(a, top)) {
            --top;
        }
        int e = top + Base;
        const int shift = e >= -1022 ? top - 52 : -1074 - Base;
        uint64_t sig = 0;
        for (int i = 0; i < 54; ++i) {
            if (get(a, i + shift)) {
                sig |= uint64_t(1) << i;
            }
        }
        const bool guard = get(a, shift - 1);
        bool sticky = false;
        const int full = (shift - 1) / 64;
        for (int i = 0; i < full; ++i) {
            sticky |= a[i] != 0;
        }
        for (int i = full * 64; i < shift - 1; ++i) {
            sticky |= get(a, i);
        }
        if (guard && (sticky || (sig & 1))) {
            ++sig;
        }
        const uint64_t sign = cmp < 0 ? 1ull << 63 : 0;
        if (e < -1022) {
            return sign | sig;
        }
        if (sig == 1ull << 53) {
            sig >>= 1;
            ++e;
        }
        if (e > 1023) {
            return sign | 0x7ff0000000000000ull;
        }
        return sign | (uint64_t(e + 1023) << 52) | (sig & 0xfffffffffffffull);
    }
    double rounded() const {
        const uint64_t b = rounded_bits();
        double v;
        std::memcpy(&v, &b, 8);
        return v;
    }
};

// RNE(x 2^shift) 2^-shift as binary64: the value the engine multiplies.
inline double quantized_restored(double x, int shift) {
    const uint64_t raw = ExactDot::bits(x);
    const unsigned encoded = unsigned((raw >> 52) & 2047);
    if (encoded == 2047) {
        throw std::invalid_argument("quantization requires finite binary64 input");
    }
    uint64_t mantissa = raw & 0xfffffffffffffull;
    if (encoded) {
        mantissa |= uint64_t(1) << 52;
    }
    if (!mantissa) {
        return x;
    }
    const int exponent = encoded ? int(encoded) - 1075 : -1074;
    const int64_t h = int64_t(exponent) + shift;
    if (h >= 0) {
        return x;
    }
    uint64_t integer = 0;
    const int64_t distance = -h;
    if (distance < 54) {
        const unsigned d = unsigned(distance);
        integer = mantissa >> d;
        const uint64_t tail = mantissa & ((uint64_t(1) << d) - 1), midpoint = uint64_t(1)
                                                                              << (d - 1);
        if (tail > midpoint || (tail == midpoint && (integer & 1))) {
            ++integer;
        }
    }
    if (!integer) {
        return std::copysign(0.0, x);
    }
    const double restored = std::ldexp(double(integer), -shift);
    if (!std::isfinite(restored)) {
        throw std::overflow_error("restored quantized value overflows binary64");
    }
    return std::copysign(restored, x);
}

// Same shift rule as the device: (bits - 1) - exponent of the largest magnitude, 0 for an all-zero line.
inline int shift_of(double largest, int bits) {
    return largest ? bits - 1 - std::ilogb(largest) : 0;
}

struct Fixture {
    int m, n, k;
    std::vector<double> a, b; // column-major, lda = m, ldb = k
    std::vector<int> sa, sb;
};

// Deterministic inputs: uniform in [-1, 1), standard normal, or "wide" = (u - 0.5) exp(4 z).
inline Fixture make_fixture(int m, int n, int k, const std::string &distribution, uint64_t seed,
                            int bits) {
    Fixture f{m,
              n,
              k,
              std::vector<double>(size_t(m) * k),
              std::vector<double>(size_t(n) * k),
              std::vector<int>(m),
              std::vector<int>(n)};
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0, 1);
    auto uniform = [&]() { return double(rng() >> 11) * 0x1p-53; };
    auto value = [&]() {
        double x;
        if (distribution == "uniform") {
            x = 2 * uniform() - 1;
        } else if (distribution == "normal") {
            x = normal(rng);
        } else if (distribution == "wide") {
            x = (uniform() - .5) * std::exp(4 * normal(rng));
        } else {
            throw std::invalid_argument("distribution: uniform, normal or wide");
        }
        if (!std::isfinite(x)) {
            throw std::runtime_error("nonfinite fixture value");
        }
        return x;
    };
    for (auto &x : f.a) {
        x = value();
    }
    for (auto &x : f.b) {
        x = value();
    }
    for (int r = 0; r < m; ++r) {
        double mx = 0;
        for (int t = 0; t < k; ++t) {
            mx = std::max(mx, std::abs(f.a[size_t(t) * m + r]));
        }
        f.sa[r] = shift_of(mx, bits);
    }
    for (int c = 0; c < n; ++c) {
        double mx = 0;
        for (int t = 0; t < k; ++t) {
            mx = std::max(mx, std::abs(f.b[size_t(c) * k + t]));
        }
        f.sb[c] = shift_of(mx, bits);
    }
    return f;
}

struct Sample {
    uint64_t index;   // column-major index into C
    double original;  // exact dot of the original inputs, rounded to binary64
    double quantized; // exact dot of the quantized inputs, rounded to binary64
};

// rows x cols samples on an evenly spaced grid (32 x 32 = 1,024 for the accuracy criterion).
inline std::vector<Sample> sampled_references(const Fixture &f, int rows, int cols) {
    std::vector<Sample> refs;
    rows = std::min(rows, f.m);
    cols = std::min(cols, f.n);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const int r = rows == 1 ? 0 : i * (f.m - 1) / (rows - 1),
                      c = cols == 1 ? 0 : j * (f.n - 1) / (cols - 1);
            ExactDot original, quantized;
            for (int t = 0; t < f.k; ++t) {
                const double x = f.a[size_t(t) * f.m + r], y = f.b[size_t(c) * f.k + t];
                original.add(x, y);
                quantized.add(quantized_restored(x, f.sa[r]), quantized_restored(y, f.sb[c]));
            }
            refs.push_back({uint64_t(c) * f.m + r, original.rounded(), quantized.rounded()});
        }
    }
    return refs;
}

struct Metrics {
    double ulp_p95 = 0, ulp_max = 0;
};

inline double p95(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x.at(size_t(std::ceil(.95 * x.size())) - 1);
}

// ULP of each sample in the binade of the exact reference.
inline Metrics metrics(const std::vector<double> &got, const std::vector<Sample> &refs) {
    std::vector<double> u;
    for (size_t i = 0; i < refs.size(); ++i) {
        const double ref = refs[i].original, x = got[i];
        double ulp = INFINITY;
        if (std::isfinite(x) && std::isfinite(ref)) {
            const long double diff = std::abs((long double)x - ref);
            const double mag = std::abs(ref), next = std::nextafter(mag, INFINITY);
            const long double spacing = std::isfinite(next)
                                            ? (long double)next - mag
                                            : (long double)mag - std::nextafter(mag, 0.0);
            ulp = double(diff / spacing);
        }
        u.push_back(ulp);
    }
    return {p95(u), *std::max_element(u.begin(), u.end())};
}

// Accuracy criterion of the measurements: p95 and maximum sampled ULP no worse than native rocBLAS.
inline bool qualified(const Metrics &a, const Metrics &native) {
    return std::isfinite(a.ulp_max) && std::isfinite(native.ulp_max) &&
           a.ulp_p95 <= native.ulp_p95 && a.ulp_max <= native.ulp_max;
}

inline uint64_t bits(double x) {
    uint64_t r;
    std::memcpy(&r, &x, 8);
    return r;
}

inline void number(double x) {
    if (std::isfinite(x)) {
        std::cout << x;
    } else {
        std::cout << "null";
    }
}

inline double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x.at(x.size() / 2);
}

inline bool has_flag(int argc, char **argv, const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

inline std::string option(int argc, char **argv, const std::string &flag,
                          const std::string &fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == flag) {
            return argv[i + 1];
        }
    }
    return fallback;
}

} // namespace host
} // namespace awe_mxfp6
