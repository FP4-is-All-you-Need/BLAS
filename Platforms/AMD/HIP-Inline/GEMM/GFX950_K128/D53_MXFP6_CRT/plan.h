#pragma once
// FP64 GEMM C = A B (8192 x 8192 x 8192, column-major, alpha = 1, beta = 0) on MXFP6 Matrix Cores:
//   1. shifts: one power-of-two scale per row of A and per column of B (53-bit integer grid)
//   2. encode: residues of every quantized input modulo the 19 moduli, as FP6 product planes
//   3. products: 44 MXFP6 GEMM launches (29 products, K split in 4096/8192 segments); each launch
//      folds its FP32 accumulator into the uint16 residue plane of its modulus
//   4. CRT: every output is reconstructed from its 19 residues and rounded once to binary64
// The result equals the exactly rounded dot product of the quantized inputs.
#include "aiter_module.h"
#include "common.h"
#include "crt.h"
#include "encode.h"
#include "layout.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace awe_mxfp6 {

struct Timings {
    float shift = 0, encode = 0, products = 0, crt = 0,
          total = 0; // milliseconds between stream events
};

class Dgemm8192 {
    static constexpr int m_ = rows, n_ = rows, k_ = rows, bits_ = table::max_bits;
    hipStream_t stream_;
    Module module_;
    std::vector<unsigned> caps_;
    Memory sa_, sb_, invalid_, shift_max_, residues_, lut_, packed_lut_, table_a_, table_b_,
        caps_device_;
    std::vector<std::unique_ptr<Memory>> planes_;
    std::array<ProductOperands, table::product_count> ha_{}, hb_{};
    hipEvent_t events_[5]{};
    size_t bytes_ = 0;

    Memory *keep(size_t bytes) {
        planes_.push_back(std::make_unique<Memory>(bytes));
        bytes_ += bytes;
        return planes_.back().get();
    }
    static void check_crt_coverage() {
        // The centred CRT result must hold any |C| <= K 2^(2 bits): require 2 K 2^(2 bits) < P.
        table::Big unit;
        unit.w[(2 * bits_) / 32] = uint32_t(1) << (2 * bits_ % 32);
        const table::Big bound = awe_wide::add_multiply(table::Big{}, unit, 2u * unsigned(k_));
        require(awe_wide::compare(table::modulus_product, bound) > 0,
                "modulus product does not cover K");
    }
    // Encoder shape of one side (rows of A or columns of B; both are padded to whole 256-row tiles).
    static EncodeShape encode_shape(int outer) {
        EncodeShape s;
        s.outer = uint64_t(outer);
        s.k = uint64_t(k_);
        s.products = table::product_count;
        s.padded_outer = (s.outer + 255) / 256 * 256;
        s.padded_full_k = (s.k + 255) / 256 * 256;
        s.segments = (s.k + 4095) / 4096;
        s.product_words = s.padded_outer * (s.padded_full_k / 16) * 3;
        s.total_words = s.products * s.product_words;
        s.grid_blocks = (s.padded_outer / 16) * (s.padded_full_k / 128);
        return s;
    }

  public:
    Dgemm8192(hipStream_t stream, const std::string &module_path)
        : stream_(stream), module_(module_path), caps_(segment_caps()) {
        hipDeviceProp_t prop{};
        int device = 0;
        hip_check(hipGetDevice(&device));
        hip_check(hipGetDeviceProperties(&prop, device));
        require(std::string(prop.gcnArchName).rfind("gfx950", 0) == 0 && prop.warpSize == 64,
                "gfx950 (wave64) required");
        check_crt_coverage();
        for (unsigned pi = 0; pi < table::product_count; ++pi) {
            const auto p = table::host_products[pi];
            require(p.bits == 6 && p.p >= 2 && p.p <= 65535, "product table");
        }
        const size_t count = size_t(m_) * n_;
        sa_ = Memory(size_t(m_) * 4);
        sb_ = Memory(size_t(n_) * 4);
        invalid_ = Memory(4);
        shift_max_ = Memory(size_t(m_) * 8);
        residues_ = Memory(size_t(table::form_count) * count * 2);
        lut_ = Memory(sizeof(table::host_lut));
        hip_check(
            hipMemcpy(lut_.p, table::host_lut, sizeof(table::host_lut), hipMemcpyHostToDevice));
        const auto packed = packed_lut::build();
        packed_lut_ = Memory(packed.size() * 4);
        hip_check(
            hipMemcpy(packed_lut_.p, packed.data(), packed.size() * 4, hipMemcpyHostToDevice));
        caps_device_ = Memory(caps_.size() * sizeof(unsigned));
        hip_check(hipMemcpy(caps_device_.p, caps_.data(), caps_.size() * sizeof(unsigned),
                            hipMemcpyHostToDevice));
        bytes_ = size_t(m_ + n_) * 4 + 4 + size_t(m_) * 8 + size_t(table::form_count) * count * 2 +
                 sizeof(table::host_lut) + packed.size() * 4;
        for (unsigned pi = 0; pi < table::product_count; ++pi) {
            const unsigned cap = caps_[pi], segments = unsigned(k_) / cap;
            ha_[pi].cap = hb_[pi].cap = cap;
            ha_[pi].segments = hb_[pi].segments = segments;
            for (unsigned s = 0; s < segments; ++s) {
                ha_[pi].data[s] = keep(plane_words(cap) * 4)->as<uint32_t>();
                hb_[pi].data[s] = keep(plane_words(cap) * 4)->as<uint32_t>();
                ha_[pi].scale[s] = keep(plane_scale_bytes(cap))->as<uint8_t>();
                hb_[pi].scale[s] = keep(plane_scale_bytes(cap))->as<uint8_t>();
            }
        }
        table_a_ = Memory(sizeof(ha_));
        table_b_ = Memory(sizeof(hb_));
        hip_check(hipMemcpy(table_a_.p, ha_.data(), sizeof(ha_), hipMemcpyHostToDevice));
        hip_check(hipMemcpy(table_b_.p, hb_.data(), sizeof(hb_), hipMemcpyHostToDevice));
        bytes_ += sizeof(ha_) + sizeof(hb_);
        for (auto &event : events_) {
            hip_check(hipEventCreate(&event));
        }
    }
    // Waits for queued work before the buffers are freed; a destructor cannot report the status.
    ~Dgemm8192() {
        (void)hipStreamSynchronize(stream_);
        for (auto event : events_) {
            if (event) {
                (void)hipEventDestroy(event);
            }
        }
    }
    Dgemm8192(const Dgemm8192 &) = delete;
    Dgemm8192 &operator=(const Dgemm8192 &) = delete;

    static constexpr int m() {
        return m_;
    }
    static constexpr int n() {
        return n_;
    }
    static constexpr int k() {
        return k_;
    }
    size_t device_bytes() const {
        return bytes_;
    }
    unsigned launches() const {
        unsigned calls = 0;
        for (unsigned cap : caps_) {
            calls += unsigned(k_) / cap;
        }
        return calls;
    }
    const uint16_t *residues() const {
        return residues_.as<uint16_t>();
    }

    // All work is enqueued on the construction stream; the call returns after the stream completes.
    Timings run(const double *a, const double *b, double *c, int lda, int ldb, int ldc) {
        require(a && b && c && lda >= m_ && ldb >= k_ && ldc >= m_,
                "pointers and leading dimensions");
        const size_t count = size_t(m_) * n_;
        hip_check(hipEventRecord(events_[0], stream_));
        hip_check(hipMemsetAsync(invalid_.p, 0, 4, stream_));
        hip_check(hipMemsetAsync(shift_max_.p, 0, size_t(m_) * 8, stream_));
        hipLaunchKernelGGL(crt::shifts_a_partial, dim3((m_ + 255) / 256, (k_ + 127) / 128),
                           dim3(256), 0, stream_, a, shift_max_.as<unsigned long long>(),
                           invalid_.as<int>(), m_, k_, lda, 128);
        hip_check(hipGetLastError());
        hipLaunchKernelGGL(crt::shifts_finish, dim3((m_ + 255) / 256), dim3(256), 0, stream_,
                           shift_max_.as<unsigned long long>(), sa_.as<int>(), m_, bits_);
        hip_check(hipGetLastError());
        hipLaunchKernelGGL(crt::shifts_b, dim3(n_), dim3(256), 0, stream_, b, sb_.as<int>(),
                           invalid_.as<int>(), n_, k_, ldb, bits_);
        hip_check(hipGetLastError());
        hip_check(hipEventRecord(events_[1], stream_));

        hipLaunchKernelGGL(encode::encode_kernel<false>, dim3(m_ / 32, k_ / 128), dim3(256), 0,
                           stream_, a, sa_.as<int>(), table_a_.as<ProductOperands>(),
                           lut_.as<uint8_t>(), invalid_.as<int>(), encode_shape(m_), lda,
                           caps_device_.as<unsigned>(), packed_lut_.as<uint32_t>());
        hip_check(hipGetLastError());
        hipLaunchKernelGGL(encode::encode_kernel<true>, dim3(n_ / 32, k_ / 128), dim3(256), 0,
                           stream_, b, sb_.as<int>(), table_b_.as<ProductOperands>(),
                           lut_.as<uint8_t>(), invalid_.as<int>(), encode_shape(n_), ldb,
                           caps_device_.as<unsigned>(), packed_lut_.as<uint32_t>());
        hip_check(hipGetLastError());
        hip_check(hipEventRecord(events_[2], stream_));

        // The first launch of a modulus writes its residue plane, later ones add to it, so the plane
        // needs no clearing. A form with mixed caps (p = 121) has products with 1 and 2 segments.
        for (unsigned fi = 0; fi < table::form_count; ++fi) {
            const auto f = table::host_forms[fi];
            bool first = true;
            for (unsigned pi = f.first; pi < f.first + f.count; ++pi) {
                for (unsigned s = 0; s < ha_[pi].segments; ++s) {
                    ModuleLaunch x{};
                    x.residues = residues_.as<uint16_t>() + size_t(fi) * count;
                    x.invalid = invalid_.as<int>();
                    x.a = ha_[pi].data[s];
                    x.b = hb_[pi].data[s];
                    x.scale_a = ha_[pi].scale[s];
                    x.scale_b = hb_[pi].scale[s];
                    x.k = ha_[pi].cap;
                    x.p = f.p;
                    x.weight = table::host_products[pi].weight % f.p;
                    x.accumulate = first ? 0u : 1u;
                    first = false;
                    module_.launch(x, stream_);
                }
            }
        }
        hip_check(hipEventRecord(events_[3], stream_));

        hipLaunchKernelGGL(crt::reconstruct_kernel, dim3((count + 255) / 256), dim3(256), 0,
                           stream_, residues_.as<uint16_t>(), c, sa_.as<int>(), sb_.as<int>(), m_,
                           n_, ldc);
        hip_check(hipGetLastError());
        hip_check(hipEventRecord(events_[4], stream_));
        hip_check(hipEventSynchronize(events_[4]));
        int invalid = 0;
        hip_check(hipMemcpy(&invalid, invalid_.p, 4, hipMemcpyDeviceToHost));
        if (invalid) {
            throw std::runtime_error(
                "nonfinite input, quantization overflow or inexact accumulator (flag " +
                std::to_string(invalid) + ")");
        }
        Timings t;
        hip_check(hipEventElapsedTime(&t.shift, events_[0], events_[1]));
        hip_check(hipEventElapsedTime(&t.encode, events_[1], events_[2]));
        hip_check(hipEventElapsedTime(&t.products, events_[2], events_[3]));
        hip_check(hipEventElapsedTime(&t.crt, events_[3], events_[4]));
        hip_check(hipEventElapsedTime(&t.total, events_[0], events_[4]));
        return t;
    }
};

} // namespace awe_mxfp6
