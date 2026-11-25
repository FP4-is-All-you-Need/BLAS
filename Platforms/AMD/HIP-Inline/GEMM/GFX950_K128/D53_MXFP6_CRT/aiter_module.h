#pragma once
// One workgroup owns an output tile through the sequence of 448-byte product records.
#include "common.h"
#include "layout.h"
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace awe_mxfp6 {

struct alignas(16) ModuleArgs {
    std::array<uint8_t, 448> bytes{};
    template <class T> void at(size_t offset, T value) {
        if (offset + sizeof(T) > bytes.size()) {
            throw std::logic_error("kernel argument block overflow");
        }
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }
    // AITER's original arguments live at 16-byte slots.
    template <class T> void slot(size_t index, T value) {
        at(index * 16, value);
    }
};
static_assert(sizeof(ModuleArgs) == 448);

struct ModuleLaunch {
    uint16_t *residues;  // this form's residue plane, column-major with stride m
    int *invalid;        // set to nonzero when an accumulator is not a small integer
    const void *a;       // A plane data of this product segment (layout.h)
    const void *b;       // B plane data
    unsigned k;          // 4096 or 8192
    unsigned p;          // modulus
    unsigned weight;     // product weight modulo p
    unsigned accumulate; // 0: write residues; 1: add to the residues already there
};

class Module {
    hipModule_t module_{};
    hipFunction_t function_{};

  public:
    static constexpr const char *symbol = "f6gemm_awe_modular_v2";

    explicit Module(const std::string &path) {
        hip_check(hipModuleLoad(&module_, path.c_str()));
        const hipError_t status = hipModuleGetFunction(&function_, module_, symbol);
        if (status != hipSuccess) {
            (void)hipModuleUnload(module_);
            hip_check(status);
        }
    }
    ~Module() {
        if (module_) {
            (void)hipModuleUnload(module_); // a destructor cannot report the status
        }
    }
    Module(const Module &) = delete;
    Module &operator=(const Module &) = delete;

    static ModuleArgs pack(const ModuleLaunch &x, bool last_in_form) {
        require(x.k == 4096 || x.k == 8192, "segment K must be 4096 or 8192");
        require(x.p >= 2 && x.p <= 65535 && x.weight < x.p && x.accumulate <= 1,
                "modular launch arguments");
        // The first reciprocal reduction leaves a value below 2p. Multiplication
        // by the weight and addition of the prior residue must fit in a dword.
        require(uint64_t(2 * x.p - 1) * x.weight + x.p - 1 <= UINT32_MAX,
                "weighted residue bound");
        ModuleArgs args;
        // AITER's argument block: output pointer, A, B, alpha, N as the output stride, M, N, K,
        // and K / 32 scale groups per row. The epilogue does not use slot 0 (the output
        // pointer), but the main loop still loads it, so it holds the residue plane.
        args.slot(0, x.residues);
        args.slot(2, x.a);
        args.slot(3, x.b);
        args.slot(4, 1.0f);
        args.slot(6, rows);
        args.slot(14, rows);
        args.slot(15, rows);
        args.slot(16, x.k);
        args.slot(19, x.k / 32);
        args.slot(21, x.k / 32);
        // Modular tail (offsets fixed by AITER/modular.s).
        args.at(384, x.residues);
        args.at(392, x.invalid);
        args.at(400, x.p);
        args.at(404, x.weight);
        args.at(408, unsigned((uint64_t(1) << 32) / x.p));
        args.at(412, ((1u << 24) + x.p - 1) / x.p * x.p); // offset all valid sums to nonnegative
        args.at(416, x.accumulate);
        args.at(420, rows);        // residue stride
        args.at(424, 16777216.0f); // |acc| must stay below 2^24 to be an exact integer
        args.at(428, unsigned(last_in_form));
        return args;
    }

    void launch(const ModuleArgs *products, hipStream_t stream) const {
        ModuleArgs args;
        args.at(0, products);
        size_t size = sizeof(args);
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
        hip_check(hipModuleLaunchKernel(function_, rows / 256, rows / 256, 1, 256, 1, 1, 0, stream,
                                        nullptr, config));
    }
};

} // namespace awe_mxfp6
