#pragma once
#include <cstddef>
struct awe_engine_configuration {
    size_t struct_size;
    int pacing, chunks, order, encoder_minb, pingpong, encoder;
};
struct awe_engine_effective {
    int pace, chunks, crt_chunks, item_order, encoder_minb, pingpong;
    int skip_b_padding, pdl, shift_form, extra_fx, encoder_fast, moduli;
};
#include <cstdlib>
#include <cstring>

// Private host ABI: operation 0 retains legacy environment selection.
struct awe_engine_options {
    int operation = 0; // 1 GEMM, 2 SYRK, 3 TRMM, 4 HERK
    bool frozen = false;
    int measured_pingpong = 0;
};
inline const char *awe_option_env(const awe_engine_options *options, const char *name) {
    if (options && options->operation) {
        if (!std::strcmp(name, "L8_SYRK")) {
            return options->operation == 2 ? "1" : nullptr;
        }
        if (!std::strcmp(name, "L8_TRMM")) {
            return options->operation == 3 ? "1" : nullptr;
        }
        if (!std::strcmp(name, "ZL8_HERK")) {
            return options->operation == 4 ? "1" : nullptr;
        }
    }
    if (options && options->frozen) {
        if (!std::strcmp(name, "L8_PINGPONG")) return options->measured_pingpong ? "1" : "0";
        return nullptr;
    }
    return std::getenv(name);
}

inline bool awe_configuration_supported(const awe_engine_configuration *o) {
    return o && o->struct_size == sizeof(*o) && o->pacing == 0 &&
        o->chunks == 0 && o->order == 0 &&
        o->encoder_minb == 0 && o->pingpong == 0 &&
        o->encoder == 0;
}
