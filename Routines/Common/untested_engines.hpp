#pragma once
#include <string>
namespace awe {
// Generated from the assembled-source records in import_k64.json.
inline constexpr const char *untested_single_engines[] = {
    "k64_s20",
    "k64_s21_k8192",
    "k64_s22_k4096",
    "k64_s22_k8192",
    "k64_s23_k8192",
    "k64_s24",
    "k64_s23_gemm",
};
inline bool needs_explicit_engine(const std::string &id) {
    for (const auto *entry : untested_single_engines)
        if (id == entry) return true;
    // Lower-width complex engines remain explicit choices.
    return id == "k64_c20" || id == "k64_c22" || id == "k64_c23" || id == "k64_d57_trsm";
}
}
