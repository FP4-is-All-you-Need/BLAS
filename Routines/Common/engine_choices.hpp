#pragma once
#include <cstring>
namespace awe {
// Names describe width, modulus count and use; backend comes from the descriptor.
struct EngineChoice { const char *name, *library; int backend, bits, moduli; };
inline constexpr EngineChoice engine_choices[] = {
    {"d53_m17_gemm", "k96_d17", 96, 53, 17},
    {"d53_m18_gemm", "k96_d18", 96, 53, 18},
    {"d53_m18_syrk", "k96_d18_syrk", 96, 53, 18},
    {"d53_m21_syrk", "k96_d21_syrk", 96, 53, 21},
    {"d53_m18_syrk_trmm", "k96_d18_syrk_trmm", 96, 53, 18},
    {"d53_m21_syrk_trmm", "k96_d21_syrk_trmm", 96, 53, 21},
    {"z53_m20_syrk_trmm", "k96_z53_syrk_trmm", 96, 53, 20},
    {"z51_m19_herk", "k96_z51_herk", 96, 51, 19},
    {"d53_m17_gemm", "k64_d17", 64, 53, 17},
    {"d53_m18_gemm", "k64_d18", 64, 53, 18},
    {"d53_m21_gemm", "k64_d21", 64, 53, 21},
    {"z53_m20_level3", "k96_z53_general", 96, 53, 20},
    {"z52_m19_syrk", "k96_z52_syrk", 96, 52, 19},
    {"z50_m18_syrk", "k96_z50_syrk", 96, 50, 18},
    {"z49_m19_herk", "k96_z49_herk", 96, 49, 19},
    {"z48_m18_level3", "k96_z48_general", 96, 48, 18},
    {"d53_m18_level3", "k96_d53_general", 96, 53, 18},
    {"d51_m17_level3", "k96_d51_general", 96, 51, 17},
    {"d52_m17_level3", "k96_d52_general", 96, 52, 17},
    {"s20_m10_level3_fused", "k96_s20_fused", 96, 20, 10},
    {"s23_m12_level3_fused", "k96_s23_fused", 96, 23, 12},
    {"s24_m12_level3_fused", "k96_s24_fused", 96, 24, 12},
    {"s22_m11_level3_fused", "k96_s22_fused", 96, 22, 11},
    {"s22_m11_level3", "k96_s22_current", 96, 22, 11},
    {"s21_m10_level3", "k96_s21_current", 96, 21, 10},
    {"s23_m9_gemm", "k96_s23_gemm", 96, 23, 9},
    {"s21_m10_gemm", "k96_s21_gemm", 96, 21, 10},
    {"s20_m10_syr2k", "k96_s20_syr2k", 96, 20, 10},
    {"c22_m10_gemm", "k96_c22", 96, 22, 10},
    {"c20_m9_gemm", "k96_c20", 96, 20, 9},
    {"i8_gemm", "k96_i8", 96, 8, 0},
    {"z53_m20_level3", "k64_z53_general", 64, 53, 20},
    {"i8_gemm", "k64_i8", 64, 8, 0},
    {"s21_m10_level3", "k64_s21", 64, 21, 10},
    {"s23_m11_level3", "k64_s23", 64, 23, 11},
    {"s23_m11_trmm_gemm", "k64_s23", 64, 23, 11},
    {"s20_m10_level3", "k64_s20", 64, 20, 10},
    {"s21_m10_k8192", "k64_s21_k8192", 64, 21, 10},
    {"s22_m11_k4096", "k64_s22_k4096", 64, 22, 11},
    {"s22_m11_k8192", "k64_s22_k8192", 64, 22, 11},
    {"s23_m12_k8192", "k64_s23_k8192", 64, 23, 12},
    {"s24_m12_level3", "k64_s24", 64, 24, 12},
    {"s23_m9_gemm", "k64_s23_gemm", 64, 23, 9},
    {"c20_m9_gemm", "k64_c20", 64, 20, 9},
    {"c22_m10_gemm", "k64_c22", 64, 22, 10},
    {"c23_m10_gemm", "k64_c23", 64, 23, 10},
    {"c24_m11_gemm", "k64_c24", 64, 24, 11},
    {"d53_m18_trsm", "k64_d53_trsm", 64, 53, 18},
    {"d57_m20_trsm", "k64_d57_trsm", 64, 57, 20},
    {"s21_m10_level3_k4096", "k96_s21_level3_k4096", 96, 21, 10},
    {"s22_m11_syr2k_k4096", "k96_s22_syr2k_k4096", 96, 22, 11},
    {"s23_m11_syrk_k4096_a", "k96_s23_syrk_k4096_a", 96, 23, 11},
    {"s23_m11_syrk_k4096_b", "k96_s23_syrk_k4096_b", 96, 23, 11},
    {"s22_m11_symm_k4096", "k96_s22_current_k4096", 96, 22, 11},
    {"s23_m11_trmm_k4096", "k96_s23_current_k4096", 96, 23, 11},
    {"s22_m8_gemm_k4096", "k96_s22_gemm_k4096", 96, 22, 8},
    {"c23_m10_gemm", "k96_c23", 96, 23, 10},
    {"z51_m19_herk_k4096", "k96_z51_herk_k4096", 96, 51, 19},
    {"z53_m20_herk", "k96_z53_herk", 96, 53, 20},
    {"d51_m17_gemm_crt", "k96_d51_gemm_crt", 96, 51, 17},
    {"d52_m17_gemm_crt", "k96_d52_gemm_crt", 96, 52, 17},
    {"d53_m18_gemm_crt", "k96_d53_gemm_crt", 96, 53, 18},
};
inline const EngineChoice *engine_choice(const char *name, int backend) {
    for (const auto &entry : engine_choices)
        if (entry.backend == backend && std::strcmp(entry.name, name) == 0) return &entry;
    return nullptr;
}
inline const char *public_engine_name(const std::string &library) {
    for (const auto &entry : engine_choices)
        if (library == entry.library) return entry.name;
    throw ApiError(AWE_RUNTIME_ERROR, "selected engine has no public name");
}
}
