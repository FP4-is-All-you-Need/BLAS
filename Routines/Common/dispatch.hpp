#pragma once
#include <AWE/awe.h>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include "untested_engines.hpp"
namespace awe {
struct ApiError : std::runtime_error {
    awe_status status;
    ApiError(awe_status s, const char *m) : std::runtime_error(m), status(s) {}
};
inline void require(bool yes, awe_status code, const char *message) {
    if (!yes) throw ApiError(code, message);
}
struct Selection { std::string engine; bool complex; int operation, bits, moduli; int element_bytes=0; bool native_adapter=false; bool general_adapter=false; };
// Float Level-3 support prepares operands for a base GEMM engine.
// Dedicated real/complex and native adapters retain their operation code.
inline int create_operation(const Selection &s) {
    if (s.native_adapter) return s.operation;
    if (s.general_adapter || s.element_bytes == 4) return 1;
    return s.operation == 9 ? 2 : s.operation > 4 ? 1 : s.operation;
}
inline Selection select_default(const awe_descriptor &d, int bits = 0, const char *explicit_library = nullptr) {
    require(d.struct_size == sizeof(d), AWE_INVALID_ARGUMENT, "descriptor size mismatch");
    require(d.backend == AWE_K96 || d.backend == AWE_K64, AWE_INVALID_ARGUMENT, "unknown backend");
    require(d.routine >= AWE_DGEMM && d.routine <= AWE_DTRSM, AWE_INVALID_ARGUMENT, "unknown routine");
    require(bits >= 0 && bits <= (d.routine == AWE_DTRSM ? 57 : 53), AWE_INVALID_ARGUMENT, "integer width is outside the routine limit");
    const bool integer = d.routine == AWE_I8GEMM;
    const bool cgemm = d.routine == AWE_CGEMM;
    const bool single = d.routine >= AWE_SGEMM && d.routine <= AWE_STRMM;
    const bool z_general = d.routine == AWE_ZGEMM || d.routine == AWE_ZSYMM ||
                           d.routine == AWE_ZHEMM || d.routine == AWE_ZSYR2K || d.routine == AWE_ZHER2K;
    const bool d_general = d.routine == AWE_DSYMM || d.routine == AWE_DSYR2K;
    const bool rank2 = d.routine == AWE_SSYR2K || d.routine == AWE_DSYR2K || d.routine == AWE_ZSYR2K || d.routine == AWE_ZHER2K;
    require(d.routine <= AWE_ZHERK || z_general || d.routine == AWE_ZSYRK || d_general || single || cgemm || integer || d.routine == AWE_DTRSM, AWE_NOT_SUPPORTED,
            "this routine's engine is not included yet");
    require(d.routine == AWE_DTRSM || (d.backend == AWE_K64 && (bits == 0 || bits == 53)) || integer || single || cgemm || bits == 0 || (d.routine == AWE_ZHERK ? (bits == 51 || bits == 49 || bits == 53) :
                         d.routine == AWE_ZSYRK ? (bits == 52 || bits == 50 || bits == 53) :
                         d.routine == AWE_ZGEMM ? (bits == 53 || bits == 48) :
                         d.routine == AWE_DGEMM ? (bits >= 51 && bits <= 53) : bits == 53), AWE_NOT_SUPPORTED,
            "requested integer width is not included for this routine");
    require(d.moduli == AWE_MODULI_AUTO || d.moduli == AWE_MODULI_17 || d.moduli == AWE_MODULI_18 || d.moduli == AWE_MODULI_21,
            AWE_INVALID_ARGUMENT, "unknown modulus set");
    require(d.m > 0 && d.n > 0 && d.k > 0 && d.m <= INT_MAX && d.n <= INT_MAX && d.k <= INT_MAX,
            AWE_INVALID_ARGUMENT, "dimensions must be positive and fit int");
    require(d.trans_a == AWE_NO_TRANSPOSE && d.trans_b == AWE_NO_TRANSPOSE && d.triangle == AWE_LOWER &&
            d.side == AWE_LEFT && d.diagonal == AWE_NON_UNIT && d.alpha.real == 1 && d.alpha.imag == 0 &&
            d.beta.real == 0 && d.beta.imag == 0, AWE_NOT_SUPPORTED,
            "only no-transpose, lower, left, non-unit, alpha=1, beta=0 is supported");
    require(d.lda >= d.m && d.ldc >= d.m && d.ldb >= (rank2 ? d.m : d.k),
            AWE_INVALID_ARGUMENT, "leading dimensions are too small (set ldb=k for rank-k)");
    require(d.m <= 65536 && d.n <= 65536, AWE_NOT_SUPPORTED, "M and N must not exceed 65536");
    const bool z = z_general || d.routine == AWE_ZSYRK || d.routine == AWE_ZTRMM || d.routine == AWE_ZHERK;
    const auto max_elements = std::numeric_limits<size_t>::max() / (single ? 4 : z ? 16 : 8);
    require(uint64_t(d.lda) <= max_elements / uint64_t(d.k) && uint64_t(d.ldb) <= max_elements / uint64_t(rank2 ? d.k : d.n) &&
            uint64_t(d.ldc) <= max_elements / uint64_t(d.n), AWE_INVALID_ARGUMENT, "matrix byte size overflows size_t");
    const bool gemm = integer || cgemm || d.routine == AWE_SGEMM || d.routine == AWE_DGEMM || d.routine == AWE_ZGEMM;
    if (d.routine == AWE_DTRSM) {
        require(d.backend == AWE_K64, AWE_NOT_SUPPORTED, "triangular solve is included only for K64");
        require(d.k == d.m && d.m <= 4096 && d.n <= 2048, AWE_NOT_SUPPORTED, "triangular solve requires K=M<=4096 and N<=2048");
        require(d.lda <= INT_MAX && d.ldb <= INT_MAX && d.ldc <= INT_MAX, AWE_INVALID_ARGUMENT, "triangular solve strides must fit int");
        const int width = bits ? bits : 57;
        require((width == 53 || width == 57) && d.moduli == AWE_MODULI_AUTO, AWE_NOT_SUPPORTED, "triangular solve supports fixed 53-bit or 57-bit engines");
        return {"k64_d" + std::to_string(width) + "_trsm", false, 10, width, width == 57 ? 20 : 18, 0, true};
    }
    if (!gemm) {
        require(d.m == d.n && d.lda == d.m && d.ldb == (rank2 ? d.m : d.k) && d.ldc == d.m,
                AWE_NOT_SUPPORTED, "Level-3 requires square output and contiguous matrices");
        if (d.routine == AWE_DTRMM || d.routine == AWE_ZTRMM || d.routine == AWE_ZSYMM || d.routine == AWE_ZHEMM || d.routine == AWE_DSYMM || d.routine == AWE_SSYMM || d.routine == AWE_STRMM)
            require(d.k == d.n, AWE_NOT_SUPPORTED, "TRMM, SYMM and HEMM require square A, B and C");
    }
    if (integer) {
        require(d.moduli == AWE_MODULI_AUTO && (bits == 0 || bits == 8), AWE_NOT_SUPPORTED,
                "INT8 uses exact 8-bit inputs without a modulus set");
        require(d.k <= 16384 && d.k * 128 * 128 <= INT_MAX, AWE_NOT_SUPPORTED,
                "INT8 K exceeds the kernel or INT32 sum limit");
        require(d.lda <= INT_MAX && d.ldb <= INT_MAX && d.ldc <= INT_MAX, AWE_INVALID_ARGUMENT,
                "INT8 leading dimensions must fit int");
        return {d.backend == AWE_K64 ? "k64_i8" : "k96_i8", false, 1, 8, 0, 1};
    }
    if (cgemm) {
        require(d.moduli == AWE_MODULI_AUTO, AWE_NOT_SUPPORTED, "complex single precision uses a fixed modulus set for each width");
        const int width = bits ? bits : d.backend == AWE_K64 ? 24 : 22;
        require(width == 22 || width == 20 || width == 23 || (d.backend == AWE_K64 && width == 24), AWE_NOT_SUPPORTED, "requested complex single-precision width is not included");
        require(d.k <= 16384, AWE_NOT_SUPPORTED, "complex single-precision K exceeds the included table");
        require(width != 23 || d.k <= 4096, AWE_NOT_SUPPORTED, "23-bit complex single precision is included up to K=4096");
        return {(d.backend == AWE_K64 ? "k64_c" : "k96_c") + std::to_string(width), true, 1, width, width == 24 ? 11 : width >= 22 ? 10 : 9, 8};
    }
    if (single) {
        if (d.backend == AWE_K64) {
            require(d.moduli == AWE_MODULI_AUTO, AWE_NOT_SUPPORTED, "single precision uses a fixed modulus set for each width");
            require(d.k <= 8192, AWE_NOT_SUPPORTED, "single-precision K exceeds the included table");
            const bool small = d.k <= 4096;
            const int width = bits ? bits : d.routine == AWE_SGEMM ? 23 : d.routine == AWE_SSYRK ? (small ? 23 : 24) :
                              d.routine == AWE_SSYR2K ? (small ? 23 : 21) : d.routine == AWE_STRMM ? (small ? 23 : 22) : 22;
            const bool valid = d.routine == AWE_SGEMM ? (width == 23 || width == 21) :
                               d.routine == AWE_SSYRK ? (width == 23 || (!small && width == 24)) :
                               d.routine == AWE_STRMM ? (small ? width == 21 || width == 23 : width == 22) :
                               d.routine == AWE_SSYMM ? (width == 22 || width == (small ? 21 : 20)) :
                               small ? (width == 22 || width == 23) : (width == 20 || width == 21);
            require(valid, AWE_NOT_SUPPORTED, "requested single-precision width is not included");
            std::string engine = "k64_s" + std::to_string(width);
            int moduli = width == 24 ? 12 : width <= 21 ? 10 : 11;
            if (d.routine == AWE_SGEMM && width == 23) { engine += "_gemm"; moduli = 9; }
            else if (width == 22) engine += small ? "_k4096" : "_k8192";
            else if (!small && (width == 21 || width == 23)) { engine += "_k8192"; if (width == 23) moduli = 12; }
            const int operation = d.routine == AWE_SGEMM ? 1 : d.routine == AWE_SSYRK ? 2 : d.routine == AWE_STRMM ? 3 : d.routine == AWE_SSYMM ? 5 : 7;
            return {engine, false, operation, width, moduli, 4, !gemm};
        }
        require(d.moduli == AWE_MODULI_AUTO, AWE_NOT_SUPPORTED, "single precision uses a fixed modulus set for each width");
        const bool small = d.k <= 4096;
        // At the same width, the normal-input measurement chooses variant a.
        const int small_width = bits ? bits : d.routine == AWE_SSYRK || d.routine == AWE_STRMM ? 23 :
                                d.routine == AWE_SGEMM ? 23 : d.routine == AWE_SSYR2K ? 21 : 22;
        std::string small_id;
        int small_moduli = 11;
        bool small_native = false;
        if (d.routine == AWE_SGEMM && small_width == 22) { small_id="s22_gemm_k4096"; small_moduli=8; }
        if (d.routine == AWE_SSYMM && small_width == 22) small_id="s22_current_k4096";
        if ((d.routine == AWE_SSYMM || d.routine == AWE_STRMM) && small_width == 21) {
            small_id="s21_level3_k4096"; small_moduli=10; small_native=true;
        }
        if (d.routine == AWE_SSYRK && small_width == 23) {
            small_id=explicit_library && std::string(explicit_library)=="k96_s23_syrk_k4096_b" ?
                     "s23_syrk_k4096_b" : "s23_syrk_k4096_a";
            small_native=true;
        }
        if (d.routine == AWE_SSYR2K && small_width == 22) { small_id="s22_syr2k_k4096"; small_native=true; }
        if (d.routine == AWE_STRMM && small_width == 23) small_id="s23_current_k4096";
        if (!small_id.empty() && (small || explicit_library) && (!explicit_library || std::string(explicit_library)=="k96_"+small_id)) {
            require(small, AWE_NOT_SUPPORTED, "this engine requires K<=4096");
            const int operation=d.routine==AWE_SGEMM ? 1 : d.routine==AWE_SSYRK ? 2 : d.routine==AWE_STRMM ? 3 : d.routine==AWE_SSYMM ? 5 : 7;
            return {"k96_"+small_id, false, operation, small_width, small_moduli, 4, small_native};
        }
        const int width=bits ? bits : d.routine==AWE_SGEMM ? 23 : d.routine==AWE_SSYRK ? 24 : d.routine==AWE_SSYR2K ? 21 : 22;
        const bool valid=d.routine==AWE_SGEMM ? (width==23 || width==21) :
                         d.routine==AWE_SSYRK ? (width==24 || width==23) :
                         d.routine==AWE_SSYR2K ? (width==21 || width==20) :
                         d.routine==AWE_SSYMM ? (width==22 || width==20) : width==22;
        require(valid, AWE_NOT_SUPPORTED, "requested single-precision width is not included");
        require(d.k<=8192, AWE_NOT_SUPPORTED, "single-precision K exceeds the included table or coverage limit");
        const bool fused=d.routine==AWE_SSYRK || d.routine==AWE_STRMM || width==20;
        const int operation=d.routine==AWE_SGEMM ? 1 : d.routine==AWE_SSYRK ? 2 : d.routine==AWE_STRMM ? 3 : d.routine==AWE_SSYMM ? 5 : 7;
        return {"k96_s"+std::to_string(width)+(d.routine==AWE_SSYR2K && width==20 ? "_syr2k" : gemm ? "_gemm" : fused ? "_fused" : "_current"), false,
                operation,width,gemm && width==23 ? 9 : width>=23 ? 12 : width==22 ? 11 : 10,4,fused};
    }
    if (d.backend == AWE_K64 && z) {
        require(d.moduli == AWE_MODULI_AUTO && (bits == 0 || bits == 53), AWE_NOT_SUPPORTED,
                "K64 complex routines use the included 53-bit modulus set");
        require(d.k <= 16384, AWE_NOT_SUPPORTED, "K64 complex K exceeds the table");
        const int operation = d.routine == AWE_ZGEMM ? 1 : d.routine == AWE_ZSYRK ? 2 : d.routine == AWE_ZTRMM ? 3 :
                              d.routine == AWE_ZSYMM ? 5 : d.routine == AWE_ZHEMM ? 6 : d.routine == AWE_ZSYR2K ? 7 : d.routine == AWE_ZHER2K ? 8 : 9;
        return {"k64_z53_general", true, operation, 53, 20, 0, !gemm};
    }
    if (d.backend == AWE_K64 && !gemm) {
        require(bits == 0 || bits == 53, AWE_NOT_SUPPORTED, "K64 real Level-3 uses the included 53-bit modulus sets");
        const int mod = d.moduli ? d.moduli : d.k <= 8192 ? 18 : 21;
        require((mod == 18 && d.k <= 8192) || (mod == 21 && d.k <= 16384), AWE_NOT_SUPPORTED,
                "K64 Level-3 requires 18 moduli up to K=8192 or 21 moduli up to K=16384");
        return {"k64_d" + std::to_string(mod), false, d.routine == AWE_DSYRK ? 2 : d.routine == AWE_DTRMM ? 3 : d.routine == AWE_DSYMM ? 5 : 7, 53, mod, 0, false, true};
    }
    if (z) {
        require(d.moduli == AWE_MODULI_AUTO, AWE_NOT_SUPPORTED, "complex routines use their fixed modulus set");
        require(d.k <= 16384, AWE_NOT_SUPPORTED, "complex K exceeds the table limit");
        if (d.routine == AWE_ZSYRK) {
            const int width = bits ? bits : d.k <= 4096 ? 53 : 52;
            if (width == 53) return {"k96_z53_syrk_trmm", true, 2, 53, 20};
            require(d.k <= (width == 52 ? 8958 : 15925), AWE_NOT_SUPPORTED,
                    "K exceeds strict complex CRT coverage at this integer width");
            return {width == 52 ? "k96_z52_syrk" : "k96_z50_syrk", true, 2, width, width == 52 ? 19 : 18};
        }
        if (d.routine == AWE_ZHERK && bits == 53)
            return {"k96_z53_herk", true, 4, 53, 20};
        if (d.routine == AWE_ZHERK && (bits == 0 || bits == 51) &&
            ((d.k <= 4096 && !explicit_library) ||
             (explicit_library && std::string(explicit_library) == "k96_z51_herk_k4096"))) {
            require(d.k <= 4096, AWE_NOT_SUPPORTED, "this engine requires K<=4096");
            return {"k96_z51_herk_k4096", true, 9, 51, 19};
        }
        if (d.routine == AWE_ZHERK && bits == 49)
            return {"k96_z49_herk", true, 9, 49, 19};
        if (d.routine == AWE_ZGEMM && bits == 48) {
            require(d.k <= 15187, AWE_NOT_SUPPORTED, "K exceeds strict 48-bit complex CRT coverage");
            return {"k96_z48_general", true, 1, 48, 18};
        }
        if (z_general) {
            const int operation = d.routine == AWE_ZSYMM ? 5 : d.routine == AWE_ZHEMM ? 6 :
                                  d.routine == AWE_ZSYR2K ? 7 : d.routine == AWE_ZHER2K ? 8 : 1;
            return {"k96_z53_general", true, operation, 53, 20};
        }
        return {d.routine == AWE_ZHERK ? "k96_z51_herk" : "k96_z53_syrk_trmm", true,
                d.routine == AWE_ZHERK ? 4 : 3, d.routine == AWE_ZHERK ? 51 : 53, 0};
    }
    if (d_general || (d.routine == AWE_DGEMM && ((bits && bits != 53) ||
        (explicit_library && (std::string(explicit_library) == "k96_d53_general" || std::string(explicit_library) == "k96_d53_gemm_crt"))))) {
        require(d.backend == AWE_K96, AWE_NOT_SUPPORTED, "this real engine is not included for K64 yet");
        const int width = bits ? bits : 53;
        const int moduli = width == 53 ? 18 : 17;
        require(d.moduli == AWE_MODULI_AUTO || d.moduli == moduli, AWE_NOT_SUPPORTED,
                "requested modulus set does not match this integer width");
        require(d.k <= (width == 53 ? 8254 : width == 51 ? 8720 : 8560), AWE_NOT_SUPPORTED,
                "K exceeds strict real CRT coverage at this integer width");
        const bool fast_crt = d.routine == AWE_DGEMM && (!explicit_library ||
            std::string(explicit_library) == "k96_d" + std::to_string(width) + "_gemm_crt");
        return {"k96_d" + std::to_string(width) + (fast_crt ? "_gemm_crt" : "_general"), false,
                d.routine == AWE_DSYMM ? 5 : d.routine == AWE_DSYR2K ? 7 : 1, width, moduli};
    }
    int mod = d.moduli;
    if (!mod) mod = gemm ? (d.k <= (d.backend == AWE_K64 ? 8192 : 4096) ? 17 : 18) : (d.k <= 8254 ? 18 : 21);
    require(mod != 17 || gemm, AWE_NOT_SUPPORTED, "17 moduli have no Level-3 patch");
    require(!(gemm && d.backend == AWE_K96 && (d.k > 8254 || mod == 21)), AWE_NOT_SUPPORTED,
            "K96 DGEMM requires K<=8254; general 21-modulus DGEMM is not included yet");
    require(!(d.backend == AWE_K64 && d.k > 8192 && mod != 21), AWE_NOT_SUPPORTED,
            "K exceeds the included table for 17/18 moduli; select AWE_MODULI_21 explicitly");
    require(d.k <= (mod == 17 ? 8192 : mod == 18 ? (d.backend == AWE_K64 ? 8192 : 8254) : 16384), AWE_NOT_SUPPORTED,
            "K exceeds strict coverage or the included table for the selected modulus set");
    std::string id = "k" + std::to_string(d.backend) + "_d" + std::to_string(mod);
    if (d.routine == AWE_DSYRK) id += "_syrk";
    if (d.routine == AWE_DTRMM) id += "_syrk_trmm";
    return {id, false, gemm ? 1 : d.routine == AWE_DSYRK ? 2 : 3, 53, mod};
}
}

#include "engine_choices.hpp"
namespace awe {
inline Selection select(const awe_descriptor &descriptor, int bits = 0, const char *name = nullptr) {
    if (!name) {
        auto selected = select_default(descriptor, bits);
        require(!needs_explicit_engine(selected.engine), AWE_NOT_SUPPORTED,
                "this source has not run on a GPU; select engine_name explicitly until full small-case and hash checks pass");
        return selected;
    }
    const auto *choice = engine_choice(name, descriptor.backend);
    require(choice, AWE_NOT_SUPPORTED, "unknown engine name for this backend");
    require(bits == 0 || bits == choice->bits, AWE_NOT_SUPPORTED, "engine name and integer width disagree");
    auto d = descriptor;
    // Explicit real modulus variants may differ from the default at this K.
    if (choice->library[4] == 'd' && d.routine != AWE_DTRSM) {
        require(d.moduli == AWE_MODULI_AUTO || d.moduli == choice->moduli,
                AWE_NOT_SUPPORTED, "engine name and modulus set disagree");
        d.moduli = static_cast<awe_moduli>(choice->moduli);
    }
    auto selected = select_default(d, choice->bits, choice->library);
    if (std::strcmp(name, "s23_m11_trmm_gemm") == 0) {
        require(d.routine == AWE_STRMM, AWE_NOT_SUPPORTED,
                "this engine name requires STRMM");
        // Exercise the same host operand preparation as the K96 GEMM path.
        selected.native_adapter = false;
    }
    require(selected.engine == choice->library, AWE_NOT_SUPPORTED,
            "engine is not supported for this routine and shape");
    return selected;
}
}
