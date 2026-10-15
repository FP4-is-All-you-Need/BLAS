#pragma once

#include <type_traits>
#include "l8_probe.cuh"

namespace l8e {

#ifndef L8E_FAST_FX
#define L8E_FAST_FX 25
#endif
constexpr int kFastFx = L8E_FAST_FX;
#ifndef L8E_FAST_MINB
#define L8E_FAST_MINB 3
#endif
constexpr int kFastMinb = L8E_FAST_MINB;
static_assert(kFastMinb >= 3 && kFastMinb <= 6, "the forms measured on B300");
static_assert(kFastFx == 1 || kFastFx == 9 || kFastFx == 17 || kFastFx == 25,
              "the forms measured on B300");

template <int R, int U, int W,
          class T = double> // T float: FP32 inputs (abs64 widens the bits, l8_probe.cuh)
__global__ void __launch_bounds__(256)
    shift_cols_p(const T *__restrict__ x, int *exponents, int *invalid, int flagval, int outer,
                 int k, int64_t ld, int bits) {
    if constexpr (W == 2 || W == 3) {
        pdl_trigger();
    }
    constexpr int C = 256 / R;
    __shared__ unsigned long long red[C][R];
    const int t = threadIdx.x, r = t % R, c = t / R, o0 = blockIdx.x * R, o = o0 + r;
    unsigned long long mx[U] = {};
    bool bad = false;
    if (o < outer) {
        int q = c;
        for (; q + (U - 1) * C < k; q += U * C) {
            unsigned long long a[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                a[u] = abs64(x[size_t(q + u * C) * ld + o]);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (!rfinite(a[u])) {
                    bad = true;
                } else {
                    mx[u] = a[u] > mx[u] ? a[u] : mx[u];
                }
            }
        }
        for (; q < k; q += C) {
            const unsigned long long a = abs64(x[size_t(q) * ld + o]);
            if (!rfinite(a)) {
                bad = true;
            } else {
                mx[0] = a > mx[0] ? a : mx[0];
            }
        }
    }
#pragma unroll
    for (int u = 1; u < U; ++u) {
        mx[0] = mx[u] > mx[0] ? mx[u] : mx[0];
    }
    if (bad) {
        atomicExch(invalid, flagval);
    }
    red[c][r] = mx[0];
    __syncthreads();
    if (t < R) {
        unsigned long long m = 0;
        for (int i = 0; i < C; ++i) {
            m = red[i][t] > m ? red[i][t] : m;
        }
        if (o0 + t < outer) {
            exponents[o0 + t] = m == 0 ? 0 : bits - 1 - qint::ilogb_bits(m);
        }
    }
    if constexpr (W == 2) {
        pdl_wait();
    }
}
template <int U, int W, class T = double>
__global__ void __launch_bounds__(256)
    shift_rows_p(const T *__restrict__ x, int *exponents, int *invalid, int flagval, int outer,
                 int k, int64_t ld, int bits) {
    if constexpr (W == 2 || W == 3) {
        pdl_trigger();
    }
    __shared__ unsigned long long red[256];
    const int o = blockIdx.x, t = threadIdx.x;
    const T *row = x + size_t(o) * ld;
    unsigned long long mx[U] = {};
    bool bad = false;
    int q = t;
    for (; q + (U - 1) * 256 < k; q += U * 256) {
        unsigned long long a[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            a[u] = abs64(row[q + u * 256]);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!rfinite(a[u])) {
                bad = true;
            } else {
                mx[u] = a[u] > mx[u] ? a[u] : mx[u];
            }
        }
    }
    for (; q < k; q += 256) {
        const unsigned long long a = abs64(row[q]);
        if (!rfinite(a)) {
            bad = true;
        } else {
            mx[0] = a > mx[0] ? a : mx[0];
        }
    }
#pragma unroll
    for (int u = 1; u < U; ++u) {
        mx[0] = mx[u] > mx[0] ? mx[u] : mx[0];
    }
    if (bad) {
        atomicExch(invalid, flagval);
    }
    red[t] = mx[0];
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) {
            red[t] = red[t + s] > red[t] ? red[t + s] : red[t];
        }
        __syncthreads();
    }
    if (t == 0) {
        exponents[o] = red[0] == 0 ? 0 : bits - 1 - qint::ilogb_bits(red[0]);
    }
    if constexpr (W == 2) {
        pdl_wait();
    }
}
// a launch with or without cudaLaunchAttributeProgrammaticStreamSerialization
template <class... KArgs, class... Args>
inline void launch_pdl(void (*kern)(KArgs...), dim3 grid, size_t smem, cudaStream_t st, bool pdl,
                       Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = pdl ? at : nullptr;
    cfg.numAttrs = pdl ? 1 : 0;
    if (cudaLaunchKernelEx(&cfg, kern, args...) != cudaSuccess) {
        throw std::runtime_error("l8e: chain launch failed");
    }
}

struct FastEncoder : Encoder {
    unsigned char *dlutx[2] = {nullptr,
                               nullptr}; // FX 16: the u16 LUT followed by the u8 copy, per side

    int minb = 0;

    bool skip_b_pad = false;

    int extra_fx = 0;
    // The 38-plane layout (FX 16384, l8_probe.cuh) for the GEMM that builds the sum plane in Shared
    // Memory: set before init (geo.planes and the packed sizes follow it); the bytes = the 50-plane form's without the 3-plane moduli's
    // plane 2 (test_nosum.cu). Only with extra_fx 0 / 256. Default false (unchanged).
    bool nosum = false;
    unsigned *cs_buf =
        nullptr; // (m + n) x L8CS_N row sums, then 2 x L8CS_N maxima and one word of failure bits (cs_result)
    size_t cs_words() const {
        return size_t(geo.m + geo.n) * L8CS_N + 2 * L8CS_N + 1;
    }
    const unsigned *cs_result() const {
        return cs_buf + size_t(geo.m + geo.n) * L8CS_N;
    }
    template <class F> void by_extra(F &&f) const {
        switch (extra_fx) {
        case 0:
            f(std::integral_constant<int, 0>{});
            return;
        case 256:
            f(std::integral_constant<int, 256>{});
            return;
        case 1024:
            f(std::integral_constant<int, 1024>{});
            return;
        case 1280:
            f(std::integral_constant<int, 1280>{});
            return;
        default:
            throw std::runtime_error("l8e: FastEncoder extra_fx must be 0, 256, 1024 or 1280");
        }
    }
    // chain(): PDL between its four kernels when pdl is set, the device has it (compute capability 9.0 and later) and the form was
    // measured with it (MINB 3 or 6, not a_direct); otherwise the same four kernels as plain launches. flag_value is what the shift
    // kernels store in *invalid on a non-finite input (1 = Encoder::shift's; a per-call value lets the caller drop the memset and
    // compare with the call's own value). pdl_kernels_without_launch_attr: tests only (the PDL kernel forms launched plainly).
    bool pdl = false;
    int flag_value = 1;

    int shift_form = 0;
    bool pdl_supported = false;
    bool pdl_kernels_without_launch_attr = false;
    void init(int bits_, int m, int n, int k, bool a_direct_ = false) {
        release_x();
        {
            int dev = 0, major = 0;
            pdl_supported = cudaGetDevice(&dev) == cudaSuccess &&
                            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                                   dev) == cudaSuccess &&
                            major >= 9;
        }
        Encoder::init(bits_, m, n, k, a_direct_);
        if (a_direct) {
            return;
        }
        if (!ct_tables_match()) {
            throw std::runtime_error("l8e: FastEncoder's constants differ from l8_tables.h");
        }
        if (nosum) { // FX 16384 keys on "3 planes" = mechanism A (l8_kind 1) in this table set
            for (int q = 0; q < L8_NMOD; ++q) {
                if ((l8_kind[q] == 1) != (l8_planes[q] == 3)) {
                    throw std::runtime_error("l8e: nosum needs the 3-plane moduli = mechanism A");
                }
            }
            geo.planes = L8NoSumPlanes;
        }
        if constexpr ((kFastFx & 16) != 0) {
            for (int side = 0; side < 2; ++side) {
                const std::vector<unsigned char> h = lut_x_host(*this, side);
                if (cudaMalloc(&dlutx[side], h.size()) != cudaSuccess ||
                    cudaMemcpy(dlutx[side], h.data(), h.size(), cudaMemcpyHostToDevice) !=
                        cudaSuccess) {
                    throw std::runtime_error("l8e: FastEncoder LUT setup failed");
                }
            }
        }
        if (cudaMalloc(&cs_buf, cs_words() * 4) != cudaSuccess) {
            throw std::runtime_error("l8e: FastEncoder row-sum buffer");
        }
        geo.cs = cs_buf;
    }
    // FX 1024 around plain launch() calls: zero the sums before encoding A and B, check after both
    void cs_zero(cudaStream_t st = 0) const {
        if (cudaMemsetAsync(cs_buf, 0, cs_words() * 4, st) != cudaSuccess) {
            throw std::runtime_error("l8e: cs_zero");
        }
    }
    void cs_check(cudaStream_t st = 0) const {
        cs_check_k<<<L8CS_N, 1024, 0, st>>>(cs_buf, geo.m, geo.n,
                                            cs_buf + size_t(geo.m + geo.n) * L8CS_N);
    }
    // T is double or float. FP32 inputs produce the same bytes as the double path given those values as doubles.
    template <class T>
    void launch(const T *x, int64_t ld, const int *shifts, unsigned char *packed, int side_b,
                cudaStream_t stream = 0) const {
        if (a_direct) {
            if constexpr (std::is_same<T, double>::value) {
                Encoder::launch(x, ld, shifts, packed, side_b, stream);
                return;
            } else {
                throw std::runtime_error("l8e: a_direct (the generic encoder) takes double only");
            }
        }
        by_extra([&](auto ex) {
            constexpr int E = decltype(ex)::value;
            if constexpr (((E & 256) != 0 && ((kFastFx & 16) == 0 || !ct_pairs_ok())) ||
                          ((E & 1024) != 0 && ((kFastFx & 24) != 24 || !cs_tables_ok()))) {
                throw std::runtime_error(
                    "l8e: extra_fx 256 needs FX 16 and the paired table set, 1024 needs FX 24 and the counting tables");
            } else if (nosum) {
                if constexpr ((E & 1024) != 0) {
                    throw std::runtime_error("l8e: nosum is with extra_fx 0 / 256 only");
                } else if (side_b && skip_b_pad) {
                    launch_minb<kFastFx | 4 | E | 16384>(x, ld, shifts, packed, side_b, stream);
                } else {
                    launch_minb<kFastFx | E | 16384>(x, ld, shifts, packed, side_b, stream);
                }
            } else {
                if (side_b && skip_b_pad) {
                    launch_minb<kFastFx | 4 | E>(x, ld, shifts, packed, side_b, stream);
                } else {
                    launch_minb<kFastFx | E>(x, ld, shifts, packed, side_b, stream);
                }
            }
        });
    }
    template <int FX, class T>
    void launch_minb(const T *x, int64_t ld, const int *shifts, unsigned char *packed, int side_b,
                     cudaStream_t stream) const {
        switch (minb ? minb : kFastMinb) {
        case 3:
            launch_x<0, true, FX, 3, T>(*this, x, ld, shifts, packed, side_b, stream,
                                        dlutx[side_b]);
            return;
        case 4:
            launch_x<0, true, FX, 4, T>(*this, x, ld, shifts, packed, side_b, stream,
                                        dlutx[side_b]);
            return;
        case 5:
            launch_x<0, true, FX, 5, T>(*this, x, ld, shifts, packed, side_b, stream,
                                        dlutx[side_b]);
            return;
        case 6:
            launch_x<0, true, FX, 6, T>(*this, x, ld, shifts, packed, side_b, stream,
                                        dlutx[side_b]);
            return;
        default:
            throw std::runtime_error("l8e: FastEncoder minb must be 0 or 3 to 6");
        }
    }
    // shift A, shift B, encode A, encode B on one stream (the call's whole encoding when neither side is reused). With PDL: shift A
    // triggers at its start; shift B (independent of A) triggers at its start and waits just before its end; encode A copies its
    // LUT and stages A (inputs only) before its wait, then reads the shifts and triggers (FX 32); encode B triggers at its start and
    // waits before every exit (FX 64): it launches only after every encode A block passed its wait, so shift B has finished, and
    // its end wait makes "encode B finished" imply "encode A finished" for whatever the stream runs next (a plain launch, or a PDL
    // launch that calls griddepcontrol.wait before reading the packed planes or the shifts). Nothing may be put on the stream
    // between these four (an event there would cut the overlap, other work would break the reasoning above).
    template <class T>
    void chain(const T *A, int64_t lda, const T *B, int64_t ldb, int *ea, int *eb, int *invalid,
               unsigned char *pa, unsigned char *pb, cudaStream_t st = 0) const {
        const int mb = minb ? minb : kFastMinb;
        if (extra_fx & 1024) {
            cs_zero(st); // before the four kernels (nothing may go between them)
        }
        const bool pdl_form =
            (pdl || pdl_kernels_without_launch_attr) && !a_direct && (mb == 3 || mb == 6);
        const bool attr = pdl_form && pdl && pdl_supported && !pdl_kernels_without_launch_attr;
        const dim3 gb(unsigned(geo.n));
        if (!pdl_form) {
            shift_a<0>(A, lda, ea, invalid, flag_value, st);
            launch_pdl(shift_rows_p<1, 0, T>, gb, 0, st, false, B, eb, invalid, flag_value, geo.n,
                       geo.k, ldb, bits);
            launch(A, lda, ea, pa, 0, st);
            launch(B, ldb, eb, pb, 1, st);
            if (extra_fx & 1024) {
                cs_check(st);
            }
            return;
        }
        shift_a<3>(A, lda, ea, invalid, flag_value, st);
        launch_pdl(shift_rows_p<1, 2, T>, gb, 0, st, attr, B, eb, invalid, flag_value, geo.n, geo.k,
                   ldb, bits);
        by_extra([&](auto ex) {
            constexpr int E = decltype(ex)::value;
            if constexpr (((E & 256) != 0 && ((kFastFx & 16) == 0 || !ct_pairs_ok())) ||
                          ((E & 1024) != 0 && ((kFastFx & 24) != 24 || !cs_tables_ok()))) {
                throw std::runtime_error(
                    "l8e: extra_fx 256 needs FX 16 and the paired table set, 1024 needs FX 24 and the counting tables");
            } else if (nosum) {
                if constexpr ((E & 1024) != 0) {
                    throw std::runtime_error("l8e: nosum is with extra_fx 0 / 256 only");
                } else {
                    chain_enc<kFastFx | E | 16384>(A, lda, B, ldb, ea, eb, pa, pb, st, attr, mb);
                }
            } else {
                chain_enc<kFastFx | E>(A, lda, B, ldb, ea, eb, pa, pb, st, attr, mb);
            }
        });
        if (extra_fx & 1024) {
            cs_check(st); // plain launch: encode B's end wait makes it follow encode A too
        }
    }

    template <int E, class T>
    void chain_fx(const T *A, int64_t lda, const T *B, int64_t ldb, int *ea, int *eb, int *invalid,
                  unsigned char *pa, unsigned char *pb, cudaStream_t st = 0) const {
        const int mb = minb ? minb : kFastMinb;
        if (!(pdl || pdl_kernels_without_launch_attr) || a_direct || (mb != 3 && mb != 6)) {
            throw std::runtime_error("l8e: chain_fx is for the PDL forms");
        }
        if (nosum != ((E & 16384) != 0)) {
            throw std::runtime_error("l8e: chain_fx's FX 16384 must match nosum");
        }
        const bool attr = pdl && pdl_supported && !pdl_kernels_without_launch_attr;
        if constexpr ((E & 1024) != 0) {
            cs_zero(st);
        }
        shift_a<3>(A, lda, ea, invalid, flag_value, st);
        launch_pdl(shift_rows_p<1, 2, T>, dim3(unsigned(geo.n)), 0, st, attr, B, eb, invalid,
                   flag_value, geo.n, geo.k, ldb, bits);
        chain_enc<kFastFx | E>(A, lda, B, ldb, ea, eb, pa, pb, st, attr, mb);
        if constexpr ((E & 1024) != 0) {
            cs_check(st);
        }
    }
    template <int F, class T>
    void chain_enc(const T *A, int64_t lda, const T *B, int64_t ldb, const int *ea, const int *eb,
                   unsigned char *pa, unsigned char *pb, cudaStream_t st, bool attr, int mb) const {
        if (mb == 3) {
            enc_pdl<F | 32, 3>(A, lda, ea, pa, 0, st, attr);
            if (skip_b_pad) {
                enc_pdl<F | 4 | 64, 3>(B, ldb, eb, pb, 1, st, attr);
            } else {
                enc_pdl<F | 64, 3>(B, ldb, eb, pb, 1, st, attr);
            }
        } else {
            enc_pdl<F | 32, 6>(A, lda, ea, pa, 0, st, attr);
            if (skip_b_pad) {
                enc_pdl<F | 4 | 64, 6>(B, ldb, eb, pb, 1, st, attr);
            } else {
                enc_pdl<F | 64, 6>(B, ldb, eb, pb, 1, st, attr);
            }
        }
    }
    // shift A in the form shift_form picks (W: 0 plain, 3 the chain's first kernel: triggers its dependents at the start)
    template <int W, class T>
    void shift_a(const T *A, int64_t lda, int *ea, int *invalid, int flagval,
                 cudaStream_t st) const {
        const int m = geo.m;
        if (shift_form == 1 && m < 768) {
            launch_pdl(shift_cols_p<2, 4, W, T>, dim3(unsigned((m + 1) / 2)), 0, st, false, A, ea,
                       invalid, flagval, m, geo.k, lda, bits);
        } else if (shift_form == 1 && m <= 2048) {
            launch_pdl(shift_cols_p<4, 4, W, T>, dim3(unsigned((m + 3) / 4)), 0, st, false, A, ea,
                       invalid, flagval, m, geo.k, lda, bits);
        } else {
            launch_pdl(shift_cols_p<8, 1, W, T>, dim3(unsigned((m + 7) / 8)), 0, st, false, A, ea,
                       invalid, flagval, m, geo.k, lda, bits);
        }
    }
    // Encoder::shift with shift_form for A (shift_form 0: Encoder::shift itself); B always today's form. Float inputs: the same
    // shifts from this file's kernels (Encoder::shift takes double only; shift_rows_p / shift_cols_p compute what it computes)
    template <class T>
    void shift(const T *x, int *shifts, int *invalid, int side_b, int64_t ld,
               cudaStream_t stream = 0) const {
        if constexpr (std::is_same<T, double>::value) {
            if (side_b || shift_form == 0) {
                Encoder::shift(x, shifts, invalid, side_b, ld, stream);
                return;
            }
            shift_a<0>(x, ld, shifts, invalid, 1, stream);
        } else {
            if (side_b) {
                launch_pdl(shift_rows_p<1, 0, T>, dim3(unsigned(geo.n)), 0, stream, false, x,
                           shifts, invalid, 1, geo.n, geo.k, ld, bits);
            } else {
                shift_a<0>(x, ld, shifts, invalid, 1, stream);
            }
        }
    }
    template <int FX, int MB, class T>
    void enc_pdl(const T *x, int64_t ld, const int *shifts, unsigned char *packed, int side_b,
                 cudaStream_t st, bool attr) const {
        static bool smem_set = false;
        if (!smem_set) {
            if (cudaFuncSetAttribute(encode_l8x<0, true, FX, MB, T>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     dyn_smem_x(0, FX)) != cudaSuccess) {
                throw std::runtime_error("l8e: encode_l8x attribute");
            }
            smem_set = true;
        }
        const unsigned short *tab =
            (FX & 16) ? reinterpret_cast<const unsigned short *>(dlutx[side_b]) : dlut[side_b];
        const dim3 grid(4, unsigned(geo.cuts), unsigned(tiles(side_b) * 2));
        launch_pdl(encode_l8x<0, true, FX, MB, T>, grid, size_t(dyn_smem_x(side_b, FX)), st, attr,
                   x, ld, shifts, packed, (const ModC *)dmods, tab, side_b, geo);
    }
    void release_x() {
        cudaFree(dlutx[0]);
        cudaFree(dlutx[1]);
        dlutx[0] = dlutx[1] = nullptr;
        cudaFree(cs_buf);
        cs_buf = nullptr;
        geo.cs = nullptr;
    }
    void release() {
        release_x();
        Encoder::release();
    }
};

} // namespace l8e
