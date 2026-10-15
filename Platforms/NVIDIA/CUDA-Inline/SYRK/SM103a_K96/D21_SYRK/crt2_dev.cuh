// Two-group CRT reconstructs the same u in [0,P) as the four-limb form.
// The moduli are split (crt2_split.h, at build time) into G1 = parts crt2_idx(0 .. CRT2_N1-1) and G2 = the rest with
// P1 = prod G1 <= P2 = prod G2 < 2^63 (the host checks it and sets sh.crt2 = 0 otherwise).
// Per group g: W_gq = B_gq ((B_gq^-1 lam0inv_q) mod p_q), B_gq = P_g / p_q, so u_g = sum_q r_q W_gq mod P_g = u mod P_g.
// s_g = sum r_q W_gq is taken mod 2^64 from two 32-bit limb column sums (each < 16 * 255 * 2^32 < 2^44).
// f_g = sum r_q fx_q with fx_q = ceil(W_gq 2^20 / P_g) <= 2^20 (f_g < 16 * 255 * 2^20 < 2^32):
// s_g / P_g <= f_g 2^-20 < s_g / P_g + 16 * 255 * 2^-20 < s_g / P_g + 1/256, so n = f_g >> 20 is floor(s_g / P_g) or one more,
// s_g - n P_g lies in [-P_g, P_g) (|.| < 2^63, exact as a signed 64-bit wrap), and one conditional add gives u_g.
// Garner: u = u1 + P1 t, t = (u2 - u1) P1^-1 mod P2 (u1 < P1 <= P2, so u2 - u1 needs one conditional add), the product mod P2 by
// Shoup's method with c' = floor(c 2^64 / P2) (result in [0, 2 P2), one conditional subtraction). u < P1 + P1 (P2 - 1) = P.
// The including file declares crt2_wl[kParts][2], crt2_fx[kParts], crt2_c[4] = {P1, P2, c, c'}, kParts, CRT2_N1, crt2_idx.
#ifndef CRT2_QUAL
#define CRT2_QUAL __device__ __forceinline__
#endif
CRT2_QUAL f64_impl::U128 crt2_reduce(const unsigned (&r)[kParts]) {
    static_assert(CRT2_N1 >= 1 && CRT2_N1 < kParts && CRT2_N1 <= 16 && kParts - CRT2_N1 <= 16,
                  "crt2 group sizes");
    unsigned long long a0 = 0, a1 = 0, b0 = 0, b1 = 0;
    unsigned fa = 0, fb = 0;
#pragma unroll
    for (int i = 0; i < kParts; ++i) {
        const unsigned rq = r[crt2_idx(i)];
        if (i < CRT2_N1) {
            a0 += (unsigned long long)crt2_wl[i][0] * rq;
            a1 += (unsigned long long)crt2_wl[i][1] * rq;
            fa += crt2_fx[i] * rq;
        } else {
            b0 += (unsigned long long)crt2_wl[i][0] * rq;
            b1 += (unsigned long long)crt2_wl[i][1] * rq;
            fb += crt2_fx[i] * rq;
        }
    }
    const unsigned long long P1 = crt2_c[0], P2 = crt2_c[1];
    long long d1 = (long long)(a0 + (a1 << 32) - (unsigned long long)(fa >> 20) * P1);
    if (d1 < 0) {
        d1 += (long long)P1;
    }
    long long d2 = (long long)(b0 + (b1 << 32) - (unsigned long long)(fb >> 20) * P2);
    if (d2 < 0) {
        d2 += (long long)P2;
    }
    const unsigned long long u1 = (unsigned long long)d1, u2 = (unsigned long long)d2;
    const unsigned long long e = u2 >= u1 ? u2 - u1 : u2 - u1 + P2;
#ifdef __CUDA_ARCH__
    const unsigned long long qh = __umul64hi(e, crt2_c[3]);
#else
    const unsigned long long qh = (unsigned long long)(((unsigned __int128)e * crt2_c[3]) >> 64);
#endif
    unsigned long long t = e * crt2_c[2] - qh * P2;
    if (t >= P2) {
        t -= P2;
    }
    return f64_impl::U128(P1) * t + u1;
}
