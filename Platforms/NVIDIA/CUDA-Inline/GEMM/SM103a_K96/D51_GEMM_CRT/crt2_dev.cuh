

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
