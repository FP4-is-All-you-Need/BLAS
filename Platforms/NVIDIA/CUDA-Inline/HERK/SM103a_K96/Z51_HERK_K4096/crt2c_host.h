

#pragma once
inline bool crt2c_make(const int *p, const unsigned *lam0inv, unsigned (*wl)[2], unsigned *fx,
                       unsigned long long *c) {
    typedef unsigned __int128 u128;
    u128 P1 = 1, P2 = 1;
    for (int i = 0; i < ZL8_NMODC; ++i) {
        const int q = crt2c_idx(i);
        if (p[q] < 2 || p[q] > 255) {
            return false;
        }
        (i < CRT2C_N1 ? P1 : P2) *= u128(unsigned(p[q]));
        if (P1 >= (u128(1) << 63) || P2 >= (u128(1) << 63)) {
            return false;
        }
    }
    if (!(P1 <= P2)) {
        return false;
    }
    for (int i = 0; i < ZL8_NMODC; ++i) {
        const int q = crt2c_idx(i);
        const unsigned pq = unsigned(p[q]);
        const u128 Pg = i < CRT2C_N1 ? P1 : P2, B = Pg / pq;
        if (B * pq != Pg) {
            return false;
        }
        unsigned inv = 0;
        for (unsigned x = 1; x < pq; ++x) {
            if (unsigned(B % pq) * x % pq == 1) {
                inv = x;
                break;
            }
        }
        if (!inv) {
            return false;
        }
        const unsigned cq = unsigned((unsigned long long)inv * (lam0inv[q] % pq) % pq);
        const u128 W = B * cq; // < Pg < 2^63
        wl[i][0] = unsigned((unsigned long long)W);
        wl[i][1] = unsigned((unsigned long long)(W >> 32));
        fx[i] = unsigned(((W << 20) + Pg - 1) / Pg); // ceil, <= 2^20
    }
    // P1^-1 mod P2 by the extended Euclid algorithm (P1, P2 coprime: distinct moduli of the table, checked by the plan)
    __int128 r0 = (__int128)P2, r1 = (__int128)(P1 % P2), s0 = 0, s1 = 1;
    while (r1) {
        const __int128 qq = r0 / r1, r2 = r0 - qq * r1, s2 = s0 - qq * s1;
        r0 = r1;
        r1 = r2;
        s0 = s1;
        s1 = s2;
    }
    if (r0 != 1) {
        return false;
    }
    if (s0 < 0) {
        s0 += (__int128)P2;
    }
    const unsigned long long cinv = (unsigned long long)s0;
    c[0] = (unsigned long long)P1;
    c[1] = (unsigned long long)P2;
    c[2] = cinv;
    c[3] = (unsigned long long)((u128(cinv) << 64) / P2);
    return true;
}
