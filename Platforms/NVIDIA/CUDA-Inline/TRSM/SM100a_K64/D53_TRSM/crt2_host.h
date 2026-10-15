// host constants of crt2_dev.cuh. p[] / lam0inv[] in part order (the plan's hm[]).
// wl[i] / fx[i] belong to part crt2_idx(i); c = {P1, P2, P1^-1 mod P2, floor(c 2^64 / P2)}. False when the build's split does not
// give P1 * P2 = P with P1 <= P2 < 2^63 (the caller then keeps the 4-limb CRT).
#pragma once
#ifndef CRT2_FXB
#define CRT2_FXB 16 // the same value as crt2_dev.cuh / crt2c_dev.cuh
#endif
inline bool crt2_make(const int *p, const unsigned *lam0inv, unsigned (*wl)[2], unsigned *fx,
                      unsigned long long *c) {
    typedef unsigned __int128 u128;
    u128 P1 = 1, P2 = 1;
    for (int i = 0; i < kParts; ++i) {
        const int q = crt2_idx(i);
        if (p[q] < 2 || p[q] > 65535) {
            return false; // 1- or 2-byte residues
        }
        (i < CRT2_N1 ? P1 : P2) *= u128(unsigned(p[q]));
        if (P1 >= (u128(1) << 63) || P2 >= (u128(1) << 63)) {
            return false;
        }
    }
    if (!(P1 <= P2)) {
        return false;
    }
    { // R_g = sum over each group of (p_q - 1) < 2^16 keeps f_g < 2^32 and the estimate within one (crt2_dev.cuh)
        unsigned long long R1 = 0, R2 = 0;
        for (int i = 0; i < kParts; ++i) {
            (i < CRT2_N1 ? R1 : R2) += unsigned(p[crt2_idx(i)]) - 1;
        }
        if (R1 >= (1ull << 16) || R2 >= (1ull << 16)) {
            return false;
        }
    }
    for (int i = 0; i < kParts; ++i) {
        const int q = crt2_idx(i);
        const unsigned pq = unsigned(p[q]);
        const u128 Pg = i < CRT2_N1 ? P1 : P2, B = Pg / pq;
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
        fx[i] = unsigned(((W << CRT2_FXB) + Pg - 1) / Pg); // ceil, <= 2^FXB
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
