

#pragma once
#include <stdexcept>
#include <string>
#include <cmath>

inline int zl8_fp4_value(unsigned code) {
    static const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    const int v = mag[code & 7u];
    return (code & 8u) ? -v : v;
}

inline void zl8_check_tables(const L8kMod *mods, int nmod, int k, int cuts, int fts,
                             int box_tiles) {
    auto fail = [](const std::string &s) { throw std::runtime_error("zl8 table: " + s); };
    if (nmod != L8K_NMOD) {
        fail("part count differs from the table's");
    }
    int gbase = 0, pbase = 0, gt = 0;
    for (int i = 0; i < nmod; ++i) {
        gt += (cuts + fts / mods[i].planes - 1) / (fts / mods[i].planes);
    }
    for (int i = 0; i < nmod; ++i) {
        const L8kMod &m = mods[i];
        const int q = zl8_part_mod[i];
        const std::string at =
            " (part " + std::to_string(i) + ", modulus " + std::to_string(m.p) + ")";
        if (q < 0 || q >= ZL8_NMODC || m.p != zl8_p_ct(q)) {
            fail("modulus of the part differs from the encoder's" + at);
        }
        const int F = zl8_planes_of_mod[q];
        if (m.planes != F || F < 1 || F > 3) {
            fail("planes differ from the encoder's" + at);
        }
        int mx[3] = {0, 0, 0};
        for (int r = 0; r < m.p; ++r) {
            const unsigned w = zl8_lut[zl8_lut_off[q] + r];
            for (int j = 0; j < F; ++j) {
                const int v = std::abs(zl8_fp4_value((w >> (4 * j)) & 15u));
                if (v > mx[j]) {
                    mx[j] = v;
                }
            }
        }
        if (m.nprod < 1 || m.nprod > 4 || m.nacc < 1 || m.nacc > 2) {
            fail("product / accumulator count" + at);
        }
        double bound[2] = {0, 0};
        int firsts[2] = {0, 0};
        for (int j = 0; j < m.nprod; ++j) {
            const unsigned e = m.prod[j];
            const int fa = e & 3, fb = (e >> 2) & 3, acc = (e >> 4) & 1, bs = (e >> 5) & 3,
                      fst = (e >> 7) & 1;
            if (fa >= F || fb >= F || acc >= m.nacc) {
                fail("product names a plane or accumulator out of range" + at);
            }
            const double wt = bs == 0 ? 1.0 : l8k_e4m3(m.code[bs - 1]) / 2.0;
            if (wt != std::floor(wt) || wt < 1) {
                fail("weight not a positive integer" + at);
            }
            bound[acc] += wt * mx[fa] * mx[fb] * double(k);
            firsts[acc] += fst;
        }
        for (int a = 0; a < m.nacc; ++a) {
            if (firsts[a] != 1) {
                fail("an accumulator without exactly one first write" + at);
            }
            if (bound[a] > double((1u << 24) - 1)) {
                fail("P1: accumulator " + std::to_string(a) + " can exceed 2^24 - 1" + at);
            }
        }
        const double tmax = bound[0] + std::abs(double(m.gamma)) * bound[1];
        if (m.bias % m.p != 0 || double(m.bias) < tmax || double(m.bias) + tmax >= 4294967296.0) {
            fail("P2: t + bias" + at);
        }
        if (m.nacc == 1 && m.gamma != 0) {
            fail("P2: gamma on a one-accumulator modulus" + at);
        }
        if (m.mmagic != unsigned((1ull << 32) / unsigned(m.p))) {
            fail("P2: mmagic" + at);
        }
        const int cpg = fts / F, groups = (cuts + cpg - 1) / cpg;
        if (m.cpg != cpg || m.groups != groups || m.gbase != gbase || m.pbase != pbase ||
            m.gt != gt) {
            fail("P4: ring" + at);
        }
        for (int g = 0; g < groups; ++g) {
            const int nc = cpg < cuts - g * cpg ? cpg : cuts - g * cpg;
            if ((nc * F) % box_tiles) {
                fail("P4: a group's plane tiles are not a multiple of the copy box" + at);
            }
        }
        gbase += groups;
        pbase += F;
    }
}
