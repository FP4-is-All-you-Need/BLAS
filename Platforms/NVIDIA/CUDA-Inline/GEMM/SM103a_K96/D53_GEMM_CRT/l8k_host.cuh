

#pragma once
#include <stdexcept>
#include <string>
#include <cmath>

inline double l8k_e4m3(unsigned code) {
    const int e = int(code >> 3) & 15, m = int(code & 7);
    return e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
}

inline void l8k_check_tables(const L8kMod *mods, int nmod, int k, int cuts, int fts, int box_tiles,
                             bool a_direct) {
    auto fail = [](const std::string &s) { throw std::runtime_error("l8k table: " + s); };
    if (nmod != L8_NMOD) {
        fail("modulus count differs from the encoder's");
    }
    int gbase = 0, pbase = 0, gt = 0;
    for (int q = 0; q < nmod; ++q) {
        gt += (cuts + fts / mods[q].planes - 1) / (fts / mods[q].planes);
    }
    for (int q = 0; q < nmod; ++q) {
        const L8kMod &m = mods[q];
        const std::string at = " (modulus " + std::to_string(m.p) + ")";
        if (m.p != l8_mod[q]) {
            fail("modulus order differs from the encoder's" + at);
        }
        const int F = (l8_kind[q] == 1 && a_direct) ? 2 : l8_planes[q];
        if (m.planes != F) {
            fail("planes differ from the encoder's" + at);
        }
        // plane maxima over all residues (the A side's sign flips do not change them)
        int mx[3] = {0, 0, 0};
        for (int r = 0; r < m.p; ++r) {
            const int a0 = l8_limbs[l8_limb_off[q] + r][0], a1 = l8_limbs[l8_limb_off[q] + r][1];
            const int v[3] = {a0, a1, a0 + a1};
            for (int j = 0; j < F; ++j) {
                mx[j] = std::max(mx[j], std::abs(v[j]));
            }
        }
        if (m.nprod < 1 || m.nprod > 4 || m.nacc < 1 || m.nacc > 2) {
            fail("product / accumulator count" + at);
        }
        double bound[2] = {0, 0};
        int firsts[2] = {0, 0};
        for (int i = 0; i < m.nprod; ++i) {
            const unsigned e = m.prod[i];
            const int fa = e & 3, fb = (e >> 2) & 3, acc = (e >> 4) & 1, bs = (e >> 5) & 3,
                      fst = (e >> 7) & 1;
            if (fa >= F || fb >= F || acc >= m.nacc) {
                fail("product names a plane or accumulator out of range" + at);
            }
            const double w = bs == 0 ? 1.0 : l8k_e4m3(m.code[bs - 1]) / 2.0;
            if (w != std::floor(w) || w < 1) {
                fail("weight not a positive integer" + at);
            }
            bound[acc] += w * mx[fa] * mx[fb] * double(k);
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
            const int nc = std::min(cpg, cuts - g * cpg);
            if ((nc * F) % box_tiles) {
                fail("P4: a group's plane tiles are not a multiple of the copy box" + at);
            }
        }
        gbase += groups;
        pbase += F;
    }
}
