// Check the table premises on the host before any launch. A failed premise throws:
// P1 every accumulator is exact in FP32: sum over its products of w * max|plane A| * max|plane B| * K <= 2^24 - 1
// w is the decoded product scale; plane maxima come from the encoder lookup tables.
// P2 the payout's t = D0 + gamma D1 + bias lies in [0, 2^32) and bias is a multiple of p (Barrett with one correction,
// mmagic = floor(2^32 / p));
// P3 products name planes < F and accumulators < nacc <= 2; every accumulator has exactly one first write;
// P4 the ring: cuts per group = FTS / F, groups = ceil(Cuts / cpg), gbase / pbase / gt are the running sums, and every
// group's plane tiles are a multiple of BoxTiles (the tensor copies of the supply).
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
        int mx[4] = {0, 0, 0, 0}; // up to 4 planes, values from the table (l8_face_value)
        if (F < 1 || F > 4) {
            fail("planes out of 1..4" + at);
        }
        for (int r = 0; r < m.p; ++r) {
            for (int j = 0; j < F; ++j) {
                mx[j] = std::max(mx[j], std::abs(l8_face_value(q, r, j)));
            }
        }
        if (m.nprod < 1 || m.nprod > 10 || m.nacc < 1 || m.nacc > 2) {
            fail("product / accumulator count" + at); // 10
        }
        double bound[2] = {0, 0};
        int firsts[2] = {0, 0};
        for (int i = 0; i < m.nprod; ++i) {
            const unsigned e = m.prod[i];
            const int fa = e & 3, fb = (e >> 2) & 3, acc = (e >> 4) & 1,
                      bs = int((e >> 5) & 3) | int((e >> 8) & 1) << 2, fst = (e >> 7) & 1,
                      as_ = int((e >> 9) & 7); // bslot bit 8, aslot 9-11
            if (fa >= F || fb >= F || acc >= m.nacc) {
                fail("product names a plane or accumulator out of range" + at);
            }
            if (bs > Geometry::scale_slots - 1 || as_ > Geometry::scale_slots - 1) {
                fail("a scale slot out of range" + at); // Geometry bounds the TMEM scale slots.
            }
            const double w = (as_ == 0 ? 2.0 : l8k_e4m3(m.code[as_ - 1])) *
                             (bs == 0 ? 2.0 : l8k_e4m3(m.code[bs - 1])) / 4.0; // A x B
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
