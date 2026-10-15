"""Compute TRMM (C = tril(A) B, A lower n x n, left) in one
launch with each tile's inner length cut at the last nonzero column of its rows. Apply after patch_syrk.py (uses its tile table).
With L8_TRMM=<mode> at create, mode 1 / 2 / 3 (m == n == k; the caller passes A = tril(A) with zeros above the diagonal, as the full
form does):
  - every tile of C is run (C is full), listed through patch_syrk's table: mode 1 = the engine's own order (bands of 8 tile rows,
    column-major inside a band) with the bands from the bottom up; modes 2 / 3 = whole tile rows from the bottom up (column order inside);
  - mode 1 / 2: the pace (s11 / s12, the A-side supply waits until every cluster has started part p - S) is off; mode 3 keeps the
    engine's default. Why (v3): with the pace on, a part step lasts as long as the heaviest tile in the wave; r7 (18603, mode-1 order with
    the pace on at 8192 / 16384) lost to the 4 row strips by 1.13 / 1.06 (r5 18591, whole rows with the pace on: 1.045 / 1.077);
  - a tile of tile row tr runs the cuts [0, ceil((tr + 1) Side / Kc) rounded up to even) only (Kc = the k of one cut: 96 on K96 trees,
    64 on K64), computed once per tile in the supply and issue loops (v3; v1 / v2 computed it per group), i.e. the supply and issue loops
    of every part take that many cuts (the groups of the last are partial); the skipped cuts multiply A's zeros only, so every element
    equals the full form's (same encoding of A and B, same sums, the same CRT);
  - d_info / z_info print "trmm":<mode>. Not together with L8_SYRK or ZL8_HERK.
usage: patch_trmm.py TREE_COPY"""
import os
import sys
from pathlib import Path

T = Path(sys.argv[1])


def rd(n):
    return (T / n).read_text(encoding='utf-8')


def wr(n, s):
    tmp = T / (n + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fo:
        fo.write(s)
    os.replace(tmp, T / n)


def rep(s, old, new, count=1, where=''):
    c = s.count(old)
    if c != count:
        raise SystemExit('%s: %d matches (want %d) for %r' % (where, c, count, old[:90]))
    return s.replace(old, new)


k = rd('l8k_kernel_rt.cuh')
if 'const unsigned* tmap;' not in k:
    raise SystemExit('apply patch_syrk.py first')
k = rep(k, '\n  int syrk;                          // When set, leave elements above the diagonal unchanged.',
        '\n  int syrk;                          // When set, leave elements above the diagonal unchanged.'
        '\n  int trmm, tcut_k;                  // TRMM mode (0 = off) and the number of K elements per cut.', where='shape')
old_col = '__device__ __forceinline__ unsigned l8_tile_col(const L8Shape& sh, unsigned tile) {\n'
i = k.index(old_col)
j = k.index('\n}\n', i) + 3
k = k[:j] + ('// Stop after the last nonzero column of this lower-triangular tile row.\n'
             '__device__ __forceinline__ int l8_tcuts(const L8Shape& sh, unsigned tile, int cuts) {\n'
             '  if (!sh.trmm) return cuts;\n'
             '  int need = int(((l8_tile_row(sh, tile) + 1u) * unsigned(Side) + unsigned(sh.tcut_k) - 1u) / unsigned(sh.tcut_k));\n'
             '  need = (need + 1) / 2 * 2;   // even, as cuts_of() rounds (the extra cut multiplies zeros only)\n'
             '  return need < cuts ? need : cuts;\n'
             '}\n') + k[j:]
# supply: the tile's cuts once per tile
k = rep(k, '      const size_t block = size_t(side_b ? l8_tile_col(sh, tile) : l8_tile_row(sh, tile)) * block_tiles;   // in plane tiles\n',
        '      const size_t block = size_t(side_b ? l8_tile_col(sh, tile) : l8_tile_row(sh, tile)) * block_tiles;   // in plane tiles\n'
        '      const int tcuts = l8_tcuts(sh, tile, cuts);   // Number of K cuts needed by this tile.\n', where='supply tcuts')
k = rep(k, '        const int F = mods[part].planes, cpg = mods[part].cpg, groups = mods[part].groups;\n',
        '        const int F = mods[part].planes, cpg = mods[part].cpg, groups = sh.trmm ? (tcuts + cpg - 1) / cpg : mods[part].groups;   // Use the tile-specific group count for TRMM.\n',
        where='supply groups')
k = rep(k, '          const int cut0 = g * cpg, nc = min(cpg, cuts - cut0);\n',
        '          const int cut0 = g * cpg, nc = min(cpg, tcuts - cut0);   // Clip the final group to the tile length.\n', where='supply nc')
# issue: the same, once per tile
issue_tile = '      const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);\n      for (int part = p0_of(w); part < p1_of(w); ++part) {\n        const int step = scnt++;\n'
k = rep(k, issue_tile,
        '      const unsigned w = unsigned(k) * sh.clusters + q, tile = tile_of(w);\n'
        '      const int tcuts = l8_tcuts(sh, tile, cuts);   // Number of K cuts needed by this tile.\n'
        '      for (int part = p0_of(w); part < p1_of(w); ++part) {\n        const int step = scnt++;\n', where='issue tcuts')
k = rep(k, '        const int first = gcnt, last = first + mods[part].groups;\n',
        '        const int first = gcnt, last = first + (sh.trmm ? (tcuts + cpg - 1) / cpg : mods[part].groups);   // Advance by the group count for this tile.\n', where='issue groups')
k = rep(k, '              const int g = G - first, nc = min(cpg, cuts - g * cpg);\n',
        '              const int g = G - first, nc = min(cpg, tcuts - g * cpg);   // Clip the final issued group to the tile length.\n', where='issue nc')
wr('l8k_kernel_rt.cuh', k)

s = rd('l8rt_plan.cuh')
anchor = '    sh.tmap = nullptr; sh.syrk = 0;\n'
blk = r'''    sh.trmm = 0; sh.tcut_k = 0;
    if (const char* e_trmm = getenv("L8_TRMM")) if (atoi(e_trmm) != 0) {   // Select the left-lower TRMM tile schedule.
      const int mode = atoi(e_trmm);
      if (mode < 1 || mode > 3) throw std::runtime_error("L8_TRMM: mode 1, 2 or 3");
      if (m != n || k != n) throw std::runtime_error("L8_TRMM needs m == n == k");
      if (const char* e_s = getenv("L8_SYRK")) if (atoi(e_s) != 0) throw std::runtime_error("L8_TRMM and L8_SYRK together");
      if (const char* e_h = getenv("ZL8_HERK")) if (atoi(e_h) != 0) throw std::runtime_error("L8_TRMM and ZL8_HERK together");
      int cutk = 0;   // the k of one cut: cuts_of(2c) == 2 and cuts_of(2c + 1) == 4
      for (int c : {64, 96}) if (l8e::cuts_of(2 * c) == 2 && l8e::cuts_of(2 * c + 1) == 4) cutk = c;
      if (!cutk) throw std::runtime_error("L8_TRMM: the cut width is neither 64 nor 96");
      if (sh.tile_rows > 65535 || sh.tile_cols > 65535) throw std::runtime_error("L8_TRMM: too many tiles");
      std::vector<unsigned> tl;
      if (mode == 1) {   // the engine's own order (bands of 8 tile rows, column-major inside), the bands from the bottom up
        for (int band = (sh.tile_rows + 7) / 8 - 1; band >= 0; --band) {
          const int rb = std::min(8, sh.tile_rows - band * 8);
          for (int tc = 0; tc < sh.tile_cols; ++tc)
            for (int r = 0; r < rb; ++r) tl.push_back((unsigned(band * 8 + r) << 16) | unsigned(tc));
        }
      } else {           // whole tile rows from the bottom up: a wave of clusters holds tiles of one or two tile rows (equal cuts)
        for (int tr = sh.tile_rows - 1; tr >= 0; --tr)
          for (int tc = 0; tc < sh.tile_cols; ++tc) tl.push_back((unsigned(tr) << 16) | unsigned(tc));
      }
      ck(cudaMalloc(&tmap_dev, tl.size() * sizeof(unsigned)));
      ck(cudaMemcpy(tmap_dev, tl.data(), tl.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
      sh.tmap = tmap_dev; sh.trmm = mode; sh.tcut_k = cutk;
    }
'''
# insert after patch_syrk's block (which starts with the anchor and ends with the closing brace of its if)
i = s.index(anchor)
j = s.index('      sh.tmap = tmap_dev; sh.syrk = ', i)
j = s.index('\n    }\n', j) + len('\n    }\n')
s = s[:j] + blk + s[j:]
# after the pace's default (the trees differ in its condition: the tailpace tree also paces the split tail)
if s.count('getenv("L8_PACE")') != 1:
    raise SystemExit('pace: L8_PACE getenv not found once')
i = s.index('getenv("L8_PACE")')
j = s.index('\n#endif\n', i) + len('\n#endif\n')
if '#if L8_S12_DEFAULTS' not in s[i:j] or 'sh.pace_s = 1;' not in s[i:j]:
    raise SystemExit('pace: the default block is not where expected')
s = s[:j] + '    if (sh.trmm == 1 || sh.trmm == 2) sh.pace_s = -1;   // Modes 1 and 2 disable pacing.\n' + s[j:]
wr('l8rt_plan.cuh', s)
for f in ('l8rt_dtab.cuh', 'zl8_dtab.cuh'):
    if not (T / f).exists():
        continue
    d = rd(f)
    d = rep(d, '\\"syrk\\":%d,', '\\"syrk\\":%d,\\"trmm\\":%d,\\"pace_on\\":%d,', where=f)
    d = rep(d, 'p.sh.syrk,', 'p.sh.syrk, p.sh.trmm, p.sh.pace_s >= 0 ? 1 : 0,', where=f)
    wr(f, d)
print('patched', T)
