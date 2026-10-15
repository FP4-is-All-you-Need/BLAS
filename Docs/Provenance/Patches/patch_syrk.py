"""Schedule the lower-triangle SYRK tiles in one launch.

With L8_SYRK=1 in the environment at create (m == n; the caller passes B = A^T for SYRK, A^T without conjugation for ZSYRK):
  - the plan lists the tiles that hold an element with row >= col (tile (tr, tc) of 256 x 224: tc * Cols < (tr + 1) * Side), in the
    order of l8_tile_row / l8_tile_col (bands of 8 tile rows, column-major inside a band), packed (tr << 16) | tc in a device table;
    sh.tiles, the residue planes, the metrics and the tail split all use that count (the tiles above the diagonal are never handed
    to a cluster, so no cluster idles on an empty tile);
  - l8_tile_row / l8_tile_col read the table when sh.tmap is set (every caller: supply, issue, payout, the in-kernel CRT, l8k_crt_after);
  - the CRT is where it was (in the kernel behind the next tile's MMAs); an element above the diagonal (col > row) is not written,
    so C's strict upper triangle keeps the caller's values (BLAS SYRK with uplo L);
  - the encoder still encodes all of A (m rows) and all of B (n columns); the arithmetic of every written element is unchanged,
    so the lower triangle is bit-identical to the full product's / the strips' (B's column exponents are per column).
  - d_info / z_info print "syrk":1.  L8_SYRK with ZL8_HERK set stops at create.
Without L8_SYRK nothing changes (sh.tmap = nullptr, sh.syrk = 0).
usage: patch_syrk.py TREE_COPY   (a copy of a request tree: d53jg_crtfix, s14_crtfix, zl8f tailpace)"""
import os
import sys
from pathlib import Path

T = Path(sys.argv[1])


def edit(name, pairs, optional=False):
    p = T / name
    if not p.exists():
        if optional:
            return
        raise SystemExit(f'{name}: missing')
    s = p.read_text()
    for old, new, count in pairs:
        c = s.count(old)
        if c != count:
            raise SystemExit(f'{name}: {c} matches (want {count}) for {old[:80]!r}')
        s = s.replace(old, new)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(s)
    os.replace(tmp, p)


# ---- kernel: the shape fields, the tile map, the store guard ----
k = (T / 'l8k_kernel_rt.cuh').read_text()
i = k.find('struct L8Shape {')
j = k.find('\n};\n', i)
if i < 0 or j < 0:
    raise SystemExit('L8Shape not found')
k = k[:j] + ('\n  const unsigned* tmap;              // Lower-triangle tile map: (row << 16) | column.'
             '\n  int syrk;                          // When set, leave elements above the diagonal unchanged.') + k[j:]
old_row = '__device__ __forceinline__ unsigned l8_tile_row(const L8Shape& sh, unsigned tile) {\n'
old_col = '__device__ __forceinline__ unsigned l8_tile_col(const L8Shape& sh, unsigned tile) {\n'
for old, new in [(old_row, old_row + '  if (sh.tmap) return __ldg(sh.tmap + tile) >> 16;   // Read the packed tile row.\n'),
                 (old_col, old_col + '  if (sh.tmap) return __ldg(sh.tmap + tile) & 0xffffu;   // Read the packed tile column.\n')]:
    if k.count(old) != 1:
        raise SystemExit('tile function: %d matches' % k.count(old))
    k = k.replace(old, new)
real_store = 'cout[size_t(global_n) * sh.ldc + row] = crt_elem_rt(r, P, er - eb[global_n], sh);'
cplx_skip = '    if (global_n >= unsigned(sh.n)) continue;\n'
if k.count(real_store) == 2:     # the CRT stage's crt_cols_rt (prefetch and plain loops): keep the prefetch rotation, skip the store
    k = k.replace(real_store, 'if (!(sh.syrk && global_n > row)) ' + real_store + '   // Store only the requested lower triangle.')
elif k.count(cplx_skip) == 1 and k.count(real_store) == 0:   # crt_cols_rt (complex and real in one loop, no prefetch)
    k = k.replace(cplx_skip, cplx_skip + '    if (sh.syrk && global_n > row) continue;   // Preserve the strict upper triangle of C.\n')
else:
    raise SystemExit('crt store form not recognized (%d / %d)' % (k.count(real_store), k.count(cplx_skip)))
tmp = T / 'l8k_kernel_rt.cuh.tmp'
tmp.write_text(k)
os.replace(tmp, T / 'l8k_kernel_rt.cuh')

# ---- plan: build the table at create, free it at destroy ----
SYRK_BLOCK = r'''    sh.tmap = nullptr; sh.syrk = 0;
    if (const char* e_syrk = getenv("L8_SYRK")) if (atoi(e_syrk) != 0) {   // Schedule only tiles that intersect the lower triangle.
      if (m != n) throw std::runtime_error("L8_SYRK needs m == n");
      if (const char* e_h = getenv("ZL8_HERK")) if (atoi(e_h) != 0) throw std::runtime_error("L8_SYRK and ZL8_HERK together");
      std::vector<unsigned> tl;
      for (int band = 0; band * 8 < sh.tile_rows; ++band) {   // l8_tile_row / l8_tile_col's order without the tiles above the diagonal
        const int rb = std::min(8, sh.tile_rows - band * 8);
        for (int tc = 0; tc < sh.tile_cols; ++tc)
          for (int r = 0; r < rb; ++r) {
            const int tr = band * 8 + r;
            if ((long long)tc * Cols < (long long)(tr + 1) * Side) tl.push_back((unsigned(tr) << 16) | unsigned(tc));
          }
      }
      if (tl.empty() || sh.tile_rows > 65535 || sh.tile_cols > 65535) throw std::runtime_error("L8_SYRK: no tile");
      ck(cudaMalloc(&tmap_dev, tl.size() * sizeof(unsigned)));
      ck(cudaMemcpy(tmap_dev, tl.data(), tl.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
      sh.tmap = tmap_dev; sh.syrk = 1; sh.tiles = int(tl.size());
    }
'''
edit('l8rt_plan.cuh', [
    ('  unsigned short* resid = nullptr;\n',
     '  unsigned short* resid = nullptr;\n  unsigned* tmap_dev = nullptr;   // Device allocation for the lower-triangle tile map.\n', 1),
    ('    sh.m = m; sh.n = n; sh.cuts = cuts;\n    sh.tile_rows = (m + Side - 1) / Side; sh.tile_cols = (n + Cols - 1) / Cols; sh.tiles = sh.tile_rows * sh.tile_cols;\n',
     '    sh.m = m; sh.n = n; sh.cuts = cuts;\n    sh.tile_rows = (m + Side - 1) / Side; sh.tile_cols = (n + Cols - 1) / Cols; sh.tiles = sh.tile_rows * sh.tile_cols;\n'
     + SYRK_BLOCK, 1),
    ('    cudaFree(pa); cudaFree(pb); cudaFree(resid);',
     '    cudaFree(tmap_dev); tmap_dev = nullptr;   // Clear the freed tile-map pointer.\n    cudaFree(pa); cudaFree(pb); cudaFree(resid);', 1),
])
for f in ('l8rt_dtab.cuh', 'zl8_dtab.cuh'):
    edit(f, [('\\"nchunk\\":%d,', '\\"nchunk\\":%d,\\"syrk\\":%d,', 1), ('p.sh.nchunk,', 'p.sh.nchunk, p.sh.syrk,', 1)],
         optional=(f == 'zl8_dtab.cuh'))
print('patched', T)
