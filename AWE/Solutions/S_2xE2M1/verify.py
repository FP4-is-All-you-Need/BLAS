# -*- coding: utf-8 -*-
"""独立検算担当 による独立検算(k<6 の直接項に対応)。

verify_sample.py の恒等式は対角 L 本 + 対 len(pairs) 本の和しか組んでおらず、
対に入っていない (i,j) を直接積 a_i b_j / a_j b_i として払う分が抜けている。
そのため k<6(r=12 など)の行は必ず verified:false になる。ここでは直接項を
足した完全な形で検算し、併せて符号化そのもの(Σ w_i a_i = n、各リムが S の元)も
独立に確かめる。Python 3.6 互換・numpy 1.14 API。
"""
from __future__ import print_function
import io, itertools, json, sys
from fractions import Fraction

import numpy as np

S = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12]
SSET = set(S)
UV_TABLE = [(1, 1), (1, -1), (1, 2), (1, -2), (2, 1), (2, -1), (2, 2), (2, -2)]


def all_pairs(L):
    return [(i, j) for i in range(L) for j in range(i + 1, L)]


out = []
for line in sys.stdin.read().splitlines():
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    L = rec["L"]
    apl = all_pairs(L)
    pairs = [apl[i] for i in rec["pairs"]]
    uvs = [UV_TABLE[i] for i in rec["uv"]]
    w = rec["best_w"]
    pmap = dict(zip(pairs, uvs))

    # --- 符号化(独立に構成し、性質も確かめる)
    enc = {}
    for t in itertools.product(S, repeat=L):
        good = True
        for (i, j), (u, v) in zip(pairs, uvs):
            if (u * t[i] + v * t[j]) not in SSET:
                good = False
                break
        if not good:
            continue
        n = sum(w[i] * t[i] for i in range(L))
        if n not in enc:
            enc[n] = t
    X = 0
    while (X + 1) in enc and -(X + 1) in enc:
        X += 1
    enc_ok = all(sum(w[i] * enc[n][i] for i in range(L)) == n
                 and all(x in SSET for x in enc[n])
                 for n in list(enc)[:5000])

    # --- 復元係数(対角・対)
    lii = []
    for i in range(L):
        x = Fraction(w[i] * w[i])
        for j in range(L):
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in pmap:
                u, v = pmap[key]
                x -= Fraction(w[i] * w[j]) * (Fraction(u, v) if i < j
                                              else Fraction(v, u))
        lii.append(x)
    lij = [Fraction(w[i] * w[j], u * v) for (i, j), (u, v) in zip(pairs, uvs)]
    lam = lii + lij
    D = 1
    for x in lam:
        D = int(D) * x.denominator // int(np.gcd(D, x.denominator))
    nk = [int(x * D) for x in lam]

    R = list(range(-X, X + 1))
    A = np.array([enc[n] for n in R], dtype=np.int64)
    acc = np.zeros((len(R), len(R)), dtype=np.int64)
    for i in range(L):
        acc += nk[i] * np.outer(A[:, i], A[:, i])
    for k, ((i, j), (u, v)) in enumerate(zip(pairs, uvs)):
        f = u * A[:, i] + v * A[:, j]
        acc += nk[L + k] * np.outer(f, f)
    # ここが追加分: 対に入っていない (i,j) の直接積
    n_direct = 0
    for i in range(L):
        for j in range(i + 1, L):
            if (i, j) in pmap:
                continue
            c = D * w[i] * w[j]
            acc += c * (np.outer(A[:, i], A[:, j]) + np.outer(A[:, j], A[:, i]))
            n_direct += 2
    Rv = np.array(R, dtype=np.int64)
    ok = bool((acc == D * np.outer(Rv, Rv)).all())

    out.append({"L": L, "r": L * L - len(pairs), "w": w,
                "X_reported": rec.get("best_X"), "X_recomputed": X,
                "X_matches": X == rec.get("best_X"),
                "D": int(D), "n_direct_products": int(n_direct),
                "encoding_ok": bool(enc_ok),
                "n_pairs_checked": len(R) ** 2,
                "verified_full": bool(ok and X == rec.get("best_X"))})

print(json.dumps(out, ensure_ascii=False, indent=1))
sys.exit(0 if all(o["verified_full"] for o in out) else 1)
