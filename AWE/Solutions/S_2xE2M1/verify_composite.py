# -*- coding: utf-8 -*-
"""合成12積(外(2,2)カラツバ⊗内(2,2)直接)の独立検算 — 独立検算担当 2026-08-19。

record関門(i): 構成担当の構成(L=4・w=(1,9,42,378)・X=1680・D=1)を、
報告のλを使わずに一から検算する:
  (1) 12積の面対から復元係数を16本の線形方程式として分数で厳密求解
  (2) 面族6本の下で合法な S^4 のリム組を独立に列挙し、±X の連続被覆を再計算
  (3) ±X × ±X の全数対で恒等式を整数(共通分母倍)で照合
  (4) 解いたλを報告のλ(集合として)と突き合わせ

積の構成(合成の定義から):
  リム順 (0,1,2,3) = 内(1,9)が隣接対・外(1,42)がブロック対。w = (1,9)⊗(1,42)。
  面族: e0..e3、S0=(u,0,v,0)、S1=(0,u,0,v)。
  12積 = ブロック0×0の内直接4 + ブロック1×1の内直接4 + 和面×和面の内直接4。
"""
from __future__ import print_function
import itertools, json, sys
from fractions import Fraction as Fr

import numpy as np

S = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12]
SSET = set(S)
W = [1, 9, 42, 378]
X_CLAIM = 1680
LAM_CLAIM = [-369, -369, -41, -3321, 15498, 15498, 1722, 139482, 42, 378, 378, 3402]


def products_of(u, v):
    e = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    s0 = (u, 0, v, 0)
    s1 = (0, u, 0, v)
    P = []
    for fa, fb in itertools.product([e[0], e[1]], repeat=2):
        P.append((fa, fb))
    for fa, fb in itertools.product([e[2], e[3]], repeat=2):
        P.append((fa, fb))
    for fa, fb in itertools.product([s0, s1], repeat=2):
        P.append((fa, fb))
    return P


def solve_exact(A_rows, b_vec):
    m = [[Fr(x) for x in row] + [Fr(bv)] for row, bv in zip(A_rows, b_vec)]
    rows, cols = len(m), len(m[0]) - 1
    piv = []
    ri = 0
    for c in range(cols):
        pr = None
        for i in range(ri, rows):
            if m[i][c] != 0:
                pr = i
                break
        if pr is None:
            continue
        m[ri], m[pr] = m[pr], m[ri]
        pv = m[ri][c]
        m[ri] = [x / pv for x in m[ri]]
        for i in range(rows):
            if i != ri and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[ri])]
        piv.append(c)
        ri += 1
        if ri == rows:
            break
    for i in range(ri, rows):
        if m[i][cols] != 0:
            return None
    x = [Fr(0)] * cols
    for i, c in enumerate(piv):
        x[c] = m[i][cols]
    return x


def main(u, v):
    P = products_of(u, v)
    # (1) 復元係数
    A_rows = [[P[k][0][i] * P[k][1][j] for k in range(12)]
              for i in range(4) for j in range(4)]
    b_vec = [W[i] * W[j] for i in range(4) for j in range(4)]
    lam = solve_exact(A_rows, b_vec)
    if lam is None:
        return {"uv": [u, v], "algebra_ok": False}
    D = 1
    for x in lam:
        D = D * x.denominator // np.gcd(D, x.denominator)
    nk = [int(x * Fr(int(D))) for x in lam]

    # (2) 合法なリム組と符号化(面はe4本+和面2本 — e面は自明に合法)
    T = np.array(list(itertools.product(S, repeat=4)), dtype=np.int64)
    ok = np.ones(len(T), dtype=bool)
    for f in ((u, 0, v, 0), (0, u, 0, v)):
        val = sum(T[:, i] * f[i] for i in range(4))
        ok &= np.isin(val, np.array(S, dtype=np.int64))
    Tm = T[ok]
    n_of = sum(Tm[:, i] * W[i] for i in range(4))
    first = {}
    for idx in range(len(Tm)):
        n = int(n_of[idx])
        if n not in first:
            first[n] = idx
    xmax = 0
    while (xmax + 1) in first and -(xmax + 1) in first:
        xmax += 1
    res = {"uv": [u, v], "algebra_ok": True, "D": int(D),
           "lambda_solved": nk, "n_legal": int(len(Tm)),
           "X_recomputed": xmax, "X_claim": X_CLAIM,
           "lambda_matches_report": sorted(nk) == sorted(LAM_CLAIM)}
    if xmax < X_CLAIM:
        res["verified_full"] = False
        return res
    enc = np.array([Tm[first[n]] for n in range(-X_CLAIM, X_CLAIM + 1)],
                   dtype=np.int64)

    # (3) 全数対の恒等式
    FA = [sum(enc[:, i] * P[k][0][i] for i in range(4)) for k in range(12)]
    FB = [sum(enc[:, i] * P[k][1][i] for i in range(4)) for k in range(12)]
    n = 2 * X_CLAIM + 1
    acc = np.zeros((n, n), dtype=np.int64)
    for k in range(12):
        acc += nk[k] * np.outer(FA[k], FB[k])
    nr = np.arange(-X_CLAIM, X_CLAIM + 1, dtype=np.int64)
    ident = bool((acc == int(D) * np.outer(nr, nr)).all())
    res["n_pairs_checked"] = int(n * n)
    res["identity_ok"] = ident
    res["verified_full"] = bool(ident)
    return res


if __name__ == "__main__":
    for (u, v) in ((1, 1), (1, -1)):
        r = main(u, v)
        print(json.dumps(r, ensure_ascii=False))
