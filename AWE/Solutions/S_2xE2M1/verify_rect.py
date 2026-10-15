# -*- coding: utf-8 -*-
"""長方形(左右独立)構成の独立検算器 — 独立検算担当 新造(2026-08-18夜)。

探索器(sweep_shard_rect.py)とは独立に、hit record だけから全てを解き直す:
  (1) 復元係数 λ: Σ_k λ_k u_k v_k^T = w v^T の 9本の線形方程式を分数で厳密求解
      (解の存在=代数の成立。ガウス消去は自前実装で探索器のコードを使わない)
  (2) 符号化: 合法なリム組(全ての面の値が S に入る S^3 の元)を独立に列挙し、
      重み和 = n となる組を各 n に割り当て。主張された X_A / X_B の連続被覆を確認
  (3) 恒等式: ±X_A × ±X_B の全数対で Σ_k λ_k (u_k・a)(v_k・b) = n_A・n_B を
      整数(共通分母 D 倍)で照合
  (4) 面の枚数: 相異なる u_k / v_k の本数(両側4枚の主張の確認)

入力: 標準入力に hit record の JSON 行(X_A, X_B, w, v, products)。
出力: 1行1件の判定 JSON、最後に集計行。全件 verified_full なら exit 0。
Python 3.6 互換・numpy 1.14 API。
"""
from __future__ import print_function
import itertools, json, sys
from fractions import Fraction as Fr

import numpy as np

S = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12]
SSET = set(S)


def solve_exact(A_rows, b_vec):
    """A x = b を分数で解く(行簡約)。解が無ければ None。"""
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
            return None          # 矛盾 = 解なし
    x = [Fr(0)] * cols
    for i, c in enumerate(piv):
        x[c] = m[i][cols]
    return x


def legal_and_encode(faces, wt, X):
    """面集合 faces の下で合法な S^3 のリム組を列挙し、±X を符号化する。

    戻り値 (encode配列 (2X+1,3) or None, 実際の最大連続被覆X_max)。
    """
    T = np.array(list(itertools.product(S, repeat=3)), dtype=np.int64)
    ok = np.ones(len(T), dtype=bool)
    for u in faces:
        val = T[:, 0] * u[0] + T[:, 1] * u[1] + T[:, 2] * u[2]
        ok &= np.array([int(x) in SSET for x in val])
    Tm = T[ok]
    n_of = Tm[:, 0] * wt[0] + Tm[:, 1] * wt[1] + Tm[:, 2] * wt[2]
    first = {}
    for idx in range(len(Tm)):
        n = int(n_of[idx])
        if n not in first:
            first[n] = idx
    xmax = 0
    while (xmax + 1) in first and -(xmax + 1) in first:
        xmax += 1
    if xmax < X:
        return None, xmax
    enc = np.array([Tm[first[n]] for n in range(-X, X + 1)], dtype=np.int64)
    return enc, xmax


out_all = []
for line in sys.stdin.read().splitlines():
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    XA, XB = rec["X_A"], rec["X_B"]
    w, v = rec["w"], rec["v"]
    prods = rec["products"]
    uks = [p[0] for p in prods]
    vks = [p[1] for p in prods]
    res = {"w": w, "v": v, "X_A": XA, "X_B": XB, "r": len(prods)}

    # (4) 面の枚数
    fa = sorted(set(tuple(u) for u in uks))
    fb = sorted(set(tuple(x) for x in vks))
    res["n_faces_A"], res["n_faces_B"] = len(fa), len(fb)

    # (1) 復元係数を独立に解く
    A_rows = [[uks[k][i] * vks[k][j] for k in range(len(prods))]
              for i in range(3) for j in range(3)]
    b_vec = [w[i] * v[j] for i in range(3) for j in range(3)]
    lam = solve_exact(A_rows, b_vec)
    res["algebra_ok"] = lam is not None
    if lam is None:
        res["verified_full"] = False
        out_all.append(res)
        continue
    D = 1
    for x in lam:
        D = D * x.denominator // np.gcd(D, x.denominator)
    nk = [int(x * Fr(int(D))) for x in lam]
    res["D"] = int(D)

    # (2) 符号化(左右独立)
    encA, xa = legal_and_encode(fa, w, XA)
    encB, xb = legal_and_encode(fb, v, XB)
    res["X_A_recomputed"], res["X_B_recomputed"] = xa, xb
    res["cover_ok"] = encA is not None and encB is not None
    if not res["cover_ok"]:
        res["verified_full"] = False
        out_all.append(res)
        continue

    # (3) 全数対の恒等式(D 倍の整数で厳密)
    FA = [encA[:, 0] * u[0] + encA[:, 1] * u[1] + encA[:, 2] * u[2] for u in uks]
    FB = [encB[:, 0] * x[0] + encB[:, 1] * x[1] + encB[:, 2] * x[2] for x in vks]
    acc = np.zeros((2 * XA + 1, 2 * XB + 1), dtype=np.int64)
    for k in range(len(prods)):
        acc += nk[k] * np.outer(FA[k], FB[k])
    na = np.arange(-XA, XA + 1, dtype=np.int64)
    nb = np.arange(-XB, XB + 1, dtype=np.int64)
    ident = bool((acc == int(D) * np.outer(na, nb)).all())
    res["n_pairs_checked"] = int((2 * XA + 1) * (2 * XB + 1))
    res["identity_ok"] = ident
    res["verified_full"] = bool(ident and res["cover_ok"] and res["algebra_ok"])
    out_all.append(res)

n_ok = sum(1 for o in out_all if o["verified_full"])
for o in out_all:
    print(json.dumps(o, ensure_ascii=False))
print(json.dumps({"summary": True, "n": len(out_all), "n_verified": n_ok},
                 ensure_ascii=False))
sys.exit(0 if n_ok == len(out_all) else 1)
