# -*- coding: utf-8 -*-
"""M=15の22積(対角の持ち上げ)の独立全対検算 — 独立検算担当 2026-08-19。

record関門: 構成担当の構成を、構成担当のコード・λを使わずに一から検算する。
  (1) 積22本を、カタログのM=12 record(pairs/uv/w)と持ち上げ則(u=6021)から自前で組む
  (2) 復元係数を25本の線形方程式として分数で厳密求解(D と |λ|max を報告と照合)
  (3) 合法なリム組を独立に列挙し、±26,422 の連続被覆を再計算
  (4) ±26,422 × ±26,422 の全数対(約27.9億対)をブロック分割で厳密照合
"""
from __future__ import print_function
import itertools, json, sys, time
from fractions import Fraction as Fr

import numpy as np

S = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12]
SSET = set(S)
W4 = [2, 10, 43, 551]
U = 6021
W5 = W4 + [U]
X = 26422
PAIRS = [(0, 1), (0, 3), (1, 3)]          # M=12 record
UVS = [(1, -1), (1, 1), (1, 1)]


def e(i):
    v = [0] * 5
    v[i] = 1
    return tuple(v)


def build_products():
    P = []
    for (i, j), (u, v) in zip(PAIRS, UVS):   # 対3本
        f = [0] * 5
        f[i], f[j] = u, v
        P.append((tuple(f), tuple(f)))
    for i in range(4):                        # 対角4本
        P.append((e(i), e(i)))
    used = set(PAIRS)
    for i in range(4):                        # 対に入らない直接 2*(6-3)=6本
        for j in range(i + 1, 4):
            if (i, j) in used:
                continue
            P.append((e(i), e(j)))
            P.append((e(j), e(i)))
    assert len(P) == 13
    for j in range(4):                        # 持ち上げ +4本
        P.append((e(j), e(4)))
    for j in range(5):                        # 持ち上げ +5本
        P.append((e(4), e(j)))
    assert len(P) == 22
    return P


def solve_exact(A_rows, b_vec):
    m = [[Fr(x) for x in row] + [Fr(bv)] for row, bv in zip(A_rows, b_vec)]
    rows, cols = len(m), len(m[0]) - 1
    piv, ri = [], 0
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


def main():
    t0 = time.time()
    P = build_products()
    A_rows = [[P[k][0][i] * P[k][1][j] for k in range(22)]
              for i in range(5) for j in range(5)]
    b_vec = [W5[i] * W5[j] for i in range(5) for j in range(5)]
    lam = solve_exact(A_rows, b_vec)
    if lam is None:
        print(json.dumps({"algebra_ok": False}))
        return 1
    D = 1
    for x in lam:
        D = D * x.denominator // np.gcd(D, x.denominator)
    nk = [int(x * Fr(int(D))) for x in lam]
    print(json.dumps({"algebra_ok": True, "D": int(D),
                      "lam_max": max(abs(x) for x in nk),
                      "lam_max_claim": 36252441}), flush=True)

    # 下位4リムの合法な組(対3本の面のみが制約)
    T4 = np.array(list(itertools.product(S, repeat=4)), dtype=np.int64)
    ok = np.ones(len(T4), dtype=bool)
    for (i, j), (u, v) in zip(PAIRS, UVS):
        val = u * T4[:, i] + v * T4[:, j]
        ok &= np.isin(val, np.array(S, dtype=np.int64))
    T4m = T4[ok]
    vals4 = sum(T4m[:, i] * W4[i] for i in range(4))
    rep4 = {}
    for idx in range(len(T4m)):
        v = int(vals4[idx])
        if v not in rep4:
            rep4[v] = idx
    # 5リムの値集合と被覆
    vset = set()
    for d in S:
        for v in rep4:
            vset.add(v + U * d)
    x5 = 0
    while (x5 + 1) in vset and -(x5 + 1) in vset:
        x5 += 1
    print(json.dumps({"n_legal4": int(len(T4m)), "X_recomputed": x5,
                      "X_claim": X}), flush=True)
    if x5 < X:
        return 1

    # 符号化表(±X): 値 -> 5リム
    enc = np.zeros((2 * X + 1, 5), dtype=np.int64)
    got = np.zeros(2 * X + 1, dtype=bool)
    for d in S:
        base = U * d
        for v, idx in rep4.items():
            N = v + base
            if -X <= N <= X and not got[N + X]:
                enc[N + X, :4] = T4m[idx]
                enc[N + X, 4] = d
                got[N + X] = True
    assert bool(got.all()), "符号化に欠け"

    # 全数対をブロックで照合
    FA = [sum(enc[:, i] * P[k][0][i] for i in range(5)) for k in range(22)]
    FB = [sum(enc[:, i] * P[k][1][i] for i in range(5)) for k in range(22)]
    n = 2 * X + 1
    nr = np.arange(-X, X + 1, dtype=np.int64)
    B = 512
    for s in range(0, n, B):
        t = min(s + B, n)
        acc = np.zeros((t - s, n), dtype=np.int64)
        for k in range(22):
            acc += nk[k] * np.outer(FA[k][s:t], FB[k])
        if not (acc == int(D) * np.outer(nr[s:t], nr)).all():
            print(json.dumps({"identity_ok": False, "block": s}), flush=True)
            return 1
        if (s // B) % 20 == 0:
            print("  block %d/%d (%.0fs)" % (s, n, time.time() - t0), flush=True)
    print(json.dumps({"identity_ok": True, "n_pairs_checked": n * n,
                      "verified_full": True,
                      "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
