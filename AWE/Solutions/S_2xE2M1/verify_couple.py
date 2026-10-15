# -*- coding: utf-8 -*-
"""成分≤2・r=5の結合段の独立再実装 — 独立検算担当 2026-08-19。

構成担当の結合段(243,000件)を、構成担当のコードを使わずに一から再現する。
入力も構成担当の成果物でなく、独立検算担当がarchive-search-a+archive-search-bで回収した片側の足切りの
成果(campaign/results/c2_all.jsonl — 45合格類・達成重み1,012本)を使う。

問い: 合格類の対(A,B)・面の1対1対応σ・達成重みの対(w_A, w_B)で、
      Σ_k λ_k u_k v_{σ(k)}ᵀ = w_A w_Bᵀ を満たす復元係数λ(全て非零)が存在するか。

方法:
  各(A,B,σ)で M = [vec(u_k v_{σ(k)}ᵀ)] (9×5) の直交補空間 Q を厳密整数で計算し、
  w_A w_Bᵀ が span(M) に入る ⟺ Q·vec(w_A w_Bᵀ) = 0 を、Q_q を3×3に畳んで
  R_q = W_A Q_q W_Bᵀ の同時零として全重み対に一括判定(numpy int64)。
  完全性: 片側の足切りは箱なしの完全リストなので、r=5解の重みは必ず達成リストに入る。
  当たり候補が出たらλを分数で厳密求解し、全λ非零(積5本)まで確かめる。
"""
from __future__ import print_function
import itertools, json, sys, time
from fractions import Fraction as Fr

import numpy as np

BASE = 'source-archive/campaign/results/'


def int_left_nullspace(M):
    """整数行列 M (9x5) の直交補空間(kernel of M^T)の整数基底。分数ガウス。"""
    rows = [[Fr(M[i][k]) for k in range(5)] + [Fr(1 if j == i else 0) for j in range(9)]
            for i in range(9)]
    # M の列空間に対する消去: 先頭5列を簡約し、ゼロ行の右側9列が補空間
    piv = []
    ri = 0
    for c in range(5):
        pr = None
        for i in range(ri, 9):
            if rows[i][c] != 0:
                pr = i
                break
        if pr is None:
            continue
        rows[ri], rows[pr] = rows[pr], rows[ri]
        pv = rows[ri][c]
        rows[ri] = [x / pv for x in rows[ri]]
        for i in range(9):
            if i != ri and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[ri])]
        piv.append(c)
        ri += 1
    out = []
    for i in range(ri, 9):
        vec = rows[i][5:]
        den = 1
        for x in vec:
            den = den * x.denominator // np.gcd(den, x.denominator)
        iv = [int(x * den) for x in vec]
        g = 0
        for x in iv:
            g = int(np.gcd(g, abs(x)))
        if g > 1:
            iv = [x // g for x in iv]
        if any(iv):
            out.append(iv)
    return out


def main():
    t0 = time.time()
    classes = []
    for line in open(BASE + 'c2_all.jsonl'):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get('n_ach') or 0) > 0:
            classes.append({"rank": r['rank'],
                            "faces": [tuple(f) for f in r['faces']],
                            "W": np.array(r['achieving'], dtype=np.int64)})
    print("合格類:", len(classes), "/ 達成重みの総本数:",
          sum(len(c['W']) for c in classes), flush=True)
    perms = list(itertools.permutations(range(5)))
    n_cases = 0
    n_pair_checks = 0
    hits = []
    for A in classes:
        UA = A['faces']
        WA = A['W']
        for B in classes:
            VB = B['faces']
            WB = B['W']
            for sg in perms:
                n_cases += 1
                M = [[UA[k][i] * VB[sg[k]][j] for k in range(5)]
                     for i in range(3) for j in range(3)]
                Q = int_left_nullspace(M)
                if not Q:          # spanが9次元 — あり得ないが安全側
                    continue
                ok = np.ones((len(WA), len(WB)), dtype=bool)
                for q in Q:
                    Qm = np.array(q, dtype=np.int64).reshape(3, 3)
                    R = WA @ Qm @ WB.T
                    ok &= (R == 0)
                    if not ok.any():
                        break
                n_pair_checks += len(WA) * len(WB)
                if ok.any():
                    for ia, ib in zip(*np.nonzero(ok)):
                        # λの厳密求解と全λ非零の確認
                        wA = [int(x) for x in WA[ia]]
                        wB = [int(x) for x in WB[ib]]
                        rowsM = [[Fr(M[r_][k]) for k in range(5)] + [Fr(wA[r_ // 3] * wB[r_ % 3])]
                                 for r_ in range(9)]
                        piv = []
                        ri = 0
                        for c in range(5):
                            pr = None
                            for i in range(ri, 9):
                                if rowsM[i][c] != 0:
                                    pr = i
                                    break
                            if pr is None:
                                continue
                            rowsM[ri], rowsM[pr] = rowsM[pr], rowsM[ri]
                            pv = rowsM[ri][c]
                            rowsM[ri] = [x / pv for x in rowsM[ri]]
                            for i in range(9):
                                if i != ri and rowsM[i][c] != 0:
                                    f = rowsM[i][c]
                                    rowsM[i] = [x - f * y for x, y in zip(rowsM[i], rowsM[ri])]
                            piv.append(c)
                            ri += 1
                        bad = any(rowsM[i][5] != 0 for i in range(ri, 9))
                        lam = [Fr(0)] * 5
                        for i, c in enumerate(piv):
                            lam[c] = rowsM[i][5]
                        if (not bad) and all(x != 0 for x in lam):
                            hits.append({"rankA": A['rank'], "rankB": B['rank'],
                                         "sigma": list(sg), "w_A": wA, "w_B": wB,
                                         "lambda": [str(x) for x in lam]})
        if len(hits) == 0 and A['rank'] % 10 == 0:
            print("  類A rank%d まで: %d 件処理 (%.0f秒)" % (A['rank'], n_cases, time.time() - t0),
                  flush=True)
    out = {"question": "成分<=2・r=5の結合段(合格類の対×面の1対1対応×達成重みの対)に解があるか",
           "inputs": "独立検算担当がarchive-search-a+archive-search-bで回収した片側の足切りの成果(c2_all.jsonl)。構成担当の成果物・コードは不使用",
           "n_classes": len(classes), "n_cases": n_cases,
           "n_weight_pair_checks": n_pair_checks,
           "n_hits": len(hits), "hits": hits,
           "scope": "面の語彙=成分<=2の原始49方向・定義B・整数重み・被覆127。片側の足切りは箱なしの完全リストなので、この243,000件で結合段は完全"}
    with open(BASE + 'couple_indep.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    print("== まとめ ==")
    print("件数 %d / 重み対の判定 %s / 当たり %d (%.0f秒)"
          % (n_cases, format(n_pair_checks, ','), len(hits), time.time() - t0))


if __name__ == "__main__":
    main()
