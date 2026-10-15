# Exact integer product constructions — solution catalogue (FP4 / E2M1 carriers)

Canonical machine-readable record: `SOLUTIONS_intN.json` in this directory. This document summarizes it.

> **Correction dated 2026-08-17 — the “K limit” column in this document (116,508 in every row) is outdated.**
> The JSON adds `c_min` / `K_ng` / `K_odd`, 3 columns that **should be used instead**.
> The old column assumed a fixed stored-value cap of \|c\|=12. Under model C, which permits a decomposition per value,
> **c can be reduced while preserving coverage of the target width**, improving K by the corresponding c² factor.
> Across 178,259 rows, the median improvement is **9 times**, with a maximum of 16 times. See “K limits by storage cap” below.
> The `verified_all_pairs` column was also rebuilt: all rows had been 0.
> `analysis/verify_catalog.py` verified every pair for the maximum-coverage row in each (L, pair set) group.

## Reading the catalogue — always keep pair sets and weights together

**Do not take a pair set in isolation and combine it with other weights.** The sweep tests permutations, so the correspondence between pairs and weights is essential.
For example, the 7-product solution `(1,18,59)` with pairs `{(0,1),(1,2)}` reaches X=204; using `{(0,1),(0,2)}` instead reduces it to **X=8** (audit record #7).

- **All pair sums** (r = L(L+1)/2) use all 3 pairs and are **permutation symmetric**. Weights are canonicalized in ascending order.
- **Partial pair sums** (r > L(L+1)/2) are asymmetric. Use the pair set exactly as shown.

## Scope of this catalogue

| L | Pair count | r | Legal limb tuples | Counting upper bound X | Maximum X | Solution count |
|---|---|---|---|---|---|---|
| 2 | 0 | 4 | 225 | 112 | **56** | 11 |
| 2 | 1 | 3 | 133 | 66 | **24** | 8 |
| 3 | 0 | 9 | 3,375 | 1,687 | **732** | 32,865 |
| 3 | 1 | 8 | 1,995 | 997 | **360** | 75,126 |
| 3 | 2 | 7 | 1,275 | 637 | **214** | 57,741 |
| 3 | 3 | 6 | 875 | 437 | **148** | 12,508 |

For L=4,5, exhaustive free-weight enumeration is impractical; only counting bounds are given:

| L | Pairs | r | Legal limb tuples | Counting upper bound X | Maximum covered N |
|---|---|---|---|---|---|
| 4 | 0 | 16 | 50,625 | 25,312 | 15 |
| 4 | 6 | 10 | 5,173 | 2,586 | 12 |
| 5 | 0 | 25 | 759,375 | 379,687 | 19 |
| 5 | 10 | 15 | 28,971 | 14,485 | 14 |

For L=4 with all pair sums (10 products), a bounded sweep (m1≤16, m2≤130, m3≤1100) reached **maximum X=624**,
which **does not cover fp16 (1023)**. The counting upper bound, 2,586, does not exclude the target;
**unexplored possibilities remain**, rather than a proof of impossibility.

## K limits by storage cap (added 2026-08-17)

The K limit is determined by the maximum stored absolute value c: **K ≤ 2^24/(n_g·c²)** (F=24 is the exact-integer width of FP32).
Model C permits a decomposition per value. Apply **the same cap c to both limbs and planes**, then require only that each value have a representation under that cap.
Reducing c while preserving the target coverage increases K.

| c | K limit(n_g=1) | Maximum X with 6 products (all pair sums) | 9 products, radix 13, direct |
|---|---|---|---|
| ≤3 | 1,864,135 | **Unreachable**(fewer than 255 legal tuples; counting bound) | X=4 |
| ≤4 | **1,048,576** | **130**(weights (8,17,48), planes generalized to u·a_i+v·a_j) | X=4 |
| ≤6 | 466,033 | 148(`(1,29,37)`) | X=4 |
| ≤8 | 262,144 | 148 | 732 |
| ≤12 | 116,508 | At least 162 | 732 |

Conclusions for INT8 (requiring \|x\|≤127):

- **The ceiling on K for 6 products is 1,048,576** (c≤4). The case c≤3 is mathematically excluded.
- Radix 13 with 9 direct products needs c=8: gaps at ±5,±7 prevent coverage with ±6.
  With n_g=3, this allows only **87,381**. The 6-product construction improves it by **12 times**.
- At c≤4, the 6-product construction stores only the **contiguous subset of E2M1**, avoiding its gaps.
  See correction #2 in `SIX_PRODUCT.md` and `analysis/verify_uv6.py` for details and verification over all pairs.

For each row, `c_min` is the smallest c needed to cover `max_N`. `K_ng` uses the conventional formula
2^24/(n_g·c_min²); `K_odd` uses the refined formula that matched 5 measured points, 2^24/(odd(n_g)·c_min²).
At the time of this record, hardware K sweeps were still needed to distinguish them, so **both** are retained.

## Best construction for each N (minimum r, then maximum K limit)

| N | Required \|x\| | r | L | Pair set | Weights | X | Planes per operand | K limit | Verification |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 7 | **3** | 2 | {(0,1)} | `(1, 6)` | 24 | 3 | 116,508 | Sweep |
| 5 | 15 | **3** | 2 | {(0,1)} | `(1, 6)` | 24 | 3 | 116,508 | Sweep |
| 6 | 31 | **4** | 2 | None (direct) | `(1, 13)` | 56 | 2 | 116,508 | Sweep |
| 7 | 63 | **6** | 3 | {(0,1),(0,2),(1,2)} | `(1, 29, 37)` | 148 | 6 | 116,508 | **All 65,025 pairs** |
| 8 | 127 | **6** | 3 | {(0,1),(0,2),(1,2)} | `(1, 29, 37)` | 148 | 6 | 116,508 | **All 65,025 pairs** |
| 9 | 255 | **8** | 3 | {(0,1)} | `(1, 9, 83)` | 360 | 4 | 116,508 | Sweep |
| 10 | 511 | **9** | 3 | None (direct) | `(1, 13, 169)` | 732 | 3 | 116,508 | Sweep |

N=11 and above cannot be covered with 3 limbs (maximum coverage with 3 planes: 732). The direct construction with L=4 and 16 products was the best known at the time of this record. Reductions for L=4 were unexplored.

## 6-product constructions: 2 solutions verified over all pairs

| | Pair-sum form | Karatsuba form |
|---|---|---|
| Weights | `(1, 29, 37)` | `(1, 6, 33)` |
| Pair set | {(0,1), (0,2), (1,2)} | {(0,1), **(1,2)**} + plane a0+a1+a2 |
| Coverage X | 148 | 132 |
| Coefficient sum Σλ | 1072 | — |
| Verification | All 65,025 pairs | All 65,025 pairs |

The alternative `(1, 11, 39)` (X=134, coefficient sum 1164) is easier to encode (greedy success: 64.3% versus 28.6%).
The fraction of coefficients supported by scales on both operands is 3/6 (main solution: 1/6), but coverage headroom is smaller (7 versus 21).

## Solutions by layer (including layers above the minimum r)

In practice, the K limit, scale compatibility, plane count, and encoding cost can favor **a layer above the minimum r**.
The JSON contains all layers; this document shows excerpts from each (highest X / E4M3-exact weights / powers of 2 as weights).
**See the JSON for the complete catalogue.**

### L=2, r=4(0 pairs, none)  — **All 11 choices**

| Category | Weights | X | Maximum N | n_g | K limit | Coefficient sum | Scale-compatible λ |
|---|---|---|---|---|---|---|---|
| Highest X | `(1, 13)` | 56 | 6 | 2 | 58,254 | 196 | 4/4 |
| Highest X | `(1, 11)` | 48 | 6 | 2 | 58,254 | 144 | 4/4 |
| Highest X | `(1, 9)` | 40 | 6 | 2 | 58,254 | 100 | 4/4 |
| Highest X | `(1, 8)` | 36 | 6 | 2 | 58,254 | 81 | 4/4 |
| Highest X | `(1, 4)` | 36 | 6 | 2 | 58,254 | 25 | 4/4 |
| Powers of 2 as weights | `(1, 2)` | 28 | 5 | 2 | 58,254 | 9 | 4/4 |

E4M3-exact weights 11 choices / Powers of 2 as weights 4 choices。

### L=2, r=3(1 pair [[0, 1]])  — **All 8 choices**

| Category | Weights | X | Maximum N | n_g | K limit | Coefficient sum | Scale-compatible λ |
|---|---|---|---|---|---|---|---|
| Highest X | `(1, 6)` | 24 | 5 | 1 | 116,508 | 31 | 3/3 |
| Highest X | `(1, 7)` | 22 | 5 | 1 | 116,508 | 43 | 3/3 |
| Highest X | `(1, 5)` | 20 | 5 | 1 | 116,508 | 21 | 3/3 |
| Highest X | `(1, 8)` | 18 | 5 | 1 | 116,508 | 57 | 3/3 |
| Highest X | `(1, 4)` | 16 | 5 | 1 | 116,508 | 13 | 3/3 |
| Powers of 2 as weights | `(1, 2)` | 10 | 4 | 2 | 58,254 | 3 | 3/3 |

E4M3-exact weights 8 choices / Powers of 2 as weights 3 choices。

### L=3, r=9(0 pairs, none)  — **All 32,865 choices**

| Category | Weights | X | Maximum N | n_g | K limit | Coefficient sum | Scale-compatible λ |
|---|---|---|---|---|---|---|---|
| Highest X | `(1, 13, 169)` | 732 | 10 | 3 | 38,836 | 33489 | 6/6 |
| Highest X | `(1, 26, 169)` | 628 | 10 | 2 | 58,254 | 38416 | 6/6 |
| Highest X | `(1, 13, 143)` | 628 | 10 | 2 | 58,254 | 24649 | 6/6 |
| Highest X | `(1, 11, 145)` | 628 | 10 | 2 | 58,254 | 24649 | 4/6 |
| Highest X | `(1, 11, 143)` | 620 | 10 | 2 | 58,254 | 24025 | 6/6 |
| E4M3-exact weights | `(1, 112, 13)` | 504 | 9 | 2 | 58,254 | 15876 | 9/6 |
| E4M3-exact weights | `(1, 56, 13)` | 504 | 9 | 2 | 58,254 | 4900 | 9/6 |
| E4M3-exact weights | `(1, 13, 112)` | 504 | 9 | 2 | 58,254 | 15876 | 9/6 |
| Powers of 2 as weights | `(1, 64, 8)` | 292 | 9 | 3 | 38,836 | 5329 | 9/6 |
| Powers of 2 as weights | `(1, 64, 4)` | 292 | 9 | 2 | 58,254 | 4761 | 9/6 |
| Powers of 2 as weights | `(1, 32, 8)` | 292 | 9 | 2 | 58,254 | 1681 | 9/6 |

E4M3-exact weights 1,549 choices / Powers of 2 as weights 52 choices。

### L=3, r=8(1 pair [[0, 1]])  — **All 75,126 choices**

| Category | Weights | X | Maximum N | n_g | K limit | Coefficient sum | Scale-compatible λ |
|---|---|---|---|---|---|---|---|
| Highest X | `(1, 9, 83)` | 360 | 9 | 2 | 58,254 | 8622 | 3/6 |
| Highest X | `(1, 83, 9)` | 360 | 9 | 2 | 58,254 | 8622 | 3/6 |
| Highest X | `(1, 9, 85)` | 352 | 9 | 2 | 58,254 | 8998 | 3/6 |
| Highest X | `(1, 85, 9)` | 352 | 9 | 2 | 58,254 | 8998 | 3/6 |
| Highest X | `(1, 8, 83)` | 350 | 9 | 2 | 58,254 | 8440 | 3/6 |
| E4M3-exact weights | `(1, 80, 11)` | 324 | 9 | 2 | 58,254 | 5824 | 6/6 |
| E4M3-exact weights | `(1, 11, 80)` | 324 | 9 | 2 | 58,254 | 5824 | 6/6 |
| E4M3-exact weights | `(1, 72, 13)` | 292 | 9 | 2 | 58,254 | 4588 | 6/6 |
| Powers of 2 as weights | `(1, 8, 64)` | 274 | 9 | 2 | 58,254 | 5305 | 8/6 |
| Powers of 2 as weights | `(1, 8, 32)` | 274 | 9 | 2 | 58,254 | 1657 | 8/6 |
| Powers of 2 as weights | `(1, 64, 8)` | 274 | 9 | 2 | 58,254 | 5305 | 8/6 |

E4M3-exact weights 4,217 choices / Powers of 2 as weights 147 choices。

### L=3, r=7(2 pairs [[0, 1], [0, 2]])  — **All 57,741 choices**

| Category | Weights | X | Maximum N | n_g | K limit | Coefficient sum | Scale-compatible λ |
|---|---|---|---|---|---|---|---|
| Highest X | `(1, 70, 43)` | 214 | 8 | 2 | 58,254 | 3756 | 1/6 |
| Highest X | `(1, 43, 70)` | 214 | 8 | 2 | 58,254 | 3756 | 1/6 |
| Highest X | `(1, 60, 43)` | 208 | 8 | 2 | 58,254 | 2896 | 2/6 |
| Highest X | `(1, 43, 60)` | 208 | 8 | 2 | 58,254 | 2896 | 2/6 |
| Highest X | `(1, 51, 9)` | 204 | 8 | 2 | 58,254 | 2191 | 3/6 |
| E4M3-exact weights | `(1, 72, 11)` | 173 | 8 | 2 | 58,254 | 4464 | 5/6 |
| E4M3-exact weights | `(1, 11, 72)` | 173 | 8 | 2 | 58,254 | 4464 | 5/6 |
| E4M3-exact weights | `(1, 48, 6)` | 168 | 8 | 2 | 58,254 | 2863 | 5/6 |
| Powers of 2 as weights | `(1, 32, 4)` | 144 | 8 | 2 | 58,254 | 1261 | 6/6 |
| Powers of 2 as weights | `(1, 4, 32)` | 144 | 8 | 2 | 58,254 | 1261 | 6/6 |
| Powers of 2 as weights | `(1, 4, 32)` | 138 | 8 | 2 | 58,254 | 973 | 6/6 |

E4M3-exact weights 3,829 choices / Powers of 2 as weights 138 choices。

### L=3, r=6(3 pairs [[0, 1], [0, 2], [1, 2]])  — **All 12,508 choices**

| Category | Weights | X | Maximum N | n_g | K limit | Coefficient sum | Scale-compatible λ |
|---|---|---|---|---|---|---|---|
| Highest X | `(1, 29, 37)` | 148 | 8 | 1 | 116,508 | 1072 | 1/6 |
| Highest X | `(1, 11, 39)` | 134 | 8 | 1 | 116,508 | 1164 | 3/6 |
| Highest X | `(1, 7, 37)` | 122 | 7 | 1 | 116,508 | 1116 | 1/6 |
| Highest X | `(1, 25, 35)` | 122 | 7 | 1 | 116,508 | 916 | 2/6 |
| Highest X | `(1, 6, 32)` | 122 | 7 | 1 | 116,508 | 831 | 5/6 |
| E4M3-exact weights | `(1, 7, 30)` | 120 | 7 | 1 | 116,508 | 703 | 6/6 |
| E4M3-exact weights | `(1, 6, 36)` | 118 | 7 | 1 | 116,508 | 1075 | 3/6 |
| Powers of 2 as weights | `(1, 4, 32)` | 116 | 7 | 1 | 116,508 | 877 | 5/6 |
| Powers of 2 as weights | `(1, 8, 32)` | 112 | 7 | 1 | 116,508 | 793 | 5/6 |
| Powers of 2 as weights | `(1, 2, 16)` | 72 | 7 | 1 | 116,508 | 211 | 5/6 |

E4M3-exact weights 688 choices / Powers of 2 as weights 28 choices。

## L=2 layer — records for scheme II (Ozaki II / CRT)

**The optimal weights differ between scheme I (integer targets) and scheme II (modular residues).** Scheme I requires the whole interval [-X,X], so it measures contiguous coverage.
Scheme II only needs every residue modulo m, not a contiguous interval, so it measures the **number of distinct values**.

| Weights w | Contiguous coverage X | Distinct value count | Use |
|---|---|---|---|
| 6 | **24** | 87 | Scheme I (maximum contiguous coverage) |
| 7 | 22 | 95 | |
| 8 | 18 | 107 | |
| 9 | 12 | 113 | |
| 12 | 4 | **125** | Scheme II (maximum distinct value count) |

### Maximum modulus

- There are **133** legal limb tuples for Karatsuba with 3 products, or **225** for the direct construction with 4 products. This is an **upper bound** on the number of covered residues, independent of weights.
- The **largest modulus admitting a complete residue system is 91** (3 products, weights such as 17/50/71/75).
  With 4 direct products it is 165. **133 is an upper bound, not an attained value.**
- With 3 products, **74 moduli** admit complete residue systems: almost all of 2..53, plus 55,56,57,58,59,61,63,64,65,
  67,69,71,73,75,77,79,81,83,85,87,89,**91**.
- Compared with the old KAR_MODULI (fixed weight 13) = {53,59,61,67,71,73,83}, **free weights add 67 moduli**.
  Beyond the old maximum 83, the new moduli 85,87,89,91 can now use 3 products.

### Implications for scheme II with 19 moduli

Among the moduli supporting 3 products, a pairwise coprime set of 19 can be chosen: `{23,31,37,41,43,47,53,59,61,64,67,71,73,79,83,85,87,89,91}`.
Their product is **2^111.5**, covering the DD (106bit) range. **If all 19 moduli support 3 products, 19×4=76 → 19×3=57 products, a reduction of 25.0%.**
The other scheme II requirements (residue ranges, accumulators, and CRT reconstruction implementation) are outside this search and need separate checks before adoption.

## Search cutoffs

- Weight sweep: for L=2, all (1,m) with m<=4000; for L=3, all (1,m1,m2) with m1<=120, m2<=1000. Component 1 is fixed to 1.
- For L>=4, exhaustive free-weight enumeration is impractical. Only counting bounds were used to record possible feasibility; solutions were not exhaustively searched.
- L=4 all-pair-sum sweep: m1≤16, m2≤130, m3≤1100 (`analysis/l4_pairsum.py`).
- **Scope of “6 is minimal” (updated 2026-08-17):** proof of r≥4, structural arguments for within-row/within-column merging, and **asymmetric u≠v searches at r=4,5 with 0 solutions** (rank-drop conditions on minors, candidate planes ‖u‖₁≤2, and 3 limbs per operand). **NVFP4 block scales give the same result**: per-position scales are absorbed into weights, while sum-plane scales only reduce legal tuples.
  Unexplored: asymmetry with ‖u‖₁≥3 (3-term sums), and different limb counts on the operands.
- **Generalized (u,v) planes (added 2026-08-17):** a pair plane can be **u·a_i + v·a_j** instead of a_i+a_j; 1/(uv) is a rational epilogue coefficient and preserves (P1). This table fixes (u,v)=(1,1). Generalization increases **L=2 with 3 products from X=24→30, and L=3 with 6 products from X=148→162 or more**. **The claim of only (1,29,37) and (1,11,39), just 2 solutions, is withdrawn.** The thresholds (N=6 requires 31; N=9 requires 255) are not crossed, so the minimum r for each N is unchanged. Enumeration of the generalized layers was ongoing at the time of this record.
- For the L=2 layer, X(w) covers w=1..4000; modulus checks exhaust m<=133 (3 products) / m<=225 (4 products).
- Scheme II moduli were selected greedily for pairwise coprimality from those supporting 3 products. Scheme II range, accumulator, and CRT reconstruction requirements were not checked.
