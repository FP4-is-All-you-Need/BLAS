# Solution schema for exact integer matrix products

A solution constructs a wide integer matrix product from narrow products whose
limbs belong to storage set S. It records mathematical facts: weights, planes, reconstruction
coefficients, coverage, and verification status. Hardware names, kernels, and
performance results belong to backend records. Dependencies point from those
implementations toward this catalogue.

## Layout

| File or directory | Contents |
|---|---|
| `schema.md` | Record format and interpretation rules |
| `S_2xE2M1/` | Solutions for S = 2×E2M1; other storage alphabets can use sibling directories such as `S_2xE3M2/` |
| `S_2xE2M1/diagonal.jsonl` | 1 record per line for square cells with L limbs and shared weights on both operands |
| `S_2xE2M1/rectangular.jsonl` | 1 record per line for independent operands, including unequal limb counts and equal-count designs with different weights or planes (`cell.type=independent`) |
| `S_2xE2M1/verify.py` | Independent reconstruction of encodings and exhaustive identity checks from search-format JSON lines on stdin |
| `hybrid.jsonl` | 1 mixed-format construction per line, outside any single storage-alphabet directory |

Hybrid records list their alphabets in `formats`. Relative product costs depend
on hardware, so `scope` must state the conversion rule and the conditions for an
advantage. The original verifier targets Python 3.6 and NumPy 1.14 APIs.

The [representation guide](../../Docs/representations.md) defines the storage
set, limbs, planes and coverage. Stored `storage.id` values remain unchanged.

## Record types

Each line is JSON, with `kind` identifying its interpretation:

- `solution`: a verified positive construction.
- `negative_sweep`: absence within a search. **Always read its `scope`; it does
  not exclude solutions outside that scope.** Distinguish the weight box from
  the allowed plane vocabulary: removing 1 restriction does not remove the
  other.
- `negative_theorem`: impossibility established by a proof rather than a finite
  search cutoff; the theorem's hypotheses still apply.
- `theorem`: a theorem with its statement, examples, and source.

## Main fields of a square solution

| Field | Meaning |
|---|---|
| `cell` | `{type: "square", L, M}`: limb count L and integer width M; coverage requires X >= 2^(M-1)-1 |
| `weights`, `X`, `D` | Limb weights w, symmetric coverage ±X, and common denominator of reconstruction coefficients (1 for integers) |
| `pairs` | Limb-index pairs used for pair products; for L=4 the order is `[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]` |
| `uv` | Coefficients (u,v), giving products `(u*a_i + v*a_j)(u*b_i + v*b_j)` |
| `r` | Total products; for L=4, r = 4 diagonal + k pair + 2*(6-k) direct = 16-k |
| `alternates`, `canonical_rule` | Other weights reaching the same X and the rule for selecting a representative |
| `verified` | `{method, n_pairs, date}` describing the independent checks |
| `symmetry` | Assumptions about whether the 2 operands share weights and planes |

Many square-cell sweeps, including minrank5, scale sweeps, and the L=4 ladder,
search only shared weights and planes. Their negative results apply to that
symmetric scope. Rectangular sweeps and counting closures permit independent
operands. Each record states its assumptions.

## Evidence requirements

Counts, coverage values, and numbers of construction families must be linked to
supporting artifacts by path and hash in `evidence`:

```json
{"evidence": [{"path": "source-archive/results_YYYYMMDD/xxx.json",
               "sha256": "...", "what": "The quantity supported by this artifact"}]}
```

A confirmed record links its numbers to supporting artifacts. The `what` field
identifies evidence produced by a later rerun. Paths may be relative archive
identifiers for artifacts kept outside BLAS.

## Verification convention

“Verified” means a checker written independently of the search program verifies
both the encoding (`sum w_i*a_i = N`, with every limb in S) and the reconstruction
identity (weighted sum of products = D*A*B) for every pair in the claimed ±X
range. Record the verification date and total pair count in `verified`.

## Collected descriptors

Each line carries `schema_version: "1.0.0-collect"` and the existing `kind`
discriminator from the record types above. This additive version
uses full coefficient vectors on both operands rather than requiring pair-form
`uv` fields; it does not change older records.

For a record, the reconstructed product is

$$
\sum_{p=1}^{r} c_p\,(u_p^T a)(v_p^T b)
= (w_A^T a)(w_B^T b).
$$

`faces.A[p]` and `faces.B[p]` contain $u_p$ and $v_p$;
`coefficients[p]` contains $c_p$ and `D` is 1. `face_gains` specifies integer
factors applied before storage; the effective coefficient for those stored
faces is $c_p/(g_{A,p}g_{B,p})$. `encoding_lut` fixes the limb representation of
every retained input. The explicit inclusive `input_domain` is authoritative;
width labels do not silently change it to a language's signed integer range.

`correction.rejected_values` lists the other inputs in the domain. Their faces
are zeroed. If either operand is rejected, the exact original product is added
once as a correction. This is a mathematical specification, with no runtime
adapter supplied here. `X`, where present, is the contiguous range needing no
correction and can be smaller than the corrected input domain.

Verification checks the coefficient matrix, storage membership, encoding and
the retained/corrected input partition.
Scale-format checks are separate; [collection notes](../../Docs/Provenance/solution_collection.json)
retain the unresolved scale observation.
