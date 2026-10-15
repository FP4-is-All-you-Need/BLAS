# Solution catalogue

AWE is the canonical source of the mathematical data and verification programs.
Updates flow from AWE to this copy. Annotations in the data may include Japanese.
[The schema](schema.md) defines fields and verification rules.

| Records | Contents |
|---|---|
| [SOLUTIONS_intN.json](SOLUTIONS_intN.json) | Integer-width constructions; use `c_min`, `K_ng` and `K_odd`, not the older fixed-cap `K_max` |
| [plane.json](plane.json) | Width pairs and hardware-dependent cost estimates |
| `gen_uv.json`, `gen_uv_ext_done.jsonl`, `cap_uv6.json`, `cap_wide.json`, `e4m3_weights.json` | Generalized pair planes, storage caps and scale compatibility |
| `rect23_r4.json`, `rect_cap.json`, `rect_ext.json` | Independent operand constructions |
| `verify_six.json`, `verify_uv6.json`, `verify_rect.json`, `minrank5.jsonl`, `minrank5.json`, `l4_pairsum.json` | Verification and bounded search records |
| [S_2xE2M1/](S_2xE2M1/), [hybrid.jsonl](hybrid.jsonl) | Storage-specific and mixed-format descriptors |
| [Collected_20260920/](S_2xE2M1/Collected_20260920/) | Collected constructions and line-S observations |

[Verification commands](../../Docs/validation.md#solution-checks) operate on the
included data. `source-archive/`, `archive-search-*`, `archive-sweep-a`,
`archive-analysis/`, `archive-rect/`, `archive-wide/` and `archive-evidence/`
are source identifiers, not local or network paths. Referenced artifacts are not
necessarily distributed here. `verify_couple.py` needs a locally supplied
`source-archive/` input tree.

The historical [PLANE](PLANE.md), [SIX_PRODUCT](SIX_PRODUCT.md) and
[SOLUTIONS_intN](SOLUTIONS_intN.md) reports remain because their complete tables
and corrections have not been matched to the JSON records. They retain their
original scope and are not current API documentation. In particular, the old
catalogue summary reports 178,259 solutions while the retained JSON has 178,260
rows. Neither count has been changed to reconcile the difference.
