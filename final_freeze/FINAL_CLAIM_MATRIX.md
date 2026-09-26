# Final claim matrix — frozen

11 claims, **8 load-bearing**. Machine-readable: `FINAL_CLAIM_MATRIX.csv`.
Numerical source of truth: `final_evidence/FINAL_PAPER_EVIDENCE.csv` (75 values).
All S-1 … S-19 binding.

Load-bearing count rises from 6 to 8 for two deliberate reasons: **C1-1** (the
framework itself) is now a claim rather than an assumption, and **C4-2** (the
score-space destination comparison) is promoted from SUPPORTING to load-bearing
main-text evidence.

| id | claim | LB | section |
|---|---|:-:|---|
| C1-1 | SAKIKO is a correction + verification + licensing pipeline | **Y** | 1–3 |
| C2-1 | a meaningful subset of tool-decision errors is improvable | **Y** | 4 |
| C2-2 | breadth to a second benchmark and action space | N | 7 |
| C3-1 | REAL beats every structural control | **Y** | 4, 8 |
| C3-2 | 0/59 matched randoms exceed real, in two models | **Y** | 4 |
| C4-1 | aggregate gain conceals preservation failure | **Y** | 5 |
| **C4-2** | **similar aggregate gain conceals different destination** | **Y** | **5** |
| C4-3 | gating controls exposure volume, not sample safety | N | 5, 8 |
| C5-1 | protocol-conditional and graded — 1 ADMIT + 2 DECLINE + Phi | **Y** | 6, 7 |
| C5-3 | the ladder also rejects (Llama, Mistral) | N | 7 |
| C5-2 | evidence requirement scales steeply with effect size | **Y** | 6 |
| C6-1 | correction quality is a design property | N | 8 |

## Verdicts carried from the regression audit

**PASS:** C2-1, C3-1, C3-2, C5-2, C4-3.
**NARROWED:** C4-1 (drop "≥1.6×"; add execution-variability range), C5-1
(model-at-protocol-dose; conservative), C2-2 (two splits, 524/1040 shared),
C6-1 (gating `SUPPORTED`, not `STRONGLY_SUPPORTED`).
**PROMOTED:** C4-2.
**REMOVED:** none.

## Global prohibitions

- "SAKIKO reliably fixes tool errors across models"
- "random directions produced no gold arrivals" — false in other settings
- 52/264 as *the* collateral rate; the estimand is 52/93
- "32 of 87" — it is 32 of 72 routed channel errors
- calling Qwen3-4B or Gemma **ADMITTED** — both are `FORMAL_DECLINE` (S-20)
- calling either **intrinsically uncorrectable** — `DECLINE != uncorrectable`
- "Level-1"/"Level-2" as licence names — not prospectively frozen (S-21)
- a nine-condition E-2 ladder — E-2 is **five**
- activation statistically superior to score-space
- gating universally necessary
- intrinsic model-level correctability
- channel orthogonality
- any prevalence claim about the field

## Provenance tiers (enforced)

Every number carries a `provenance_tier` in `final_evidence/FINAL_PAPER_EVIDENCE.csv`:

| tier | meaning | count |
|---|---|---:|
| **A** | row-level records, recomputed from per-sample data | 49 |
| **B** | committed structured artifact; no per-sample records survive | 35 |
| **C** | secondary audit document only | 9 |

**Rule: no load-bearing claim may rest on Tier C.** Verified mechanically by
`final_evidence/tier_check.py`. C5-1 was narrowed to satisfy it; Llama and
Mistral now carry Tier C evidence under the SUPPORTING claim C5-3.

**C2-1 and C3-1 have a Tier B floor** — MetaTool has no per-sample records, and
four of the five Phi control arms are aggregate-only. Both are disclosed
(limitations 5 and 6) and neither can be raised without data that does not exist.
