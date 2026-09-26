# Historical SAKIKO licence stress test

**`RETROSPECTIVE_HISTORICAL_RE-ADJUDICATION`. CPU-only. No model run, no GPU, no artifact modified.**
Not preregistered, not prospective, not confirmatory, not sealed, not a replication.

## Principal verdict

`HISTORICAL_LICENSE_STRESS_TEST_STRENGTHENS_PAPER`

## The result in one line

SAKIKO's own headline historical result — **Phi-3.5, Net +55, superior to all four placebos**
— **would be refused a modern correction licence**, on a collateral figure the 2026-03
analysis itself recorded and published: **52 / 264 = 0.1970**, against a 0.05 bound.

## Why it is robust

| | |
|---|---|
| exactness | the per-sample file reproduces every committed aggregate — Fixed 107, Broke 52, Net +55, clean_damage 52/264, routed 137/83/73 — **all ✓** |
| denominator-proof | fails on the frozen formal denominator (0.1970, 3.9×) **and** on eligible-at-risk (52/93 = 0.5591, 11.2×) |
| already recorded | `p0_final_test_eval.json` contains `clean_damage: {n: 264, damaged: 52, rate: 19.7}` — **the data did not change; the licence did** |
| adverse to interest | it refuses the project's own prior conclusion |
| gating verified | 0 of 255 unrouted rows changed prediction |

## What Phi passes

Support, historical direction-specificity (real +55 vs best placebo +14), zero control, and —
importantly — **destination correctness**: target-hit 0.6897–0.7353 and Target Gain +0.1864 to
+0.2857, *comparable to the Qwen3-8B ADMIT's 0.7308 / +0.2759*. Phi fails on a different axis
entirely.

## The three-case pattern — three answers, three different reasons

```
Qwen3-8B   conventional success → ADMIT         (nothing fails)
Qwen3-4B   conventional success → DECLINE       (destination intervals; collateral fine)
Phi-3.5    conventional success → WOULD-DECLINE (destination fine; collateral 3.9x over)
```

The two failures are **orthogonal**, which empirically excludes a single-threshold
explanation of the licence.

## Registry outcome

Of 18 modern criteria: **7 PASS_ON_RECORDED_EVIDENCE · 3 FAIL_ON_RECORDED_EVIDENCE ·
6 UNADJUDICABLE_FROM_HISTORY · 2 NOT_APPLICABLE/NOT_COMPARABLE.** Missing was never counted
as pass or as fail.

## Not re-adjudicable, and left that way

Qwen2.5 (`QWEN2_5_NOT_RESCORABLE` — git-LFS pointers); MetaTool (binary, `|A\{g,s}| = 0`, so
`OTHER_WRONG` is undefined); Mistral and Llama (no intervened per-sample records). Nothing was
fetched, rerun or inferred to fill these.

## Files

`HISTORICAL_EVIDENCE_MANIFEST.json` · `MODERN_CRITERION_REGISTRY.csv` · `PHI_ARTIFACT_AUDIT.md` ·
`PHI_DESTINATION_RECONSTRUCTION.csv` · `PHI_COLLATERAL_RECONSTRUCTION.csv` ·
`PHI_LICENSE_STRESS_TEST.md` · `PHI_STATISTICAL_UNCERTAINTY.json` · `PHI_CONTROL_MAPPING.md` ·
`QWEN2_5_RESCORABILITY_AUDIT.md` · `METATOOL_RESCORABILITY_AUDIT.md` ·
`LLAMA_MISTRAL_BOUNDARY_NOTE.md` · `HISTORICAL_LICENSE_COMPARISON.csv` ·
`THREE_CASE_CANONICAL_COMPARISON.md` · `PAPER_IMPLICATIONS.md` · `CLAIM_BOUNDARY.md` ·
`FINAL_EXPERIMENT_STOP_DECISION.md` · inventory / scan / hashes / handoff.
