# Human Git Handoff — historical SAKIKO licence stress test

**CPU-only retrospective analysis. No model, no GPU, no artifact modified. Nothing staged.**

## Verdict

`HISTORICAL_LICENSE_STRESS_TEST_STRENGTHENS_PAPER` → `FREEZE_EXPERIMENTATION_AND_WRITE`

## Staging

```
git add -- research_exploration/historical_sakiko_license_stress_test_v1
```

Note: `research_exploration/iclr_final_scientific_sufficiency_audit_v1/` from the previous
task is still untracked. Stage it separately if you want it recorded:

```
git add -- research_exploration/iclr_final_scientific_sufficiency_audit_v1
```

No `git add .`. Do not stage `final/results/decisive_upgrade_audit/` or the stray root-level
`Under` file.

## Proposed commit

```
audit: retrospective licence stress test refuses the historical Phi-3.5 result
```

## Read first

`PHI_LICENSE_STRESS_TEST.md`, then `THREE_CASE_CANONICAL_COMPARISON.md`, then
`PAPER_IMPLICATIONS.md`.

## The one number to carry into the manuscript

Phi-3.5 historical: **52 / 264 = 0.1970** baseline-correct predictions broken (95% CI
[0.1507, 0.2501]); **52 / 93 = 0.5591** of those actually exposed. Modern bound 0.05.
Recorded in 2026-03 as `clean_damage`, not treated as decisive.

## Mandatory labelling wherever this is used

Retrospective and partial. **Never** write "Phi failed the complete modern formal protocol" —
6 of 18 criteria are unadjudicable from the historical record.

## What was not touched

No existing experiment package modified. Qwen3-8B ADMIT and Qwen3-4B DECLINE intact. No LFS
payload fetched, no historical model rerun, no Qwen3.5, no dataset search.
