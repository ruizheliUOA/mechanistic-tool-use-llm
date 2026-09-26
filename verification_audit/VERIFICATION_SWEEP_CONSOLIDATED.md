# Consolidated verification sweep

Every claim traced to a primary artifact and independently recomputed.

## Verified exactly — no discrepancy

| setting | claim | artifact value | status |
|---|---|---|---|
| Qwen3-8B | target-hit 38/52 = 0.7308 | 0.7308 | **exact** |
| Qwen3-8B | conjunction 10/10, ADMIT | 10/10 | **exact** |
| Qwen3-8B | preservation 0/6 at record level | 6 correct rows, all routed, 0 breaks | **exact** |
| Qwen3-8B | add-one p = 0.016667 | 1/60 | **exact** |
| Gemma | 8/10, failed conditions 3 and 6 | conditions 3, 6 | **exact** |
| Gemma | McNemar 17:0 → 7.629e-06 | 0.5^17 | **exact** |
| Gemma | dose trajectory, all 5 doses | DOSE_SELECTION.json | **exact** |
| Qwen3-4B | 37/64, McNemar 31:0 / 37:0 | record level | **exact** |
| WildToolBench | 1024 turns, acc 0.5703, macro-F1 0.5222 | all | **exact** |
| WildToolBench | CLARIFY→TOOL_CALL 75, 66 dialogues | 75 / 66 | **exact** |
| Qwen3.5 | DEV ROC-AUC 0.727 | `dev_roc_auc: 0.727` | **exact** |
| Qwen3.5 | 12 channels enumerated, 1 eligible | 12 / rfi→tc | **exact** |
| Phi-3.5 | 52/93 = 0.5591 eligible-at-risk | CI [0.4524, 0.6620] | **exact** |
| ACEBench | Qwen2.5-7B accuracy 0.7562 | R0.1 | **exact** |
| ACEBench | 9/10 criteria pass, R0.9 fails | CONDITIONAL PASS | **exact** |

**Not one artifact value was wrong.**

## Defects found — all in the description layer

| # | defect | where | number changed? |
|---|---|---|---|
| 1 | transition-schema handling | adapter built to fix it | no |
| 2 | Clopper–Pearson quoted as frozen CI | my reporting | no |
| 3 | unrouted denominator padding | my new V2 code | no — caught pre-use |
| 4 | "delta 0.000000" overstated | my claim | no |
| 5 | Phi row unlabelled as retrospective | my summary | no |
| 6 | "mechanism failure 0 of 4" scope | my summary | no |
| 7 | abbreviated provenance paths | manuscript CSV | no |
| 8 | `FORMAL_RESULTS.json` ambiguous | manuscript CSV | no |
| 9 | one absent artifact, one filename typo | manuscript CSV | no |
| 10 | claim D1 does not name its channel | manuscript CSV | no |

**Numbers wrong that reached anything: 1** — my Rule-B `n=199`, superseded, never
load-bearing.

## Seed robustness — no verdict is seed-dependent

10 seeds, both interval conditions, all three formal settings. Every margin is
8–25x its seed spread. The ADMIT is an ADMIT under every seed; both DECLINEs fail
under every seed. The seed can only touch conditions 3 and 6 — the rest are
counts, exactness checks, or deterministic Clopper–Pearson.

## New detail worth carrying into the paper

**Qwen3.5's `cannot_answer→tool_call` had 170 DEV errors but only 29 reference
rows**, and was excluded on *reference* support, not error support. The one
support-eligible channel (`request_for_info→tool_call`) then failed readability
at AUC 0.727 against the 0.75 floor, with `train_auc 0.9999` — a large
train/DEV gap.

That is a cleaner story than "Qwen3.5 had no channel": it had abundant errors on
the historical channel and was stopped by a different denominator.

## Outstanding fixes before submission

1. Expand all provenance paths to repo-root-relative
2. Disambiguate `FORMAL_RESULTS.json` (8 Gemma claims currently resolve to the Qwen3-4B file by basename)
3. Fix `FINAL_REPORT.md` → `MISTRAL7B_W2C_SAKIKO_CA_FINAL_REPORT.md`
4. Restore or re-source `PHASE6_EVIDENCE_RECONSTRUCTION_AND_AUDIT.md` (1 claim)
5. Add the channel identifier to claim D1
6. Replace the `same` shorthand with explicit paths
