# Phi-3.5 licence stress test — result

**`RETROSPECTIVE_HISTORICAL_RE-ADJUDICATION`.** Modern criteria applied *only where
mechanically recoverable* to a frozen historical result. This does not convert the 2026-03
experiment into a modern formal trial and does not claim it does.

## Headline

> **`WOULD_DECLINE_ON_RECORDED_COLLATERAL`**

The historical Phi-3.5 P0 locked-test result — Net +55, +10.04 accuracy points, superior to
all four placebos, reported as a success — **would not receive a modern SAKIKO correction
licence**, on a criterion that was measured at the time and simply not treated as decisive.

## The decisive criterion

Modern rule (`final/results/qwen3_stage2_formal`, conditions 7 and 8):
`clean_collateral_rate ≤ 0.05` at the point estimate **and** at the 95% upper bound.

| denominator | broken | n | rate | Clopper–Pearson 95% | vs 0.05 | verdict |
|---|---:|---:|---:|---|---:|---|
| **A. all baseline-correct** *(the frozen formal metric)* | 52 | 264 | **0.1970** | [0.1507, 0.2501] | **3.9×** | **FAIL** on point and interval |
| **B. Router-fired ∩ baseline-correct** *(eligible-at-risk)* | 52 | 93 | **0.5591** | [0.4524, 0.6620] | **11.2×** | **FAIL** |
| C. unrouted baseline-correct *(never exposed)* | 0 | 171 | 0.0000 | [0.0000, 0.0213] | — | PASS (gating intact) |

Break destinations: **BROKEN_TO_SOURCE 4, BROKEN_TO_OTHER 48.**

Two points make this robust:

1. **It fails on both denominators.** The Qwen3-8B metric-compression audit showed that the
   frozen all-correct denominator can flatter a result by dilution. Here dilution works the
   other way: the eligible-at-risk rate is *worse*. No denominator argument rescues it.
2. **The number was already recorded.** `p0_final_test_eval.json` contains
   `clean_damage: {n: 264, damaged: 52, rate: 19.7}`. The 2026-03 analysis measured it,
   printed it, and still called the result a success — because the success definition was
   `Net > 0` plus a placebo battery. **The data did not change; the licence did.**

## What Phi passes

This is not a story about a bad result.

| criterion | value | verdict |
|---|---|---|
| channel support | 112 / 77 / 59 errors | PASS |
| direction specificity (historical Net-based) | real +55 vs random +14, reverse +14, wrong-layer +3, mismatched −17 | PASS on the old definition |
| zero control (via unrouted rows) | 0 of 255 changed | PASS |
| target-hit (point) | 0.7353 / 0.7143 / 0.6897 | PASS — comparable to the Qwen3-8B ADMIT's 0.7308 |
| Target Gain (point) | +0.2857 / +0.2338 / +0.1864 | PASS — comparable to the ADMIT's +0.2759 |
| mechanical integrity | every committed aggregate reproduced exactly | PASS |

**Phi's destination evidence is as good as the Qwen3-8B ADMIT's.** It fails on a different
axis entirely.

## What is unadjudicable, and must be stated as such

K = 59 fresh randoms with an add-one p-value; the frozen bootstrap interval machinery; the
score-space comparator; sealed one-shot execution; modern Router AUC/τ-precision. None was
recorded in 2026-03. **Missing ≠ pass, and missing ≠ fail.** Of the eighteen registry
criteria: 7 `PASS_ON_RECORDED_EVIDENCE`, 3 `FAIL_ON_RECORDED_EVIDENCE`, 6
`UNADJUDICABLE_FROM_HISTORY`, 2 `NOT_APPLICABLE` / `NOT_COMPARABLE`.

## Permitted and forbidden statements

**Permitted:** *"The modern SAKIKO licence would refuse the frozen Phi result on the
collateral criterion, which is exactly recoverable from committed per-sample records."*

**Forbidden:** *"Phi failed the complete modern formal protocol."* Six criteria cannot be
evaluated from the historical record, so no complete conjunction was applied and none may be
claimed.

## Why this is the sharpest available evidence for the licensing thesis

The refusal is **self-inflicted**. SAKIKO's own headline historical result — the one that
justified the project continuing — does not survive SAKIKO's own current standard. A protocol
that only ever ratifies its authors' prior conclusions is a wrapper. This one overturns one.
