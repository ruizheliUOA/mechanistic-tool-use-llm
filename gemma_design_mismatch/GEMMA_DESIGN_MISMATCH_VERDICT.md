# Gemma design-mismatch verdict

## `GEMMA_DESIGN_MISMATCH_SUPPORTED`

Located precisely: **the dose-selection rule**, not the estimator and not the site.

## The evidence

Frozen DEV dose trajectory (`DOSE_SELECTION.json`, n = 193 channel errors,
exposed-correct 23, all-correct denominator 471). No GPU was used; these are
already-computed artifacts.

| arm | exits | gold | other | target-hit | target-gain | broke | collateral(all) | admissible |
|---|---|---|---|---|---|---|---|---|
| zero @ 0.0 | 0 | 0 | 0 | — | 0.000 | 0 | 0.000 | False |
| **d_grad @ 0.125** | **51** | **34** | 17 | **0.667** | **0.088** | **0** | 0.000 | **True ← SELECTED** |
| d_grad @ 0.25 | 95 | 62 | 33 | 0.653 | 0.150 | 3 | 0.0064 | True |
| d_grad @ 0.5 | 132 | 100 | 32 | 0.758 | 0.352 | 9 | 0.0191 | True |
| d_grad @ 1.0 | 152 | 126 | 26 | 0.829 | 0.518 | 11 | 0.0234 | True |
| d_grad @ 2.0 | 158 | 133 | 25 | 0.842 | 0.560 | 12 | 0.0255 | True |
| d_Lobs_diffmean @ all | 2–21 | 0–7 | 2–14 | 0.000–0.333 | **negative** | 0 | 0.000 | False |

## What this shows

1. **Target-hit rises monotonically with dose** — 0.667 → 0.842. This is *not* the
   familiar "more movement, worse destination" pattern. Gemma's higher doses are
   better on destination quality, not merely noisier.
2. **Target-gain rises 6.4×** — 0.088 → 0.560.
3. **Exits rise 3.1×** — 51 → 158. Since both failing conditions were
   *interval-width* failures driven by `source_exits = 27`, this is the direct
   remedy for what actually failed.
4. **Every dose from 0.125 to 2.0 was admissible**, and all-correct collateral
   stayed under the 0.05 budget throughout (max 0.0255 at q = 2.0).
5. **The rule `min(admissible)` therefore selected the single worst cell on every
   destination metric**, because it optimises for safety margin and is blind to
   evidence budget.
6. **The estimator was not the problem.** `d_Lobs_diffmean` produces near-zero
   gold arrivals and *negative* target gain at every dose — the gradient
   estimator was the correct choice. The mismatch is not estimator-side.

## Interpretation — required wording

> A retrospective/exploratory design search identifies a Gemma configuration with
> stronger destination-correct behaviour, supporting intervention-design mismatch
> as a plausible explanation for the historical DECLINE. The mismatch localises to
> the dose-selection rule `min(admissible)`, which chose the dose with the
> smallest evidence budget among six admissible options.

**This is not an ADMIT and cannot become one on this data.** The trajectory is
DEV, already consumed during development. `GEMMA_FORMAL_DECLINE` remains
immutable. Only a fresh prospective population can convert this into a formal
result.

## The honest caveat

Exposed-correct collateral is the unresolved risk. On DEV, breaks rise 0 → 12 as
dose rises. The all-correct denominator (471) keeps the *frozen* endpoint under
budget, but the *exposed* denominator is only 23 rows on DEV — far too small to
bound the exposure-conditional rate at α = 0.05, which needs ≥ 59 exposed-correct
rows with zero breaks. At q = 1.0 with 11 breaks, the exposure-conditional rate
would almost certainly fail. **A higher dose buys destination evidence and spends
preservation evidence.** Any C* must be judged on both, and the preservation side
is where it is most likely to fail.
