# Paper implications

## The objection this addresses

Reviewer A, from the sufficiency audit: *"Given 2605.07990 and PRISMS, the readable+steerable
contribution is gone, and what remains is an evaluation wrapper on standard components."*

A wrapper ratifies. This one **refuses** — including its authors' own prior headline result.

## Assessment of the licensing-materiality claim

> SAKIKO's modern licensing protocol is not a cosmetic evaluation wrapper, because applying
> its recoverable criteria to frozen historical outcomes changes which interventions would be
> considered admissible.

**`SUPPORTED`**, with a stated scope limit.

Supported because: the Phi-3.5 reversal is exact, not approximate; it is driven by a quantity
(`clean_damage` 52/264) that the historical analysis *itself recorded and published*; it is
robust to the denominator debate (fails on both, 3.9× and 11.2×); and it is adverse to the
authors' interest.

Scope limit: one historical case. Qwen2.5 is LFS-blocked, MetaTool is structurally binary,
Mistral and Llama lack intervened per-sample records. The claim is "changes at least one
conclusion", not "changes conclusions in general".

## Assessment of the stronger claim

> Aggregate steering success is insufficient evidence of trustworthy correction.

**`SUPPORTED`**, and more strongly than before this analysis, because the three cases now
fail on **different, non-substitutable dimensions** (see `THREE_CASE_CANONICAL_COMPARISON.md`).
A single-threshold explanation of the licence is empirically excluded.

## How to use it in the manuscript

**Figure 4 becomes real.** The sufficiency audit listed it as conditional on this analysis;
it can now be drawn: three settings, one row each, coloured by which criterion fails.

**One paragraph in the Discussion**, phrased self-critically:

> Applying the licence to our own earlier headline result refuses it. The Phi-3.5 intervention
> that motivated this project — Net +55, superior to four placebo controls — breaks 52 of 264
> baseline-correct predictions (19.7%; 55.9% of those it actually touched), against a 5%
> bound. The collateral figure was recorded in 2026 and not treated as decisive. The data did
> not change; the licence did.

**Required labelling, everywhere it appears:** retrospective, partial, and explicitly *not* a
claim that the complete modern protocol was applied to Phi.

## Residual limits this does not fix

- Still one dataset; cross-dataset transfer untested.
- Still two prospective formal settings.
- The licence's *thresholds* remain inherited from the Qwen3-8B programme and are not
  independently validated — a reviewer may still ask why 0.05 rather than 0.10. The honest
  answer is that they were frozen before the Qwen3 outcomes and applied unchanged here.
