# S1 — Post-treatment conditioning is prohibited. Part A verdict corrected.

## The error

T1.8 Part A conditioned destination on `M_pre` while controlling for
standardised `dM = M_post − M_pre`.

**`dM` is measured after the intervention and lies on the causal path from
treatment to outcome.** Conditioning on a post-treatment variable induces
collider/mediator bias. The observed coefficient inflation

```
M_pre coefficient  +1.8671  ->  +3.3948   (+82%)
```

is a mechanical consequence of that conditioning, not evidence about the dose
confound. **The test is uninformative in either direction.**

I proposed this test, ran it, and reported SURVIVES. That was wrong, and the
82% inflation should have been the tell — a covariate that *strengthens* the
predictor it is meant to control for is the signature of conditioning on a
mediator, not of a suppressor relationship worth reporting.

## Corrected verdict

```
SURVIVES  ->  UNTESTABLE
```

## What the evidence actually says

Realised standardised displacement is rank-aligned with dose and
inverse-rank-aligned with margin dependence across all three models:

| model | std displacement | dose | margin g |
|---|---:|---:|---:|
| Qwen3-8B | 1.991 | 1.0 | 0.684 |
| Qwen3-4B | 1.526 | 0.25 | 1.577 |
| Gemma | 0.880 | 0.125 | 2.080 |

**The available evidence is CONSISTENT WITH the mechanical account.** A weaker
effective push crosses only near-boundary samples, producing high margin
dependence; a stronger push also crosses distant ones, producing low margin
dependence. That is exactly the observed pattern, on the realised push rather
than nominal dose.

This must be reported in the mechanism section, not buried.

## Standing rule for this project

**Post-treatment conditioning is prohibited.** No analysis may control for a
quantity measured after the intervention when estimating an effect of, or
predictor of, that intervention's outcome.

A valid control for the dose confound requires **pre-outcome variation**:
within-model dose sweeps compared at matched exit rates. The formal runs are
single-dose (Qwen3-8B q=1.0, Qwen3-4B q=0.25, Gemma q=0.125), and the one DEV
sweep that exists lacks `scores_base`. **Not available. No substitute
conditioning scheme will be attempted.**

## Consequence for the paper

The cross-model margin claim is withdrawn to:

> *Margin dependence varies across the three models and is rank-aligned with
> realised intervention displacement. We cannot separate a mechanistic account
> from a push-magnitude account with single-dose artifacts.*

The **within-model** Qwen3-8B result (`d_act − d_ss = −1.538 [−2.417, −0.749]`)
is unaffected: it compares two interventions on the same samples at the same
dose, with no post-treatment conditioning anywhere.
