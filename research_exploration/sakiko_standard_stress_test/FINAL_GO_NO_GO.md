# Final verdicts and decision

## Verdicts

| axis | verdict |
|---|---|
| **Benchmark** | `SERIOUS_BUT_NONFATAL_LIMITATION` — fatal to transfer claims, not to existence/refusal/taxonomy claims |
| **Licence calibration** | `CONSERVATIVE_BUT_UNCALIBRATED` — N=3, no external criterion, thresholds are conventions |
| **Standard usefulness** | `MODERATE_INCREMENTAL_VALUE` — 4 of 8 settings reported differently, but 2 of those 4 only enforce CI reporting that good practice already implies |
| **Novelty** | `DEFENSIBLE_BUT_NARROW` — see limitation below |
| **ICLR readiness** | `BORDERLINE` — panel mean 5.2, weak accept conditional on narrowing |

## Why the standard deserves to exist

Because it demonstrably changes what gets reported. Four of eight adjudicated settings would
have been written up more favourably without it — three as successes — and one of those three
**was** published as a success by this project in 2026-03, with `clean_damage: 19.7%` sitting
unremarked in the same results file. The licence is not a theory of correctability; it is a
prespecified rule that makes destination composition, collateral exposure and evidential
precision mandatory rather than optional, and its own authors' prior headline result is the
proof that optional was not enough.

## Why a reviewer could reasonably conclude it does not

Because most of the filtering is done by controls the field already expects. `Net > 0` plus a
matched-random null removes three of nine units on its own; adding destination accounting
removed **none** beyond that on any observed formal unit. What remains uniquely SAKIKO's is a
collateral estimand that the audit shows is itself mis-specified in the frozen primary
(1/223 vs 1/11), a non-vacuity check, and a confidence-interval requirement that is ordinary
statistical hygiene. A reviewer can fairly say: this is careful practice, formalised, on one
benchmark, with one positive result and uncalibrated thresholds — a workshop contribution
rather than a conference one. **That reading is defensible and I cannot refute it from the
evidence.**

## The strongest claim that survives universal DECLINE

> A prespecified staged licence, applied prospectively to nine tool-decision settings across
> four model families, localises where evidential legitimacy fails — support, readability,
> specificity, destination, preservation, or certification — and refuses interventions that
> conventional aggregate reporting accepts, including the authors' own prior published result.

This is true whether or not any future setting ever ADMITs, which is the test that the
contribution is about evaluation rather than cherry-picked success.

## What would falsify the licence's value

Concretely: (a) a rule of the form `specificity ∧ CI-qualified destination` reproducing all
nine decisions — **partly true already**, since destination accounting added nothing beyond
specificity here; (b) exposure-conditional collateral proving uninformative where the frozen
denominator is informative — the audit found the reverse; (c) two independent analysts
assigning different failure classes to the same unit; (d) the taxonomy failing to recur on a
second action ontology.

## Decision

```
B. SUBMIT, BUT NARROW THE CLAIM SUBSTANTIALLY
```

Submit. Do not run another experiment — neither candidate (second benchmark, estimator×site)
changes any verdict, and the benchmark one is not runnable without building a new instrument.

**The required narrowing:**

1. **Claim Level 2, not Level 3.** A prespecified evidential standard for when movement may be
   called correction. **Not** a classifier of which errors are correctable.
2. **State that destination accounting changed no decision beyond specificity** on the observed
   formal units, and that it is conceptually load-bearing but empirically untriggered here.
3. **Report both collateral denominators everywhere, including for the ADMIT** (0/6, CP95 upper
   ≈0.46; DEV estimate 26.7% at the same channel and dose). The frozen primary stays primary;
   the exposure-conditional quantity is reported beside it.
4. **Describe thresholds as a conservative operationalisation**, benchmark-agnostic in
   definition but plausibly W2C-specific in numerical calibration.
5. **Keep "channel-dependent readability" at `SUGGESTIVE`** — channel and benchmark are
   perfectly confounded.
6. **Disclose the low-yield gap**: a near-zero-exit intervention passes the formal conjunction.

## Stated limitation of this audit

I did **not** run a fresh primary-source literature sweep in this phase; I am relying on the
prior novelty audit (which located arXiv 2605.05715, *Decodable but Not Corrected by Fixed
Residual-Stream Linear Steering*, as the closest neighbour). The `DEFENSIBLE_BUT_NARROW` novelty
verdict is therefore carried forward, not re-established. Before submission, the Related Work
section needs a current search against the reliability/abstention/conformal-risk literature —
the closest conceptual prior may not be a steering paper at all, and that check has not been done.
