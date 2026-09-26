# The three-case canonical comparison

Three interventions. All three look successful by conventional criteria. The modern licence
returns three different answers, and — decisively — **for three different reasons**.

| | **Case A — Qwen3-8B** | **Case B — Qwen3-4B** | **Case C — Phi-3.5 (historical)** |
|---|---|---|---|
| evidence class | FORMAL_CONFIRMATORY | PROSPECTIVE_FORMAL_DECLINE | RETROSPECTIVE stress test |
| conventional view | Net +40, Fixed 40 / Broke 0 | Net +39, Fixed 40 / Broke 1 | Net +55, acc +10.04pp, beats 4 placebos |
| direction specificity | 0 / 59 randoms, p = 0.0167 | 0 / 59 randoms, p = 0.0167 | real +55 vs best placebo +14 (Net-based) |
| reverse | 1 exit | **0 exits** | +14 vs +55 |
| target-hit | **0.7308** [0.6042, 0.8462] | 0.5781 **[0.4559, 0.6970]** | 0.6897–0.7353 (point only) |
| Target Gain | **0.2759** [0.1149, 0.4253] | 0.0806 **[−0.0484, 0.2016]** | +0.1864 to +0.2857 (point only) |
| collateral (all-correct) | **0/211 = 0.0000** | **1/214 = 0.0047** | **52/264 = 0.1970** |
| collateral (eligible-at-risk) | 0/6 | 1/50 | **52/93 = 0.5591** |
| **licence** | **ADMIT** | **DECLINE** | **WOULD-DECLINE** |
| **failing dimension** | none | **destination confidence** (interval) | **collateral** |

## The pattern

```
conventional success  →  ADMIT           (A: everything holds)
conventional success  →  DECLINE         (B: destination intervals fail; collateral fine)
conventional success  →  WOULD-DECLINE   (C: destination fine; collateral fails 3.9x)
```

## Why this is the strongest form of the argument

If all three had failed on the same criterion, a reviewer could reasonably say the licence is
one threshold wearing a conjunction's clothing. They do not.

- **B passes collateral and fails destination.** Target Gain 0.0806 with an interval spanning
  zero, collateral 0.0047.
- **C passes destination and fails collateral.** Target Gain up to +0.2857 and target-hit
  0.7353 — *better point estimates than the ADMIT on one channel* — with collateral 0.1970.

The two failures are **orthogonal**. That is direct empirical evidence that destination
correctness and collateral control are non-redundant evidence dimensions, which is precisely
what a conjunction is for and what an aggregate metric cannot express.

## The sentence this licenses

> Aggregate steering success is insufficient evidence of trustworthy correction — and the
> insufficiency is not one-dimensional: interventions with equally favourable aggregates fail
> for different, non-substitutable reasons.

## What it does not license

Case C is retrospective and partial: six modern criteria are unadjudicable from the 2026-03
record. Cases A and B used a single frozen protocol; Case C had three competing routers, so
its per-channel destination numbers are cascade-level and `NOT_COMPARABLE`. The comparison is
about **which evidence dimension fails**, not a like-for-like three-way trial.
