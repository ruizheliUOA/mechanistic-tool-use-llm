# SAKIKO Figure 1 — information hierarchy

## TIER 1 — must be obvious within 5-10 seconds

If a reviewer cannot get these from a glance, the figure has failed.

1. The transformer is **frozen** (theta never updated).
2. There are **two passes**: diagnose, then intervene.
3. `h_obs` is read at the observation site.
4. A lightweight **Router** decides whether to act.
5. `h' = h + alpha*d_c` is written at the injection site.
6. The output is a **multiclass action**, not a scalar.
7. Three destinations: `SOURCE_RETAINED` / `GOLD_ARRIVAL` / `OTHER_WRONG`.
8. `Readable -> Steerable -> Correctable` is an ordered chain of property claims.
9. **`Correctable` and `Licensable` are different kinds of object.**

Item 9 is the paper's contribution. If the figure gets everything else right and
9 wrong, it is worse than no figure.

## TIER 2 — must be present, visually secondary

- offline channel discovery from the model's own errors
- direction estimation producing a channel-specific `d_c`
- the preservation population (baseline-correct, router-exposed)
- causal controls (real vs zero / random / reverse / wrong-layer)
- Adjudicability as an entry condition
- the five licensing criteria

These may be smaller, greyer, or compressed. They may not be omitted.

## TIER 3 — caption or Methods; not necessarily drawn

- K = 59 and the exact random-direction construction
- the full 10-condition formal conjunction
- the complete estimator list and all score-space variants
- dose grid, support-gate numbers, tau grid
- power rules, CI procedures, bootstrap parameters
- any numerical threshold

A designer who is short of space should cut from Tier 3 first, then compress
Tier 2, and never touch Tier 1.
