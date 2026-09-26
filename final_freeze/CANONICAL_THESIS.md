# Canonical thesis — frozen

> Tool-using language models can make consequential errors before execution by
> selecting the wrong action path. SAKIKO first identifies model-specific
> directional error channels and selectively intervenes on their internal
> representations, showing that a meaningful subset of these errors can be
> behaviourally improved across multiple model families and scales. Extending the
> analysis with structural controls, destination resolution and preservation
> accounting reveals that aggregate improvement, destination correctness and
> preservation of already-correct behaviour are **distinct properties that do not
> imply one another**: a real and structurally specific gain can coexist with
> majority damage to the correct samples the router touches, and two interventions
> with near-identical aggregate gain can land their corrections in substantially
> different places. SAKIKO therefore couples correction with destination-resolved
> verification and evidence-based licensing, assigning only the strongest
> claim, up to repair, supported by structural specificity, destination composition,
> preservation and statistical evidence — and explicitly declining stronger
> conclusions the evidence does not carry. **The procedure declines in practice,
> not only in principle:** of three prospectively evaluated modern models, one was
> admitted and two were formally declined despite favourable destination point
> estimates, because their intervals crossed the frozen boundary.

## Why this wording and not the draft

Two changes from the target paragraph, both forced by evidence:

- **"across multiple model families and scales"** — never "controlled scaling
  study" (S-3 population discipline; the models were not matched on protocol).
- **the explicit "do not imply one another"** — this is the paper's actual
  finding, and it is now carried by *two* independent examples (Phi preservation;
  activation vs score-space destination), not one.

## One-line form

> SAKIKO turns a correction from an *assumed* repair into an *adjudicated* one.

## Amendment — 2026-09-10 (terminology only)

Aligned with the title *When Does Correction Become Repair?*: "target-directed
repair" → "destination correctness"; "the strongest correction claim" → "the
strongest claim, up to repair,"; one-line form rewritten. Previous one-line form:
"SAKIKO turns selective intervention from an *assumed* correction mechanism into an
*adjudicated* correction process." No claim changed. See
`writing/TERMINOLOGY_MIGRATION_RECORD.md`.
