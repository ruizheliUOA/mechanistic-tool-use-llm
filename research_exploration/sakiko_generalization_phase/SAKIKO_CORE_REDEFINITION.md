# SAKIKO-Core — benchmark-agnostic redefinition

## The separation that resolves the previous audit's central confusion

```
Adjudicable → [ Readable → Steerable → Correctable ] → Licensable
   (instrument)      (properties of M,c,I)              (evidence)
```

**Adjudicable** and **Licensable** are properties of the *study*. **Readable / Steerable /
Correctable** are properties of the *setting*. The 10-condition licence conflated them, which is
why "Gemma DECLINE" was ambiguous between "the intervention doesn't work" and "we couldn't tell."

Under the separation, Gemma reads correctly and unambiguously:

> Adjudicable ✓ · Readable ✓ (0.9260) · Steerable ✓ (0/59 randoms, reverse dead) ·
> Correctable **point-positive** (t-hit 0.630, TG +0.073) · **Licensable ✗** (CI lower 0.4475,
> −0.0327) · Preservation **untested at usable precision** (1/11 exposed).

Nothing in that sentence says Gemma is uncorrectable, and nothing hides that the evidence failed.

## Definitions independent of When2Call

Let `A_D` be the dataset-native admissible action set at a decision point, `g,s ∈ A_D`, `g≠s`,
channel `c = (g→s)`, intervention `I`.

- **Adjudicable(c, D)** — the instrument yields deterministic gold, an error population, a
  reference population, an exposed-correct population, and `|A_D \ {g,s}| ≥ 1`.
- **Readable(c, M, Φ)** — under probe family `Φ` at an observation site, channel-error samples
  are separable from reference samples above a task-calibrated bound.
- **Steerable(c, M, I)** — `I` produces a causal change exceeding matched-random, reverse, zero
  and wrong-site nulls. *Not* "the prediction changed."
- **Correctable(c, M, I)** — conditional on movement, arrivals concentrate on `g` rather than on
  `A_D \ {g,s}`. Requires `|A_D| ≥ 3`; at `|A_D| = 2` leaving the source **is** arriving at gold
  and the property is undefined. This is why MetaTool is structurally ineligible.
- **Preserving(c, M, I)** — `P(correct → wrong | exposed to I)` is acceptably small.
- **Licensable** — the evidence supports the above at stated uncertainty `δ`.

## Where preservation belongs

**Not inside Correctable.** The audit showed destination and preservation fail independently:
Phi passes destination (t-hit 0.69–0.74) and fails preservation (52/93 exposed = 55.9%);
Gemma `ca→direct` passes destination (0.8636) with preservation *untestable*. Fusing them makes
both unfalsifiable. Preservation is a fourth property, reported with its own uncertainty.

## Consequence for the thesis

The original insight — *readable ⇏ steerable ⇏ correctable* — survives and is **strengthened**
by being freed from the threshold apparatus. The 10-condition licence becomes **one conservative
operationalization** of `Licensable`, not the definition of `Correctable`.
