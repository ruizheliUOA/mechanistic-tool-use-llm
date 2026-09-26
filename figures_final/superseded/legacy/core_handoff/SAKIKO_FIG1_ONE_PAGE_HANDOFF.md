# SAKIKO Figure 1 — one-page handoff

## What is SAKIKO?

A framework for deciding whether a behavioural change produced by an internal
intervention may honestly be called a **correction**. It operates on a frozen
language model making a pre-execution multiclass tool decision, conditionally
edits one hidden state, and then adjudicates the evidence.

## What must Figure 1 show?

Two things at once:
1. **Mechanically** — how SAKIKO observes and conditionally intervenes inside a
   frozen transformer.
2. **Scientifically** — what additional evidence is required before the resulting
   change may be called a correction.

The transformer is the visual backbone. The claim structure wraps it.

## What is the true computation order?

**TWO PASSES.** `L_inj = 21` executes *before* `L_obs = 26` (zero-based, 36-layer
model), so diagnosis and intervention cannot share a forward pass.

```
PASS 1  input -> frozen model -> cache h_obs at layer 26
        [ Router scores the cached vector OFFLINE, in NumPy ]
PASS 2  same input -> frozen model -> h' = h + alpha*d_c at layer 21
                   -> remaining frozen layers -> action
```

## What is the key scientific distinction?

**`Correctable` is a property claim. `Licensable` is a judgement about that claim.**

- Readable / Steerable / Correctable — claims about the tested setting.
- Adjudicable — an entry condition on the *study*.
- Licensable — an epistemic decision about whether the evidence justifies
  asserting Correctable.

Correctable and Licensable must be visually different **kinds of object**, not
consecutive boxes in a row.

And: for a `K >= 3` action space, leaving the wrong answer is not the same as
reaching the right one. Three destinations, not two.

## What must never be drawn?

- The two-pass implementation as one pass, or `h_obs -> Router -> h_inj` inline.
- `Licensable` as a fourth scientific property.
- Any model name under a failing property (this reads as "Correctable = FAIL"
  when what failed was the licence).
- "Aggregate correctness separates SOURCE_RETAINED from OTHER_WRONG" — backwards.
  It collapses them.
- Activation space as the only correction surface.
- `DECLINE` as intrinsic uncorrectability.
- A historical `ADMIT` as a safety certificate.
- Any empirical number or CI in Figure 1.

## What may the designer change freely?

Everything visual: composition, panel arrangement, shape vocabulary, spacing,
typography, colour palette, arrow style, visual metaphor, aspect ratio, and
whether the claim layer sits below, beside, or around the mechanism.

Constraints that are **not** visual: execution order, causal direction, the
three-way destination partition, the two disjoint populations, and the
property/licence distinction.

## Target format

ICLR single column, `\textwidth` = **5.5 in**. Design natively at that width. No
text below ~6 pt at final size. Explanatory prose belongs in the caption.
