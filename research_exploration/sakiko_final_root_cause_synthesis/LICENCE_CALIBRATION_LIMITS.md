# Licence calibration limits

## The count

Modern formal licence outcomes: **three**. Qwen3-8B ADMIT, Qwen3-4B DECLINE, Gemma DECLINE —
all on When2Call, two of them on the identical channel `cannot_answer → tool_call`.

## What cannot be estimated

With N=3 (effectively one matched triple), neither

`P(reliable correction | ADMIT)` nor `P(reliable correction | DECLINE)`

is estimable at any useful precision. There is also no independent ground truth for "reliable
correction" — the licence *defines* the outcome it would need to be validated against, so the
quantity is not merely under-sampled but circular without an external criterion.

## The correct characterisation

> **The SAKIKO licence is a conservative evidential rule, not an empirically calibrated
> classifier of intrinsic correctability.**

This is not automatically a flaw. Preregistered evidential rules in other fields are routinely
conventions rather than utility-derived optima. But it must be stated, because the natural
misreading — that a DECLINE estimates a low probability of correctability — is exactly what the
evidence does not support.

## What the simulation does establish, and what it does not

**Does:** the frozen conjunction admits ~93% of trials at the Qwen3-8B effect size and ~19–25%
at the DECLINE effect sizes, given their observed n. That is an operating characteristic, and
it is real.

**Does not:** that this operating point is correct. Nothing in the project establishes what
effect size *ought* to earn a correction licence. Fixing that needs a decision-utility model —
the cost of a wrong-to-wrong redirection versus a repaired error versus a broken correct action
— which does not exist here.

## Consequence for the paper

The paper may claim that the licence is conservative, that it is applied consistently, and that
it localises failure by stage. It may **not** claim that the licence is calibrated, that its
thresholds are optimal, or that DECLINE carries a probabilistic interpretation about the model.
