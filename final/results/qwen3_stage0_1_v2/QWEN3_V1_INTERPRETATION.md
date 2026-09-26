# What the V1 result `NO_SUPPORT_ELIGIBLE_CHANNELS` does and does not establish

## What it is

`NO_SUPPORT_ELIGIBLE_CHANNELS` is an **estimator-feasibility result**.

It states exactly one thing: under the frozen outcome-conditioned DiffMean
estimator and the frozen support gate
(train_error >= 60,
train_correct_reference >= 60,
dev_error >= 30,
dev_correct_reference >= 30),
the V1 allocation of the authorized non-test rows left no directed channel with
enough rows on both sides of the contrast to estimate a direction and evaluate
it on held-out development data.

It is a statement about **population sufficiency under one allocation**, not
about the model's mechanism.

## What it is not

The V1 run stopped after the baseline and the channel ledger. Execution never
reached direction estimation, Router fitting, dev geometry, or any
intervention. Therefore this result is **not** evidence that:

- **Qwen3 directions are ineffective.** No direction was ever estimated. The
  `directions/` namespace was never created.
- **Qwen3 errors are unreadable.** No Router was ever fitted. No ROC-AUC,
  PR-AUC, precision or recall exists for any channel.
- **SAKIKO interventions fail.** No intervention was applied, and no dose was
  ever frozen.
- **Real directions are equivalent to random directions.** No real direction
  and no matched-random control was constructed, so the comparison was never
  made.
- **Qwen3 is not correctable.** No behavioural correction was attempted or
  measured.
- **The hypothesis about readability versus goldward alignment is refuted.**
  Both quantities were unmeasured.

Each of those claims would require quantities that do not exist in the V1
outputs. Asserting any of them from this result would be an inference from
absence.

## What V1 did positively establish

- The frozen readout executes correctly on Qwen3-8B end to end: pilot replay
  was bit-exact (maximum absolute score deviation 0.0, identical prediction and
  runner-up vectors across two repetitions).
- Zero rows were skipped; all four candidate scores were finite for every one
  of the 3104 authorized rows; there were no unresolved ties.
- Substantial directed error structure exists. Several transitions carry large
  error populations, so the model does make the kind of mistakes the programme
  studies.
- The limiting factor was specifically correct-reference support, which is a
  property of the accuracy profile and the allocation, and is analysed in
  `OUTCOME_CONDITIONED_REFERENCE_SUPPORT_WINDOW.md`.

## Status

V1 is complete and immutable. Its artifacts, runner history, retry ledger,
attention-guard history and this mechanical result are preserved unchanged.
The V2 amendment that follows changes only the allocation of already authorized
non-test rows and the formal adoption of the audited attention backend; it does
not reinterpret this result.
