# Outcome-conditioned reference support has an accuracy-dependent feasibility window

This document states a scoping finding exposed by the V1 result
`NO_SUPPORT_ELIGIBLE_CHANNELS`. Every support count and threshold is read from the
committed V1 artifacts; none is restated from memory.

## The estimator that produces the constraint

Read directly from the committed V1 protocol
`QWEN3_8B_STAGE0_1_PROTOCOL.json`, key `direction`:

- estimator: `DiffMean`
- formula: `mean(reference activations)-mean(error activations)`
- reference population: `TRAIN gold=g and baseline prediction=g`
- error population: `TRAIN gold=g and baseline prediction=p`
- sign: `positive intended to increase gold relative to source`

Frozen support gate, read from key `channel_discovery.support_gate`:

| criterion | threshold |
|---|---:|
| train_error_min | 60 |
| train_correct_reference_min | 60 |
| dev_error_min | 30 |
| dev_correct_reference_min | 30 |

## The structural point

For a channel `g -> p`, both populations are drawn from the *same* gold class
`g`, partitioned by the model's own outcome:

- reference rows are the ones the model got right (`prediction = g`);
- error rows are the ones it got wrong in the specific direction
  (`prediction = p`).

Within a gold class of fixed size, these two populations are complementary and
compete for a finite number of rows. Raising accuracy on class `g` necessarily
moves mass from the error side to the reference side, and lowering it does the
reverse.

Consequently, **outcome-conditioned contrastive estimation is feasible only in
an intermediate accuracy band**:

- at very high accuracy on `g`, the error population is starved;
- at very low accuracy on `g`, the correct-reference population is starved;
- only intermediate accuracy supports both sides of the contrast at once.

V1 fell off the low-accuracy side of this window for its high-error classes,
and off the high-accuracy side for its accurate class. Both failure modes
appear simultaneously in the same run.

## The V1 measurements

Complete 12-transition ledger as committed:

| channel | train error | train correct-ref | dev error | dev correct-ref | support eligible | exclusion |
|---|---:|---:|---:|---:|---|---|
| tool_call__to__direct | 32 | 849 | 8 | 190 | False | TRAIN_ERROR_LT_60;DEV_ERROR_LT_30 |
| tool_call__to__request_for_info | 13 | 849 | 4 | 190 | False | TRAIN_ERROR_LT_60;DEV_ERROR_LT_30 |
| tool_call__to__cannot_answer | 4 | 849 | 1 | 190 | False | TRAIN_ERROR_LT_60;DEV_ERROR_LT_30 |
| direct__to__tool_call | 0 | 0 | 0 | 0 | False | TRAIN_ERROR_LT_60;TRAIN_CORRECT_REFERENCE_LT_60;DEV_ERROR_LT_30;DEV_CORRECT_REFERENCE_LT_30 |
| direct__to__request_for_info | 0 | 0 | 0 | 0 | False | TRAIN_ERROR_LT_60;TRAIN_CORRECT_REFERENCE_LT_60;DEV_ERROR_LT_30;DEV_CORRECT_REFERENCE_LT_30 |
| direct__to__cannot_answer | 0 | 0 | 0 | 0 | False | TRAIN_ERROR_LT_60;TRAIN_CORRECT_REFERENCE_LT_60;DEV_ERROR_LT_30;DEV_CORRECT_REFERENCE_LT_30 |
| request_for_info__to__tool_call | 570 | 128 | 122 | 26 | False | DEV_CORRECT_REFERENCE_LT_30 |
| request_for_info__to__direct | 39 | 128 | 10 | 26 | False | TRAIN_ERROR_LT_60;DEV_ERROR_LT_30;DEV_CORRECT_REFERENCE_LT_30 |
| request_for_info__to__cannot_answer | 8 | 128 | 3 | 26 | False | TRAIN_ERROR_LT_60;DEV_ERROR_LT_30;DEV_CORRECT_REFERENCE_LT_30 |
| cannot_answer__to__tool_call | 400 | 102 | 81 | 16 | False | DEV_CORRECT_REFERENCE_LT_30 |
| cannot_answer__to__direct | 347 | 102 | 81 | 16 | False | DEV_CORRECT_REFERENCE_LT_30 |
| cannot_answer__to__request_for_info | 64 | 102 | 6 | 16 | False | DEV_ERROR_LT_30;DEV_CORRECT_REFERENCE_LT_30 |

Support-eligible channels: **0**.

The binding criterion is `dev_correct_reference_min`
(30). Channels with abundant error support
fail on the reference side, while the one gold class with abundant reference
support produces too few errors.

## Scope limits of this finding

**This finding does not apply to all CAA / ActAdd-style methods.** It applies
specifically to estimators whose *reference* state is defined by the model
producing the correct outcome naturally.

Classical prompt-pair contrastive methods construct their contrast from forced
or externally supplied continuations. They do not require the model to answer
correctly, so they have no comparable accuracy-dependent starvation of the
reference side. Nothing here is evidence against those methods.

Two further limits:

- **Cross-fold re-estimation cannot repair an absolute shortage.** Rotating
  which rows serve as held-out data does not create additional correct rows in
  a gold class; if the class contains too few correct rows in total, every fold
  inherits the shortage.
- **External or synthetic references are not a cost-free workaround.** Sourcing
  reference states from another model, from forced decoding, or from
  constructed text redefines what the reference state *is*. That is a different
  estimator with a different claim, not a repair of this one, and it would
  require its own prospective declaration.

## What this does not claim

It does not claim the thresholds are wrong, that Qwen3 lacks readable error
structure, or that directions estimated some other way would fail. Those
quantities were never reached in V1.
