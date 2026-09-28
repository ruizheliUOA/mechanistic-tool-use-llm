# Qwen3-8B Stage 0/1 Protocol V2

Status: **FROZEN_BEFORE_V2_SPLIT_AND_BEFORE_ANY_V2_CHANNEL_COUNT**

Protocol version `QWEN3_8B_STAGE0_1_V2`, superseding `QWEN3_8B_STAGE0_1_V1`.

V2 is **one disclosed development-stage allocation amendment**. It does not
lower support thresholds, change the direction estimator, select channels
manually, or access the sealed evaluation split.

## Scope of the amendment

Only these change:

- allocation of the already authorized 3104 non-test rows
- V2 split index and split hashes
- V2 population manifests
- formal adoption of the audited long-sequence attention backend

Everything else is **inherited verbatim from V1**. The keys `model`, `thinking_mode`, `readout`, `channel_discovery`, `direction`, `layer_mapping`, `router`, `geometry`, `stage2_branches`, `determinism`, `output_schemas`, `retry_failure`, `forbidden` were copied
programmatically from the committed V1 protocol JSON
(SHA256 `83a66bcefb8d3dcc69d7631a4380d3804fc3c393724fa885d74f4e7d21575b7c`), not restated from memory.

Unchanged and explicitly re-affirmed:

- model `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, tokenizer revision identical
- chat template SHA256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- thinking rule: deterministic empty think block, `enable_thinking=False`
- candidate scoring: 4 teacher-forced forwards, mean log probability over complete candidate tokens
- support gate: train error >= 60, train correct reference >= 60, dev error >= 30, dev correct reference >= 30
- direction estimator: mean(reference activations)-mean(error activations)
- `L_obs` = 26, `L_inj` = 21, site `model.model.layers[L].mlp forward output, after MLP internals and before decoder residual addition`
- Router family, preprocessing, tau grid [0.4, 0.5, 0.6, 0.7, 0.8], tau rule `smallest tau with precision >=0.50`, eligibility `support + direction + valid full-effect dev geometry + dev ROC-AUC >=0.75 + selected-tau precision >=0.50`
- geometry primary `Q_sum_channel=sum(a_offaxis)/sum(a_dec)`, secondary `negative_sign_fraction`
- Stage 2 endpoint definitions and control requirements
- sealed evaluation identity

## Allocation

See `QWEN3_V2_SPLIT_RULE.md`. Targets are TRAIN 2000 and DEV
1104 over the inherited 3,104 non-test rows, gold-stratified by
deterministic UUID hashing with the fixed salt `SAKIKO_QWEN3_V2_ALLOC_V1|`.

## Attention backend

The audited long-sequence guard is formally adopted as the V2 execution backend
for sequences exceeding 4096 tokens; shorter sequences use the unmodified
upstream implementation. Disposition
`ATTENTION_BACKEND_ACCEPTED`.

It is **not** claimed to be bitwise identical to upstream at all lengths. The
audited evidence is: 864/864 all-layer comparisons bitwise identical across the 24 V1-affected sequences; not universal across lengths.

## Artifact reuse

Outcome `CPU_SUFFICIENT`. Baseline rows and both frozen
activation sites are inherited from V1 and are a pure function of the sample,
so reallocation cannot change any score, prediction, runner-up or margin.

## K branches (copied verbatim from V1)

- K >= 5 → branch `A`: exact one-sided Kendall tau: lower dev Q_sum predicts higher evaluation destination selectivity
- K = 4 → branch `B_LOW_POWER`: same exact one-sided Kendall test; success only for extremely strong ordering
- 1 <= K <= 3 → branch `C`: real direction versus prospectively generated matched-random directions on destination selectivity for every eligible channel; predictor claim withdrawn
- K = 0 → branch `D`: `NO_ELIGIBLE_CHANNELS`

There is no "three channels or stop" rule. Project-management deadlines are not
scientific thresholds.

## Atomic output rule

Before the first V2 computation the output namespace must be empty and the
split file, support ledger and state file must not exist. Outputs are written
to a temporary file and promoted with an atomic replace. A crash before
promotion permits retrying the same committed runner, logged in the retry
ledger. After promotion there is no recomputation, no reseeding, no resizing
and no alternative split.

## Disclosure

- V1 used 2556 train / 548 dev and returned `NO_SUPPORT_ELIGIBLE_CHANNELS`
- the failure exposed an outcome-conditioned reference-support bottleneck
- V2 changes only non-test development allocation and the audited attention backend
- the evaluation split remained sealed and unaccessed
- no support threshold was lowered, no channel was manually included, no alternative V2 split was tried

Future evidence must be labelled:

> prospective evaluation following one disclosed development-stage allocation amendment

Stage 2 is **not** authorized by this protocol.
