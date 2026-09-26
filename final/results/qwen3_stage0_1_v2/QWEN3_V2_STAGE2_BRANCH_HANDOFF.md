# Qwen3 V2 Stage 2 branch handoff

## STAGE_2_NOT_AUTHORIZED

This task ends before Stage 2 preregistration and before any evaluation/test access.
No Stage 2 preregistration, runner, execution freeze, preflight or test prediction was created.

K (support-eligible) = **3**
K_stage2 (support + direction + Router + geometry) = **3**

Inherited branch: **C**

Future primary: real direction versus prospectively generated matched-random directions on destination selectivity for every eligible channel

## Claim boundary

Future evidence from this line must be labelled:

> prospective evaluation following one disclosed development-stage allocation amendment

not a pristine no-amendment confirmation.

## Disclosure

- V1 allocation: {'dev': 548, 'sealed_evaluation': 548, 'train': 2556}
- V1 result: NO_SUPPORT_ELIGIBLE_CHANNELS
- V1 failure cause: outcome-conditioned reference-support bottleneck
- V2 changed: non-test development allocation and the audited attention backend only
- evaluation split sealed and unaccessed: True
- support threshold lowered: False
- channel manually included: False
- alternative V2 split tried: False

## Support-eligible channels ordered by dev Q_sum (low to high)

| rank | channel | Q_sum full | negative-sign fraction |
|---:|---|---:|---:|
| 1 | cannot_answer__to__direct | 0.611052 | 0.522013 |
| 2 | request_for_info__to__tool_call | 0.710761 | 0.557377 |
| 3 | cannot_answer__to__tool_call | 0.724542 | 0.808383 |

## Complete eligibility

| channel | support | direction | Router | geometry | Stage2 | exclusion |
|---|---|---|---|---|---|---|
| tool_call__to__direct | False | False | False | False | False | NOT_YET_EVALUATED |
| tool_call__to__request_for_info | False | False | False | False | False | NOT_YET_EVALUATED |
| tool_call__to__cannot_answer | False | False | False | False | False | NOT_YET_EVALUATED |
| direct__to__tool_call | False | False | False | False | False | NOT_YET_EVALUATED |
| direct__to__request_for_info | False | False | False | False | False | NOT_YET_EVALUATED |
| direct__to__cannot_answer | False | False | False | False | False | NOT_YET_EVALUATED |
| request_for_info__to__tool_call | True | True | True | True | True | — |
| request_for_info__to__direct | False | False | False | False | False | NOT_YET_EVALUATED |
| request_for_info__to__cannot_answer | False | False | False | False | False | NOT_YET_EVALUATED |
| cannot_answer__to__tool_call | True | True | True | True | True | — |
| cannot_answer__to__direct | True | True | True | True | True | — |
| cannot_answer__to__request_for_info | False | False | False | False | False | NOT_YET_EVALUATED |

## Direction Gram matrix

`{"channels": ["cannot_answer__to__direct", "cannot_answer__to__tool_call", "request_for_info__to__tool_call"], "flags_abs_cosine_gt_0_4": [], "signed_cosine_matrix": [[0.999999999157464, 0.38486166655469317, 0.059348993378407044], [0.38486166655469317, 0.9999999982425577, 0.3245188376527677], [0.059348993378407044, 0.3245188376527677, 1.0000000006346297]]}`

Evaluation rows, IDs, labels, predictions, scores, support counts, aggregates and channel
membership were not accessed at any point.

