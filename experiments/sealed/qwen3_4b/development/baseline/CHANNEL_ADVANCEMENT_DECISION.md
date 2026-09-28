# Channel Advancement Decision

## Status: NOT REACHED — Stage D blocked by a hardware limit

The lexicographic advancement gate could not be evaluated, because direction estimation did
not complete. No channel advanced, and no channel was declined on scientific grounds.

## Where each eligible channel stands

| channel | support | readability | direction geometry | DEV steerability | advancement |
|---|---|---|---|---|---|
| `cannot_answer→tool_call` | ELIGIBLE (TRAIN 436 / DEV 220 errors; ref 68/40) | **READABLE** — DEV AUC 0.9237, τ=0.011564, τ-precision 0.500, fires 408/877 | not computed | not computed | **NOT REACHED** |
| `cannot_answer→direct` | ELIGIBLE (TRAIN 174 / DEV 114 errors; ref 68/40) | **READABLE** — DEV AUC 0.8799, τ=0.0, τ-precision 0.864, fires 132/132 | not computed | not computed | **NOT REACHED** |

`request_for_info→tool_call` is the largest transition (TRAIN 517 / DEV 273) and is
`SUPPORT_INELIGIBLE`: its correct-reference population is 54 < 60 on TRAIN. This is the same
outcome-conditioned reference-support bottleneck that produced Qwen3-8B's V1
`NO_SUPPORT_ELIGIBLE_CHANNELS`. The threshold was not lowered to admit it.

## Observations carried forward, not acted on

`cannot_answer→direct` selects τ = 0, meaning its Router fires on every DEV row whose
baseline prediction is `direct`. It satisfies the frozen rule (smallest τ with precision
≥ 0.50) but is effectively ungated, which would matter for collateral exposure and should
weigh in any future advancement decision. Both channels show TRAIN AUC 1.0000, which
reflects 2560-dimensional separability over a few hundred positives; DEV AUC is the
meaningful figure and both clear 0.75 on DEV.

No channel was selected, and no formal primary-channel rule was applied, because no channel
reached the geometry gate. Nothing here should be read as evidence for or against
correctability.
