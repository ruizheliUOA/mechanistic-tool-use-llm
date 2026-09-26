# Formal access policy

## One-shot access ledger

```
formal_run_count:            0
scientific_endpoint_observed: false
access_marker_created:        false
sealed_rows_read:             0
```

## Status

**SEALED ACCESS NOT AUTHORISED.** No access marker exists. No SEALED label, prompt, output,
score or count has been viewed at any point in this study.

## Procedure, when and only when a human authorises

1. Generate the K=59 fresh matched-random directions with seeds disjoint from the DEV seeds
   `20260811-20260818`; write the matrix sha256 into `FORMAL_RANDOM_NULL.json`.
2. Commit the complete freeze. Replay the gate on the resulting HEAD.
3. Create the access marker.
4. Execute exactly one formal run: 65 arms (zero, real, reverse, wrong-layer, DiffMean,
   ungated, score-space, and 59 randoms) over the 548 sealed rows, in a single invocation.
5. Any incomplete arm execution, automatic retry or resume is a mechanical VOID, never a
   scientific result.

## Prohibitions

No endpoint may be inspected before the complete battery is authorised and executed. A
scientific failure may never be converted to VOID. The dose, direction, Router, tau, channel
and thresholds are frozen by `FORMAL_CONFIG.json` and may not be changed after any SEALED read.
