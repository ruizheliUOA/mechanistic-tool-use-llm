# Phi-3.5 historical artifact audit

**Analysis class: `RETROSPECTIVE_HISTORICAL_RE-ADJUDICATION`.** Not preregistered, not
prospective, not confirmatory, not sealed, not an independent replication.

## Identity, as recorded

| item | recorded value | source |
|---|---|---|
| model | Phi-3.5-mini-instruct, 3.8B, 32 layers, hidden 3072, fp16 | `FINAL_RESULTS_SUMMARY.md` |
| checkpoint revision | **not recorded** | — |
| dataset | `trace_db/w2c_phi35.jsonl`, 3,652 rows, When2Call 4-candidate avg-logp ranking | `FINAL_RESULTS_SUMMARY.md` |
| split | seed 42, 70/15/15 → train 2,556 / val 548 / **test 548** | `p0_split_info.json` |
| config | val-selected: τ = 0.4 all three channels; α = 5.0 / 2.0 / 10.0; obs L18; inj L14 / L16 / L16 | `p0_locked_config.json` |
| headline | acc 48.18 → 58.21 (+10.04pp), **Fixed 107, Broke 52, Net +55** | `p0_final_test_eval.json` |
| channels | `rfi→tc` (112), `ca→tc` (77), `ca→direct` (59) | `p0_final_test_eval.json` |

## Reconciliation of the per-sample file against committed aggregates

`p0_final_test_details.jsonl` (548 rows; fields `idx`, `gold`, `clean_pred`, `int_pred`,
`route`, `route_prob`, `error_type`) reproduces **every** committed aggregate exactly:

| quantity | recomputed | committed | match |
|---|---:|---:|:-:|
| Fixed | 107 | 107 | ✓ |
| Broke | 52 | 52 | ✓ |
| Net | +55 | +55 | ✓ |
| clean_damage n | 264 | 264 | ✓ |
| clean_damage damaged | 52 | 52 | ✓ |
| routed per channel | 137 / 83 / 73 | 137 / 83 / 73 | ✓ |

The artifact is authoritative and the reconstruction below is exact, not approximate.

## Gating integrity

Of the 255 rows with `route == "none"`, **zero** changed prediction. The intervention is
genuinely gated, and the unrouted population functions as an implicit zero control.

## The comparability limit that must be disclosed

Phi ran **three routers concurrently**, and they compete for the same rows. Route assignment
within each channel-error population:

| channel-error population | own router | other router | none |
|---|---:|---:|---:|
| `rfi→tc` (112) | 66 | 2 (`ca_tc`) | 44 |
| `ca→tc` (77) | 45 | 23 (`ca_direct` 18, `rfi_tc` 5) | 9 |
| `ca→direct` (59) | 32 | 12 (`ca_tc` 8, `rfi_tc` 4) | 15 |

So 18 of 77 `ca→tc` error rows were edited by the `ca→direct` direction. The modern protocol
is one channel, one Router, one direction. **Per-channel destination numbers are therefore
`NOT_COMPARABLE` to a modern single-channel measurement** and are reported as
cascade-level observations.

**The collateral finding is unaffected by this**, because its numerator and denominator are
defined over baseline-correct rows and their exposure, not over channel identity.
