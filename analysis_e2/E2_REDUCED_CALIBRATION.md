# E-2 — Reduced five-condition historical calibration (Phi-3.5)

```
VERDICT: E2_REDUCED_CALIBRATION_SUPPORTS_DISCRIMINATION
```

**POST-HOC / HISTORICAL.** Not prospective confirmation. Phi-3.5, seed 42.

## Task 1 — regression gate: PASS

Recomputed from `final/results/clean/p0_final_test_details.jsonl` (548 rows,
recovered via Git LFS from `exp/sakiko-followup-archive`):

| quantity | recomputed | archived |
|---|---:|---:|
| n | 548 | 548 |
| baseline-correct | 264 | 264 |
| Fixed | 107 | 107 |
| Broke | 52 | 52 |
| Net | +55 | +55 |
| Router-fired baseline-correct | **93** | 93 |
| broken among exposed | 52 | 52 |

**Firing rule recovered:** `route != 'none'`, equivalently `route_prob >= 0.4`
(seed-42 locked threshold). Both give the identical 293 rows — 200 baseline
errors + 93 baseline-correct. `route` alone is the channel *assignment* and is
populated for all 548 rows; treating it as the firing decision yields 264 and
fails the gate.

Seed 42 confirmed against `p1_multiseed_table.csv` (107 / 52 / +55).

## Task 2 — REAL_LOCKED destination audit

**Denominator = 200 routed baseline errors.**

| outcome | n |
|---|---:|
| SOURCE_RETAINED | 47 |
| **GOLD_ARRIVAL** | **107** |
| OTHER_WRONG | 46 |
| source exits | 153 |

```
target-hit      = 107/153 = 0.6993
wrong-to-wrong  =  46/153 = 0.3007
```

**Preservation, denominator = 93 Router-fired baseline-correct:**

```
BROKEN 52/93 = 0.5591     <- exposed-correct collateral
(all-baseline-correct 52/264 = 0.1970 is the diluted figure; NOT the estimand)
```

Per-channel, row-level:

| channel | errors | exits | gold | other | target-hit | exposed-corr | broken |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rfi_tc` | 87 | 82 | 59 | 23 | **0.720** | 50 | 37 |
| `ca_tc` | 59 | 35 | 18 | 17 | **0.514** | 24 | 7 |
| `ca_direct` | 54 | 36 | 30 | 6 | **0.833** | 19 | 8 |

## Task 3/5 — five-condition comparison

Source: `final/results/clean/p2_placebo_controls.json`.

| condition | structural validity | Fixed | Broke | Net | F/B | destination audit | preservation audit |
|---|---|---:|---:|---:|---:|---|---|
| **REAL_LOCKED** | intended | **107** | **52** | **+55** | **2.06** | t-hit 0.699, 107G/46O/47S | 52/93 = 0.5591 |
| RANDOM_DIR | direction destroyed | 71 | 57 | +14 | 1.25 | N/A (historical aggregate only) | N/A |
| REVERSE_DIR | sign reversed | 71 | 57 | +14 | 1.25 | N/A (historical aggregate only) | N/A |
| WRONG_LAYER | site disrupted | 66 | 63 | +3 | 1.05 | N/A (historical aggregate only) | N/A |
| MISMATCHED_CHANNEL | channel mismatched | 38 | 55 | **-17** | 0.69 | N/A (historical aggregate only) | N/A |

**REAL ranks 1 of 5 on every comparable metric; all 4 placebos are worse on all 4.**

`n_routed` is identical across arms (137 / 83 / 73), so routing was held fixed and
only intervention structure was corrupted — a like-for-like comparison.

No ordinal ranking is asserted among the placebos: RANDOM_DIR and REVERSE_DIR are
identical on every aggregate (71/57/+14) and differ only in per-channel precision.

## Tasks 6-7 — what this does and does not establish

**Claim E2-A — intervention discrimination: SUPPORTED.**

> The real matched SAKIKO intervention achieves stronger aggregate correction
> evidence than every control that disrupts its direction, layer, or channel
> structure, while its row-level record additionally reveals destination and
> preservation properties that aggregate metrics hide.

**Claim E2-B — full licensing calibration: `NOT_IDENTIFIABLE_FROM_SURVIVING_HISTORY`.**

Four placebo arms have no row-level `int_pred`. Target-hit, destination splits and
exposed-correct collateral cannot be computed for them and **must never be
reconstructed from aggregate Fixed/Broke/Net**.

## The finding aggregate metrics conceal

Net +55 reads as a clear success. The row-level audit shows the same run
**breaking 52 of 93 exposed-correct rows — 55.9%**, against a 5% modern budget.
And `ca_tc` reaches target-hit **0.514**, barely above chance redistribution.

That is the paper's thesis demonstrating itself on historical data: the licence
and the aggregate metric disagree, and the disagreement is visible only at
destination resolution.

## Prohibited wordings

- "the full licensing score monotonically ranks all five interventions"
- any target-hit, GOLD_ARRIVAL, OTHER_WRONG or row-level collateral for a placebo arm
- 52/264 as the collateral rate (that is the diluted all-correct denominator)
- a nine-condition ladder; Ungated and the three `remove_*` arms are unrecoverable
- treating five non-independent arms as independent samples
