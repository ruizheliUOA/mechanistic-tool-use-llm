# Qwen3 V2 exposure ledger

This ledger continues
`final/results/qwen25_branch_determination/EXPOSURE_LEDGER.md`. It records
behavioural-value exposure relevant to evaluation admissibility for the Qwen3
line.

Structural reads of paths, filenames, file sizes, Git metadata, script source,
serializer definitions, schema field names and headers are permitted and are
not behavioural-value exposures.

## Inherited boundary

The Qwen2.5 ledger records historical **test-split aggregate** exposure on the
Qwen2.5 evaluation population: aggregate counts were observed; no test row, ID,
label, prediction vector, score vector or per-sample result was observed. That
population was already independently classified as used.

The Qwen3 evaluation partition is a distinct 548-row sealed partition. It has
never been opened in the Qwen3 line.

## Qwen3 V1 (protocol V1, completed, immutable)

| UTC timestamp | task | granularity | metric category | cause | admissibility effect |
|---|---|---|---|---|---|
| 2026-07-29 (V1 Block 0 through Block 4) | Qwen3-8B Stage 0/1 V1 | none | none | The selective TRAIN/DEV byte index emitted only authorized non-test offsets; every consumer direct-seeked through it. The evaluation manifest was never opened, no evaluation offset was emitted, and zero non-allowlisted payloads were parsed. | **No new behavioural-value exposure.** The Qwen3 evaluation partition remains sealed and unaccessed. |

V1 mechanical result: `NO_SUPPORT_ELIGIBLE_CHANNELS`. Execution stopped after
the baseline and channel ledger; directions, Routers, geometry and
interventions were never reached, so no evaluation-relevant quantity was
produced.

## Qwen3 V2 (this amendment)

| UTC timestamp | task | granularity | metric category | cause | admissibility effect |
|---|---|---|---|---|---|
| 2026-07-30 Block 0 | V1 archive artifacts | non-test aggregates only | TRAIN/DEV accuracy, macro-F1, confusion matrices, prediction distribution, channel supports | All values were read from committed V1 non-test artifacts (`QWEN3_STAGE1_BASELINE_SUMMARY.json`, `QWEN3_STAGE1_COMPLETE_CHANNEL_LEDGER.csv`) and the committed V1 protocol. | **No test exposure.** Every quantity is derived exclusively from the 3,104 authorized non-test rows. |

| 2026-07-30 Blocks 1-2 | attention and artifact audit | non-test only | attention tensors, activation/baseline schemas | Q/K/V were captured from six authorized TRAIN rows; artifacts inspected were the non-test baseline and activation files. | **No test exposure.** |
| 2026-07-30 Blocks 3-4 | V2 freeze and allocation | non-test aggregates only | gold-class totals, correct-reference totals, support counts | The split rule and sizing justification used only gold labels, immutable UUIDs and aggregate non-test totals. Support counts were recomputed from the existing non-test baseline artifact. | **No test exposure.** The sealed partition was inherited by reference; its manifest was not opened and its hash was not recomputed. |
| 2026-07-30 Blocks 5-6 | directions, Routers, dev geometry | non-test only | activations, gradients, Router metrics, geometry | All estimation used V2 TRAIN rows; all geometry used V2 DEV channel-error rows. Both partitions are subsets of the 3,104 authorized non-test rows. | **No test exposure.** |
| 2026-07-30 same-layer L21 control | post-hoc layer-transposition control | non-test only | L21 activations, persisted DEV gradients, Qwen2.5 layer configuration | CPU-only reuse of stored V1 activations and persisted V2 DEV gradients; Qwen2.5 audit read committed configuration artifacts only, with no model load and no gradient recomputation. | **No test exposure.** No GPU, no model load, no forward or backward pass. |
| 2026-07-30 layer-control consolidation | hook repair, sign-orientation audit, Gram decomposition, Qwen2.5 layer resolution | non-test only | L21/L26 component means, persisted DEV gradients, archived Qwen2.5 direction vectors and 7B activation caches | CPU-only. Qwen2.5 `ca_direct` DiffMean was recomputed from the archived 3584-dim L16 and L20 activation caches to resolve a layer discrepancy; those caches and the Qwen2.5 baseline cover the full 3652-row dataset, which for Qwen2.5 is a historical population already classified as used. No Qwen3 evaluation row was touched. | **No Qwen3 test exposure.** The Qwen2.5 recomputation is on the historical Qwen2.5 population, whose aggregate status was already recorded in the inherited boundary; it produces a direction vector, not an evaluation metric. |
| 2026-07-31 Stage 1.5 gradient-mean | TRAIN-only estimator and prospective DEV transfer | non-test only | TRAIN L21 gradients, persisted DEV L21 gradients | GPU computed gradients for 1031 frozen V2 TRAIN error rows; the DEV stage was CPU-only and reused the persisted V2 DEV gradients with no new forward. | **No test exposure.** The sealed 548-row partition was never opened, enumerated, hashed or scored. |
| 2026-07-31 Stage 2A dose declaration | prospective TRAIN-only dose-unit definition | non-test only | TRAIN L21 activation norms, committed Router tensors | CPU-only. The scaling population was reconstructed from the committed Routers and baseline predictions on TRAIN rows; s_c is the median L21 activation norm over that population. No DEV behavioural intervention was run and no DEV outcome was consulted. | **No test exposure.** No GPU, no model load, no forward or backward pass, no intervention. |
| 2026-08-03 Stage 2A V2 DEV calibration | DEV-only behavioural intervention, controls, dose gate | non-test only | intervened four-mode scores and destination flows on DEV Router-gated rows | GPU applied the frozen dose rule at L21 to DEV rows selected by the committed Router, re-scored the four modes, and resolved destinations; 14293 per-sample records across dose and control arms. | **No test exposure.** The sealed 548-row partition was never opened; only its committed structural count was used for a cost projection. |
| 2026-08-03 Stage 2 formal freeze | preregistration, random null, preflight, DEV engineering check, gate-only | none | none | All verification was structural or on authorized DEV rows. The formal random vectors are synthetic. The formal execution body is absent from the runner and --run-formal refuses without an external approval token. | **No test exposure.** The evaluation payload was not opened, no evaluation ID was enumerated and no evaluation prediction exists; only the committed structural count 548 was referenced. |
| 2026-08-03 Stage 2 formal implementation | execution body, versioned freezes V2-V4, preflight, DEV check, gate-only | none | none | All verification was structural or on authorized DEV rows. The evaluation-access class was never constructed and the access marker was never written. | **No test exposure.** The evaluation payload was not opened, no evaluation ID was enumerated and no evaluation prediction exists. |

### Sealed-partition handling in V2

The 548-row Qwen3 evaluation partition is inherited **by reference** from the
frozen V1 manifest. In this task it is not read, materialized, re-hashed,
enumerated, scored, counted or inspected. Its identity is carried forward as a
committed hash only, and its hash is not recomputed from payload.

### Closing statement for V2

New accidental behavioural-value exposures in the Qwen3 line: **none**.

No Qwen3 test row, ID, label, prediction, score, support count, aggregate or
channel membership was accessed at any point in V1 or V2. The sealed 548-row
evaluation partition was never opened, never enumerated, never re-hashed from
payload and never scored. Its identity was carried forward only as a committed
hash inherited from the frozen V1 manifest.

Stage 2 was not authorized, not prepared and not run. No Stage 2
preregistration, runner, execution freeze, preflight or test prediction exists.

Any accidental exposure would be recorded honestly here and never concealed or
erased. None occurred.
