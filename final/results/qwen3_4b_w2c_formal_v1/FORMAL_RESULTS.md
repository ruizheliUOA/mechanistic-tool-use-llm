# Qwen3-4B × When2Call — one-shot formal SAKIKO result

**Channel `cannot_answer → tool_call`. One authorised SEALED run. Result final.**

## Principal verdict

## `QWEN3_4B_FORMAL_DECLINE`

The complete preregistered ten-condition conjunction did **not** pass. Two conditions
failed, both confidence-interval conditions. This is a scientific outcome and it consumes
the experiment.

## SEALED baseline

548 evaluation rows, scored once before any arm. Channel-error support **124**
(gold `cannot_answer`, baseline prediction `tool_call`); 118 of those Router-fired.
Baseline-correct rows 214, of which 50 were Router-fired and therefore actually exposed to
the gated intervention.

## Full formal conjunction

| # | Frozen rule | Observed | PASS/FAIL |
|---|---|---|---|
| 1 | channel-error support ≥ 30 | 124 | **PASS** |
| 2 | real `target_gain_rate` > 0 | 0.0806 | **PASS** |
| 3 | bootstrap 95 % **lower** bound for `target_gain_rate` > 0 | **−0.0484** | **FAIL** |
| 4 | `p_add_one` ≤ 0.05 vs K=59 randoms | **0.016667** | **PASS** |
| 5 | real `target_hit` > 0.50 | 0.5781 | **PASS** |
| 6 | bootstrap 95 % **lower** bound for `target_hit` > 0.50 | **0.4559** | **FAIL** |
| 7 | `clean_collateral_rate` ≤ 0.05 | 0.004673 | **PASS** |
| 8 | bootstrap 95 % **upper** bound for collateral ≤ 0.05 | 0.014019 | **PASS** |
| 9 | zero-control equality exact | exact on all 218 routed rows | **PASS** |
| 10 | no structural or provenance failure | none | **PASS** |

8 of 10 pass. The conjunction is an intersection claim with no subset substitution, so the
result is DECLINE. There is no partial credit.

## Where the evidence ladder broke

**Statistical precision on the destination endpoints — not direction specificity.**

The intervention moved behaviour, moved it specifically, and moved it preferentially
toward gold. What it did not do is move it *decisively enough on 124 SEALED errors* for the
interval requirements to hold. Both failures are lower-bound failures with passing point
estimates:

- Target Gain 0.0806, 95 % CI [−0.0484, +0.2016] — spans zero
- target-hit 0.5781, 95 % CI [0.4559, 0.6970] — spans 0.50

## Formal activation result and destination flow

Real `d_grad`, q = 0.25, ‖Δ‖ = 6.1748905405, at L21, Router-gated:

| destination | n |
|---|---:|
| `SOURCE_RETAINED` | 60 |
| `GOLD_ARRIVAL` | **37** |
| `OTHER_WRONG` | 27 |
| source exits | 64 |

target-hit 0.5781 · Target Gain count 10 · Target Gain rate 0.0806 · Fixed 40 · Broke 1 ·
Net +39.

## Fresh matched-random specificity — the strongest part of the result

| | |
|---|---|
| K | 59 |
| real `target_gain_rate` | **0.0806** |
| random min / median / max | −0.0484 / −0.0161 / **0.0000** |
| randoms ≥ real | **0** |
| real rank | **1 of 60** |
| `p_add_one` | **0.016667** |

Not one of the 59 fresh matched-random directions, at identical perturbation norm and
Router gate, reached the real direction. The effect is direction-specific.

## Controls

| arm | exits | gold | other-wrong | target-hit | Target Gain | collateral |
|---|---:|---:|---:|---:|---:|---:|
| zero | 0 | 0 | 0 | — | 0.0000 | 0.0000 |
| **real `d_grad`** | 64 | 37 | 27 | 0.5781 | **+0.0806** | 0.0047 |
| reverse | **0** | 0 | 0 | — | 0.0000 | 0.0000 |
| wrong-layer (L26) | 15 | 6 | 9 | 0.4000 | −0.0242 | 0.0000 |
| L26 DiffMean | 4 | 1 | 3 | 0.2500 | −0.0161 | 0.0000 |
| ungated | 65 | 37 | 28 | 0.5692 | +0.0726 | 0.0280 |
| score-space | 75 | 44 | 31 | 0.5867 | **+0.1048** | 0.0280 |

- **Zero**: exact baseline reproduction on every routed row. Integrity intact.
- **Reverse**: zero source exits. The reversed direction does nothing at all; it does not
  reproduce the effect and must not be read as bidirectional controllability.
- **Wrong-layer**: 15 exits but negative Target Gain — the same direction at L26 breaks the
  effect. Site-specific.
- **DiffMean**: near-inert, 4 exits, negative Target Gain. Estimator dependence on this
  channel at this dose; **no universal claim about DiffMean follows.**
- **Ungated**: nearly identical repair (37 gold) at 6× the collateral (6 vs 1 events;
  eligible-at-risk 186 vs 50). Router gating limits unintended intervention chiefly by
  reducing exposure, not by changing repair quality.

## Score-space comparator

**The frozen score-space comparator outperformed the activation intervention.** Target Gain
0.1048 vs 0.0806; gold arrivals 44 vs 37; exits 75 vs 64; wrong-to-wrong 31 vs 27; collateral
6 vs 1 events.

`b_c = 1.4013434649` was frozen on DEV and was not tuned on SEALED.

Per the frozen interpretation rule this is binding: **do not claim the activation
intervention provides behaviour unavailable to a simple frozen two-mode score shift.** On
this channel, at this dose, on this population, it does not. The frozen conjunction never
required activation superiority, so this did not by itself cause the DECLINE — but it
materially constrains what may be claimed.

## Collateral — both denominators, claim lock carried

| metric | value |
|---|---|
| **frozen formal endpoint** (all baseline-correct) | **1 / 214 = 0.004673**, 95 % CI [0.0000, 0.014019] — **PASS** |
| **eligible-at-risk diagnostic** (Router-fired ∩ baseline-correct) | **1 / 50 = 0.0200** |
| rows in the frozen denominator never perturbed | 164 of 214 (76.6 %) |
| collateral destinations | CORRECT_RETAINED 213 · BROKEN_TO_SOURCE 0 · BROKEN_TO_OTHER 1 |

> **Passing the frozen formal collateral endpoint must not be interpreted as demonstrated
> deployment safety.**

Prohibited and not claimed: safe deployment, proven low-risk intervention, zero-risk,
deployment-safe.

## Cross-model interpretation

Modern-protocol evidence now reads:

- Qwen3-8B × When2Call × `cannot_answer→tool_call` → **FORMAL ADMIT**
- Qwen3-4B × When2Call × `cannot_answer→tool_call` → **FORMAL DECLINE** (this result)
- Qwen3-4B × `cannot_answer→direct` → DEV DECLINE on collateral qualification

The prospective cross-model replication **did not** replicate. What Qwen3-4B adds is
narrower and still real: on a previously untouched evaluation population, a
development-selected frozen direction produced direction-specific, destination-preferential
movement (0 of 59 randoms reached it) that nonetheless failed the frozen precision
requirements for correctability.

This is precisely the case SAKIKO's ladder exists to separate. Readable, steerable, and
destination-preferential is **not** the same as correctable, and DEV success did not imply
SEALED success. The protocol did its job.

## Scientific limitations

- **Same benchmark.** When2Call only; `DATASET_KNOWN / MODEL_UNTOUCHED`. Not an
  independent-dataset replication and must never be described as one.
- **No dataset-generalisation claim.**
- **No scaling-law claim.** One 4B result against one 8B result on one channel is not a
  size trend, and the two were not tested for a shared causal mechanism.
- **No universal activation superiority.** Here the score-space comparator was stronger.
- **No deployment-safety claim.** See the collateral claim lock.
- **No generalisation** from one channel to all Qwen3-4B channels, all models, or all
  tool-use tasks.
- The channel, estimator and dose were selected on TRAIN/DEV evidence while the evaluation
  partition was sealed — a disclosed, not an untouched, choice.

## Execution integrity

One authorised run, invocation `e3d9d0b5-92f9-42da-86c2-d8ca2606466c`, access marker
fsynced before the first SEALED load. 14,404 records across 65 arms, 0 duplicate samples
within an arm, 0 non-finite scores, all 59 randoms present and no additional random,
destination categories partition every population exactly, zero-control exact.

A prior attempt was terminated by host teardown mid-random-battery and was declared
`QWEN3_4B_FORMAL_VOID` / `INCOMPLETE_ALL_ARM_EXECUTION` before any endpoint existed; it was
not resumed, and every arm was re-executed from scratch under explicit human authorisation.
