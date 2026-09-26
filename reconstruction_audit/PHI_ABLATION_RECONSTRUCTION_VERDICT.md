# Phi-3.5 ablation reconstruction — verdict

```
HISTORICAL_RECONSTRUCTION_RUNNER_FAILED_REGRESSION
```

The four missing arms were **not** rerun. The mandatory Phase D regression gate
fails, and it fails on surviving historical evidence — no download, no GPU, and
no new inference was required to establish it.

The frozen five-condition E-2 result is unchanged and not reopened.

---

## 1. Historical-definition recovery

| arm | definition recovered? | source artifact | result on locked 548? |
|---|---|---|---|
| **UNGATED** | **YES — exact** | `sakiko/scripts/phi_placebo/p2_independent_baselines.py` L150-154 | **YES** — 143/175/**−32** |
| **REMOVE_CA_DIRECT** | **YES — exact** | `sakiko/scripts/phi_mainline/run_v31.py` L105-119 (`v31_ablation_2ch`) | no — only n=1096 |
| **REMOVE_RFI_TC** | mechanical only | pattern from `v31_ablation_2ch`; no arm-specific artifact | **no** |
| **REMOVE_CA_TC** | mechanical only | pattern from `v31_ablation_2ch`; no arm-specific artifact | **no** |

### UNGATED is *not* tau=0 — the brief's warning was correct

The historical runner defines it as **threshold zero AND removal of the target
gate**:

```python
cfg0["threshold_rfi"]       = 0.0
cfg0["threshold_ca_tc"]     = 0.0
cfg0["threshold_ca_direct"] = 0.0
# "threshold=0 + no TARGET_GATE; always apply top1 router channel"
```

This is option **B and C combined**, not B alone. The paper must never equate
historical `Ungated` with `tau=0`: the target gate removal is a second,
independent relaxation.

### Removal semantics (recovered, unambiguous)

`v31_ablation_2ch` removes a channel by `inj_layer=None, alpha=0, threshold=1.0`,
and `_train_pipeline` then leaves that router `None`. `cascade_route` sorts
candidates by **probability first** (`key=lambda x: (-x[1], priority[...])`),
with channel priority only as a tiebreak. Therefore removing a channel lets
affected samples **fall through** to the next above-threshold channel; cascade
ordering does not otherwise change. This is mechanical and identical for all
three channels.

**But no `remove_rfi_tc` or `remove_ca_tc` arm was ever run** — no config, no
result, no log, in any of the 169 commits, on any branch, in LFS, or among
deleted objects.

---

## 2. Regression gate — FAILED, and this is the central finding

`p2_independent_baselines.py` imports its entire evaluation path from the frozen
locked runner:

```python
from phi_mainline.run_v31         import SEED, load_data, build_meta, load_model,
                                         collect_activations, cascade_route,
                                         predict_with_correction, compute_metrics
from phi_mainline.p0_run_locked_eval import (_train_pipeline, evaluate_on_split,
                                         extract_metrics, load_locked_config,
                                         build_locked_cfg, load_splits)
```

Same code. Same `SEED`. Same `p0_locked_config.json`. Same split manifest. Same
548-row TEST population. Same `baseline_accuracy = 48.18`.

It reports its own REAL_LOCKED as:

| run | Fixed | Broke | Net |
|---|---:|---:|---:|
| `p2_placebo_controls.json` (**frozen E-2 reference**) | **107** | **52** | **+55** |
| `p2_independent_baselines.json` (same code, same seed) | 106 | 44 | +62 |

**The original authors' own contemporaneous code, run twice under identical
settings, did not reproduce Fixed=107 / Broke=52.** The discrepancy is
concentrated in `Broke` (52 vs 44, a 15% relative swing) — the fragile quantity,
since breaking a baseline-correct row requires a borderline generation flip.
The residual source is non-determinism in hooked `model.generate` reductions,
which no seed controls.

Phase D requires exact reproduction of 107/52 before any missing arm may be run.
That gate is **provably unpassable**: the target was not reproducible in 2026 by
the code that produced it. Per the brief, execution stops here.

### Secondary blocker

`microsoft/Phi-3.5-mini-instruct` weights are **absent from this machine** (no
`models/` entry, no HF cache entry, no safetensors on disk). No model revision
or commit hash was ever recorded in any config or manifest, so even a fresh
download could not be pinned to the historical revision.

---

## 3. Reconstruction results

**None.** No arm was rerun. Nothing was downloaded, tuned, or retried.

---

## 4. Calibration consequence

The expanded evidence **complicates** the original discrimination claim in one
respect and **strengthens** it in another.

**Strengthens.** A surviving locked-test `UNGATED` result exists and is strongly
negative — **Fixed 143 / Broke 175 / Net −32**, against its own within-run
reference of +62. Removing the gate does not merely reduce the benefit; it
**inverts the sign**. Router gating is necessary, not decorative. This is a
legitimate *within-run* contrast: both arms come from one execution of one
script and share one set of trained routers and directions.

**Complicates.** The run-to-run instability above is new information about the
historical Phi pipeline. It does not invalidate E-2 — all five frozen conditions
were computed inside a single run of `p2_placebo_controls.py` and are therefore
internally consistent and comparable to each other. But it means the frozen
point estimates carry an undeclared run-to-run component of roughly ±8 on
`Broke`, and the derived collateral rate 52/93 = 0.5591 should not be presented
as a stable constant.

**The `UNGATED` row must not be inserted into the five-condition E-2 table.**
Its companion reference is 106/44/+62, not 107/52/+55; placing it beside the
frozen rows would compare an arm from run A against controls from run B. It is
reportable only as a separately labelled within-run contrast.

```
E2_REMAINS_REDUCED_FIVE_CONDITION
UNGATED_AVAILABLE_AS_SEPARATE_WITHIN_RUN_CONTRAST
```

---

## 5. Strongest permitted wording

> In the historical Phi-3.5 setting, a surviving locked-test control shows that
> removing the Router gate and target gate — applying the top-1 channel
> correction to every sample — inverts the aggregate effect from +62 to −32 net
> corrections within the same run. Router gating is therefore necessary to the
> observed effect, not incidental to it. This control originates from a separate
> execution whose reference arm differs from the frozen calibration run
> (106/44 vs 107/52), so it is reported as a within-run contrast and is not
> pooled with the five-condition calibration table.

**Forbidden:**
- calling historical `Ungated` a `tau=0` condition
- any nine-condition ladder
- placing `UNGATED` in the same table as the five frozen conditions
- reporting `remove_rfi_tc` or `remove_ca_tc` as executed arms
- presenting 52/93 as a run-stable constant without the instability caveat

---

## 6. Exact next action

```
E3_METATOOL_READJUDICATION
```
