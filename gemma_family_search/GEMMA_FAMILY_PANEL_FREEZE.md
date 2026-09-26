# Gemma family panel freeze

**GEMMA_SEARCH_STATUS: `PANEL_FROZEN`.** GPU-hours: 0.0. Cells run: 0. None hidden.

## Corrected infrastructure state

An earlier turn of this session reported the Gemma weights absent and the
programme disk-blocked. **Both claims were wrong.** They came from inspecting
`hf_cache/models--google--gemma-2-9b-it`, which is an empty 217-byte husk, and
not looking further.

The weights are present and complete:

```
/root/autodl-tmp/models/gemma-2-9b-it-11c9b309   18 GB
model-0000{1,2,3,4}-of-00004.safetensors         4/4 present
GEMMA_LOCAL_SHA256SUMS.txt                       local integrity manifest
```

Revision `11c9b309` matches the pinned C0 revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819`. **No download is required and there
is no disk blocker.** 24.5 GB VRAM is entirely free.

## Architecture correction — load-bearing for Figure 1

`config.json` reports **`num_hidden_layers: 42`**, hidden size 3584.

The Figure-1 handoff brief instructs recording a "36-layer model". That is wrong
for Gemma-2-9b-it. Under 42 layers the C0 sites sit at:

| site | layer | normalized depth |
|---|---|---|
| `L_obs` | 30 | 0.732 |
| `L_est` = `L_inj` | 24 | 0.585 |

`L_inj < L_obs` still holds, so the **two-pass topology lock is unaffected** —
injection executes before observation, and a single continuous forward pass
remains impossible. Only the denominator changes. Any topology lock must record
42, not 36.

## Frozen panel

Executable today: **1 model.**

1. `google/gemma-2-9b-it` @ `11c9b309…` — local, complete, 42L, the C0 model.

A second member (`gemma-2-2b-it`, ~5.2 GB) fits in the 16 GB free and is the only
realistic addition. It is **not** frozen into the panel here, because adding it
after the 9B outcomes are seen would violate the §17 stop rule. If a scale
comparison is wanted it must be declared and downloaded **before** G2 runs.

`gemma-2-27b-it` is permanently ineligible (~54 GB bf16 vs 24 GB VRAM;
quantisation would change the readout). Gemma-3 is multimodal with an unverified
text-only path against the frozen 4-mode scored readout.

With a one-model panel, §10 cross-model scale analysis returns
**`INSUFFICIENT_EVIDENCE` by construction.** Stated now, before any GPU time.

## Existing machinery — the factorial should reuse, not rebuild

The complete C0 pipeline is already committed at
`research_exploration/sakiko_final_model_breadth_panel_v1/gemma/`:
`run_baseline.py`, `run_router.py`, `run_directions.py`, `run_dev_ladder.py`,
`run_controls.py`, plus frozen `BASELINE_ROWS.jsonl`,
`BASELINE_LOBS_ACTIVATIONS.npz`, `CHANNEL_FREEZE.json`, `DIRECTION_RECORDS.jsonl`
and `DEV_INTERVENTION_RECORDS.jsonl`.

Two consequences for the §5 factorial:

- **`Q_GRID` is frozen and explicitly marked not extendable**
  (`run_dev_ladder.py:2,13`). The dose axis {0, 0.125, 0.25, 0.5, 1.0, 2.0} is
  already fully traversed and verified. The dose factorial is **complete** — no
  GPU needed.
- `run_dev_ladder.py` already computes a **DiffMean direction at `L_inj`**
  (`diffmean_Linj_sha`). Part of the estimator axis therefore already exists in
  frozen DEV artifacts and should be read out before any re-run.

The genuinely new GPU work is the **write-site axis** (`L_inj` ∈ bounded grid)
crossed with estimator. Sites are hardcoded in the runner, so this needs a
parameterised variant — which must be written as a *new* script, leaving the
frozen runner untouched.

## Result already secured without GPU

`gemma_design_mismatch/GEMMA_DESIGN_MISMATCH_VERDICT.md` →
**`GEMMA_DESIGN_MISMATCH_SUPPORTED`**, localised to the dose rule
`q = min(admissible)`. The user-supplied trajectory was verified against
`DOSE_SELECTION.json` and matches exactly at all five doses: target-hit rises
monotonically 0.667 → 0.842 while `min(admissible)` selected the worst cell.

This stands independently of whether the family search runs.

## Next executable step

Read the estimator evidence already frozen in `DEV_INTERVENTION_RECORDS.jsonl`
and `DIRECTION_RECORDS.jsonl` before spending GPU. Only the write-site axis
requires new computation.
