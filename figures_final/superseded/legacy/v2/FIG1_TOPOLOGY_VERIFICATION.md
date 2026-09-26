# FIG1 topology verification

Verified by reading committed source, not summaries. All line numbers are in the
`sakiko-paper` branch working tree.

## VERIFIED OBSERVATION SITE

- **Layer index:** `L_OBS = 26` — `scripts/qwen3_8b_stage0_1.py:60`
- **Module:** `model.model.layers[L_OBS].mlp` — `scripts/qwen3_8b_stage0_1.py:1332`
- **Hook type:** `register_forward_hook`, output capture only (no gradient, no write)
- **Tensor position:** final prompt token; recorded in the frozen router spec as
  `"feature": "L{L_OBS} MLP output final prompt token"` — `qwen3_8b_stage0_1.py:1055`

## VERIFIED INJECTION SITE

- **Layer index:** `L_INJ = 21` — `scripts/qwen3_8b_stage0_1.py:61`
- **Module:** `model.model.layers[L_INJ].mlp` — `qwen3_8b_stage0_1.py:1068`,
  and at formal time via `make_hook(model, layer, delta)` with
  `layer = fz.L_INJ` — `qwen3_stage2_formal.py:296`
- **Operation:** constant additive delta on the MLP output

## LAYER NUMBERING CONVENTION

**Zero-based.** Both sites index the HuggingFace `model.model.layers` ModuleList
directly (`layers[21]`, `layers[26]`), which is zero-based. `NEW_NUM_LAYERS = 36`,
so valid indices are 0–35 and both sites are interior.

The normalized-depth mapping asserted at `qwen3_8b_stage0_1.py:490-493` is
consistent with zero-based indexing on both ends:
`round(20 / (28-1) * (36-1)) = 26` and `round(16 / (28-1) * (36-1)) = 21`.

## EXECUTION ORDER

Layers execute in increasing index order, so within a single forward pass:

```
... -> layers[21] (L_inj) -> ... -> layers[26] (L_obs) -> ...
```

**The injection site executes BEFORE the observation site.** A same-pass
`observe -> route -> inject` pipeline is therefore impossible, exactly as the
brief suspected.

## ONE-PASS or TWO-PASS

**TWO-PASS. Confirmed.**

1. **Pass 1 — baseline / capture.** `qwen3_8b_stage0_1.py:1331-1335` registers
   capture hooks on *both* `layers[26].mlp` and `layers[21].mlp` and runs the
   model over TRAIN and DEV with no intervention. `h_obs` is written to
   `activations_L26` and persisted.
2. **Router scoring — offline, on cached activations.** `router_prob()` takes
   `A26_rows: np.ndarray` and applies a stored scaler and logistic weights in
   NumPy — `qwen3_stage2_formal_v2.py:227-229`. **No model forward is involved.**
   The router never runs inside the transformer graph.
3. **Pass 2 — intervention.** `score_with_intervention()`
   (`qwen3_stage2a_v2_dev_calibration.py:204`) attaches `make_hook(model, layer,
   delta)` at `L_INJ` and re-runs a fresh forward for each of the four modes,
   then removes the handle.

Activations are cached in pass 1; the router decision is computed per-sample
between passes; no activation is carried live from pass 1 into pass 2 other than
through the router's scalar probability.

## SOURCE FILES / FUNCTIONS

| Role | File | Symbol |
|---|---|---|
| Layer constants | `scripts/qwen3_8b_stage0_1.py` | `L_OBS`, `L_INJ` (lines 60–61) |
| Capture pass | `scripts/qwen3_8b_stage0_1.py` | `modules={...}` (line 1332) |
| Router weights + scoring | `scripts/qwen3_stage2_formal_v2.py` | `router_prob` (line 227) |
| Delta construction | `scripts/qwen3_stage2_formal_v2.py` | `build_delta` (line 233) |
| Intervention forward | `scripts/qwen3_stage2a_v2_dev_calibration.py` | `score_with_intervention` (line 204) |
| Formal arm driver | `scripts/qwen3_stage2_formal.py` | `run_formal` (line 360) |

## MANUSCRIPT WORDING THAT MUST BE CORRECTED

1. **Nothing in the manuscript is reversed.** `L_obs = 26` and `L_inj = 21` are
   reported with the correct values. The defect is one of *omission*, not
   inversion.
2. **The two-pass structure is never stated.** Methods describes observation and
   injection without saying that injection precedes observation in execution
   order and that the procedure therefore requires two forward passes. A reader
   who assumes a single pass will construct an impossible mental model — and a
   reviewer who notices `21 < 26` will read it as an error in the work rather
   than an omission in the writing.
3. **Add explicitly:** that the router operates on cached activations in NumPy
   and is not part of the model graph. This matters for the cost claim: SAKIKO's
   gate is not a second network evaluated online.
4. **Rejected figure v1 drew a one-pass topology.** It is archived in
   `figures/archive_rejected/` and must not be reinstated.

## CONSEQUENCE FOR FIGURE 1

The figure draws two labelled passes over the same frozen stack, with the router
sitting *between* them on cached `h_obs` rather than inline. This is the verified
topology and is not negotiable for visual convenience.
