# V2 artifact reusability audit

CPU only. Files were inspected directly; nothing is inferred from file size.

## 2.1 Activation tensor inventory

`final/results/qwen3_stage0_1/QWEN3_STAGE1_ACTIVATIONS.safetensors`
SHA256 `0c0ef438e6f3c4837f28a8ec6540d5566d4111f76bb943cde5406e0b1c709efc`

| key | shape | dtype | layer | residual-stream site | token position |
|---|---|---|---:|---|---|
| `activations_L21` | [3104, 4096] | F32 | 21 | mlp_output_last_prompt_token | last prompt token |
| `activations_L26` | [3104, 4096] | F32 | 26 | mlp_output_last_prompt_token | last prompt token |

- storage schema: **one vector per authorized row** — not per row and position,
  not per row and candidate. The capture hook recorded the activation at the
  final prompt token during the first (`tool_call`) candidate forward only, so
  the stored feature is candidate-independent by construction.
- row count: **3104**
- unique project indices: **3104**
- unique immutable UUIDs: **3104**
- row ordering: baseline row order: TRAIN by ascending project index, then DEV by ascending project index
- UUID mapping: `metadata sample_order_json, index-aligned with tensor rows`
- V1 split composition of the stored rows: {'train': 2556, 'dev': 548}
- **all 3,104 authorized non-test rows represented: True**

The V1 split labels are stored only as metadata. The tensors themselves are a
per-row feature matrix and carry no split dependence, so a new allocation of the
same rows reuses them without recomputation.

## 2.2 Frozen layer comparison

Read from the committed V1 protocol `layer_mapping`:

- `L_obs` = **26**, `L_inj` = **21**, zero-based = True, sweep = False
- site = `model.model.layers[L].mlp forward output, after MLP internals and before decoder residual addition`
- token positions = `last prompt token for feature/direction; all positions for full-effect gradient`

Stored metadata reports `layers = [26, 21]` and
`site = mlp_output_last_prompt_token`.

- `activations_L26` present: **True**
- `activations_L21` present: **True**

Layer identity is taken from the safetensors metadata and the committed
protocol, not from the filename.

## 2.3 Baseline artifact inventory

`final/results/qwen3_stage0_1/QWEN3_STAGE1_BASELINE_ROWS.jsonl`
SHA256 `83c0836f4fd1b52df7d44e81e44254a4e3599bad97eb6239e8744571729a02c3`

- rows: **3104**, unique UUIDs: **3104**
- every row carries immutable ID, gold label, four candidate scores, predicted
  mode, all 12 ordered margins, runner-up, four candidate lengths, prompt-token
  count and ordering metadata: **all present, 0 rows missing any field**
- four finite scores on every row: **yes**, non-finite scores: **0**
- prediction and runner-up disagreeing with the score vector: **0 rows**
- 1:1 index alignment with the activation rows: **verified on all 3,104 rows**

### Split independence

- **Structural:** `score_sample(...)` receives only the sample. The split is a
  loop label attached to the output record after scoring. No branch in the
  scoring path reads it.
- **Empirical:** the fixed pilot scored 8 rows twice *before* the baseline loop,
  in a separate execution context. Those 64 score comparisons against the final
  baseline rows agree with **maximum deviation 0.0**, and predictions and
  runner-ups are identical.

Baseline values are therefore a pure function of the sample and do not depend
on whether a row was assigned to V1 train or V1 dev. Reallocating the same rows
cannot change any score, prediction, runner-up or margin.

## 2.4 Outcome

**CPU_SUFFICIENT.**

- non-test baseline coverage is complete (3,104 of 3,104);
- row identity and ordering are exact and verified;
- the captured features are sufficient for the frozen direction estimator and
  the frozen Router specification at both required sites, `L_obs = 26`
  and `L_inj = 21`.

Direction estimation and Router fitting for V2 therefore require **no new
forward pass**.

This outcome concerns direction and Router estimation only. The frozen dev
geometry is defined as candidate-conditioned **gradients** at `L_inj`, which are
not activations and were never stored by V1; computing them necessarily requires
a GPU pass in Block 5 regardless of this outcome.
