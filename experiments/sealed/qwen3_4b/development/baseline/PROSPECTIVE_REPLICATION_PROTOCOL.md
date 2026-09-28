# Prospective Cross-Model Replication Protocol — Qwen3-4B × When2Call

## Object

Whether the full modern SAKIKO protocol can prospectively identify and formally ADMIT at
least one genuinely correctable error channel in a second modern host model. The host model
changes from Qwen3-8B to Qwen3-4B; the measurement environment does not.

`DATASET_KNOWN / MODEL_UNTOUCHED`. This is **not** an independent-dataset replication and
must never be described as one.

## Frozen inheritance (resolved from committed artifacts, not from the task prompt)

- **Dataset** `nvidia/When2Call` @ `0582f7749df63a96fdc3070932e83e72396ace53`, config `test`,
  split `mcq`, 3,652 rows; raw payload sha256 verified against the committed manifest.
- **Splits** `numpy.random.default_rng(42).permutation(3652)`, `project_index i → raw[perm[i]]`,
  verified 3104/3104 on uuid and gold. TRAIN 2000 / DEV 1104 / SEALED 548.
- **Readout** the committed `qwen3_8b_stage0_1.py` functions reused verbatim: four
  teacher-forced forwards per sample, mean log-probability over complete candidate tokens,
  official Qwen3 template with `enable_thinking=False`, same determinism block and
  long-sequence attention guard. Not redesigned.
- **Ontology** `tool_call`, `direct`, `request_for_info`, `cannot_answer`.
- **Support gate** TRAIN ≥60 errors and ≥60 correct reference; DEV ≥30 and ≥30.
- **Router** L_obs MLP output at the final prompt token, StandardScaler,
  `LogisticRegression(C=1.0, L2, liblinear, max_iter=2000, tol=1e-4, random_state=42)`,
  fit on TRAIN only, positives = channel TRAIN errors, negatives = all correct TRAIN rows;
  eligibility DEV ROC-AUC ≥ 0.75 and selected-τ precision ≥ 0.50; τ = smallest value with
  precision ≥ 0.50 on DEV rows whose baseline prediction equals the channel source.
- **Primary estimator** `d_grad = unit(mean_i unit(G_i,gold − G_i,source))` at L_inj, TRAIN only.
  Prespecified comparator: same-site L_inj DiffMean. No PCA/LDA/low-rank/nonlinear.
- **Layer mapping** `round(old/(old_n−1)×(new_n−1))` from the Qwen2.5 base (28 layers,
  L_obs 20, L_inj 16). Recomputed for Qwen3-4B, not copied.

## Model freeze

`Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`, pinned before any inference.
Verified 36 layers, hidden 2560 — the layer count coincides with Qwen3-8B by computation,
the width does not. bf16, eager attention, torch 2.1.2+cu121, transformers 4.51.0,
tokenizers 0.21.1, same chat template.

## Anti-shopping

One checkpoint, one template, one tokenizer, one precision, one readout, one split. No
samples removed. No thresholds changed after seeing results. **Qwen3.5-9B untouched** — not
downloaded, loaded, baselined or inspected, despite a pre-existing unrelated cache entry.

## Split firewall

TEST (548 rows) is inaccessible during development. Every runner reconstructs the sealed
index and aborts with `QWEN3_4B_TEST_CONTAMINATION` on any sealed project_index. Sealed
identity is pinned by a hash of the ordered uuid list without opening contents. No Qwen3-4B
TEST prediction was computed.

## Stage ladder and where this run stopped

A baseline validity → B automatic channel discovery → C readability → **D direction
estimation (BLOCKED)** → E DEV intervention gate → advancement → freeze → one-shot formal TEST.

Stages A–C completed. Stage D was blocked by a hardware limit documented in
`STAGE_D_BLOCKER.json`; see that file for why no workaround was applied inside this study.
