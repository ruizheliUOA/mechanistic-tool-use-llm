# Phase 5 — Locked Protocol (Mistral-7B-Instruct-v0.3 × When2Call)

**Written before inspecting any full test result.** All selection/tuning uses train/val
only; each locked-test arm runs once. Layer candidates are defined by **normalized model
depth over Mistral's 32 layers** — no Qwen absolute indices are copied.

| # | item | locked value |
|---|---|---|
| 1 | model id / revision | `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71` (SHA-verified) |
| 2 | precision | **bf16** (no 4-bit/fp16 fallback; hard guard aborts otherwise) |
| 3 | W2C dataset version | `nvidia/When2Call`, config `test`, split `mcq`, N=3652 (local raw jsonl == HF order, verified) |
| 4 | split construction | **pilot:** archived fixed indices `final/results/splits/` (train 2556 / val 548 / test 548; stratified on Phi errors, model-independent). **multiseed:** deterministic `StratifiedShuffleSplit` 70/15/15 on Mistral's own `etype` (rare<5 pooled), `random_state=seed` |
| 5 | seeds | pilot 42; multiseed {42, 123, 456, 789, 2024} |
| 6 | prompt / chat template | Mistral `apply_chat_template([system,user], add_generation_prompt=True)`; system = "You are a helpful assistant." + tools JSON (or "No tools are available.") + "Given the user's question, choose the most appropriate response from the provided options." |
| 7 | candidate response-mode strings | the dataset `answers[label]` **texts** for `direct` / `tool_call` / `request_for_info` / `cannot_answer` |
| 8 | candidate scoring | length-normalized average token log-prob (`avg_logp`) of the candidate conditioned on the prompt; **argmax = prediction** |
| 9 | length normalization | divide summed token log-probs by candidate token count |
| 10 | R0 baseline-validity | PASS requires: acc > majority-gold + 0.01; **not collapsed** (rule: collapse iff majority_pred_share > 0.60 AND acc < majority_gold); all scores finite; deterministic replay identical; ≥1 transition with train-count ≥ 30 |
| 11 | channel-discovery thresholds | per-transition status via `stable_train_min=50, stable_eval_min=15, weak_train_min=20` (default), plus strict/lenient reported; 200-bootstrap stability frequency; `eval_support = min(val,test)` |
| 12 | scope filter (intervention eligibility) | gold ∈ {tool_call, request_for_info, cannot_answer}; `from_pred` is a real predicted label ≠ gold; **train error count ≥ 30** and **val support ≥ 8** and **test support ≥ 8**; status ∈ {stable, weak} at default; and an R1-surviving obs layer exists |
| 13 | bootstrap | 200 resamples of the full labeled set; report per-transition stable-frequency |
| 14 | router family | `LogisticRegression(C=1.0, solver=liblinear, max_iter=2000)` on StandardScaler-ed activations; **positives** = channel train errors; **negatives** = matched-pred (same baseline pred, different etype) |
| 15 | observation-layer grid | `[13, 18, 22, 27]` = round({0.40, 0.55, 0.70, 0.85}·32) |
| 16 | injection-layer grid | per channel: obs + {0, −2, −4} (−2 ≈ −0.06·L, −4 ≈ −0.12·L), clipped ≥0 |
| 17 | direction candidates | **DiffMean** (`unit(ref_mean − err_mean)`) default; **PCA-1** (sign-aligned to DiffMean delta) added iff `cos(DiffMean, PC1) < 0.6` (R2); winner chosen on val |
| 18 | degenerate-direction rejection (R1) | **model-agnostic relative gate:** reject obs layers with norm-ratio (DiffMean norm / median activation norm) < 0.05; among survivors pick **max norm-ratio**, tie-break router val-AUC. *(The archived absolute floor `norm < 5.0` was calibrated on Qwen, whose activation norms are ~5–10× Mistral's — Mistral's median activation norm is itself < 5 — so an absolute floor is not portable. Relative-ratio selection is exactly what the method's own §4 R1 specifies; this is a train-only geometry calibration fixed before any test inspection. See PHASE5_ENVIRONMENT_AND_PREFLIGHT / GEOMETRY.)* |
| 19 | alpha grid | `[0.5, 1.0, 2.0, 4.0, 6.0]` |
| 20 | threshold grid | `[0.4, 0.5, 0.6, 0.7, 0.8]` |
| 21 | validation objective | maximize **Net = Fixed − Broke** on val; tie-break (−Broke, then channel-own Fixed) |
| 22 | one-shot locked-test policy | lock all per-channel {obs, inj, method, router, threshold, alpha, gating, cascade order} on val; run each test arm **once**; no re-tuning after seeing test |
| 23 | placebo / control protocol | at the locked config, per selected channel: **real**, **reverse** (−unit), **≥20 matched-norm random** directions, **ungated** (all from_pred samples), **wrong-layer** (inject at L5 ≈ 0.15·L); identical router/threshold/alpha/layer/eligible-set/direction-norm |
| 24 | pilot→multiseed gate | proceed to multiseed only if all 10 mandatory criteria (see §Gate) hold on seed-42 |
| 25 | outcome definitions | see §Outcomes |

## Intervention operator

`h' = h + α · median_norm · unit(direction)` added to the **MLP output** at the injection
layer, all positions, during candidate scoring. `median_norm` = median L2 norm of train
activations at the channel's obs layer. **Gate:** a channel touches sample *i* only if
`baseline_pred[i] == channel.from_pred` **and** its router score ≥ threshold. Multi-channel
**cascade**: single correction per sample, by a fixed priority order locked on val
(over-call channels before under-call, ties by val-Net).

## Arms (seed-42 pilot)

- **Arm A** — Mistral W2C baseline (R0).
- **Arm C_i** — single-channel target-native intervention, one per scope-passing channel.
- **Arm D** — automatic multi-channel SAKIKO-CA cascade over channels that pass individual
  validation (utility gate).
- **Arm F (diagnostic only)** — the historically important W2C channels
  {rfi_tc, ca_tc, ca_direct} run under the *same* target-native validation rules,
  regardless of whether they pass Mistral's gate. Never forced into the final system if it
  fails Mistral validation.

## Gate (pilot → multiseed): all 10 must hold

1. R0 passes. 2. ≥1 auto-discovered channel has locked-test Net > 0. 3. Its own residual
decreases meaningfully (own_after ≤ 0.75·own_before). 4. Broke controlled (Broke ≤
Fixed/2). 5. Existing error channels not severely worsened (each other channel's residual ≤
baseline + 2). 6. Gain is not mere redirection (own→gold > own→other-wrong). 7. real Net >
reverse Net. 8. Defensible random advantage (real ≥ 75th pct of random **and** real Net >
mean+std of random). 9. No test-driven tuning. 10. Multi-channel composition shows
acceptable overlap/interference (cascade Net ≥ best single-channel Net − small slack, no
destructive interference).

## Outcomes

- **SUCCESS (robust):** gate passes **and** multi-seed shows ≥1 channel with positive Net
  in ≥4/5 seeds, direction-specific (real ≫ random/reverse), Broke controlled.
- **PARTIAL SUCCESS:** positive, direction-specific gain on seed-42 (+ some seeds) but not
  ≥4/5, or only one channel survives.
- **BEHAVIORAL-ONLY:** positive, gated Net but random directions frequently match it
  (not direction-specific) → report under a pre-declared behavioral-only interpretation.
- **NO-GO:** R0 fails, or no scope-passing channel passes the utility gate, or gains are
  non-specific *and* not behaviorally meaningful → honest negative; do not force channels.

## Artifact-control rules (this phase)

Model weights and all activation `.npy` caches live **outside the repo**
(`/root/autodl-tmp/models/…`, `/root/autodl-tmp/phase5_mistral_cache/`) and are never
staged. Per-example baseline details are written to the out-of-repo cache and mirrored into
`baseline/` (LFS-routed `*.jsonl`; the human decides commit-via-LFS vs local-only). No
secrets, tokens, or proxy config appear in any Phase-5 file.
