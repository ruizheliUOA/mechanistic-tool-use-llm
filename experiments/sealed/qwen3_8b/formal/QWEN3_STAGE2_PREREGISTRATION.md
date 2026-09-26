# Qwen3 Stage 2 formal preregistration

Status: **FROZEN_BEFORE_FORMAL_EXECUTION**. Tier: FORMAL PROTOCOL PREPARATION / NO SEALED-EVALUATION ACCESS.

## The formal question

On the untouched evaluation population, does the frozen TRAIN-only d_grad direction produce destination-correct behaviour that exceeds a prospectively generated matched-random null?

This is a **single-channel confirmatory test following disclosed
development-stage channel, estimator and dose selection**. It is explicitly not:
a cross-channel predictor validation, an untouched choice among all possible channels, a claim that gradient-based steering is novel in general, a confirmation that all SAKIKO channels are correctable.

## Frozen configuration

| item | value |
|---|---|
| channel | `cannot_answer__to__tool_call` (gold `cannot_answer` → source `tool_call`) |
| model | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` |
| L_obs / L_inj | 26 / 21 |
| site | `model.model.layers[L].mlp forward output, after MLP internals and before decoder residual addition` |
| positions | all sequence positions, matching the frozen full-effect aggregation |
| Router tau | 0.4 |
| relative dose q | 1.0 |
| TRAIN-only scale s_c | 41.56598488897048 |
| absolute ‖Δh‖ = q·s_c | 41.56598488897048 |
| score comparator b_c | 1.673107087612152 |
| batch size / dtype / attention | 1 / bfloat16 / eager |
| formal null size K | **59** |

## Primary statistic and conjunction

`target_gain_rate = (gold_arrivals - wrong_to_wrong) / n_channel_error`

Null: K fresh matched-random directions at the identical absolute perturbation norm and Router gate.
`p_add_one = (1 + count[target_gain_rate_random >= target_gain_rate_real]) / (K + 1)`

**Confirmatory success requires the conjunction of all 10 conditions:**

1. evaluation channel-error support n_channel_error >= 30
2. real target_gain_rate point estimate > 0
3. bootstrap 95% lower bound for real target_gain_rate > 0
4. p_add_one <= 0.05 against the K fresh randoms
5. real target_hit point estimate > 0.50
6. bootstrap 95% lower bound for real target_hit > 0.50
7. clean_collateral_rate point estimate <= 0.05
8. bootstrap 95% upper bound for clean_collateral_rate <= 0.05
9. zero-control equality passes exactly
10. no structural or provenance failure occurs

This is an **intersection claim**. Failure of any condition prevents the full
destination-correct-and-safe confirmation, and the conjunction **must not** be
replaced with a best-looking subset after execution.

## Secondary, ordered and non-promotable

1. d_grad versus frozen L26 DiffMean: paired difference in target_gain_rate
2. d_grad versus frozen score-space comparator: paired difference in target_gain_rate and wrong_to_wrong rate
3. target_hit
4. source_exits
5. gold_arrivals
6. wrong_to_wrong
7. Fixed / Broke / Net
8. continuous target-source margin change
9. reverse contrast
10. wrong-layer contrast
11. ungated collateral

Secondary results **must not rescue a failed primary conjunction**.

Multiplicity: none required: one formal primary channel and one primary null comparison.

## Random budget

| option | cells | hours | +15% contingency | within 3 h |
|---|---:|---:|---:|---|
| K=39 | 4647.6 | 1.641 | 1.887 | True |
| K=59 | 6335.3 | 2.236 | 2.572 | True |
| K=79 | 8022.9 | 2.832 | 3.257 | False |

**Selected K = 59**, chosen on engineering cost only; independent of DEV rankings, effect sizes, anticipated p-value, random geometry or any evaluation information.

## Arm battery

baseline, zero, real TRAIN-only d_grad, K fresh matched-random directions, frozen original L26 DiffMean, reverse -d_grad, wrong-layer d_grad, ungated d_grad, frozen score-space target/source comparator.

Excluded: same-layer L21 DiffMean, another channel's d_grad, orthogonalized mismatch, pooled or shared direction, PCA or rank-2 direction, any new estimator, any additional dose.

## Populations

- baseline: single baseline over all 548 evaluation rows computed first
- channel-error: `gold = cannot_answer AND baseline prediction = tool_call`
- safety denominator: `baseline prediction = gold`

Neither population is ever redefined using intervention results.

## Bootstrap

10000 resamples; seed derived as uint64 of first 8 bytes of SHA256(seed_salt) from salt
`SAKIKO_QWEN3_CA_TC_STAGE2_BOOTSTRAP_V1`. Frozen before execution and not changed afterwards.

## Execution order

1. environment and hash verification
2. model load
3. evaluation baseline
4. zero arm
5. exact zero-equality gate
6. real d_grad
7. frozen DiffMean
8. reverse
9. wrong-layer
10. ungated
11. K formal randoms in seed order
12. score comparator
13. endpoint computation
14. formal report generation

## Support rule

If `n_channel_error < 30`: return
**`FORMAL_SUPPORT_INSUFFICIENT_NO_CONFIRMATORY_CLAIM`**. The run remains valid, all
descriptives are reported, and this **must not** be reinterpreted as intervention
failure.

## VOID rules

Permanently VOID if any of these occurs after evaluation access begins:
model/tokenizer/hash mismatch, evaluation population mismatch, zero-control failure, arm missing or reordered, direction/vector mismatch, Router mismatch, dose mismatch, batch-size change, scientific runner modification, automatic retry or resume, endpoint modification, random-vector replacement, incomplete all-arm execution.

Natural scientific outcomes are **not** void. A crash before evaluation access is
repairable only through a new versioned freeze and separate approval; a crash
after evaluation access begins is VOID and must not be automatically retried.

## Interpretation rules, frozen now

- **d_grad_significantly_exceeds_comparator** — Report only as a secondary result supporting a non-reducible destination pattern under the frozen comparator.
- **diffmean_ineffective** — Report estimator dependence; do not claim DiffMean is universally invalid.
- **full_conjunction_passes** — Evidence supports destination-correct and safety-bounded SAKIKO correction for one prospectively evaluated Qwen3 channel following disclosed development-stage channel, estimator and dose selection.
- **no_generalization** — Do not generalize from one channel to all Qwen3 channels, all models or all tool-use tasks.
- **positive_behaviour_but_not_beating_random** — Behavioural movement occurred, but direction specificity is not confirmed.
- **real_beats_random_but_target_hit_or_collateral_fails** — Direction-specific steering is supported, but target-correctable deployment is not supported.
- **score_comparator_matches_or_exceeds** — Do not claim that activation intervention provides behaviour unavailable to a simple frozen score-space shift.
