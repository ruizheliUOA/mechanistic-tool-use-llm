# Formal preregistration — Qwen3-4B × When2Call, `cannot_answer → tool_call`

**Status: `FROZEN_BEFORE_ANY_SEALED_ACCESS`. Prepared, NOT executed.**
The 548-row evaluation partition was not opened by this task. No Qwen3-4B TEST prediction
exists. Machine-readable twin: `FORMAL_PROTOCOL_FROZEN.json`.

## Question

On the untouched evaluation population, does the frozen TRAIN-only `d_grad` direction
produce destination-correct behaviour that exceeds a prospectively generated matched-random
null?

## Frozen configuration

| item | value |
|---|---|
| model | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` |
| layers / hidden | 36 / 2560 |
| observation layer | **L_obs = 26** |
| injection layer | **L_inj = 21** |
| site | `model.model.layers[L].mlp` forward output, after MLP internals, before decoder residual addition |
| positions | all sequence positions, matching the frozen full-effect aggregation |
| dtype / attention / batch | bfloat16 / eager / 1 |
| Router | `ROUTER_cannot_answer__tool_call.npz`, sha256 `f45d6b7f…`, **refit forbidden** |
| Router threshold | **τ = 0.011564** |
| gate | baseline prediction == `tool_call` AND `predict_proba` ≥ τ |
| s_c | 24.6995621618 |
| dose | **q = 0.25** |
| absolute perturbation norm | **6.1748905405** |
| primary direction | `PRIMARY_DIRECTIONS.safetensors::d_grad__cannot_answer__to__tool_call`, sha256 `84c10294…` |
| DiffMean comparator | `DIFFMEAN_DIRECTIONS.safetensors::L26_diffmean_frozen__…` |
| score-space comparator | `b_c = 1.4013434649`, applied as `e_gold += b_c/2 ; e_source −= b_c/2` |
| loader | committed `load_model()` semantics: `eval()`, `use_cache=False`, all parameters `requires_grad_(False)` |

## Arm battery, in frozen execution order

environment/hash verification → model load → evaluation baseline → **zero** → exact
zero-equality gate → **real d_grad** → **frozen L26 DiffMean** → **reverse −d_grad** →
**wrong-layer d_grad at L26** → **ungated d_grad** → **K = 59 formal randoms in seed
order** → **score-space comparator** → endpoint computation → report.

Excluded, and not addable: same-layer L21 DiffMean (DEV diagnostic only, not promoted),
another channel's `d_grad`, orthogonalised mismatch, pooled or shared direction, PCA or
rank-2 directions, any new estimator, any additional dose.

## K = 59 random generation rule

```
for i in 0..58:
    seed  = uint64( SHA256("SAKIKO_QWEN3_4B_CA_TC_STAGE2_RANDOM_V1|" + decimal(i))[:8] )
    g     = Generator(PCG64DXSM(seed))
    z     = g.standard_normal(2560)          # float64
    v_i   = (z / ||z||)  in float64, cast once to float32
```

Frozen and materialised: `FORMAL_RANDOMS.safetensors`, matrix sha256 `063bb9d2…`.
No rejection, no orthogonalisation, no cosine filtering, no outcome matching, no post-hoc
selection, no regeneration. Cosines with `d_grad` (min −0.0462, median −0.0053, max
0.0360) were recorded **after** the complete set was frozen; no vector was removed on their
basis.

**Seed disjointness.** The salt differs from the Qwen3-8B formal salt, so no vector is
shared with that program. The 59 formal seeds are 64-bit hash-derived and intersect neither
the eight DEV development seeds (20260811–20260818) nor the historical seeds
(20260729–20260801).

## Endpoints

```
source_exits          = count(baseline pred = source AND intervention pred != source)
gold_arrivals         = count(baseline pred = source AND intervention pred = gold)
wrong_to_wrong        = count(baseline pred = source AND pred != source AND pred != gold)
target_hit            = gold_arrivals / source_exits
target_gain_count     = gold_arrivals - wrong_to_wrong
target_gain_rate      = target_gain_count / n_channel_error
clean_collateral_rate = baseline-correct rows made incorrect / baseline-correct rows
```

Destinations resolve to exactly `SOURCE_RETAINED` / `GOLD_ARRIVAL` / `OTHER_WRONG`;
collateral to `CORRECT_RETAINED` / `BROKEN_TO_SOURCE` / `BROKEN_TO_OTHER`.

Baseline is computed once over all 548 evaluation rows **before** any arm. The channel-error
population and the safety denominator are fixed from that baseline and are never redefined
using intervention results.

## Primary analysis

Statistic `target_gain_rate`. Null: the K = 59 fresh matched-random directions at the
identical absolute perturbation norm and Router gate.

```
p_add_one = (1 + count[ target_gain_rate_random >= target_gain_rate_real ]) / (K + 1)
```

Bootstrap: 10,000 draws, seed `uint64(SHA256("SAKIKO_QWEN3_4B_CA_TC_STAGE2_BOOTSTRAP_V1")[:8])`
= `4482706006845774666`, frozen before execution.

### Formal success conjunction — an intersection claim, no subset substitution

1. evaluation channel-error support `n_channel_error ≥ 30`
2. real `target_gain_rate` point estimate > 0
3. bootstrap 95% lower bound for real `target_gain_rate` > 0
4. `p_add_one ≤ 0.05`
5. real `target_hit` point estimate > 0.50
6. bootstrap 95% lower bound for real `target_hit` > 0.50
7. `clean_collateral_rate` point estimate ≤ 0.05
8. bootstrap 95% upper bound for `clean_collateral_rate` ≤ 0.05
9. zero-control equality passes exactly
10. no structural or provenance failure occurs

Secondary endpoints are ordered, non-promotable, and **must not rescue a failed primary**.

## TEST-access rule

SEALED TEST may be opened only by a separate, explicitly authorised one-shot formal
execution task, and only after (a) this freeze is committed and pushed, and (b) a gate-only
verification passes — hashes, model and tokenizer identity, software versions, channel and
configuration, direction hashes, random count and hashes, seed disjointness, arm
completeness and fixed order, batch size 1, single-load path, zero gate ordering, record
schema, primary/secondary hierarchy, bootstrap spec, support branch, VOID rules, sealed
firewall, and an empty formal output namespace. **This task did not open it.**

## Mechanical VOID / restart rule

VOID is mechanical only. Reason codes: `MODEL_HASH_MISMATCH`, `EVAL_POPULATION_MISMATCH`,
`ZERO_CONTROL_FAILURE`, `ARM_MISSING_OR_REORDERED`, `DIRECTION_MISMATCH`,
`ROUTER_MISMATCH`, `DOSE_MISMATCH`, `BATCH_SIZE_CHANGE`, `RUNNER_MODIFIED`,
`AUTOMATIC_RETRY_OR_RESUME`, `ENDPOINT_MODIFIED`, `RANDOM_VECTOR_REPLACED`,
`INCOMPLETE_ALL_ARM_EXECUTION`.

Once VOID is declared, endpoints must not be computed, automatic retry is forbidden, and
restart is manual. **A natural scientific outcome — including insufficient evaluation
support, or a failed conjunction — is NOT void.**

## Formal runner

Not written in this task. It must import the committed frozen readout and these frozen
artifacts, add no estimator and no arm, and pass gate-only verification before any sealed
row is opened. The Stage D/E development runners are hash-pinned in
`RUNNER_AND_PROTOCOL_HASHES_STAGE_D_E.json` so that any later drift is detectable.

## Interpretation rules, fixed in advance

- **Full conjunction passes** — evidence supports destination-correct and safety-bounded
  SAKIKO correction for one prospectively evaluated Qwen3-4B channel, following disclosed
  development-stage channel, estimator and dose selection.
- **Positive behaviour but not beating random** — behavioural movement occurred; direction
  specificity is not confirmed.
- **Real beats random but target-hit or collateral fails** — direction-specific steering is
  supported; target-correctable deployment is not.
- **Score comparator matches or exceeds** — do not claim the activation intervention
  provides behaviour unavailable to a simple frozen score-space shift.
- **DiffMean ineffective** — report estimator dependence; do not claim DiffMean is
  universally invalid.
- **No generalisation** from one channel to all Qwen3-4B channels, all models, or all
  tool-use tasks.

## What this is not

Not a cross-channel predictor validation. Not an untouched choice among all possible
channels. Not an independent-dataset replication — the measurement environment is the
established When2Call one and only the host model changed
(`DATASET_KNOWN / MODEL_UNTOUCHED`). Not a claim that gradient-based steering is novel in
general. Not a confirmation that all Qwen3-4B channels are correctable.
