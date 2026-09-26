# Collateral claim lock — frozen BEFORE any SEALED access

Binding on the formal report and on every downstream claim. Frozen by this gate, before the
formal experiment is authorised, so it cannot be negotiated after the outcome is seen.

## Source

`research_exploration/qwen3_4b_collateral_integrity_audit_v1/`, verdict
**`QWEN3_4B_ADVANCEMENT_CONFIRMED`**, committed at `59a1628` and hash-valid (18/18).

## The two quantities, kept distinct

### Frozen formal metric — all-baseline-correct denominator

`clean_collateral_rate = baseline-correct rows made incorrect / baseline-correct rows`
(`QWEN3_STAGE2_ENDPOINT_SPEC.md:12`; `qwen3_stage2a_v2_analyze.py:94,127`; and
`FORMAL_PROTOCOL_FROZEN.json → endpoints.clean_collateral_rate`).

On DEV: **3 / 455 = 0.006593**, frozen percentile bootstrap 95 % **[0.000000, 0.015385]** —
frozen criterion passes (point ≤ 0.05 and CI upper ≤ 0.05).

This is the endpoint the formal conjunction evaluates. It is unchanged by this gate.

### Exposure-conditioned diagnostic — Router-fired / actually perturbed denominator

Router-fired ∩ baseline-correct, verified set-identical to the actually-perturbed rows.

On DEV: **3 / 106 = 0.028302**, same frozen bootstrap method **[0.0000, 0.0660]**,
Clopper–Pearson exact **[0.0059, 0.0805]**. Point estimate under 0.05; **both upper bounds
exceed 0.05**.

76.7 % of the frozen denominator (349 of 455 rows) was never perturbed and could not have
broken.

## The lock

> **Passing the frozen formal collateral endpoint must not be interpreted as demonstrated
> deployment safety.**

Concretely, the formal report must:

1. report the collateral rate on **both** denominators, side by side, wherever collateral is
   reported;
2. state the exposure-conditioned uncertainty, which is too wide to establish
   deployment-level safety;
3. never describe the frozen rate as a demonstrated safety property, and never make a
   zero- or low-collateral-risk claim or any deployment-readiness claim;
4. carry the disclosure that the channel, estimator and dose were selected on TRAIN/DEV
   evidence while the evaluation partition was sealed — not an untouched choice among all
   channels;
5. carry the disclosure that this is `DATASET_KNOWN / MODEL_UNTOUCHED` cross-model
   replication on the established When2Call environment, never described as an
   independent-dataset replication.

## Precedent

This mirrors the Qwen3-8B metric-compression claim lock
(`final/results/qwen3_stage2_metric_compression/`), which kept the frozen metric as the
primary endpoint, labelled the eligible-at-risk reconstruction `POST-HOC EXPLORATORY`, and
added "zero collateral risk" to its prohibited-claims table. Qwen3-4B inherits that
discipline. Audit verdict on the comparison: `PRECEDENT_CONSISTENT`.

## What this lock does NOT do

It does not alter the formal success conjunction, the collateral endpoint definition, the
denominator, the threshold or the CI method. Those remain exactly as frozen in
`FORMAL_PROTOCOL_FROZEN.json`. Changing the gate metric now — in either direction — would
itself be a post-outcome protocol change and is forbidden.
