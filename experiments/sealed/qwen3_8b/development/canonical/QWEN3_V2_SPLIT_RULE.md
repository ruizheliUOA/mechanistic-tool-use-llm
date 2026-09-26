# Qwen3 V2 split rule

One rule. One split. One support recount. This rule is frozen **before** the
split is computed and before any V2 channel count exists.

## Population

The fixed **3,104 non-test immutable UUIDs inherited from V1**. No row enters or
leaves this population. The sealed 548-row evaluation partition is not touched:
its identity is inherited by reference from the frozen V1 manifest and its hash
is never recomputed from payload.

## Targets

- TRAIN: **2000**
- DEV: **1104**

## Exact rule

1. Group rows by **gold label only**.
2. For each row compute

   `split_hash = SHA256("SAKIKO_QWEN3_V2_ALLOC_V1|" + immutable_uuid)`

3. Sort each gold stratum by `split_hash` ascending, with `immutable_uuid`
   ascending as the tie breaker.
4. Allocate dev quotas proportionally to gold-class size using the
   **largest-remainder** method, with quotas summing exactly to 1104.
5. Assign the first `quota` rows of each stratum to **DEV**.
6. Assign every remaining non-test row to **TRAIN**.

## Resulting quotas

These follow deterministically from the already-known gold-class totals and the
target sizes. They involve no prediction, no outcome and no channel.

| gold class | gold total | DEV quota | TRAIN quota |
|---|---:|---:|---:|
| cannot_answer | 1097 | 390 | 707 |
| request_for_info | 906 | 322 | 584 |
| tool_call | 1101 | 392 | 709 |

## Admissible and inadmissible dependencies

The rule may depend only on: `immutable UUID`, `gold label`, `fixed literal salt`, `total target sizes`.

The rule must not depend on: `baseline prediction`, `correct/error status`, `channel identity`, `candidate score`, `margin`, `sequence length`, `attention-guard status`, `Router result`, `geometry`.

The six guard-triggering rows receive **no special split treatment**; they
follow the same gold-stratified hash rule as every other row.

## Prohibited

- trying another seed or salt;
- trying another dev size;
- changing stratification;
- using channel membership, predictions, geometry or Router results;
- changing the split after seeing support counts.

## Sizing justification

Inputs used: total gold-class counts over the 3104 non-test rows, total correct-reference counts over the same rows, hypergeometric expectation and standard deviation, frozen support thresholds.

Inputs deliberately **not** used: any candidate V2 partition, per-channel V2 membership, any alternative seed, any alternative size.

The dev fraction is 1104 / 3104 = **0.3557**.

Under the frozen support gate the binding criteria for the reference side are
`train_correct_reference >= 60` and
`dev_correct_reference >= 30`. Treating the
allocation as a draw without replacement inside each gold stratum, the
correct-reference counts are hypergeometric:

| gold class | gold total | correct total | DEV quota | E[dev correct] | sd above 30 | E[train correct] | sd above 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cannot_answer | 1097 | 118 | 390 | 42.0 ± 4.9 | 2.43 | 76.0 ± 4.9 | 3.27 |
| request_for_info | 906 | 154 | 322 | 54.7 ± 5.4 | 4.57 | 99.3 ± 5.4 | 7.25 |
| tool_call | 1101 | 1039 | 392 | 369.9 ± 3.7 | 92.77 | 669.1 ± 3.7 | 166.22 |

Aggregate feasibility calculations indicate that this allocation places the
dominant reference classes inside the intended support window: every gold class
carries an expected dev correct-reference count above the dev threshold and an
expected train correct-reference count above the train threshold, with the
tightest class more than two standard deviations clear on the dev side.

Raising the dev fraction further would push the train reference side of the
weakest class toward its own boundary; the chosen fraction keeps both sides
clear simultaneously.

**This does not guarantee any particular K or any particular channel identity.** This calculation is
about aggregate reference availability only. It says nothing about which
directed channels will pass, because channel membership additionally requires
error support that is not part of this calculation.
