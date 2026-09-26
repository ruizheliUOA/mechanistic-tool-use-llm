# Qwen2.5-7B rescorability audit

## Verdict: `QWEN2_5_NOT_RESCORABLE`

## Evidence

| file | status |
|---|---|
| `qwen25_7b_locked_test_details.jsonl` | **git-LFS pointer**, 129-byte stub, payload not materialised |
| `qwen25_7b_split_baseline_details.jsonl` | **git-LFS pointer**, payload not materialised |
| `multiseed/qwen25_7b_multiseed_details.jsonl` | present, but rows are **per-seed aggregates** (`base_acc`, `sakiko_acc`, `delta_acc`) with no per-sample predictions |

No committed non-LFS artifact records per-sample intervened predictions, and no aggregate
carries a `clean_damage`-equivalent field.

## Consequence

Neither destination-resolved accounting nor collateral can be reconstructed. The historical
Qwen2.5 conclusion — 5 channels discovered, 4 passing the older utility gate, accuracy
0.4380 → 0.5529 — rests on aggregate improvement alone.

**Aggregate improvement alone is insufficient for modern re-adjudication.** It is exactly the
evidence class the licensing thesis argues is inconclusive, so declaring Qwen2.5 either
licensed or refused would beg the question.

## What was deliberately not done

The LFS payload was **not** fetched and the model was **not** rerun. §11 forbids filling
missing evidence, and §18 forbids new inference. The gap is left open and labelled.
