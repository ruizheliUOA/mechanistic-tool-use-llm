# Formal arm inventory — resolving 65 vs 66

**Answer: 65.** Determined from the committed Qwen3-4B formal records, not from assumption.

## Canonical evidence

`QWEN3_4B_FORMAL_RECORDS.jsonl` contains **14,404 records across exactly 65 distinct arms** —
6 named forward arms plus `formal_random_0..58`.

| arm | n |
|---|---:|
| `zero` | 218 |
| `real_d_grad` | 218 |
| `d_L26_diffmean` | 218 |
| `reverse_d_grad` | 218 |
| `wrong_layer_d_grad` | 218 |
| `ungated_d_grad` | **452** (all `pred==source`, not just routed) |
| `formal_random_0..58` | 218 each |

`(5 + 59) x 218 + 452 = 14,404`. Exact.

## Why the naive 66 is wrong

The naive count treats **score-space as a 66th arm**. It is not. Score-space performs **zero
model forward passes** and writes **zero records**: `b_c` is the median margin shift computed
from the *real arm's already-stored scores*, then applied arithmetically in score space. It is a
**derived comparator**, not an intervention arm.

`zero` **is** a distinct forward arm — it runs the full forward pass with `delta=None` and must
reproduce baseline predictions exactly.

## Gemma

Identical semantics, with `d_Lobs_diffmean` substituting for `d_L26_diffmean` (L30 for L26) and
`wrong_layer` at L30. Expected record count is `64 x n_routed + n_pred_eq_source`, both of which
depend on sealed Router firing and are therefore verified **at execution**, not pre-declared.

**No arm was created or removed to make the count equal 65.**
