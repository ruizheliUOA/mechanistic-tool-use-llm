# Automatic channel discovery — Gemma-2-9B-it

All 12 ordered non-diagonal transitions enumerated. **No channel preselected.** The Qwen3
`cannot_answer→tool_call` channel was given no privileged status.

## Allocation correction — recorded, not hidden

Channel discovery was first run under the **V1** TRAIN/DEV allocation (2556/548) and returned
**`NO_SUPPORT_ELIGIBLE_CHANNELS`**, binding on `dev_correct_reference_min = 30`
(`request_for_info` 28, `cannot_answer` 12).

That was **my error, not a result.** The frozen modern protocol is
`QWEN3_8B_STAGE0_1_V2`, which supersedes V1 as *one disclosed development-stage allocation
amendment*: TRAIN 2000 / DEV 1104 over the same 3104 non-test rows, gold-stratified by
deterministic UUID hashing with the fixed salt `SAKIKO_QWEN3_V2_ALLOC_V1|`. V2 **did not**
lower thresholds, change the estimator, or select channels manually — `support_thresholds_lowered:
false`, `manual_channel_selection: false`.

V1 returned `NO_SUPPORT_ELIGIBLE_CHANNELS` for **Qwen3-8B too**, for the same reason. Reporting
Gemma's V1-allocation result as a scientific outcome would have been a false negative produced
by a superseded protocol version.

The V2 allocation depends only on immutable UUID, gold label, the fixed salt and target sizes,
and explicitly **must not** depend on baseline prediction, correct/error status, channel
identity, score, margin or sequence length. It is therefore model-independent and applies to
Gemma unchanged. UUID join: **3098/3098**.

## Result under the frozen V2 allocation — TRAIN 1997 / DEV 1101

| channel | train err | train ref | dev err | dev ref | eligible |
|---|---:|---:|---:|---:|---|
| `request_for_info → tool_call` | 499 | 71 | 261 | 48 | **YES** |
| `cannot_answer → tool_call` | 401 | 77 | 224 | 38 | **YES** |
| `cannot_answer → direct` | 202 | 77 | 110 | 38 | **YES** |
| `cannot_answer → request_for_info` | 27 | 77 | 16 | 38 | no |
| `request_for_info → direct` | 7 | 71 | 11 | 48 | no |
| `tool_call → direct` | 5 | 696 | 4 | 385 | no |
| `request_for_info → cannot_answer` | 5 | 71 | 2 | 48 | no |
| `tool_call → request_for_info` | 4 | 696 | 2 | 385 | no |
| `tool_call → cannot_answer` | 3 | 696 | 0 | 385 | no |
| `direct → tool_call` | 0 | 0 | 0 | 0 | no |
| `direct → request_for_info` | 0 | 0 | 0 | 0 | no |
| `direct → cannot_answer` | 0 | 0 | 0 | 0 | no |

**3 support-eligible channels — the same three the Qwen3-8B ADMIT setting produced.** No
threshold was altered. Every unsuccessful channel is reported above.

`direct` has zero support because it is **never a gold label** anywhere in When2Call
(population gold: `cannot_answer` 1295, `request_for_info` 1062, `tool_call` 1295). It is a
distractor option only. This is a property of the benchmark, identical for Qwen3-8B.
