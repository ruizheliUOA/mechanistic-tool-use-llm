# Predeclared hardware structural exclusion — frozen BEFORE any baseline

## What is excluded

**6 rows of 3652 = 0.1643%**, raw indices `[217, 985, 1487, 2078, 2574, 3342]`.

## Why

Under the frozen configuration — BF16, **eager attention (unchanged)**, `requires_grad=False` on
all parameters, `expandable_segments` allocator, and gradient checkpointing validated
bit-identical — the four-mode full-effect gradient stack at `L_inj = 24` exceeds the 23.52 GiB
RTX 4090 D on these rows and only these rows. Eager attention is O(n²) in sequence length and
§12 forbids changing the attention implementation to obtain a result, so this is a hard limit.

## The threshold is not a tuned parameter

The population has a **natural gap**: the longest feasible row is `max_seq = 3169` and the
shortest infeasible row is `max_seq = 4572`. **No row lies between them.** Every threshold in
`[3200, 4572]` selects exactly the same 6 rows. The exclusion is therefore determined by the
data's own structure plus the hardware, not by a choice made to obtain an outcome.

| threshold | rows excluded |
|---|---|
| ≥ 2500 | 14 (0.383%) |
| ≥ 3000 | 8 (0.219%) |
| **≥ 3200 … ≥ 4572** | **6 (0.164%)** |

## Split incidence — CORRECTED

An earlier version of this file computed incidence under the **superseded V1** allocation and
stated that all 6 rows fall in TRAIN with DEV untouched. That was wrong. Under the frozen
**V2** allocation (TRAIN 2000 / DEV 1104) the incidence is:

| split | size | excluded | remaining |
|---|---:|---:|---:|
| TRAIN | 2000 | **3** (0.150%) | 1997 |
| DEV | 1104 | **3** (0.272%) | 1101 |
| **SEALED** | **548** | **0** | **548** |

| raw idx | gold | V2 split |
|---:|---|---|
| 217 | cannot_answer | dev |
| 985 | cannot_answer | dev |
| 1487 | request_for_info | train |
| 2078 | request_for_info | train |
| 2574 | tool_call | dev |
| 3342 | tool_call | train |

**SEALED remains bit-identical and completely untouched** — all six excluded rows lie in the
3104-row non-test population. But DEV is *not* untouched: it loses 3 of 1104 rows (0.272%).
Any DEV-stage quantity is computed on 1101 rows, and that must be disclosed wherever a DEV
number is reported for Gemma.

## Status and honesty conditions

This is a **`HARDWARE_STRUCTURAL_EXCLUSION`**, of the same kind as the frozen protocol's
existing `MAX_TOKENS = 8192` predeclared structural exclusion, but tighter and driven by
measurement on this GPU.

- Frozen **before** any baseline was observed. No outcome was known when it was set.
- The excluded indices are enumerated above and are fixed; they are excluded consistently at
  every stage.
- It is a deviation from "D held exactly fixed" and **must be disclosed as such** wherever the
  Gemma setting is reported. `D` for Gemma is When2Call minus 6 TRAIN rows.
- It must **not** be described as a scientific exclusion, and the affected rows must not be
  characterised in any way — they were never scored.

## What would remove it

A GPU with more memory. §12 lists "larger GPU" as an allowed remedy; none is available here.
