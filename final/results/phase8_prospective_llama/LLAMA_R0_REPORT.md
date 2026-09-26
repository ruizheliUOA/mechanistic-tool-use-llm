# Llama-3.1-8B-Instruct — R0 Readout Validity (Part 1)

**Model:** `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`,
bf16. **All 4 weight shards SHA256-verified byte-identical to the official Meta repo**
(see `PHASE8_EXECUTION_MANIFEST.md` §2). The primary model named in the frozen protocol was
used; **the pre-registered OLMo fallback was NOT used**.

**Config:** `llama`, 32 layers, hidden 4096, 32 heads / 8 KV heads, vocab 128256,
max_pos 131072, bf16. Derived frozen grids (from L=32, not copied):
obs {13, 18, 22, 27}; inj offsets {0, −2, −4}; wrong-layer L5.

## R0 verdict: **PASS**

| metric | value |
|---|---|
| N evaluated / skipped | 3652 / **0** |
| accuracy | **0.44031** |
| balanced accuracy | 0.43005 |
| macro-F1 (3-cls) | 0.41850 |
| majority-gold baseline | 0.35460 |
| **acc − majority** | **+0.08571** |
| total errors | 2044 |
| deterministic replay (n=40) | **pred 40/40 identical, avg_logp 40/40 identical** |
| readout collapse flag | **False** |
| score margin (mean / median / p05 / min) | 0.519 / 0.442 / 0.036 / 0.00008 |

**R0 checks:** meaningful_above_majority ✅ · not_collapsed ✅ · finite_scores ✅ ·
deterministic_replay ✅ · adequate_transition_support ✅ · n_pred_classes = 4.

## Prediction distribution — over-call profile

Gold: tool_call 1295 / cannot_answer 1295 / request_for_info 1062.
Pred: **tool_call 2337 (64.0%)** / direct 627 / request_for_info 451 / cannot_answer 237.

| model | acc | macro-F1(3) | acc − majority | **tool_call share** |
|---|---|---|---|---|
| Phi-3.5-mini | 0.4822 | 0.4526 | +0.128 | 0.676 |
| Qwen2.5-7B | 0.4381 | 0.4006 | +0.084 | 0.614 |
| Mistral-7B-v0.3 | 0.4269 | 0.3319 | +0.072 | **0.821** |
| **Llama-3.1-8B** | **0.4403** | **0.4185** | **+0.086** | **0.640** |

Llama is the **strongest baseline of the four** on accuracy and macro-F1, and its over-call
profile resembles **Qwen** (0.61), not the saturated Mistral (0.82). This mattered for the
prospective prediction: a Qwen-like over-call share did **not** imply Qwen-like actionability
(see §fragility below).

## Error transitions (all / train / val / test)

| gold → pred | all | train | val | test |
|---|---|---|---|---|
| request_for_info → tool_call | **678** | 471 | 101 | 106 |
| cannot_answer → tool_call | **535** | 383 | 79 | 73 |
| cannot_answer → direct | **463** | 316 | 73 | 74 |
| cannot_answer → request_for_info | 99 | 68 | 10 | 21 |
| tool_call → direct | 91 | 66 | 13 | 12 |
| request_for_info → direct | 73 | 54 | 11 | 8 |
| tool_call → request_for_info | 66 | 45 | 9 | 12 |
| request_for_info → cannot_answer | 25 | 18 | 5 | 2 |
| tool_call → cannot_answer | 14 | 11 | 2 | 1 |

Machine-readable: `llama_r0_summary.json`, `llama_confusion_matrix.csv`,
`llama_error_transition_matrix.csv`.

## Firewall note

Baseline predictions were computed on **all 3652 rows** — this is the R0 requirement and the
first precondition of firewall rule #5 ("baseline validity is established"). The **locked-test
intervention** (steering) was firewalled until `LLAMA_FROZEN_GATE_DECISIONS.json` was written
and its SHA256 verified; `phase8_locked_test.py` refuses to run otherwise (self-tested:
`REFUSING: frozen gate decisions + hash must exist before any test access`).
