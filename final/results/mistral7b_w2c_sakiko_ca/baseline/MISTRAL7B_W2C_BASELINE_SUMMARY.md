# Mistral-7B-Instruct-v0.3 — R0 W2C MCQ Baseline

- Model: `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71` (bf16), 32 layers, hidden 4096
- Protocol: avg_logp response-mode scoring, seed-42 W2C test/mcq, N=3652 (skipped 0)
- Split: train 2556 / val 548 / test 548

## Headline

| metric | value |
|---|---|
| accuracy | 0.4269 |
| balanced accuracy | 0.4056 |
| macro-F1 (3-cls) | 0.3319 |
| macro-F1 (4-opt) | 0.2490 |
| majority-gold baseline | 0.3546 |
| acc − majority | +0.0723 |
| total errors | 2093 |
| **R0 verdict** | **PASS** |

## Gold / prediction distribution

- gold: {'tool_call': 1295, 'cannot_answer': 1295, 'request_for_info': 1062}
- pred: {'tool_call': 2999, 'request_for_info': 103, 'direct': 327, 'cannot_answer': 223}  (majority pred class: tool_call, share 0.821)

## Confusion matrix (rows=gold, cols=pred)

| gold\pred | tool_call | direct | request_for_info | cannot_answer |
|---|---|---|---|---|
| tool_call | 1278 | 15 | 1 | 1 |
| direct | 0 | 0 | 0 | 0 |
| request_for_info | 948 | 21 | 76 | 17 |
| cannot_answer | 773 | 291 | 26 | 205 |

## Per-class precision / recall / F1

| class | P | R | F1 | support |
|---|---|---|---|---|
| tool_call | 0.426 | 0.987 | 0.595 | 1295 |
| direct | 0.000 | 0.000 | 0.000 | 0 |
| request_for_info | 0.738 | 0.072 | 0.131 | 1062 |
| cannot_answer | 0.919 | 0.158 | 0.270 | 1295 |

## Top error transitions (candidate channels; count_all / train / val / test)

| gold → pred | all | train | val | test |
|---|---|---|---|---|
| request_for_info → tool_call | 948 | 662 | 141 | 145 |
| cannot_answer → tool_call | 773 | 541 | 117 | 115 |
| cannot_answer → direct | 291 | 205 | 43 | 43 |
| cannot_answer → request_for_info | 26 | 18 | 2 | 6 |
| request_for_info → direct | 21 | 17 | 2 | 2 |
| request_for_info → cannot_answer | 17 | 15 | 0 | 2 |
| tool_call → direct | 15 | 9 | 4 | 2 |
| tool_call → cannot_answer | 1 | 1 | 0 | 0 |
| tool_call → request_for_info | 1 | 1 | 0 | 0 |

## R0 gate

```
{
  "meaningful_above_majority": true,
  "not_collapsed": true,
  "finite_scores": true,
  "deterministic_replay": true,
  "adequate_transition_support": true,
  "n_pred_classes": 4
}
R0_PASS = True
```

Deterministic replay: {'n': 40, 'pred_agreement': 40, 'score_bitmatch': 40, 'pred_identical': True, 'score_identical': True}
