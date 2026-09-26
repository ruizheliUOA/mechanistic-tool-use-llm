# Qwen3-8B V1 baseline behaviour: tool_call overprediction

Source of every number in this document: the committed V1 artifacts
`QWEN3_STAGE1_BASELINE_SUMMARY.json` and
`QWEN3_STAGE1_COMPLETE_CHANNEL_LEDGER.csv`. Nothing here is restated from
memory and nothing here involves the sealed evaluation partition.

**This is a baseline behavioural finding, not an intervention result.** No
direction was estimated, no Router was fitted, no geometry was computed and no
intervention was applied at any point in V1. The quantities below describe only
what the frozen four-candidate teacher-forced readout produced.

## Headline metrics

| split | n | accuracy | macro-F1 (4-mode) |
|---|---:|---:|---:|
| TRAIN | 2556 | 0.422144 | 0.273266 |
| DEV | 548 | 0.423358 | 0.264602 |

TRAIN and DEV agree closely (0.4221 vs 0.4234
accuracy), so the behaviour is a stable property of the model on this
population rather than an artefact of one partition.

## Prediction distribution

| mode | TRAIN predicted | TRAIN share | DEV predicted | DEV share |
|---|---:|---:|---:|---:|
| tool_call | 1819 | 0.7117 | 393 | 0.7172 |
| direct | 418 | 0.1635 | 99 | 0.1807 |
| request_for_info | 205 | 0.0802 | 36 | 0.0657 |
| cannot_answer | 114 | 0.0446 | 20 | 0.0365 |

## Gold support

| mode | TRAIN gold | DEV gold |
|---|---:|---:|
| tool_call | 898 | 203 |
| direct | 0 | 0 |
| request_for_info | 745 | 161 |
| cannot_answer | 913 | 184 |

## The overprediction

`tool_call` is predicted on **1819 of 2556
TRAIN rows (71.2%)** and **393 of
548 DEV rows (71.7%)**, while gold `tool_call` accounts for
only 898 TRAIN rows (35.1%) and 203 DEV
rows (37.0%).

The model therefore emits `tool_call` roughly
2.03 times more often than the gold
distribution warrants on TRAIN, and 1.94
times on DEV.

## Per-gold accuracy

TRAIN:

| gold | n | correct | accuracy |
|---|---:|---:|---:|
| tool_call | 898 | 849 | 0.945434 |
| direct | 0 | 0 | n/a (no gold rows) |
| request_for_info | 745 | 128 | 0.171812 |
| cannot_answer | 913 | 102 | 0.111720 |

DEV:

| gold | n | correct | accuracy |
|---|---:|---:|---:|
| tool_call | 203 | 190 | 0.935961 |
| direct | 0 | 0 | n/a (no gold rows) |
| request_for_info | 161 | 26 | 0.161491 |
| cannot_answer | 184 | 16 | 0.086957 |

The asymmetry is the important part: the model is accurate when the gold answer
really is `tool_call` and weak on every other gold class, because its errors
overwhelmingly flow *into* `tool_call`.

## Complete confusion matrices

TRAIN (n=2556):

| gold \ predicted | tool_call | direct | request_for_info | cannot_answer | row total |
|---|---:|---:|---:|---:|---:|
| **tool_call** | 849 | 32 | 13 | 4 | 898 |
| **direct** | 0 | 0 | 0 | 0 | 0 |
| **request_for_info** | 570 | 39 | 128 | 8 | 745 |
| **cannot_answer** | 400 | 347 | 64 | 102 | 913 |
| **column total** | 1819 | 418 | 205 | 114 | 2556 |

DEV (n=548):

| gold \ predicted | tool_call | direct | request_for_info | cannot_answer | row total |
|---|---:|---:|---:|---:|---:|
| **tool_call** | 190 | 8 | 4 | 1 | 203 |
| **direct** | 0 | 0 | 0 | 0 | 0 |
| **request_for_info** | 122 | 10 | 26 | 3 | 161 |
| **cannot_answer** | 81 | 81 | 6 | 16 | 184 |
| **column total** | 393 | 99 | 36 | 20 | 548 |

## Consequence recorded for later blocks

Because errors concentrate into `tool_call`, the gold classes that generate
large error populations (`cannot_answer`, `request_for_info`) are exactly the
gold classes left with few correct rows. That coupling is analysed in
`OUTCOME_CONDITIONED_REFERENCE_SUPPORT_WINDOW.md`.
