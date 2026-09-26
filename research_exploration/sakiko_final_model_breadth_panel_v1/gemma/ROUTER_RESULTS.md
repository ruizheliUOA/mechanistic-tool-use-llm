# Router readability — Gemma-2-9B-it

Frozen modern Router family, hyperparameters verbatim: `StandardScaler` +
`LogisticRegression(C=1.0, penalty=l2, solver=liblinear, max_iter=2000, tol=1e-4,
random_state=42, class_weight=None, fit_intercept=True)` on the **L_obs = 30** MLP output at the
final prompt token. τ grid `[0.4, 0.5, 0.6, 0.7, 0.8]`, τ rule *smallest τ with
source-matched DEV precision ≥ 0.50*, eligibility *DEV comparable ROC-AUC ≥ 0.75 **and**
selected-τ precision ≥ 0.50*. **No gate modified.**

DEV denominator is the actual **1101** rows, not 1104.

| channel | TRAIN AUC | DEV ROC-AUC | 95% CI | PR-AUC | τ | precision | recall | eligible |
|---|---:|---:|---|---:|---:|---:|---:|---|
| `request_for_info → tool_call` | 0.9999 | **0.7067** | [0.6666, 0.7456] | 0.5302 | 0.4 | 0.5131 | 0.6015 | **NO** |
| `cannot_answer → tool_call` | 1.0000 | **0.9260** | [0.9009, 0.9484] | 0.8450 | 0.4 | 0.8319 | 0.8616 | **YES** |
| `cannot_answer → direct` | 1.0000 | **0.9190** | [0.8831, 0.9510] | 0.8036 | 0.4 | 0.9773 | 0.7818 | **YES** |

## Firing detail at the selected τ

| channel | fired channel-errors | fired baseline-correct (DEV) | fired off-channel errors | DEV rows predicting source |
|---|---:|---:|---:|---:|
| `rfi → tc` | 157 | 126 | 97 | 610 |
| `ca → tc` | 193 | 50 | 121 | 232 |
| `ca → direct` | 86 | 37 | 138 | 88 |

## Verdicts

- **`request_for_info → tool_call` — `READABILITY_ROUTER_DECLINE`**, on
  `DEV_ROC_AUC_LT_0.75`. The failure is unambiguous: the entire 95% interval
  [0.6666, 0.7456] lies below the 0.75 gate. The channel is **retained in the record** and was
  not dropped for convenience; it simply does not advance.
- **`cannot_answer → tool_call`** and **`cannot_answer → direct`** are `ROUTER_ELIGIBLE`.

## Two things this does not mean

1. **Router success is `READABLE` only.** It is not evidence of steerability and certainly not
   of correctability. Two channels being readable says nothing yet about whether any
   intervention on them deserves a licence.
2. **TRAIN AUC ≈ 1.0 is in-sample overfitting**, not a finding — 3584 features against ~1000
   rows. Only the DEV comparable AUC is load-bearing, which is why the frozen rule uses it.

**2 of 3 channels advance to direction estimation.** `cannot_answer → tool_call` carries no
privileged status from having been Qwen3-8B's formal channel; it advanced on Gemma's own DEV
numbers, and so did `cannot_answer → direct`.
