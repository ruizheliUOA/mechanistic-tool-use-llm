# Gemma historical configuration C0

Reconstructed verbatim from `sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/FORMAL_CONFIG.json`
and `FORMAL_PRINCIPAL_VERDICT.json`. Immutable reference for all comparisons.

## C0

| Field | Value |
|---|---|
| Model | `google/gemma-2-9b-it` @ `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Channel | `cannot_answer → tool_call` (gold `cannot_answer`, source `tool_call`) |
| Router | StandardScaler + LogisticRegression, C=1.0, l2, liblinear, seed 42, **τ = 0.4** |
| `L_obs` | 30 |
| `L_est` | 24 (same as write site) |
| `L_inj` | 24 |
| Site | `model.model.layers[L].mlp` forward output |
| Hidden | 3584 |
| Estimator | `d_grad = unit(mean_i unit(G_i,gold − G_i,source))`, 4-mode full-effect, TRAIN-only, n=401 |
| Direction sha256 | `e939665b4da3e8e2…` |
| Norm | `s_c = 1.27609`, absolute delta norm `0.15951` |
| **Dose** | **q = 0.125**, grid {0, 0.125, 0.25, 0.5, 1.0, 2.0}, **rule = min(admissible)** |
| Gating | Router-gated, τ = 0.4 |
| Surface | rank-1 activation, additive |
| Destination | SOURCE_RETAINED \| GOLD_ARRIVAL \| OTHER_WRONG |
| Preservation | both denominators: all-correct (frozen) and Router-fired-correct (exposed) |
| Population | TRAIN 1997 / DEV 1101 / SEALED 548 |

## Outcome: `GEMMA_FORMAL_DECLINE`, 8/10

SEALED: routed 114, channel errors 96, **source_exits 27**, GOLD_ARRIVAL 17, OTHER_WRONG 10.

| Condition | Value | Pass |
|---|---|---|
| 1 support ≥ 30 | 96 | ✓ |
| 2 target gain > 0 | 0.0729 | ✓ |
| **3 target gain CI lower > 0** | **−0.0313** | **✗** |
| 4 real > K=59 random null | 0/59, p = 0.0167 | ✓ |
| 5 target hit > 0.50 | 0.6296 | ✓ |
| **6 target hit CI lower > 0.50** | **0.4444** | **✗** |
| 7 collateral point ≤ 0.05 | 1/223 = 0.0045 | ✓ |
| 8 collateral CI upper ≤ 0.05 | 0.0247 | ✓ |
| 9 zero reproduces baseline | exact | ✓ |
| 10 no structural failure | — | ✓ |

**Both failures are interval-width failures on the destination layer.** Every
point estimate passed. Specificity was clean (0/59 randoms ≥ real). The binding
constraint was `source_exits = 27` — only 28% of channel errors moved at all,
which is what made both CIs too wide.

Exposed-at-risk collateral was 1/11 = 9.09%, above the 5% budget; the 1/223
all-correct pass must never be read as deployment safety.
