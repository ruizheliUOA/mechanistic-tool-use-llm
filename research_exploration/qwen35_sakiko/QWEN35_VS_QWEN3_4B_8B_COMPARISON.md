# Qwen3.5-9B vs Qwen3-8B / Qwen3-4B / Gemma-2-9B

All figures recomputed from committed per-row artifacts. `D` is held fixed at When2Call
throughout, so every difference below is a host-model difference.

## The central finding: readability is a property of the channel, not the model

DEV comparable Router ROC-AUC, frozen family and hyperparameters, gate = 0.75:

| channel | Qwen3-8B | Gemma-2-9B | Qwen3.5-9B |
|---|---:|---:|---:|
| `cannot_answer → tool_call` | **0.9387** | **0.9260** | *not testable* |
| `cannot_answer → direct` | **0.9329** | **0.9190** | *not testable* |
| `request_for_info → tool_call` | **0.7700** | **0.7067** | **0.7270** |

Across three architectures and two model families, `cannot_answer → *` is strongly readable
(0.92–0.94) and `request_for_info → tool_call` is marginal-to-unreadable (0.71–0.77). Qwen3-8B's
`rfi→tc` cleared the gate by 0.02 and was still not the channel its protocol selected for formal
evaluation.

**This is the most reproducible quantity in the entire panel.** It is a statement about the
error channel, not about any model.

## Baseline topology

| | Qwen3-8B | Gemma-2-9B | Qwen3.5-9B |
|---|---:|---:|---:|
| non-sealed n | 3104 | 3098 | 3104 |
| accuracy | 0.4224 | 0.4245 | **0.4533** |
| max predicted-class share | 0.7126 | 0.7960 | **0.6633** |
| recall `cannot_answer` | 0.108 | 0.105 | **0.087** |
| recall `request_for_info` | 0.170 | 0.132 | **0.326** |
| recall `tool_call` | 0.944 | 0.984 | 0.924 |
| reference pool `cannot_answer` | 118 | 115 | **95** |
| reference pool `request_for_info` | 154 | 119 | **295** |
| support-eligible channels | 3 | 3 | **1** |

## Margins — Qwen3.5 errors are the closest to flipping

Median baseline `gold − source` score on channel-error rows (less negative = nearer the boundary):

| channel | Qwen3-8B | Gemma-2-9B | Qwen3.5-9B |
|---|---:|---:|---:|
| `request_for_info → tool_call` | −1.2296 | −0.7940 | **−0.4947** |
| `cannot_answer → tool_call` | −1.3454 | −0.8550 | **−0.7104** |

On the naive expectation that near-boundary errors are the easiest to repair, Qwen3.5 should
have been the most favourable setting in the panel. It never reached a correctability test.

## Terminal outcomes, and the failure stage of each

| setting | verdict | first failing stage |
|---|---|---|
| Qwen3-8B × `ca→tc` | **FORMAL ADMIT** (10/10) | none |
| Qwen3-4B × `ca→tc` | **FORMAL DECLINE** (8/10) | destination intervals — CI, not effect |
| Gemma-2-9B × `ca→tc` | **FORMAL DECLINE** (8/10) | destination intervals — CI, not effect |
| Qwen3.5-9B | **NO ACTIONABLE CHANNEL** | support (`ca→*`), then readability (`rfi→tc`) |

Effect magnitude, evidence precision and formal verdict must be kept apart. Qwen3-4B and Gemma
both produced positive point effects (Target Gain +0.0806 and +0.0729, target-hit 0.578 and
0.630) and both were refused because the intervals could not exclude the thresholds. Neither is
a statement that the intervention did nothing.

## What Qwen3.5 does and does not add

**Adds:** a third prospective host-model instantiation; a cross-family replication of the
`rfi→tc` readability failure; a demonstration that the support gate refuses a preferred channel
by one row.

**Does not add:** any evidence about correctability in Qwen3.5, any generational trend, any
scaling claim, any architecture-causation claim, and nothing whatsoever about cross-benchmark
generalisation.
