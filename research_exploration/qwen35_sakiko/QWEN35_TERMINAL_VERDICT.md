# Qwen3.5-9B — terminal developmental verdict

**SEALED never accessed. No intervention was run. No formal freeze constructed.**

## Verdict

```
QWEN35_NO_ACTIONABLE_CHANNEL
```

The chain breaks in two distinct places, and both are structural rather than about
correctability:

- **SUPPORT** for every `cannot_answer → *` channel, including the exact channel that carried
  the Qwen3-8B formal ADMIT and Gemma's formal candidate.
- **READABILITY** for the single channel that did pass support.

## Where exactly it breaks

### 1. Support — the informative channels are not testable

| channel | TRAIN err | TRAIN ref | DEV err | DEV ref | eligible |
|---|---:|---:|---:|---:|---|
| `request_for_info → tool_call` | 365 | 195 | 207 | 100 | **YES** |
| `cannot_answer → tool_call` | 300 | 66 | 170 | **29** | no — `DEV_CORRECT_REFERENCE_LT_30` |
| `cannot_answer → direct` | 275 | 66 | 160 | **29** | no — same |
| `cannot_answer → request_for_info` | 66 | 66 | 31 | **29** | no — same |
| (8 others) | ≤25 | — | ≤26 | — | no — error support |

`cannot_answer → tool_call` has **abundant** error support (300 TRAIN / 170 DEV) and fails
purely on the *reference* side, by **one row**: 29 against a frozen minimum of 30.

**The threshold was not lowered and will not be.** A gate that moves when it is one row from
admitting a preferred channel is not a gate. This is the outcome-conditioned feasibility window
the project already documented for Qwen3-8B V1: within a gold class, reference and error rows
are complementary and compete for a fixed number of rows. Qwen3.5 answers `cannot_answer`
correctly only **8.7%** of the time — the lowest of the three models — which starves the
reference side of every `ca → *` channel.

### 2. Readability — the one supported channel fails

`request_for_info → tool_call`: TRAIN AUC 0.9999 (in-sample, not load-bearing),
**DEV ROC-AUC 0.7270, 95% CI [0.6865, 0.7664]**, PR-AUC 0.5131. The gate is 0.75 and the
**entire interval lies below it**. At τ=0.4 precision is 0.5021 — barely over the 0.50 floor —
firing on 119 channel errors, 121 baseline-correct rows and 93 off-channel errors. Population
purity is poor.

**`READABILITY_ROUTER_DECLINE`.**

### 3. A genuine cross-model replication

Gemma-2-9B's `request_for_info → tool_call` failed the identical gate at **0.7067
[0.6666, 0.7456]**. Two unrelated architectures, the same channel, the same failure, overlapping
intervals. `rfi → tc` looks intrinsically less linearly readable at the observation site than
`cannot_answer → *`, which reached 0.9260 and 0.9190 in Gemma.

## Why Qwen3.5 differs — baseline topology, not intervention

| | Qwen3-8B | Gemma-2-9B | **Qwen3.5-9B** |
|---|---:|---:|---:|
| accuracy | 0.4224 | 0.4245 | **0.4533** |
| max predicted-class share | 0.7126 | 0.7960 | **0.6633** |
| recall `cannot_answer` | 0.108 | 0.105 | **0.087** |
| recall `request_for_info` | 0.170 | 0.132 | **0.326** |
| reference pool `ca` | 118 | 115 | **95** |
| reference pool `rfi` | 154 | 119 | **295** |
| median gold−source, `rfi→tc` | −1.2296 | −0.7940 | **−0.4947** |
| median gold−source, `ca→tc` | −1.3454 | −0.8550 | **−0.7104** |

Qwen3.5 is the **most accurate and least collapsed** of the three, and its errors sit
**closest to the decision boundary**. On the naive expectation that near-boundary errors are
easier to correct, it should have been the most promising setting. It never reached the test,
because the improvement is unevenly distributed: `request_for_info` recall nearly doubles
(0.326) while `cannot_answer` recall *falls* to 0.087. That redistribution moves the testable
channel from `ca → *` to `rfi → tc` — and `rfi → tc` is precisely the channel that is not
readable.

**This is a topology outcome, not evidence about correctability.** Nothing here says Qwen3.5 is
uncorrectable. It says the frozen protocol has no admissible channel to test in this model.

## What was NOT done, and why

No direction estimation, no estimator×site control, no DEV intervention, no dose curve, no
destination or collateral accounting, no adjudicability gate, no formal freeze. The frozen stop
rule terminates the study when the Router validity gate fails. Reporting any of those stages
would mean running them on a channel the protocol has already declined.

## Claim boundary

Permitted: *the modern SAKIKO ladder was instantiated on a newer Qwen generation and terminated
before intervention because no channel was simultaneously support-eligible and readable.*

Forbidden: that Qwen3.5 is uncorrectable; that correctability declines across Qwen generations;
that scale or architecture causes this; any cross-benchmark or cross-action-space claim. The
formal evidence base remains When2Call only, and this result does not change that.
