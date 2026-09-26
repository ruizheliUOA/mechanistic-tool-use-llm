# The second benchmark instantiation already exists — and is mis-filed as a dead end

`final/results/acebench_generation_readout/` is currently recorded as **Outcome D — DATASET
NO-GO**, retired from the intervention line. For the *intervention* claim that is correct. For
the **framework-transfer** claim it is the evidence the paper is missing.

## ACEBench is a genuinely different action space

| | When2Call | **ACEBench** |
|---|---|---|
| action ontology | `tool_call`, `direct`, `request_for_info`, `cannot_answer` | **`tool_call`, `ask_user`, `cannot_comply`, `flag_param_error`** |
| readout mechanism | candidate scoring (4 teacher-forced forwards) | **generation-based + rule parser** |
| gold source | benchmark-native MCQ | benchmark-native, different construction |

`flag_param_error` has **no When2Call analogue**. The measurement mechanism is different too —
generation plus parsing rather than candidate scoring — so this is not the same instrument
pointed at new data.

## SAKIKO ran end-to-end and stopped interpretably

Criteria R0.1–R0.10 were **locked before the run** (`READOUT_LOCK_MANIFEST.json`, 8 sha256).

| result | value |
|---|---|
| accuracy | **0.7562** |
| macro-F1 | **0.6868** |
| balanced accuracy | 0.6205 |
| max predicted-class share | **0.675** — *better balanced than any W2C model* |
| modes ≥5% of predictions | **4 / 4** |
| gold classes with recall ≥0.10 | **4 / 4** (0.892 / 0.420 / 0.420 / 0.750) |
| replay determinism | **100%** (50/50) |
| parser mislabel | 0.98% strict (102-row human audit) |
| **R0.9 channel support** | **FAIL — 0 of ≥2 transitions clear train≥30 / val≥5 / test≥5** |

**9 of 10 prospectively locked criteria pass.** Channel discovery ran and found real directed
channels — `ask_user→tool_call` (40), `cannot_comply→ask_user` (22), `tool_call→ask_user` (7),
`flag_param_error→ask_user` (6) — and the support gate refused all of them.

The stop cause is structural and was **predicted pre-hoc**: ACEBench special modes hold only 100
examples each, so per-split support is arithmetically unreachable.

## What this licenses the paper to claim

> The SAKIKO ladder was instantiated prospectively on a **second, independent action ontology**
> with a **different measurement mechanism**, under criteria locked before execution. The
> measurement layer transferred (9/10 criteria, 4/4 classes represented, 100% replay
> determinism); channel discovery transferred (4 directed channels found automatically); and the
> ladder terminated at the **adjudicability** rung for a structural reason the protocol had
> predicted in advance.

That is **framework generalisation without an intervention claim** — precisely the
`Adjudicable` rung doing its job on data it was never tuned on. It answers Reviewer D's
external-validity attack without requiring a second ADMIT.

## The independent scientific finding buried in the same package

The pre-registered paraphrase check **failed** (0.56 vs an 0.80 gate). The reason is
substantive: **restraint modes are wording-fragile while calling is stable** — `ask_user` 0.36
versus calling 0.84, and a semantically equivalent rewording moves the model **+22 points toward
calling**.

That is a construct-validity result about tool-use benchmarks in general, obtained under a
pre-registered criterion, and it sits alongside arXiv 2607.02577's finding of 9.8–30.5%
evaluator error across four major benchmarks. It is currently reported as a reason to abandon
ACEBench rather than as a finding.

## Honest limits

- ACEBench is a prior SAKIKO development corpus. It **cannot** support a confirmatory licence
  outcome, and no transfer claim about correctability may be made from it.
- The claim is about the **framework instantiating and stopping**, not about any intervention.
- The support failure is a property of ACEBench's size, not evidence about correctability
  anywhere.

## Recommendation

Promote this from "DATASET NO-GO" to a **second-benchmark framework-transfer section**. The
work is done, prospectively locked, committed and hash-verified. What is missing is the framing.
