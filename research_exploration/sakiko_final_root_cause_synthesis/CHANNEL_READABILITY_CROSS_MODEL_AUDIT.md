# Cross-model channel-readability audit

## The pattern

DEV comparable Router ROC-AUC, identical Router family and hyperparameters, gate 0.75:

| channel | Qwen3-8B | Gemma-2-9B | Qwen3.5-9B |
|---|---:|---:|---:|
| `cannot_answer → tool_call` | 0.9387 | 0.9260 | not testable |
| `cannot_answer → direct` | 0.9329 | 0.9190 | not testable |
| `request_for_info → tool_call` | **0.7700** | **0.7067** | **0.7270** |

## Testing the strong claim

I previously wrote that "readability is a property of the error channel, not the model."
**That is too strong.** Auditing it against the required checks:

| check | finding |
|---|---|
| observation sites comparable in normalized depth? | **YES** — all at normalized depth 0.7407 under the same frozen rule (L26/L30/L23) |
| Router family identical? | **YES** — StandardScaler + LogisticRegression, C=1.0, liblinear, seed 42, identical τ grid |
| reference/error definitions identical? | **YES** — same frozen definitions |
| support sizes comparable? | **NO** — `rfi→tc` DEV positives 244 / 261 / 207; `ca→tc` 167 / 224 / n-a. Not matched. |
| AUC uncertainties overlapping? | **YES for the three `rfi→tc` values**: [n/r], [0.6666,0.7456], [0.6865,0.7664] overlap heavily. **NO between `rfi→tc` and `ca→*`**: 0.71–0.77 vs 0.92–0.94, non-overlapping by a wide margin. |
| counterexamples? | **YES, one that matters** — Llama-3.1-8B's `rfi_tc` Router reached val AUC **0.893** at L13 and 0.855 at L18, far above the Qwen/Gemma `rfi→tc` band. Different protocol generation and site, so not directly comparable, but it is a genuine violation of a strict channel-only account. |
| could When2Call itself create the hierarchy? | **YES, and this is not separable.** All four models are evaluated on one benchmark. `request_for_info` gold rows may simply be more heterogeneous *in this dataset* — a dataset property presenting as a channel property. |

## Verdict

```
SUGGESTIVE_CHANNEL_ASSOCIATION
```

Not `STRONG`. Three models agree closely on `rfi→tc` (0.71–0.77, overlapping intervals) and two
agree closely on `ca→*` (0.92–0.94), and the separation between the two groups is large and
non-overlapping. But support sizes are unmatched, Llama contradicts the pattern under an older
protocol, and — decisively — **channel identity and benchmark are perfectly confounded**, since
every measurement comes from When2Call.

The defensible phrasing is:

> Across the models tested on When2Call at matched normalized depth with an identical Router
> family, error-channel identity accounted for readability differences more consistently than
> model identity did.

The word **intrinsic** is not justified and is not used.

## What would separate the hypotheses

A second multiclass pre-execution benchmark with the same four-mode structure. If `rfi→tc`
remained the weakest channel there, the channel account would strengthen considerably; if the
ordering changed, it was a When2Call property throughout. No such benchmark is currently
eligible.
