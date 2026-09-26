# Single-benchmark claim scope

## Construct validity — is When2Call the right object?

| requirement | When2Call | verdict |
|---|---|---|
| multiclass first-action decision | 3 gold classes (`cannot_answer`, `request_for_info`, `tool_call`) + `direct` as distractor | **YES** |
| meaningful gold actions | human-curated, benchmark-native | YES |
| wrong-to-wrong transitions possible | \|A\\{g,s}\| ≥ 2 | **YES** — the property MetaTool lacks |
| support/reference populations definable | outcome-conditioned, both sides observed | YES |
| Router false-positive exposure measurable | yes, though often tiny (0, 6, 11, 23 rows) | **PARTIAL** |
| destination-resolved outcomes | yes | YES |

**Construct validity holds.** One benchmark is therefore sufficient for *existence* claims.

## What one benchmark can and cannot support

| claim | 1 benchmark? | why |
|---|:-:|---|
| There exists a setting where a correction proposal satisfies the licence | **YES** | existence |
| Aggregate metrics can hide wrong destinations | **YES** | one counterexample suffices (Llama, and the score-space arms) |
| The licence can refuse a conventionally successful intervention | **YES** | Qwen3-4B, Gemma, Phi |
| The failure taxonomy occurs across models | **YES within W2C** | 9 units, 6 rungs, 4 families |
| The licence is applied consistently under prospective discipline | **YES** | one-shot sealed, hash-frozen |
| SAKIKO generalises across tool-use tasks | **NO** | needs a second action space |
| Licence thresholds transfer across datasets | **NO** | thresholds are plausibly W2C-calibrated |
| Correctability structure is benchmark-invariant | **NO** | channel/benchmark perfectly confounded |
| `rfi→tc` is intrinsically less readable | **NO** | confounded with W2C's ontology |

## Is one benchmark fatal to any claim the paper makes?

**No — provided the paper claims only the rows marked YES.** Every NO-row claim must be absent.

The sharpest residual risk is not the existence claim; it is that **channel identity and
benchmark are perfectly confounded**, so the "channel-dependent readability" observation
(0.92–0.94 for `ca→*` vs 0.71–0.77 for `rfi→tc`, three models) may be a When2Call ontology
property presented as a channel property. That claim must stay at `SUGGESTIVE`.

## Verdict

```
SERIOUS_BUT_NONFATAL_LIMITATION
```

Fatal to generalisation claims. Not fatal to the existence, refusal and taxonomy claims — which
is what the paper actually rests on.
