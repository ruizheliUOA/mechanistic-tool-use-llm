# RUT-Bench deep screen — result

Data opened directly: `Miaow-Lab/RUT-Bench`, `RUT-Bench.jsonl`, 38.15 MB, ungated,
**all 1638 samples inspected** (not a sample, not the abstract).

## Verdict

```
CATEGORY D — NOT SUITABLE
```

## The decisive facts

| field | value |
|---|---|
| `task_type` | **`execute` for 1638 / 1638** |
| `micro_tasks.task_type` | **`execute` × 2785** — every subtask |
| `refusal_requirements` | **non-empty in 0 / 1638** — field exists, never populated |
| `forbidden_actions` | empty in the inspected rows |

**`|A_D| = 1`.** There is no multiclass first-action gold. Every task's correct action is
"execute the tool trace." Adapter contract requirement 2 (`|A_D| ≥ 3` with a meaningful
alternative beyond `{g,s}`) fails outright, which means `Correctable` is undefined here for the
same structural reason it is undefined on MetaTool.

## What RUT-Bench actually varies

Not the action, the **user**: 1403 unstable vs 235 stable, across six behaviour categories at
~233 samples each — `information_overload`, `contradictory_constraints`, `underspecification`,
`impatience_and_hostility`, `goal_switching`, `fabricated_parameters`. 824 single-turn,
814 multi-turn, 59 environments.

It is a benchmark about **executing correctly despite messy users**, not about **choosing among
actions**. Those are different scientific objects.

## The tempting opening, and why I am not taking it

`underspecification` (234) and `fabricated_parameters` (232) are exactly the conditions under
which a good agent should **ask** rather than **execute** — 466 samples of genuine ask-vs-act
tension. But the benchmark labels them `execute`.

Constructing an ask/execute gold over those rows would mean **inventing labels the dataset does
not contain**, which is the definition of Category D under the adapter contract and precisely
the failure mode arXiv 2607.02577 documents across the field. It would manufacture the second
benchmark rather than find one.

If RUT-Bench's own evaluation code adjudicates clarification behaviour (its "informational
honesty" and "tool discipline" metrics suggest it may), that adjudication lives in the harness,
not in per-sample gold — so it cannot supply a deterministic first-action label either.

## Cost of this screen

~15 minutes, CPU and network only. Nothing consumed, no formal population touched.
