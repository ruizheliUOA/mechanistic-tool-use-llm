# How hardcoded is the implementation to When2Call?

Audited from source, not memory.

| file | literal W2C mode names | `MODES` symbol uses |
|---|---:|---:|
| `scripts/qwen3_8b_stage0_1.py` | 41 | 29 |
| `stage_d_e_common.py` | 8 | 1 |

`MODES` has **one definition point** (`qwen3_8b_stage0_1.py:55`). The core scientific functions
are already generic over labels:

- `channel_name(gold, source)` — parameterised
- `geometry_metrics(matrix, direction, gold, source, variant)` — parameterised
- destination classification — `SOURCE_RETAINED if pred==source else GOLD_ARRIVAL if pred==gold
  else OTHER_WRONG` — **fully generic over any label set**
- `build_channel_ledger` — iterates `for gold in MODES: for pred in MODES`, generic over `K`

**The algorithmic core is already action-space agnostic.** The 41 literal mentions are
concentrated in prompt construction, the `answers[mode]` candidate lookup, and the thinking-mode
readout assertion — i.e. the **instrument**, not the science.

## What generalization actually requires

Not a rewrite. Three substitutions at the instrument boundary:

1. `MODES` → `A_D` supplied by the adapter.
2. `sample["answers"][mode]` → `adapter.candidate_text(sample, action)`.
3. the Qwen-specific thinking-suffix assertion → `adapter.validate_readout(prompt)`.

The frozen historical code is not modified. This is a specification for a successor
implementation.

## Verdict

```
CORE IS GENERIC; INSTRUMENT IS W2C-SPECIFIC
```

The claim "SAKIKO is a When2Call checklist" is **false at the level of the algorithm** and
**true at the level of the current harness**. That distinction is defensible in print and is
cheap to demonstrate, because the destination classifier — the conceptually distinctive part —
never references a W2C label at all.
