# Random-control wording — correction required in the manuscript

## Verified from primary records

`QWEN3_STAGE2_FORMAL_RECORDS.jsonl`, 59 random arms, channel `cannot_answer→tool_call`:

| model | random gold arrivals | random exits | arms with ≥1 gold | pooled random target-hit |
|---|---:|---:|---:|---:|
| **Qwen3-8B** | **126** | 393 | **47 / 59** | **0.3206** |
| **Gemma-2-9B** | **0** | 19 | **0 / 59** | 0.000 |

## The wording that must not be used

> ~~"random directions produced no gold arrivals"~~

For **Qwen3-8B this is false.** Its random directions produced 126 gold arrivals
across 47 of 59 arms. Stating otherwise as a general claim would be a factual
error in the paper.

## The correct wording

> **0 of 59 matched-random arms exceeded the real intervention on the frozen
> Target Gain Rate statistic** (`randoms_ge_real = 0`, add-one p = 0.0167).

That is the preregistered comparison, and it holds for both models.

## What the numbers actually mean

Random target-hit of **0.3206** for Qwen3-8B is close to **1/3** — chance, once
the source action is excluded from a four-way space. The correct reading:
**random directions move behaviour, but their destination is uninformative.**
The real intervention reaches 0.7308.

Gemma's randoms differ in kind, not degree: they produce almost no movement at
all (19 exits pooled across 59 arms, ~0.32 exits per arm), and none reach gold.

So the two models fail the null in different ways, and the paper should say so
rather than flattening them:

- **Qwen3-8B**: randoms cause movement at chance destination; the real direction
  is destination-selective.
- **Gemma**: randoms cause almost no movement; the real direction is both
  movement-inducing and destination-selective.

## Statements I made this session, re-checked

- "Gemma's 59 random directions produced 0 gold arrivals across 19 exits" —
  **verified true**, and correctly scoped to Gemma.
- I did **not** generalise it to Qwen3-8B. But the contrast was never stated,
  which risks a reader inferring it. The manuscript must carry the Qwen3-8B
  figure explicitly.

## Also verified in the same pass

**Canonical adapter regression: PASS.** Reading Qwen3-8B's transition-format
records (`tool_call->cannot_answer`) through the adapter reproduces the frozen
headline exactly — gold **38**, exits **52**, other-wrong **14**. A naive
categorical filter on that file returns zero gold arrivals for every arm; the
adapter is doing real work.

**Preservation 0/6 confirmed at record level.** 6 baseline-correct rows in the
formal records, all Router-fired, 0 breaks. Matches `PRESERVATION_STATUS_FINAL`.

**Denominator note.** The adapter finds **72** routed channel errors where the
frozen file reports `n_channel_error = 87`. Both are correct: the records contain
only Router-fired rows, the frozen count includes unrouted ones. The same pattern
holds for Qwen3-4B (118 routed vs 124 total). **Target-hit is computed on exits,
which match exactly (52), so no endpoint is affected** — but any future analysis
must state which denominator it uses.
