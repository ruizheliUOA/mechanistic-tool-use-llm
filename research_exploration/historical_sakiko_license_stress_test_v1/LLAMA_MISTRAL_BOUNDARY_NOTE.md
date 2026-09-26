# Llama and Mistral — claim boundary, not re-scoring

Neither is retro-scored. Required fields are absent and no inference was run to create them.

## Mistral-7B-v0.3

| item | status |
|---|---|
| per-sample records | baseline only (`gold`, `pred`, `correct`, …); **no intervened prediction field** |
| rescorable? | **No** |

Historical conclusion, preserved in its own units: 3 channels discovered, **0 passed the
utility gate**; 2 of 3 not direction-specific (`rfi_tc`: 17 of 20 randoms ≥ real); the one
specific channel (`ca_tc`) had catastrophic ungated behaviour (Net −97, 191 broken). Raw Net
rose to +82 while the specific component collapsed ~4.5×.

Use as: *readable and discoverable, but the causal-specificity claim does not transfer.*

## Llama

| item | status |
|---|---|
| per-sample records | **not present** |
| rescorable? | **No** |

**Correction preserved from the sufficiency audit.** Llama is *not* a clean formal
"steerable but misdirected" rung. On the locked test the gate **rejected all three channels**,
and the two audited rejections were both **DENIED** (`ca_tc` z = 0.879, 5/20 randoms ≥ real;
`ca_direct` z = 1.256, 3/20). The "specificity without correction" pattern — ~35 exits with
~8 reaching gold — was observed at **validation**, and did not survive as direction-specific
on the locked test.

Use as: *evidence that a prospective gate can reject correctly* — 0 over-admissions,
0 over-rejections. **Do not overstate Llama in the paper** as a destination-failure rung.

## Why this matters for the manuscript

The failure taxonomy must not be presented as four clean rungs measured in one metric
language. Only Qwen3-8B, Qwen3-4B and (for collateral) Phi-3.5 are expressible in modern
terms. Mistral and Llama belong in a clearly demarcated older-protocol corroboration block.
Presenting them otherwise invites the accurate criticism that incomparable protocols were
merged.
