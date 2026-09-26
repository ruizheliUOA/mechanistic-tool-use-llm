# Final formal status table — locked

Frozen hierarchy, not collapsed:
`Adjudicable → Readable → Steerable → Correctable → Preserving → Licensable`

## Modern, prospectively evaluated (sealed protocol)

| model | observed effect | Correctable | Preserving | Licensable | **formal verdict** |
|---|---|---|---|---|---|
| **Qwen3-8B** | 38 gold / 52 exits, t-hit **0.7308** [0.6042, 0.8462] | **YES** — arrivals concentrate on gold (38 vs 14) | **YES on E1** (0/211, CP upper 0.0141); **VACUOUS on E2** (n=6) | **YES** | **`FORMAL_CONFIRMATORY_SUCCESS`** |
| **Qwen3-4B** | 37 gold / 64 exits, t-hit **0.5781**, CI lower **0.456** | **descriptively yes** — gold 37 > other 27 | **YES** — E2 1/50 = 2.0% | **NO** — interval crosses 0.50 | **`QWEN3_4B_FORMAL_DECLINE`** |
| **Gemma-2-9b** | 17 gold / 27 exits, t-hit **0.6296**, CI lower **0.444** *(frozen artifact; corrected 2026-09-11 from the recomputed 0.440)* | **descriptively yes** — gold 17 > other 10 | **NO** — E2 1/11 = 9.1%, weakly powered | **NO** — fails conditions 3 and 6 | **`GEMMA_FORMAL_DECLINE`** |

## Historical (post-hoc protocol)

| model | observed effect | Correctable | Preserving | Licensable | status |
|---|---|---|---|---|---|
| **Phi-3.5-mini** | Net **+55**, t-hit 0.6993 | yes — 107 gold vs 46 other | **NO** — E2 52/93 = 55.9% | not adjudicated (historical) | Steerable; **preservation failure** |
| **Qwen2.5-7B** | +79 real vs +16 reverse | **undefined** on MetaTool (binary) | not adjudicated | not adjudicated | Steerable |
| **Llama-3.1-8B** | 19 gold vs **44 other-wrong** | no — arrivals do not concentrate | — | — | fails **Steerable** (5/20, 3/20 randoms ≥ real) |
| **Mistral-7B** | real +12 < random mean +29 | — | — | — | fails **Steerable** (17/20 randoms ≥ real) |

## The dissociation this table exists to show

**Correctable and Licensable are different properties, and two models sit exactly
in the gap.** Qwen3-4B and Gemma show arrivals concentrating on gold — the
Correctable property — yet the evidence is not precise enough to license the
claim, with 64 and 27 exits respectively.

**Preserving is a fourth, independent property.** Gemma is descriptively
Correctable and **not** Preserving; Qwen3-4B is descriptively Correctable **and**
Preserving, and still not Licensable. The three fail independently, exactly as the
frozen definitions specify.

## Binding language

- `DECLINE ≠ intrinsically uncorrectable` — the evidence is insufficient under this
  protocol and sample support, nothing more.
- `historical ADMIT ≠ preservation-certified safe`.
- `Licensable` is a property of the **evidence**, not of the model or setting.
- Never describe Qwen3-4B or Gemma as formally admitted (S-20).
- Never use "Level-1"/"Level-2" as licence names (S-21).
