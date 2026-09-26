# SAKIKO — full-model failure taxonomy and licence root-cause synthesis

Every row is a `(M, D, c, I)` unit that reached adjudication. `D` = When2Call throughout.
All figures recomputed from committed per-row artifacts.

## The ladder, and where each unit stopped

```
SUPPORT → READABILITY → DIRECTION → SPECIFICITY → DESTINATION → COLLATERAL → PRECISION → ADMIT
```

| # | setting × channel | terminal outcome | first failing rung | the number that decided it |
|---|---|---|---|---|
| 1 | Qwen3-8B × `ca→tc` | **FORMAL ADMIT** | — (10/10) | target-hit 0.7309, TG +0.2759, 0/59 randoms, p 0.0167 |
| 2 | Qwen3-4B × `ca→tc` | **FORMAL DECLINE** | **PRECISION** | TG CI lower **−0.0484**; target-hit CI lower **0.4545** |
| 3 | Gemma-2-9B × `ca→tc` | **FORMAL DECLINE** | **PRECISION** | TG CI lower **−0.0327**; target-hit CI lower **0.4475** |
| 4 | Phi-3.5 × 3 channels | **WOULD-DECLINE** (retrospective) | **COLLATERAL** | 52/264 = 0.1970, 3.9× the bound |
| 5 | Gemma-2-9B × `ca→direct` | **DEV DECLINE** | **COLLATERAL (vacuity)** | **0** exposed baseline-correct rows |
| 6 | Qwen3-4B × `ca→direct` | **DEV DECLINE** | **COLLATERAL (vacuity)** | eligible-at-risk n = 0 |
| 7 | Gemma-2-9B × `rfi→tc` | **READABILITY DECLINE** | **READABILITY** | DEV ROC-AUC 0.7067, CI [0.6666, 0.7456] |
| 8 | Qwen3.5-9B × `rfi→tc` | **READABILITY DECLINE** | **READABILITY** | DEV ROC-AUC 0.7270, CI [0.6865, 0.7664] |
| 9 | Qwen3.5-9B × `ca→*` (3) | **NO SUPPORT** | **SUPPORT** | DEV reference **29** vs minimum **30** |

**Nine adjudicated units. One ADMIT. Eight refusals, distributed over four distinct rungs.**

## Root-cause classes

### A. Precision-limited (units 2, 3) — the dominant formal failure mode

Both formal DECLINEs have **positive point effects that survive every specificity control**
and fail only on interval width.

| | Qwen3-4B | Gemma-2-9B |
|---|---:|---:|
| target-hit point | 0.5781 | 0.6296 |
| Target Gain point | +0.0806 | +0.0729 |
| randoms ≥ real (K=59) | **0** | **0** |
| add-one p | 0.0167 | 0.0167 |
| channel errors on SEALED | 124 | **96** |

Gemma's DEV→SEALED comparison isolates the mechanism: target-hit moved 0.6667 → 0.6296 and
Target Gain +0.0881 → +0.0729 — essentially unchanged — while n halved from 193 to 96 and the
95% half-width grew from 0.072 to 0.106. Clearing zero at that effect size needs ~199 channel
errors; SEALED supplied 96.

**Root cause: the confirmatory partition is roughly half the size of the development
partition.** The V2 allocation moved rows into DEV (1104) to cure reference starvation, leaving
SEALED at 548. That made development well-powered and confirmation under-powered. This is a
protocol design property, not a model property, and it is the single most actionable finding in
the taxonomy.

### B. Collateral-limited (unit 4) and collateral-vacuous (units 5, 6)

Two different failures that must not be conflated:

- **Unit 4, real breakage.** Phi's 52/264 = 19.7% is 3.9× the bound, and 52/93 = 55.9% on the
  exposed denominator — worse on both. The number was recorded in 2026 and simply not treated
  as decisive.
- **Units 5 and 6, vacuity.** Both have `0` exposed baseline-correct rows, so their 0/N
  collateral could not have failed. Both had the *stronger* destination evidence of their
  respective models — Gemma's `ca→direct` reached target-hit 0.8636 and TG +0.1860, passing 13
  of 14 conditions. They are refused because an untestable safety criterion is not a passed one.

The same structure appeared independently in two models, which is why condition 11 exists.

### C. Readability-limited (units 7, 8) — a channel property, not a model property

| channel | Qwen3-8B | Gemma-2-9B | Qwen3.5-9B |
|---|---:|---:|---:|
| `cannot_answer → tool_call` | 0.9387 | 0.9260 | not testable |
| `cannot_answer → direct` | 0.9329 | 0.9190 | not testable |
| `request_for_info → tool_call` | **0.7700** | **0.7067** | **0.7270** |

Across three architectures and two families, `ca→*` is strongly readable and `rfi→tc` is
marginal-to-unreadable. Qwen3-8B's `rfi→tc` cleared the gate by 0.02 and was still not the
channel its protocol selected. **This is the most reproducible quantity the project produced,
and it is a statement about the error channel.**

### D. Support-limited (unit 9)

`cannot_answer → *` in Qwen3.5 fails on DEV reference support **29 vs 30**. Reference and error
rows within a gold class are complementary and compete for a fixed row budget; Qwen3.5's
`cannot_answer` recall of 0.087 starves the reference side. The same mechanism produced
`NO_SUPPORT_ELIGIBLE_CHANNELS` for Qwen3-8B under the V1 allocation.

**Root cause: outcome-conditioned reference estimation is feasible only in an intermediate
accuracy band.** Too accurate on a class and the error side starves; too inaccurate and the
reference side starves.

## What the taxonomy shows about the licence

1. **The licence is binding, and binds in different places.** Eight refusals across four
   distinct rungs is not a protocol that rubber-stamps. It refused its authors' own prior
   headline result (unit 4) and refused the strongest destination evidence in the panel (unit 5).
2. **No refusal was rescued.** The one-row support miss (unit 9) is the sharpest test — moving
   the gate by one row would have opened the channel that carried the project's only ADMIT.
   It was not moved.
3. **Effect, precision and verdict are three different things.** Units 2 and 3 have real,
   direction-specific, positive effects and are correctly refused. A DECLINE is a statement
   about evidence sufficiency, never a claim that the intervention did nothing.
4. **The failures are not concentrated in weak models.** Qwen3.5 has the best accuracy, least
   collapse and errors nearest the boundary in the panel, and it terminated earliest.

## Concentration risk, stated plainly

The single ADMIT is one `(M, D, c, I)` unit: Qwen3-8B × When2Call × `cannot_answer → tool_call`
× gradient-direction activation addition at L21. Everything else refuses. The existential claim
is real and prospectively earned, but it rests on one cell, and both replication attempts at the
same channel failed on interval width rather than on effect.

## The two changes this taxonomy actually motivates

Recorded as findings, not as authorised work:

1. **The confirmatory partition is too small relative to development.** Two of three formal
   attempts failed on precision alone with effects intact. Any future prospective design should
   size SEALED from a power calculation at the DEV effect size, not from a fixed 548-row legacy.
2. **Collateral exposure must be projected before sealing.** Three units reached the collateral
   rung with 0, 6, 11 or 23 exposed rows. Exposure should be a pre-registration quantity, not a
   post-hoc discovery.
