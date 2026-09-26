# The expanded SAKIKO ladder — audit

Proposed: `Adjudicable → Readable → Steerable → Correctable → Preservable → Certifiable`.

Each rung is tested for a precise definition, a distinct estimand, a real empirical example,
and a non-redundant role.

| rung | estimand | frozen test | real example | non-redundant? |
|---|---|---|---|:-:|
| **Adjudicable** | channel error and reference support | TRAIN ≥60/≥60, DEV ≥30/≥30 | **Qwen3.5 `ca→*`** — DEV reference 29 vs 30 | **YES** — nothing downstream can be posed without it |
| **Readable** | DEV comparable Router ROC-AUC | ≥ 0.75 with τ-precision ≥ 0.50 | **Gemma & Qwen3.5 `rfi→tc`** — 0.7067, 0.7270 | **YES** |
| **Steerable** | direction-specificity vs matched randoms and reverse | real exceeds every random; reverse does not reproduce | **Mistral `rfi_tc`** — real +12 < random mean +29, reverse +19 > real, 17/20 randoms ≥ real; **Llama locked TEST** — 5/20 and 3/20 randoms ≥ real | **YES** |
| **Correctable** | destination composition: gold vs other-wrong among exits | target-hit > 0.50, gold > wrong-to-wrong | **weakly instantiated** — see below | **QUALIFIED** |
| **Preservable** | collateral on exposed baseline-correct rows | ≤ 0.05 point and CI | **Phi-3.5** — 52/264 = 19.7%, 52/93 = 55.9%; **Gemma & 4B `ca→direct`** — vacuous, 0 exposed | **YES** |
| **Certifiable** | lower confidence bounds on the destination endpoints | TG CI lower > 0, t-hit CI lower > 0.50 | **Qwen3-4B and Gemma `ca→tc`** | **YES** — and it is the *only* rung that has ever decided a formal case |

## The one rung that is weakly supported

**Correctable** — as a *separately failing* rung — has no clean example. The candidate was
Llama's destination redistribution (`ca→tc` locked TEST: 19 gold vs 44 other-wrong, target-hit
≈ 0.30), which looks exactly like "moves but misdirected." But **Llama failed specificity first**
(5/20 randoms ≥ real, z=0.879), so it never established steerability and cannot demonstrate a
rung above it. The redistribution is descriptively real and scientifically interesting; it is
not a licensed example of "steerable but not correctable."

At the formal stage, the destination *point* conditions (2, 5) have never failed while the
destination *interval* conditions (3, 6) failed twice. So in practice Correctable and
Certifiable have not been observed to separate — only the certification half has ever bound.

## Verdict on the ladder

**Five of six rungs are supported with distinct estimands and real examples.** Adjudicable,
Readable, Steerable, Preservable and Certifiable each have a case where they and only they
decided the outcome. **Correctable is currently a definitional rung rather than an empirically
separated one**, and the paper should say so rather than implying six independently
demonstrated stages.

That is still a strict improvement on `Readable → Steerable → Correctable`, which collapses
adjudicability, preservation and certification into one word.
