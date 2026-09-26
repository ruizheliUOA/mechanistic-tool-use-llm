# Conjunction bottleneck audit

## The frozen 10 conditions

1 support ≥ 30 · 2 Target Gain > 0 · 3 TG CI lower > 0 · 4 real beats K=59 random null ·
5 target-hit > 0.50 · 6 target-hit CI lower > 0.50 · 7 collateral point ≤ 0.05 ·
8 collateral CI upper ≤ 0.05 · 9 zero reproduces baseline exactly · 10 no structural failure.

## Observed failure incidence across the three formal trials

| condition | Qwen3-8B | Qwen3-4B | Gemma | times binding |
|---|:-:|:-:|:-:|:-:|
| 1 support | pass | pass | pass | 0 |
| 2 TG > 0 | pass | pass | pass | 0 |
| **3 TG CI lower** | pass | **FAIL** | **FAIL** | **2** |
| 4 random null | pass | pass | pass | 0 |
| 5 target-hit > 0.50 | pass | pass | pass | 0 |
| **6 t-hit CI lower** | pass | **FAIL** | **FAIL** | **2** |
| 7 collateral point | pass | pass | pass | 0 |
| 8 collateral CI | pass | pass | pass | 0 |
| 9 zero exact | pass | pass | pass | 0 |
| 10 structural | pass | pass | pass | 0 |

**Every formal refusal in the project was decided by conditions 3 and 6, and by nothing else.**

## Are 3 and 6 redundant?

Partly. Both are lower-confidence-bound requirements on destination quality, and they are
strongly dependent: target-hit is `gold/exits` and Target Gain is `(gold − other)/n_err`, so
they share the same `gold` numerator and move together. In both DECLINEs they failed *jointly*,
never singly — which is the signature of two conditions measuring one underlying uncertainty.

That is not automatically a defect. Requiring both means the licence demands that the
intervention be *both* net-beneficial per channel error *and* reliably targeted per exit. But
it does mean the conjunction is effectively **8 independent conditions plus one
double-counted destination-precision requirement**, and its refusal behaviour is dominated by
that single axis.

## Conditions never observed to bind

Conditions 1, 2, 4, 5, 7, 8, 9, 10 have never decided a formal case. Four of them (2, 5, 7, 9)
are point-estimate or structural preconditions that a serious candidate passes almost by
construction. The random-null condition 4 has never failed at the formal stage — though it
*has* failed decisively at earlier stages in Mistral and Llama, which is where it earns its
place.

## Is the conjunction brittle or intentionally conservative?

**Conservative, and under-calibrated rather than brittle.** The simulation shows P(ADMIT) ≈ 0.93
at the Qwen3-8B effect and ≈ 0.19–0.25 at the DECLINE effects. A gate that admits large effects
reliably and refuses modest ones four times in five is behaving as a conservative evidential
rule, not as a broken one. What is missing is any calibration: no analysis ever established what
effect size the licence *should* admit, so the operating point is a convention rather than a
derived threshold.

**No threshold is proposed here.** Recording that conditions 3 and 6 are the sole binding pair,
and that they are dependent, is a finding — not a licence to relax either.
