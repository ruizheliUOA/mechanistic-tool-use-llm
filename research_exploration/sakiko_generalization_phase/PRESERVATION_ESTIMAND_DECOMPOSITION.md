# Preservation — three estimands, and what each one licenses

## Definitions

**E1 — historical frozen estimand.** `#(baseline-correct → wrong) / #(all baseline-correct)`.
The quantity the preregistered protocol used. Retained for comparability. Not modified.

**E2 — exposure-conditional causal risk.**
`R_pres(I) = P(Y_I ≠ Y_0 ∧ Y_0 = Y* | E_I = 1, Y_0 = Y*)` — broken among **exposed**
baseline-correct rows. For a gated intervention this is the causal population at risk: a row the
Router never fires on cannot be damaged, and the zero control verifies exactly that.

**E3 — deployment population risk.** `P(exposed) × R_pres(I)`. A different question again, and
the one a deployment decision actually needs. **Never computed in this project.**

The three must not share a denominator.

## Historical reanalysis

| setting | E1 historical | E2 exposure-conditional | **95% one-sided upper on E2** | certifiable at α=0.05? |
|---|---|---|---:|---|
| **Qwen3-8B `ca→tc` (the ADMIT)** | 0/211 = 0.0000 | **0/6** = 0.0000 | **0.3930** | **NO** |
| Qwen3-4B `ca→tc` | 1/214 = 0.0047 | 1/50 = 0.0200 | 0.0914 | **NO** |
| Gemma `ca→tc` | 1/223 = 0.0045 | **1/11** = 0.0909 | **0.3644** | **NO** |
| Gemma `ca→direct` | 0/471 = 0.0000 | **0/0 undefined** | n/a | **VACUOUS** |
| Qwen3-4B `ca→direct` | — | **0/0 undefined** | n/a | **VACUOUS** |
| Phi-3.5 cascade | 52/264 = 0.1970 | 52/93 = 0.5591 | 0.6468 | **NO** |

## The finding

**Not one setting in the project can certify exposure-conditional preservation at α = 0.05 —
including the ADMIT.**

The Qwen3-8B ADMIT's frozen record reads `clean_collateral_rate: 0.0`. On the causally exposed
population that is **0 of 6 rows**, whose one-sided 95% upper bound is **0.393**. The frozen
endpoint is compatible with a true damage rate on exposed correct decisions of up to **39%**.

This is not a criticism of the frozen verdict, which was adjudicated prospectively on E1 and
stands. It is a statement about what E1 can and cannot support: **E1 passing is not evidence of
safety, and the gap is roughly two orders of magnitude wide.**

## Why exposure is so small

E1's denominator is dominated by rows the intervention never touches. For Gemma, 212 of 223
baseline-correct rows were never exposed — 95% of the denominator carries no risk and cannot
contribute a numerator event. Router gating, which is what makes the intervention safe, is also
what makes its safety unmeasurable at these sample sizes.

## Rule of three

To certify `R_pres ≤ α` with **zero** observed breaks requires roughly `3/α` exposed rows:

| α | exposed rows needed (0 breaks) |
|---:|---:|
| 0.05 | **59** |
| 0.10 | 29 |
| 0.20 | 14 |

Maximum exposure achieved anywhere in the project: **93** (Phi — which broke 52 of them).
Maximum exposure in a *passing* setting: **50** (Qwen3-4B). The ADMIT had **6**.

## And more data would not have rescued Gemma

At Gemma's observed 9.1% exposure rate, the bound converges toward the point estimate:

| n exposed | 95% one-sided upper |
|---:|---:|
| 11 | 0.3644 |
| 100 | 0.1518 |
| 1000 | **0.1074** |

It never reaches 0.10, because the underlying rate is ~0.09. **Gemma's preservation problem is
not a power problem — it is a risk problem that the diluted denominator concealed.**
