# Collateral estimand audit

## Three different populations have been called "collateral"

| estimand | definition | Gemma | Qwen3-4B | Qwen3-8B | Phi-3.5 |
|---|---|---|---|---|---|
| **A. historical formal** | broken / all baseline-correct rows in the evaluation split | 1/223 = **0.45%** | 1/214 = **0.47%** | 0/211 = **0.00%** | 52/264 = **19.7%** |
| **B. causal exposure** | broken / Router-fired ∩ baseline-correct | 1/11 = **9.1%** | 1/50 = **2.0%** | 0/6 = **0.0%** | 52/93 = **55.9%** |
| **C. deployment** | expected damage rate under a deployment gating policy | not estimated | not estimated | not estimated | not estimated |

## Which population is actually at risk

SAKIKO modifies **only Router-fired rows**. A baseline-correct row the Router never fires on
cannot be damaged by the intervention — its outcome is unchanged by construction, and the zero
control verifies this exactly. Under a causal reading, **estimand B is the correct risk
denominator** and estimand A is diluted by rows that were never exposed.

The dilution is not small. Gemma's denominator A contains 223 rows of which **212 were never
exposed** — 95% of the denominator carries no risk and cannot contribute a numerator event.

## Why this is more serious than a reporting quibble

The simulation in `LICENCE_POWER_CURVES.md` shows the two estimands **disagree about the
verdict at scale**. Gemma passes A comfortably (0.45%, CP95 upper 0.0247) while its B rate of
9.1% is nearly twice the 0.05 bound. With 11 exposed rows the B interval is
[0.0023, 0.4128] — uninformative — so nothing binds. Increase exposure at the same underlying
rate and B binds hard.

So estimand A did not merely flatter Gemma cosmetically. **It concealed the condition that
would actually have decided the case in a better-powered study.**

## What must not be done

Estimand B must **not** retroactively replace the frozen primary in any historical result.
Qwen3-8B's ADMIT was adjudicated on A, prospectively, and rewriting the denominator after the
fact would be exactly the post-hoc estimand substitution the project exists to refuse. The
0/211 stands as the frozen result.

But the same audit applies to the ADMIT: Qwen3-8B's exposure was **6 rows**. `0/6` has CP95
upper ≈ 0.46. Its collateral condition passed on a denominator that could not have failed in
any practical sense. The DEV estimate at the same channel and dose was 4/15 = 26.7%, and
P(0 of 6 | 0.267) = 0.156 — so the sealed zero is entirely compatible with a materially unsafe
true rate.

## Recommended reporting standard, prospectively only

Report **A and B side by side, always, with exposure n stated**. Neither replaces the other:
A is the frozen comparability metric; B is the causal risk quantity. A pass on A with
single-digit exposure must never be described as safety — and that applies to the ADMIT as much
as to the DECLINEs.

Any future prospective design should treat **exposed-correct n as a pre-registered power
quantity**, sized so that estimand B can actually reject at the bound. Three of the units in
this project reached the collateral rung with 0, 6 or 11 exposed rows.
