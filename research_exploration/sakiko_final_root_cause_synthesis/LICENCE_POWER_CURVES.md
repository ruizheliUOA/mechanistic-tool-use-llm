# Licence operating characteristics and power

CPU-only Monte Carlo using the frozen endpoint definitions (destination multinomial;
bootstrap CIs on Target Gain and target-hit; Clopper–Pearson on collateral) over a 420-cell
grid. Full grid in `LICENCE_OPERATING_CHARACTERISTICS.csv`.

## The headline: P(ADMIT) at each setting's own observed point effect

| setting | n_err | exit rate | target-hit | implied TG | **P(ADMIT \| observed n)** | n for 80% power |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B (ADMIT) | 87 | 0.598 | 0.731 | +0.2759 | **0.930** | **75** |
| Qwen3-4B (DECLINE) | 124 | 0.516 | 0.578 | +0.0806 | **0.190** | not reached ≤1000 |
| Gemma-2-9B (DECLINE) | 96 | 0.281 | 0.630 | +0.0729 | **0.250** | not reached ≤1000 |

**Qwen3-8B's ADMIT was not a lucky draw.** At its own effect size the frozen licence admits
93% of the time, and 75 channel errors would have sufficed. It had 87.

**Both DECLINEs were substantially under-powered at their own effect sizes** — 19% and 25%
certification probability. Under the frozen licence, a true effect of that size is *expected*
to be refused roughly four times in five.

## The finding that corrects my own earlier claim

I previously wrote that Gemma "just needed ~199 channel errors." That was computed against the
Target-Gain condition alone and **is wrong as a statement about the full conjunction.**

Certification probability as n grows, with exposed-correct scaled proportionally (as it would
be in a genuinely larger population):

| | n=obs | n=200 | n=400 | n=1000 |
|---|---:|---:|---:|---:|
| Qwen3-4B P(ADMIT) | 0.200 | 0.280 | 0.485 | **0.115** |
| dominant failure | precision | precision | precision | **collateral** |
| Gemma P(ADMIT) | 0.205 | 0.420 | 0.460 | **0.020** |
| dominant failure | precision | precision | **collateral** | **collateral** |

**P(ADMIT) rises, peaks near n≈400, then collapses.** The binding condition migrates from
*precision* to *collateral*. More evidence would not have converted these DECLINEs into ADMITs;
it would have converted a precision failure into a collateral failure.

The mechanism is visible in the raw data. Gemma broke **1 of 11** exposed correct rows — an
exposure-conditional rate of **9.1%**, well above the 0.05 bound. With only 11 exposed rows the
interval was far too wide to detect that, so the frozen all-correct denominator (1/223 = 0.45%)
passed. At larger exposure the same underlying rate becomes measurable, and binds.

Holding exposure *fixed* at the observed level isolates the pure precision effect: both settings
then certify comfortably (Qwen3-4B 0.935, Gemma 0.995 at n=1000). The difference between those
two rows is the entire finding.

## Stated limitation of this simulation

The collateral denominator is held at 223 while exposed-correct is scaled. That is internally
inconsistent for large n: in a real larger population the all-correct denominator would grow
too, which would soften the collateral collapse. **The direction of the effect is robust; the
magnitude at n=1000 is not.** The defensible claim is that the binding constraint *migrates*
from precision toward collateral as evidence grows — not that P(ADMIT) falls to 0.02.

## What this does to the "just under-powered" reading

It kills the simple version of it. The correct statement for Gemma and Qwen3-4B is:

> Their point effects were direction-specific and positive, and the frozen population was too
> small to certify them at the required destination precision. But their observed
> exposure-conditional collateral rates mean that a larger confirmatory population would not
> reliably have produced an ADMIT either — it would have moved the binding condition to
> preservation.

This is a stronger and less flattering conclusion than the one I reported earlier, and it is
the one the evidence supports.
