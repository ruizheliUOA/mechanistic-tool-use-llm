# Can preservation be risk-controlled? Feasibility verdict

```
PRESERVATION_UNCERTIFIABLE (at α = 0.05, all settings)
```

## Obstacle diagnosis

Per §23-D, the obstacle is **BOTH**, and it differs by setting:

| setting | obstacle |
|---|---|
| Qwen3-8B (ADMIT), Gemma `ca→tc` | **too little exposure** — 6 and 11 rows |
| Gemma `ca→direct`, Qwen3-4B `ca→direct` | **zero exposure** — structurally vacuous |
| Phi-3.5 | **high observed risk** — 55.9% on exposed |
| Qwen3-4B `ca→tc` | both — 50 rows at 2.0%, upper 0.0914 |

α was **not** relaxed after seeing these results, and no settings were pooled to shrink a bound.

## Fixed policy ≠ conformal calibration

For the historical runs the policy is completely frozen — channel, Router, τ, direction, dose
all fixed before execution, with no calibration parameter. **Conformal risk control is the wrong
tool here**, because there is nothing to calibrate. The correct instrument is exact finite-sample
risk estimation, which is what `PRESERVATION_FINITE_SAMPLE_BOUNDS.csv` reports.

Forcing a CRC interpretation onto a frozen policy would be a category error, and I am not doing it.

## Where CRC does belong — prospectively only

For a **future** SAKIKO, define a nested policy family `{π_λ}` where λ controls Router
confidence / eligible exposure set — *not* dose, since preservation risk is not established to be
monotone in dose. Then:

```
maximise  U(π_λ)   subject to   R_pres(π_λ) ≤ α   with finite-sample control
```

This requires an exchangeable calibration population of exposed baseline-correct rows. Given the
rule of three, that means designing for **≥59 exposed correct rows at α = 0.05** — an order of
magnitude more than any setting in this project achieved.

**Exposure must become a pre-registered power quantity**, sized before sealing. That is the
single most actionable protocol change this analysis produces, and it is future work.

## Relation to the literature — stated without inflation

SAKIKO did not invent finite-sample risk control, and must not claim to. Angelopoulos et al.'s
conformal risk control, the structured-LLM CRC work (arXiv 2606.29054), CORA (2604.09155) and
role-stratified tool-call CRC (2607.24343) all predate any such use here. **2607.24343 already
measures over-intervention on benign fields with a distribution-free guarantee** — the closest
prior, and it has the machinery SAKIKO lacks.

The only defensible novelty is the **application target**: exposure-conditional risk on
*intervention-exposed correct tool decisions*, inside a Readable→Steerable→Correctable ladder.
That is narrow, and it should be stated as narrow.

## The impossibility result reframes the whole picture

arXiv 2606.29054 proves that above a base-risk threshold any distribution-free certification must
abstain heavily. Combined with the rule of three, this says SAKIKO's high refusal rate is
**structurally expected**, not a tuning artifact — and that **uncertifiability at these exposure
counts was predictable a priori**. That is a genuine finding, and it is not a flattering one:
it means the project's preservation evidence was never going to be conclusive at the sample
sizes it was designed around.
