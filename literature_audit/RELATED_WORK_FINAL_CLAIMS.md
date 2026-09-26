# Related-work final claims

## Corpus

6 papers, full text verified. **CONVENIENCE SAMPLE, not a prevalence estimate.**

| paper | full text | key coding |
|---|---|---|
| CAST (Lee et al., ICLR 2025) | VERIFIED | detector-gated conditional intervention |
| RepE (Zou et al.) | VERIFIED | unconditional control, reading + LoRRA |
| CAA (Rimsky et al.) | VERIFIED | N=50/behaviour, binary A/B |
| ITI (Li et al., NeurIPS 2023) | VERIFIED | 817 Q, random-direction control |
| Tan et al. 2024 | VERIFIED | 500 test/behaviour, binary A/B |
| AxBench (Wu et al. 2025) | VERIFIED | open-ended, judge-scored |

## RETRACTION — Session A's exchange-rate table is withdrawn

Session A applied SAKIKO's Level-2 design-sensitivity curve to all four papers
then verified. **That was invalid.** The curve applies only to the estimand
"source exits -> gold arrivals vs non-gold arrivals, H0 p_target <= 0.50".

None of the six papers reports a compatible source-error population, source
exits, or gold/other-wrong destination structure. All six are coded:

```
NOT COMPARABLE UNDER SAKIKO LEVEL-2 ESTIMAND
```

The numbers 0.6849 / 0.5440 / 0.5557 / 0.9167 are withdrawn and must not appear.
Reporting a minimum-certifiable rate for a paper that does not measure that
quantity is a category error, not a conservative approximation.

## Inferential-unit corrections (A2.4)

| paper | unit reported in Session A | correct primary claim unit | verdict |
|---|---|---|---|
| CAA | 50 items | **behaviour** (7 behaviours, 50 items each) | claims are per-behaviour effect sizes |
| ITI | 817 questions | **question**, 2-fold CV | claim is accuracy over questions |
| Tan et al. | 500 test items | **dataset** (40 datasets) — claims concern generalisation *across* datasets | **INFERENTIAL_UNIT_AMBIGUOUS** |
| AxBench | 10 instructions | **concept** (500 concepts) — the 10 instructions are within-concept replicates | Session A used the smaller unit; corrected |

AxBench in particular: taking N=10 was wrong. The paper's claims are made across
500 concepts, not within one concept's 10 instructions.

## Novelty matrix — component-level

| dimension | YES | PARTIAL | NO |
|---|---:|---:|---:|
| INTERNAL_CAUSAL_INTERVENTION | **6** | 0 | 0 |
| DETECTOR_OR_ROUTER_GATE | **1** (CAST) | 0 | 5 |
| RANDOM_DIRECTION_CONTROL | **1** (ITI) | 0 | 5 |
| PRESERVATION_ENDPOINT | 0 | 2 | 4 |
| POWER/DESIGN SENSITIVITY | 0 | 1 (Tan) | 5 |
| K >= 3 NATIVE ACTION SPACE | 0 | 0 | 6 |
| DESTINATION_RESOLVED_ACCOUNTING | 0 | 0 | 6 |
| CHANNEL_DISCOVERY | 0 | 0 | 6 |
| REVERSE / WRONG_LAYER CONTROL | 0 | 0 | 6 |
| PROSPECTIVE_FROZEN_CRITERIA | 0 | 0 | 6 |
| ADMIT_DECLINE_OR_REFUSAL_RULE | 0 | 0 | 6 |

## Components SAKIKO cannot claim as novel

- **Internal causal intervention** — all six (universal).
- **Detector-gated conditional intervention** — **CAST**, §3.1:
  `h' <- h + f(sim(h, proj_c h)) * alpha * v`, with `f = 1 if sim > theta else 0`.
  **This is SAKIKO's Router role, already published at ICLR 2025 Spotlight.**
- **Random-direction controls** — ITI, §4.3 Table 3.
- Activation addition, representation reading, layer/site selection.

## Strongest prior collision

**CAST (Lee et al., ICLR 2025).** It is the only verified work performing
sample-conditional intervention gated by a learned detector. SAKIKO's Router is
architecturally the same idea.

What CAST does **not** do (all §-verified): binary refuse/comply label space; no
destination categories when an output changes; no random, reverse, or
wrong-layer controls; preservation only as a harmful-minus-harmless proxy; and
threshold selection by **post-hoc grid search**, explicitly not a prospectively
frozen criterion.

## Residual compositional novelty

```
FULL_COMBINATION_FOUND: NO within the verified corpus
```

No verified paper combines destination-resolved multiclass accounting +
target-directed controls (random/reverse/wrong-layer) + a preservation endpoint +
prospectively frozen criteria + an explicit ADMIT/DECLINE rule.

## Permitted wording

> "Among the closest works identified by our search, we found no method
> combining destination-resolved multiclass accounting, target-directed
> intervention controls, a preservation endpoint, prospectively frozen evidence
> criteria, and an explicit admit/decline rule. Detector-gated conditional
> intervention is established prior work (CAST); our contribution is the
> adjudication procedure around it, not the gating mechanism."

## Forbidden wording

- "No prior work does this."
- "The field generally / most steering papers / destination evaluation is uncommon."
- Any proportion (0/6, 1/6) presented as a field rate.
- Any minimum-certifiable target-hit for a paper not measuring the Level-2 estimand.

## Answers

```
DOES THIS SUPPORT A CLOSEST-WORK NOVELTY CLAIM: YES (composition only)
DOES THIS SUPPORT A FIELD-PREVALENCE CLAIM:     NO
```

A prevalence claim requires a separate survey with a defined population,
exhaustive or random sampling, multiple coders, inter-rater agreement, and
proportion CIs. **None was executed.**
