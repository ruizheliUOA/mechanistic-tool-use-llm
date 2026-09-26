# Preservation — final status and repairability

Method verified: Clopper–Pearson exact, one-sided 95% upper bound. Recomputed this session.

| setting | exposed | rate | one-sided 95% upper |
|---|---|---:|---:|
| Qwen3-8B (the ADMIT) | **0/6** | 0.0000 | **0.3930** |
| Qwen3-4B | 1/50 | 0.0200 | 0.0914 |
| Gemma-2-9B | **1/11** | 0.0909 | 0.3644 |
| Phi-3.5 | 52/93 | 0.5591 | 0.6468 |

## Three distinct questions

**A. Is the historical Qwen3-8B ADMIT still valid under the frozen licence?**
**YES.** It was adjudicated prospectively against the endpoint that was frozen (E1, all
baseline-correct), passed 10/10, and nothing here revokes it.

**B. Is it preservation-certified under the exposure-conditional 5% interpretation?**
**NO.** 0/6 gives an upper bound of 0.393.

**C. Can B be made YES by re-analysing the existing six samples?**
**NO — arithmetically impossible.** The best attainable result on n=6 is zero breaks, which still
yields 0.393. Certifying α=0.05 with zero observed breaks requires **n ≥ 59**; 0/58 gives 0.0503
and only 0/59 reaches 0.0495. With one break it requires n ≥ 93; with two, n ≥ 124; with three,
n ≥ 153.

No re-analysis, denominator substitution, α adjustment or pooling can close a gap of that size,
and none was attempted.

## Verdicts

```
PRESERVATION_CURRENT_NARROW_CLAIM_SUFFICIENT:  YES
PRESERVATION_CERTIFICATION:                    NOT ESTABLISHED
PRESERVATION_REPAIRABILITY:                    NEW_PROSPECTIVE_DATA_REQUIRED
                                               (also: NOT_REQUIRED_FOR_CURRENT_NARROW_CLAIM)
```

Both labels hold simultaneously. The paper does not claim preservation certification; it reports
that no setting attains it, which is a true statement fully supported by the data.

## What a future preservation study would need

Exposure of **≥59 correct rows** at zero breaks. Achieving that requires one or more of: a lower
Router threshold (which trades destination precision for exposure), a substantially larger
evaluation population, or a dedicated preservation population sized in advance. None is a
re-analysis; all are new prospective data collection. **Exposure must become a pre-registered
power quantity** — that is the single actionable protocol lesson of this project.
