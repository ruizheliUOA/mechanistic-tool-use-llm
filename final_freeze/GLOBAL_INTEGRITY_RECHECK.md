# Final global integrity recheck

```
GLOBAL_INTEGRITY_PASS_UNIFIED_SAKIKO_LOCKED
```

Every load-bearing number recomputed from row-level primary artifacts in a single
pass. No summary file was trusted.

## 1. V1/V2 integration — `V1_V2_INTEGRATION_COHERENT`

Each stage is motivated by evidence, not narrative:

| stage | origin | what forces it |
|---|---|---|
| Discover | historical | 3 channels = ~78% of errors; dominant channel *flips* between models |
| Detect | historical | exposure is the only lever on damage; ungated inverts sign in 3 of 4 settings |
| Correct | historical | REAL beats random/reverse/wrong-layer/mismatched; 0/59 randoms |
| **Verify** | later | 46 of 153 exits land on a third wrong action; equal-Net configs differ |
| **License** | later | two of three modern models formally declined on interval evidence |

The later work is not appended — it answers a question the earlier work raised and
could not settle: Phi's Net +55 alongside 52/93 breakage.

## 3–6. Recomputation — ALL PASS

```
Qwen3-8B    ch=72  gold=38  other=14  exits=52  t-hit=0.7308  CI_lo=0.6034
Qwen3-4B    ch=118 gold=37  other=27  exits=64  t-hit=0.5781  CI_lo=0.4561
Gemma-2-9b  ch=96  gold=17  other=10  exits=27  t-hit=0.6296  CI_lo=0.4400
  (frozen unit = channel errors, 10,000 draws)

matched randoms   Qwen3-8B real 38, K=59, max 8,  0/59 exceed
                  Gemma    real 17, K=59, max 0,  0/59 exceed

score-space       activation gold 38 other 14 exits 52 t-hit 0.7308 Net +38
                  score-space gold 37 other 21 exits 58 t-hit 0.6379 Net +35

Phi  n=548 fired=293 routed_err=200 exposed=93
     SOURCE_RETAINED=47 GOLD=107 OTHER=46 exits=153
     target-hit 107/153 = 0.6993   BROKEN=52 retained=41  E2 52/93 = 0.5591
```

**Paired-test status verified:** McNemar exact two-sided **p = 1.0** and **0.189**;
the artifact further records *"all-548 paired correctness was NOT prespecified as a
paired comparison."* The score-space comparison is therefore **descriptive and
post-hoc**. Permitted: *similar aggregate improvement, different observed
destination composition.* Forbidden: any superiority or inferiority claim.

## 7. Collateral denominators

| model | num | denom | estimand | rate | status |
|---|---:|---:|---|---:|---|
| Phi-3.5 | 52 | 93 | **E2** | 55.9% | fails budget, well-powered |
| Phi-3.5 | 52 | 264 | **E1** | 19.7% | fails budget |
| Qwen3-8B | 0 | **6** | **E2** | — | **VACUOUS** |
| Qwen3-8B | 0 | 211 | **E1** | 0% (CP upper 0.0141) | passes; what the ADMIT used |
| Qwen3-4B | 1 | 50 | **E2** | 2.0% | passes |
| Gemma | 1 | **11** | **E2** | 9.1% | exceeds budget, weakly powered |
| Qwen3-4B `ca→direct` | 0 | **0** | — | — | **VACUOUS / UNADJUDICABLE** |

## 11. Three-part turning point

| | demonstration | verdict |
|---|---|---|
| **A** preservation — Phi Net +55, 52/93 | **STRONGLY_SUPPORTED** |
| **B** destination — activation vs score-space | **SUPPORTED** (descriptive; ns paired tests) |
| **C** evidence sufficiency — two formal DECLINEs | **STRONGLY_SUPPORTED** (artifact verdicts) |

Together they support: *behavioural improvement, destination-correct repair,
preservation, and formally supportable correction are **distinct properties***.
They fail at three different pipeline stages, so the argument rests on no single
failure mode.

## 12. Claim rows

All **PASS or NARROW**. **No load-bearing claim REMOVED.** C5-1 narrowed to the
ADMIT/DECLINE split; C4-1 keeps the execution-variability range; C2-2 keeps the
split caveat; C6-1 keeps gating as SUPPORTED.

## 13. Risk classification

| risk | class |
|---|---|
| CAST overlap | MITIGATED (positioning) |
| one modern full-gate positive | LIMITATION |
| Qwen3-4B / Gemma declines | **RESOLVED — now evidence for License** |
| Phi execution variability | MITIGATED (disclosure) |
| preservation robustness | RESOLVED |
| collateral denominator complexity | RESOLVED (S-16) |
| vacuous preservation | RESOLVED (S-17) |
| dose dependence | RESOLVED (interpretation) |
| score-space near-equivalence | **RESOLVED — promoted to evidence** |
| MetaTool heterogeneity | LIMITATION (disclosed) |
| historical/modern heterogeneity | MITIGATED (provenance separation) |
| placebo destination logs | LIMITATION |
| mechanism depth | LIMITATION |
| benchmark breadth | LIMITATION |
| **anonymity** | **OPERATIONAL_BLOCKER** |
| **stale figures** | **OPERATIONAL_BLOCKER** |

**No SCIENTIFIC_BLOCKER remains.**

## 18. Positioning — `POSITIONING_STABLE`

Framework, not a steering primitive. CAST acknowledged at method introduction.
Novelty rests on the integration of channel-specific correction, multiclass
destination resolution, preservation accounting, structural controls and
evidence-graded formal licensing.
