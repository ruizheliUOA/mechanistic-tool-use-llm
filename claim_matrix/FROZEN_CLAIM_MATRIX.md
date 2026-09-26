> **SUPERSEDED 2026-08-28.** This matrix is retained for provenance only.
> The authoritative version is `final_freeze/FINAL_CLAIM_MATRIX.{md,csv}`.
> In particular the line *"never call Qwen3-4B or Gemma a DECLINE — ADMITTED AT
> LEVEL-1"* is **FALSIFIED** by the primary artifacts: both returned
> `FORMAL_DECLINE`. See S-20 and
> `final_freeze/FORMAL_VERDICT_RECONCILIATION.md`.

# SAKIKO — Frozen Claim Matrix

```
VERDICT: CLAIM_MATRIX_FROZEN_READY_TO_DRAFT
```

10 claims across 5 contribution blocks. **6 load-bearing.** Machine-readable
companion: `FROZEN_CLAIM_MATRIX.csv`.

---

## Canonical thesis (Task 8)

> SAKIKO first demonstrates that structured pre-execution tool-use decision
> errors can be selectively corrected through channel-specific, Router-gated
> internal intervention. Extending the analysis reveals that behavioral
> improvement alone does not determine how strongly a correction may be
> interpreted: a real and structurally specific gain need not support a reliable
> repair claim. SAKIKO therefore pairs correction with destination-resolved
> verification and evidence-based licensing, separating intervention effects,
> target-directed correction, and majority repair, and explicitly declining
> stronger conclusions the evidence does not carry.

---

## Central evidence chain (Task 3)

| # | question | verdict | evidence |
|---|---|---|---|
| Q1 | Can SAKIKO produce real correction effects? | **STRONGLY_SUPPORTED** | 4 models with positive Net; Phi 5/5 seeds +53 to +67 |
| Q2 | Are the effects structurally meaningful? | **STRONGLY_SUPPORTED** | REAL ranks 1/5 vs four corrupted controls; 0/59 randoms exceed real in two modern models |
| Q3 | Does behavioral gain imply reliable correction? | **STRONGLY_SUPPORTED (negatively)** | Phi Net +55 with 52/93 = 55.9% exposed-correct breakage |
| Q4 | Can correction strength be graded? | **SUPPORTED** | 7 models across 5 rungs; exchange rate 30→620 exits |
| Q5 | What mechanism is supported? | **PARTIALLY_SUPPORTED** | d_act−d_ss = −1.538; dose confound UNTESTABLE |

**Q3 is the paper's turning point** and the one place the evidence is strongest
in the negative direction. That is the contribution.

---

## Load-bearing claims — six, no more

| id | claim | why load-bearing |
|---|---|---|
| **C2-1** | a meaningful subset of tool-decision errors is correctable | Act 1 collapses without it |
| **C3-1** | REAL beats every structural control (Phi, 1/5) | answers "the framework only rejects" |
| **C3-2** | 0 of 59 matched randoms exceed real | direction specificity in the modern protocol |
| **C4-1** | aggregate Net conceals a preservation failure | Act 2's turning point |
| **C5-1** | seven models across five rungs | the empirical content of the ladder |
| **C5-2** | the evidence requirement scales steeply with effect size | explains why licensing fires once |

Everything else is IMPORTANT, SUPPORTING, or APPENDIX. C4-3 (activation vs
score-space) is **SUPPORTING only** — all paired tests were ns after Bonferroni.

---

## Reviewer objections (Task 5)

| # | objection | severity | current evidence | resolved? | new experiment? |
|---|---|---|---|---|---|
| 1 | novelty overlaps CAST (detector-gated steering) | **HIGH** | CAST §3.1 verified as prior art | **partly** — cite it; claim composition only | no |
| 2 | one Level-2 licence is thin breadth | **HIGH** | 7 models, 1 Level-2, 2 Level-1 | **no** — stated as a limitation | no (pincer: constructed populations leak, natural ones too small) |
| 3 | historical/modern protocol heterogeneity | MEDIUM | POST-HOC block separated; Phi flagged throughout | **yes** | no |
| 4 | preservation criterion may be too strict | MEDIUM | conditional bound vacuous at n=6 | **no** | would need a positive control; **not load-bearing** |
| 5 | dose confound in the mechanism analysis | MEDIUM | displacement rank-inverse to margin dependence | **no** — reported as UNTESTABLE | no; requires dose sweeps that do not exist |
| 6 | placebo destination logs missing | LOW | 4 of 5 arms aggregate-only | **no** — stated | no; artifacts do not survive |
| 7 | benchmark concentration on When2Call | MEDIUM | WildToolBench commissioned, Level-2 undecidable there (~83 exits, MDE 0.62) | **partly** | no |

---

## Task 6 — experiment necessity

```
NO_LOAD_BEARING_EXPERIMENT_REQUIRED
```

No candidate passes all four tests. The two strongest gaps fail on test 2 or 3:

- **Second Level-2 licence** — fails test 2 in the strict sense but more
  decisively fails feasibility: `SECOND_ADMIT_NOT_REALISTIC` under the pincer
  diagnosis. Not load-bearing: the paper's thesis is the licensing procedure and
  the rung distribution, not the count of licences.
- **Positive control on a known-strong steering effect** — would address
  objection 4, but objection 4 attaches to no load-bearing claim. It is upside,
  not a gap.

---

## Paper structure (Task 9)

1. Introduction
2. Problem: pre-execution tool decisions and what aggregate metrics cannot see
3. SAKIKO correction method (cite CAST as closest prior art)
4. Correction results — Act 1
5. **Why behavioral gain is not enough — the Phi turning point**
6. Verification and licensing: destination, specificity, preservation, evidence
7. Cross-model evidence hierarchy and the exchange rate
8. Structural calibration and controls (E-2, five conditions)
9. Mechanistic analysis — **EXPLORATORY** throughout
10. Limitations and conclusion

---

## Standing prohibitions carried into drafting

- never "SAKIKO reliably fixes tool errors across models"
- never "random directions produced no gold arrivals" — false for Qwen3-8B
  (126 gold across 47/59 arms)
- never 52/264 as the collateral rate; the estimand is **52/93**
- never "32 of 87"; it is **32 of 72 routed channel errors**
- never call Qwen3-4B or Gemma a DECLINE — **ADMITTED AT LEVEL-1**
- never 1.41% standing alone without the vacuous conditional bound beside it
- never a nine-condition ladder; E-2 is **five**
- never claim margin dependence explains rung placement — dose confound UNTESTABLE
- never revive the field-prevalence claim
- never present detector-gated conditional steering as novel
