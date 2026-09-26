# Formal verdict reconciliation

```
S5_RECONCILED_READY_TO_DRAFT
```

One pre-draft correction pass. No experiments, no threshold changes, no
reinterpretation of frozen outcomes. Additive commit; superseded wording retained.

---

## Part 1 — formal verdicts verified from primary artifacts

| model | artifact | verdict | target-hit | frozen CI95 | outcome |
|---|---|---|---:|---|---|
| **Qwen3-8B** | `QWEN3_STAGE2_FORMAL_RESULTS.json` + `FORMAL_REPORT.md` | **`FORMAL_CONFIRMATORY_SUCCESS`** | 0.7308 | **[0.60416, 0.84615]** | ADMIT |
| **Qwen3-4B** | `FORMAL_PRINCIPAL_VERDICT.txt` | **`QWEN3_4B_FORMAL_DECLINE`** | 0.5781 | lower ≈ 0.456 | DECLINE |
| **Gemma-2-9b** | `FORMAL_PRINCIPAL_VERDICT.json` | **`GEMMA_FORMAL_DECLINE`** | 0.6296 | lower ≈ 0.440 | DECLINE |

Gemma passes **8 of 10** conditions, failing exactly
`3_target_gain_CI_lower_gt_0` and `6_target_hit_CI_lower_gt_0.50`
(`failed_evidence_layer: "destination interval conditions (3 and 6)"`).

**Reproduction.** 10,000-draw percentile bootstrap resampling **channel errors**
(the frozen unit — resampling exits is invalid): CI lowers 0.6034 / 0.4561 /
0.4400. The Qwen3-8B value differs from the frozen 0.60416 only by Monte-Carlo
noise, and the predeclared threshold **0.6042 is that frozen bound rounded** — it
was derived from this result, so it sits at the boundary by construction. Both
DECLINEs reproduce on the exact condition the artifacts name.

`n_channel_error = 87` in the results file versus **72** routed; target-gain uses
the full 87 (`pad_unrouted`), target-hit uses the 52 exits. Both conventions
confirmed consistent.

**Part 1: verdicts reproduce. Not blocked.**

---

## Part 2 — provenance of "Level-1"

```
POST_HOC_DESCRIPTIVE_RUNG   (status B — introduced after the formal outcomes)
```

The prospectively frozen ladder, in
`research_exploration/sakiko_paper_freeze/FINAL_SAKIKO_DEFINITIONS.md`, is:

```
Adjudicable -> Readable -> Steerable -> Correctable -> Preserving -> Licensable
```

with outcomes **ADMIT / DECLINE**. **"Level-1" and "Level-2" appear nowhere in it.**
Across the whole archive only four files match "Level 1/2", and each is unrelated
(dataset grouping levels in a feasibility study; benchmark gate matrices). The
terminology appears in fifteen files on `sakiko-paper` — all authored *after* the
formal declines were returned.

The frozen definitions already carry the needed consequence:

```
DECLINE != intrinsically uncorrectable
Licensable is a property of the EVIDENCE, not of the setting
Preserving is a FOURTH property, not a component of Correctable
```

**Conclusion:** the honest framing was already available and is stronger than the
invented rung. Qwen3-4B and Gemma are **Correctable on the point estimate but not
Licensable** — a dissociation the frozen ladder was explicitly designed to express.

---

## Part 3 — canonical status language, frozen

Three things, never merged:

**A. Point estimate / observed effect.** Qwen3-4B target-hit **0.5781**, Gemma
**0.6296**, both with gold arrivals exceeding wrong-to-wrong. Supports:
*"observed destination tendency was positive."*

**B. Post-hoc descriptive category.** If retained at all, label
**descriptive / post-hoc**. Never "formally admitted".

**C. Frozen formal verdict — always takes precedence for confirmatory status.**
Qwen3-8B ADMIT; Qwen3-4B DECLINE; Gemma DECLINE.

### Canonical sentence

> Under the prospectively frozen gate, one of three modern models (Qwen3-8B) was
> **admitted**; Qwen3-4B and Gemma-2-9b returned a **formal DECLINE**, each failing
> the destination-interval conditions. Both declined models show *positive*
> destination point estimates (0.5781 and 0.6296) with intervals that cross the
> frozen boundary — the evidence is insufficient under this protocol and sample
> support (64 and 27 exits), **not** evidence that the models are uncorrectable.

---

## Parts 4–6, 14 — files corrected

| file | change |
|---|---|
| `constraints/STANDING_CONSTRAINTS.md` | **S-5 marked SUPERSEDED**, wording retained; **S-20** (verdict precedence) and **S-21** (rung vocabulary) added |
| `final_freeze/FINAL_CLAIM_MATRIX.{md,csv}` | C5-1 restated as ADMIT + two DECLINEs; prohibitions rewritten |
| `final_freeze/FINAL_MODEL_EVIDENCE_MAP.md` | verdict column replaces rung labels; CI bounds shown |
| `final_freeze/UNIFIED_SAKIKO_SPECIFICATION.md` | frozen six-rung ladder replaces the invented five |
| `final_freeze/FINAL_CONTRIBUTIONS.md` | contribution 3 restated on the frozen ladder |
| `final_freeze/FINAL_LIMITATIONS.md` | limitation 1 rewritten; "not uncorrectable" guard added |
| `final_freeze/FINAL_FIGURE_PLAN.md` | **Figure 4 redesigned** as a forest plot with the decision boundary |
| `final_freeze/CANONICAL_THESIS.md`, `ABSTRACT_LOGIC.md` | "declines in practice, not only in principle" |
| `final_freeze/FINAL_REVIEWER_RISK_CLOSURE.md` | R2 restated |
| `final_evidence/FINAL_PAPER_EVIDENCE.md` | stale PASS row corrected |
| `claim_matrix/FROZEN_CLAIM_MATRIX.md` | **annotated SUPERSEDED**, not edited — provenance preserved |

No stale contradictory status survives.

---

## Parts 7–9 — the declines as positive evidence

The central arc now has **three** complementary demonstrations, not two:

| | demonstration | what it shows |
|---|---|---|
| **A** | Phi: Net +55, collateral 52/93 | aggregate gain is insufficient |
| **B** | activation vs score-space: Net +38/+35, target-hit 0.7308/0.6379 | destination composition matters |
| **C** | Qwen3-4B / Gemma: positive point estimates, formal DECLINE | **even a favourable destination is insufficient without statistical support** |

**This is stronger than the previous two-part structure.** A, B and C fail at three
*different* points of the pipeline — preservation, destination, and evidential
support — so the argument for `Correct → Verify → License` no longer rests on any
single failure mode. C is also the only one that demonstrates the *License* stage
doing work, which was previously argued rather than shown.

**Contribution 3, strengthened:**

> SAKIKO distinguishes observed correction effects from formally supportable
> repair claims: two modern models exhibit favourable destination point estimates
> yet are declined by prospectively frozen uncertainty criteria.

**Never:** that DECLINE proves the models uncorrectable. The supported reading is
*evidence insufficient for the stronger claim under the evaluated protocol and
sample support.* Exit counts are 64 and 27; no required sample size is asserted
beyond the already-frozen exchange-rate curve.

---

## Part 13 — did the correction help?

```
NET_STRENGTHENED
```

- **Rigor — materially up.** The single largest reviewer risk was a claim
  contradicting a primary artifact in the same repository. That is now impossible.
- **Turning point — up.** Two demonstrations became three, failing at three
  different pipeline stages.
- **Evidence-hierarchy clarity — up.** The frozen six-rung ladder with ADMIT/DECLINE
  replaces an invented five-rung scheme that had no frozen definition.
- **Novelty — unchanged.** Positioning is untouched.
- **Significance — slightly up.** "Our gate declines two of three models" is a
  sharper demonstration that licensing has content than any admission count.

The one cost is presentational: the headline can no longer say three models
reached a licensed level. It never could — the artifacts always said otherwise.
