> Anonymous display derivative. This is not the original frozen review.
> Only identity mentions were replaced; scientific content is unchanged.

# Final pre-draft ICLR go/no-go review

```
ICLR_READY_WITH_ONE_PRE_DRAFT_FIX
```

The fix is a **labelling correction**, not an experiment. It is mandatory.

---

## THE PRE-DRAFT FIX — S-5 contradicts the primary artifacts

**S-5 states:** *"Qwen3-4B and Gemma are ADMITTED AT LEVEL-1, not DECLINE."*

**The primary artifacts state the opposite:**

```
final/results/qwen3_4b_w2c_formal_v1/FORMAL_PRINCIPAL_VERDICT.txt
  -> QWEN3_4B_FORMAL_DECLINE

.../gemma/formal_freeze/FORMAL_PRINCIPAL_VERDICT.json
  -> GEMMA_FORMAL_DECLINE   (8/10 conditions pass)
     failed: 3_target_gain_CI_lower_gt_0, 6_target_hit_CI_lower_gt_0.50
     failed_evidence_layer: "destination interval conditions (3 and 6)"
```

**Independently reproduced.** Recomputing the frozen 10,000-draw percentile
bootstrap **resampling channel errors** (the frozen unit, not exits):

| model | ch. errors | gold/exits | target-hit | **CI95 lower** | cond. 6 (>0.50) |
|---|---:|---:|---:|---:|:-:|
| Qwen3-8B | 72 | 38/52 | 0.7308 | **0.6034** | **PASS** |
| Qwen3-4B | 118 | 37/64 | 0.5781 | **0.4561** | **FAIL** |
| Gemma-2-9b | 96 | 17/27 | 0.6296 | **0.4400** | **FAIL** |

My reproduction independently confirms both DECLINEs on the exact condition the
frozen verdict names. **"Level-1" has no prospectively frozen definition anywhere
in the archive** — it is a post-hoc descriptive rung (point estimate > 0.50 and
gold > wrong-to-wrong) introduced *after* the declines were returned.

Per the standing rule — a constraint is binding *unless a primary artifact
mechanically falsifies it* — **S-5 is falsified and must be rewritten.**

### Why this is a fix and not a catastrophe

The science is untouched. Both statements are true about different gates. What is
untenable is publishing only the ADMIT half. A reviewer reading "admitted at
Level-1" beside artifacts reading `FORMAL_DECLINE` would treat it as
misrepresentation, and the internal logic — *declined, then defined a weaker rung
they pass* — is visible without the artifacts.

### The reframe makes the paper stronger

Both models failed on **conditions 3 and 6 — the destination interval**, not on
the point estimate. That is precisely the failure mode the preservation
preregistration named as *thesis-reinforcing*: *"fails on destination interval
(3 or 6): consistent with the exchange-rate story — the effect is real but the
exit count is small."* Qwen3-4B has 64 exits and Gemma 27, against the ~145
required at the predeclared threshold.

**"Our frozen gate declined two of the three modern models, on exactly the
condition our exchange-rate analysis predicts"** is a far stronger demonstration
that the gate has content than "three models admitted at some level."

### Required manuscript actions

1. Rewrite S-5: *Qwen3-4B and Gemma returned `FORMAL_DECLINE` under the
   prospectively frozen gate, failing the destination-interval conditions (3 and
   6). They satisfy a post-hoc descriptive rung (Level-1) defined on point
   estimates after the declines.*
2. Every table/figure showing Level-1 must carry the DECLINE and the failing
   condition.
3. Label Level-1 **POST-HOC**, never "licensed" or "admitted".
4. Update `FINAL_CLAIM_MATRIX` C5-1 and `FINAL_MODEL_EVIDENCE_MAP`.

---

## 1. Canonical rung wording

**Inclusive attainment is not usable**, because Level-1 is post-hoc while Level-2
is prospectively frozen. Mixing them implies a single ladder that was never
prospectively defined.

**Canonical wording the paper must use:**

> Under the prospectively frozen licensing gate, one of three modern models
> (Qwen3-8B) was **ADMITTED at Level-2 (majority repair)**; the other two
> (Qwen3-4B, Gemma-2-9b) returned a **formal DECLINE**, each failing the
> destination-interval conditions. Both declined models satisfy a **post-hoc
> descriptive rung** (target-hit point estimate > 0.50 with gold arrivals
> exceeding wrong-to-wrong redistribution), which we report as descriptive
> evidence, not as a licence.

Never *"3 models reached Level-1"* without the DECLINE in the same sentence.

---

## 2. Blocker re-audit

| # | risk | current evidence | status | manuscript action |
|---|---|---|---|---|
| 1 | CAST overlap | §3.1 verified; composition distinct 0/6 | **mitigated** | cite in §3 at method description |
| 2 | one Level-2 | 1 ADMIT, 2 DECLINE — gate discriminates | **limitation, now an asset** | lead with rung distribution |
| 3 | Phi execution variability | 107/52 vs 106/44; routing 293 vs 243; cause undeterminable | **mitigated by disclosure** | proactive Methods paragraph |
| 4 | Phi preservation robustness | all realizations > budget; min 6.1% vs 5% | **resolved** | quote range beside point estimate |
| 5 | E1 vs E2 denominator | 1.3% vs 5.7% on same data | **disclosed** | mandatory denominator column |
| 6 | vacuous preservation | exposure 0 (4B ca→direct); n=6 (8B) | **disclosed** | label VACUOUS |
| 7 | dose/protocol dependence | exit-matched Δ flips sign | **resolved by interpretation** | protocol-conditional wording |
| 8 | score-space | 37 vs 38; ns after Bonferroni | **converted to evidence** | Figure 3 Panel B |
| 9 | MetaTool heterogeneity | 524/1040 rows shared | **disclosed** | "two partially overlapping splits" |
| 10 | historical/modern | 4 historical, 3 modern | **mitigated** | visual provenance separation |
| 11 | placebo row-level gaps | 4/5 E-2 arms aggregate-only | **irreducible limitation** | state it |
| 12 | mechanism depth | multi-PC, no circuit | **irreducible limitation** | "decision-associated structure" |
| 13 | benchmark breadth | licensing on W2C alone | **irreducible limitation** | state it |
| 14 | figure staleness | all stale | **operational gate** | regenerate first |
| 15 | **anonymity** | double-blind confirmed in force; public repo `<anonymised>/<anonymised>`; method name searchable | **OPERATIONAL BLOCKER** | anonymous mirror + rename or scrub before submission |

---

## 3. Thesis link audit

| link | verdict |
|---|---|
| A structured error channels exist | **STRONGLY_SUPPORTED** — 78% in 3 channels, split-stable |
| B error states are readable | **STRONGLY_SUPPORTED** — AUC 0.77–0.99 across models |
| C a subset is selectively correctable | **STRONGLY_SUPPORTED** — 5 models positive, controls pass |
| D effects are structurally specific | **STRONGLY_SUPPORTED** — 0/59 in two models; full control battery |
| E aggregate gain ≠ destination-correct repair | **STRONGLY_SUPPORTED** — Phi 46 other-wrong; score-space 0.731 vs 0.638 |
| F gain can coexist with preservation failure | **STRONGLY_SUPPORTED** — 52/93, robust across realizations |
| G evidence can be graded | **SUPPORTED** — 1 ADMIT, 2 DECLINE, plus Effect/Steerable failures |

**Does the thesis require any unsupported link? No.** Every link is at least
SUPPORTED; six of seven are STRONGLY_SUPPORTED.

---

## 4. The two turning-point demonstrations

**Phi — holds.** *A real, structurally specific, aggregate-positive correction
effect can coexist with severe preservation failure.* Robust to execution
variability: every artifact-backed realization exceeds budget. Point estimate must
carry the range.

**Score-space — holds, descriptively.** *Similar aggregate improvement can
correspond to different observed destination composition.* Net +38 vs +35, gold 38
vs 37, but OTHER_WRONG 14 vs 21 and target-hit 0.7308 vs 0.6379. No superiority
claim; ns after Bonferroni; n=6 collateral not used as evidence.

**Together they justify Section 5 as the central empirical section.** One example
reads as a single-model anomaly; two independent ones read as a methodological
finding. **And with the DECLINE reframe there is now a third**: two models whose
effects are real but whose intervals cannot certify repair.

---

## 5–7. ICLR assessments

**Novelty: `NOVELTY_SUFFICIENT_BUT_POSITIONING_SENSITIVE`.** Under the framework
framing, the composition is not matched in the verified corpus. Under a steering
framing it is borderline-to-insufficient against CAST. Everything depends on
positioning.

*Strongest objection:* "The Router is CAST; what is new?"
*Supported response:* "We agree, and say so where the mechanism is introduced. Our
contribution is the adjudication procedure — destination resolution, preservation
accounting, frozen criteria, an explicit decline rule — which CAST does not have
(binary label space, post-hoc grid-searched threshold, no destination categories,
no reverse/wrong-layer controls). Our gate declined two of three modern models;
CAST has no mechanism that could decline."

**Significance: `SIGNIFICANCE_SUFFICIENT`.** The strongest contribution is **the
combination**, and specifically *destination accounting + licensing*. The knowledge
transferred is a **negative methodological result the field can act on**:
aggregate improvement does not identify correction, demonstrated three ways.
Benchmark gains are not the contribution and must not be sold as such.

**Rigor: `RIGOR_STRONG`.** This is the paper's best dimension. Row-level
regeneration of every load-bearing number; 0/59 nulls recomputed independently;
frozen criteria with genuine declines; E1/E2 discipline; vacuity labelled;
proactive reproducibility disclosure; twelve real limitations. Rigor is strong
**only if** the S-5 fix lands — publishing "admitted" over `FORMAL_DECLINE` would
invert this into the paper's greatest weakness.

---

## 8. Submission hygiene

- **Page limit: 9 pages main text at submission, 10 during rebuttal.**
- **Deadlines: abstract 18 Sep 2026, full paper 25 Sep 2026 (AOE).** From
  28 Aug 2026 that is **21 and 28 days**.
- **Double-blind confirmed in force** for review; submissions later become public
  with names attached. The public repo `<anonymised>/<anonymised>` and the searchable
  method name are live deanonymization vectors. **Operational blocker.**
- **Reproducibility statement is truthful** as frozen.

---

## 9. Reviewer simulation

**Reviewer A — steering expert. Borderline → weak accept.**
*Positive:* control battery is unusually complete (random, reverse, wrong-layer,
mismatched, ungated, zero, K=59).
*Negative:* the mechanism is CAST's; and score-space nearly matches at 37 vs 38.
*Rebuttal viable?* **Yes** — both are already disclosed and reframed; the
score-space result is presented as a finding of the framework.

**Reviewer B — statistics expert. Weak accept → accept**, *if* the S-5 fix lands;
**reject** if it does not.
*Positive:* frozen criteria, real declines, E1/E2 discipline, row-level
regeneration. Rare.
*Negative:* one Level-2; historical Phi variability; post-hoc Level-1.
*Rebuttal viable?* **Yes if disclosed up front; no if discovered.** This reviewer
decides the paper.

**Reviewer C — agents/tool-use expert. Borderline.**
*Positive:* pre-execution decisions matter and are under-studied.
*Negative:* When2Call-centric; 3.8B–9B only; no end-to-end agent task; practical
takeaway is a caution, not a deployable gain.
*Rebuttal viable?* **Partially** — breadth is irreducible without new experiments.

---

## 10. Acceptance probability

**Current project, competent execution: ~25–35%.**
**Strong execution: ~40–50%.**

Drivers of the gap, in order: (1) CAST positioning — mishandled it caps at weak
reject; (2) the S-5 correction, which converts Reviewer B from the biggest threat
to the strongest advocate; (3) Figure 3 carrying both demonstrations; (4)
disciplined claim language; (5) anonymity.

No false precision: ICLR variance at this level is large, and a Reviewer-C draw
who wants agent-scale results can sink it regardless.

**Strongest rejection scenario:** a reviewer reads the contribution as
"CAST + an evaluation protocol", finds one Level-2 licence on one benchmark at
≤9B scale, notes the score-space comparator nearly matches, and concludes the
empirical payoff does not justify the machinery. The defence is that the negative
methodological result is the payoff — which lands only if Section 5 is the paper's
centre and the declines are foregrounded.

---

## 11. Verdict

```
ICLR_READY_WITH_ONE_PRE_DRAFT_FIX
```

**Fix:** correct S-5 and every derived table to state the formal DECLINEs and
relabel Level-1 as post-hoc. Non-experimental. Hours of work.

**Separately, an operational blocker before submission (not before drafting):**
anonymity — anonymous artifact mirror, and a decision on the method name.
