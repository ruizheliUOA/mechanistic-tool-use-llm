# Final adversarial story and ICLR audit

```
FINAL_PACKAGE_STRUCTURALLY_SOUND
NO_ACTIVE_NARRATIVE_CONTRADICTIONS
FINAL_STORY_CONFIRMED_BUT_POSITIONING_BORDERLINE
```

**The framing is not wrong, but it is not yet the strongest one available.** A
stronger framing is derived in Part 5 and it is more faithful to the evidence,
not less.

---

## Part 1 — package audit

93 rows / 13 cols · 12 claims (8 LB, 4 SUP) · ids unique · tiers A=49 B=35 C=9 ·
every LB claim mapped · none on Tier C · all Tier A sources tracked and real.

Spot-checks against source, one per family — all exact:

```
Phi   E2 collateral 52/93 = 0.5591   (row-level)
Q8B   target-hit    38/52 = 0.7308   (row-level)
MetaTool  8.6 / 13.0 / 21.6          (Tier B artifact)
mech  ca_tc·ca_direct = 0.7055       (Tier B artifact)
```

Tier B limitations disclosed (limitations 5, 6, 9). No claim contradicts S-20/S-21.

## Part 14 — contradiction scan

Every regex hit resolved to a **prohibition list or the retired-claims table** —
i.e. the guardrails firing, which is correct. `NO_ACTIVE_NARRATIVE_CONTRADICTIONS`.

---

## Part 3 — reconstruction from first principles

**Q1 — narrowest accurate object:** *pre-execution tool-decision errors, and the
question of what evidence establishes that an intervention has repaired one.*
Not activation steering (that is the instrument). Not hallucination. Not
evaluation in general.

**Q2 — original problem:** can structured, recurring wrong-action selections be
corrected by intervening on internal state, rather than by prompting or retraining?

**Q3 — what the original work established.** True positive: a meaningful subset is
improvable, with structural specificity (0/59 randoms, mismatched −17). Design:
gating, estimator adaptivity, site separation. Supporting: MetaTool bidirectionality.
**Does not survive:** universal gating necessity, channel orthogonality, intrinsic
model correctability, activation superiority over score-space.

**Q4 — what emerged.** Phi produced Net **+55** *and* broke **55.9%** of the correct
rows it touched. That single observation is not a story choice; it forces the
question "what does improvement actually establish?" Verify and License are the
answer to a question the data raised, not a framework selected in advance.

## Part 4 — integration audit

| transition | verdict | forcing evidence |
|---|---|---|
| Discover → Detect | **WELL_MOTIVATED** | channels are model-specific (dominant channel flips Phi↔Qwen); exposure is the only damage lever |
| Detect → Correct | **INEVITABLE_FROM_EVIDENCE** | Router AUC flat ~0.96 across layers where the direction degrades to noise (cos 0.0002, bootstrap 0.82) and benefit moves +13→+46 |
| Correct → Verify | **INEVITABLE_FROM_EVIDENCE** | 46 of 153 Phi exits land on a *third* wrong action; equal-Net configs (+49) differ in composition |
| Verify → License | **INEVITABLE_FROM_EVIDENCE** | two models with favourable destination point estimates returned `FORMAL_DECLINE` on interval evidence |

No transition is `ARTIFICIALLY_APPENDED`.

**Adversarial question — one object or two papers glued?** It reads as one object
**only if** the Phi result appears early. If Act 1 is presented as a clean success
and the problems arrive in Section 5, it reads as two papers. The Phi
Net-plus-breakage observation must appear **in the Introduction**, as the reason
the rest exists.

---

## Part 5 — the true central discovery

| candidate | verdict |
|---|---|
| A "new steering method" | **dies to CAST.** Do not use |
| B "evaluation framework for steering" | defensible, but ICLR systematically discounts eval-framework papers |
| C "errors correctable, but improvement ≠ repair" | strong, concrete, well-supported |
| D "correctability is evidence-graded and protocol-conditional" | true but abstract; belongs in Discussion |
| E "SAKIKO combines X, Y, Z, W" | a component list, not an idea — the classic weak-novelty signal |

### Derived candidate F — recommended

> **Behavioural improvement, target-directed repair, preservation of correct
> behaviour, and evidential sufficiency are four empirically independent
> properties of an intervention.** We exhibit each dissociating from the others on
> real tool-decision corrections, and derive an adjudication procedure that
> reports them separately rather than collapsing them into one number.

**Every dissociation is artifact-backed:**

| dissociation | evidence |
|---|---|
| improvement ⊥ preservation | Phi: Net **+55**, collateral **52/93 = 55.9%** |
| improvement ⊥ destination | score-space: Net +38 vs +35, target-hit **0.7308 vs 0.6379**, other-wrong 14 vs 21 |
| destination ⊥ evidential sufficiency | Qwen3-4B **0.5781**, Gemma **0.6296**, both `FORMAL_DECLINE` |
| correctable ⊥ preserving | Gemma: gold 17 > other 10, yet E2 collateral **9.1%** |

**Why F beats C.** F is a *finding*; C is a *caution*. ICLR rewards findings.
F also converts the framework from the contribution into a **consequence** of the
finding — which is both better received and more honest, since the frozen
definitions already asserted these properties were independent (*"Preserving is a
fourth property, not a component of Correctable: the two fail independently"*) and
the evidence confirms it. The paper's own definitions predicted the result.

**Ranking:** F > C > D > B > E > A.

---

## Part 6 — strongest narrative, and where the DECLINEs go

1. **Is Section 5 the intellectual centre?** Yes — under F it is not a caveat
   section but the results section.
2. **Complementary or redundant?** Complementary. A and B both attack *aggregate*
   sufficiency from different endpoints; C attacks *destination* sufficiency. Two
   levels, three demonstrations. Not redundant.
3. **Does one weaken another?** One risk: B's paired tests are ns, so if B is
   presented first the reader discounts the section. **Order A → C → B.** A is the
   strongest and fully row-level; C is artifact-backed formal verdicts; B is
   descriptive and lands better once the pattern is established.
4. **DECLINEs before or after the framework definition?** **After.** A decline is
   only meaningful once the reader knows what the gate demanded. Introduce the
   frozen conditions, then show the gate refusing.
5. **Do negatives read as findings?** Only with F. Under "we built a framework"
   they read as failed experiments; under "these properties dissociate" they *are*
   the result.
6. **Is Act 1 still substantive?** Yes — see Part 7.

## Part 7 — is Act 1 load-bearing?

**If Verify + License were removed:** Act 1 is a competent but post-CAST paper —
channel-specific gated steering with a good control battery. Publishable
somewhere; **not an ICLR novelty case**.

**If Act 1 were removed:** Verify + License has no object. An adjudication
procedure with nothing to adjudicate is not a paper.

**Both are load-bearing, but asymmetrically.** The *novelty* concentrates in
Verify + License; the *scientific legitimacy* depends on Act 1 being a real,
specific, non-trivial correction result. Under framing F, Act 1 gets a sharper
role: **it is the existence proof that makes the dissociation meaningful.** You
cannot show improvement dissociating from repair without producing genuine
improvement first. Act 1 should be written to that purpose, not as a
benchmark-gains section.

---

## Part 8 — novelty red team

```
NOVELTY_SUFFICIENT_POSITIONING_SENSITIVE
```

**Conceded without defence:** conditional/detector gating (CAST §3.1), Router
gating, additive activation steering, probing, random-direction controls (ITI).

**What remains:** directional error-channel formulation over a K≥3 native action
space · multiclass destination accounting (SOURCE_RETAINED / GOLD_ARRIVAL /
OTHER_WRONG) · exposure-conditional preservation · reverse/wrong-layer/mismatched
structural placebos · prospectively frozen evidence criteria · demonstrated
willingness to decline favourable point estimates · and the property distinction
Correctable vs Preserving vs Licensable. **No verified paper has any of the last
five.**

**The sentence that must appear in the Introduction:**

> Detector-gated conditional intervention is established prior work; our
> contribution is not the gate but the demonstration that behavioural improvement,
> target-directed repair, preservation and evidential sufficiency dissociate — and
> the adjudication procedure that reports them separately.

## Part 9 — significance red team

```
SIGNIFICANCE_SUFFICIENT
```

**Transfers:** the dissociation itself, and the measurement discipline —
destination-resolved accounting, exposure-conditional denominators, and
prospectively frozen decision criteria apply to *any* intervention on a
multi-class decision, not just tool use. The E1/E2 denominator trap and the
zero-exposure vacuity trap are errors other groups are certainly making.

**Does not transfer:** the specific channels, doses, layers, models; any claim
about intervention efficacy in general.

**Honest limit:** this is a methodology contribution demonstrated on one benchmark
family. It is sufficient for ICLR, not exceptional.

## Part 10 — rigor red team

```
RIGOR_STRONG
```
(not *exceptionally* strong — see below)

Genuinely strong: frozen criteria that produced **two real declines**; row-level
regeneration of every load-bearing number; E1/E2 discipline; vacuity labelled;
full placebo battery; provenance tiering enforced by a script that fails on drift;
score-space non-significance disclosed rather than buried; Phi variability
disclosed proactively; dose confound tested and reported as confounded.

**The single most dangerous remaining rigor criticism:** *the predeclared Level-2
threshold 0.6042 is Qwen3-8B's own CI lower bound.* The single formal success is
adjudicated against a boundary derived from itself. This is defensible for
**future** effects — that is what a predeclared threshold is for — but a sharp
reviewer will note the sole ADMIT could not have failed the threshold it defined.
**It must be disclosed in the text, not left to be discovered.** Nothing else in
the package comes close to this in danger.

---

## Part 11 — tier stress test

**C2-1** (Tier B floor, MetaTool): **not misleading.** MetaTool carries *breadth*,
not the correction claim; the load-bearing content is Phi/Q8B/Q4B/Gemma, all Tier A.
Disclose that MetaTool has no per-sample records.

**C3-1** (Tier B floor, four of five Phi controls aggregate-only): **not
misleading, but must be stated in the caption.** The claim is an *ordering* over
five arms, and the ordering is what the aggregate artifact supports. Do not present
destination breakdowns for the four control arms — they do not exist.

No headline depends on Tier C.

## Part 12 — model/scale framing

Combining historical small and modern large models produces **both** breadth and
heterogeneity confusion. Exact permitted framing:

> We evaluate across four model families and 3.8B–9B parameters, under two
> protocol generations that we report separately. Scale was not varied under a
> common protocol, and we make no claim about how correctability varies with scale.

Forbidden: scaling law, larger-is-more-correctable, architecture determines rung.

## Part 13 — mechanism story

**Correctly classified as strong supporting.** The shortest version that earns its
place — three sentences, no more:

> Readability does not locate a usable intervention: Router AUC stays flat at ~0.96
> across layers where the recovered direction degrades to noise and behavioural
> benefit varies four-fold. Correction quality therefore depends on estimator,
> channel matching, injection site, gating and dose rather than on the model alone.
> And the damaged samples are a fixed subpopulation invariant to direction choice
> and unidentified by Router confidence, so preservation can only be bought by not
> firing — which is why verification and licensing are separate stages.

Anything longer competes with the headline.

---

## Part 15 — reviewer simulation

**Reviewer A (steering).** *Borderline.* Positive: the control battery is unusually
complete. Negative: the mechanism is CAST's, and score-space nearly matches at
37 vs 38. Answerable — under framing F the score-space result is a *product* of the
contribution, and the Introduction sentence pre-empts the misread. **Rebuttal can
move them.**

**Reviewer B (statistics).** *Weak accept → accept*, and the most likely advocate.
Positive: frozen criteria with real declines, tiered provenance, denominator
discipline. Negative: the 0.6042 self-derived threshold; one formal success; Tier B
floors. **Rebuttal works only if disclosed in the submission.** If they find the
threshold circularity themselves, it becomes a reject.

**Reviewer C (agents/tool-use).** *Borderline / weak reject.* Positive: pre-execution
decisions are under-studied and matter. Negative: one benchmark family carries
licensing, ≤9B, no end-to-end agent task, and the practical takeaway is a caution
rather than a deployable gain. **Rebuttal only partially works** — this is
irreducible without new experiments, which are out of scope.

## Part 16 — ICLR 2027 criteria

```
ICLR_BORDERLINE_COMPETITIVE
```

Rigor **strong**; limitations honesty **strong**; reproducibility **strong**;
related-work positioning **adequate and fully dependent on execution**; novelty
**sufficient but framing-sensitive**; significance **sufficient**; clarity **the
binding constraint** — four models, two protocol generations, three dissociations,
five stages and a tier system into **9 pages** is genuinely hard. Deadlines:
abstract 18 Sep, paper 25 Sep 2026.

## Part 17 — acceptance range (judgement, not calibration)

**Scenario A, competent execution: ~25–35%.**
**Scenario B, strong execution: ~45–55%** — raised from my earlier 40–50% *only if*
framing F is adopted, because it converts a framework paper into a findings paper,
which reviewers score differently.

Main movers: (1) framing F vs a framework pitch; (2) whether the 0.6042 circularity
is disclosed or discovered; (3) CAST handled at method introduction; (4) whether
Section 5 is unmistakably the centre; (5) 9-page legibility.

**Strongest rejection scenario:** a reviewer reads it as "CAST plus an evaluation
checklist", notes one formal success on one benchmark family at ≤9B, observes that
the cheap score-space comparator nearly matches, and concludes the empirical payoff
does not justify the machinery — while a second reviewer independently finds that
the licensing threshold was derived from the single result it licenses. Those two
observations together are fatal, and both are pre-emptable in the writing.

---

## Part 18 — final story

**A. One sentence.**
> SAKIKO corrects pre-execution tool-decision errors by intervening on internal
> state, and shows that behavioural improvement, target-directed repair,
> preservation and evidential sufficiency are four independent properties — so it
> adjudicates them separately instead of reporting one number.

**B. One paragraph.**
> Tool-using models choose the wrong action before execution, and these errors
> recur as a small number of directional channels. Intervening on channel-specific
> internal directions improves them measurably and specifically: the real direction
> beats random, reversed, wrong-layer and mismatched controls, and no matched random
> direction reaches it in either model tested. But improvement turns out not to mean
> repair. One setting improves aggregate accuracy while breaking the majority of the
> correct samples it touches; two interventions with near-identical aggregate gain
> send their corrections to measurably different places; and two models with
> favourable destination estimates are formally declined because the evidence is too
> imprecise. SAKIKO therefore resolves every outcome into where behaviour actually
> went and what it cost, and licenses only the strongest claim the evidence carries.

**C. Introduction beats (six).**
1. Pre-execution tool decisions are consequential and fail in structured ways.
2. Those failures are internally readable and can be selectively corrected.
3. **But our strongest correction breaks most of what it touches** — stated here,
   with the number, not deferred to Section 5.
4. Improvement, repair, preservation and evidential sufficiency are therefore
   distinct; we show each dissociating.
5. Detector-gated intervention is prior work; the adjudication is ours.
6. Contributions.

**D. Reviewer takeaway six months later.**
> *"That's the paper where a +55 net improvement broke 55% of the correct samples
> it touched — and where their own gate refused two of three models."*

## Part 19 — contribution order

1. **The four-way dissociation** (improvement / repair / preservation / evidential
   sufficiency) — the finding.
2. **Destination-resolved verification** — the instrument that makes it visible.
3. **Evidence-graded licensing with demonstrated declines** — the procedure.
4. **Channel-specific correction across four families** — the object that makes 1–3
   meaningful.

Mechanism is supporting. **Reproducibility is not a headline contribution.**

---

## Part 20 — before figures

```
ONE_NONEXPERIMENTAL_FIX
```

**Disclose in the manuscript that the predeclared 0.6042 threshold is the CI lower
bound of the single licensed effect.** It is defensible as a standard for future
effects, and indefensible if discovered rather than stated. One paragraph in
Section 6. No analysis required — the number is already in the frozen artifact
(`target_hit_ci95 = [0.60415684, 0.84615385]`).

## Part 21 — verdict

```
FINAL_STORY_CONFIRMED_BUT_POSITIONING_BORDERLINE
```

Borderline **on positioning, not on science**. The evidence is sound and the chain
holds. Whether this is accepted turns on whether it is written as a findings paper
(framing F) or a framework paper (framing E) — and on one disclosure.

**Final unified chain:** `Discover → Detect → Correct → Verify → License`
**Final paper-level thesis:** improvement, repair, preservation and evidential
sufficiency are independent; SAKIKO adjudicates them separately.
**Final ICLR positioning:** a findings paper about intervention evaluation, whose
framework is the consequence of the finding — not a new steering primitive.
**Exact next action:** `REGENERATE_FINAL_FIGURES_FROM_93_ROW_DATABASE`
