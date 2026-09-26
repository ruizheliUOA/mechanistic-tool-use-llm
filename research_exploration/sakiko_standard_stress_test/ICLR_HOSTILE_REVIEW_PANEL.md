# Hostile review panel

Each reviewer must answer: *if SAKIKO did not exist, what mistake would researchers be more
likely to make?*

## Reviewer A — activation steering / representation editing

**Contribution:** destination-resolved accounting and a refusal rule. **Objection:** every
component is prior art — Router gating, DiffMean, activation addition, random nulls. The one
ADMIT uses a standard method. **Useful?** Marginally: the incremental-value audit shows
specificity alone does most of the filtering, and specificity controls are already expected.
**Mistake prevented:** calling movement correction without checking destination.
**Score 5, confidence 4.**

## Reviewer B — tool-use agents

**Contribution:** pre-execution action errors are the right target and the taxonomy is real.
**Objection:** `direct` is never a gold label, so the "4-way" space is effectively 3-way; and
When2Call is single-turn MCQ, not an agent acting. **One benchmark fatal?** For any transfer
claim, yes. **Mistake prevented:** publishing Net +39 as cross-model replication.
**Score 5, confidence 3.**

## Reviewer C — statistics / selective inference

**Strongest reviewer for this paper.** **Contribution:** prospective one-shot sealed
evaluation with hash-frozen configuration is genuinely rare, and the power simulation showing
P(ADMIT) = 0.93 vs 0.19–0.25 is the kind of operating-characteristic analysis almost no
steering paper does. **Objection:** the licence is uncalibrated (N=3), the two destination CI
conditions are near-redundant, and the collateral estimand is diluted — Gemma's 1/223 = 0.45%
passes while its exposure-conditional 1/11 = 9.1% exceeds the bound. That is a **construct
error in the primary endpoint**, not a reporting nicety. **Mistake prevented:** treating an
underpowered null as mechanism failure. **Score 6, confidence 4.**

## Reviewer D — benchmarks / external validity

**Objection:** all formal evidence on one benchmark, and channel identity is perfectly
confounded with it. The "channel-dependent readability" result may be a When2Call ontology
artifact. **One benchmark fatal?** To generalisation, yes; to existence, no. The 9-candidate
triage is unusually strong diligence. **Mistake prevented:** assuming a channel that works in
one model transfers. **Score 4, confidence 4.**

## Reviewer E — mechanistic interpretability

**Objection:** no mechanism. Three mechanism searches (forecasting, gold-cone geometry,
preservation) returned negative or data-limited. The paper is about evaluation, not
understanding. **Useful?** Yes, but to practice rather than to interpretability.
**Mistake prevented:** inferring representational structure from behavioural movement.
**Score 5, confidence 3.**

## Reviewer F — general significance

**Objection:** one ADMIT, one benchmark, a rule whose thresholds are conventions. Is a
negative-result framework paper an ICLR contribution? **Counter:** the without-SAKIKO
counterfactual is concrete — four of eight settings would have been reported more favourably,
three as successes, and one of those was actually published as a success by these authors.
**Mistake prevented:** the field's default of reporting Net and calling it correction.
**Score 6, confidence 3.**

## Convergent answer to the panel question

Every reviewer named a real mistake, and the four recurring ones are: **calling movement
correction; ignoring wrong-to-wrong; ignoring collateral or measuring it on a diluted
denominator; treating underpowered evidence as mechanism failure.** Reviewers A and C note that
existing best practice already covers the first and part of the fourth. The two that are *not*
covered by ordinary practice are the **collateral estimand** and **non-vacuity**.
