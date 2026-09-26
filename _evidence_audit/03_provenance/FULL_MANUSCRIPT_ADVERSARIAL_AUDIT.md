# Full-manuscript adversarial audit

Six reviewers against the complete draft set, checked against the claim matrix, number
provenance, closest-work matrix and novelty statement.

## A — activation steering / representation editing

**Strongest contribution:** destination resolution among K≥3 wrong actions; no work in the
closest-work matrix marks YES on it. **Most serious issue:** none factual — the draft concedes
every component as prior art in Related Work paragraph 2. **Novelty objection:** "the machinery is
all borrowed." Fair, and the paper says so first. **Experimental objection:** one licensed
setting. **Overreach:** none found; Discussion 6.5 volunteers that specificity did most of the
filtering. **Missing citation:** none identified. **Score 5, confidence 4.**

## B — tool-use / agent benchmarks

**Strongest contribution:** pre-execution action selection is the right target, and automatic
channel discovery over all ordered transitions is not standard. **Most serious issue:** the
benchmark is single-turn and its gold has three effective classes, which the draft states.
**Novelty objection:** mild. **Experimental objection:** one benchmark for all licence outcomes.
**Overreach:** watch that "framework instantiation" never drifts to "validation." **Missing
citation:** the 2026 evaluator-validity audit is cited in Related Work; keep it.
**Score 5, confidence 3.**

## C — statistics / selective inference

**Strongest and most favourable reviewer.** **Strongest contribution:** prospective sealed
one-shot execution with hash-frozen configuration, plus operating-characteristic simulation
showing the licence admits 93% at the licensed effect and 19–25% at the refused ones — analysis
almost no steering paper performs. **Most serious issue:** thresholds are conventions with no
calibration argument; the paper concedes this. **Novelty objection:** conformal risk control
predates the preservation reframing; the draft states this explicitly in Discussion 6.4.
**Experimental objection:** N=3 formal outcomes cannot calibrate anything. **Overreach:** none —
the exposure-conditional table with 0/6 → 0.393 is volunteered in Results 3.1, not buried.
**Score 6, confidence 4.**

## D — external validity / benchmarks

**Strongest contribution:** the six-population eligibility screen with a named binding gate per
candidate. **Most serious issue:** channel identity and benchmark are perfectly confounded, so the
cross-ontology channel observation cannot be elevated. Discussion 6.6 keeps it as an observation.
**Novelty objection:** none. **Experimental objection:** licence outcomes on one benchmark whose
gold is synthetic and unaudited by the authors. **Overreach:** none found. **Missing citation:**
none. **Score 4, confidence 4.**

## E — mechanistic interpretability

**Strongest contribution:** three documented negative mechanism searches, reported as negatives.
**Most serious issue:** the paper explains nothing about *why* the effects occur. **Novelty
objection:** it is an evaluation contribution, not an interpretability one. **Experimental
objection:** no mechanism-level evidence. **Overreach:** none — Discussion 6.9 lists mechanism as
unknown. **Score 5, confidence 3.**

## F — general ICLR

**Strongest contribution:** the without-SAKIKO counterfactual — four of eight settings would have
been reported more favourably, three as successes, one of which the authors had published.
**Most serious issue:** is a negative-result evaluation framework an ICLR contribution?
**Novelty objection:** narrow after concessions. **Experimental objection:** one ADMIT, one
benchmark. **Overreach:** none; §6.8 states the sceptical case in full before answering it.
**Score 6, confidence 3.**

## Meta-review

**Scores 5, 5, 6, 4, 5, 6 — mean 5.2.** Borderline; no champion, one clear detractor (D), two weak
advocates (C, F). Consistent with every prior adversarial pass, which is itself a signal that the
drafts are not inflating.

The manuscript's defensive position is unusually strong for a borderline paper: **every objection
the panel raises is already stated in the paper, in the authors' own words, before the reviewer
reaches it.** §6.8 concedes the entire sceptical reading; §6.5 volunteers that destination
accounting changed no observed decision beyond specificity; §6.4 and Results 3.1 disclose that the
licensed setting's preservation claim is uncertifiable. That is the correct posture for this
evidence base and should not be softened during LaTeX assembly.

**Recommendation: weak accept, conditional on the claim discipline surviving formatting.** The
single largest remaining risk is not scientific but editorial — that "framework instantiation"
becomes "validated on two benchmarks" somewhere in compression.
