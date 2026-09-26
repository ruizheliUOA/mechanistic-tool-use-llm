# Final novelty positioning

## The frozen statement

> SAKIKO turns a correction from an **assumed** repair into an **adjudicated** one,
> by coupling channel-specific
> intervention with multiclass destination resolution, preservation accounting,
> structural specificity controls and evidence-graded licensing.

## Against CAST (Lee et al., ICLR 2025) — the nearest work

**What CAST already does, verified in §3.1:** sample-conditional intervention
gated by a learned detector, `h' ← h + f(sim(h, proj_c h))·α·v` with
`f = 1 if sim > θ else 0`. **This is architecturally SAKIKO's Router.** It is
published, at a top venue, and SAKIKO must not claim it.

**What CAST does not do** (all section-verified): binary refuse/comply label space
only; no destination categories when an output changes; no random, reverse or
wrong-layer controls; preservation only as a harmful-minus-harmless proxy; and
threshold selection by **post-hoc grid search**, explicitly not a prospectively
frozen criterion.

## The novelty boundary — exact

**Must not claim as novel:** conditional or detector-gated intervention · Router
gating · activation addition · representation reading · layer/site selection ·
random-direction controls (ITI, §4.3).

**Remains distinct — as a composition, verified against six full-text papers:**
multiclass destination-resolved accounting (0/6) + target-directed controls
including reverse and wrong-layer (0/6) + a preservation endpoint (0/6 full) +
prospectively frozen evidence criteria (0/6) + an explicit admit/decline rule
(0/6) + a K≥3 native action space (0/6).

```
FULL_COMBINATION_FOUND: NO within the verified corpus
```

## Permitted related-work wording

> "Among the closest works identified by our search, we found no method combining
> destination-resolved multiclass accounting, target-directed intervention
> controls, a preservation endpoint, prospectively frozen evidence criteria and an
> explicit admit/decline rule. Detector-gated conditional intervention is
> established prior work (CAST); our contribution is the adjudication procedure
> around it, not the gating mechanism."

**Forbidden:** "no prior work does this" · "the field generally" · "destination
evaluation is uncommon" · any proportion (0/6, 1/6) presented as a field rate. The
corpus is a **convenience sample of six**, not a survey.
