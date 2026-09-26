# Part 13 — the surviving mechanism claim, at three scales

Derived from Parts 1–12, not chosen in advance.

## Full internal scientific statement

> Tool-decision errors concentrate into a small number of recurring directional
> transitions (~78% of errors in three channels), whose internal states are
> linearly readable (probe AUC 0.77–0.96) as separable directions inside
> overlapping clouds rather than as distinct clusters. Channel-specific directions
> recovered from that structure are behaviourally load-bearing — they outperform
> random, reversed, wrong-layer and mismatched-channel controls, and are stable
> under bootstrap resampling (|cos| 0.92–0.995). But readability does not locate a
> usable intervention: Router AUC stays flat at ~0.96 across layers where the
> recovered direction degenerates to noise (cos with PC1 = 0.0002, bootstrap
> stability 0.82) and behavioural benefit varies four-fold. Correction quality is
> therefore a property of the full design — channel matching, estimator choice
> adapted to channel geometry, separated observation/estimation/injection sites,
> Router gating, dose and exposure — not of the model alone. Gating improves
> intervention quality in many evaluated settings, but not all: removing it reverses
> the sign of the effect in two model-corpus settings, while on Gemma-2-9b the
> ungated arm is as strong as the gated one. The correction is distributed
> across the representation subspace rather than carried by a single semantic axis,
> and the samples it damages are a fixed subpopulation invariant to the direction
> chosen, so preservation can only be bought by not firing. Finally, movement is
> not arrival: about 30% of Phi-3.5's source exits (46 of 153) land on a third
> wrong action, equal-Net
> configurations differ in destination composition, and destination quality varies
> sharply and non-monotonically with dose. Aggregate improvement is therefore not
> sufficient evidence of repair, which is why SAKIKO pairs correction with
> destination-resolved verification and evidence-graded licensing.

## Main-paper mechanism claim (1–2 sentences)

> Tool-decision errors form linearly readable, channel-specific structure that
> supports targeted intervention, but readability does not imply a usable
> correction direction and behavioural movement does not imply arrival at the
> correct action: correction quality depends on channel matching, estimator,
> injection site, gating and dose; removing the gate reverses the sign of the effect
> in two model-corpus settings but not in others.

## Abstract-safe phrase (one clause)

> *— correction depends on channel-specific structure and on design choices such
> as gating, and
> behavioural movement alone does not establish that the intervention arrives at
> the correct action.*

## Forbidden wording

- "we identify a causal circuit"
- "correctability is intrinsic to architecture"
- "margin determines repair"
- "all channels share one correction direction"
- "activation steering is universally superior"
- "Router readability predicts repair"
- "dose-independent correctability"
- any orthogonality claim from cross-layer cosines
- Phi's +62/−32 pair quoted against the frozen +55 reference
