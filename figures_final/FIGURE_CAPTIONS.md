# Paper-ready figure captions — frozen

> **Superseded for the manuscript (2026-09-11):** final captions, in the final figure
> numbering, are in `manuscript/figure_captions.md`. This file describes the current
> figure files and is kept for provenance.

Every number resolves to `final_evidence/FINAL_PAPER_EVIDENCE.csv` (113 rows).
Captions are self-contained: each carries the caveat that prevents its own
misreading.

---

## Figure 0 — `fig0_dissociation_map.pdf` · MAIN (opens Section 5)

> **Four questions, not one.** An intervention that changes a model's action must
> pass four distinct questions before a repair claim is supported, and improving
> behaviour settles only the first. Each dashed branch marks a question at which an
> intervention in this work fails despite having improved behaviour: 46 of 153
> source exits reach a third wrong action rather than the correct one, and 52 of
> the 93 already-correct rows the detector fires on are broken (both Phi-3.5,
> frozen realization); and two settings whose target-hit point estimates are
> favourable, 0.5781 and 0.6296, are refused by prospectively frozen criteria
> because their intervals are too wide at the available sample support. A refusal
> denotes insufficient evidence under the evaluated protocol, not intrinsic
> uncorrectability. Panel content is conceptual; the underlying measurements appear
> in Figures 3 and 4.


## Figure 1 — `fig1_framework.pdf` · MAIN

> **SAKIKO follows a tool-decision correction through a complete evidence pipeline.** A
> tool-using model commits to an action over a multiclass space before any tool
> executes. SAKIKO first discovers the directional error channels a given model
> actually exhibits, `c = (y* → ŷ₀)`; detects per input whether the observed state
> belongs to one of them, intervening only when a channel fires; and applies a
> channel-specific correction configured by estimator, observation and injection
> site, and a frozen dose. It then resolves the post-intervention decision `ŷ₁`
> into retention of the source error, arrival at the correct action, or
> redistribution to a third wrong action, while separately tracking damage to
> decisions that were already correct. These outcomes are adjudicated against the
> prospectively frozen hierarchy from Adjudicable to Licensable. Separating
> Correctable, Preserving and Licensable is what prevents behavioural movement, or
> a favourable point estimate, from being read as reliable repair: a formal DECLINE
> denotes insufficient evidence under the evaluated protocol, not intrinsic
> uncorrectability. The three horizontal layers show, respectively, what happens to
> the model, what SAKIKO does, and how the evidence is interpreted. No measured
> data appear in this figure.

## Figure 2 — `fig2_correction.pdf` · MAIN

> **Correction is real, stable and structurally specific.**
> **(a)** Net corrections on locked test sets: Phi-3.5 across five seeds
> (+53 to +67, 5/5 positive) and Qwen2.5-7B (+79; 92 fixed, 13 broken). The two
> settings use different protocols and are not a controlled scale comparison.
> **(b)** Corrupted variants of the intervention. For Phi-3.5, replacing the
> direction (+14), reversing it (+14), moving the injection layer (+3) or applying
> a channel's direction to the wrong channel (−17) does not reproduce the real
> effect (+55); for Qwen2.5-7B, reverse gives +16 against a random mean of +25.2.
> These control arms are aggregate-only; no per-sample records survive for them.
> **(c)** Under the sealed modern protocol, none of 59 matched random directions
> reaches the real direction's gold arrivals in either model (max 8 of 38;
> max 0 of 17).

## Figure 3 — `fig3_dissociation.pdf` · MAIN — the central figure

> **Behavioural improvement does not establish repair.**
> **(a)** On Phi-3.5 the intervention yields +55 net corrections while breaking 52
> of the 93 baseline-correct rows the Router fires on. The 56% figure is the frozen
> realization; across artifact-backed historical executions the broken count ranges
> from 16 to 52, and every realization exceeds the 5% preservation budget.
> **(b)** On Qwen3-8B an activation intervention and a score-space comparator
> achieve near-identical aggregate gains (channel-level Net +38 vs +35; 38 vs 37 gold arrivals) but
> differ in destination composition (14 vs 21 other-wrong; target-hit 0.731 vs
> 0.638). Paired comparisons are not significant after multiplicity correction
> (McNemar exact p = 1.0 and 0.189) and the comparison was not prespecified; this
> panel is descriptive and does not establish superiority of either intervention.
> **(c)** Target-hit with 95% percentile-bootstrap intervals resampling channel
> errors (10,000 draws). The 0.50 criterion was fixed before any sealed evaluation
> and applied unchanged to all three settings. Qwen3-8B clears it; Qwen3-4B and
> Gemma-2-9b have favourable point estimates (0.578, 0.630) whose intervals do
> not, and both return a formal DECLINE. Bounds are those recorded in the frozen
> verdict artifacts; independent re-draws differ in the third decimal, and no
> verdict depends on that precision.

## Figure 4 — `fig4_licensability.pdf` · MAIN

> **Correctable is not Licensable.** Descriptive property evidence (left) against
> the frozen adjudication (right) for every evaluated setting; historical
> post-hoc settings are shaded and were not adjudicated under the sealed protocol.
> Qwen3-4B satisfies every property yet is declined, because its target-hit
> interval does not clear the criterion at 64 source exits; Gemma-2-9b is declined
> on the same interval conditions and additionally breaks 1 of 11 exposed-correct
> rows. Qwen3-8B's preservation is marked partial: it is certified on all 211
> baseline-correct rows (0/211, one-sided 95% upper 0.0141) but vacuous on the
> exposure-conditional estimand, where only 6 rows were exposed. Correctable is
> undefined for Qwen2.5-7B's binary corpus, in which leaving the source mode is
> necessarily arriving at gold.

## Figure 5 — `fig5_mechanism.pdf` · APPENDIX

> **Why the stages are separated.**
> **(a)** For the `ca_direct` channel, Router AUC is approximately 0.96 at every
> layer while the recovered direction's norm grows from 2.12 to 14.26 and
> best-validation net rises from +13 to +46. At the lowest layer the direction is
> essentially noise (cosine with PC1 = 0.0002). Detectability does not locate a
> usable intervention.
> **(b)** Where the difference-in-means direction is poorly aligned with the
> leading principal component (cosine 0.41), the PCA-1 estimator reaches +50 on
> the locked test against +36 for difference-in-means.
> **(c)** Gold arrivals per broken row, gated versus ungated. Gating raises this
> ratio in all three settings where both arms are directly comparable. It does not
> always raise raw net: in two settings the ungated arm attains a similar or larger
> net by firing on substantially more samples and absorbing more damage.

---

## Caption prohibitions

No significance markers anywhere. No "outperforms" for the score-space panel. No
"admitted" for Qwen3-4B or Gemma. Never draw or describe 0.6042 as a boundary in
Figure 3c: it is Qwen3-8B's own interval lower bound, not a criterion. Every sealed
gate used 0.50 (S-24).
