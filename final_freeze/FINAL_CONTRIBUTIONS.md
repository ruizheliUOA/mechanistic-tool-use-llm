# Final contributions — four

### 1. A correction framework for pre-execution tool-decision errors
Channel-specific, Router-gated intervention on directional error channels
(`gold → predicted wrong action`), evaluated across **seven models spanning five
families and 3.8B–9B**, on two benchmarks. Establishes that a meaningful subset of
these errors is behaviourally improvable, with structural specificity: real
intervention beats random, reverse, wrong-layer and mismatched-channel controls,
and **0 of 59** matched randoms reach it in either of two models tested.

### 2. Destination-resolved verification
Multiclass accounting that separates `SOURCE_RETAINED` / `GOLD_ARRIVAL` /
`OTHER_WRONG` for errors and `CORRECT_RETAINED` / `BROKEN` for correct rows.
Demonstrates two things aggregate metrics cannot express: a Net-positive
intervention breaking **55.9%** of the correct samples it touches, and two
interventions with near-identical aggregate gain (channel-level Net +38 vs +35) landing at
substantially different destinations (target-hit 0.7308 vs 0.6379).

### 3. Evidence-graded licensing
The prospectively frozen ladder — Adjudicable, Readable, Steerable, Correctable,
Preserving, Licensable — with frozen criteria and an
explicit decline rule, plus a design-sensitivity curve showing the exit count
required to certify that most source exits reach the correct action grows steeply
as the true target-hit approaches 0.50. Correctability is treated as **protocol-conditional**, not as a
model property.

### 4. Design analysis explaining why the stages are separated
Readability does not locate a usable intervention (Router AUC flat at ~0.96 across
layers where the direction degrades to noise and benefit varies four-fold);
correction quality depends on estimator, channel matching, injection site, gating
and dose; and gating controls exposure volume rather than sample safety. This
motivates Verify and License rather than decorating them.

**Not claimed as a contribution:** conditional/detector-gated intervention,
activation steering, representation reading, or the Router mechanism — all prior
art. The reproducibility audit is reported as diligence, not as novelty.
