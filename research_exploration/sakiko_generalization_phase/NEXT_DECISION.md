# Next decision

## What generalization changed, honestly

**Improved:**
- The property/licence separation resolves the "DECLINE ≠ uncorrectable" ambiguity that the
  adversarial audit identified as the framework's central confusion. This is a real gain and it
  costs nothing.
- Ten conditions collapse to **six benchmark-agnostic constructs** with no loss of pathology
  coverage, and construct **F (yield) repairs the demonstrated low-yield gap**.
- The algorithmic core is **already action-space agnostic** — the destination classifier never
  references a W2C label. "SAKIKO is a W2C checklist" is false at the algorithm level.
- **arXiv 2607.02577** (evaluator error 9.8–30.5% across four major tool benchmarks)
  independently vindicates the eligibility discipline and adds requirement 10 to the contract.
- **arXiv 2606.29054**'s impossibility result gives a *theoretical* reason to expect high
  refusal rates, reframing 8-of-9 refusals as structurally expected.
- **RUT-Bench** is a genuinely new candidate the previous shortlist did not contain, differing
  from W2C on data source, user distribution and interaction structure simultaneously.

**Worsened:**
- **arXiv 2607.24343** measures "over-intervention" on benign fields **with conformal risk
  control**. The exposure-conditional collateral contribution is no longer unique, and the
  neighbour has a finite-sample guarantee where SAKIKO has a 0.05 convention. Reviewer C's
  attack is now backed by a published alternative.

**Unchanged:**
- One benchmark, one ADMIT, uncalibrated thresholds.

## Decision

```
B. GENERALIZED CORE MATERIALLY IMPROVES PAPER — REWRITE, NO NEW BENCHMARK EXPERIMENT
```

Not C or D. A second-benchmark feasibility study is now **less** attractive than it looked: the
best candidates carry 9.8–30.5% evaluator error, and SAKIKO's endpoints are differences of tens
of samples. RUT-Bench is worth an adapter screen as *future work*, not as a paper-critical
dependency — and screening it properly means validating its gold against the standard 2607.02577
just showed the field is failing.

Not A: freezing as-is would ship the conflation the audit found, and would omit the yield gap
and the six-construct simplification, both of which are free.

Not E: generalization did not rescue the calibration problem, but it did produce three
defensible gains (separation, minimal core, surface-agnostic exposure) and a Layer-C novelty
position that survives the fresh search.

## The thesis to write

> Tool-use intervention reliability is not one number. SAKIKO decomposes a candidate correction
> into **adjudicability, readability, causal specificity, destination correctness, preservation
> and evidential precision**, and localises which of them fails. Applied prospectively to nine
> settings across four model families, it refused eight at six distinguishable stages —
> including the authors' own prior published result.

## The one thing to add before submission, and it is not an experiment

Re-express **preservation** as a conformal risk-controlled quantity. It is CPU-only on existing
per-sample records, it replaces the least defensible part of the framework with the most
defensible machinery available, and it directly answers the strongest reviewer objection. Its
limitation must be stated plainly: exposed-correct populations of 0, 6, 11 and 23 rows may be
too small for a useful bound, in which case 2606.29054's impossibility result *is* the finding.
