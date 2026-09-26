# Reproducibility and integrity statement

## For the paper (Methods or Appendix)

> All load-bearing reported outcomes were mechanically regenerated from archived
> row-level records during a final evidence audit. Destination counts, exposure
> denominators and specificity nulls were recomputed from per-sample predictions
> rather than copied from summary files; the frozen values for Qwen3-8B
> (72/38/14/52), Qwen3-4B (118/37/27/64), Gemma-2-9b (96/17/10/27) and Phi-3.5
> (293 fired / 200 routed errors / 47 / 107 / 46 / 93 exposed-correct / 52 broken)
> reproduced exactly. The matched-random nulls were re-derived across all 59 arms
> per model. A machine-readable evidence database backs every number in the
> manuscript.

Reported as diligence. **Not claimed as a contribution.**

## Required proactive disclosure — Phi historical variability

> One caveat is disclosed proactively. The frozen historical Phi run reports
> Fixed 107 / Broke 52 / Net +55. A separate contemporaneous execution importing
> the same evaluation code, the same seed, the same locked configuration and the
> same split reports 106 / 44 / +62, and fired on 243 rows rather than 293.
> Because routing is deterministic given trained routers, generation
> non-determinism cannot account for the difference, and the cause cannot be
> established: the source repository was squashed into a single initial commit, so
> no file-level history survives to show whether the shared runner changed between
> executions. We therefore report the Phi collateral point estimate as
> execution-dependent. **The qualitative conclusion is unaffected:** across every
> artifact-backed realization — five seeds and one independent execution — the
> exposed-correct collateral remains above the 5% preservation budget, with a
> minimum margin of 6.1% against 5% at the highest threshold examined.

**Prohibited:** presenting 52/93 without this caveat; claiming the historical Phi
pipeline is exactly reproducible; attributing the divergence to a known cause.
