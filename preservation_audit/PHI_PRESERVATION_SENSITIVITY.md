# Phi preservation sensitivity audit

```
PHI_PRESERVATION_FAILURE_ROBUST
```

No new inference. All figures below come from surviving TEST artifacts.
Validation sweep cells were excluded from evidence.

---

## The identity that makes this auditable

For the frozen run, summing `channel_stats[*].n_routed` gives the number of rows
the Router actually **fired** on:

```
R = 137 + 83 + 73 = 293      routed source errors = 200
R - 200 = 93 == frozen Router-exposed baseline-correct   OK
```

`fixed` and `broke` also sum exactly across channels (107, 52). So R is a valid
firing count, and since exposed-correct ≤ R and exposed-correct ≤ E1n = 264,

```
collateral_rate = broke / exposed_correct  >=  broke / min(R, 264)
```

This is a **rigorous lower bound**, computable for every run that recorded
`channel_stats`, without row-level records. For seed 42 the bound is 19.7% and
the true value is 55.9% — the bound is loose, which is what makes it safe.

---

## All artifact-backed Phi TEST realizations

| run | provenance | thr | α | Fixed | Broke | Net | R fired | E1 n | E1 rate | **LB exposed-correct rate** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **p0 / p1 seed 42** | **frozen locked** | 0.4 | 10.0 | 107 | 52 | +55 | 293 | 264 | 19.7% | **19.7%** (true **55.9%**) |
| p2_placebo REAL_LOCKED | frozen locked | 0.4 | 10.0 | 107 | 52 | +55 | 293 | — | — | 19.7% |
| p1 seed 123 | same locked config | 0.4 | 10.0 | 105 | 51 | +54 | 304 | 264 | 19.3% | **19.3%** |
| p1 seed 456 | multiseed variant | 0.6 | 8.0 | 94 | 41 | +53 | 259 | 264 | 15.5% | **15.8%** |
| p1 seed 789 | multiseed variant | **0.9** | 8.0 | 76 | **16** | +60 | 202 | 264 | 6.1% | **7.9%** |
| p1 seed 2024 | multiseed variant | 0.6 | 8.0 | 107 | 40 | +67 | 250 | 264 | 15.2% | **16.0%** |
| p2_independent REAL_LOCKED | same config, separate execution | 0.4 | 10.0 | 106 | 44 | +62 | **243** | — | — | **18.1%** |

Only seeds 42 and 123 share the frozen locked configuration (thr 0.4 / α 10.0).
Seeds 456, 789 and 2024 **re-selected their own threshold and alpha** on their own
validation splits and are therefore configuration variants, not seed replicates
of the frozen arm. They are reported separately and not pooled.

---

## Primary question — answered

> Does any artifact-backed Phi TEST realization bring exposed-correct collateral
> close to or below the frozen preservation budget (5%)?

**No.** The minimum lower bound across all seven realizations is **7.9%**, which
is 1.6x the budget — and it is a *lower* bound, so the true rate is higher.

**The Broke=16 realization is the trap, and it does not help.** Seed 789 looks
dramatically better on raw Broke (16 vs 52) only because it ran at threshold
**0.9**: the Router fired on 202 rows instead of 293. Fewer breaks reflect **less
exposure, not better preservation**. Its collateral rate remains at least 7.9%.
Reading Broke=16 as evidence of preservation would be a denominator error of
exactly the kind the 52/264-vs-52/93 prohibition already guards against.

Across every realization, lowering exposure lowered Broke and Fixed together;
no configuration converted the effect into a preserving one.

---

## Run-to-run variation — source, and a correction

Broke ranges **16 to 52** and Net ranges **+53 to +67** across surviving TEST
runs. Two components:

1. **Configuration variation** (seeds 456/789/2024) — thresholds and alphas were
   re-selected per seed. Expected, and not a reproducibility problem.
2. **Same-configuration divergence** — `p2_independent_baselines.json` reports
   106/44/+62 where the frozen run reports 107/52/+55, using imported code, the
   same `SEED`, the same locked config and the same splits.

**Correction to the earlier reconstruction verdict:** that document attributed
(2) to non-determinism in hooked generation. That attribution is not supported.
The independent run fired on **243** rows versus **293** — a routing-level
difference, and routing is deterministic given activations and trained routers.
Generation non-determinism cannot produce it. The actual cause is **not
determinable**: the entire repository was squashed into a single
`Initial commit: SAKIKO project` on 2026-03-31, so no file-level history exists
to show whether the shared runner changed between the two executions.

The regression-gate verdict is unaffected — the target was still not reproduced,
which is what the gate tests. Only the explanation changes, from "known cause" to
**"undetermined cause, and undeterminable from surviving history."** That is the
weaker and more honest statement.

---

## C4-1 — revised wording

**Old:** *aggregate Net conceals a preservation failure* (52/93 = 55.9%).

**Revised strongest permitted wording:**

> In the historical Phi-3.5 setting, aggregate improvement coexists with severe
> damage to the baseline-correct samples the Router fires on. The point estimate
> varies across executions — Broke ranges from 16 to 52 and Net from +53 to +67
> across surviving TEST runs, partly because per-seed threshold selection changes
> how many samples are exposed — but **every** artifact-backed realization
> violates the preservation budget by at least a factor of 1.6, and the exactly
> resolved case reaches 52/93 = 55.9%. The qualitative preservation failure is
> robust to the specific realization; the exact rate is not.

**Prohibited:**
- 52/93 presented as a stable constant, or without the execution-variation caveat
- 52/264 = 19.7% as *the* collateral rate (that is the E1 estimand)
- treating Broke=16 (seed 789) as improved preservation — it is reduced exposure
- pooling seeds 456/789/2024 with the frozen arm as if same-configuration replicates
- attributing the 107/52 vs 106/44 divergence to generation non-determinism

C4-1 **survives**, with the point estimate downgraded to a range and the
qualitative claim retained.
