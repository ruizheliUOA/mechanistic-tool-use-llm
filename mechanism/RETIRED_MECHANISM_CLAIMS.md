# Retired mechanism hypotheses

Internal record. Purpose: prevent retired interpretations from returning.

| hypothesis | why plausible | experiment | result | status |
|---|---|---|---|---|
| a single universal global steering direction suffices | one "tool-call propensity" axis is parsimonious | global steering baseline (Phi); MetaTool dual-channel | GLOBAL_STEERING **+27/−27 = Net 0**; MetaTool improves both directions on disjoint sets | **FALSIFIED** |
| readable ⇒ actionable | high probe AUC feels like a usable direction | ca_direct layer sweep | AUC flat ~0.96 while Net +13→+46; L16 direction is noise (cos 0.0002, bootstrap 0.82) | **FALSIFIED** |
| steerable ⇒ correctable | changed behaviour looks like fixed behaviour | Phi destination resolution | 153 exits → 107 gold, **46 other-wrong** | **FALSIFIED** |
| cross-layer cosine indicates semantic alignment | cosine is the natural similarity | same-layer recompute | native-layer 0.007/0.012 vs common-layer **0.488/0.706** | **FALSIFIED** (artifact) |
| representational similarity transfers intervention | high R² mapping should carry the direction | Phi→Qwen mapping | ridge R²=0.68 → Net +10; procrustes R²=0.39 → +23 but **wrong-layer +35 > real** | **FALSIFIED** |
| correction is carried by one semantic PC | a single interpretable axis would be elegant | full 20-PC audit | no single PC exceeds rfi_net=7; only PC9 removal helps | **FALSIFIED** — correction is multi-PC emergent |
| damage depends on the chosen direction | better direction ⇒ less damage | PC-subset comparison | Broke set **Jaccard = 1.000** across configurations | **FALSIFIED** — damage is a subpopulation property |
| baseline margin is a sufficient destination predictor | margin is the obvious boundary statistic | displacement / margin calibration | rank-inverse to margin dependence; dose confound | **NOT_IDENTIFIABLE** |
| activation beats score-space in general | activation is "deeper" | paired destination tests | d_act − d_ss = −1.538; **all ns after Bonferroni** | **REFINED** → SUPPORTING only |
| correctability is intrinsic to a model | model families "differ" | exit-rate-matched comparison | Δtarget_hit **flips sign** with match level | **CONFOUNDED** |
| exact additivity is independent MetaTool evidence | exact B+C=D looks striking | overlap analysis | `partition_overlap = 0` makes additivity near-mechanical | **REFINED** → consistency check only |
| differential/source suppression is the primary mechanism | suppressing the source mode explains exits | destination decomposition | 46 other-wrong; ca_tc reverse retains 46% of fixes | **PARTIAL** — a real component, not the mechanism |
| first-order gold-cone geometry explains arrival | linear reachability toward gold | off-axis analysis | Q_channel 0.236–0.663, substantial off-axis energy | **NOT_IDENTIFIABLE** |
| zero collateral means preservation succeeded | 0.0% looks ideal | exposure check | `router_fired_baseline_correct_exposure = 0` at every dose | **FALSIFIED** — **VACUOUS** |
