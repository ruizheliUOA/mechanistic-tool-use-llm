# Robustness and validation — final audit

## The seven sources of randomness, separated

| source | exists? | tested? | finding |
|---|---|---|---|
| **A. model/generation** | **NO** | n/a | greedy, `do_sample=False`, `use_cache=False`, TF32 off, `cudnn.deterministic`. Replay byte-identical: 144/144 (V1R3), 50/50 (ACEBench), 4-token generation bitwise (Qwen3.5 smoke). **Deterministic, not stochastic.** |
| **B. data split** | YES | **NEWLY QUANTIFIED** | 30 group-preserving alternative splits — see below. **The one material unquantified source, now measured.** |
| **C. Router training** | **effectively NO** | **NEWLY TESTED** | 20 `random_state` values on identical data: max coefficient difference **0.000e+00**, pairwise cosine **1.000000000000**, AUC spread **0.000e+00**. `ROUTER_CONDITIONAL_DETERMINISM`. |
| **D. direction estimation** | sampling variance only | YES | TRAIN→DEV cosine 0.9927 / 0.9932; wrong-sign fraction 0.0000; `wbar_norm` 11.1× and 6.5× the `1/√n` chance reference; split-half 0.908 (Llama). |
| **E. random-intervention null** | YES | YES | K=59 salt-derived per formal trial; 8 frozen DEV seeds; **0/59 randoms ≥ real in all three formal trials**, p = 0.016667. |
| **F. bootstrap / CI Monte Carlo** | YES | YES | 10,000-draw percentile bootstrap, fixed seeds; Clopper–Pearson exact for collateral. |
| **G. parser / measurement** | YES | PARTIAL | W2C: 0 unparsed, 0 ties, 0 exclusions across 4 models. ACEBench: 0.98% on a **102-row audited subset** whose stratification is undocumented. |

**K=59 is a specificity null, not a 59-seed replication.** It varies the intervention direction
under a fixed pipeline. It says nothing about split or training variance and must never be
described as end-to-end repetition.

## The new result: structural split sensitivity

30 alternative splits, gold-stratified with V2's exact quotas (390/322/392), group constraints
preserved, salt varied. Only pre-intervention quantities recomputed. No SEALED outcome inspected.

| model · channel | eligible across 30 splits | dev_ref min/med/max | verdict |
|---|---:|---|---|
| **Qwen3.5 `ca→tool_call`** | **56.7%** | 28 / 34 / 43 | **SPLIT_SENSITIVE** |
| **Qwen3.5 `ca→direct`** | **56.7%** | 28 / 34 / 43 | **SPLIT_SENSITIVE** |
| Qwen3.5 `ca→request_for_info` | 36.7% | 28 / 34 / 43 | SPLIT_SENSITIVE |
| Qwen3.5 `rfi→tool_call` | **100%** | 93 / 101 / 117 | ROBUSTLY_ELIGIBLE |
| Gemma `ca→tool_call` | **100%** | 33 / 41 / 49 | ROBUSTLY_ELIGIBLE |
| Gemma `ca→direct` | **100%** | 33 / 41 / 49 | ROBUSTLY_ELIGIBLE |
| Gemma `rfi→tool_call` | **100%** | 34 / 41 / 49 | ROBUSTLY_ELIGIBLE |

**This must be disclosed.** Qwen3.5's `ca→*` channels missed the reference floor by one row
(29 vs 30) under the frozen split — and the frozen split placed them in the **lower tail** of
the achievable distribution (median 34). Under a different, equally legitimate pre-registered
allocation they would have been eligible **more often than not**.

The verdict does not change: the split was frozen prospectively, which is the entire point of
prospective evaluation. But `QWEN35_NO_ACTIONABLE_CHANNEL` is a **split-sensitive** outcome, not
a structural property of the model, and the paper must say so.

By contrast **Gemma's eligibility was not luck** — 100% across all 30 splits on all three
channels. And Qwen3.5's `rfi→tc` is robustly eligible, so its **readability** failure (ROC-AUC
0.7270, CI upper 0.7664) is robust.

## Historical multi-seed evidence (older protocol generations)

Qwen2.5-7B × W2C, seeds {42, 123, 456, 789, 2024}: **Net +92.0 ± 19.8** (min 63, max 124),
**5/5 positive**, mean Δacc +16.8pp. Reverse per seed {−2, +9, +10, +21, +17} against real
{63, 85, 98, 124, 90} — retention 3–21%.

**Known weakening, already recorded in the archive and carried forward here:** the "real ≥ all
random in 5/5 seeds" claim rests on `n_random = 10` for seeds {42, 123} but `n_random = 1` for
{456, 789, 2024}. The 5/5 claim is real; its per-seed statistical strength is heterogeneous. This
is historical-generation evidence and must not be merged with modern formal results.

## Does full formal multi-seed rerun have scientific value?

```
NO_SCIENTIFIC_VALUE_IN_FULL_FORMAL_MULTI_SEED
```

Decoding is deterministic (A). The Router is conditionally deterministic — coefficient difference
exactly **0.000e+00** across 20 seeds (C). Directions are deterministic functions of frozen TRAIN
populations (D). Populations, channel, τ, dose and thresholds are all frozen. The intervention
null is already varied 59 ways (E), and CI Monte Carlo is seeded (F). **The only live stochastic
component was the split, and it has now been quantified on CPU without touching SEALED.**

Rerunning the formal pipelines under new seeds would re-execute a deterministic function and
consume a sealed population for no information.

## Verdicts

| dimension | rating | evidence |
|---|---|---|
| multi-seed — historical | **ADEQUATE** | 5 seeds, 5/5 positive, heterogeneous null strength disclosed |
| multi-seed — modern intervention null | **STRONG** | K=59 salt-derived, disjointness audited, 0/59 in all three trials |
| multi-seed — modern pipeline construction | **ADEQUATE** | deterministic by design; split sensitivity now quantified over 30 splits |
| ablation completeness | **STRONG** | 65 formal arms; 7-rule ladder; leave-one-out; 9-pathology battery |
| causal control completeness | **STRONG** | zero, reverse, wrong-layer, cross- and same-layer DiffMean, ungated, score-space |
| measurement validation | **ADEQUATE** | W2C 0/0/0 across 4 models; ACEBench 9/10 locked, parser audit unstratified |
| benchmark validation | **PARTIAL** | W2C gold never independently re-audited by us |
| statistical validation | **STRONG** | bootstrap, Clopper–Pearson, add-one p, 420-cell operating characteristics |
| preservation validation | **WEAK** | uncertifiable at α=0.05 in every setting including the ADMIT (0/6 → upper 0.393) |

## What remains untested, stated plainly

1. **W2C gold provenance** — we never independently audited its labels. Given arXiv 2607.02577's
   9.8–30.5% evaluator error across four other benchmarks, this is a fair reviewer question. A
   blinded stratified human audit is proposed as optional future work; **we did not run one and
   do not claim one.**
2. **ACEBench parser audit stratification** — 0.98% is a point estimate on 102 audited rows
   (95% upper ≈ 4.6%), with no documented stratification by language × gold × predicted × output
   form. Report as *"0.98% on the audited subset."*
3. **Preservation certification** — no setting certifies; disclosed as a result.
4. **Multi-seed on the split for the formal trials themselves** — quantified retrospectively on
   pre-intervention quantities only; the sealed endpoints were not recomputed under alternative
   splits and must not be.
