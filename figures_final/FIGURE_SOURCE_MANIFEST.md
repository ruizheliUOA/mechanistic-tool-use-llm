# Final figure source manifest

**Every number in every panel resolves to a row of the 113-row
`final_evidence/FINAL_PAPER_EVIDENCE.csv`.** No manually positioned numeric label.
Scripts read the database at render time; nothing is transcribed.

All previously existing figures are **STALE** and none is reused.

## Part 0 — figure story specification

The figures must communicate the **finding first**, the framework as its
consequence.

> Correction-related properties dissociate: behavioural improvement,
> destination correctness, preservation, and evidential sufficiency need not
> coincide, and repair requires the last three together. That dissociation is why
> `Correct → Verify → License` exists.

| figure | the one question it answers |
|---|---|
| **F1** | What is SAKIKO? |
| **F2** | Is the correction real, stable and specific? *(Act 1)* |
| **F3** | Does improvement establish repair? *(the finding)* |
| **F4** | What does the evidence license, per setting? |
| **F5** | Why does the pipeline need every stage? *(appendix)* |

Panel order inside F3 is **A → C → B** per the adversarial audit: A is fully
row-level and strongest, C is artifact-backed formal verdicts, B is descriptive
and lands only after the pattern is established.

---

## Panel manifest

| fig | panel | evidence variables | artifact | population | denominator | statistical status | caption claim |
|---|---|---|---|---|---|---|---|
| F1 | — | none (conceptual) | — | — | — | n/a | framework schematic; no data |
| F2 | A | `phi_e2_real_locked`, `phi_e2_random_dir`, `phi_e2_reverse_dir`, `phi_e2_wrong_layer`, `phi_e2_mismatched` | `p2_placebo_controls.json` | Phi test n=548 | none (counts) | descriptive ordering; **4 of 5 arms aggregate-only (Tier B)** | real intervention ranks first against four corrupted variants |
| F2 | B | `q8b_gold_arrival`, `q8b_random_gold_max`, `q8b_matched_random_exceed`, `gemma_gold_arrival`, `gemma_random_gold_max`, `gemma_matched_random_exceed` | formal RECORDS.jsonl (both) | routed channel errors | K=59 | exact count, recomputed | no matched random direction reaches real in either model |
| F3 | A | `phi_fixed`, `phi_broke`, `phi_net`, `phi_exposed_correct`, `phi_broken_e2`, `phi_collateral_e2` | `p0_final_test_details.jsonl` | 548 rows; 93 exposed-correct | **E2 n=93** | row-level exact | aggregate gain and preservation failure coexist |
| F3 | C | `q8b_gold_arrival`, `q8b_other_wrong`, `q8b_target_hit`, `q8b_score_space_*` | `QWEN3_STAGE2_FORMAL_RECORDS.jsonl` | 72 routed channel errors | 52 / 58 exits | **ns after Bonferroni** (McNemar p = 1.0, 0.189); **descriptive only** | similar aggregate improvement, different destination composition |
| F3 | B | `q8b/q4b/gemma_target_hit`, `*_ci_lo`, `*_ci_hi`, `*_formal_verdict` | formal RECORDS + PRINCIPAL_VERDICT | source exits 52 / 64 / 27 | source exits | 10,000-draw percentile bootstrap **resampling channel errors** | favourable point estimates can still be formally declined |
| F4 | — | the three `*_formal_verdict` rows, `phi_collateral_e2`, `llama_*`, `mistral_*` | mixed | per setting | per setting | mixed A/B/C — tier shown | property evidence ≠ formal licence |
| F5 | A–C | mechanism rows (`mech_*`), `phi_confidence_cliffs_delta` | Tier B/A | mixed | mixed | exploratory | representation structure is intervention-relevant but does not determine destination |

## Known measurement caveats carried into captions

1. **Interval source.** Plotted bounds are the values recorded in the frozen
   verdict artifacts — lowers 0.60416 / 0.4559 / 0.4444, uppers
   0.84615 / 0.6970 / 0.8148 (Qwen3-8B / Qwen3-4B / Gemma-2-9b). Independent
   10,000-draw reproductions differ in the third decimal (e.g. Gemma lower 0.4400);
   no verdict depends on this. *(Corrected 2026-09-11; see
   `final_evidence/EVIDENCE_CORRECTIONS.md`.)*
2. **0.50 is the only destination boundary.** Every sealed gate used a 95% lower
   bound above 0.50, fixed before the first sealed evaluation. 0.6042 is Qwen3-8B's
   own CI lower bound, never a criterion; it is not drawn or described as a
   threshold (S-24). *(Replaces the earlier "0.6042 reference" caveat.)*
3. **F2-A is Tier B** — four of the five Phi control arms have no surviving
   per-sample records. Plot Fixed/Broke/Net only; no destination breakdown.
4. **F3-C is descriptive.** No significance markers of any kind.
5. **Phi collateral is execution-dependent** — Broke ranged 16–52 across
   artifact-backed realizations. Caption must say so.

## Prohibited in every figure

Level-1 / Level-2 as formal hierarchy · orthogonal channels · significance stars ·
"admitted" for Qwen3-4B or Gemma · activation superiority · historical and modern
settings in one undifferentiated visual band · leaderboard aesthetics.


---

## Update after the maximum-value audit (113-row database)

Figure 2 was rebuilt to carry evidence that had been absent from the database:

| panel | added variables |
|---|---|
| F2-a | `phi_multiseed_nets`, `phi_multiseed_all_positive`, `qwen25_real_net`, `qwen25_real_fixed_broke` |
| F2-b | `qwen25_reverse_net`, `qwen25_random_mean_net` (beside the five Phi conditions) |
| F2-c | unchanged |
| F4 | `q8b/q4b/gemma_formal_verdict`, `*_target_hit`, `*_ci_lo`, `phi_broken_e2`, `llama_*`, `mistral_*` |
| F5 | `mech_ca_direct_*`, `mech_estimator_pca1_vs_diffmean`, `gating_precision_gated/ungated` |

**F4 is deliberately not a forest plot.** Figure 3c already carries the interval
evidence for the three modern settings; duplicating it would waste a main-text
figure. F4 instead covers all seven evaluated settings and separates descriptive
property evidence from the frozen verdict — information that appears nowhere else.

**F3 panel order is Phi → score-space → declines.** The formal declines close the
argument and hand directly to the licensing section, and the descriptive
score-space panel is not left carrying the section's conclusion.

**Checks at freeze:** tier check PASS · additive-only 96 → 113 rows, 0 altered ·
no orphan variables.


## Figure 0 — Introduction teaser (added at Stage 1B)

| fig | panel | evidence variables | artifact | statistical status | caption claim |
|---|---|---|---|---|---|
| F0 | — | `phi_other_wrong`, `phi_source_exits`, `phi_broken_e2`, `phi_exposed_correct`, `q4b_target_hit`, `gemma_target_hit` | `p0_final_test_details.jsonl`, formal RECORDS | Tier A throughout; conceptual layout | improving behaviour settles only the first of four questions |

Deliberately distinct from F1 (pipeline architecture) and F4 (per-setting property
matrix): F0 is an attrition view of a single intervention, and is the only figure
that states the paper's finding without plotting data. Final figure numbering is
decided at layout; the file name is stable.


## Rebuild in the final numbering (2026-09-11)

The whole set was rebuilt in the ggplot2-style theme with the Lancet palette
(`FIGURE_STYLE.md`). Earlier figures moved to `superseded/`. The rule above is
widened in one respect: values without a CSV row are read, read-only, from frozen
verdict artifacts and asserted against the CSV wherever a row exists.

| fig | file | panel | evidence variables | frozen artifacts read | population · denominator |
|---|---|---|---|---|---|
| 1 | `fig1_thesis` | left | `phi_net` | — | Phi test n=548 |
| 1 | | middle | `phi_source_exits`, `phi_gold_arrival`, `phi_other_wrong`, `phi_exposed_correct`, `phi_broken_e2` | — | 153 exits · 93 exposed-correct (E2) |
| 1 | | right | `{q8b,q4b,gemma}_target_hit`, `_ci_lo`, `_ci_hi`, `_formal_verdict` | — | source exits |
| 2 | `fig2_framework` | — | none (conceptual) | — | — |
| 3 | `fig3_dissociation` | a | `*_gold_arrival`, `*_other_wrong`, `*_matched_random_exceed` | 59 random target gains per sealed setting; Qwen3-4B channel-error count | all channel errors (87 / 124 / 96) · K = 59 |
| 3 | | b | `q8b_gold_arrival`, `q8b_other_wrong`, `q8b_score_space_*`, `q8b_gated_net`, `q8b_target_hit` | score-space breaks (channel-level Net +35) | 72 routed channel errors |
| 3 | | c | `phi_routed_errors`, `phi_gold_arrival`, `phi_other_wrong`, `phi_source_retained`, `phi_exposed_correct`, `phi_broken_e2`, `phi_router_fired`, `phi_collateral_e1` (denominator 264), `phi_target_hit` | — | 293 fired · E2 n=93 · E1 n=264 |
| 3 | | d | `{q8b,q4b,gemma}_target_hit`, `_ci_lo`, `_ci_hi`, `_formal_verdict`, `_source_exits` | — | source exits 52 / 64 / 27 |
| 4 | `fig4_licensability` | — | `*_matched_random_exceed`, `*_target_hit`, `*_collateral_e2`, `*_exposed_correct`, `*_formal_verdict`, `phi_broken_e2`, `phi_e2_random_dir`, `qwen25_real_ge_all_random`, `llama_randoms_ge_real`, `mistral_randoms_ge_real` | Qwen3-4B random count | per setting |
| 5 | `fig5_mechanism` | a | `mech_ca_direct_layers`, `_best_val_net`, `_dm_norm`, `_auc_flat`, `_cos_dm_pc1_l16` | — | layer sweep, development data |
| 5 | | b | `mech_estimator_pca1_vs_diffmean` | — | locked test |
| 5 | | c | `gating_precision_gated`, `gating_precision_ungated`, `qwen25_real_net`, `qwen25_ungated_net`, `qwen25_*_fixed_broke`, `q8b_gated_net`, `q8b_ungated_net`, `gemma_gated_net`, `gemma_ungated_net` | — | per arm |
| D1 | `figD1_historical` | a | `phi_e2_*` | — | Phi test n=548 |
| D1 | | b | `phi_multiseed_nets`, `phi_collateral_e1` | `final/results/clean/p1_multiseed_table.csv` | E1 n=264 per seed |
| D1 | | c | `qwen25_real_net`, `qwen25_reverse_net`, `qwen25_random_mean_net`, `qwen25_random_max_net` | — | K = 10 randoms |
