# Figure and table source freeze

**All existing historical figures remain `STALE / NOT PAPER CLEARED`.** Nothing is
generated in this task. Every figure below must be built from
`final_evidence/FINAL_PAPER_EVIDENCE.csv`.

| fig | source data | denominator | claim | caption constraints |
|---|---|---|---|---|
| **F1** unified pipeline | none (conceptual) | — | C1-1 | no model names, no numbers |
| **F2** correction + controls | `phi_e2_*`, `q8b_*`, `q4b_*`, `gemma_*`, `*_matched_random_exceed` | routed channel errors, per bar | C2-1, C3-1, C3-2 | historical/modern bands separated; state K=59 |
| **F3A** Phi turning point | `phi_net`, `phi_collateral_e2` | **E2 n=93** | C4-1 | quote the 16–52 Broke range beside the point estimate |
| **F3B** activation vs score-space | `q8b_*`, `q8b_score_space_gold` | 72 channel errors | C4-2 | **must state:** McNemar p = 1.0 / 0.189, not prespecified, descriptive only; no superiority claim; do **not** show 0/6 vs 2/6 as preservation evidence |
| **F4** formal evidence forest plot | `q8b/q4b/gemma_target_hit` + frozen CIs | source exits | C5-1 | show point, interval, boundary, verdict; **caption must state DECLINE ≠ uncorrectable**; exits 52/64/27 |
| **F5** mechanism *(if budget)* | Router AUC by layer, DiffMean norm, bootstrap stability | — | C6-1 | label EXPLORATORY |

| table | content | mandatory column |
|---|---|---|
| **T1** | model / scale / dataset / protocol / **formal verdict** | protocol generation |
| **T2** | Act-1 correction + structural controls | historical vs modern block |
| **T3** | destination / preservation / licensing | **denominator label (E1 / E2 / VACUOUS)** |
| **T4** *(appendix)* | exit-rate-matched cross-model | matching error |
