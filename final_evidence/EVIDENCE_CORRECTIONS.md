# Evidence corrections — appended record

Corrections to `FINAL_PAPER_EVIDENCE.csv` (a derived index). No frozen artifact under
`final/results/` or `research_exploration/` is modified; each correction makes the
index agree with the frozen artifact it cites. No verdict changes.

## 2026-09-11 — target-hit intervals for the two declined settings

The index carried **recomputed** bootstrap bounds (independent 10,000-draw
reproductions from the 2026-08-28 verdict reconciliation) where the frozen verdict
artifacts record their own bounds. Under S-20 (verdict precedence) the frozen
values are authoritative.

| variable | was (recomputed) | now (frozen) | frozen source | tier |
|---|---:|---:|---|---|
| `q4b_target_hit_ci_lo` | 0.4561 | **0.4559** | `final/results/qwen3_4b_w2c_formal_v1/QWEN3_4B_FORMAL_RESULTS.json` → `arms.real_d_grad.target_hit_ci95` | B |
| `q4b_target_hit_ci_hi` | 0.6984 | **0.6970** | same | A → B |
| `gemma_target_hit_ci_lo` | 0.4400 | **0.4444** | `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/FORMAL_PRINCIPAL_VERDICT.json` → `headline.target_hit_ci95` | B |
| `gemma_target_hit_ci_hi` | 0.8148 | 0.8148 (value unchanged; source corrected) | same | A → B |

`q8b_target_hit_ci_lo` / `_hi` (0.60416 / 0.84615) already carried the frozen values.

**Why it matters.** The Gemma lower bound moved by 0.0044, more than the ±0.003
Monte-Carlo noise the captions had allowed, so the plotted and quoted value must be
the recorded one. Both lower bounds remain below 0.50; both DECLINEs are unchanged.

**Downstream updates in the same change:** `manuscript/05_correction_to_repair.md`
(0.440 → 0.444); `final_freeze/FINAL_FORMAL_STATUS_TABLE.md`,
`final_freeze/FINAL_MODEL_EVIDENCE_MAP.md`, `final_freeze/FINAL_FIGURE_PLAN.md`;
`figures_final/FIGURE_CAPTIONS.md` (Fig 3c) and `FIGURE_SOURCE_MANIFEST.md`;
`fig3_dissociation` and `fig4_licensability` regenerated from the corrected index.
