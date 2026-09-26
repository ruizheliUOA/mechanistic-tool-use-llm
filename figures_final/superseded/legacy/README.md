# Figures

`main/` — the four main-text figures. `appendix/` — supplementary. `source/` — generating
scripts and any intermediate data.

Specifications live in `manuscript/02_tables_figures/MAIN_FIGURE_SPEC.md` and `FIGURE_PLAN.md`.

| id | figure | data source | status |
|---|---|---|---|
| F1 | staged ladder with empirical stopping points | schematic; labels from `CORRECTABILITY_OUTCOME_LEDGER.csv` | not generated |
| F2 | destination flow for the licensed setting (87→52→38/14) + K=59 null | `qwen3_stage2_formal/QWEN3_STAGE2_FORMAL_RECORDS.jsonl` | not generated |
| F3 | refusal despite favourable conventional metrics | `qwen3_4b_w2c_formal_v1/` + `gemma/formal_freeze/FORMAL_RECORDS.jsonl` | not generated |
| F4 | preservation certification gap (E1 vs E2 vs upper bound) | `sakiko_generalization_phase/PRESERVATION_HISTORICAL_REANALYSIS.csv` | not generated |

**Rule:** every figure must be regenerable from a committed artifact by a script in `source/`.
No hand-drawn numbers. F1 is the sole exception — it is a schematic, and its annotations must
match the ledger exactly.
