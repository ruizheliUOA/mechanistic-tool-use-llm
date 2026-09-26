# Artifact reading index

| Question | Protocol / evidence | Code |
|---|---|---|
| Admitted sealed setting | `final/results/qwen3_stage2_formal/`: preregistration, records, results, random manifest | `scripts/qwen3_stage2_formal.py`, `scripts/qwen3_stage2_formal_v2.py` |
| Qwen3-4B decline | `final/results/qwen3_4b_w2c_dose/`: frozen protocol; `final/results/qwen3_4b_w2c_formal_v1/`: records, results, principal verdict | `scripts/qwen3_4b_w2c_formal.py` |
| Gemma decline | `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/`: config, baseline, records, principal verdict | Same directory: `run_gemma_formal.py` |
| Historical net gain and collateral | `final/results/clean/`: seed-42 rows, multi-seed summaries; `analysis_e2/`, `preservation_audit/`, historical stress-test package | `scripts/`; source constraints limit unrecorded placebo destinations |
| Canonical non-test split | `final/results/qwen3_stage0_1_v2/`: protocol, split index, support/geometry/Router manifests | `scripts/qwen3_8b_stage0_1_v2.py` |
| MetaTool binary ontology | `metatool_audit/`, historical summary packages | `scripts/convert_metatool_binary.py`, `scripts/metatool_binary_sakiko_pilot.py` |
| ACEBench readout and support stop | `final/results/acebench_generation_readout/`: protocols, audit judgments, dataset manifest | `scripts/convert_acebench_decision.py`, `scripts/acebench_generation_baseline.py`, `scripts/acebench_parse_outputs.py` |
| Qwen3.5 screening stop | `research_exploration/qwen35_sakiko/`: baseline rows, support ledgers, terminal verdict | Package-local code and manifests |
| Claim/figure provenance | `final_freeze/FINAL_CLAIM_MATRIX.csv`, `final_evidence/FINAL_PAPER_EVIDENCE.csv`, `final_evidence/EVIDENCE_CORRECTIONS.md`, `_evidence_audit/03_provenance/` | `final_evidence/tier_check.py`, `figures_final/_panels.py` |

Start verification with `tools/verify_artifact.py`, not an experiment runner.
Older audit prose and `claim_matrix/` retain superseded claims as provenance and
cannot override principal formal verdicts or later corrections. Historical references
to manuscript drafting directories describe the original layout; those drafts are
not part of this artifact. The Qwen3.5 infrastructure note has a separately named,
clearly marked anonymous derivative; it is not its original frozen artifact.
