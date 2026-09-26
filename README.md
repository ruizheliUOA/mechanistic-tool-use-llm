# SAKIKO experimental artifact

Experimental data, frozen configurations and the code for *When Does Correction
Become Repair?* Start with the three sealed result packages below. Historical
experiments and settings stopped at screening are separated from sealed results.

## Data layout

| Directory | Contents |
|---|---|
| [experiments/sealed/qwen3_8b/](experiments/sealed/qwen3_8b/) | Development split/dose, preregistration, 6,372 formal arm records, controls, result summary; **FORMAL_CONFIRMATORY_SUCCESS**. |
| [experiments/sealed/qwen3_4b/](experiments/sealed/qwen3_4b/) | Development configuration/directions, 14,404 formal arm records, per-arm exports, destination/collateral results; **DECLINE**. |
| [experiments/sealed/gemma_2_9b/](experiments/sealed/gemma_2_9b/) | Development data/arrays, 548-row formal baseline, 7,731 arm records, result and principal verdict; **DECLINE**. |
| [experiments/historical/](experiments/historical/) | Phi-3.5, Qwen2.5-7B, Mistral, Llama and MetaTool locked results/control tables. These are not additional prospective licences. |
| [experiments/screening/](experiments/screening/) | ACEBench readout/support data and Qwen3.5 screening results/stop decision. |
| [experiments/analysis/](experiments/analysis/) | Statistical sensitivity, preservation and secondary-analysis tables. |
| [experiments/paper_evidence.csv](experiments/paper_evidence.csv) | Existing numerical index; individual frozen results and principal verdicts take precedence. |
| [scripts/](scripts/) | Experiment runners, preprocessing, analysis and their local dependencies. See [code map](scripts/README.md). |
| [manifests/](manifests/) | Delivered-byte hashes, original-to-current path map, payload availability and preserved pre-execution provenance. |

## Verify without running experiments

Python 3.10 or newer; no third-party packages or GPU required:

```sh
python3 tools/verify_artifact.py
```

The command reads only: file hashes, full payload availability, syntax, destination
counts, zero controls, 59 random arms per sealed setting, selected denominators,
frozen intervals and principal verdicts. It does not run a model or alter results.

All 40 recovered LFS objects are included as **full bytes**, not pointers. Raw
benchmark downloads and model weights are acquired separately. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [data/model sources](docs/data_and_models.md).

## Reading the results correctly

- Aggregate net gains +40/+39 are different from channel gold arrivals 38/37.
- Population collateral and exposed-correct collateral have different denominators.
  In particular, Qwen3-8B's 0/211 does not establish a tight bound from its 0/6 exposure.
- Positive point estimates do not override the two frozen DECLINE verdicts.
- Known old-manifest hash discrepancies and incomplete historical inputs are
  disclosed in [limitations](docs/limitations.md). A matching release hash does not
  certify every assertion in an old manifest.

This distribution focuses on experiments. Editorial drafts, internal handoffs,
reviewer simulations, writing audits, obsolete plots and duplicate prose reports
are not included. Retained scientific data and frozen protocols preserve their
bytes; the separate path map records the new layout.
