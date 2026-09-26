# Reproducibility and inspection

This artifact preserves results; it does not authorize another sealed experiment.
The checks below read existing bytes and never load a model, contact a service,
rewrite a protocol, or generate a new formal verdict.

## Runnable checks

Python 3.10+, from the artifact root; no dependency installation:

```sh
python3 tools/verify_artifact.py
python3 final_evidence/tier_check.py
```

Redirect output to a separate audit directory if desired. To check a specific file:

```sh
python3 tools/verify_artifact.py --require-payload final/results/qwen3_stage2_formal/QWEN3_STAGE2_FORMAL_RECORDS.jsonl
```

A missing file or LFS pointer produces a nonzero exit. A pointer is never interpreted
as zero observations. The release manifest covers all delivered files except itself;
it verifies delivered bytes, not every historical hash assertion. Original
inconsistencies are reported separately in `INTEGRITY_FINDINGS.md`.

## Available row-level checks

| Setting | Records | Directly reconstructed counts |
|---|---|---|
| Qwen3-8B sealed | 6,372 arm records | `d_grad`: 72 routed errors, 52 exits, 38 gold, 14 other wrong, 6 exposed correct, 0 broken, 40 fixed, net +40. The 87-error channel denominator is recoverable from the arm-baseline union. |
| Qwen3-4B sealed | 14,404 forward-arm records, separate baseline and comparator records | `real_d_grad`: 118 routed errors, 64 exits, 37 gold, 27 other wrong, 50 exposed correct, 1 broken, 40 fixed, net +39. Bundled baseline subset: 452 rows, 186 correct; population denominator 214 is preserved from the summary. |
| Gemma sealed | 7,731 arm records, 548 baseline rows | `real_d_grad`: 96 routed errors, 27 exits, 17 gold, 10 other wrong, 11 exposed correct, 1 broken, net +16. Full baseline: 223 correct, 112 channel errors. Frozen target-gain denominator remains 96. |
| Phi-3.5 historical, seed 42 | 548 rows | 107 fixed, 52 broken, net +55; exposed-correct denominator 93, using archived firing rule `route != 'none'`. |

All three sealed settings have zero controls and 59 random directions. The helper
checks zero-arm predictions and scores, random count, and whether random target-gain
numerators reach the real numerator. It compares existing intervals and verdicts
to the index. It does not rerun bootstrap selection or substitute intervals.

**Limits:** Complete 548-row baselines are not separately bundled for Qwen3-8B or
Qwen3-4B. Population-correct denominators 211 and 214 are preserved summary
values, not complete baseline recounts here. Its exposed denominator 6 and zero breaks are directly
checkable. Missing historical destination outcomes cannot be reconstructed from
Fixed/Broke/Net. CPU checks do not validate model weights or the original GPU runtime.

## Command classes

| Component | Prerequisites and effect |
|---|---|
| `tools/verify_artifact.py` | Read-only standard-library inspection of bundled bytes. |
| `final_evidence/tier_check.py` | Read-only standard-library check of the index mapping. |
| `figures_final/_panels.py:load_all()` | Optional figure-source consistency check; needs NumPy and Matplotlib. No figure-writing function is needed. |
| Dataset conversion examples | External inputs required; run in a separate staging copy. New serialization is not proof of historical byte identity. |
| `scripts/qwen3_stage2_formal.py --gate-only` | Historical command: requires input arrays/model files and writes gate/retry ledgers. Not a safe quick-start check. |
| Formal runners with `--run-formal` | Historical provenance only. Require exact inputs, runtime, paths, and one-shot authorization state. Do not run to inspect this artifact. |
| Arbitrary script imports | Not a supported safety check: some create output directories at import time. Syntax is checked without importing them. |

The optional figure check requires an isolated venv with `numpy` and `matplotlib`.
Set `MPLCONFIGDIR` and `PYTHONPYCACHEPREFIX` outside the checkout and invoke only
`_panels.load_all()`. The experiment dependencies in `requirements.txt` are an
inventory, not a universal environment lock. Package-local frozen records take
precedence: Qwen3-8B and Gemma formal records specify Transformers 4.51.0,
PyTorch 2.1.2+cu121, CUDA 12.1; older ACEBench/Llama records may specify 4.49.0.
Versions absent from the original record remain unknown.

No sample, label, split, seed, direction, dose, threshold, statistical rule,
preregistration, or formal verdict was changed. Unresolved original-record
inconsistencies prevent a blanket reproducibility or release PASS.
