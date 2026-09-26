# Reproducibility

## 1. Inspect the preserved results

From the repository root, with Python 3.10+:

```sh
python3 tools/verify_artifact.py
```

No dependencies need installing. Expected output includes
`AVAILABLE_EVIDENCE_CHECKS_PASSED`; this applies to the stated checks, not to
missing historical inputs or an assertion that all historical hashes match.
A missing file, changed byte or LFS pointer causes a nonzero exit.

| Setting | Real arm | Exits / gold / other wrong | Exposed correct / broken | Whole-arm net |
|---|---|---|---|---|
| Qwen3-8B | `d_grad` | 52 / 38 / 14 | 6 / 0 | +40 |
| Qwen3-4B | `real_d_grad` | 64 / 37 / 27 | 50 / 1 | +39 |
| Gemma-2-9B | `real_d_grad` | 27 / 17 / 10 | 11 / 1 | +16 |
| Phi-3.5, historical seed 42 | archived locked run | 153 / 107 / 46 | 93 / 52 | +55 |

The three sealed verdicts remain success, decline, decline. The checker reads
frozen intervals rather than regenerating them. Original rows, thresholds, seeds,
doses, labels and splits are not changed by packaging.

## 2. Understand file hashes and locations

`manifests/release.json` covers every delivered file except itself.
`manifests/payloads.json` records original LFS payload hashes/sizes, all verified.
`manifests/retained_artifacts.json` distinguishes byte-identical files from the four
Phi usage-docstring derivatives. Four further Qwen3-4B files have separately named
anonymous derivatives that replace a private session/cache path only; see
`manifests/anonymous_derivatives.json`. `manifests/paths.json` resolves the old relative
paths referenced inside preserved scripts/manifests to this curated layout.

Original frozen manifests are unchanged. They describe their original packages,
including administrative files intentionally outside this data-focused selection.
Use the path map for retained files. An absent historical reference is not treated
as verified; old self/cross-hash discrepancies remain in `docs/limitations.md`.

## 3. Inspect or prepare the experiment code

The final Qwen3-8B executed runner is `scripts/qwen3_stage2_formal_v4.py`, as stated
in the frozen V4 implementation record. Older superseded formal runners are not
advertised as final code. Qwen3-4B's development modules and Phi's locked-evaluation
modules were recovered from existing Git history, not recreated from outcomes.

For an independent historical-layout workspace, outside this checkout:

```sh
python3 tools/prepare_legacy_layout.py --out /path/to/new-empty-workspace
```

This optional command copies retained files to their original relative locations.
It refuses an existing destination and performs no inference, download, installation
or Git operation. It restores paths, not missing inputs or a historic machine.

Experiment runners retain their frozen environment/path assumptions. They require
benchmark inputs and licensed model weights, and some can write files even with
`--gate-only` or at import time. Do not import arbitrary runners as integrity tests.
Formal one-shot runs must not be repeated to inspect this artifact.

## 4. External inputs and limits

[docs/data_and_models.md](docs/data_and_models.md) gives official sources, known
revisions, preprocessing and environment records. `requirements.txt` is an optional
import inventory, not a universal reproduction lock. No GPU runtime was recreated.

Qwen3-8B/4B full-population denominators 211/214 are preserved in summaries; their
complete 548-row baseline exports are not separately included. The Qwen3-4B
baseline export intentionally covers 452 rows entering at least one arm. Gemma's
full baseline has 112 channel errors, while its frozen target-gain uses 96 routed
errors. Missing historical placebo destinations cannot be inferred from net gains.
