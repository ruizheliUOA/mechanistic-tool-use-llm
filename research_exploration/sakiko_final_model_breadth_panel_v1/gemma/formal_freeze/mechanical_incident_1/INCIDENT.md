# Mechanical incident 1 — frozen runner had no execution path

**Class: mechanical. No scientific definition, constant, threshold or population changed.**

## Defect

`run_gemma_formal.py` as first frozen implemented only `--gate-only`. The `--run-formal`
branch printed a refusal and exited 3. The runner was therefore **unexecutable**: authorisation
could be granted but the frozen artifact could never perform the run. This is the same class as
the Qwen3-4B formal runner's mechanical incident 1.

Discovered **before** any SEALED access, before the marker existed, with the ledger pristine.

## Repair

Added the `run_formal()` execution path: gate in `formal-run` phase, fsynced access marker
before the first SEALED row is loaded, SEALED baseline, Router application with the committed
coefficients, the 65 committed arms, per-row record writing, ledger finalisation.

**Zero scientific constants touched.** `q`, `tau`, `L_inj`, `L_obs` are byte-identical between
the before and after files (verified by diff). The direction, Router, random matrix, dose,
thresholds, collateral denominators and exclusion manifest are all read from the committed
`FORMAL_CONFIG.json` / artifacts and are not defined in the runner.

## Hashes

| | sha256 |
|---|---|
| BEFORE | `1aa81b6eddb721651151e7ee50b51e07f2bf81924ccfedb6365d2067034c4d38` |
| AFTER | `66d81956fb54fa85082a38a16987ce26c67dcf343c9844b7a3285ae18e798588` |

The before-file is preserved verbatim at `run_gemma_formal.BEFORE.py`.

## Consequence

Per the freeze policy the hash is invalidated, the package is re-frozen, and the **complete
pre-formal gate is re-run** on the repaired runner before any SEALED access.
