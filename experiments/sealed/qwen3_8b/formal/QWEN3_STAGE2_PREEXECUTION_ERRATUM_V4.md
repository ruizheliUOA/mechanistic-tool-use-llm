# Pre-execution erratum V4 — gate-only mode not reachable

## The defect

Block 2 of the implementation task specified three strictly separated modes, so
V2 and V3 wired exactly `--preflight`, `--dev-engineering-check` and
`--run-formal`. The `gate_only` verification function was carried forward in the
source but was **not reachable from the command line**, so Block 6's required
23-item gate-only run exited with an argparse error.

Found before gate-only ran and before any evaluation access. No result depends on
it.

## The correction, in V4

`--gate-only` is wired to the existing `gate_only` function. The function body is
unchanged; only the argument parser and dispatch table changed.

## What did not change

No scientific quantity, no endpoint, no arm, no lock condition. V3's corrected
authorization lock is carried forward verbatim: anchor-or-descendant HEAD, runner
SHA256 match, exact CLI flag, exact environment token, divergence 0 0, clean
worktree, empty formal namespace and full frozen-artifact hash match.

All earlier runners are preserved unmodified and superseded.
`scripts/qwen3_stage2_formal_v4.py` is now the only candidate for formal
authorization, and preflight, the DEV engineering check and gate-only are all
re-run against it.
