# Pre-execution erratum V3 — authorization-lock defect

## The defect

The V2 authorization lock required `HEAD == frozen_implementation_head`. That
condition is **unsatisfiable in principle**: the frozen head must be recorded
inside a committed file, and writing it creates a new commit, so HEAD is
immediately one ahead of the recorded value. Every subsequent preparation commit
(preflight output, engineering check, gate-only) moves it further.

Concretely: the implementation commit was `a8b6c1a3`; recording that value
produced commit `d1f9881`, so the V2 lock would have refused every future
invocation, including a properly authorized one.

This was found **before** any preflight, engineering check or gate-only ran
against V2, and before any evaluation access. No result depends on it.

## The correction, in V3

`HEAD` must be the anchor commit `a8b6c1a3a61092a03b7c2ccd817d301d72c58475`
**or a descendant of it**, verified with `git merge-base --is-ancestor`, **and**
the runner's own SHA256 must equal the value recorded in this freeze, **and** the
existing conditions still apply: the exact CLI flag, the exact environment token,
divergence `0 0`, a clean worktree, an empty formal output namespace, and a full
frozen-artifact hash match.

## Why this is tighter, not looser

A commit-id comparison only proves *which commit* is checked out. The V3 lock
additionally proves *that the runner file is byte-identical* to the frozen one
and that every frozen artifact still hashes correctly. A modified runner on the
correct commit would have passed V2's check and fails V3's.

The anchor rule still prevents running from an unrelated branch or from before
the implementation existed, and combined with `divergence 0 0` and a clean
worktree it pins the repository to published, unmodified state.

## What did not change

No scientific quantity. The V2 runner is preserved unmodified and is superseded;
`scripts/qwen3_stage2_formal_v3.py` is now the only candidate for formal
authorization. Preflight, the DEV engineering check and gate-only are all run
against V3.
