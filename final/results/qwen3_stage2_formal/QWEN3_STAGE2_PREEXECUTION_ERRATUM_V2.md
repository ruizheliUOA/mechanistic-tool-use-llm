# Pre-execution erratum V2 — formal execution body

## What the previous freeze contained

The freeze at HEAD `42261b30255c167975357f225f49c427474c1fd4` contained the
**complete scientific protocol** — preregistration, channel-selection disclosure,
endpoint specification, control specification, execution freeze, the 59 formal
random vectors, the VOID schema and template, and passing preflight,
DEV-engineering and gate-only runs — **but no formal execution body**.
`scripts/qwen3_stage2_formal.py --run-formal` refused without a token and, even
with one, exited stating that the body must be added under a new versioned
freeze.

## What did not happen

**No evaluation access occurred.** The evaluation payload was never opened, no
evaluation ID was enumerated, no evaluation prediction was generated, and no
`FORMAL_ACCESS_STARTED.json` marker was ever written.

## What V2 adds

Implementation only:

- an `EvaluationAccess` class, the single component able to read the evaluation
  partition, deriving its indices as the complement of the committed train and
  dev manifests and refusing to construct unless the access marker exists;
- the formal access marker and its VOID semantics;
- the one-process, one-load formal path with the frozen arm order;
- baseline over all 548 rows, the zero arm and the exact zero-equality gate;
- population derivation from the formal baseline only;
- all nine frozen arms including the 59 randoms in seed order;
- raw-record writing to a temporary path with atomic promotion after every arm;
- endpoint computation gated on completeness, order, hash and norm checks;
- mechanical selection of the frozen interpretation branch.

## What did not change

**No scientific quantity changed.** The formal channel, direction and its hash,
Router weights, tau, `L_obs`/`L_inj`, intervention site and positions, `q`,
`s_c`, `K`, all 59 random vectors and their order, the arm battery and order, the
primary statistic, the ten-condition conjunction, the secondary hierarchy, the
bootstrap method and seed, the support rule, `b_c`, the zero-control rule, the
VOID rules, batch size and the numerical environment are all read from the
committed artifacts at runtime and are unchanged.

## Artifact status

All previous artifacts remain preserved and unmodified. The previous runner
remains in the tree and remains non-runnable for the formal test. The new runner
`scripts/qwen3_stage2_formal_v2.py` is the **only** candidate for future formal
authorization.
