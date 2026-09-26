# §12 BF16 mechanical VRAM smoke test — Gemma-2-9B-it

**Result: PASS**, on the population that remains after a predeclared hardware structural
exclusion of 6 TRAIN rows (0.164%). See `HARDWARE_STRUCTURAL_EXCLUSION.md`.

## Configuration — frozen, nothing substituted

BF16 exact · `attn_implementation="eager"` (**unchanged**) · `device_map="cuda:0"` ·
`model.eval()` semantics · `use_cache=False` · **all parameters `requires_grad=False`**, only
the activation leaf at `L_inj = 24` carries gradient · determinism as committed
(`use_deterministic_algorithms(True)`, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).

**No quantisation. No precision change. No estimator substitution. No attention-implementation
change.**

## The four escalating attempts, all within §12's allowed remedies

| # | change | p90 (1448 tok) | longest (4528 tok) | peak alloc |
|---|---|---|---|---|
| v1 | frozen config as-is | **OOM** | **OOM** | 24.64 GB |
| v2 | + `expandable_segments` allocator, per-mode cache release | **OOM** | **OOM** | 24.64 GB |
| v3 | + gradient checkpointing (inactive — HF needs `training=True`) | **OOM** | **OOM** | 24.61 GB |
| v4 | + train-mode activation, gated on a dropout audit | **OK** 19.50 GB | OOM | 18.94 GB |

v2's allocator change alone cut reserved memory from 22.38 → 18.58 GB at the median.

## Why train mode is mechanically equivalent here

HF dispatches the checkpointed path only when `module.training` is True. Train mode differs
from eval mode **only** through dropout, so it was gated on an explicit audit:

- `config` dropout fields: `{"attention_dropout": 0.0}`
- `nn.Dropout` modules with `p != 0`: **none**

and then validated numerically against the un-checkpointed eval-mode gradient:

| quantity | value |
|---|---|
| score absolute difference | **0.0** |
| gradient relative L2 difference | **0.0** |
| gradient max absolute difference | **0.0** |
| peak allocated, checkpointing off → on | 21.476 → **18.869 GB** |

**Bit-identical.** §12 permits gradient checkpointing "if numerically validated"; this is the
validation, and it is exact rather than merely within tolerance.

## Recorded measurements

GPU: RTX 4090 D, 23.52 GiB usable. Weights after load: **18.483 GB allocated**.
Gemma-rendered `max_seq` over all 3652 rows: min 78, p50 687, p90 1566, p95 1686, p99 2352,
max 5247. Feasibility boundary: last feasible **3169**, first infeasible **4572**, with **no
row in between**.

## Verdict

**No `MODEL_HARDWARE_NO_GO`.** The frozen four-mode full-effect gradient stack runs on this
hardware for 3646 of 3652 rows, including every DEV and every SEALED row. The v1–v3 `MODEL_HARDWARE_NO_GO`
readings were intermediate states of this test, not the outcome, and are retained above rather
than deleted.
