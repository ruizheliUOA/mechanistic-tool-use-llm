# Formal implementation gap audit

Performed without opening evaluation data.

## A. Scientific definitions already frozen — MUST NOT CHANGE

Frozen at HEAD `42261b3` and read from committed artifacts at runtime, never
restated in code: formal channel `cannot_answer → tool_call`; model and revision;
`L_obs=26`, `L_inj=21`, site and intervention positions; Router weights and
`tau=0.4`; `d_grad` and its hash; `q=1.0`, `s_c=41.56598488897048` and the
absolute norm; `K=59` and all 59 random vectors in seed order; the nine-arm
battery and its order; the primary statistic; the ten-condition conjunction; the
ordered non-promotable secondary hierarchy; the bootstrap method and derived
seed; the support rule; `b_c`; the zero-control rule; the thirteen VOID rules;
batch size 1; the numerical environment; candidate scoring; prompt template;
thinking-mode handling; attention implementation; checkpointing.

## B. What already exists in `scripts/qwen3_stage2_formal.py`

| function | status |
|---|---|
| `utc_now`, `sha256_file`, `wj`, `append_retry`, `load_module` | complete utilities |
| `bootstrap_seed` | complete — derives the frozen seed from the salt |
| `Frozen.__init__` / `Frozen.verify` | complete — loads and verifies every frozen input |
| `preflight` | complete and verified PASS |
| `_delta` | complete — float64 construction, single cast, relative norm guard |
| `dev_engineering` | complete and verified PASS |
| `gate_only` | complete and verified PASS |
| `run_formal` | **stub only** — refuses without token, then exits stating the body is absent |

## C. Shared with the verified Stage 2A V2 implementation

`scripts/qwen3_stage2a_v2_dev_calibration.py` supplies, already exercised on
14,293 DEV records: `make_hook` (constant additive intervention at a layer's MLP
output, all positions), `score_with_intervention` (four-mode re-scoring under the
hook), and `Ctx.router_prob` (exact Router reconstruction, validated against all
15 committed tau-grid cells). These are reused rather than rewritten.

**DEV-specific logic that must NOT be carried over.** `Ctx.dev_pred_eq_source`,
`Ctx.dev_routed` and `Ctx.dev_channel_errors` all filter on
`split[sample_id] == "dev"`. They are unusable for the formal population and are
deliberately not reused; the formal runner defines its populations from the
formal baseline instead.

## D. Engineering code still required

1. **An evaluation accessor.** `TrainDevData.sample` raises
   `"firewall: requested non-authorized project index"` for anything outside the
   3,104 authorized rows, so it cannot reach the evaluation partition by
   construction. A separate accessor is required, deriving the evaluation project
   indices as the complement of the committed train and dev manifests over the
   3,652 pinned rows and direct-seeking through the raw source.
2. **The formal access marker** `FORMAL_ACCESS_STARTED.json`, written atomically
   immediately before the first evaluation read.
3. **Formal baseline** over all 548 rows, then the **zero arm** and the exact
   equality gate.
4. **Formal population derivation** from the baseline only.
5. **All nine arms** in the frozen order, including the 59 randoms in seed order.
6. **Raw-record writing** to a temporary path with atomic promotion only after
   every arm completes.
7. **Endpoint computation** after completeness, order, hash and norm checks.
8. **Mechanical interpretation selection** from the frozen branches.
9. **VOID emission** conforming to the frozen schema, with no retry or resume.

## E. Every location where evaluation access could occur

In the current runner: **none**. `preflight`, `gate_only` and `dev_engineering`
touch only committed artifacts and authorized DEV rows, and `run_formal` exits
before any data path.

In the new runner, evaluation access is confined to exactly one place: the
`EvaluationAccess` class, which is constructed **only** inside `run_formal`,
**only** after the approval lock passes and the access marker is promoted.
Module import performs no dataset access and no model loading. `preflight` and
`--dev-engineering-check` never construct it.
