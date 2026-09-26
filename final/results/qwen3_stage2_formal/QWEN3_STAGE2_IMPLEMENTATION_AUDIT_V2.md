# Implementation audit V2

## Reused verbatim, already exercised on 14,293 DEV records

- `make_hook` and `score_with_intervention` from the Stage 2A V2 calibration
  runner — the constant additive intervention at the frozen MLP site across all
  positions, and four-mode re-scoring under it.
- `load_model` and the frozen scoring/rendering/tokenization from the V1 runner.
- `install_chunk_checkpointed_attention` from the V2 runner — the verified memory
  path.

## Written for V2

`EvaluationAccess`, the formal access marker, `arm_plan` (expanding the frozen
arm order, never restating it), `build_delta` (float64 construction, single cast,
relative norm guard), `endpoints`, `score_changes`, the ten-condition conjunction
evaluation, the add-one p-value, the mechanical interpretation selection, VOID
emission, and the six-part authorization lock.

## DEV-specific logic deliberately NOT reused

`Ctx.dev_pred_eq_source`, `Ctx.dev_routed` and `Ctx.dev_channel_errors` all
filter on `split == "dev"`. The formal runner derives its populations from the
formal baseline instead, so no DEV filter can leak into the formal path.

## Evaluation-access surface

Exactly one class. It refuses to construct unless `FORMAL_ACCESS_STARTED.json`
exists, and it is referenced only inside `run_formal`. Verified: importing the
module loads no dataset, does not import torch, and creates no marker.

## Verified properties before freeze

- module import performs zero dataset access and zero model loading;
- `--run-formal` refuses without the environment token;
- the previous runner is untouched.
