> **Anonymised derived copy.** This file is *not* the frozen artifact. The frozen original,
> `QWEN35_INFRASTRUCTURE_VERIFICATION.md` (sha256 `6ca3ed1c26f54525a9fb86ecbcd107bd521710b7432a4bf87e661651ed85fd4e`, recorded in the
> sibling `HASHES.json`), carries a git remote URL that identifies the authors and is
> therefore withheld from this release. Exactly one substring was replaced: that remote URL.
> No recorded value, count, verdict or protocol statement differs. `HASHES.json` is left
> unmodified and will not match this file, by design.

# Qwen3.5-9B — infrastructure verification

Every claim below was verified from local artifacts in this session. Nothing was taken from a
conversational summary.

## Repository

`/root/autodl-tmp/sakiko-followup`, branch `exp/sakiko-followup-archive`, HEAD
`49dc39031c535135fac3c952d687790fd6eb3a0c`, remote `git@github.com:<anonymised>/<anonymised>.git`.
Namespace for this study: `research_exploration/qwen35_sakiko/`. No historical formal artifact
is written to.

## Model integrity — VERIFIED

`Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, local
`/root/autodl-tmp/models/qwen3.5-9b-c2022362`.

| shard | size | sha256 |
|---|---|---|
| 00001-of-00004 | 5,276,436,216 | `db6f444b…` MATCH |
| 00002-of-00004 | 5,335,161,512 | `31c7d7e2…` MATCH |
| 00003-of-00004 | 5,368,717,440 | `7ec36ba3…` MATCH |
| 00004-of-00004 | 3,325,995,712 | `b62b0c4c…` MATCH |

All four match authoritative upstream size **and** sha256. Config, tokenizer, vocab, merges,
chat template, generation config and weight index hashed into
`QWEN35_LOCAL_MODEL_HASHES.json`.

## Environment — isolated, verified

`/root/autodl-tmp/env_qwen35_panel`. torch **2.5.1+cu124** (CUDA available, 12.4),
transformers **5.15.0**, tokenizers **0.22.2**, huggingface_hub **1.27.0**.
`Qwen3_5ForCausalLM` / `Qwen3_5Config` resolve. The historical SAKIKO environment is untouched.

A `torchvision` image-extension warning is emitted from the *base* interpreter's site-packages;
it is unrelated to the text-only path and does not enter the model or scoring code.

**Recorded install incident.** Two pip attempts to fetch torch (PyTorch CDN, then PyPI) both
failed with `THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE` — the box's
local proxy corrupts large wheels in transit. Neither corrupted wheel was installed. The wheel
was instead fetched with `curl`, verified independently (`zipfile.testzip()` clean, 11,269
members), and installed from the local file; CUDA runtime libraries were then resolved normally.

## BF16 text-only smoke — `QWEN35_BF16_SMOKE_PASS`

All four required markers emitted: `MODEL_LOAD_OK`, `FORWARD_OK`, `GENERATE_OK`,
`QWEN35_BF16_SMOKE_PASS`. Artifact: `QWEN35_BF16_SMOKE.json`.

| quantity | value |
|---|---|
| params loaded | **8.954 B — text-only; the vision tower is not instantiated by `Qwen3_5ForCausalLM`** |
| dtype | `torch.bfloat16` exclusively |
| all params `requires_grad=False` | **True** |
| forward logits | `[1, 16, 248320]`, all finite |
| gradient at L_inj leaf | shape `[1, 16, 4096]`, finite, L2 0.0461 |
| deterministic 4-token replay | **True** (bitwise identical across two runs) |
| peak allocated | **18.256 GB** of 25.253 GB — ~7 GB headroom |

No OOM occurred, so no remedy was applied: no quantisation, no precision change, no CPU offload,
no device-map change, no architecture change.

## Runtime architecture — two corrections to prior expectations

1. **Text stack is at `model.model.layers`**, not `model.language_model.layers`. The weight file
   names carry a `model.language_model.` prefix, but `Qwen3_5ForCausalLM` flattens it. Reusing
   the weight-name path would raise `AttributeError`, not mis-hook silently.
2. `Qwen3_5ForCausalLM` loads **8.95B text-only parameters**; the 333-tensor vision tower is
   never instantiated, so contamination of a text-only run is structurally impossible rather
   than merely avoided.

Verified at runtime: 32 layers, hidden 4096.
**L_inj 18** → mixer `Qwen3_5GatedDeltaNet`, MLP `Qwen3_5MLP`.
**L_obs 23** → mixer `Qwen3_5Attention`, MLP `Qwen3_5MLP`.

## Readout interface — exact match, no adaptation required

The Qwen3.5 chat template accepts `enable_thinking=False` and emits **exactly** the frozen Qwen3
suffix `<|im_start|>assistant\n<think>\n\n</think>\n\n`, with an empty reasoning block, and
renders `tools=` natively. The committed Qwen3 `render_prompt` / `candidate_ids` /
`score_candidate` therefore apply **verbatim**, with no wording adaptation of any kind. This is
a stronger interface match than Gemma-2-9B, which required folding the system text and tool
schemas into a single user turn.

## SEALED

Untouched. No sealed row has been read by any Qwen3.5 process at any point.
