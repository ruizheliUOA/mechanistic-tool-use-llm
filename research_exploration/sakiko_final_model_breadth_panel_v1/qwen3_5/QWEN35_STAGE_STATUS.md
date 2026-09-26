# Qwen3.5-9B — stage status

**SEALED untouched. No model loaded. No inference. No behaviour observed.**

| stage | status |
|---|---|
| repository / protected-package gate | **COMPLETE** |
| panel freeze verification (MODEL Q) | **COMPLETE** |
| canonical V2 split verification | **COMPLETE** — TRAIN 2000 / DEV 1104 / SEALED 548 |
| architecture audit from local config | **COMPLETE** |
| layer mapping | **COMPLETE** — L_obs 23, L_inj 18 |
| architecture-interface decision | **COMPLETE — ACCEPTED** |
| weight download | **IN PROGRESS** — 0.76 / 19.31 GB, 1.79 MB/s, ETA ~2.9 h |
| isolated environment | **IN PROGRESS** — transformers 4.57.6 + tokenizers 0.22.2 installed; torch 2.5.1+cu121 downloading (780 MB) |
| local model hashes | NOT REACHED — requires complete shards |
| BF16 hardware smoke | NOT REACHED |
| TRAIN+DEV baseline | NOT REACHED |
| channel discovery | NOT REACHED |

## Resume procedure

1. Confirm `ALL_SHARDS_COMPLETE` in the downloader log and that torch 2.5.1 imports.
2. Hash all shards + metadata into `LOCAL_MODEL_HASHES.json`; verify against upstream.
3. `ENVIRONMENT_FREEZE.json`: pin python/torch/transformers/tokenizers/hub/CUDA, and record
   whether `causal-conv1d` / fast Gated-DeltaNet kernels are present or Transformers falls back
   to the reference PyTorch path (a fallback is permitted only if it is an official supported
   path and is disclosed).
4. BF16 smoke at L_inj 18 via `model.language_model.layers[18].mlp`, all params
   `requires_grad_(False)`, final-token logit slicing, float32 SSM state contribution reported.
5. TRAIN+DEV baseline, then automatic 12-transition channel discovery.

## Two mechanical facts that will matter at resume

- The hook path is **`model.language_model.layers[L].mlp`**, not `model.model.layers[L].mlp` —
  Qwen3.5 wraps the text stack under a multimodal parent. Reusing the Qwen3/Gemma path verbatim
  would raise `AttributeError`, not silently mis-hook.
- The venv is `--system-site-packages` and inherits torch 2.1.2, which **cannot** import
  transformers 4.57.6 (`register_pytree_node` renamed in torch 2.2). torch 2.5.1 must finish
  installing into the venv before any Qwen3.5 code runs.
