# Final model panel freeze

Written **before any baseline result exists**. No model was loaded. No inference was run. No
dataset was touched. No scientific outcome was observed for either slot.

## The two slots — closed permanently

### MODEL G — primary cross-family

- **Checkpoint:** `google/gemma-2-9b-it`
- **Upstream revision:** `11c9b309abf7` (resolved from the HF API, 2026-08-11)
- **Weights:** 18.48 GB across 4 safetensors shards
- **Gating:** `gated=manual`
- **Reason for inclusion:** non-Qwen architecture; approximately the Qwen3-8B parameter regime;
  instruction tuned; large enough that the result cannot be dismissed as a tiny-model artefact.
- **Prior SAKIKO exposure:** the **family** was used at Gemma-2-2B under an earlier,
  non-modern protocol generation. **This checkpoint has never been tested.** The correct
  description is *modern-protocol cross-family replication on a previously untested Gemma
  checkpoint* — **never** "untouched family".
- **Fallback:** `google/gemma-7b-it` (rev `9c5798d27f58`, 17.08 GB), permitted **only** on a
  pre-inference mechanical memory/compatibility gate, only before any scientific baseline is
  observed, and never because of unattractive results. If used, record
  `HARDWARE_TRIGGERED_MODEL_FALLBACK` and that Gemma-7B is an older-generation architecture.
- **Immutable file hashes: NOT OBTAINED.** The checkpoint could not be fetched (see
  `MODEL_COMPATIBILITY_AUDIT.csv`).

### MODEL Q — secondary cross-generation

- **Checkpoint:** `Qwen/Qwen3.5-9B`
- **Upstream revision:** `c20223623576` (resolved from the HF API, 2026-08-11)
- **Weights:** 19.31 GB across 4 safetensors shards
- **Gating:** none
- **Architecture (from the retrieved `config.json`, which *was* obtained):**
  top-level `model_type: qwen3_5`, **multimodal** — a 27-layer / 1152-hidden vision tower plus
  `text_config` with `model_type: qwen3_5_text`. Text stack: **hidden_size 4096, 32 layers**,
  intermediate 12288, 16 attention heads, head_dim 256, 4 KV heads, vocab 248320, bfloat16,
  `tie_word_embeddings: false`, max_position_embeddings 262144.
  **Hybrid attention:** `layer_types` is `linear_attention` except full attention at layers
  **3, 7, 11, 15, 19, 23, 27, 31** (a 3:1 linear:full pattern), with `linear_conv_kernel_dim 4`
  and `mamba_ssm_dtype float32`. Also carries a 1-layer MTP head.
  Declares `transformers_version: 4.57.0.dev0`.
- **Reason for inclusion:** same Qwen lineage, different generation, hybrid architecture, ~9B.
  Tests whether frozen SAKIKO reasoning survives a substantially changed host architecture.
  This is **cross-generation / architecture breadth, not cross-family evidence**, and must not
  be compared to Qwen3-8B as a scaling law.
- **Prior SAKIKO exposure:** **none.** The standing project constraint that Qwen3.5-9B remain
  untouched was honoured up to this task: the HF cache contained an empty
  `models--Qwen--Qwen3.5-9B` directory with 0 blobs, no snapshots and no refs. This task, under
  explicit authorisation, retrieved **config and tokenizer metadata only (4.22 MB)**. No
  weights were obtained, no forward pass was run, no activation was inspected.
- **Immutable file hashes: NOT OBTAINED** for weights.

## Hardware and environment at freeze time

- **GPU:** 1 × NVIDIA GeForce RTX 4090 D, 24564 MiB, 0 MiB in use, driver 580.76.05
- **Disk:** `/root/autodl-tmp` 100 GB total, **32 GB free**
- **Base environment:** Python 3, torch 2.1.2+cu121, CUDA 12.1, transformers 4.51.0,
  tokenizers 0.21.1, huggingface_hub 0.36.2
- **Repository:** `exp/sakiko-followup-archive` @ `f5ba3a47f63f373df234013ed568ac6f43b2f8c9`

## Prohibition on replacement

Both slots are scientific outcomes regardless of what they return. A negative outcome from
either does **not** authorise replacing it. No third family, no larger Qwen, no larger Gemma,
no model chosen after observing results. See `ANTI_MODEL_SHOPPING_FIREWALL.md`.

## Outcome table, frozen before inference

`GEMMA ADMIT` strongest breadth improvement (model breadth only, never cross-dataset) ·
`GEMMA DECLINE` cross-family licensing tested, correctability did not replicate positively ·
`QWEN3.5 ADMIT` positive replicated across a changed Qwen generation, not independent-family
evidence · `QWEN3.5 DECLINE` Qwen lineage does not guarantee correctability ·
`BOTH ADMIT` strong model breadth, still one dataset · `ONE/ONE` licences are setting-specific
rather than family or parameter-count guarantees · `BOTH DECLINE` no further model; the
Qwen3-8B existential ADMIT stands and the paper strengthens on selectivity, not breadth.
