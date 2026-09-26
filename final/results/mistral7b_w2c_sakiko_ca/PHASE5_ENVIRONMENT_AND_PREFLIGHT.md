# Phase 5 — Environment & Preflight

**Date:** 2026-07-15 · Repo `/root/autodl-tmp/sakiko-followup` · branch
`exp/sakiko-followup-archive` (unchanged). No git writes performed.

## 1. Model acquisition (recorded)

| field | value |
|---|---|
| source | official Hugging Face (via `hf-mirror.com` transparent mirror; **byte-identity verified**) |
| repo id | `mistralai/Mistral-7B-Instruct-v0.3` |
| revision (commit) | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| gated | No |
| precision | bf16 (native `torch_dtype: bfloat16`) |
| location (outside repo) | `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3` |
| total size | 14 GB (3 HF-format safetensors shards) |

**File manifest (downloaded):** `config.json`, `generation_config.json`,
`model-0000{1,2,3}-of-00003.safetensors`, `model.safetensors.index.json`, `params.json`,
`special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`.
**Deliberately excluded:** `consolidated.safetensors` (14.5 GB single-file duplicate of the
sharded weights — not needed by HF `from_pretrained`), `*.pth`, `original/*`.

**SHA256 integrity (downloaded == official HF LFS oids):**

| file | sha256 | verdict |
|---|---|---|
| model-00001-of-00003.safetensors | `ce6fb6f6…b0bd5c19` | ✅ OK |
| model-00002-of-00003.safetensors | `8c0e72f1…44f92ae0` | ✅ OK |
| model-00003-of-00003.safetensors | `905dd405…90cfb4ca` | ✅ OK |
| tokenizer.model | `37f00374…c6ca1f89` | ✅ OK |

Mirror use is justified per Phase-5 §4: the proxy stalled the official LFS CDN (0 B/s);
the mirror served the identical files and **exact model identity is established by SHA256
equality to the official repo's LFS oids** (captured from
`api/models/…/tree/main?recursive=1`). No tokens were stored in any script or report.

**Config (verified):** `MistralForCausalLM`, `num_hidden_layers=32`, `hidden_size=4096`,
`intermediate_size=14336`, `num_attention_heads=32`, `num_key_value_heads=8`,
`vocab_size=32768`, `max_position_embeddings=32768`, `rope_theta=1e6`,
`sliding_window=null`, `torch_dtype=bfloat16`.

## 2. Environment report

| component | value |
|---|---|
| conda env | `sakiko-phase3` |
| Python | 3.10.8 |
| torch | 2.1.2+cu121 (CUDA 12.1, available=True) |
| bf16 supported | True |
| transformers | 4.49.0 |
| tokenizers | 0.21.4 |
| datasets | 3.2.0 |
| scikit-learn | 1.7.2 |
| numpy | 1.26.3 |
| GPU | NVIDIA GeForce RTX 4090 D, 25.25 GB total, **24.88 GB free** at load |
| disk (`/root/autodl-tmp`) | 22 GB free after model download |

`transformers 4.49.0` natively supports Mistral-v0.3 (standard `MistralForCausalLM`;
no `trust_remote_code` needed). `nvidia-smi`: single RTX 4090 D, 0 MiB used before load.

## 3. Tokenizer / chat-template behavior (verified)

- `vocab_size=32768`; `bos=<s>`, `eos=</s>`, `pad=None` → set `pad=eos` at load.
- **Chat template accepts a `system` role**: `[{system},{user}]` renders to
  `'<s>[INST] {system}\n\n{user}[/INST]'` — identical to folding the system text into the
  user turn. The archived `build_prompt_messages` (system + user) therefore works
  unchanged; no Mistral-specific prompt rewrite is required.
- Candidate response-mode strings are the dataset's `answers[label]` **texts** (not bare
  label names); all 4 modes (`direct` / `tool_call` / `request_for_info` /
  `cannot_answer`) are present and non-empty for **all 3652** samples; each tokenizes to
  ≥1 token → all scoreable.

## 4. Data reconstruction (verified)

- `load_dataset("nvidia/When2Call","test","mcq").shuffle(seed=42)` is **bit-identical** in
  ordering to the local-raw-jsonl shim recipe (`phase5_mistral_lib.load_ds_raw`): all 3652
  uuids match in order. → the archived split indices (train 2556 / val 548 / test 548) are
  directly portable to Mistral.

## 5. Smoke test (exact planned baseline path)

`python scripts/phase5_mistral_baseline.py --smoke 5 --replay 5`:

| requirement | result |
|---|---|
| model loads in bf16 | ✅ `dtype=torch.bfloat16`, VRAM 14.63 GB |
| no quantization fallback | ✅ (hard guard: rejects non-bf16 params / any quantization_config) |
| no OOM | ✅ |
| candidate scores finite | ✅ (all avg_logp finite) |
| all W2C response modes scoreable | ✅ (4/4) |
| deterministic rerun → identical predictions | ✅ **pred 5/5 identical** |
| deterministic rerun → identical scores | ✅ **avg_logp 5/5 bit-identical** |
| output schema correct | ✅ (`uuid, gold, pred, correct, avg_logp{4}, margin, n_prompt_tok`) |

Throughput ≈ 2 samples/s at seq-scoring (4 candidates/sample) → full R0 ≈ 30 min;
activation extraction ≈ 30 min. Fits comfortably in 24 GB.

## Preflight verdict: **PASS** — cleared to run the full R0 baseline.
