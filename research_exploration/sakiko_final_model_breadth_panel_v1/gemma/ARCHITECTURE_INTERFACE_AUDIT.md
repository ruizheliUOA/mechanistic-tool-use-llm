# Gemma-2-9B-it — architecture / interface compatibility audit

**Status: COMPLETE.** Derived entirely from the official `config.json` at the frozen revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819`. **No weights were loaded. No inference was run.**
This is a pre-inference compatibility record, not a result.

## Architecture

`gemma2` / `Gemma2ForCausalLM` — **42 layers, hidden_size 3584**, intermediate 14336,
16 attention heads, 8 KV heads, head_dim 256, vocab **256000**, bfloat16,
max_position_embeddings 8192.

## Frozen layer mapping — applied, not searched

```
L_obs = round(20/27 × 41) = round(30.370) = 30
L_inj = round(16/27 × 41) = round(24.296) = 24
```

The same rule reproduces Qwen3-8B's committed `L_OBS = 26` / `L_INJ = 21` from its 36 layers,
so it is applied here unchanged. **No architecture-specific layer search was performed and none
is authorised.**

## Interface: the frozen site has a direct analogue

| SAKIKO requirement | Gemma-2 analogue | status |
|---|---|---|
| residual stream at `L_inj` | standard pre-norm decoder residual | **PRESENT** |
| per-block MLP/FFN output at `L_obs` | `mlp` output, `GeGLU`, width 14336 | **PRESENT** |
| final-prompt-token readout | standard causal decoder | **PRESENT** |
| four-mode scoring interface | causal LM logits over 256000 vocab | **PRESENT** |
| native chat template | official Gemma-2 template | **PRESENT** |

**No `ARCHITECTURE_INTERFACE_NO_GO`.** The frozen representation site is mechanically defined
on this architecture.

## Four architecture differences recorded BEFORE inference

These are logged now so that no post-hoc adaptation can later be presented as a design choice.

1. **`final_logit_softcapping = 30.0`.** Gemma-2 applies `tanh` compression to output logits.
   The gradient estimator differentiates a score/margin with respect to activations, and `tanh`
   saturates — gradient magnitude is compressed where logits are large. This does not
   invalidate the estimator, and **the estimator family must not be changed to compensate.**
   It is recorded as a known cross-family difference that may affect effective dose scaling.
2. **`attn_logit_softcapping = 50.0`** — same class of difference, inside attention.
3. **Alternating sliding-window attention**, `sliding_window = 4096`, against
   max_position 8192. When2Call prompts are far below both, so no prompt is expected to
   straddle the boundary — but this must be verified from actual token lengths at baseline,
   not assumed.
4. **`head_dim × n_heads = 256 × 16 = 4096 ≠ hidden_size 3584`.** A Gemma-2 quirk. It does not
   affect the residual-stream or MLP-output sites, both of which are `hidden_size`-dimensional,
   so `s_c` (median ‖h‖ at `L_inj`) and the direction estimator operate on **3584** dimensions.

## §12 VRAM smoke test — PREDICTION ONLY, NOT RUN

Requires weights. Static arithmetic on the 24564 MiB (25.76 GB) RTX 4090 D:

| quantity | value |
|---|---|
| BF16 weights (9.24B params) | 18.48 GB |
| headroom | **7.28 GB** |
| saved activations + full logits @ seq 1024 | 3.61 GB |
| saved activations + full logits @ seq 2048 | 7.21 GB |

**Provisional read: feasible but tight.** Two conditions look load-bearing — slicing logits to
the final prompt token (0.5 MB rather than 0.5–1.05 GB) and holding `requires_grad = False` on
all parameters as the corrected Qwen3 implementation does, so no parameter gradient buffer is
ever allocated. At seq 2048 with full logits the margin is thin enough that the test could
genuinely fail.

**This prediction does not substitute for the smoke test and no verdict rests on it.**
`MODEL_HARDWARE_NO_GO` remains a live possible outcome, to be decided by measurement.
