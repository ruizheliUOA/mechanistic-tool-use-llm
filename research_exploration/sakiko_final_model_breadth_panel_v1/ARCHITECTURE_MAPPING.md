# Architecture mapping

## The frozen normalized-depth rule

SAKIKO does not copy Qwen absolute layer indices. The frozen mapping from the Qwen2.5 base
(28 layers, L_obs 20, L_inj 16) is

```
new_index = round( old_index / (old_n - 1) * (new_n - 1) )
```

giving normalized observation depth **20/27 = 0.7407** and normalized injection depth
**16/27 = 0.5926**.

**Validation.** Qwen3-8B has 36 layers: obs `round(0.7407 × 35) = round(25.93) = 26`, inj
`round(0.5926 × 35) = round(20.74) = 21`. These are exactly the committed `L_OBS = 26` and
`L_INJ = 21` in `scripts/qwen3_8b_stage0_1.py`. The rule reproduces the frozen protocol and is
applied here unchanged.

## MODEL Q — Qwen/Qwen3.5-9B (derived, not executed)

Text stack: **32 layers**, hidden_size **4096** (identical to Qwen3-8B), intermediate 12288,
16 heads, head_dim 256, 4 KV heads, vocab 248320.

| quantity | value |
|---|---|
| normalized observation depth | 0.7407 |
| normalized injection depth | 0.5926 |
| **derived L_obs** | `round(0.7407 × 31) = round(22.96)` = **23** |
| **derived L_inj** | `round(0.5926 × 31) = round(18.37)` = **18** |

**Interface assessment — provisional, and explicitly not a verdict.** The residual stream, the
per-block MLP/FFN output and the final-prompt-token readout all have direct analogues, so the
frozen representation site is *prima facie* mechanically defined. Derived L_obs 23 falls on a
`full_attention` layer and L_inj 18 on a `linear_attention` (SSM-style) layer; residual-stream
injection is architecture-agnostic and does not obviously break on a linear-attention block.
Two open questions could not be resolved without loading the model: whether the recurrent SSM
state interacts with a single-token residual edit in a way that has no analogue in the frozen
protocol, and whether the `float32` SSM state path preserves the gradient semantics the
estimator assumes.

**No `ARCHITECTURE_INTERFACE_NO_GO` is declared.** That verdict requires demonstrating the
frozen site has no meaningful analogue, and the model was never loaded. Declaring it here would
be an unsupported scientific claim. Equally, no post-hoc alternative site was invented.

## MODEL G — google/gemma-2-9b-it (derived; audit COMPLETE, model not loaded)

Access resolved in round 3. `gemma2` / `Gemma2ForCausalLM`: **42 layers, hidden_size 3584**,
intermediate 14336, 16 heads, 8 KV heads, head_dim 256, vocab 256000, bfloat16.

| quantity | value |
|---|---|
| normalized observation depth | 0.7407 |
| normalized injection depth | 0.5926 |
| **derived L_obs** | `round(0.7407 × 41) = round(30.370)` = **30** |
| **derived L_inj** | `round(0.5926 × 41) = round(24.296)` = **24** |

The frozen site has a direct analogue on this architecture — residual stream, per-block
GeGLU MLP output, final-token readout and causal-LM scoring are all present. **No
`ARCHITECTURE_INTERFACE_NO_GO`.** Full audit, including the four cross-family differences
recorded before inference (`final_logit_softcapping 30.0`, `attn_logit_softcapping 50.0`,
alternating 4096 sliding-window attention, and `head_dim × n_heads = 4096 ≠ hidden 3584`), is
in `gemma/ARCHITECTURE_INTERFACE_AUDIT.md`.

Note that `s_c` and the direction estimator operate on **3584** dimensions, not 4096: both the
residual stream and the MLP output are `hidden_size`-dimensional.
