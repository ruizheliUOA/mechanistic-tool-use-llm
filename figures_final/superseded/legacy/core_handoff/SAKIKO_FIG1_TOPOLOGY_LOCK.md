# SAKIKO Figure 1 — topology lock

**This section is load-bearing. Violating it makes the figure scientifically false.**

## Verified implementation

| | Value | Source |
|---|---|---|
| Backbone | 36 layers, zero-based `model.model.layers[...]` | `scripts/qwen3_8b_stage0_1.py:57-61` |
| **Observation site** | `layers[26].mlp` output, final prompt token | `qwen3_8b_stage0_1.py:1332`, spec at `:1055` |
| **Injection site** | `layers[21].mlp` output | `qwen3_8b_stage0_1.py:1068`; formal: `qwen3_stage2_formal.py:296` |
| Router scoring | NumPy on cached rows, `router_prob(A26_rows)` | `qwen3_stage2_formal_v2.py:227` |
| Intervention forward | `score_with_intervention(...)` re-runs the model | `qwen3_stage2a_v2_dev_calibration.py:204` |

Indexing is **zero-based** (direct HuggingFace `ModuleList` indexing). Both sites
are interior. The normalized-depth mapping is self-consistent:
`round(20/(28-1)*(36-1)) = 26` and `round(16/(28-1)*(36-1)) = 21`.

## The consequence

Layers execute in increasing index order:

```
... -> layers[21] (L_inj) -> ... -> layers[26] (L_obs) -> ...
```

**The injection site executes BEFORE the observation site.**

Therefore a same-pass `observe -> route -> inject` pipeline is **physically
impossible**. The tested configuration is **TWO-PASS**.

## The two passes

```
PASS 1 - DIAGNOSE
  input -> frozen model -> cache h_obs at layers[26].mlp
  (no intervention; activation written to disk)

  [ router scores the cached activation OFFLINE, in NumPy,
    outside the model graph entirely ]

PASS 2 - INTERVENE            (only if p_c >= tau)
  same input -> frozen model
             -> h' = h + alpha*d_c at layers[21].mlp
             -> remaining frozen layers 22..35
             -> action readout
```

## FORBIDDEN

An illustrator must **never** draw:

```
h_obs -> Router -> h_inj        [inside one continuous forward pass]
```

This is wrong in three ways at once: it reverses execution order, it implies the
router runs inside the model graph, and it hides that the input is processed
twice. The rejected v1 figure did exactly this and is archived in
`figures/archive_rejected/`.

Also forbidden:
- drawing the router as a neural network or a second LLM;
- drawing a single stack with the tap upstream of the write;
- omitting the "same input" relationship between the two passes.

## Permitted visual treatments

Any of these correctly express the two-pass structure:
- two horizontal lanes, one per pass, over the same frozen stack;
- one stack drawn once with numbered passes and an explicit re-entry arc;
- a vertical stack with the router offset to one side between passes.

The designer chooses freely among these. The ordering constraint does not.
