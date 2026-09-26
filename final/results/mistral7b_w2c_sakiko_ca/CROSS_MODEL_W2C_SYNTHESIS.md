# Cross-Model W2C Synthesis — Phi-3.5-mini vs Qwen2.5-7B vs Mistral-7B-v0.3

All three evaluated on the **same** dataset (`nvidia/When2Call` test/mcq, N=3652, seed-42
shuffle), the **same** split (train 2556 / val 548 / test 548), the **same** avg_logp
response-mode readout, and the **same** semantic label space. Layers are compared by
**normalized depth**, never by absolute index. **No Net is quoted without its error
opportunity, test size, channel support, Fixed/Broke, and a matched random reference.**

*(Machine-readable: `cross_model_w2c_summary.csv`. Mistral rows = this phase; Qwen/Phi rows =
archived committed files cited in `PHASE5_EXISTING_EVIDENCE_AUDIT.md`.)*

---

## 1. Baseline error structure

| model | layers | acc | macro-F1(3) | **pred `tool_call` share** | total errors | test-split channel opportunity |
|---|---|---|---|---|---|---|
| Phi-3.5-mini | 32 | 0.4822 | 0.4526 | 0.676 | 1891 | — (summary-level) |
| Qwen2.5-7B | 28 | 0.4381 | 0.4006 | 0.614 | 2052 | 253 / 548 |
| **Mistral-7B-v0.3** | 32 | **0.4269** | **0.3319** | **0.821** | **2093** | **303 / 548** |

Shared gold: tool_call 1295 / cannot_answer 1295 / request_for_info 1062. All three
**over-commit to tool-calling**; Mistral is the most extreme (82.1%) and predicts
`request_for_info` on only 2.8% of items. Mistral is the weakest baseline and the most
saturated — this drives every downstream difference.

## 2. Discovered channel counts & identity

| channel (gold→pred) | Phi | Qwen | **Mistral** |
|---|---|---|---|
| rfi_tc (request_for_info→tool_call) | 746 | 622 | **948** |
| ca_tc (cannot_answer→tool_call) | 511 | 513 | **773** |
| ca_direct (cannot_answer→direct) | 393 | 465 | **291** |
| ca_rfi (cannot_answer→request_for_info) | 93 | **219 (stable)** | **26 (insufficient)** |
| tc_rfi (tool_call→request_for_info) | 37 | **133 (stable)** | **1 (insufficient)** |
| **n stable channels** | **3** | **5** | **3** |

**Verdict: Mistral ≡ Phi (matching); Mistral ⊂ Qwen (strict subset).** Qwen's fourth
*validated* channel `ca_rfi` collapses to 26 on Mistral, and `tc_rfi` to 1 — because Mistral
funnels nearly everything into `tool_call`, the "destination = request_for_info" channels
effectively do not exist. **Channel structure is model-specific and derived, not assumed** —
the central SAKIKO-CA premise, confirmed on a third model. All 3 Mistral channels are
bootstrap-stable at **1.0**.

## 3. Selected vs utility-rejected channels

| model | discovered stable | **selected (passed utility gate)** | **rejected by utility gate** |
|---|---|---|---|
| Phi-3.5 | 3 | 3 (manual v1 design) | — |
| Qwen2.5-7B | 5 | **4** (rfi_tc, ca_tc, ca_direct, **ca_rfi**) | **tc_rfi** (Net≈0, random-saturated) |
| **Mistral-7B** | 3 | **0** | **all 3** |

On Mistral the **automatic system admits nothing**: `rfi_tc` fails 4/8 criteria (incl. both
specificity criteria), `ca_direct` fails 2/8 (both specificity), `ca_tc` fails **only**
criterion 4 (its own errors redistribute into ca_direct, 43→65 — not new damage, but
pre-registered as a failure).

## 4. Router separability, direction geometry, method preference, depth

| | Qwen2.5-7B | **Mistral-7B-v0.3** |
|---|---|---|
| obs layer (absolute) | L20 | **L22** |
| **obs normalized depth** | **0.71** | **0.69** ✅ converges |
| inj depth | L16–L18 (0.57–0.64) | L18–L20 (0.56–0.63) ✅ converges |
| router val-AUC | rfi_tc **0.75–0.78** (detection-limited → R3) | rfi_tc **0.878**, ca_tc **0.964**, ca_direct **0.938** (all ≫ 0.8) |
| median activation norm | ~10–40 | **1.6–4.5 (5–10× smaller)** |
| DiffMean norms | ca_direct L16 **2.12** (degenerate) vs L20 **10.41** | **0.24–1.14** (all below the archived absolute floor 5.0) |
| norm-ratio | selects L20 | **0.14–0.27**; selects L22 for all 3 |
| DiffMean vs PCA-1 | **PCA-1 for `ca_tc`** (cos 0.41) | **PCA-1 for `ca_direct`** (cos 0.560); DiffMean for rfi_tc (0.788), ca_tc (0.735) |

Two rules **replicate across architectures**: (i) depth-normalized obs selection (0.69 vs
0.71 despite 32 vs 28 layers); (ii) **R2** (`cos(DM,PC1) < 0.6 → try PCA-1`) fires on exactly
one channel per model and its PCA-1 candidate **wins on validation** both times — though on a
*different* channel each time (Qwen `ca_tc`; Mistral `ca_direct`): the **rule** transfers, the
**instance** does not.

One rule **fails to transfer**: **R1's absolute norm floor (< 5.0)**. Mistral's activation
norms are 5–10× smaller, so the Qwen-calibrated constant rejected *every* layer of *every*
channel. Only the **relative norm-ratio** formulation is portable.

**Router AUC ⊥ utility ⊥ specificity** — refuted on Mistral: `ca_direct` (AUC 0.938) is
non-specific and its *wrong-layer* control (+24) beats real (+10); `rfi_tc` (AUC 0.878) is
beaten by random (+29 vs +12). Mistral's routers are *uniformly better* than Qwen's, and its
mechanism is *worse*.

## 5. Intervention outcome — with full context (never Net alone)

| | Qwen2.5-7B (pilot, seed 42) | **Mistral-7B-v0.3 (Arm F, seed 42)** |
|---|---|---|
| test size | 548 | 548 |
| baseline acc | 0.4124 | 0.4252 |
| **channel-error opportunity** | **253** | **303** |
| Fixed / Broke | 92 / 13 | **88 / 6** |
| **Net** | **+79** | **+82** |
| Δ accuracy | +14.4 pts | **+15.0 pts** |
| **fixed-rate of opportunity** | **36.4 %** | **29.0 %** |
| 3-channel residual before→after | 253 → 126 (−50 %) | 303 → 174 (−43 %) |
| touched / damage-on-correct | 202 / 13 | 270 / 6 |

Mistral's Net and Δacc are **higher**, but from a **larger opportunity pool** and at a
**lower fixed-rate**. Raw Net therefore *flatters* Mistral.

## 6. Placebo specificity — matched cascade-level protocol (n_random = 10)

| control | Qwen2.5-7B | **Mistral-7B-v0.3** |
|---|---|---|
| real | **+79** | **+82** |
| reverse | +16 → **20 % retention** | +72 → **88 % retention** |
| random mean ± std | **25.2 ± 12.8** | **70.1 ± 19.9** |
| random max | 50 | **119 (> real)** |
| **real ≥ all random** | ✅ **TRUE** (0/10 ≥ real) | ❌ **FALSE** (1/10 ≥ real) |
| **marginal over random mean** | **+53.8** | **+11.9** |
| multi-seed | real ≥ all random **5/5 seeds** | **not run** (gate failed) |

Per-channel (Mistral, n_random = 20): `ca_tc` real +62, 95th pct, 1/20 ≥ real → the only
direction-specific channel; `rfi_tc` real +12 vs random mean +29 (17/20 ≥ real) → **worse than
noise**; `ca_direct` real +10, reverse +12, wrong-layer +24 → non-specific.

> **The decisive contrast.** Same protocol, dataset, split and readout. Mistral **wins on Net**
> (+82 vs +79) and **loses catastrophically on mechanism**: specific margin **4.5× smaller**,
> reverse retention **4.4× worse**, and a random direction **outperforms** the learned cascade.
> Qwen's gain is dominated by **learned direction**; Mistral's by **gated magnitude**.

## 7. Gating dependence

| model / arm | gated Net (broke) | ungated Net (broke) | reading |
|---|---|---|---|
| Qwen2.5-7B cascade | +79 (13) | **+94** (22) | ungated *raises* Net but doubles damage → gating trades Net for safety |
| **Mistral cascade** | +82 (6) | **−21** (130) | **gating is load-bearing**: without it the intervention is net-destructive |
| **Mistral `ca_tc`** | +62 (2) | **−97** (191) | strongest gating dependence in the program |

Gating (the router) does **more** work on Mistral than on Qwen — consistent with the reading
that Mistral's mechanism is *"perturb the right samples"*, not *"perturb along the right
direction."*

## 8. What the three-model picture now supports

1. **Channel structure is dataset × model specific** — third confirmation (3 / 5 / 3 stable; Mistral ⊂ Qwen, ≡ Phi).
2. **Depth-normalized layer selection is architecture-portable** (0.69 vs 0.71 across 32 vs 28 layers).
3. **R2 (cos → PCA-1) is portable as a rule**, though it fires on a different channel per model.
4. **R1's absolute norm floor is NOT portable** — must be relative (norm-ratio).
5. **Router AUC predicts neither utility nor specificity.**
6. **Direction-specificity is model-specific and does not follow from Net.** Held on Qwen (5/5 seeds); fails on Mistral *with a larger Net*.
7. **Saturation is a confound for interventional claims.** The more a model collapses onto one response mode, the more *any* large perturbation "fixes" errors and the weaker every placebo becomes. This parallels the archived MetaTool binary-space saturation caveat — but here it arises in a 3-class space from **model** saturation rather than **label-space** size, which is a new and more general form of the problem.
8. **Selecting α by val-Net alone self-selects the non-specific regime** (α saturates at the grid top on Mistral for all 3 channels — exactly where random becomes competitive).

**Conservative program-level statement.** The quantitative mechanistic story remains **one
model family (Qwen2.5-7B: multi-seed Net +92 ± 20, real ≥ all random 5/5)**. Phi contributes
summary-level baseline/channel evidence. **Mistral contributes a boundary**: the procedure
generalizes, the discovered structure is coherent, and the behavioral gains are real and
large — but the **causal-specificity claim does not transfer**, and **Net-only cross-model
comparison is actively misleading**.
