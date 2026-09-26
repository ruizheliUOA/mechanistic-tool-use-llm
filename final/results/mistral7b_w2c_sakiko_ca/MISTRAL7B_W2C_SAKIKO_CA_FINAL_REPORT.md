# Mistral-7B-Instruct-v0.3 × When2Call — Target-Native SAKIKO-CA (Phase 5)

**Date:** 2026-07-15 · Branch `exp/sakiko-followup-archive` (unchanged; no git writes).
**Question:** can SAKIKO-CA automatically discover and validate *model-specific* behavioral
channels in a second, independent 7B architecture — without reusing Qwen's channels, layers,
routers, or directions?

**One-line answer:** **The procedure transfers; the causal-specificity claim does not.**
The full pipeline ran end-to-end target-natively and produced large behavioral gains
(manual 3-channel cascade **Net +82, accuracy +15.0 pts** on the locked test — *exceeding*
Qwen's archived +79 / +14.4 pts), but the pre-registered placebo battery shows **2 of 3
discovered channels are not direction-specific**, the **cascade fails `real ≥ all random`**
(which Qwen passes), reverse retains **88%** of the effect (Qwen: 20%), and the automatic
utility gate admits **zero** channels. Multi-seed was **not** run (pre-registered gate not
satisfied).

> **Headline for reviewers:** on a second architecture SAKIKO-CA gets a *bigger* number and a
> *worse* mechanism. Raw Net went **up** (+82 vs +79) while the specific component collapsed
> **4.5×** (marginal-over-random +11.9 vs +53.8). Phase 5's contribution is this boundary,
> not a second confirmation.

---

## 1. R0 — Mistral W2C baseline · **PASS**

| metric | value |
|---|---|
| N evaluated / skipped | 3652 / **0** |
| accuracy | **0.4269** |
| balanced accuracy | 0.4056 |
| macro-F1 (3-cls) | 0.3319 |
| majority-gold baseline | 0.3546 |
| acc − majority | **+0.0723** |
| total errors | 2093 |
| deterministic replay (40) | **bit-identical** (pred 40/40, avg_logp 40/40) |
| readout collapse flag | **False** |

Gold: tool_call 1295 / cannot_answer 1295 / request_for_info 1062.
Pred: **tool_call 2999 (82.1%)** / direct 327 / cannot_answer 223 / request_for_info 103.

> **Mistral over-commits to tool-calling far harder than either predecessor**
> (tool_call share: Mistral **0.821** vs Phi 0.676 vs Qwen 0.614). It almost never asks for
> information (2.8% of predictions). This single fact drives every result below.

## 2. Automatic channel discovery (train/val only)

| channel | gold → pred | all | train | val | test | status | bootstrap | scope |
|---|---|---|---|---|---|---|---|---|
| rfi_tc | request_for_info → tool_call | 948 | 662 | 141 | 145 | stable | **1.0** | PASS |
| ca_tc | cannot_answer → tool_call | 773 | 541 | 117 | 115 | stable | **1.0** | PASS |
| ca_direct | cannot_answer → direct | 291 | 205 | 43 | 43 | stable | **1.0** | PASS |
| ca_rfi | cannot_answer → request_for_info | 26 | 18 | 2 | 6 | insufficient | 0.0 | — |
| rfi_direct / rfi_ca / tc_direct / tc_ca / tc_rfi | … | 21/17/15/1/1 | | | | insufficient | 0.0 | — |

**Structure verdict: STRICT SUBSET of Qwen's.** Mistral yields exactly the three
over-commitment channels; Qwen's fourth validated channel **`ca_rfi` (219 on Qwen) collapses
to 26 on Mistral**, and `tc_rfi` (133 on Qwen) to **1**. Because Mistral funnels nearly
everything into `tool_call`, the "destination = request_for_info" channels essentially do not
exist. Relative to **Phi**, the structure is **matching** (same three).

*Discovery recovered the manual SAKIKO-v1 set exactly, and found **no extra** channels — a
different outcome from Qwen, and derived rather than assumed.*

## 3. Geometry (R1/R2) — target-native

| channel | R1 obs (norm. depth) | DiffMean norm | norm-ratio | router val-AUC | cos(DM,PC1) | R2 methods |
|---|---|---|---|---|---|---|
| rfi_tc | **L22 (0.69)** | 0.723 | 0.247 | 0.878 | 0.788 | diffmean |
| ca_tc | **L22 (0.69)** | 0.794 | 0.271 | **0.964** | 0.735 | diffmean |
| ca_direct | **L22 (0.69)** | 0.515 | 0.176 | 0.938 | **0.560** | diffmean, **pca1** |

Two independent cross-model confirmations of the *rules* (not the results):
1. **Normalized depth converges.** All three Mistral channels independently select **L22 =
   0.69 depth**; Qwen's chosen obs was **L20 = 0.71 depth**. Absolute indices differ (22 vs
   20), normalized depth agrees — vindicating depth-normalized layer grids.
2. **R2 fired correctly and was vindicated.** Only `ca_direct` has cos(DM,PC1) < 0.6 (0.560)
   → PCA-1 added as a candidate → **PCA-1 won on validation** (and was locked). This
   reproduces, on a new architecture, the Qwen `ca_tc` PCA-1 finding.

**Methodological correction (train-only, pre-test).** The archived R1 gate rejects layers
with *absolute* DiffMean norm < 5.0. That constant is **Qwen-calibrated and does not
transfer**: Mistral's MLP-output norms are ~5–10× smaller (median activation norm 1.6–4.5),
so the absolute floor rejected **every** layer of **every** channel. Phase 5 therefore uses
the **model-agnostic relative criterion the method's own spec defines** — reject
norm-ratio < 0.05, then select by **max norm-ratio** (tie-break AUC). No test data was
involved. *Finding: absolute norm thresholds are not portable across architectures; the
relative ratio is.*

## 4. Locked configurations (validation-tuned; locked before test)

| channel | method | obs | inj | α | thr | router val-AUC | val Net |
|---|---|---|---|---|---|---|---|
| rfi_tc | diffmean | 22 | 20 | **6.0** | 0.6 | 0.878 | +11 |
| ca_tc | diffmean | 22 | 18 | **6.0** | 0.8 | 0.964 | +54 |
| ca_direct | **pca1** | 22 | 18 | **6.0** | 0.4 | 0.938 | +13 |

> **α saturates at the grid maximum (6.0) for all three channels.** Val-Net rewards large
> perturbations; §6 shows this is precisely what makes random directions competitive.

## 5. Locked test — run once per arm (n = 548)

Baseline: acc **0.42518**, macro-F1(3) 0.32262; residuals rfi_tc 145 / ca_tc 115 / ca_direct 43.

| arm | channel(s) | acc | Δacc | Fixed | Broke | **Net** | touched | damage on correct |
|---|---|---|---|---|---|---|---|---|
| A | baseline | 0.4252 | — | — | — | — | 0 | — |
| C | **ca_tc** | **0.5383** | **+11.3 pts** | 64 | 2 | **+62** | 106 | 2 |
| C | rfi_tc | 0.4471 | +2.2 | 16 | 4 | +12 | 131 | 4 |
| C | ca_direct | 0.4434 | +1.8 | 10 | 0 | +10 | 38 | 0 |
| **D** | **auto SAKIKO-CA cascade** | — | — | — | — | **NOT RUN — empty** | — | — |
| F | manual {ca_tc, rfi_tc, ca_direct} *(diagnostic)* | **0.5748** | **+15.0 pts** | 88 | 6 | **+82** | 270 | 6 |

Per-channel residuals (before → after):

| arm | rfi_tc | ca_tc | ca_direct |
|---|---|---|---|
| C_ca_tc | 145→142 | **115→14** | 43→**65** ⚠ |
| C_rfi_tc | 145→117 | 115→111 | 43→43 |
| C_ca_direct | 145→145 | 115→115 | 43→24 |
| F_manual | 145→116 | 115→12 | 43→46 |

**Arm D is empty because no channel passed the 8-criteria utility gate (§7).** Arm F is a
*diagnostic* showing what the historically-important channels do under identical target-native
rules — it is **not** the automatic system's output and is not claimed as one.

## 6. Placebo & specificity controls — the decisive evidence

Identical router/threshold/α/layer/eligible-set/direction-norm; 20 matched-norm random directions.

| channel | **real** | reverse | random mean ± std (max) | real percentile | #random ≥ real | ungated (broke) | wrong-layer L5 | **verdict** |
|---|---|---|---|---|---|---|---|---|
| **rfi_tc** | +12 | **+19** | **29.1 ± 14.0 (54)** | **15th** | **17/20** | +18 (30) | −13 | **NON-SPECIFIC** |
| **ca_tc** | **+62** | +41 | 35.5 ± 13.9 (67) | **95th** | **1/20** | **−97 (191)** | +53 | **direction-specific (with a large generic component)** |
| **ca_direct** | +10 | **+12** | 8.7 ± 4.3 (20) | 65th | **7/20** | +12 (0) | **+24** | **NON-SPECIFIC** |

Readings:

- **`rfi_tc` — non-specific, decisively.** Random directions average **+29** versus the
  learned direction's **+12**; reverse (+19) also beats real; 17/20 randoms ≥ real. The
  learned direction is *worse than noise*. (Its wrong-layer control, −13, does show the
  **layer** matters even though the direction does not.)
- **`ca_direct` — non-specific.** Reverse ≥ real, 7/20 randoms ≥ real, and the
  **wrong-layer control (+24) beats real (+10)** — a textbook non-specificity signature
  (identical to the archived Phi→Qwen mapping failure mode).
- **`ca_tc` — the one real signal.** Real (+62) sits at the **95th percentile** of the random
  distribution with only **1/20** random ≥ real, and beats reverse (+41). It is the only
  channel where the learned direction carries genuine, measurable information (≈ +27 Net over
  generic disruption). It is also **gating-critical**: ungated collapses to **−97** (191
  broke) — the router is doing indispensable work. *But* reverse retains **66%** and
  wrong-layer retains **85%** of the effect, so a large **generic suppression** component sits
  underneath the specific one.

### 6b. Cascade-level placebo (Arm F) — matched to the archived Qwen protocol

The archived Qwen headline carries a **cascade-level** placebo; the per-channel table above is
not directly comparable to it. So the identical control was run on Mistral's Arm F cascade
(same locked configs, n_random = 10, no selection changed):

| control | **Qwen2.5-7B** (archived pilot) | **Mistral-7B-v0.3** (Arm F) |
|---|---|---|
| real Net | +79 (Fixed 92 / Broke 13) | **+82** (Fixed 88 / Broke 6) |
| Δ accuracy | +14.4 pts (0.4124→0.5566) | **+15.0 pts** (0.4252→0.5748) |
| reverse Net | **+16 → 20% retention** | **+72 → 88% retention** |
| random ×10 | mean **25.2 ± 12.8**, max 50 | mean **70.1 ± 19.9**, max **119** |
| **real ≥ all random** | **TRUE (0/10 ≥ real)** | **FALSE (1/10 ≥ real)** |
| **marginal over random mean** | **+53.8** | **+11.9** |
| ungated | +94 (broke 22) | −21 (broke 130) |
| channel-error opportunity (test) | 253 / 548 | 303 / 548 |
| fixed-rate of opportunity | **92/253 = 36.4%** | 88/303 = **29.0%** |

> **This is the central result of Phase 5.** Mistral's raw Net (+82) and Δaccuracy (+15.0 pts)
> **exceed** Qwen's (+79, +14.4 pts) — from a *larger* error-opportunity pool and at a *lower*
> fixed-rate. Yet its **specific component is ~4.5× smaller** (+11.9 vs +53.8 marginal over
> random), **reverse retains 88%** of the effect (vs 20% on Qwen), and **one random direction
> (+119) outright beat the real cascade**. A Net-only evaluation would have declared Mistral a
> *better* result than Qwen. The placebo battery says the opposite: on Mistral the cascade is
> **not direction-specific at the system level**.

**Mechanistic interpretation.** Because Mistral is *saturated* toward `tool_call` (82%), that
mode is fragile: at α = 6 **any** large perturbation at mid-late depth knocks the model off
`tool_call`, and since routed samples are by construction non-`tool_call`-gold, such
de-saturation scores as "Fixed" regardless of direction. This is why random means are large
(+29/+35) rather than ≈ 0. Only for `ca_tc` does the learned direction add signal beyond
de-saturation. **A large share of the headline behavioral gain on Mistral is gated
magnitude, not learned direction.**

**Methodological consequence (new, actionable).** Selecting α by **val-Net alone drives the
system into exactly the non-specific magnitude regime**: val-Net is maximized at the grid top
(α=6) for all three channels, and that is where random directions become competitive. A
specificity-aware selection objective (e.g. penalize configurations whose gain is matched by
matched-norm random directions on *validation*) is required before SAKIKO-CA can claim
direction-specificity on a saturated model. This is a concrete defect of the current rule set
exposed only by running it on a second architecture.

## 7. Utility gate (8 criteria, pre-registered) — **0 / 3 pass**

| criterion | rfi_tc | ca_tc | ca_direct |
|---|---|---|---|
| 1 Net > 0 | ✅ | ✅ | ✅ |
| 2 own residual ↓ ≥25% | ❌ (145→117) | ✅ (115→14) | ✅ (43→24) |
| 3 Broke controlled | ✅ | ✅ | ✅ |
| 4 other channels not worsened | ✅ | ❌ (ca_direct 43→**65**) | ✅ |
| 5 not mere redirection | ❌ (13 to-gold vs 15 to-other-wrong) | ✅ (64 vs 37) | ✅ (10 vs 9) |
| 6 real > reverse | ❌ (12 < 19) | ✅ (62 > 41) | ❌ (10 < 12) |
| 7 random advantage | ❌ (15th pct) | ✅ (95th pct) | ❌ (65th pct) |
| 8 no test tuning | ✅ | ✅ | ✅ |
| **PASS** | **NO** | **NO** | **NO** |

**`ca_tc` fails on criterion 4 alone (7/8 pass), including *both* specificity criteria.** Its
failure mode is *redistribution*: fixing 64 of 101 touched `ca_tc` errors, it converts 22 into
`ca_direct` errors and 15 into `ca_rfi`-type errors. Those samples were **already wrong**
(gold `cannot_answer`, predicted `tool_call`), so this is **not new damage** — Net (+62) and
accuracy (+11.3 pts) already account for it. Under a *pre-declared alternative* that treats
redistribution among already-wrong labels as non-damage, `ca_tc` would pass 8/8. **I am not
applying that alternative post hoc**: the pre-registered gate says fail, and the reported
verdict is fail. Both readings are stated so a reviewer can judge.

## 8. Pilot → multi-seed gate: **FAILED → multi-seed NOT run**

Of the 10 mandatory criteria, **#5** (existing error channels not severely worsened — ca_tc→ca_direct)
and **#10** (multi-channel composition; no auto-cascade could be formed) fail; #7/#8 fail for
2 of 3 channels. Per the locked protocol ("proceed only if all mandatory criteria hold"; "if
specificity fails but behavioral gain remains: stop"), **the run stops at the pilot.** No
multi-seed numbers are reported, and none should be inferred.

## 9. Answers to the Phase-5 secondary questions

1. **Which channels emerge?** rfi_tc (948), ca_tc (773), ca_direct (291) — exactly 3, all bootstrap-stable 1.0.
2. **Match Phi/Qwen?** **Matching Phi; strict subset of Qwen** (Qwen's ca_rfi/tc_rfi are absent: 26/1).
3. **Which are behaviorally intervenable?** All 3 give positive locked-test Net; but only **ca_tc** survives specificity, and **none** passes the full pre-registered gate.
4. **Best layers vs Phi/Qwen?** **L22 = 0.69 depth for all three**, vs Qwen L20 = 0.71 — different absolute index, **same normalized depth**.
5. **DiffMean or PCA-1?** DiffMean for rfi_tc/ca_tc; **PCA-1 for ca_direct** (R2-triggered at cos 0.560, won on val).
6. **Does router AUC predict utility?** **No — explicitly refuted.** `ca_direct` has AUC 0.938 yet is non-specific (wrong-layer beats real); `rfi_tc` AUC 0.878 yet random beats real. Only ca_tc (AUC 0.964) has a specific direction. AUC ordering ≠ specificity ordering.
7. **Are gains direction-specific?** **Mostly no.** 2/3 channels: random ≥ real. 1/3 (ca_tc): yes, at the 95th percentile, but with reverse retaining 66% and wrong-layer 85%.
8. **Does channel adaptation beat a fixed copied-channel protocol?** **Not demonstrable here** — adaptation *discovered* the same set the fixed protocol uses (Arm F), so the two coincide on Mistral; and the adaptive gate admitted nothing. Channel adaptation's value on Mistral is **negative information** (it correctly reports that ca_rfi/tc_rfi don't exist), not a gain.

## 10. What this establishes

**Positive.**
- SAKIKO-CA's **procedure** is architecture-portable: discovery → scope filter → geometry
  (R1/R2) → target-native routers/directions → val-only tuning → one-shot locked test →
  controls all ran on a new 7B without reusing a single Qwen artifact.
- **Depth-normalized layer selection is validated across architectures** (0.69 vs 0.71).
- **R2 (cos(DM,PC1) < 0.6 → try PCA-1) reproduced on a new model** and won on val.
- **Gating is causally essential** where the direction is real (ca_tc ungated: −97).
- **Large behavioral gains exist** (ca_tc +62; manual cascade +82 / +15.0 pts).

**Negative (equally important).**
- **Direction-specificity does not transfer — at channel *or* system level.** Per-channel:
  random beats real for 2/3 channels. Cascade-level (matched to Qwen's protocol): **real ≥ all
  random is FALSE** on Mistral (TRUE on Qwen), reverse retains **88%** (vs 20%), and the
  specific margin over random is **+11.9 vs Qwen's +53.8**.
- **Raw Net is an actively misleading cross-model metric.** Mistral scores *higher* than Qwen
  on Net (+82 vs +79) and Δacc (+15.0 vs +14.4 pts) while being *far less* mechanistically
  specific.
- **The automatic system admits zero channels** under its own pre-registered gate.
- **Absolute R1 norm thresholds are not portable** across architectures.
- **Router AUC does not predict intervention utility or specificity.**
- **Val-Net-only selection self-selects the non-specific magnitude regime** (α saturation).

---

### Verdict

**PARTIAL / BEHAVIORAL-ONLY.** Procedure generalizes; one channel (`ca_tc`) shows genuine
single-seed direction-specificity but fails the pre-registered gate on collateral
redistribution; the other two are behavioral-only (non-specific). Multi-seed not run.
**Phase 5 does not license any claim that SAKIKO-CA's causal-specificity result generalizes to
a second architecture.** See `PHASE5_CLAIM_BOUNDARY.md`.
