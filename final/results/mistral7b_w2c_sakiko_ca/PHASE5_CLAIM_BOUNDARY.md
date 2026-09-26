# Phase 5 — Claim Boundary (Mistral-7B-Instruct-v0.3 × W2C)

Every row below is tied to a locked experiment. Nothing here may be strengthened without new
evidence. **Multi-seed was not run** (pre-registered gate failed), so *nothing* in Phase 5 is
"Robust".

---

## A. DEMONSTRATED (locked experiments, single seed 42)

| # | claim | evidence |
|---|---|---|
| A1 | Mistral-7B-v0.3's W2C response-mode readout is valid and non-collapsed: acc 0.4269 vs majority 0.3546 (+0.072), 0 skipped, deterministic replay bit-identical (40/40). | `baseline/mistral7b_w2c_baseline_summary.json` (R0_PASS) |
| A2 | Mistral over-commits to `tool_call` on 82.1% of items — more than Phi (67.6%) or Qwen (61.4%). | baseline pred distribution |
| A3 | Automatic discovery yields **exactly three** stable, bootstrap-stable (1.0) channels: rfi_tc 948, ca_tc 773, ca_direct 291. | `channel_discovery/` |
| A4 | Mistral's channel structure is a **strict subset of Qwen's** and **matches Phi's**: Qwen's validated `ca_rfi` (219) collapses to 26; `tc_rfi` (133) to 1. | `cross_model_channel_comparison.csv` |
| A5 | All three channels independently select obs **L22 = 0.69 normalized depth** (Qwen: L20 = 0.71) — different absolute index, same normalized depth. | `geometry/` |
| A6 | R2 fires only for `ca_direct` (cos(DM,PC1)=0.560 < 0.6) and **PCA-1 then wins on validation**. | `geometry/`, `pilot/locked_configs.json` |
| A7 | Single-channel locked-test Nets are positive: ca_tc **+62** (acc +11.3 pts), rfi_tc +12, ca_direct +10. | `pilot/pilot_key_table.csv` |
| A8 | The manual 3-channel cascade (Arm F, diagnostic) reaches **Net +82 / accuracy 0.4252→0.5748 (+15.0 pts)**, Broke 6. | `pilot/` arm `F_manual_diag` |
| A9 | For `ca_tc`, **gating is causally essential**: ungated Net collapses to **−97** (191 broke) versus +62 gated. | `placebos/` |
| A10 | The **automatic** SAKIKO-CA system admits **zero** channels under its pre-registered 8-criteria gate → Arm D empty. | `pilot/pilot_summary.json` gate |

## B. DIRECTION-SPECIFIC (supported by random/reverse/wrong-layer controls)

| # | claim | evidence |
|---|---|---|
| B1 | **`ca_tc` only.** Real +62 vs 20 matched-norm random directions (mean +35.5 ± 13.9): **95th percentile, 1/20 ≥ real**; real > reverse (+41). | `placebos/placebo_all_controls.json` |
| B2 | Even for `ca_tc`, specificity is **partial**: reverse retains **66%** and wrong-layer(L5) retains **85%** of the effect ⇒ a large generic-suppression component underlies it. | same |
| B3 | **At system (cascade) level, specificity FAILS.** Matched to the archived Qwen protocol (n_random=10): Mistral Arm F real +82 but **real ≥ all random = FALSE** (1/10 ≥ real; max random **+119 > +82**), reverse **+72 = 88% retention**, marginal over random mean only **+11.9**. Qwen: real +79, real ≥ all random **TRUE** (0/10), reverse +16 = 20%, marginal **+53.8**. | `placebos/cascade_placebo_armF.json` vs `7b_w2c_sakiko/qwen25_7b_placebo_controls.json` |

## C. BEHAVIORAL-ONLY (positive Net without directional specificity)

| # | claim | evidence |
|---|---|---|
| C1 | **`rfi_tc` is non-specific**: random mean **+29.1** > real **+12**; reverse +19 > real; **17/20** random ≥ real (15th percentile). The learned direction is worse than noise. | `placebos/` |
| C2 | **`ca_direct` is non-specific**: reverse (+12) ≥ real (+10); **7/20** random ≥ real; **wrong-layer (+24) beats real (+10)**. | `placebos/` |
| C3 | Their positive Nets are therefore attributable to **gated magnitude perturbation** (de-saturating a `tool_call`-saturated model), not to a learned direction. | §6 of final report |

## D. NEGATIVE FINDINGS (asserted)

| # | claim | evidence |
|---|---|---|
| D1 | **SAKIKO-CA's causal-specificity result does not transfer to Mistral**, at channel *and* system level. Qwen: real ≥ all random (5/5 seeds; pilot 0/10 random ≥ real). Mistral: random ≥ real for 2/3 channels, and the cascade fails real ≥ all random (1/10; max random +119 > real +82). | placebos vs archive |
| D1b | **Raw Net is a misleading cross-model metric.** Mistral *beats* Qwen on Net (+82 vs +79) and Δacc (+15.0 vs +14.4 pts) while its specific margin over random is **4.5× smaller** (+11.9 vs +53.8) and reverse-retention is 4.4× worse (88% vs 20%). Any cross-model Net comparison must be accompanied by error opportunity (303 vs 253), fixed-rate (29.0% vs 36.4%), Fixed/Broke, and a matched random reference. | cascade placebo + archive |
| D2 | **Router AUC does not predict utility or specificity.** ca_direct AUC 0.938 → non-specific (wrong-layer beats real); rfi_tc AUC 0.878 → random beats real. | geometry + placebos |
| D3 | **Absolute R1 norm thresholds are not architecture-portable.** The Qwen-calibrated floor (norm < 5.0) rejected *every* layer of *every* Mistral channel (Mistral median activation norm 1.6–4.5). Only the **relative** norm-ratio criterion is portable. | `geometry/mistral7b_geometry_table.csv` |
| D4 | **Selecting α by val-Net alone self-selects the non-specific regime**: α saturates at the grid maximum (6.0) for all 3 channels, which is exactly where random directions become competitive. | locked configs + placebos |
| D5 | **Channel adaptation yields no gain over the fixed protocol on Mistral** — it discovers the same set the fixed design uses, and its gate admits nothing. Its value here is *negative information* (correctly reporting ca_rfi/tc_rfi absent). | discovery + gate |
| D6 | `ca_tc` redistributes: 22 of its own errors become `ca_direct` errors (43→65), 15 become RFI-type. Not new damage (already wrong; Net accounts for it), but it fails pre-registered criterion 4. | arm `C_ca_tc` |

## E. PILOT-ONLY (single seed 42; NOT robust)

Everything in A, B, C above is **seed-42 only**, on one fixed split (train 2556 / val 548 /
test 548). The pre-registered pilot→multi-seed gate **failed** (criteria 5 and 10), so
multi-seed was not run. **No variance, no wins/losses, no ± figures exist for Mistral.**

## F. NOT DEMONSTRATED — do not claim

- ❌ That SAKIKO-CA generalizes to *every* architecture (n = 3 models, 1 dataset for the quantitative story).
- ❌ Any **robust/multi-seed** Mistral result (not run).
- ❌ That Mistral's +82 cascade is a *mechanistic* result — it is largely **gated magnitude**, and Arm F is a **diagnostic**, not the automatic system's output.
- ❌ That `ca_tc`'s specificity replicates across seeds (single seed).
- ❌ Cross-model direction transfer (archived negative; untouched here).
- ❌ OOD tool/task transfer, prompt-wording invariance, scale generalization beyond 7B.
- ❌ That the three channels are *clusters* — silhouette evidence forbids cluster language; they are linear decision directions.
- ❌ That α=6 is optimal — it is the **grid maximum**; the grid was pre-registered and not extended (extending it post-hoc would be test-driven tuning).

## G. Honest framing for the paper

> On a second, independent 7B architecture (Mistral-7B-Instruct-v0.3), the SAKIKO-CA
> **procedure** ran end-to-end target-natively and recovered a coherent, bootstrap-stable
> channel structure — a strict subset of Qwen's — with depth-normalized layer selection and the
> R2 direction rule both reproducing. It produced a large behavioral improvement
> (+15.0 accuracy points, Net +82). **However, its pre-registered specificity controls show
> that most of this gain is not direction-specific**: for two of three channels, matched-norm
> random directions match or exceed the learned direction, and the automatic utility gate
> admits no channel at all. Only `cannot_answer→tool_call` shows a genuine directional
> signal (95th percentile against 20 random directions), and even it carries a large generic
> component and fails the no-collateral criterion. **We therefore report Mistral as a
> behavioral-only / partial result and an explicit boundary on the method's causal claims**,
> not as a second confirmation. The headline mechanistic claim remains one model family
> (Qwen2.5-7B, multi-seed +92 ± 20).

**Chief lesson.** Cross-architecture generalization of an *interventional* method must be
judged by its placebo battery, not its Net. Mistral produces a *larger* single-arm accuracy
gain than Qwen's pilot while being *less* mechanistically specific — a result that a Net-only
evaluation would have reported as a triumph.
