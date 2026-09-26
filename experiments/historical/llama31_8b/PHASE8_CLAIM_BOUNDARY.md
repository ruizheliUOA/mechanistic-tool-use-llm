# Phase 8 — Claim Boundary

**VERDICT: B — PROSPECTIVE REJECTOR SUPPORTED, ADMISSION UNRESOLVED.**

---

## A. Demonstrated (locked, one-shot, on an architecture that never influenced the gate)

| # | claim | evidence |
|---|---|---|
| A1 | The frozen Gate v2 was applied to Llama-3.1-8B **without modification**: lock manifest 15/15 artifacts unchanged at HEAD `1ff1f6f`; decisions hashed (`45a3f8a7…`) and hash-verified before the test driver would execute. | `PHASE8_INTEGRITY_AUDIT.md`, `GATE_V2_LOCK_MANIFEST.json` |
| A2 | The **primary** model named in the protocol was used — `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2…`, **4/4 shards SHA256-verified byte-identical to official**. The OLMo fallback was not used. | `PHASE8_EXECUTION_MANIFEST.md` §2 |
| A3 | **R0 PASS**: acc 0.4403 vs majority 0.3546 (+0.086), macro-F1(3) 0.4185, 0 skipped, replay 40/40 bit-identical, not collapsed. Strongest baseline of the four models. | `LLAMA_R0_REPORT.md` |
| A4 | Llama's discovered topology: 3 Stage-1-eligible channels (rfi_tc 678, ca_tc 535, ca_direct 463); `ca_rfi` (99) and 5 smaller transitions excluded by the frozen power/support floors. | `LLAMA_CHANNEL_TOPOLOGY_REPORT.md` |
| A5 | **Third architecture converging on normalized depth:** all three channels independently selected obs L22 = **0.688** (Mistral 0.688, Qwen 0.714). | `LLAMA_FROZEN_GATE_DECISIONS.json` (geometry) |
| A6 | Gate decisions (validation only, pre-test): **0 ADMIT / 3 REJECT**, all **boundary-seeking**. | `LLAMA_FROZEN_GATE_DECISIONS.json` |
| A7 | **Bounded rejection audit: 2/2 DENIED at locked test** → **TN=2, FP=0, FN=0**, rejection precision **1.0**; zero F-1 and zero F-2 events. | `LLAMA_PROSPECTIVE_RESULTS.{md,json}` |
| A8 | The audit was **adversarial**: slots went to the two rejections most likely to be wrong (ca_tc, val z=5.46 interior; ca_direct, best Net +23), not to the obvious one (rfi_tc). Both still denied. | `LLAMA_PROSPECTIVE_RESULTS.md` §3 |
| A9 | **Pre-registered fragility predictions (hashed before any intervention) were correct 2/2** on the audited channels — and predicted non-actionability from a *Qwen-like* over-call share (0.64), i.e. from margin structure, not over-call rate. | `registered_predictions.json` (+ `.sha256`) |
| A10 | **Net and AUC again fail to establish mechanism**, prospectively: rejected channels had Nets +16…+23 and AUCs 0.837–0.959; rfi_tc's reverse **equalled** real (+18/+18) with random mean 24.1 > real; ca_direct's **wrong-layer (+61) was 3× its real effect (+19)**; ca_tc's ungated arm was **−140**. | `llama_rho_curves.json`, `LLAMA_PROSPECTIVE_RESULTS.md` |
| A11 | **New taxonomy entry — direction-specific redistribution:** ca_tc @ρ=4 (val) had real +8 vs random −0.3±1.5, **0/20 ≥ real, z=5.46**, reverse 0, residual −44%, at an interior ρ — yet failed on `not_redirection` (≈27 of 35 moved errors landed on other wrong labels). Specificity without correction; the audit confirmed the rejection (test z=0.879). | `llama_rho_curves.json`, audit |
| A12 | OOD is **feasible** on W2C via a pre-existing tool-domain grouping (SEEN 3266 / UNSEEN 386; all three channels clear the floors); design frozen before any run. | `OOD_PROTOCOL.md` |

## B. NOT demonstrated — must not be claimed

| # | not demonstrated |
|---|---|
| B1 | **The predictive gate.** Admission precision is **undefined** — Llama produced no admitted channel, so the gate's positive predictive value remains **untested on any unseen architecture**. Gate v2 is a **reliable rejector, not a validated admitter**. |
| B2 | **That Gate v2 generalizes as a decision rule.** 2 audited decisions on 1 architecture is not a statistic. TN=2 is consistent with the gate being correct *and* with a gate that rejects nearly everything. |
| B3 | **Llama actionability of any kind.** No Llama channel is actionable under the frozen rule. |
| B4 | **Multiseed anything (Llama).** Not run — the pre-registered prerequisite (≥1 admitted channel confirming) was unmet. No variance, no ±, no wins/losses exist. |
| B5 | **OOD generalization.** Not run — gated behind multiseed. Feasibility ≠ result. |
| B6 | **That "no linear correction direction exists" in Llama or Mistral.** The search covers **MLP-output injection sites only** (4 obs depths × 3 inj offsets × 32× ρ range). **Residual-stream and attention-output sites are untested on every architecture.** This is the strongest live alternative and it is **open**. |
| B7 | **That the fragility statistic is a validated predictor.** It is 2/2 on audited channels — encouraging, and explicitly *not* a statistic. |
| B8 | **Cross-architecture actionability of SAKIKO-CA as an intervention method.** One test-confirmed actionable channel (Qwen ca_rfi) exists across four architectures. |
| B9 | **Anything about scale, other datasets, prompt-wording invariance, or cross-model direction transfer.** Untouched here. |

## C. Claims that must be WITHDRAWN or restated

- **Withdrawn:** any statement that Gate v2 "predicts actionability". Only the rejection half
  has prospective support. Correct wording: *"a prospectively validated **rejector**;
  admission remains diagnostic and unvalidated."*
- **Restated:** the programme's contribution is **evaluative/diagnostic** — a protocol that
  determines whether a detectable error channel has credible, low-damage intervention
  geometry — **not** a deployable corrective method. Per `ICLR_REVIEWER_ATTACK_MATRIX.md` #9,
  as a corrective tool it has one demonstrated channel in four architectures.
- **Unchanged:** Phase-5's Mistral verdict and Phase-7's A-global finding are untouched. No
  Mistral rescue was attempted; no Gate v3 exists.

## D. Is strict OOD justified? **No.**

The frozen prerequisite (multiseed confirmation of an admitted channel) is unmet, and running
OOD without a validated mechanism reproduces exactly the Phase-5 error: behaviourally
impressive, mechanistically uninterpretable numbers. `OOD_PROTOCOL.md` is written and frozen
so that *if* a future architecture yields a confirmed channel, the design cannot be fitted
after the fact.

## E. Is a new GPU experiment necessary? **One, and only one is defensible.**

Per the Phase-7 decision tree, the single question that (i) is still live, (ii) cannot be
answered from existing data, and (iii) distinguishes two real explanations is **H2 vs H4**:

> Is the absence of correction geometry a property of the *model*, or of the *site* we inject
> at (MLP output)?

A **pre-registered site/estimator probe** (residual-stream + attention-output × 2–3 depths ×
{DiffMean, LDA, probe-weight}, fixed ρ bracket, validation-only, z≥3 hit criterion with
multiple-comparison discipline, **no rescue permitted**) would either upgrade the negative to
"site/estimator-robust" — much stronger for the paper — or localize the mechanism. Everything
else (more architectures, more seeds, OOD) should **not** be run: it cannot distinguish live
explanations and risks manufacturing a positive.

## F. Strongest honest ICLR framing

> **"Detectability is not actionability: a prospectively-validated rejector for
> activation-intervention claims."**
>
> Automatic error-channel discovery recovers coherent, bootstrap-stable channels across four
> architectures, and normalized-depth layer selection converges (0.69–0.71) across three. But
> across ~8 controlled channels, **positive Net and high router AUC never establish
> mechanism**: matched-norm random directions match or beat the learned direction in most
> cases, reverse controls sometimes equal it exactly, and a wrong-layer control can triple it.
> We formalize an actionability gate, **freeze and hash it**, and evaluate it prospectively on
> an untouched architecture (Llama-3.1-8B): it admits nothing, and a **deliberately adversarial
> bounded audit confirms both of its most-suspect rejections (2/2)**, while a pre-registered
> pre-intervention statistic predicts both outcomes correctly. The gate's **rejections are
> prospectively correct; its admissions remain unvalidated** — we report it as a rejector, and
> we report that only one channel in four architectures has ever been confirmed actionable.

This framing is defensible line-by-line against `ICLR_REVIEWER_ATTACK_MATRIX.md`. A framing
that promised cross-architecture correction would not be — and the evidence says it would be
false.
