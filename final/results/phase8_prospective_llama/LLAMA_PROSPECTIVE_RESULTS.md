# Llama-3.1-8B — Prospective Actionability-Gate Validation (Parts 3–5)

**Claim under test:** does the *frozen* Gate v2 make correct actionability **decisions** on an
architecture that did not influence its design? Endpoint = **decision accuracy**, not Net.

**Verdict: B — PROSPECTIVE REJECTOR SUPPORTED, ADMISSION UNRESOLVED.**

Gate v2 (hash `45a3f8a7…`, verified unchanged before test access) admitted **0** channels and
rejected **3**. The bounded rejection audit tested 2 of those rejections on the locked test:
**both DENIED**, as the gate predicted. Zero over-admissions (F-1), zero over-rejections (F-2).
The pre-registered fragility predictions (hashed before any intervention) were **correct on
both audited channels**.

---

## 1. Gate decisions (validation only, frozen + hashed BEFORE any test access)

| channel | decision | response shape | reason |
|---|---|---|---|
| rfi_tc | **REJECT** | boundary-seeking | best Net +18 @ρ=8; fails broke_controlled, real_gt_reverse, z_ok, nonparam_ok, interior_rho |
| ca_tc | **REJECT** | boundary-seeking | best Net +23 @ρ=8; fails not_redirection, interior_rho |
| ca_direct | **REJECT** | boundary-seeking | best Net +23 @ρ=8; fails interior_rho |

Stage 1 excluded `ca_rfi` (99 errors) and 5 smaller transitions on the **power floor**
(own-val-errors < 30) — the Phase-7 threshold behaving as designed.
Stage 2/3 passed for all three (AUC 0.837/0.959/0.937; split-half 0.898/0.889/0.755).
**All three independently selected obs L22 = 0.688 normalized depth** — the third architecture
to converge there (Mistral 0.688, Qwen 0.714). R2 did not fire on any channel (cos > 0.6).

## 2. Full validation ρ-response curves (`llama_rho_curves.json`; fig2)

**rfi_tc** (DiffMean, inj L22, thr 0.4, AUC 0.837):

| ρ | real | reverse | random μ±σ (max) | n≥real | z | own residual |
|---|---|---|---|---|---|---|
| 0.25 | +1 | +1 | 0.6±0.8 (3) | 9/20 | 0.50 | 101→100 |
| 0.5 | 0 | +1 | 0.9±0.9 (3) | 20/20 | −1.03 | 101→101 |
| 1 | +1 | +1 | 1.1±1.2 (5) | 14/20 | −0.09 | 101→101 |
| 2 | +1 | 0 | 0.9±0.7 (2) | 14/20 | 0.14 | 101→101 |
| 4 | +3 | +1 | 2.4±1.3 (6) | 6/20 | 0.51 | 101→99 |
| **8** | **+18** | **+18** | **24.1±8.7 (38)** | 16/20 | **−0.71** | 101→30 |

At the boundary the **reverse direction is identical to real (+18)** and the **random mean
(24.1) exceeds real** — a pure magnitude effect with no directional content.

**ca_tc** (DiffMean, inj L18, thr 0.4, AUC 0.959):

| ρ | real | reverse | random μ±σ (max) | n≥real | z | own residual |
|---|---|---|---|---|---|---|
| 0.25–2 | 0 | 0/+1 | 0.8–1.2 | 19–20/20 | −1.0 … −1.4 | 79→76/77 |
| **4** | **+8** | **0** | **−0.3±1.5 (3)** | **0/20** | **+5.46** | 79→44 |
| 8 | +23 | 0 | 4.5±7.1 (20) | 0/20 | +2.62 | 79→12 |

**ca_direct** (DiffMean, inj L18, thr 0.4, AUC 0.937):

| ρ | real | reverse | random μ±σ (max) | n≥real | z | own residual |
|---|---|---|---|---|---|---|
| 0.25–2 | 0…+2 | +1/+2 | 0.8–4.5 | 16–20/20 | −1.1 … −0.2 | 73→57…73 |
| 4 | +6 | +2 | 5.5±2.1 (11) | 9/20 | 0.26 | 73→51 |
| 8 | +23 | +1 | 7.0±5.5 (21) | 0/20 | **2.92** | 73→33 |

### A new taxonomy entry: **direction-specific redistribution**

`ca_tc` @ ρ=4 is the most interesting single point in the program: real +8 vs random
**−0.3±1.5 with 0/20 ≥ real (z = 5.46)**, reverse 0, residual 79→44 (−44%), at an **interior**
ρ. It fails the gate on **`not_redirection` alone**: 35 errors left the ca_tc bucket but only
~8 reached gold, so ~27 landed on *other wrong* labels. The direction is genuinely specific —
it reliably pushes the model **off** `tool_call` — but it does not deliver them to
`cannot_answer`. **Specificity without correction.** The locked-test audit (§3) confirms the
gate was right to refuse it: at test, z collapsed to 0.879.

## 3. One-shot locked test — bounded rejection audit (Part 4)

**Audit selection (disclosed, made before test access, adversarial to the gate):** the
protocol allows "up to 2". The slots went to the rejections **most likely to be wrong** —
(a) max interior specificity (`ca_tc`, val z=5.46) and (b) highest best-Net (`ca_direct`, +23,
tie-broken over rfi_tc's +18). This selection can only make the gate look *worse*, never
better. `rfi_tc` (the most obviously-correct rejection: random > real, reverse = real) was
therefore **not** audited — a deliberate choice to spend the budget on falsification.

| audited channel | locked cfg | routed | real Net | reverse | random μ (max, n≥real) | **z** | ungated | wrong-layer | **verdict** |
|---|---|---|---|---|---|---|---|---|---|
| **ca_tc** | ρ=8, inj L18, DiffMean, thr 0.4 | 69 | +16 (19 fix / 3 broke) | **−2** | 6.75 (max **36**, **5/20**) | **0.879** | **−140** | +2 | **DENIED** |
| **ca_direct** | ρ=8, inj L18, DiffMean, thr 0.4 | 67 | +19 (19 fix / 0 broke) | +0 | 8.65 (max 30, **3/20**) | **1.256** | +25 | **+61** | **DENIED** |

Both fail the frozen confirmation rule on `z_ok` and `nonparam_ok` (they pass
real>reverse, broke_controlled, residual_down — i.e. they *look* behaviourally fine and are
still not direction-specific).

Two damning details:
- **ca_direct's wrong-layer control (+61) is 3× its real effect (+19)** — injecting the same
  vector at L5 (a layer with no claimed role) works *better*. Textbook non-specificity.
- **ca_tc's ungated arm is −140** — the router is the only thing preventing catastrophe, and
  the direction still isn't specific.

## 4. Gate scoring vs locked-test truth

| | test CONFIRMS | test DENIES |
|---|---|---|
| **gate ADMIT** | TP = **0** | FP (F-1 over-admission) = **0** |
| **gate REJECT** | FN (F-2 over-rejection) = **0** | TN = **2** |

- **admission precision:** undefined (no admissions to test) → **admission remains unvalidated**
- **rejection precision:** **1.0 (2/2)**
- **F-1 events: 0** · **F-2 events: 0** · **F-3: registered predictions correct 2/2**

## 5. Registered prediction check (hashed pre-intervention, `registered_predictions.json`)

| channel | F1 = P(gold=runner-up) | F2 = med(top−gold) | flag | predicted | actual | correct? |
|---|---|---|---|---|---|---|
| rfi_tc | 0.652 | 0.663 | FRAGILE | non-actionable | not audited | n/a |
| ca_tc | 0.407 | 0.636 | FRAGILE | non-actionable | **non-actionable** | ✅ |
| ca_direct | 0.440 | 0.624 | FRAGILE | non-actionable | **non-actionable** | ✅ |

The Phase-6 fragility statistic made a **falsifiable pre-registered call on an unseen
architecture and was right both times** (2/2; F-3 threshold allows ≤1 error). Notably it
predicted non-actionability from a *Qwen-like* over-call share (0.64) — the prediction came
from the **margin structure**, not the over-call rate.

## 6. Parts 5–6: multiseed and OOD — **NOT RUN** (prerequisites unmet)

- **Multiseed:** the frozen rule permits it only if the pilot admission rule holds (≥1 admitted
  channel confirming at Stage 6). Zero channels were admitted ⇒ **not run**. Running seeds to
  hunt for a favourable one is explicitly prohibited.
- **OOD:** requires multiseed confirmation first ⇒ **not run**. Feasibility *was* established
  (tool-domain split; SEEN 3266 / UNSEEN 386; all three channels clear the floors) and the
  design is frozen in `OOD_PROTOCOL.md` so it cannot be fitted after the fact.

## 7. What this establishes, precisely

**Established:** on an architecture that never influenced its design, the frozen gate's
**rejections were prospectively correct (2/2)**, with the strongest candidate over-rejection
(`ca_tc`, val z=5.46) confirmed as a true rejection at test; and a pre-registered
pre-intervention statistic predicted both outcomes correctly.

**Not established:** the **admission** half. Llama produced no admitted channel, so the gate's
positive predictive value remains untested on any unseen architecture. Gate v2 therefore stays
what Phase 7 called it: **a reliable rejector, not a validated admitter.**
