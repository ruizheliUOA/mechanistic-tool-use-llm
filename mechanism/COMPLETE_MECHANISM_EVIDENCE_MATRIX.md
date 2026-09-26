# Complete mechanism evidence matrix

25 evidence rows. Machine-readable companion: `COMPLETE_MECHANISM_EVIDENCE_MATRIX.csv`.
Stage distribution: DISCOVER 4 · DETECT 3 · CORRECT 11 · VERIFY 3 · LICENSE 4.

## Part 2 — DISCOVER

```
CHANNEL_DECOMPOSITION_SUPPORTED
```

- **D1 — yes.** The three SAKIKO channels are the dominant over-commitment
  transitions: **622 / 513 / 465 of 2052 errors ≈ 78%**, threshold- and
  split-stable. Tool-decision errors are not one homogeneous binary failure.
- **D2 — yes.** Geometry differs by channel: at ca_tc's native layer
  `cos(DiffMean, PC1) = 0.41`, at rfi_tc `0.91`. Different channels want
  different estimators — and that difference is behaviourally load-bearing (M-C6).
- **D3 — partially overlapping.** At a **common layer (L20)**: rfi_tc·ca_tc
  **0.569**, rfi_tc·ca_direct **0.488**, ca_tc·ca_direct **0.706**. Not identical,
  not orthogonal. The near-zero values (0.007, 0.012) in the native-layer matrix
  are **cross-layer artifacts** and must not be read as orthogonality.
- **D4.** Probe AUC 0.77–0.95 with silhouette **−0.01…−0.05**: channels are
  **linearly separable directions inside overlapping clouds**, which is precisely
  the regime where per-channel linear steering beats one global direction. MetaTool
  adds the behavioural argument: over-call and under-call improve simultaneously on
  disjoint populations, which no single global tool-call bias can produce.
  Mechanical additivity is **not** counted as independent evidence.

## Part 3 — DETECT

**R1 — no. Decoding does not establish that a usable correction direction exists.**
This is artifact-backed twice over:

- **M-R3, the decisive case.** For ca_direct, Router AUC is **flat at ~0.96 from
  L16 to L22**, while DiffMean norm rises 2.12 → 5.88 → 10.41 → 14.26 and best-val
  Net rises **+13 → +24 → +46**. Detectability is constant across a range where
  actionability varies four-fold. At L16, `cos(DiffMean, PC1) = 0.0002` — a noise
  direction — and bootstrap stability collapses to **0.8235** (p05 0.687) against
  0.92–0.995 for healthy directions. **Two independent diagnostics converge on the
  same layer.**
- **M-R1.** Across three Qwen2.5 channels, detectability ordering
  (rfi_tc 0.751 < ca_tc 0.937 < ca_direct 0.958) does not match off-axis geometry
  ordering (0.236 / 0.663 / 0.648). Descriptive only — not a law, not a predictor.

**R2.** Readable-and-steerable: Phi (all three channels), Qwen2.5-7B, Qwen3-8B,
Qwen3-4B, Gemma-2-9b, MetaTool nt_tc/tc_nt. Readable-but-not-qualified:
ca_direct@L16 (readable, degenerate direction). Rescorability-blocked: Llama-3.1
and Mistral-7B carry no intervened-prediction field.

> **Router success identifies an internal error-associated state; it does not
> itself license intervention.**

## Part 4 — CORRECT

The intervention is not "add a vector". The pipeline is: channel assignment →
Router firing at τ → channel-specific direction → observation site distinct from
injection site → dose scaling `h' = h + q·s_c·d` → selective single-channel
injection → unmodified continuation of the forward pass.

### C4.1 Gating necessity — `STRONGLY_SUPPORTED`

| setting | gated | ungated |
|---|---:|---:|
| Phi-3.5 / W2C (within one run) | **+62** | **−32** |
| Qwen2.5-7B / MetaTool nt_tc | **+3** | **−39** |
| Qwen2.5-7B / MetaTool tc_nt | **+11** | **−34** |

**Gating does not merely reduce collateral — it preserves the sign of the effect.**
Two models, two corpora, three channels, sign inversion in every case. The Phi
figure is a *within-run* contrast (its reference is +62, not the frozen +55) and
must be quoted that way.

### C4.2 Channel matching — `SUPPORTED`

REAL **+55** vs MISMATCHED_CHANNEL **−17** on the frozen Phi test: applying a
channel's direction to the wrong channel is worse than doing nothing. With M-D2
(channel-specific geometry) and MetaTool bidirectionality, correction is
**channel-conditional, not a generic tool-use bias**.

Honest caveat retained: ca_tc's reverse control retains **46%** of its channel
fixes — a partial generic push-off-tool_call component exists.

### C4.3 Direction specificity — `SUPPORTED`

Historical aggregate (Phi, frozen): REAL +55 · RANDOM +14 · REVERSE +14 ·
WRONG_LAYER +3 · MISMATCHED −17. Qwen2.5-7B: real +79 vs reverse +16 vs random
mean 25.2 (max 50); real ≥ all random in 5/5 seeds. Modern sealed: **0 of 59**
matched randoms exceed real. Caveat: in the **binary** MetaTool space 2/25 randoms
≥ real — binary label spaces weaken specificity tests.

### C4.4 Layer/site dependence

Three distinct sites, **not interchangeable**:

| site | Phi | evidence it is separate |
|---|---|---|
| observation | L18 | Router AUC flat across layers (M-R3) |
| direction estimation | L14/L16 | L16 vs L20 DiffMean differ in norm 2.12 vs 14.26 |
| injection | L14 (mlp_all) / L16 (mlp_prompt) | WRONG_LAYER control collapses to +3 |

Raw cross-layer cosine is **not** evidence of semantic orthogonality and is
classified conservatively throughout. The behavioural evidence that site matters
(WRONG_LAYER +3 vs REAL +55; ca_direct L16→L20 Net +13→+46) is retained as valid.

## Part 5 — estimator evolution

The sequence is a genuine scientific finding and was previously omitted:

1. **E1** historical DiffMean, outcome-conditioned.
2. **E2** off-axis / wrong-sign / cross-layer concerns surface;
   `Q_channel` 0.236–0.663 shows substantial off-axis energy.
3. **E3** same-layer DiffMean control isolates layer from method —
   *"ca_direct@L16's problem is the layer, not the method."*
4. **E4** rule-driven estimator selection: PCA-1 where `cos(DM, PC1)` is low
   (ca_tc, 0.41) → **locked test +50 vs +36**; DiffMean where it is high
   (rfi_tc, 0.91). On MetaTool the same rule chose PCA-1 for under-call and
   DiffMean for over-call on **5/5 seeds, decided on validation only**.

> **The failure was of a particular estimator, not evidence against the existence
> of target-associated structure. Representation structure and estimator
> recoverability are distinct properties.**

Scope: supported on the named channels only. Not generalised.

## Part 6 — representation geometry

- **G1/G2 — multi-PC emergence.** No single PC achieves more than `rfi_net = 7`;
  only PC9 removal helps (+5 val, +1 test); PC19 removal rejected. **Correction is
  distributed across the subspace, not carried by one semantic axis.** The PCA
  subspace is recorded as *exhausted* — all 20 PCs audited.
- **G2b — the Broke set is sample-constant.** `Jaccard = 1.000` between full,
  minus-PC9 and cum-PC0–6 configurations; single-PC configs break strict subsets
  of the same 22 samples. **Damage falls on a fixed vulnerable subpopulation and is
  a property of the samples, not of the particular direction.** This is the single
  most useful preservation insight in the historical record.
- **G4 — channel geometry:** same-layer cosines 0.488–0.706 (above).
- **G5 — bootstrap stability:** healthy directions **0.92–0.995**; PCA-1 uniformly
  0.9925–0.9952. Learned directions are **not** training-sample noise.

Language held at **"decision-associated linear structure"** — no causal-circuit
claim. G3 (prompt-length nuisance correlation) was **not located** in the surviving
artifacts and is therefore **not asserted**.

## Part 7 — VERIFY

- **V1 — yes.** Phi: 153 source exits → **107 gold, 46 other-wrong**. A direction
  can move behaviour decisively while about 30% of the movement (46 of 153 exits)
  lands on a third wrong action.
- **V2 — yes.** τ=0.8 and τ=0.9 both give **Net +49** with F/B 2.96 vs 3.58 and
  exposure 209 vs 185. Equal aggregates, different composition.
- **V3 — yes.** Binary changed/not-changed accounting cannot express `OTHER_WRONG`.
  MetaTool proves the point negatively: in a binary space `OTHER_WRONG` is empty by
  construction and `target_hit ≡ 1`, so the quantity that separates Steerable from
  Correctable is undefined there.

> **Movement ≠ destination correctness**, mechanistically: exits are cheap, arrivals are not.

## Part 8 — score-space and boundary

- **S1 — yes**, decision-associated structure pre-exists the intervention (G1/G2).
- **S2 — no**, not reducible to one boundary position: multi-PC emergence and the
  non-monotone dose→target_hit curve both contradict it.
- **S3 — partially**; the score-space comparator is **SUPPORTING only** — all
  paired tests were ns after Bonferroni.
- **S4 — no.** Similar aggregate gain does not imply identical sample-level
  destination (V2).

**Permitted synthesis, unchanged:** *activation intervention interacts with
pre-existing decision-associated structure, but its finite-dose outcome is not
fully characterised by a single score-space displacement.*

## Part 9 — dose and exposure

Correctability is protocol-conditional over `(M, D, c, I, q, τ, ℓ)`. Notation is
**internal only**. Established: exposure explains most Broke variation
(`Broke/R` flat 13.2%→10.3%); `target_hit` varies 0.53→0.91 with dose and is
non-monotone; exit-rate matching reverses cross-model destination comparisons.

## Part 10 — preservation as mechanism

```
gating -> exposure -> preservation
```

Router gating is not a compute optimisation. It is the **only** control on
exposure, and exposure is the dominant driver of damage (M-P2). Ungated
intervention inverts the sign in two independent settings (M-C1, M-C2). And the
damaged set is fixed across directions (M-C10), so preservation cannot be fixed by
choosing a better direction — only by not firing.

E1/E2 discipline preserved exactly. `exposure = 0` ⇒ **VACUOUS**, never
"preserved" (M-P3). The frozen gate uses E1 and is therefore permissive relative
to E2 (M-P4).
