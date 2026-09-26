# Unified SAKIKO specification — frozen

# Discover → Detect → Correct → Verify → License

There is no longer a "v1 paper" and a "v2 paper". Historical and modern evidence
are one framework, separated by provenance and protocol strength, not by story.

## Object of study

**Pre-execution tool-use decision failure**: the model selects the wrong action
path *before* any tool is executed. Action space includes `tool_call`,
`request_for_info`, `cannot_answer`, `direct_answer`.

The unit of analysis is a **directional error channel**: `gold action → predicted
wrong action`.

**Explicitly not**: tool-argument generation errors, execution failures, generic
factual hallucination, or binary tool/no-tool classification. (MetaTool-Binary is
included only as a *binary* special case, and its destination structure is
degenerate — see Limitations.)

---

## Discover

**Question:** how does this model, on this task, actually fail?
**Operation:** enumerate directional transitions; select channels by support and
stability.
**Why it exists:** tool-decision errors are **structured and heterogeneous**, not
one homogeneous binary tool-use error. Three channels carry **622/513/465 of 2052
errors (~78%)**, threshold- and split-stable.
**Strongest evidence:** the error-transition matrix; the dominant channel
**flips** between Phi and Qwen on the same data — channel structure is
model-dependent, so it must be discovered per model, not assumed.
**Permitted characterisation:** *partially overlapping but behaviourally distinct
error-associated directions* — same-layer cosines **0.488 / 0.569 / 0.706**.
**Never:** orthogonal channels. The near-zero cosines are cross-layer artifacts.
**Limitation:** channel structure is label-space dependent — 3 channels on W2C,
1 on MetaTool-Binary, 0 quantifiable on ToolSandbox.

## Detect

**Question:** is *this* sample an instance of that failure mode?
**Operation:** per-channel Router on observation-layer activations; fire above τ.
**Why it exists:** it is the **only** control on exposure, and exposure is the
dominant driver of collateral damage.
**Strongest evidence:** probe AUC 0.77–0.95 (W2C), Router AUC 0.927–0.991
(MetaTool).
**Governing principle — `Readable ≠ Steerable`:** for `ca_direct`, Router AUC is
**flat at ~0.96 from L16 to L22** while the recovered direction degrades to noise
(`cos(DiffMean, PC1) = 0.0002`, bootstrap stability **0.8235** vs 0.92–0.995
healthy) and behavioural benefit varies **+13 → +46**. Decoding an error state
does not establish that a usable correction direction exists there.
**Limitation:** Router confidence does **not** identify *which* exposed-correct
samples are safe — broken vs retained route probabilities are indistinguishable
(Cliff's δ −0.094, p 0.44). Gating is a **volume** mechanism, not a selection
mechanism.

## Correct

**Question:** can the error be moved toward the correct action?
**Operation:** channel-specific direction, estimator chosen by channel geometry,
injected at a site distinct from the readout site, at frozen dose
`h' = h + q·s_c·d`, on fired samples only, then normal forward pass.
**Why it exists:** this is Act 1 — the demonstration that a meaningful subset of
these errors is behaviourally improvable at all (a correction effect; Correctable
is assessed in Verify).
**Strongest evidence:** REAL **+55** vs RANDOM +14 / REVERSE +14 / WRONG_LAYER +3
/ MISMATCHED **−17** (Phi, frozen); **0 of 59** matched randoms exceed real on
**both** Qwen3-8B (real 38 gold, random max 8) and Gemma (real 17, random max 0);
estimator adaptivity PCA-1 vs DiffMean **+50 vs +36** on locked test.
**Gating — permitted wording (S-18, S-23):** *gating improves intervention quality
in many evaluated settings, but not all.* Removing it reverses the sign of the
effect for Phi (+62 vs −32, within one run) and for both MetaTool channels
(+3/+11 vs −39/−34); it lowers net for Qwen3-8B (+38 vs +13); it gives **no
benefit for Gemma** (+16 gated vs +17 ungated).
**Never:** that gating is necessary, universally or in any setting (S-23).

## Verify

**Question:** did it actually arrive, and what did it break?
**Operation:** resolve every outcome.

| baseline-error rows | baseline-correct rows |
|---|---|
| `SOURCE_RETAINED` | `CORRECT_RETAINED` |
| `GOLD_ARRIVAL` | `BROKEN` |
| `OTHER_WRONG` | |

**Why it exists:** `movement ≠ destination correctness`, and `aggregate gain ≠
repair`.
**Strongest evidence — two independent demonstrations:**
1. **Phi:** 153 exits → 107 gold but **46 other-wrong**; Net **+55** while
   **52/93 = 55.9%** of exposed-correct rows break.
2. **Qwen3-8B activation vs score-space:** channel-level Net +38 vs +35 and gold 38 vs 37 —
   near-identical aggregates — but OTHER_WRONG **14 vs 21** and target-hit
   **0.7308 vs 0.6379**.
**Limitation:** undefined in binary action spaces, where `OTHER_WRONG` is empty by
construction and `target_hit ≡ 1`.

## License

**Question:** what is the strongest claim this evidence permits?
**Operation:** assign the highest rung whose frozen conditions are met.

The **prospectively frozen** ladder (`FINAL_SAKIKO_DEFINITIONS.md`):

| rung | estimand |
|---|---|
| **Adjudicable** | population, instrument and support permit the question; `\|A_D \ {g,s}\| ≥ 1` |
| **Readable** | channel-error states discriminable under the declared probe at the declared site |
| **Steerable** | direction-specific effect beyond zero, reverse, matched-random and wrong-site nulls |
| **Correctable** | arrivals concentrate on `g` rather than `A_D \ {g,s}`; **undefined** at `\|A_D\| = 2` |
| **Preserving** | `P(correct→wrong \| exposed)` acceptably bounded — a *fourth* property, fails independently |
| **Licensable** | evidence precise enough under the prespecified rule at declared δ, α |

Outcomes are **ADMIT / DECLINE**. "Level-1"/"Level-2" have no prospectively frozen
definition and are not used as licence names (S-21).

**Frozen consequences that must appear in the paper:**
`DECLINE ≠ intrinsically uncorrectable` · `historical ADMIT ≠ preservation-certified safe` ·
`Licensable` is a property of the **evidence**, not of the setting.

**Repair** (title vocabulary, amended 2026-09-10) is the composite claim —
destination correctness + preservation + sufficient evidence — **not a rung** and
never added to the ladder. A frozen ADMIT is the only outcome that licenses it
(conditions 2, 3, 5, 6 destination; 7, 8 preservation on E1). See
`writing/TERMINOLOGY_MIGRATION_RECORD.md`.

The intermediate rungs carry most of the empirical content: **Correctable** and
**Licensable** dissociate, and two of three modern models sit exactly in that gap.

**Correctability is protocol-conditional:**
`model × dataset × channel × intervention protocol × frozen dose`.
Never an intrinsic model property — exit-matched cross-model destination
differences **flip sign** with the matching level.
**Limitation:** the frozen gate adjudicates preservation on **E1**, which is
permissive relative to **E2** (Qwen3-4B dev q=0.5: E1 1.3% passes, E2 5.7% would
fail). Zero exposure ⇒ **VACUOUS**, never "preserved".
