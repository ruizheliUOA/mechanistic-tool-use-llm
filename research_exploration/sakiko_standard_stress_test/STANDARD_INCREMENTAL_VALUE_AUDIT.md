# Does the licence change decisions relative to simpler practice?

Nine units with recoverable endpoints, run through progressively richer rules. Historical
verdicts unchanged; this is retrospective methodological comparison.

| rule | accepts | what the added component removed |
|---|---:|---|
| R0 `Net > 0` | **9** | — (accepts everything, including Mistral, Llama, Phi) |
| R1 `+ specificity` | **6** | Llama ×2, Mistral `rfi_tc` — directions no better than random |
| R2 `+ target-hit > 0.5` | 5 | Mistral `ca_tc` — **but by data unavailability, not a demonstrated filter** |
| R3 `spec + TG > 0` | 5 | no change vs R2 |
| R4 **destination without specificity** | 6 | *re-admits* Llama `ca_direct` — worse than R1 |
| R5 `+ collateral ≤ 0.05` | **4** | **Phi-3.5** — the only rule that catches 19.7% real breakage |
| R5b `+ non-vacuous exposure` | **3** | **Gemma `ca→direct`** — 0 exposed rows, untestable safety |
| R6 **full SAKIKO** | **1** | **Qwen3-4B and Gemma `ca→tc`** — the CI layer |

## Findings

**1. `Net > 0` is worthless.** It accepts all nine units, including two where the learned
direction was no better than random (Mistral `rfi_tc`: real +12 vs random mean +29, reverse +19;
17/20 randoms ≥ real) and one with 19.7% collateral.

**2. Specificity is the single highest-yield condition** — it alone removes three units. Any
paper reporting matched-random and reverse controls already captures most of the available value.

**3. Destination accounting without specificity is actively worse** (R4 re-admits Llama
`ca_direct`). Destination is not a substitute for a causal control.

**4. Each remaining layer removes exactly one distinct unit.** Collateral catches Phi and
nothing else catches Phi. Non-vacuity catches Gemma `ca→direct` and nothing else catches it.
The CI layer catches Qwen3-4B and Gemma `ca→tc` and nothing else catches them.

**5. The honest limitation: the CI layer is the *only* thing separating SAKIKO from
`specificity + destination + collateral + non-vacuity`.** Everything SAKIKO uniquely does at the
formal stage is **certification**, not destination measurement. On these units, destination
accounting never changed a decision relative to a rule that already had specificity.

This materially qualifies the earlier claim that "destination-resolved accounting adds
information beyond Net." It adds information beyond Net **alone**; it did not add decision-
relevant information beyond Net + specificity on any observed formal unit.

## Where destination accounting *would* bind

Simulation confirms it is not vacuous: the pathology "many exits, wrong destination"
(80 exits, 25 gold, 55 other-wrong) passes specificity and collateral and is caught **only** by
the destination conditions. Llama's locked TEST is the closest real instance (19 gold vs 44
other-wrong, target-hit 0.30) but failed specificity first. So destination is conceptually
load-bearing and empirically untriggered.
