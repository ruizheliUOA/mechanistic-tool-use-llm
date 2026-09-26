# Final reviewer-risk closure

```
ALL_MAJOR_RISKS_CLOSED_OR_DISCLOSED
```

A risk is closed only with a **named manuscript action**. Listing is not closing.

---

## R1 — Historical Phi reproducibility → `MITIGATED_BY_ROBUSTNESS_AND_DISCLOSURE`

**Type: scientific issue, mitigated by robustness + proactive disclosure.**

**Cannot be recovered:** the cause of the divergence. The frozen run gives
107/52/+55 on 293 fired rows; an independent contemporaneous execution importing
the same code, seed, locked config and splits gives 106/44/+62 on **243** fired
rows. Routing is deterministic given trained routers, so generation
non-determinism cannot explain it, and the repository was squashed into a single
initial commit — no file-level history survives.

**Remains reproducible:** everything else. All Phi quantities regenerate exactly
from `p0_final_test_details.jsonl` — 293/200/93, 47/107/46, exits 153, target-hit
0.6993, 52 broken.

**Qualitative conclusion that survives:** across five seeds and one independent
execution, exposed-correct collateral remains above the 5% budget in every case
(thinnest margin 6.1% vs 5% at τ=0.9).

**Disclosure wording:** as frozen in `REPRODUCIBILITY_STATEMENT.md`.
**Prohibited:** 52/93 as a stable constant; any exact-reproduction claim; naming a
cause.
**Manuscript action:** disclosure paragraph in Methods **and** the range quoted
beside the point estimate wherever 52/93 appears.

---

## R2 — One formal ADMIT, two formal DECLINEs → `NON_BLOCKING_LIMITATION`

**Type: irreducible limitation.**

**Does it threaten a load-bearing claim?** **No.** No load-bearing claim asserts
that majority repair is generally achievable. C5-1 claims the *distribution* of
evidence strength; C5-2 explains analytically why licences are rare.

**Why the distribution is the empirical target:** the hierarchy discriminates in
both directions — one ADMIT, **two artifact-backed DECLINEs** on the
destination-interval layer, Steerable-but-preservation-failing once (Phi),
Effect-but-not-steerable twice (Llama, Mistral), not-adjudicable once (Qwen3.5). A framework that only ever admitted would be uninformative; the
declines are evidence that the gate has content.

**Limitation wording:** *"ADMIT at Level-2 fired once in seven models. We cannot
exclude that the licensing conjunction is too strict; no positive control on a
literature-established effect was completed."*
**Headline wording that avoids overclaim:** lead with *"the seven evaluated
models occupy distinct rungs"*, never *"SAKIKO certifies repair."*
**Manuscript action:** Section 7 presents the distribution; Section 9 states the
limitation; the abstract omits the count entirely.

---

## R3 — CAST overlap → `POSITIONING_RESOLVED`

**Type: presentation issue — fatal if mishandled, fully resolvable.**

**What CAST already does (§3.1, verified):** sample-conditional intervention gated
by a learned detector, `h' ← h + f(sim(h, proj_c h))·α·v`. **Architecturally
SAKIKO's Router.**

**SAKIKO must not claim:** conditional/detector-gated intervention · Router gating
· activation addition · representation reading · layer selection · random-direction
controls.

**What remains distinct — as composition:** multiclass destination resolution +
reverse/wrong-layer/mismatched controls + a preservation endpoint + prospectively
frozen criteria + an explicit admit/decline rule + a K≥3 native action space. No
verified paper combines these.

**Positioning:** cite CAST in **Section 3, at the point of method description**,
not defensively in Related Work. Stating the overlap where the mechanism is
introduced converts the strongest objection into an acknowledged design choice.
**Manuscript action:** one sentence in §3 naming CAST as the closest prior art for
the gating mechanism, plus the frozen related-work paragraph in §9.

---

## R4 — Score-space near-equivalence → `PROMOTED_TO_MAIN_EVIDENCE`

**Type: was a perceived threat; is in fact supporting evidence for Verify.**

Reframed: if the contribution were "activation steering works", 37-vs-38 would be
damaging. Because the contribution is **the adjudication procedure**, the
comparison is a *product* of the framework — it is precisely the distinction
aggregate metrics cannot draw, produced by the verification stage.

**Frozen wording:**
> similar aggregate improvement, different observed destination composition

**Prohibited:** activation is statistically superior · score-space is inferior ·
any inferential claim (paired tests ns after Bonferroni) · using 0/6 vs 2/6 as
preservation evidence.
**Manuscript action:** Figure 3 Panel B; C4-2 promoted to load-bearing; caption
carries the ns disclaimer and the n=6 exposure caveat.

---

## R5 — Historical/modern protocol heterogeneity → `MITIGATED_BY_PROVENANCE_SEPARATION`

**Type: presentation issue.**

**Visual separation rule, binding on every figure and table:**
- **historical / post-hoc** — Phi, Qwen2.5-7B, Llama, Mistral: separate panel or
  table block, muted/hatched treatment, labelled *historical protocol*.
- **modern / prospective / sealed** — Qwen3-8B, Qwen3-4B, Gemma: primary
  treatment, labelled *sealed protocol*.
- Never a shared axis implying one comparison; never a trend line across the two.

**Manuscript action:** provenance band is a required field in the figure
regeneration checklist; Table 1 carries an explicit `protocol generation` column.

---

## R6 — Stale figures → `REGENERATION_REQUIRED_BEFORE_DRAFT`

**Type: presentation issue, and a hard gate.**

**Every** existing figure is stale — the full set predates S-11, S-12, S-13, S-16,
S-17, S-18, S-19, E4, the mechanism synthesis and the route-confidence audit.
Requiring regeneration: F1 unified pipeline · F2 correction + controls · F3
turning point (**new Panel B**) · F4 rung distribution · F5 mechanism · all
archived channel-geometry and confusion figures if retained.

**No stale figure may receive `PAPER_CLEARED`.**
**Manuscript action:** figure regeneration from `FINAL_PAPER_EVIDENCE.csv` is
**step 1** of the writing sequence, before any prose.

---

## R7 — Dose/protocol dependence → `INTERPRETATION_RESOLVED`

**Type: scientific issue, resolved by interpretation discipline.**

**Frozen statement:** rung placement is conditional on
`model × dataset × channel × intervention protocol × dose`.

Supported by E4: exit-matched cross-model Δtarget-hit **flips sign** with the
matching level; target-hit varies 0.53→0.91 within a model on dose alone and is
non-monotone. Mitigating fact worth stating: the dose rule is frozen,
scale-matched, non-outcome-conditioned and selects the **smallest admissible
dose**, so published rungs are **conservative lower bounds**.

**Prohibited:** intrinsic model correctability · "model X is more correctable" ·
margin explains rung placement.
**Manuscript action:** Section 7 states the conditionality explicitly when the
rung distribution is introduced; the E4 exit-matched table goes to the appendix as
its support.

---

## Summary

| risk | verdict | type |
|---|---|---|
| R1 Phi reproducibility | `MITIGATED_BY_ROBUSTNESS_AND_DISCLOSURE` | scientific, disclosed |
| R2 one ADMIT / two DECLINEs | `NON_BLOCKING_LIMITATION` | irreducible, and evidentially useful |
| R3 CAST overlap | `POSITIONING_RESOLVED` | presentation |
| R4 score-space | `PROMOTED_TO_MAIN_EVIDENCE` | already resolved — now an asset |
| R5 protocol heterogeneity | `MITIGATED_BY_PROVENANCE_SEPARATION` | presentation |
| R6 stale figures | `REGENERATION_REQUIRED_BEFORE_DRAFT` | presentation, hard gate |
| R7 dose dependence | `INTERPRETATION_RESOLVED` | scientific |

**No risk is blocking. R6 is a gate on drafting figures, not on the freeze.**
