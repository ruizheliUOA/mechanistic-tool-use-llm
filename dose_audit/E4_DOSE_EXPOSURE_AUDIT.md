# E4 — dose / exposure confound audit, and review of the previous pass

```
PART 0a  PHI_PREVIOUS_PASS_NEEDS_NARROWING
PART 0b  METATOOL_PREVIOUS_PASS_NEEDS_NARROWING
PART 5   EXIT_RATE_MATCHING_IDENTIFIABLE
PART 7   E4_DOSE_CONFOUND_PARTIALLY_REDUCED
```

No GPU. No new inference. Validation/DEV artifacts only for all dose analysis.

---

# PART 0a — Phi review

## What verified exactly

The full identity chain was recomputed from `p0_final_test_details.jsonl` (548 rows):

```
sum(channel_stats.n_routed) = 137+83+73 = 293
|route != 'none'|           = 293      |route_prob >= 0.4| = 293   (identical sets)
293 = 200 routed source errors + 93 Router-exposed baseline-correct
baseline-correct (E1)       = 264
SOURCE_RETAINED 47 · GOLD_ARRIVAL 107 · OTHER_WRONG 46 · exits 153
target-hit  107/153 = 0.6993      collateral 52/93 = 0.5591
```

Every frozen quantity reproduces to the digit. `cascade_route` fires **at most one
channel** per row, so summing `n_routed` double-counts nothing. Points 1, 2, 4
and 5 of the review are **confirmed**: the exposure identity is legitimate, the
bound `broke / min(R, 264)` is valid because `broke ⊆ exposed_correct ⊆ fired`,
no validation cell entered the table, and the configuration differences between
seeds were correctly represented.

## What needs narrowing

**The frozen 5% preservation gate is adjudicated on E1, not E2.** The previous
pass reported margins against the **E2 lower bound** and concluded "at least a
factor of 1.6". On the estimand the gate actually uses:

| seed | thr | Broke | **E1 = Broke/264** | E1 margin | E2 lower bound | E2 margin |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.4 | 52 | **19.7%** | 3.94x | 19.7% | 3.94x |
| 123 | 0.4 | 51 | 19.3% | 3.86x | 19.3% | 3.86x |
| 456 | 0.6 | 41 | 15.5% | 3.10x | 15.8% | 3.17x |
| **789** | **0.9** | **16** | **6.1%** | **1.22x** | 7.9% | 1.58x |
| 2024 | 0.6 | 40 | 15.2% | 3.04x | 16.0% | 3.20x |

**Corrected wording:** *every artifact-backed Phi TEST realization exceeds the 5%
preservation budget on both estimands, but the margin at the highest-threshold
configuration (seed 789, τ=0.9) is thin — 6.1% against 5% on the estimand the
gate uses, a factor of 1.22, not 1.6.* The qualitative conclusion **survives**;
the "1.6x" quantifier does not and must be withdrawn.

`PHI_PRESERVATION_FAILURE_ROBUST` stands, with that margin stated.

---

# PART 0b — MetaTool review

Points 1, 3, 4, 5, 6 **confirmed**. The channels are disjoint by construction
(opposite gold *and* opposite pred), the two corrections move tool use in
genuinely opposite directions, the global-suppressor explanation is refuted
within MetaTool's binary setting, no stronger claim is licensed, and `Steerable`
is the strongest defensible rung.

Two narrowings:

**(i) Exact additivity is largely mechanical, not independent evidence.** With
`partition_overlap = 0`, total Fixed and total Broke are sums over disjoint
eligibility sets, so `Net_D = Net_B + Net_C` follows almost by construction. The
archived bookkeeping confirms interaction exists at the *attribution* level —
"each channel's Broke reappears as the other channel's residual (seed 42:
nt_tc 13→5, where 5 = 2 own + 3 inherited)" — while totals stay additive.
Additivity should be reported as a **consistency check**, not as evidence that
the channels do not interact.

**(ii) The five seeds do not share one population.** Archived bookkeeping:
*"the regenerated seed-42 split does not byte-match the archived split files
(524/1040 rows agree); the archived files were used for the pilot
(authoritative), regenerated splits for 123/456/789/2024."* Seed 42 and the other
four seeds therefore agree on only **~50% of rows**. The seed-42 specificity
controls (reverse, matched-random) sit on the archived split; the multiseed Nets
sit on regenerated splits.

**Corrected wording:** *both directions improved in 5/5 runs spanning two
partially overlapping splits (~524/1040 rows shared), with direction-specificity
established once, on the archived split. This is robust behavioural evidence
across seeds and splits, not five replications on one fixed population.*

---

# PART 2/3 — availability

| model | source | dose grid | exposure | exits | destination | collateral denom | row-level |
|---|---|---|---|---|---|---|---|
| Phi-3.5 | `p0_val_threshold_sweep` | τ 0.4–0.9 @ α2 | **yes** (`n_routed`) | no | no | E1 only | no |
| Phi-3.5 | `p0_val_alpha_sweep` | α 2–12 @ τ0.4 | **yes, constant 272** | no | no | E1 only | no |
| Phi-3.5 | `p0_joint_grid_val` | 16 cells | no | no | no | no | no |
| **Qwen3-4B** | `DEV_DOSE_TABLE.csv` | q 0–2 | **yes** | **yes** | **yes** | **E1 + E2** | no |
| **Qwen3-8B** | `QWEN3_STAGE2A_V2_DEV_DOSE_TABLE.csv` | q 0–2 | **yes** | **yes** | **yes** | **E1 + E2** | no |

The confounds are separable here: **A injection magnitude**, **B exposure**,
**C exit rate**, **D destination quality**, **E preservation** are all distinct
recorded fields in the modern tables.

---

# PART 4 — Phi validation sweeps

| τ (α=2) | R fired | Fixed | Broke | Net | F/B | **Broke/R** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 272 | 96 | 36 | +60 | 2.67 | 13.2% |
| 0.5 | 257 | 90 | 33 | +57 | 2.73 | 12.8% |
| 0.6 | 240 | 85 | 31 | +54 | 2.74 | 12.9% |
| 0.7 | 229 | 80 | 29 | +51 | 2.76 | 12.7% |
| 0.8 | 209 | 74 | 25 | +49 | 2.96 | 12.0% |
| 0.9 | 185 | 68 | 19 | +49 | 3.58 | 10.3% |

| α (τ=0.4) | R fired | Fixed | Broke | Net | F/B |
|---:|---:|---:|---:|---:|---:|
| 2 | **272** | 96 | 36 | +60 | 2.67 |
| 4 | **272** | 108 | 35 | +73 | 3.09 |
| 6 | **272** | 112 | 40 | +72 | 2.80 |
| 8 | **272** | 116 | 40 | +76 | 2.90 |
| 10 | **272** | 117 | 40 | +77 | 2.92 |
| 12 | **272** | 116 | 41 | +75 | 2.83 |

**The alpha sweep is an exposure-matched dose experiment by construction** —
R is exactly 272 at every dose, because α scales injection magnitude while τ
alone determines routing. Dose and exposure are orthogonal in this design.

- **Q4.1** — No, not symmetrically. At fixed exposure, α 2→10 raises Fixed by 21
  (+22%) and Broke by only 4 (+11%). Dose buys correction faster than damage.
- **Q4.2** — **Yes.** Raising τ improves preservation almost entirely by removing
  exposure: `Broke/R` is nearly flat (13.2% → 10.3%) while Broke falls 36 → 19.
- **Q4.3** — **Yes.** τ=0.8 and τ=0.9 both give Net +49, with F/B 2.96 vs 3.58 and
  exposure 209 vs 185. Net is not a sufficient statistic for the trade-off.
- **Q4.4** — **Substantially yes, on validation.** Per-exposed-row damage rate is
  near-constant, so exposure explains most absolute Broke variation. It does not
  fully carry over to TEST: seed 789 shows `16/202 = 7.9%` against seed 42's
  `52/293 = 17.7%`, a 2.2x gap in per-exposed rate that exposure alone cannot
  explain (α and seed also differ).

---

# PART 5 — exit-rate matching

```
EXIT_RATE_MATCHING_IDENTIFIABLE
```

Both modern DEV tables record `source_exits`, `gold_arrivals`, `wrong_to_wrong`
and `target_hit` per dose, on two **shared channels**. Matching rule taken from
the brief (exact count, else nearest neighbour on absolute exit count), fixed
before comparing destinations.

### `cannot_answer__to__direct`

| 4B q | exits | tgt_hit | 8B q | exits | tgt_hit | \|Δexits\| | **Δ tgt_hit** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 16 | 0.688 | 0.125 | 18 | 0.500 | 2 | **−0.188** |
| 0.25 | 34 | 0.794 | 0.25 | 39 | 0.487 | 5 | **−0.307** |
| 0.5 | 63 | 0.762 | 0.5 | 56 | 0.804 | 7 | **+0.042** |
| 1.0 | 76 | 0.737 | 1.0 | 91 | 0.747 | 15 | +0.010 |
| 2.0 | 93 | 0.645 | 1.0 | 91 | 0.747 | 2 | **+0.102** |

### `cannot_answer__to__tool_call`

| 4B q | exits | tgt_hit | 8B q | exits | tgt_hit | \|Δexits\| | **Δ tgt_hit** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 59 | 0.525 | 0.25 | 46 | 0.565 | 13 | +0.040 |
| 0.25 | 91 | 0.648 | 1.0 | 88 | 0.795 | 3 | **+0.147** |
| 0.5 | 103 | 0.738 | 1.0 | 88 | 0.795 | 15 | +0.058 |
| 1.0 | 150 | 0.827 | 2.0 | 138 | 0.761 | 12 | **−0.066** |
| 2.0 | 124 | 0.911 | 2.0 | 138 | 0.761 | 14 | **−0.150** |

**The sign of the cross-model destination difference flips with the matching
level, on both shared channels.** Within a model, `target_hit` moves over a huge
range with dose alone (4B ca→tc: 0.525 → 0.911; 8B ca→direct: 0.487 → 0.906) and
is **not monotone** (4B ca→direct peaks at q=0.25 and declines to 0.645).

**Answer:** when exposure is approximately matched, previously observed
cross-model destination differences **do not survive** — they are not stable in
magnitude or in sign. Any "model X is more correctable than model Y" reading is
confounded by dose.

---

# PART 6 — cross-model claim check

| claim | verdict |
|---|---|
| "model family differs in correctability / more correctable" | **CONFOUNDED** — sign flips under exit matching |
| "more off-axis / easier to fix" as an intrinsic property | **CONFOUNDED** — same reason |
| "margin predicts correction / explains rung placement" | **NOT_IDENTIFIABLE** — already prohibited; E4 confirms with data |
| rung placement as an **outcome of the frozen protocol** | **ROBUST_TO_E4** — see below |
| C2-1 correctable subset exists | **ROBUST_TO_E4** — within-model, single dose |
| C3-1 / C3-2 structural controls | **ROBUST_TO_E4** — controls share the real arm's dose and exposure |
| C4-1 preservation turning point | **ROBUST_TO_E4**, with the Part 0a margin narrowing |
| C5-2 exchange rate | **ROBUST_TO_E4** — analytic, not empirical |

### Why rung placement survives

`DOSE_DECLARATION.json` records `conditioned_on_dev_or_evaluation: false`, a
frozen grid `q ∈ {0, .125, .25, .5, 1, 2}` that
`may_not_be_extended_after_observing_dev`, scale-normalised injection
`h' = h + q·s_c·d` (so a given q is a **matched activation budget** across models),
an identical 9-condition admissibility gate, and selection of the **smallest
admissible dose** (`selected_q = 0.25` = min of admissible, both 4B channels).

Dose was therefore never shopped. And because `target_hit` generally *rises* with
dose while the rule takes the *smallest* admissible one, the protocol
systematically selects a dose whose destination quality is **below** what the
model could achieve. **Published rungs are conservative lower bounds**, which is
the safe direction.

### New finding — the gate is permissive relative to E2

`clean_collateral_rate` is computed on `baseline_correct_n` (E1, n=455), while
`router_fired_baseline_correct_exposure` (E2) is 106. For 4B `ca→tc` at q=0.5:

```
E1: 6/455 = 1.3%  -> passes the 5% gate
E2: 6/106 = 5.7%  -> would FAIL the same gate
```

The frozen gate uses the more permissive estimand. This is the **same E1/E2
distinction** that Part 0a surfaced on Phi, appearing independently in the modern
protocol. It is a disclosed limitation, not an error — but it means preservation
was certified on the weaker denominator throughout.

Also noted: for `cannot_answer__to__direct`, `router_fired_baseline_correct_exposure = 0`
at **every** dose. Its 0.0% collateral is **vacuous** — zero exposure, zero
denominator — and must never be quoted as evidence of preservation.

---

# PART 7 — verdict

```
E4_DOSE_CONFOUND_PARTIALLY_REDUCED
```

**Reduced.** Dose and exposure are separately identified: Phi's alpha sweep holds
exposure exactly constant, its threshold sweep shows a near-flat per-exposed
damage rate, and the modern dose selection was frozen, scale-matched,
non-outcome-conditioned and conservative. Licensing outcomes are not dose
artifacts.

**Remains.** Cross-model destination-quality comparisons are confounded — the
sign of the difference flips with the matching level, and `target_hit` varies
more within a model across dose than between models at matched exits.

---

# PART 8 — paper consequence

**What must be narrowed:** every cross-model statement of relative correctability
becomes a statement about *a model at its protocol-selected dose*, never about
intrinsic model properties.

**Strongest surviving comparison:**

> Under a frozen, scale-matched dose grid and a common admissibility gate that
> selects the smallest admissible dose, the seven models occupy five distinct
> evidence rungs. Because destination quality varies substantially with dose
> within every channel we measured, these placements characterise
> model-at-protocol-dose, not intrinsic model correctability, and — since the
> rule selects the smallest admissible dose while destination quality generally
> rises with dose — they are conservative.

**Thesis unaffected.** `Correct → Verify → License` does not depend on any
cross-model mechanism comparison. E4 constrains the interpretation layer only.

---

# PART 9 — claim matrix updates

| claim | previous | new evidence | final wording | forbidden |
|---|---|---|---|---|
| **C4-1** | 52/93 = 55.9%, "≥1.6x budget" | E1 margins 1.22–3.94x | preservation failure robust on both estimands; **thinnest margin 6.1% vs 5% (1.22x) at τ=0.9** | "at least 1.6x"; 52/93 as stable constant |
| **C5-1** | seven models across five rungs | dose-dependence of target_hit; frozen conservative dose rule | rungs as **model-at-protocol-dose**, conservative lower bounds | any intrinsic-correctability reading |
| **C6-1** (MetaTool) | both directions, 5/5, additive | split heterogeneity 524/1040 | 5/5 across **two partially overlapping splits**; specificity once, archived split | "five replications on one population"; additivity as independent evidence |
| **Q5 mechanism** | dose confound UNTESTABLE | now **tested** and confirmed confounded | dose confound **demonstrated**, not merely untestable | any margin-explains-rung claim |

New standing constraints:
- **S-16** — `clean_collateral_rate` is the **E1** estimand; never present it as exposure-conditional.
- **S-17** — a 0.0% collateral with `router_fired_baseline_correct_exposure = 0` is **vacuous**; never quote it.

---

# PART 10 — was E4 worth doing?

**Scientific value: it mainly clarified a limitation, and found two real defects.**
It did not strengthen a headline claim. It converted "dose confound UNTESTABLE"
into a *tested and demonstrated* confound — which is a genuine upgrade in
rigour — and it surfaced the E1/E2 gate asymmetry and the vacuous zero-exposure
collateral, neither of which was previously flagged.

**Paper placement: appendix, plus two sentences in limitations.** The exit-matched
table belongs in an appendix; the narrowing of rung interpretation belongs in the
main text where the rung distribution is introduced.

**Effect on unified SAKIKO: none on the thesis.** `Correct → Verify → License` is
untouched. E4 refines the mechanism/interpretation layer and makes the
cross-model reading strictly weaker and more defensible.

Not inflating this: E4's honest value is that it removes a claim the paper could
have been attacked on, and it slightly weakens one quantifier in C4-1.

---

# EXACT NEXT ACTION

```
MECHANISM_SYNTHESIS
```

E4 cleanly bounds the confound: separable within-model, confounded across models,
with the direction of bias known and conservative. Not executed.
