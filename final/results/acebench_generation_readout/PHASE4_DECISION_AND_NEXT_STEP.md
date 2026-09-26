# Phase 4 — Step 13 · Decision and Next Step

**Question this phase asked.** *Can ACEBench support a valid multi-class evaluation of SAKIKO-CA?*
Not "does SAKIKO-CA work on ACEBench" — this phase deliberately did **not** run an intervention.

---

# ⇒ OUTCOME: **D — DATASET NO-GO**
### Retire ACEBench from the quantitative (intervention) main line; it may remain a descriptive datapoint.

---

## 1. How the evidence maps to the outcome

| Finding | Result |
|---|---|
| Generation readout validity (R0.1–R0.8, R0.10) | **all PASS** — acc 0.756, macro-F1 0.687, max share 0.675, UNKNOWN 7.25%, replay 100%, parser mislabel 0.98% |
| R0 verdict (locked rule) | **CONDITIONAL PASS** (only R0.9 fails) |
| R0.9 — channel support ≥2 @ train≥30/val≥5/test≥5 | **FAIL — 0 qualifying**; bootstrap stability 0.31 / 0.215 / 0.0 |
| §20 channel-discovery feasibility | **NOT met** — `n_scope_pass=0`, `n_eligible_boot80=0` |
| §17 paraphrase eligibility gate (≥0.80) | **FAIL — 0.56** (also 0.79 format-insensitive, 0.65 latent-call) |

### Why not A, B, or C

- **A (GO)** requires `R0 PASS` **and** §20 feasibility met. Neither holds. ✗
- **B (CONDITIONAL GO)** requires `R0 PASS with §20 marginal` **or** `R0 CONDITIONAL PASS **with the
  paraphrase check passed**`. The paraphrase check **failed** (0.56 vs 0.80), on all three metrics. ✗
- **C (READOUT NO-GO)** requires `R0 FAIL with evidence localizing the failure **to the readout**`.
  The evidence says the **opposite**: the readout is valid and non-collapsed, and **136/137 (99.3%)**
  of native-label errors came from official/canonical parse rules — the instrument reads the model
  faithfully. Assigning C would misattribute a dataset-side limit to the instrument. ✗

### Disclosed deviation: the locked taxonomy has a gap

`§21` never enumerated the cell we landed in — **CONDITIONAL PASS + paraphrase FAIL**. Rather than
silently force a poor fit, this is recorded explicitly. **D** is assigned because:
1. the locked §17 rule makes the consequence unambiguous — *intervention not eligible; descriptive
   analysis only*; and
2. both blockers are **dataset-side**, not readout-side, so D's action (*retire from the quantitative
   main line*) is the one the evidence supports.

**Two honest qualifications on D's wording:** D is phrased as *"official semantics/data quality
prevent objective evaluation (e.g., gold labels incompatible with official definitions)."*
(i) That illustrative mechanism is **not** what was found — the gold labels are fine and the audit
raised no material gold-incompatibility. (ii) Objective *descriptive* evaluation **was** achieved
(acc 0.756 on a validated readout). What ACEBench cannot support is the **intervention study**, not
evaluation as such. D is assigned on its *action and attribution*, not its parenthetical example.

---

## 2. The two dataset-side blockers

**Blocker 1 — structurally insufficient channel support (arithmetic, not luck).**
Each special mode has exactly **100** examples → 70/15/15 gives train 70 / val 15 / test 15. One error
cell would need ≥30 of 70 train (43%), ≥5 of 15 val, ≥5 of 15 test, *while* ≥30 of the same 70 train
rows stay correct for the reference pool — i.e. **≥86% of the class concentrated in two cells**.
`flag_param_error→tool_call` reaches 93% concentration (38 error + 27 correct of 70) **and still
fails** (test 4 < 5; ref pool 27 < 30), with bootstrap scope-stability **0.31**. The protocol
predicted this in advance (§17, R0.9: *"special classes cap at 100 examples, so W2C's default 50/15
tier is arithmetically out of reach"*). For contrast, Phase 3's intervention-validated W2C `ca_rfi`
channel had **219** examples (151/36/32) drawn from 3652 rows.

**Blocker 2 — the candidate channels sit on the least stable axis.**
The paraphrase check shows the *default* behaviour (calling) is stable (0.84–0.92) while the
**restraint** behaviours are not (`ask_user` 0.36, `flag_param_error` 0.52, `cannot_comply` 0.52);
a semantically-equivalent rewording moves the model **+22 points toward calling** (43%→65% latent).
The two candidate channels — `flag_param_error→tool_call` and `ask_user→tool_call` — are *exactly*
restraint→calling. Steering a behaviour that moves under a prompt paraphrase risks measuring
prompt-sensitivity rather than a stable internal mechanism.
*(Caveat, per `PARAPHRASE_ROBUSTNESS_REPORT.md §3`: the paraphrase's format directive is weaker than
the official's, which inflates the raw gap. The licensed claim is only the one the gate required:
**stability ≥0.80 was not demonstrated** — on any metric.)*

Individually each blocker would justify caution; **together** they make an ACEBench SAKIKO-CA pilot
uninterpretable at the archived evidential standard.

---

## 3. What this does NOT say

- **Not** "SAKIKO-CA fails to transfer to ACEBench." **No transfer claim is made in either
  direction.** The experiment was not run, because it cannot be run here at the archived standard.
  *Insufficient channel support is not evidence of absent effect.*
- **Not** "the generation readout failed." It passed 9 of 10 criteria and fixed the prior
  candidate-scoring collapse (acc 0.196→0.756, macro-F1 0.26→0.687, one-class 74.5%→67.5%).
- **Not** "ACEBench is a bad benchmark." It is a poor fit for *this* mechanistic-intervention
  pipeline, which needs many examples per error channel and wording-robust behaviour.
- **Not** licence to re-split, relax thresholds, or re-roll the paraphrase to manufacture
  eligibility. All were considered and **rejected**: the thresholds are the same ones that made the
  Phase-3 W2C result trustworthy.

## 4. What was learned (descriptive, licensed)

- A **generation-based** readout rescues ACEBench from the candidate-scoring collapse — the
  readout-design lesson generalizes: score the model's *emitted behaviour*, not a forced choice.
- Qwen2.5-7B's dominant ACEBench failure is **over-calling**: **92 of 137** native errors (67%) point
  into `tool_call` — it calls when it should flag a bad value (52) or ask for a missing one (40).
- `flag_param_error→tool_call` has **no W2C/MetaTool analogue** — a genuinely *new* candidate channel
  family. It is recorded as a **hypothesis**, not an intervention target.
- The model's restraint behaviours are far more prompt-fragile than its calling behaviour — itself a
  finding worth carrying into benchmark selection.

---

## 5. Recommended next step (single)

**Proceed to the second-model architecture validation** already queued in
`research_synthesis/NEXT_EXPERIMENT_ROADMAP.md` — i.e. test whether the Phase-3-validated W2C channel
result (`ca_rfi`: Net +16, direction-specific, 5/5 multiseed wins) reproduces on a **different model**
using the **same dataset that has adequate channel support**, rather than spending GPU on a dataset
that cannot support the measurement.

**Rationale.** Phase 3 established that an automatically discovered channel can yield a real
target-native intervention *when support is adequate*. The open question is **architecture
generality**, not benchmark breadth. ACEBench cannot answer either question.

**If ACEBench is ever revisited**, the pre-registered requirements are:
1. a **larger special-mode pool** (≥300/mode, or pooled across ACEBench variants) so a channel can
   clear train≥30/val≥5/test≥5 *and* ref≥30 with bootstrap ≥0.80;
2. a **format-matched, multi-paraphrase** robustness study (declared in advance) demonstrating
   agreement ≥0.80 on the restraint modes specifically;
3. a raised generation cap (>256 tokens) to remove the 31 truncation-driven UNKNOWNs.
Absent (1) and (2), no ACEBench intervention should be run.

---

## 6. Cost and honesty ledger

- GPU spend: 800-row baseline (22.4 min) + 50-row replay + 100-row paraphrase ≈ **35 min** — the
  cheapest possible route to a defensible NO-GO. No intervention GPU time was spent on an
  unmeasurable target.
- **One criterion failed and is reported as failed.** No threshold moved after results were seen; the
  paraphrase was run once and not re-rolled; the taxonomy gap is disclosed rather than papered over;
  the one parser mislabel (`0499`) and two flagged conflicts are counted against us.

**Motto carried from Phase 3:** *discovery is not sufficient evidence of intervention utility* —
and here, **measurability is a precondition for both**.
