# Phase 4 — Step 11 · R0 Baseline-Validity Decision

**Scope.** This decides whether the *generation-based* ACEBench readout can support a valid
multi-class evaluation of SAKIKO-CA. It evaluates the readout, **not** the model's competence, and it
does **not** run any intervention.

**Inputs (locked before results were seen).** Criteria R0.1–R0.10 and the verdict rule are fixed in
`PHASE4_GENERATION_PROTOCOL.md §17`, hashed in `READOUT_LOCK_MANIFEST.json` (protocol v1.0, parser
v1.0) **before** the full run. Evidence: `baseline/acebench_generation_baseline_summary.json` (800
rows), `baseline/replay_determinism_check.json`, `GENERATION_READOUT_HUMAN_AUDIT.md`,
`channel_discovery/acebench_channel_discovery.json`.

> **No threshold was moved after seeing results.** One criterion fails; it is reported as a failure.

---

## 1. Criterion-by-criterion

| # | Criterion (locked) | Observed | Verdict |
|---|---|---|---|
| **R0.1** | acc ≥ 0.675 **OR** (bal-acc ≥ 0.45 AND macro-F1 ≥ 0.40) | acc **0.7562**; bal-acc 0.6205; macro-F1 0.6868 | **PASS** (both routes) |
| **R0.2** | macro-F1 ≥ 0.40 | **0.6868** | **PASS** |
| **R0.3** | max pred share ≤ 0.70 AND ≥3 of 4 modes ≥5% | max **0.675**; **4/4** modes ≥5% (67.5 / 9.6 / 9.6 / 6.0%) | **PASS** |
| **R0.4** | UNKNOWN + unparseable ≤ 10% | **7.25%** | **PASS** |
| **R0.5** | rule-compliance ≥95%; parser mislabel ≤5% | **100%**; **0.98%** strict (2.94% conservative) | **PASS** |
| **R0.6** | 50-row replay ≥98% identical | **100%** (50/50) | **PASS** |
| **R0.7** | non-empty ≥99%; EOS-before-256 ≥95% | **100%**; **95.87%** | **PASS** (narrow) |
| **R0.8** | no label leakage | parser takes text only; gold joined post-hoc; rules metadata-blind (静态 audit) | **PASS** |
| **R0.9** | **≥2 gold→pred transitions with train ≥30 AND val ≥5 AND test ≥5** | **0 transitions qualify** | **FAIL** |
| **R0.10** | ≥3 of 4 gold classes with recall ≥0.10 | **4/4** (0.892 / 0.420 / 0.420 / 0.750) | **PASS** |

### Verdict rule applied verbatim (§17)
- `PASS` = R0.1–R0.10 all hold → **not met** (R0.9 fails).
- `CONDITIONAL PASS` = R0.2–R0.8 hold **AND** bal-acc ≥ 0.35, but R0.1, R0.9, **or** R0.10 fails.
  → R0.2–R0.8 all hold ✓; bal-acc 0.6205 ≥ 0.35 ✓; R0.9 (a member of the permitted set) fails ✓.

# ⇒ VERDICT: **CONDITIONAL PASS**

Consequence per the locked rule: **descriptive error analysis only.** Intervention eligibility
additionally requires the pre-registered paraphrase check (second, semantically equivalent
non-official-wording system prompt; 100-row stratified subset; prediction agreement ≥ 0.80) —
reported in `paraphrase_check/` and adjudicated in `PHASE4_DECISION_AND_NEXT_STEP.md`.

---

## 2. The readout itself is sound — the prior failure is fixed

The earlier **candidate-scoring** readout failed R0 by collapsing. The generation readout does not:

| | candidate-scoring (archived, failed) | **generation (this study)** |
|---|---|---|
| accuracy | 0.196 | **0.7562** |
| macro-F1 | 0.26 | **0.6868** |
| largest predicted class | 74.5% (collapse) | **67.5%** |
| modes ever predicted | effectively 1 | **4 of 4** |
| UNKNOWN | n/a | 7.25% |

Crucially the errors are **model-attributable, not parser-attributable**: of the 137 native-label
confusion-cell errors, **136 (99.3%)** were produced by an official/canonical parse rule — i.e. the
model really did emit that other mode's official output. Only **1/800** was a parser misfire
(`0499`, documented in the audit). The instrument reads the model faithfully.

**So R0.9 is not a readout defect.** It is a statement about *dataset support*, which is why it is
the only criterion that fails.

---

## 3. Why R0.9 fails — a structural property of ACEBench, confirmed pre-hoc

All 10 gold→pred transitions, per split (UNKNOWN predictions excluded):

| transition | all | train | val | test | ref pool (train-correct) | scope | bootstrap |
|---|---|---|---|---|---|---|---|
| `flag_param_error→tool_call` | 52 | 38 ✓ | 10 ✓ | **4** ✗ | **27** ✗ | ✗ | **0.31** |
| `ask_user→tool_call` | 40 | **29** ✗ | 5 ✓ | 6 ✓ | **28** ✗ | ✗ | **0.215** |
| `cannot_comply→ask_user` | 22 | 14 ✗ | 4 ✗ | 4 ✗ | 53 ✓ | ✗ | 0.0 |
| `tool_call→ask_user` | 7 | 7 | 0 | 0 | — | ✗ | 0.0 |
| `flag_param_error→ask_user` | 6 | 5 | 0 | 1 | — | ✗ | 0.0 |
| `tool_call→flag_param_error` | 5 | 3 | 1 | 1 | — | ✗ | 0.0 |
| `cannot_comply→tool_call` | 2 | 2 | 0 | 0 | — | ✗ | 0.0 |
| remaining 3 | 1 each | — | — | — | — | ✗ | 0.0 |

Required: train ≥30, val ≥5, test ≥5, reference pool ≥30. **Qualifying: 0** (need ≥2).

**This is not a near-miss of luck; the support is genuinely unstable.** The two leading channels each
fail on *two* counts, and 200-resample bootstrap stability under the scope filter is **0.31** and
**0.215** — far below the 0.80 the archived pipeline requires. Resampling clears the filter less than
a third of the time.

**The arithmetic cause (anticipated in the locked protocol's own R0.9 note).** Each special mode has
exactly **100** examples → a 70/15/15 split gives **train 70 / val 15 / test 15**. To qualify, one
single error cell must hold ≥30 of 70 train (43%), ≥5 of 15 val (33%), ≥5 of 15 test (33%), *while*
≥30 of the same 70 train rows are correct for the reference pool — i.e. ≥60 of 70 train rows (**≥86%**)
must sit in just two cells. `flag_param_error` gets remarkably close (38 error + 27 correct = 65/70 =
93% concentrated) **and still fails** on test (4) and reference pool (27). ACEBench's 100-example
special classes make the archived intervention-scope thresholds **arithmetically near-unreachable**,
independent of model behaviour. The protocol said this in advance:

> *"special classes cap at 100 examples, so W2C's default 50/15 tier is arithmetically out of reach
> for special-gold transitions"* — `PHASE4_GENERATION_PROTOCOL.md §17, R0.9`

For comparison, W2C's `ca_rfi` channel (Phase 3, intervention-validated) had **219** examples
(151/36/32) from a 3652-row pool — an order of magnitude more support per channel.

Lenient-tier (30/10) discovery status, reported alongside as the protocol requires: every transition
is `weak` or `insufficient`; **none** reaches `stable` at any tier.

---

## 4. What this does and does not license

**Licensed now:**
- Descriptive error analysis of ACEBench decision behaviour under a validated, non-collapsed readout.
- The finding that Qwen2.5-7B's dominant ACEBench failure is **over-calling**: it emits a real tool
  call when it should have flagged a bad value (52) or asked for a missing parameter (40) — 92 of 137
  native errors (67%) point into `tool_call`.
- Reporting `flag_param_error→tool_call` as a *candidate* new channel family (no W2C/MetaTool
  analogue) — as a hypothesis, not an intervention target.

**Not licensed:**
- Any ACEBench SAKIKO-CA intervention claim. §20 feasibility (≥2 scope-passing channels with
  bootstrap ≥0.80, ≥1 touching a special mode) is **not met**: `n_scope_pass=0`, `n_eligible=0`.
- Any claim that ACEBench "refutes" or "confirms" SAKIKO-CA transfer. **Insufficient channel support
  is not evidence of absent effect** — it means the experiment cannot be run at the archived
  standard on this dataset as split.
- Re-splitting or threshold-relaxing to manufacture eligibility. Both were considered and **rejected**:
  the thresholds are locked and are the same ones that made the Phase-3 W2C result trustworthy;
  weakening them for ACEBench would break comparability with the archived pipeline and inflate a
  channel whose bootstrap stability is already 0.31.

---

## 5. Honest limitations of this decision

- **R0.7 passes narrowly** (EOS 95.87% vs 95% floor). 31 of 58 UNKNOWNs are 256-token truncations of
  long multi-call lists; a larger cap would likely convert some UNKNOWNs into `tool_call`, slightly
  raising accuracy and *possibly* shifting transition counts. The cap was locked pre-run and was not
  changed. This is a known, disclosed sensitivity.
- **UNKNOWN exclusion.** Transitions exclude UNKNOWN predictions (locked policy: UNKNOWN carries no
  target mode). 41 UNKNOWNs have gold `tool_call`; had they parsed, they would mostly *not* create
  new special-gold transitions, so this exclusion does not plausibly rescue R0.9.
- **Single model, single decoding.** Greedy, Qwen2.5-7B only. A different model could concentrate
  errors differently; that is a separate study, not a rescue of this one.
- The **parser mislabel `0499`** and **2 flagged conflicts** are counted honestly and do not change
  any criterion's verdict.

---

**Step-11 verdict: `CONDITIONAL PASS` — exactly one, assigned under the locked rule.**
The readout is valid and non-collapsed (R0.1–R0.8, R0.10 all pass); ACEBench's per-channel support is
structurally insufficient for the archived intervention-scope standard (R0.9 fails, 0 of ≥2).
