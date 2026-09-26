# Phase 4 — Pre-registered Paraphrase Robustness Check (CONDITIONAL-PASS eligibility gate)

**Why this ran.** The R0 verdict is **CONDITIONAL PASS** (`R0_BASELINE_VALIDITY_DECISION.md`). The
locked rule (`PHASE4_GENERATION_PROTOCOL.md §17`) states that under CONDITIONAL PASS, *intervention
eligibility additionally requires*:

> *"a second, semantically equivalent (non-official-wording) system prompt on a 100-row stratified
> subset with prediction agreement ≥ 0.80."*

This is a **black-box prompt probe**, not an intervention: no activations, routers, directions,
steering, or SAKIKO. Script: `scripts/acebench_paraphrase_check.py`. Data:
`paraphrase_agreement.json`; raw generations in gitignored
`.cache/acebench_phase4/generation_details_paraphrase.jsonl`.

**Design.** Model, decoding (greedy, 256 tokens, bf16), parser v1.0, user prompt, and the three
official **output-format template strings** are all held **identical** to the locked baseline. The only
manipulated variable is the *prose of the SPECIAL-data system prompt*, reworded into different English
and Chinese while preserving the task semantics and the answer schema. Subset: 100 rows, balanced 25
per gold mode (13 EN / 12 ZH each), lowest sample_id — deterministic.

---

## 1. Result — **FAIL**, on every metric

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| **Agreement (pre-registered — this is the gate)** | **0.5600** (56/100) | ≥ 0.80 | **FAIL** |
| Agreement, format-insensitive (both sides native, n=68) | 0.7941 | ≥ 0.80 | fail (marginal) |
| Agreement, post-hoc latent-call sensitivity | 0.6500 | ≥ 0.80 | fail |

**The verdict does not depend on which metric is chosen** — all three fall below 0.80. Only the
first is the pre-registered gate; the other two are disclosed sensitivity analyses (§3).

Per-gold-mode agreement (pre-registered metric):

| gold mode | agreement | latent-call variant |
|---|---|---|
| `tool_call` | **0.84** | 0.92 |
| `flag_param_error` | 0.52 | 0.64 |
| `cannot_comply` | 0.52 | 0.52 |
| `ask_user` | **0.36** | 0.52 |

---

## 2. What changed — the model becomes markedly more call-happy

Prediction distribution over the same 100 rows:

| | `tool_call` | `ask_user` | `flag_param_error` | `cannot_comply` | UNKNOWN |
|---|---|---|---|---|---|
| baseline (official prompt) | 43 | 20 | 16 | 19 | 2 |
| paraphrase (locked parse) | 36 | **4** | 15 | 13 | **32** |
| paraphrase (latent-call view) | **65** | 4 | 15 | 13 | 3 |

Under the paraphrase, **29 of 30** new UNKNOWNs are *quoted-call* outputs — `["ApiName(args)"]` —
i.e. the model **attempted a tool call** but wrapped it as a string literal, which the conservative
locked parser refuses to coerce. Reading those as latent calls, the `tool_call` share rises
**43% → 65% (+22 points)**, while `ask_user` collapses **20 → 4**.

This is **not merely cosmetic reformatting of the same decision.** Worked examples:

- `0254` (gold `flag_param_error`): baseline **flagged** the malformed email `user!example.com`;
  under the paraphrase the model **called** `AuthenticateAndRecycle(email='user!example.com', …)` —
  passing the bad value through instead of flagging it.
- `0305` (gold `ask_user`): baseline **asked** for the missing parameters; under the paraphrase the
  model **called** `device_rollback_software_update(device_id='my_device_id', …)` — inventing values.
- `0352` (gold `cannot_comply`): baseline emitted the official refusal template; under the paraphrase
  it emitted the prose *"The request is outside what your current APIs can do."*

Plus **14 native→native mode flips** with no format component at all
(`cannot_comply→flag_param_error` ×5, `ask_user→flag_param_error` ×4, `ask_user→tool_call` ×2, …).

**Pattern:** the *default* behaviour (calling) is stable (0.84–0.92); the **restraint** behaviours —
ask, flag, refuse — are the wording-fragile ones. ZH drifts more than EN (29 vs 15 disagreements).

---

## 3. Honest limitations of this check (read before citing it)

**3.1 The paraphrase carries a weaker format directive than the official prompt.** The official text
says, for each special case, *"The output format should be: [...]"*. The paraphrase presents the same
template after a colon, without that explicit lead-in. This plausibly **inflates** the format drift
(the quoted-call and prose-echo outputs) and therefore inflates the gap between 0.56 and 0.79. Part of
the measured failure is attributable to the construction of the paraphrase, not solely to the model.

**3.2 Therefore the strong claim is *not* licensed.** This check does **not** cleanly establish
"the model's ACEBench decision is wording-fragile." It establishes the weaker, sufficient thing the
gate actually asked for: **robustness ≥ 0.80 was not demonstrated.** Eligibility required a positive
demonstration of stability; the burden was not met — on any of the three metrics.

**3.3 The check was run once and is not re-rolled.** A "better" paraphrase would very likely score
higher, and re-running until the gate passes is precisely the practice this protocol exists to
prevent. The pre-registered result stands as measured. A future study wanting a clean decomposition
should **pre-register a format-matched paraphrase** (rewording only the decision rationale while
holding the format directive's explicitness constant) and, ideally, several paraphrases — that is a
legitimate follow-up design, declared in advance, not a repair of this one.

**3.4 Single paraphrase, single model, greedy decoding.** n=100, one rewording, Qwen2.5-7B only.

---

## 4. Consequence

`R0 = CONDITIONAL PASS` **and** `paraphrase check = FAIL` ⇒ the §17 eligibility condition is **not**
met ⇒ **ACEBench intervention is not eligible**; descriptive error analysis only. This also removes
outcome **B (CONDITIONAL GO)**, which required *"R0 CONDITIONAL PASS with the paraphrase check
passed."* Adjudicated in `PHASE4_DECISION_AND_NEXT_STEP.md`.

Note the compounding with R0.9: the two candidate channels (`flag_param_error→tool_call`,
`ask_user→tool_call`) sit **exactly on the axis this check finds least stable** (restraint → calling).
A channel that both (a) lacks the split support to be fit and tested at the archived standard and
(b) moves under a prompt rewording is not a sound intervention target — steering it would risk
measuring prompt-sensitivity rather than a stable internal mechanism.
