# Phase 4 — Step 10 · Generation-Readout Human Audit

**Purpose.** Manually adjudicate a stratified sample of the *locked* full-baseline generations against
the parser's assigned labels, to measure (a) **rule-compliance** — does the parser's output follow the
locked rules for the given text — and (b) **parser mislabel rate** — did the parser assign a *wrong
native label* that misreads the model's actual output. These feed R0.5
(`rule-compliance ≥ 95% AND parser mislabel ≤ 5%`).

**Inputs (all from the locked run — no re-generation, no rule changes):**
- Predictions: `.cache/acebench_phase4/generation_details_full.jsonl` (800 rows, parser v1.0).
- Audit sample: `.cache/acebench_phase4/audit_sample.jsonl` (102 rows), built deterministically by
  `acebench_readout_audit.py audit-build` (seed 7), stratified over: every populated confusion cell
  (≤5/cell), UNKNOWN (≤25), conflict, ambiguous-fallback, and bracketed-unclassified strata.
  It stays **gitignored**: it carries raw generations, which embed ACEBench dataset text, and the
  locked artifact policy (`PHASE4_GENERATION_PROTOCOL.md`, *Compute & artefact policy*) keeps raw
  generations out of the repo. It is exactly regenerable from the command in §4.
- Per-row judgments: `audit/audit_judgments.jsonl` (**committable** — metadata only, no raw text);
  roll-up: `audit/audit_summary.json`.

The audit is **gold-blind for the parser** (the parser never saw gold); the adjudication *does* look at
gold + raw text to decide whether a label is semantically right, which is the point of an audit.

---

## 1. Headline result

| Metric | Value | R0.5 threshold | Verdict |
|---|---|---|---|
| Rows audited | 102 | ≥ 100 | ✓ |
| **Rule-compliance** (label follows locked rules) | **102/102 = 100%** | ≥ 95% | **PASS** |
| **Parser mislabel — strict** | **1/102 = 0.98%** | ≤ 5% | **PASS** |
| Parser mislabel — conservative (incl. both flagged conflicts) | 3/102 = 2.94% | ≤ 5% | PASS |

Judgment breakdown (102): `faithful_native_read` 48 · `abstain` (UNKNOWN) 45 ·
`fallback_correct` 6 · `conflict_flagged` 2 · `fallback_misfire` **1**.

---

## 2. What the audit actually found

**2.1 Native-label predictions are faithful reads of real model modes (48/48 clean).**
Every row where the parser emitted a native mode via an *official / canonical* rule
(`bracket.official.*`, `bracket.call_list`) matched what the model literally produced. This holds even
when `gold ≠ pred`: e.g. rows where gold is `ask_user`/`flag_param_error`/`cannot_comply` but the model
emitted a real `[func(...)]` call (→ `tool_call`), or emitted the official *"Missing necessary
parameters …"* template (→ `ask_user`). These are **genuine model decision errors, correctly read by
the parser** — exactly what a valid readout must surface. Confirmed run-wide: of all **137**
native-label confusion-cell errors (gold≠pred, pred≠UNKNOWN) in the 800-row run, **136 (99.3%)** were
produced by an official/canonical rule; only **1** came from a fallback tier.

**2.2 UNKNOWN is a disciplined abstention, not a failure of reading (45/45).**
All 45 UNKNOWN rows in the sample are honest abstentions on malformed/truncated output. Run-wide the 58
UNKNOWNs decompose as: **57/58 outputs begin with `[`** (a call-list attempt) — the model tried to call
a tool but produced a non-official variant the conservative parser refuses to coerce:
- **17/58** in *quoted-call* form `["func(...)"]` (the call is a string literal, not the official
  bracketed call) — finished on EOS;
- **31/58** truncated at the 256-token cap (`finish=length`) mid call-list → unbalanced brackets;
- the remainder malformed nesting (`[[`, `[Wrapper(Call(...)]`).

These are **model formatting failures**, not parser errors. The parser correctly declines to invent a
label. (1/58 UNKNOWN is `no_rule` on non-bracketed prose.)

**2.3 The single parser mislabel — `acebench_decision_0499` (documented locked limitation).**
gold `tool_call` → pred `cannot_comply`, rule `bracket.fallback.capability_refusal_cue`, `ambiguous=1`.
The model emitted a real two-call list, but it **ran to the 256-token cap** (`finish=length`), so the
call-list block never closed (unbalanced) and the canonical call parse failed. The parser then dropped
to its fallback cue tier, which matched the Chinese phrase **"…有时不能满足请求…"** ("… sometimes cannot
satisfy the request …") appearing **inside a `user_feedback` data string**, and mislabeled the row
`cannot_comply`. This is the **only** `bracket.fallback.*` firing in the entire 800-row run (0.125%).
It is a real parser error and is **counted as a mislabel**. It is a *known, pre-disclosed* limitation
(PARSER_VALIDATION_REPORT §6: fallback cues are conservative and always flagged `ambiguous`); no
post-lock rule change is permitted, so it is reported rather than patched.

**2.4 Conflicts are transparent, not silent (2, both flagged).**
Two rows carried two official templates in one block and were resolved by the *locked* priority with
`conflict=1, ambiguous=1`:
- `0431`: value-error + missing-param → `flag_param_error` (value-error outranks missing) — and the
  model did state a value error, so the label is defensible.
- `0755`: missing-param + capability-refusal → `ask_user` (gold `cannot_comply`) — the model emitted
  *both*; the locked priority picks `ask_user`. Borderline. Counted in the **conservative** mislabel
  figure (still 2.94% ≤ 5%).

Both are surfaced by the `conflict` flag; neither is a silent misread.

---

## 3. Why this matters for R0 (readout validity, not model quality)

The audit separates two things the R0 gate must not conflate:
- **Parser fidelity** — does a predicted label reflect the text? → **≥ 99% clean** (1 misfire / 800
  native reads run-wide; 0.98% of the audit sample). The readout is a trustworthy instrument.
- **Model correctness** — does the model pick the gold mode? → *not* the parser's job; the confusion
  cells (e.g. `flag_param_error→tool_call` 52, `ask_user→tool_call` 40, `cannot_comply→ask_user` 22)
  are **real behavioral transitions** the model produced, faithfully recorded. These are the candidate
  error channels Step 12 examines.

The prior candidate-scoring readout failed R0 by *collapsing* (74.5% one class, acc 0.196). This
generation readout does not collapse (max pred share 0.675, all four native modes predicted, UNKNOWN
7.25%) **and** its errors are model-attributable rather than parser-attributable — the two properties a
valid multi-class readout needs.

---

## 4. Reproducibility

```bash
# rebuild the exact audit sample (seed 7, deterministic — byte-identical every run):
python scripts/acebench_readout_audit.py audit-build \
  .cache/acebench_phase4/generation_details_full.jsonl \
  .cache/acebench_phase4/audit_sample.jsonl
# per-row judgments + summary are regenerated by the audit block in this step (audit_summary.json).
```

**Step-10 verdict: PASS** — rule-compliance 100% (≥95%), parser mislabel 0.98% strict / 2.94%
conservative (≤5%). Feeds R0.5.
