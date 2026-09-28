# Phase 4 — Parser Validation Report

**Parser:** `scripts/acebench_parse_outputs.py`, PARSER_VERSION **1.0** (sha256 in
`READOUT_LOCK_MANIFEST.json`). Validated **before any model inference**, per
`PHASE4_GENERATION_PROTOCOL.md §18`. Verdict at the bottom.

## 1. Unit-test suite — **30/30 PASS**

Coverage (all run via `scripts/acebench_readout_audit.py unittests`):
- canonical **official** outputs for every ACEBench mode (call list incl. parallel calls;
  missing-parameters template; incorrect-value template; capability-limitation template);
- official templates with case/singular variations;
- markdown-fence and `assistant:` role-prefix wrappers;
- **ZH and EN free-text fallbacks** for ask/flag/cannot (non-official formats → correct label with
  `ambiguous=true`);
- bare call without brackets (→ `tool_call`, flagged ambiguous);
- malformed outputs: empty, whitespace, unbalanced bracket/paren, bracketed non-call prose;
- **wrong/missing-argument calls** (`f()`, `f(city=)`) → still `tool_call` (argument validity is
  not a mode change; recorded via extracted calls);
- **mixed-mode conflict**: two official special templates in one block → resolved by the official
  priority (value-error > missing), `conflict=true`;
- call block followed by refusal prose → first-block rule applies (call wins; prose outside block);
- direct answers with no actionable content → `UNKNOWN(no_rule)`;
- clarification without a question mark → stays UNKNOWN (locked conservative rule);
- multiple blocks → first balanced block wins.

## 2. Determinism — **PASS**

Every case parsed 3×: byte-identical results (`deterministic on 3x repeats: True`). The parser is
pure string processing — no randomness, no state.

## 3. Gold-independence — **PASS** (static audit)

- `parse()` signature accepts **only** the raw text — there is no argument through which gold,
  sample id, filename, split, or language could even be passed.
- Rule-function source scan: no `gold` / `sample_id` / `source_file` / `metadata` / `filename` /
  `split` / `lang` token in any rule function.
- CLI wrapper passes only `output` text into `parse()` (sample_id used solely to key the result).
- No regex keys on dataset metadata; no mode inferred from sample ID or filename (rules operate on
  generation text only).

## 4. Explicit UNKNOWN handling — **PASS**

UNKNOWN is a first-class outcome with reasons: `empty_generation`, `bracketed_unclassified`,
`no_rule`, `conflicting_modes` (call+special conflict). Ambiguous fallback parses are labeled but
flagged; nothing is silently forced into a native label.

## 5. Real-schema check on actual dataset rows — **22/22 PASS after one pre-lock fix**

Stratified rows (all 4 modes × both languages) were paired with hand-written expected-behavior
outputs (canonical official formats instantiated with the rows' real function names/values,
written from the input + schemas only) and parsed: 22/22 correct.

**Pre-lock revision (documented per protocol changelog policy):** the real ACEBench schema contains
**digit-leading function names** (`5G_City_Application_Summary`, `5G_Technology_Report_Finder`);
the initial call-identifier regex `[A-Za-z_]\w*` would have mis-parsed legitimate calls to these
functions as UNKNOWN. Fixed to `[A-Za-z0-9_][\w.]*` and the corresponding unit case updated
(+1 new digit-name case). This change happened **during Step-6 validation, before the smoke test,
the lock, and any model inference** — exactly the revision window the protocol allows.

## 6. Known, accepted limitations (locked behavior)

- Free-text (non-official-format) outputs are classified only by conservative bilingual cue rules
  and always flagged `ambiguous` — the Step-10 audit quantifies their error rate; R0.5 gates on it.
- A clarification request without `?`/`？` is UNKNOWN by design (conservatism over recall).
- Argument *values* are not validated against schemas (mode classification only).

**VERDICT: PASS — the parser is approved for the smoke test and, contingent on the smoke gate, the
locked full baseline.**
