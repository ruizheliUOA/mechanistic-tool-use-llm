# Phase 4 — Existing Evidence Audit (ACEBench generation-readout validation)

**Date:** 2026-07-14 · branch `exp/sakiko-followup-archive` · HEAD `b9f5aae` (clean tree at start) ·
written BEFORE any protocol lock or model run. Scope: readout validation only — no intervention, no
activation extraction, no routers, no SAKIKO.

## 1. Exact ACEBench source and version already used

- **Source:** `github.com/ACEBench/ACEBench`, files `data_all/data_en/` + `data_all/data_zh/`
  (recorded provenance in the archived converter and split-info). Paper: **arXiv:2501.12851**
  ("ACEBench: Who Wins the Match Point in Tool Learning?").
- The Phase-1 download survives in a session scratchpad and was copied to gitignored
  `.cache/acebench_raw/` (12 files, sha256 manifest in `dataset_manifest.json`). File-level line
  counts match the Phase-1 review exactly (per language: single 100, parallel 100, similar_api 50,
  special_incomplete 50, special_error_param 50, special_irrelevant 50).
- No pinned commit hash of the ACEBench repo was recorded in Phase 1; the raw files themselves are
  hash-manifested now, which pins the actual data used.

## 2. Dataset license

**MIT** (repository footer, verified 2026-07-14; also recorded in archived provenance).

## 3. Available files and splits

- Raw: the 6 single-turn categories × {en, zh} above (row schema `{id, question, function[], time}`;
  zero schema violations across all 1,200 × 2 raw rows scanned — 800 used).
- Converted decision dataset (archived): 800 rows — **500 `tool_call` / 100 `ask_user` /
  100 `flag_param_error` / 100 `cannot_comply`**, EN 400 + ZH 400, deterministic stratified
  70/15/15 split, seed 42 (`acebench_decision_split_info.json`, plain JSON, readable).
- **The archived converted JSONLs are LFS pointers with no local objects** (same LFS situation as
  Phase 3; fetching is off-limits). The dataset was **regenerated** from the raw files with the
  archived converter logic (`scripts/acebench_inspect_schema.py`) and verified: all 24
  per-(mode,lang,split) cells match the archived split-info **exactly**; gold distribution exact;
  the archived pilot's first-200 sample-order gold check matches. Regeneration adds the raw `time`
  field (needed by the official prompt template; the archived converter dropped it; the split logic
  never touches it). Location: gitignored `.cache/acebench_phase4/`.

## 4. Native label names and meanings (converter mapping, file-category-native)

| native mode | source files | meaning |
|---|---|---|
| `tool_call` | normal_single_turn_single_function / _parallel_function / similar_api | request is answerable by calling the provided function(s) now |
| `ask_user` | special_incomplete | a **required** parameter is missing → ask the user for it |
| `flag_param_error` | special_error_param | a provided parameter value violates the schema format/constraint → point it out |
| `cannot_comply` | special_irrelevant | no provided function can accomplish the request → say so |

## 5. Examples per native label

500 / 100 / 100 / 100 (see §3); per-language 250 / 50 / 50 / 50.

## 6. Per-example or per-turn labels?

**Per-example.** Every row is a single-turn request; the gold mode is the ACEBench *file category* —
fully native, no relabeling, no per-turn derivation.

## 7. What the previous readout did

`scripts/eval_acebench_decision_baseline.py`: 4-way **candidate scoring** — four fixed neutral
response strings (language-matched), scored by average token log-probability given the prompt
(system = function schemas + "decide how to respond"); pred = argmax. Plus an alternative-phrasing
sensitivity check on the first 200 EN rows.

## 8. Why the previous readout failed (archived numbers)

R0 collapse, recorded in `new_benchmark_pilot/acebench_pilot_baseline_summary.json` and the Phase-1
decision:
- accuracy **0.196** ≪ majority-gold 0.625;
- **one-class collapse**: `ask_user` predicted 596/800 (74.5%); tool_call recall 0.032;
- **phrasing instability**: main-vs-alternative candidate agreement **0.595**; on the same 200-row
  subset the main phrasing scored 0.0 accuracy vs 0.26 for the alternative — the "prediction" was a
  property of the candidate strings, not of the model's decision;
- ZH worse than EN (0.157 vs 0.235).
The Phase-1 verdict: *"The dataset is suitable; the cheap readout is not. A generation-based readout
is a protocol project."* That is this phase. The failure must not be attributed to ACEBench itself.

## 9. Reusable parts of the prior attempt

- The **converted decision dataset design** (native 4-mode labels, EN+ZH pooling with per-language
  reporting, stratified 70/15/15 seed-42 split) — regenerated and verified (§3).
- The **leakage-control principle** (no category vocabulary tied to file provenance in prompts).
- The archived **R0 rule family** (Phase-2 §8: majority-margin, collapse rule, validity of scores,
  ≥1 transition with support, stability, no leakage) and the **discovery pipeline**
  (`discover_sakiko_channels.py`-style transition analysis with default/strict/lenient thresholds +
  bootstrap) — reusable unchanged on generation predictions.
- The failed pilot artifacts remain untouched as the negative reference
  (`final/results/cross_dataset_channel_discovery/new_benchmark_pilot/`).

## 10. What must change in a generation-based readout

- The model must **generate** its decision (greedy, deterministic), not have fixed strings scored.
- Output classification must come from a **deterministic parser over observable behavior** (a
  parsable function call, or one of the official bracketed special responses), never from logprobs
  of curated candidates.
- The output schema should follow the **official ACEBench special-data protocol** (verified from the
  official prompt file, §"official semantics" below), which uniformly instructs, for *every*
  example: emit `[ApiName(key=...)]` when solvable; else one of three official bracketed
  responses (incorrect value / missing parameters / capability exceeded). This is the native
  schema, applied identically to all rows — it cannot leak per-example gold.

## 11. Risks of label leakage

- Using different prompts for normal vs special files would leak the category → **one unified
  prompt for all 800 rows** (the official special-data prompt already covers both scenarios).
- Prompt must not contain file names, "normal/special" vocabulary, or the converted mode names.
- Gold mode must never enter the generation context; it is joined only at evaluation time.
- Parser must not key on `sample_id`, `source_file`, or metadata (explicit check in validation).

## 12. Risks of subjective post-hoc labeling

- Free-text classification of refusals/questions is judgment-laden, especially bilingual output.
  Mitigation: the official bracketed formats make the primary parse path exact-string/regex;
  free-text fallback rules are locked before the run, are conservative (UNKNOWN rather than forced),
  and every fallback decision carries a rule-path tag + ambiguity flag for audit.
- The audit (Step 10) checks *parser-rule compliance*, not relabeling; predictions are never edited.

## 13. Stop conditions for this phase

1. Raw-data verification failure (counts/schema/split mismatch vs archive) — **checked: passed**.
2. Parser fails validation (unit tests, determinism, gold-independence) → stop before inference.
3. Smoke test shows model-loading failure, schema collapse (>50% UNKNOWN), or truncation → revise
   protocol (documented) before full run; if unresolvable → READOUT NO-GO.
4. Full-run R0 FAIL → no channel claims; Phase-4 outcome C (readout NO-GO) or D (dataset NO-GO)
   depending on where the failure localizes.
5. Any evidence that official semantics cannot be operationalized without seeing gold → outcome D.

---

## Official semantics verification (Step 2; primary sources only, accessed 2026-07-14)

| source | URL | what was taken |
|---|---|---|
| Official repo README | https://github.com/ACEBench/ACEBench | MIT license; category overview; paper link |
| Official paper (abstract + HTML body) | https://arxiv.org/abs/2501.12851 · https://arxiv.org/html/2501.12851v4 | Special-subcategory definitions and expected behaviors; binary special-accuracy judging |
| Official EN prompt file | https://raw.githubusercontent.com/ACEBench/ACEBench/main/model_inference/prompt_en.py | `SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN` + `USER_PROMPT_EN`, saved verbatim to `.cache/acebench_raw/official_prompt_en.py` |
| Official ZH prompt file | https://raw.githubusercontent.com/ACEBench/ACEBench/main/model_inference/prompt_zh.py | `SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH` + `USER_PROMPT_ZH`, saved verbatim |
| Official checker | https://raw.githubusercontent.com/ACEBench/ACEBench/main/model_eval/checker.py | saved verbatim for reference |

Per-mode official documentation:

| native mode | official definition (paper) | expected behavior | observable output evidence (official formats) | ambiguity notes | identifiable w/o gold? |
|---|---|---|---|---|---|
| `tool_call` | information clear; solvable with candidate functions | emit the API request(s) | output begins `[` and parses as `ApiName(key='value', ...)` call list | multiple valid calls allowed (parallel); optional params only if mentioned | **yes** |
| `ask_user` (incomplete) | "key information required for the function call is missing … such as absence of 'required' parameters" | identify + ask for the missing parameter(s) | `["Missing necessary parameters (key1, key2, ...) for the api (ApiName)"]` | must be distinguished from wrong-value case (official note: priority order, check value-error first) | **yes** |
| `flag_param_error` (error_param) | "parameters … do not meet the required format or constraints" | identify the erroneous parameter | `["There is incorrect value (value) for the parameters (key) in the conversation history."]` | boundary vs `ask_user` when a value is both odd and missing context | **yes** |
| `cannot_comply` (irrelevant) | "instruction exceeds the function's capabilities; none of the candidate functions can resolve the issue" | inform the user it cannot be fulfilled | `["Due to the limitations of the function, I cannot solve this problem."]` | model may instead attempt a near-miss call (that is exactly the error mode of interest) | **yes** |

Two protocol-relevant official facts: (a) the special-data system prompt covers *both* the
call-emitting scenario and the three failure scenarios in one instruction — a single unified prompt
per language is the official pattern; (b) even the **ZH** prompt specifies the three special
responses **in English**, so the primary parse path is language-uniform.

**No W2C mapping is invented anywhere in this phase: ACEBench stays in its native 4-mode space.**
(Analogy notes — `ask_user`≈request_for_info, `cannot_comply`≈cannot_answer, `flag_param_error` has
no W2C counterpart — are discussion-only, as in the Phase-1 review.)
