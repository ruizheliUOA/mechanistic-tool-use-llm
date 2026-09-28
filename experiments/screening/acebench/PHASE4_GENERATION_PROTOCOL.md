# Phase 4 — Locked Generation Protocol: ACEBench decision readout (Qwen2.5-7B)

**Date locked: 2026-07-14, BEFORE parser validation, smoke test, and the full baseline.** The full
baseline may only run after (a) the parser passes validation (Step 6) and (b) the smoke test passes
(Step 7). Any post-hoc change to prompt/parser/params creates a new protocol version and requires a
fresh run. No intervention of any kind in this phase.

## 1–2. Dataset, version, subset, split
- ACEBench single-turn decision subset, regenerated from the Phase-1 raw download
  (`.cache/acebench_raw/`, sha256 manifest in `dataset_manifest.json`; source
  github.com/ACEBench/ACEBench `data_all/data_{en,zh}`, MIT, arXiv:2501.12851).
- 800 rows: 500 `tool_call` / 100 `ask_user` / 100 `flag_param_error` / 100 `cannot_comply`
  (EN 400 + ZH 400); archived stratified 70/15/15 split, seed 42 — regeneration verified cell-exact
  vs `acebench_decision_split_info.json` (24/24) + archived first-200 order check.
- Dataset file: `.cache/acebench_phase4/acebench_decision_all_with_time.jsonl`
  (sha256 in `READOUT_LOCK_MANIFEST.json`). The full 800 rows are evaluated once; per-split metrics
  reported; no tuning on any split after the lock.

## 3–4. Model & precision
- `Qwen2.5-7B-Instruct`, local path `.cache/modelscope/Qwen/Qwen2___5-7B-Instruct` (present, 4
  shards, 15.231 GB index-verified in Phase 3). **bf16** on cuda:0 (RTX 4090 D); no 4-bit/8-bit, no
  silent fallback. Env `sakiko-phase3` (torch 2.1.2+cu121, transformers 4.49.0).

## 5–8. Prompts and tools
- **One unified prompt for all 800 rows** (leakage control §11 of the audit): the **official ACEBench
  special-data system prompt, verbatim**, per language — `SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN` for EN
  rows, `SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH` for ZH rows (saved verbatim in
  `.cache/acebench_raw/official_prompt_{en,zh}.py`; upstream
  raw.githubusercontent.com/ACEBench/ACEBench/main/model_inference/prompt_{en,zh}.py, accessed
  2026-07-14). It instructs, identically for every example: emit `[ApiName(key='value', ...)]` calls
  when solvable; otherwise one of the three official bracketed responses (incorrect value / missing
  parameters / capability limitation).
- Template slots: `{function}` ← `json.dumps(row.functions, ensure_ascii=False)`;
  `{time}` ← the row's raw `time` string (may be empty).
- **User message** = official `USER_PROMPT_{EN,ZH}`: `Conversation history 1..t:\n{question}` with
  the question restored to its raw form `user: {question}` (the converter stripped the prefix; the
  builder re-adds it; rows whose raw question lacked the prefix are used as-is — counted in the
  baseline report).
- Applied with the Qwen chat template (`apply_chat_template`, `add_generation_prompt=True`).
- **Tools are simulated (schemas only), never executed** — identical to the official single-turn
  protocol. No tool results exist in a single-turn decision.
- The prompt contains **no** file names, no normal/special vocabulary beyond the official
  instruction text (which is uniform across rows), and no converted mode names.

## 9–13. Generation parameters (locked)
- Greedy decoding: `do_sample=False`, `num_beams=1`; `max_new_tokens=256`; default EOS as the only
  stop; `pad_token = eos`; batch size 1; sample order = ascending `sample_id`.
- Max prompt length 8192 tokens (longer → skipped + counted; none expected).
- Determinism: greedy bf16 on a single GPU, batch=1, fixed order. A fixed 50-row replay subset
  (§18) quantifies residual kernel nondeterminism.

## 14–16. Output parser & policies
- Parser: `scripts/acebench_parse_outputs.py` **v1.0** (deterministic, versioned, gold-blind; sha256
  in the lock manifest). Rule summary (full spec in the script header):
  1. Extract the first balanced `[...]` block of the generation.
  2. Inside the block, in **locked priority order**: official *incorrect-value* pattern →
     `flag_param_error`; official *missing-parameters* pattern → `ask_user`; official
     *capability-limitation* pattern → `cannot_comply`; else a parsable call list
     `Name(arg=..., ...)` → `tool_call`; else `UNKNOWN(bracketed_unclassified)`.
     (Value-error before missing mirrors the official prompt's stated priority.)
  3. If both a parsable call AND an official special pattern occur in the same block →
     `UNKNOWN(conflict)` — conservative, never forced (multiple-action policy).
  4. No bracket → conservative bilingual fallback cues (question asking for missing info →
     `ask_user`; incorrect/invalid + parameter → `flag_param_error`; cannot/无法 + capability →
     `cannot_comply`; bare `Name(...)` call syntax → `tool_call`), every fallback flagged
     `ambiguous=true`; else `UNKNOWN(no_rule)`.
- Every parse records: sample_id, raw output, label, rule_path, ambiguity flag, unparseable reason,
  extracted calls. Gold is joined **after** parsing, for scoring only.
- UNKNOWN is a first-class outcome; UNKNOWN rows count as errors for accuracy and are excluded from
  channel-discovery transitions (they have no predicted mode).

## 17. R0 criteria (locked; majority-gold share = 0.625)
- **R0.1** accuracy ≥ **0.675** (majority + 0.05) **OR** (balanced accuracy ≥ **0.45** AND
  macro-F1 ≥ **0.40**) — the balanced alternative is the pre-registered task-specific route for a
  62.5%-majority 4-class task (chance balanced-acc = 0.25).
- **R0.2** macro-F1 ≥ 0.40 (prior failed readout: 0.26).
- **R0.3** no collapse: largest predicted-class share ≤ 0.70 (prior failure: 0.745) AND ≥3 of 4
  modes each ≥ 5% of predictions.
- **R0.4** UNKNOWN + unparseable ≤ 10%.
- **R0.5** parser audit (Step 10): rule-compliance ≥ 95%; estimated parser mislabel rate ≤ 5%.
- **R0.6** determinism: 50-row fixed replay ≥ 98% identical labels.
- **R0.7** non-empty generations ≥ 99%; no systematic truncation (≥ 95% end with EOS before 256).
- **R0.8** no label leakage (procedural checklist: uniform prompt, gold joined post-hoc, parser
  metadata-blind — verified in parser validation).
- **R0.9** discovery support: ≥ 2 gold→pred transitions with **train ≥ 30 AND val ≥ 5 AND test ≥ 5**
  (the archived SAKIKO-CA intervention-scope thresholds; justified pre-hoc: special classes cap at
  100 examples, so W2C's default 50/15 tier is arithmetically out of reach for special-gold
  transitions — the lenient-tier discovery thresholds (30/10) are reported alongside, as in the
  archived pipeline).
- **R0.10** multi-class structure: ≥ 3 of 4 gold classes with recall ≥ 0.10.

**Verdict rule (exactly one):** PASS = R0.1–R0.10 all hold. **CONDITIONAL PASS** = R0.2–R0.8 hold
AND balanced accuracy ≥ 0.35, but R0.1, R0.9, or R0.10 fails → descriptive error analysis only;
intervention eligibility additionally requires the pre-specified check: a second, semantically
equivalent (non-official-wording) system prompt on a 100-row stratified subset with prediction
agreement ≥ 0.80. **FAIL** otherwise. Thresholds may not move after results are seen.

## 18. Parser validation criteria (Step 6 gate, before any model run)
- Unit tests: canonical outputs for all 4 modes (EN+ZH), malformed, mixed-mode/conflict,
  missing-field, wrong-argument calls, clarifications, cannot-proceed, direct answers — 100% of
  expected labels; determinism: byte-identical results across 3 repeated runs over the suite;
  gold-blindness: parser signature takes only the raw text (no metadata argument at all);
  no rule references sample ids, file names, or dataset metadata (code review + grep).
- Additionally 20 real dataset inputs (10 EN / 10 ZH, stratified) get *hand-written expected
  behavioral outputs* (written from the input + schemas only, gold unseen at writing time) run
  through the parser.

## 19. Human-audit sampling protocol (Step 10; after the full run; no prediction edits)
- Stratified ≥ 100: ≥ 5 per gold mode × major predicted modes, every UNKNOWN (cap 25), every
  conflict, 10 random ambiguity-flagged fallback parses, 10 near-boundary bracketed parses.
- Audit records rule-compliance, genuine-ambiguity, gold-plausibility; verdict per row.
  This audits the parser, not the labels; predictions are never altered.

## 20. Channel-discovery feasibility criteria (Step 12; only if R0 allows)
- Run the archived transition-discovery procedure (all gold→pred transitions; default/strict/lenient
  thresholds; 200-resample bootstrap; UNKNOWN rows excluded) on the locked predictions.
- Feasibility = ≥ 2 transitions passing the intervention-scope filter of R0.9 with bootstrap
  stability ≥ 0.80, at least one of which involves a special-mode gold or prediction (the
  potential *new channel family* vs W2C/MetaTool: any transition touching `flag_param_error`).

## 21. GO / CONDITIONAL-GO / NO-GO mapping (Step 13; exactly one)
- **A GO:** R0 PASS and §20 feasibility met → recommend ACEBench target-native SAKIKO-CA pilot.
- **B CONDITIONAL GO:** R0 PASS but §20 marginal (exactly 1 qualifying channel, or stability
  0.6–0.8), or R0 CONDITIONAL PASS with the paraphrase check passed → one narrowly defined
  follow-up validation study.
- **C READOUT NO-GO:** R0 FAIL with evidence localizing the failure to the readout (e.g., high
  UNKNOWN, parser or format collapse) → do not run ACEBench intervention; proceed to second-model
  architecture validation.
- **D DATASET NO-GO:** official semantics/data quality prevent objective evaluation (e.g., audit
  shows gold labels incompatible with official definitions at a material rate) → retire ACEBench
  from the quantitative main line.

## Smoke-test plan (Step 7 gate)
- 48 **train-split-only** rows: 12 per mode (6 EN + 6 ZH), lowest-sample_id deterministic pick.
- Exact final code path (same builder/model-loader/parser). Report the Step-7 checklist (env,
  dtype, params, lengths, parse rates, prediction distribution).
- Gate: model loads bf16; 0 OOM; parser non-UNKNOWN ≥ 70% on smoke; EOS-termination ≥ 90%;
  no repeated-boilerplate collapse (≤ 50% identical outputs). Failing → documented revision loop
  (each revision recorded in this file's changelog) before any full run.

## Compute & artefact policy
- Estimated full-run cost: 800 greedy generations ≤ 256 new tokens ≈ 1–2.5 h on the 4090 D.
- Raw generations + parsed details stay in gitignored `.cache/acebench_phase4/` (they embed dataset
  text); committable outputs are small JSON/CSV/MD summaries under
  `final/results/acebench_generation_readout/`.
- No archived artifact is overwritten; the failed candidate-scoring pilot stays as negative
  reference.

## Changelog
- v1.0 (2026-07-14): initial lock. (Smoke-test-driven revisions, if any, will be listed here with
  reasons; none may follow the full run.)
