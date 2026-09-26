# Phase 5 — Existing-Evidence Audit (before any Mistral code)

**Project:** SAKIKO-CA — mechanistic intervention for LLM tool-use behavior.
**Phase-5 question:** can the *complete* SAKIKO-CA pipeline discover and validate
*model-specific* behavioral channels in a second, independent 7B architecture
(**Mistral-7B-Instruct-v0.3 × When2Call**) without reusing Qwen's channels, layers,
routers, or correction directions?

**Repo:** `/root/autodl-tmp/sakiko-followup` · branch `exp/sakiko-followup-archive`
(verified at startup; working tree clean, no unrelated modified/untracked files).
This audit reconstructs the program *from the repository*, not from memory. All facts
below are sourced from committed files; source paths are cited inline.

---

## 1. Exact W2C dataset and split protocol

- **Dataset:** `nvidia/When2Call`, config `test`, split `mcq` → **3652 examples**.
  Loaded via `datasets.load_dataset("nvidia/When2Call","test",split="mcq")` and cached
  locally under `/root/.cache/huggingface/datasets/nvidia___when2_call`.
  A byte-equivalent raw copy is at `data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl`
  (gitignored, 20 MB).
- **Fields used:** `uuid`, `question`, `tools` (list of tool JSON/strings), `answers`
  (dict with keys `direct` / `tool_call` / `request_for_info` / `cannot_answer`),
  `correct_answer` (gold label ∈ the 4 keys).
- **Canonical ordering:** `.shuffle(seed=42)` on the loaded dataset. **Verified this
  Phase-5 session:** the HF-shuffled order is *bit-identical* to
  `Dataset.from_list(union-normalized raw rows).shuffle(seed=42)` — the archived
  "LFS-less shim" recipe (`phase3_lib.load_ds_raw`). First-5 uuids match; all 3652 rows
  match. This means the local raw jsonl reproduces the exact evaluation order, so split
  indices are portable across models.
- **Archived fixed split** (`final/results/splits/` == `sakiko_v3/results/splits/`):
  `train_idx=2556`, `val_idx=548`, `test_idx=548` (70/15/15). These are index arrays
  into the seed-42-shuffled order, *stratified on Phi-3.5 error types* (a fixed held-out
  partition; independent of the target model). Used for the seed-42 pilot.
- **Per-seed splits (multiseed):** `qwen7b_native_sakiko_multiseed.gen_split()` —
  deterministic `StratifiedShuffleSplit` 70/15/15 stratified on **the target model's own
  per-sample error type** (`etype`), rare classes (<5) pooled as `_rare`,
  `random_state=seed`. Seeds `{42,123,456,789,2024}`.
- **Scoring protocol (identical Phi→Qwen→Mistral):** for each of the 4 candidate
  response-mode strings `answers[label]`, compute the **length-normalized average
  log-prob** (`avg_logp`) of the candidate tokens conditioned on the chat-templated
  prompt; **argmax = prediction**. No generation, no activation use at baseline.
  Prompt = system("You are a helpful assistant." + tools JSON or "No tools are
  available." + "Given the user's question, choose the most appropriate response from
  the provided options.") + user(question), then `apply_chat_template(...,
  add_generation_prompt=True)`. Source: `scripts/eval_w2c_7b_baseline.py`,
  `scripts/qwen7b_native_sakiko.py`.
- **Label space is a semantic response-mode space** (tool_call / request_for_info /
  cannot_answer / direct), *not* native tool-call execution syntax. `direct` is a W2C
  distractor and is **never a gold label** (gold ∈ {tool_call, request_for_info,
  cannot_answer}). This is retained verbatim for Mistral — Phase 5 evaluates
  response-mode *choice*, not Mistral's tool-call format.

## 2. Archived Phi and Qwen baselines (W2C, n=3652)

| model | source | acc | macro-F1 (3-cls) | pred distribution (dominant) |
|---|---|---|---|---|
| **Phi-3.5-mini-instruct** | `trace_db/w2c_phi35_summary.json` | **0.4822** | 0.4526 | tool_call 2470 / direct 471 / request_for_info 380 / cannot_answer 331 |
| **Qwen2.5-7B-Instruct** | `final/results/7b_w2c_baseline/qwen25_7b_error_decomposition.json` | **0.4381** | 0.4006 | tool_call 2244 / request_for_info 745 / direct 560 / cannot_answer 103 |

- Shared **gold distribution:** tool_call 1295 / cannot_answer 1295 / request_for_info 1062.
- Both models **over-commit to `tool_call`** (predict it ~2.2–2.5× its true frequency) —
  the dominant failure mode is calling a tool when the correct behavior is to ask for
  info or decline. This is the behavioral phenomenon SAKIKO targets. **Whether Mistral
  shows the same over-call bias is an open empirical question for R0.**

## 3. Channels discovered for Phi and Qwen

A **channel = a gold→baseline-pred error transition**. Discovery (exact per-sample,
`scripts/discover_sakiko_channels.py`, thresholds `stable_train_min=50`,
`stable_eval_min=15`, `weak_train_min=20`, 500-bootstrap stability) yields:

**Qwen2.5-7B (exact per-sample; `final/results/channel_adaptive/channel_discovery_summary.csv`):**

| channel | gold → pred | count_all | count_train | status@default |
|---|---|---|---|---|
| rfi_tc | request_for_info → tool_call | 622 | 432 | stable |
| ca_tc | cannot_answer → tool_call | 513 | 352 | stable |
| ca_direct | cannot_answer → direct | 465 | 332 | stable |
| **ca_rfi** | cannot_answer → request_for_info | 219 | 151 | stable |
| **tc_rfi** | tool_call → request_for_info | 133 | 90 | stable |
| tc_direct | tool_call → direct | 53 | 38 | stable |
| rfi_direct | request_for_info → direct | 42 | 30 | stable |

**Phi-3.5 (summary-level; test-split counts from `sakiko_v3/results/p0_final_test_eval.json`):**
rfi_tc (746 all / 112 test), ca_tc (511 / 77), ca_direct (393 / 59) — the same 3
dominant over-commitment transitions. Phi is the origin of the manually fixed
"three-channel" SAKIKO v1.

**Cross-dataset (context):** MetaTool-Binary → exactly **1** channel (no_tool→tool_call);
ToolSandbox → **0** quantitative channels (qualitative-only). Channel structure is
**dataset × model specific** — the number and identity of channels is *not* fixed.

## 4. Which channels passed / failed intervention validation

- **SAKIKO v1 manual set = {rfi_tc, ca_tc, ca_direct}** — all three are behaviorally
  intervenable on Qwen; multi-seed cascade **Net +92 ± 20 (5/5 positive)**
  (`final/results/7b_w2c_sakiko/multiseed/`).
- **Phase-3 auto-discovered channels absent from the manual set**
  (`final/results/w2c_qwen_new_channel_intervention/`):
  - **`ca_rfi` (cannot_answer→request_for_info): PASSED.** Direction-specific,
    multi-seed validated. Locked config: obs L20, inj L18, α=2.0, thr=0.8, DiffMean,
    dir_norm 16.0, router val-AUC 0.918, val-Net +16 (fixed 18 / broke 2). All 8 gate
    criteria + all placebos pass. Added to cascade: **+15.8 Net mean over 5 seeds (5/5
    wins)**.
  - **`tc_rfi` (tool_call→request_for_info): FAILED utility gate.** Discovered & stable,
    router val-AUC 0.849, val-Net +6 — but **Net ≈ 0 on locked test, fails 5/8 gate
    criteria, random directions saturate the placebo (11/20 ≥ real)**. Honest negative:
    *discovery ≠ utility*.
- **Key lesson for Mistral:** every discovered channel must independently pass the
  behavioral **utility gate** and the **direction-specificity placebo battery**; a stable
  transition and a high router AUC do **not** entitle a channel to inclusion.

## 5. Archived SAKIKO and SAKIKO-CA rules

**SAKIKO v1 (fixed-channel):** 3 hard-coded channels, each = {LR router (fires on likely
channel error) + DiffMean steering direction (correct-reference mean − error mean)},
injected at an MLP output layer, **gated** by baseline prediction == channel's "from"
class, **cascaded** by priority `rfi_tc → ca_tc → ca_direct`, thresholds tuned on val,
evaluated once on a locked test.

**SAKIKO-CA (channel-adaptive) pipeline** (`SAKIKO_METHOD_REFRAMING.md` §3):
`baseline eval → transition discovery → candidate selection (support/stability/scope) →
channel geometry validation (R1/R2/R5) → router+direction extraction on train →
validation-only tuning (thresholds, inj/α; R3/R4) → locked test once → causal/specificity
controls`. Boundary rule: if no stable channel survives, the dataset is *qualitative-only*
— do not force channels.

## 6. R0–R5 definitions (as used in the archive)

- **R0 — baseline-validity gate** (`discover_sakiko_channels.transition_discovery`):
  readout is INVALID if `majority_pred_share > COLLAPSE_PRED_SHARE (0.90)` **and**
  `acc < majority_gold_share` (collapsed/degenerate readout). Also requires meaningful
  accuracy above majority, non-collapsed predictions, finite scores, deterministic
  replay, adequate transition support. **Gate before any activation work.**
- **R1 — degeneracy gate (obs-layer/direction):** reject obs layers with DiffMean norm
  `< DEGEN_NORM (5.0)`; among survivors pick max relative norm-ratio
  (norm / median activation norm), tie-break router AUC. (Killed Qwen ca_direct@L16:
  norm 2.12, bootstrap-unstable 0.823 → moved to L20 norm 10.41.)
- **R2 — direction method:** DiffMean is default; if `cos(DiffMean, PC1) < COS_THR (0.6)`
  also try PCA-1, decide winner on val. (Qwen ca_tc: cos 0.41 → PCA-1 won +50 vs +36.)
- **R3 — detection-limited channels** (router AUC ≲ 0.8): favour precision, optionally a
  val-tuned abstention rule (router-score floor + projection cap).
- **R4 — sanity gate:** require real ≫ random on val; flag high reverse-retention channels
  (partially generic directions).
- **R5 — structural check:** high probe-AUC + low silhouette = expected regime (linear
  direction in overlapping clouds) → proceed; low probe-AUC → abstain from the channel.
  (Silhouette ≈ 0 ⇒ channels are **linear decision directions, not clusters** — never use
  "cluster" language.)

## 7. Existing layer- and direction-selection logic

- **Intervention:** `h' = h + α · median_norm · unit(direction)` added to the **MLP output**
  at the injection layer, all positions, during candidate scoring
  (`qwen7b_native_sakiko.score_hooked`). `median_norm` = median L2 norm of train
  activations at the obs layer.
- **Obs-layer grid (Qwen, 28 layers):** `OBS_LAYERS=[12,16,20]` (≈0.43/0.57/0.71 depth).
  Phase-3 used `[12,16,20,24]`.
- **Injection grid:** `INJ_OFFSETS=[0,-2,-4]` relative to obs (Phase-3 also derives inj
  candidates near obs). **Wrong-layer control** injects at L4.
- **Router:** `LogisticRegression(C=1.0, liblinear, max_iter=2000)` on `StandardScaler`-ed
  activations. Positives = channel train errors; negatives = **correct** examples
  (comparability AUC) or **matched-pred** (samples with same baseline pred, different
  etype) in Phase-3. Obs layer chosen by **val AUC**.
- **Direction:** DiffMean `unit(ref_mean − err_mean)` (`qwen7b_method_diagnostics.diffmean`)
  or PCA-1 sign-aligned to the DiffMean delta (`pca_dir`). `ref` = correct examples whose
  gold == channel's ref_gold; `err` = channel error examples.
- **α grid** `[0.5,1,2,4,6]`, **threshold grid** `[0.4,0.5,0.6,0.7,0.8]`.
- **Selection objective:** maximize **Net = Fixed − Broke** on val (tie-break −broke, then
  channel-fixed). **Never select on AUC alone.**

## 8. Known failure modes (carry into Phase 5)

1. **Discovery ≠ utility** (tc_rfi): stable + decent AUC, yet Net≈0 / random-saturated.
2. **Layer/direction degeneracy** (ca_direct@L16): high AUC can coexist with a degenerate,
   bootstrap-unstable direction (norm 2.12) that fails silently → R1.
3. **Router-separability ceiling** (rfi_tc AUC ~0.75–0.78): a detection-limited channel
   carries most of the Broke (11/13) → R3.
4. **Partially-generic directions** (ca_tc): reverse retains ~46% of the fix ⇒
   "partially direction-specific", not fully specific → R4.
5. **Binary-space placebo saturation** (MetaTool 2/25 random ≥ real): 2-class spaces make
   random-direction controls weak; W2C's 3-gold space is safer but stay alert.
6. **Cross-model mapping is weak/non-specific** (ridge/PCA/Procrustes/PLS Phi→Qwen: best
   +23 vs native +79; wrong-layer +35 > real). ⇒ **target-native extraction only.**

## 9. Model-agnostic components (reuse safely, unchanged logic)

- W2C dataset loader, seed-42 shuffle, label vocabulary, candidate scoring, avg_logp
  math, prompt/chat-template construction (uses the *target's own* chat template).
- The archived **fixed split indices** (stratified on Phi errors — model-independent) for
  the seed-42 pilot; the **per-seed stratified-split generator** for multiseed.
- Discovery framework (`transition_discovery`), thresholds, bootstrap-stability,
  R0 validity gate, cross-model comparison logic.
- Router/direction *math* (`fit_router`, `diffmean`, `pca_dir`), Net/Fixed/Broke metrics,
  destination/transition tables, the 8-criteria utility gate, placebo battery
  (real/reverse/random×20/ungated/wrong-layer), multiseed harness structure.

## 10. Components that MUST be relearned for Mistral (no reuse)

- **Model weights, tokenizer, chat template** → Mistral's own.
- **Baseline predictions** (`etype` per sample) → recomputed on Mistral (R0).
- **Which channels are discovered / stable** → re-run discovery on Mistral's confusion.
- **Which channels pass the utility gate** → re-validate; do **not** assume Qwen's set.
- **Activations, routers, directions, thresholds, α, obs/inj layers** → all re-extracted
  and re-selected on Mistral's own activation space.
- **Absolute layer indices** → **forbidden to copy** from Qwen. Layer grid defined by
  **normalized depth** on Mistral's **32 layers** (see PHASE5_PROTOCOL.md):
  0.40L→L13, 0.55L→L18, 0.70L→L22, 0.85L→L27 (obs), inj offsets ~{0, −0.06L≈−2, −0.12L≈−4}.

## 11. Available local data, caches, model files

- **W2C:** HF cache present + local raw jsonl (verified identical order). ✅
- **Splits:** `final/results/splits/{train,val,test}_idx.json`. ✅
- **Qwen model:** `.cache/modelscope/Qwen/Qwen2___5-7B-Instruct` (not needed for Phase 5). ✅
- **Mistral-7B-Instruct-v0.3:** downloading (in progress this session) to
  `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3` (**outside the repo**), official HF
  repo `mistralai/Mistral-7B-Instruct-v0.3`, revision
  `c170c708c41dac9275d15a8fff4eca08d52bab71`, **not gated**, bf16, HF sharded format
  (3 safetensors shards; `consolidated.safetensors` deliberately excluded). Config
  confirmed: **32 layers, hidden 4096, vocab 32768, max_pos 32768, MistralForCausalLM.**
- **Env:** conda `sakiko-phase3` — Python 3.10.8, torch 2.1.2+cu121 (CUDA ✅),
  transformers 4.49.0, datasets 3.2.0, sklearn 1.7.2, numpy 1.26.3. Supports Mistral-v0.3.
- **GPU:** RTX 4090 D, 24 GB, **free** (0 MiB used). bf16 7B fits (~15 GB) + scoring.
- **Disk:** `/root/autodl-tmp` 36 GB free (model lands here); `/root` overlay only 16 GB
  free (do NOT put weights there).

## 12. Missing dependencies / LFS objects

- **Git LFS is unavailable** in this clone (as in Phase 3). `.gitattributes` routes
  `*.npy/*.npz/*.jsonl` to LFS, but LFS objects are not fetched. Consequence: **large
  per-example artifacts and activation caches must be regenerated locally and kept
  gitignored** (exactly the Phase-3 pattern). Baseline accuracy is reproduced
  deterministically from the local raw jsonl; claims rest on drift-immune in-process
  marginals.
- `modelscope` python package not installed (not needed — using official HF).
- No Mistral activation cache exists yet (will be generated under a gitignored path).

## 13. Risks that could invalidate the experiment

1. **R0 collapse / weak baseline.** If Mistral's readout collapses (e.g. predicts one
   class > 90% with acc < majority) or has too few errors per transition, discovery is
   unsupported → stop at R0 with a blocker (do not force channels).
2. **Different / absent over-call structure.** Mistral may not over-call; its dominant
   channels may differ from Qwen's. That is a *finding*, not a failure — but it means the
   intervention may target different transitions (or none).
3. **Chat-template / tokenization mismatch.** Mistral-v0.3 uses `[INST]…[/INST]` and a
   different tokenizer (vocab 32768, `tokenizer.model.v3`). Candidate strings must remain
   scoreable and finite; must verify tokenization and template in preflight.
4. **Determinism.** Must confirm bit-identical deterministic replay of predictions before
   trusting Fixed/Broke marginals.
5. **Discovery-without-utility.** High risk of a discovered channel that fails the utility
   gate (cf. tc_rfi). Mitigated by the mandatory gate + placebo battery.
6. **Test leakage / test-driven tuning.** All selection on train/val only; locked test
   once per arm. Enforced by writing PHASE5_PROTOCOL.md and locked configs *before*
   inspecting test.
7. **Disk pressure** (36 GB free, model ~14.5 GB, activation caches ~4×64 MB/layer).
   Manageable; monitor.
8. **bf16 requirement.** No silent 4-bit/fp16 fallback; if bf16 cannot run, stop and
   report.

---

### Reuse decision

New Phase-5 code lives under `scripts/phase5_mistral_*.py` and reuses the archived
*math/harness* (discovery, router/direction, metrics, gate, placebo, multiseed structure)
via a thin Mistral-specific library — **without** importing any Qwen model path, Qwen
layer constant, Qwen router/direction, or Qwen channel set. All Mistral outputs are
isolated under `final/results/mistral7b_w2c_sakiko_ca/`. No archived file is modified.
