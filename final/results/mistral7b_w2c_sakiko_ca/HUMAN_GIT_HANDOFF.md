# Phase 5 — Human Git Handoff

> The assistant performed **no** git write and **no** git network operation. Only read-only
> `git status / branch / log / diff / check-attr` were used. This document is instructions for
> the human researcher (the corresponding author) to review and commit.

## 1. Starting state (verified read-only at session start)

- **Branch:** `exp/sakiko-followup-archive` (unchanged throughout; `main` untouched)
- **Starting HEAD:** `1ff1f6f Add ACEBench generation-based readout validation` (unchanged)
- **Working tree at start:** clean — no unrelated modified/untracked files.

## 2. Every new file (34 results files + 8 scripts)

**Scripts (`scripts/`, all new, small text):**
```
phase5_mistral_lib.py               # Mistral-specific lib (data, scoring, hooks, geometry, metrics)
phase5_mistral_baseline.py          # R0 baseline
phase5_mistral_extract_acts.py      # obs-layer activation caching
phase5_mistral_discovery.py         # automatic channel discovery + scope filter + cross-model cmp
phase5_mistral_pilot.py             # geometry(R1/R2) + val sweep + locked arms + placebos + gate
phase5_mistral_cascade_placebo.py   # Arm-F cascade placebo, matched to archived Qwen protocol
phase5_mistral_multiseed.py         # present but NOT RUN (pilot gate failed)
phase5_mistral_report.py            # cross-model synthesis assembly
```

**Results (`final/results/mistral7b_w2c_sakiko_ca/`):**
```
PHASE5_EXISTING_EVIDENCE_AUDIT.md          PHASE5_PROTOCOL.md
PHASE5_ENVIRONMENT_AND_PREFLIGHT.md        PHASE5_CLAIM_BOUNDARY.md
MISTRAL7B_W2C_SAKIKO_CA_FINAL_REPORT.md    CROSS_MODEL_W2C_SYNTHESIS.md
PHASE5_REPRODUCIBILITY_AND_SAFETY_CHECKLIST.md   HUMAN_GIT_HANDOFF.md
cross_model_w2c_summary.csv
baseline/    MISTRAL7B_W2C_BASELINE_SUMMARY.md, *_summary.json, *_details.jsonl,
             *_confusion_matrix.csv, *_transition_counts.csv, *_smoke.json
channel_discovery/  MISTRAL7B_CHANNEL_DISCOVERY.md, *_discovery.{json,csv},
             cross_model_channel_comparison.csv
geometry/    MISTRAL7B_GEOMETRY.md, mistral7b_geometry.json, mistral7b_geometry_table.csv,
             mistral7b_activation_extraction.json
pilot/       locked_configs.json, pilot_summary.json, pilot_key_table.csv, _state.json
placebos/    PLACEBO_AND_SPECIFICITY_SUMMARY.md, placebo_all_controls.json,
             CASCADE_PLACEBO_ARMF.md, cascade_placebo_armF.json
figures/     fig1_specificity_qwen_vs_mistral.{png,pdf}, FIGURE_CAPTIONS.md
```
*(`multiseed/` does not exist — the pre-registered gate failed, so it was not run.)*

## 3. Modified existing files

- **NONE.** No archived result, thesis, report source, config, `.gitignore`, or `.gitattributes`
  was edited. `scripts/discover_sakiko_channels.py` is **imported and reused unchanged**.

## 4. Local-only files (outside the repo — never staged)

| path | size | note |
|---|---|---|
| `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3/` | 14 GB | model weights |
| `/root/autodl-tmp/phase5_mistral_cache/acts_L{13,18,22,27}.npy` | 4 × 57 MB | activation caches (regenerable) |
| `/root/autodl-tmp/phase5_mistral_cache/mistral7b_w2c_baseline_details.jsonl` | 1 MB | full per-example copy |
| `/root/autodl-tmp/phase5_mistral_cache/*.log` | small | run logs |
| `/root/autodl-tmp/hf_home/` | — | HF download cache |

## 5. Model download location & manifest

`mistralai/Mistral-7B-Instruct-v0.3` @ **`c170c708c41dac9275d15a8fff4eca08d52bab71`**, bf16,
public/ungated, **no token used**. Stored at `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3`
(outside repo). Files: `config.json`, `generation_config.json`,
`model-0000{1,2,3}-of-00003.safetensors`, `model.safetensors.index.json`, `params.json`,
`special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`.
Excluded: `consolidated.safetensors` (14.5 GB duplicate).
**SHA256 of all 3 shards + `tokenizer.model` verified byte-identical to the official HF LFS
oids** (fetched from the official `api/models/.../tree/main`). Downloaded via the `hf-mirror.com`
mirror after the proxy stalled the official LFS CDN; identity established by SHA equality.

## 6. LFS decision required from the committer

`.gitattributes` routes `*.jsonl` (and `*.npy/*.npz`) through Git LFS. The only LFS-routed file
in the staged set is `baseline/mistral7b_w2c_baseline_details.jsonl` (**1.05 MB**). Either:
- **(a)** commit via LFS (git-lfs 3.5.1 is installed) — consistent with the archive, **or**
- **(b)** keep it local: `git reset` that path after `git add` and add an ignore rule.

No scientific claim depends on it (accuracy is deterministically regenerable; replay was
bit-identical).

## 7. Scan results (final re-run, read-only)

- **Secret/token scan:** clean. No API keys, HF/GitHub tokens, passwords, bearer/cookies, or
  private keys. **No token was used** (repo ungated). **No proxy address in any committed file.**
- **PII scan:** clean, with one **intended** exception — the researcher's git author name/email
  inside the git command block of this file, as specified by the Phase-5 brief.
- **Large-file scan:** no file > 20 MB in the staged set; largest is the 1.05 MB details jsonl.
  Weights and `.npy` caches are outside the repo.
- `git diff --check`: **clean**. `git status --short`: only the new Phase-5 paths (untracked).

## 8. Safety verdict

**PASS** — see `PHASE5_REPRODUCIBILITY_AND_SAFETY_CHECKLIST.md`.

## 9. Scientific conclusion (one paragraph)

On a second, independent 7B architecture (Mistral-7B-Instruct-v0.3 × When2Call), the SAKIKO-CA
**procedure** ran end-to-end target-natively — no Qwen channel, layer, router, or direction was
reused. R0 passed (acc 0.4269 vs majority 0.3546; deterministic replay bit-identical). Discovery
returned **exactly three** bootstrap-stable channels (rfi_tc 948, ca_tc 773, ca_direct 291) — a
**strict subset of Qwen's** (Qwen's validated `ca_rfi` collapses to 26; `tc_rfi` to 1) and a
**match to Phi's**. Two SAKIKO-CA rules replicated across architectures: depth-normalized obs
selection (Mistral L22 = 0.69 vs Qwen L20 = 0.71) and R2 (`cos(DM,PC1)<0.6 → PCA-1`, which fired
on `ca_direct` and won on validation). One rule **failed to transfer**: R1's absolute norm floor
is Qwen-calibrated and rejected every Mistral layer; only the relative norm-ratio is portable.
The intervention produced **large behavioral gains** — locked-test `ca_tc` **Net +62**, manual
3-channel cascade **Net +82 / accuracy 0.4252→0.5748 (+15.0 pts)** — **exceeding** Qwen's
archived pilot (+79 / +14.4 pts). **However, the pre-registered placebo battery refutes the
mechanistic reading:** at cascade level under the *identical* Qwen protocol, `real ≥ all random`
is **FALSE** on Mistral (1/10 random ≥ real; max random +119 > real +82), reverse retains **88%**
(Qwen: 20%), and the marginal-over-random is **+11.9** (Qwen: **+53.8**). Per channel, only
`ca_tc` is direction-specific (95th pct, 1/20); `rfi_tc` is *worse than noise* (random mean +29
vs real +12) and `ca_direct` is non-specific (wrong-layer +24 > real +10). The automatic
8-criteria utility gate therefore admits **zero** channels, and the pilot→multi-seed gate failed,
so **multi-seed was not run**. Mistral's gain is dominated by **gated magnitude** (de-saturating
an 82%-`tool_call` model), not learned direction — Phase 5 contributes a **boundary**, not a
second confirmation.

## 10. Exact claim boundary

See `PHASE5_CLAIM_BOUNDARY.md`. Headline constraints: **nothing here is multi-seed/robust**
(not run); the +82 cascade is **behavioral-only, not direction-specific**, and Arm F is a
**diagnostic**, not the automatic system's output (Arm D is empty); only `ca_tc` carries a
single-seed directional signal, and it fails the gate on collateral redistribution. The
program's quantitative mechanistic story remains **Qwen2.5-7B only** (multi-seed +92 ± 20).
**Do not claim** universal architecture transfer, OOD/prompt/scale generalization, or
cross-model direction transfer.

## 11. Suggested commit message

```
Add Mistral-7B-v0.3 W2C target-native SAKIKO-CA evaluation (Phase 5)

Second-architecture test of SAKIKO-CA. Procedure transfers; causal-specificity
does not. R0 pass (acc .427, 82% tool_call over-call). Discovery: 3 stable
channels (strict subset of Qwen's). Depth-normalized obs (L22=0.69) and R2/PCA-1
replicate; R1's absolute norm floor does not (relative ratio required).
Locked test: ca_tc +62; manual cascade +82 (+15.0 pts) > Qwen's +79 — but
cascade placebo fails real>=all-random (1/10; margin +11.9 vs Qwen +53.8),
reverse retains 88%. Utility gate admits 0 channels; multi-seed not run per
pre-registered gate. Reported as behavioral-only boundary, not confirmation.
```

## 12. Exact manual git commands (for the human — the assistant did NOT run these)

```bash
git status --short

git add final/results/mistral7b_w2c_sakiko_ca
git add scripts/phase5_mistral_lib.py scripts/phase5_mistral_baseline.py \
        scripts/phase5_mistral_extract_acts.py scripts/phase5_mistral_discovery.py \
        scripts/phase5_mistral_pilot.py scripts/phase5_mistral_cascade_placebo.py \
        scripts/phase5_mistral_multiseed.py scripts/phase5_mistral_report.py
# no archived/synthesis files were modified, so nothing else to add

# OPTIONAL (§6 option b) — keep the LFS-routed per-example file local:
# git reset final/results/mistral7b_w2c_sakiko_ca/baseline/mistral7b_w2c_baseline_details.jsonl

git diff --cached --name-status
git diff --cached --stat
git diff --cached --check

GIT_AUTHOR_NAME="the corresponding author" \
GIT_AUTHOR_EMAIL="<anonymised>" \
GIT_COMMITTER_NAME="the corresponding author" \
GIT_COMMITTER_EMAIL="<anonymised>" \
git commit -m "Add Mistral W2C target-native SAKIKO-CA evaluation"

git push origin exp/sakiko-followup-archive
```
