# Phase 5 — Reproducibility & Safety Checklist

**Repo:** `/root/autodl-tmp/sakiko-followup` · branch `exp/sakiko-followup-archive`
(unchanged; no git write/network operations were performed by the assistant).

> **Status: finalized 2026-07-15** after all Phase-5 outputs were written; scans below are the
> final re-run.

## 1. Reproducibility

**Model.** `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71`,
bf16, HF sharded format. SHA256 of all 3 weight shards + `tokenizer.model` verified equal to
the official HF LFS oids (see PHASE5_ENVIRONMENT_AND_PREFLIGHT). Weights live **outside the
repo** at `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3` and are never staged.

**Environment.** conda `sakiko-phase3`: Python 3.10.8, torch 2.1.2+cu121, transformers 4.49.0,
tokenizers 0.21.4, datasets 3.2.0, sklearn 1.7.2, numpy 1.26.3; GPU RTX 4090 D (24 GB).

**Data.** `nvidia/When2Call` test/mcq (N=3652), seed-42 shuffle; the local raw jsonl
reproduces the HF order bit-identically (verified). Fixed split indices
`final/results/splits/` (train 2556 / val 548 / test 548); per-seed splits are deterministic
`StratifiedShuffleSplit` on Mistral `etype`, `random_state=seed`.

**Determinism.** avg_logp scoring is deterministic; R0 replay was **bit-identical** on 40
samples (predictions and scores). All routers/PCA use fixed `random_state=42`.

**Regeneration order (from the repo root, env `sakiko-phase3`):**
```
python scripts/phase5_mistral_baseline.py --replay 40      # R0 baseline (~35 min GPU)
python scripts/phase5_mistral_discovery.py                 # channel discovery (CPU)
python scripts/phase5_mistral_extract_acts.py              # obs-layer activations (~8 min GPU)
python scripts/phase5_mistral_pilot.py --stage geo         # geometry / R1 / R2 (CPU)
python scripts/phase5_mistral_pilot.py --stage run         # val sweep, locked tests, controls, gate (~2h GPU)
python scripts/phase5_mistral_cascade_placebo.py           # Arm-F cascade placebo, Qwen-matched (~35 min GPU)
python scripts/phase5_mistral_multiseed.py                 # NOT RUN — pilot gate failed (see final report §8)
python scripts/phase5_mistral_report.py                    # cross-model synthesis (CPU)
```
Observed wall-clock on 1× RTX 4090 D: baseline 35 min, extraction 8 min, pilot 2 h 05 m,
cascade placebo 35 min. `phase5_mistral_pilot.py --stage run` is resumable via
`pilot/_state.json`.
Activation caches and per-example details are regenerable and kept **out of the repo**
(`/root/autodl-tmp/phase5_mistral_cache/`).

## 2. Artifact control

**Kept out of git (never staged):**
- Model weights: `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3/*` (14 GB).
- Activation caches: `/root/autodl-tmp/phase5_mistral_cache/acts_L*.npy` (4 × 57 MB).
- Per-example baseline details full copy: `/root/autodl-tmp/phase5_mistral_cache/…jsonl`.
- HF download cache: `/root/autodl-tmp/hf_home`, and the model dir's `.cache/`.
- Proxy/clash config, shell history, task-runner logs (`/root/autodl-tmp/phase5_mistral_cache/*.log`).

**Committed (small, text):** the `final/results/mistral7b_w2c_sakiko_ca/` reports + JSON/CSV,
and `scripts/phase5_mistral_*.py`. One LFS-routed file, `baseline/…_details.jsonl`
(~1.5 MB), is mirrored into the results tree; the human decides commit-via-LFS vs local-only
(see HUMAN_GIT_HANDOFF).

## 3. Secret / privacy scan

Method: regex scan of all new `scripts/phase5_*.py` and everything under
`final/results/mistral7b_w2c_sakiko_ca/` for api keys, tokens (`hf_…`, `sk-…`, `ghp_…`),
passwords, bearer/cookie headers, private keys, `.env`, the local proxy host:port literal,
and personal identifiers (personal email domains, the researcher's name/email, prior
workstation home paths).

- No credentials, tokens, or private keys. **No HF token was used or stored** (the model repo
  is public/ungated; the download required no authentication).
- **No proxy address in any committed file.** The preflight documents proxy *behavior* (it
  stalled the LFS CDN) without the host:port. *(One earlier draft of this checklist quoted the
  literal address while describing the scan; it was removed — flagged here for transparency.)*
- No personal identifiers in committed Phase-5 files, with one **intended** exception: the
  researcher's git author name/email appear inside the git command block of
  `HUMAN_GIT_HANDOFF.md`, as explicitly specified by the Phase-5 brief.
- Absolute paths present are non-sensitive workstation paths (`/root/autodl-tmp/…`), documented.

## 4. Large-file scan (re-run 2026-07-15, final)

`find … -size +20M` over all staged-candidate paths → **none**. Largest Phase-5 files:

| size | file |
|---|---|
| 1.05 MB | `baseline/mistral7b_w2c_baseline_details.jsonl` (LFS-routed; see §2) |
| 0.12 MB | `figures/fig1_specificity_qwen_vs_mistral.png` |
| 0.10 MB | `pilot/_state.json` (resumable state; small, text) |

Model weights (14 GB) and activation caches (4 × 57 MB `.npy`) are **outside the repo** and
were confirmed absent from the staged tree. `git diff --check` → clean.

*Note (pre-existing, not Phase 5):* `sakiko_v3/cache/acts_L*.npy` exist in the archive and are
already tracked from earlier phases — untouched by this work.

## 5. Git safety

No `git add/commit/push/fetch/pull/switch/checkout/branch/merge/reset/clean/config` was run.
Only read-only `git status/branch/log/check-attr/diff` were used. Branch unchanged
(`exp/sakiko-followup-archive`); `main` untouched.

## 6. Scientific-integrity checks

- **Protocol locked before test inspection** (`PHASE5_PROTOCOL.md` written pre-inference).
- **All tuning on train/val only**; each locked-test arm run exactly once.
- **No test-driven tuning**: the α grid was *not* extended even though α saturated at the grid
  maximum (6.0) — extending post-hoc would have been test-driven. Reported as a limitation.
- **One protocol deviation, documented and pre-test:** R1's absolute norm floor (Qwen-calibrated,
  `norm < 5.0`) was replaced by the model-agnostic **relative norm-ratio** criterion, because the
  absolute constant rejected every layer on Mistral (activation norms 5–10× smaller). Decided on
  train-side geometry only, before any test run; recorded in `PHASE5_PROTOCOL.md` item 18 and
  the final report §3.
- **Pre-registered gate honored:** the pilot→multi-seed gate failed, so **multi-seed was not
  run** and no multi-seed numbers are reported — despite the headline Net being large.
- **Negative results reported in full** (2/3 channels non-specific; automatic system admits
  zero channels; cascade fails `real ≥ all random`).

---

**VERDICT: PASS**
