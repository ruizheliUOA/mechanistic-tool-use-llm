# Final model breadth panel

**No model was downloaded, loaded or evaluated. No dataset was touched. No sealed population
was accessed. No scientific outcome exists for either slot.**

## Panel status

`FINAL_MODEL_PANEL_MECHANICALLY_BLOCKED` — **interim mechanical status, not a scientific
outcome.** It describes the machine, not the models. Neither slot has been scientifically
adjudicated and neither may be described as having failed.

### Mechanical history (transparent, not rewritten)

| round | MODEL G block | MODEL Q block |
|---|---|---|
| 1 | `GatedRepoError` 401 | bandwidth 18 KB/s |
| 2 (retry) | 401 unchanged | bandwidth 42 KB/s sustained |
| 3 (access resolved) | **RESOLVED — access PASS** | bandwidth (deprioritised) |

**Correction to round 1 and 2.** The Gemma 401 was **self-inflicted by this audit**, not a
license condition. Setting `HF_HOME=/root/autodl-tmp/hf_cache` relocates `HF_TOKEN_PATH` to
`$HF_HOME/token`, which does not exist, so every Gemma request was sent unauthenticated
(`get_token()` → `False` under that `HF_HOME`, `True` under the default). The earlier 401 is
not evidence about license state in either direction. Fixed by passing `HF_TOKEN` explicitly.

### Round 3 progress

- **Access gate: PASS**, independently re-verified — HTTP 302 → CDN, then 200 on
  `model-00001-of-00004.safetensors` at revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`.
- **Architecture/interface audit: COMPLETE** — `gemma/ARCHITECTURE_INTERFACE_AUDIT.md`.
  42 layers, hidden 3584; derived **L_obs 30, L_inj 24** from the unchanged mapping rule; the
  frozen site has a direct analogue, so **no `ARCHITECTURE_INTERFACE_NO_GO`**. Four
  cross-family differences recorded before inference, including `final_logit_softcapping 30.0`.
- **Download destination corrected** — root fs has 11 GB free, too small for 18.48 GB; moved to
  `/root/autodl-tmp/hf_cache` (32 GB free).
- **Sole remaining blocker: bandwidth.** Measured **7283 B/s** on a Gemma shard, degraded from
  earlier rounds. 18.48 GB → **~29 days**. The §12 smoke test and the entire prospective ladder
  are gated behind the weights arriving.

## Per-model verdicts

**None returnable.** Both slots terminated at pre-inference mechanical gates.

## What blocked each slot

| slot | checkpoint | block |
|---|---|---|
| MODEL G | `google/gemma-2-9b-it` | **`GatedRepoError` 401** — the stored HF token has not accepted the Gemma license. The fallback `gemma-7b-it` returns the same 401. |
| MODEL Q | `Qwen/Qwen3.5-9B` | **network throughput** — 4.22 MB of metadata retrieved, then 0 B/s. Best measured rate on a real weight shard: 18.2 KB/s → **12.3 days** for 19.31 GB. Also: `transformers` 4.51.0 has no `qwen3_5`, and the required upgrade stalled on a 12 MB wheel. |

## Retry (same session, after the first block)

Both blockers were re-tested, not restated.

- **Gemma: unchanged.** `gemma-2-9b-it` and `gemma-7b-it` both still return `GatedRepoError`
  401. This needs a license acceptance on the HuggingFace account owning the token — it cannot
  be resolved from here.
- **Qwen3.5: re-measured, still blocking.** A single-connection burst reached **146 KB/s**, 8×
  the first measurement, so the download was restarted with 8 workers. Over a 5-minute window
  it moved **12.6 MB — 42 KB/s sustained**, in ~12 MB bursts separated by full stalls.
  Parallelism did not raise the aggregate, so the ceiling is total bandwidth. **ETA 127 hours
  (5.3 days)** for the remaining 19.3 GB.
- **Environment: independently fatal.** `pip install transformers==4.57.6` was restarted with
  the correct interpreter and still had not completed a single **12 MB** wheel after ~7 minutes.
  Even with weights in hand, `transformers` 4.51.0 cannot load `qwen3_5`.

The download was left running. If it ever completes, the architecture mapping in
`ARCHITECTURE_MAPPING.md` (L_obs 23, L_inj 18) is ready to use.

Two further blockers were identified but never became binding: **disk** (32 GB free vs 37.79 GB
for both checkpoints) and **VRAM** (18.5–19.3 GB of BF16 weights on a 24 GB card leaves ~5 GB
for the four-mode full-effect gradient stack — a real `HARDWARE_NO_GO` risk that only the §12
smoke test could settle).

## Why no per-model scientific verdict

All eight permitted per-model verdicts assert an adjudication. Nothing was adjudicated. Writing
`MODEL_HARDWARE_NO_GO` or `MODEL_ARCHITECTURE_INTERFACE_NO_GO` would put a scientific-sounding
label on a licensing and bandwidth problem and would permanently misdescribe these settings in
the record.

## What was produced

The frozen panel document, full provenance, the compatibility audit with measured numbers, and
the architecture mapping — including the derived Qwen3.5-9B sites **L_obs 23 / L_inj 18**, from
the frozen normalized-depth rule validated by reproducing Qwen3-8B's committed 26/21. All are
pre-inference records, not results.

## Effect on the manuscript

**None.** Model breadth is unchanged: two Qwen sizes formally, plus Phi-3.5 retrospectively.
Cross-family evidence remains absent. Breadth was already ranked risk 2 and classified
`FIXABLE_BY_WRITING (claim scope)` with `requires_experiment: NO`; that stands.

**A blocked slot is not a negative result.** It must never be written that Gemma or Qwen3.5
failed to replicate.
