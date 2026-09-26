# Claim boundary

## Verified in this task

- Repository gate: HEAD advanced `59b133a3` → `f5ba3a47`; the two previously untracked
  exploration packages are now tracked (33 and 23 files). All seven §3 packages verified
  intact and unmodified; five carry self-verifying `HASHES.json` (30/30, 20/20, 32/32, 18/18,
  22/22, 22/22 all ok), the Qwen3-8B ADMIT package predates that convention.
- `google/gemma-2-9b-it` and `google/gemma-7b-it` both return `GatedRepoError` 401 to the
  stored token.
- `Qwen/Qwen3.5-9B` is ungated; its config was retrieved and is recorded verbatim in the freeze.
- Measured throughput on a real weight shard: **18179 B/s** direct, **9151 B/s** via
  hf-mirror; HF metadata API 5126 B/s; pypi mirror 16916 B/s.
- `transformers` 4.51.0 supports `gemma2`, `gemma`, `qwen3`, `qwen3_moe`; it does **not**
  support `qwen3_5` or `qwen3_next`.
- Derived from the frozen mapping rule, not executed: Qwen3.5-9B L_obs **23**, L_inj **18**.
  The rule was validated by reproducing Qwen3-8B's committed 26/21.

## Must not be claimed from this package

- That Gemma or Qwen3.5 failed, declined, lacked eligible channels, or did not replicate.
  **Nothing was measured.**
- Any per-model scientific verdict from the eight permitted values.
- That cross-family breadth was tested. It was not.
- `ARCHITECTURE_INTERFACE_NO_GO` for Qwen3.5 — the hybrid linear/full-attention stack raises
  real questions recorded in `ARCHITECTURE_MAPPING.md`, but the model was never loaded and the
  frozen site was never shown to lack an analogue.
- That the derived L_obs 23 / L_inj 18 were validated. They are arithmetic from a frozen rule.

## Mandatory disclosures

- The `post-commit` hook rewrote the supplied commit message to "audit: archive final readiness
  and preservation analyses" and pushed to `origin` automatically. The push was not requested.
- 4.22 MB of Qwen3.5-9B config/tokenizer metadata was retrieved. Prior to this task the
  checkpoint was genuinely untouched (empty cache stub, 0 blobs); the long-standing constraint
  was lifted by explicit authorisation for this panel.
- A `env_qwen35_final_panel` venv was created and deleted after the transformers install
  stalled. The historical Qwen3 formal environment was **not** modified.

## Unchanged by this task

Every claim, limitation and prohibition in
`research_exploration/iclr_final_submission_readiness_audit_v2/CLAIM_BOUNDARY.md` stands
exactly as written.
