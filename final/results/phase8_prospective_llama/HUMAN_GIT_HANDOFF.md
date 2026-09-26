# Phase 8 — Human Git Handoff

> The assistant performed **no** git write and **no** git network operation. Nothing was staged,
> committed, or pushed.

## 1. Branch and HEAD

- **Branch:** `exp/sakiko-followup-archive` (main untouched)
- **HEAD:** `16e1e8c Add Mistral diagnostics and Actionability Gate development`
  (you committed Phase 5–7 between phases; Phase 8 began from the Gate-v2 freeze at `1ff1f6f`
  and the lock manifest verified **15/15 unchanged**, so the freeze is intact across that commit)

## 2. Files created (Phase 8) — all safe to commit

**Reports (`final/results/phase8_prospective_llama/`):**
```
PHASE8_INTEGRITY_AUDIT.md              PHASE8_EXECUTION_MANIFEST.md   (M: model choice recorded)
PHASE8_PRE_RESULT_PREDICTIONS.json     phase8_part0_hashes.json
LLAMA_R0_REPORT.md                     LLAMA_CHANNEL_TOPOLOGY_REPORT.md
LLAMA_FROZEN_GATE_DECISIONS.json       LLAMA_FROZEN_GATE_DECISIONS.sha256
registered_predictions.json            registered_predictions.sha256
LLAMA_PROSPECTIVE_RESULTS.{md,json,csv}
ACTIONABILITY_ATLAS.{md,csv}           ICLR_REVIEWER_ATTACK_MATRIX.md
PHASE8_CLAIM_BOUNDARY.md               OOD_PROTOCOL.md
PHASE8_REPRODUCIBILITY_AND_SAFETY_CHECKLIST.md   HUMAN_GIT_HANDOFF.md
llama_r0_summary.json  llama_confusion_matrix.csv  llama_error_transition_matrix.csv
llama_channel_discovery.{json,csv}  llama_rho_curves.json  llama_activation_extraction.json
figures/fig1_cross_architecture_topology.{png,pdf}
figures/fig2_llama_rho_curves.{png,pdf}
figures/fig3_detectability_vs_actionability.{png,pdf}
figures/fig4_gate_decisions_vs_truth.{png,pdf}
```
**Scripts (`scripts/`):** `phase8_lib.py`, `phase8_baseline.py`, `phase8_discovery.py`,
`phase8_extract_acts.py`, `phase8_gate_eval.py`, `phase8_locked_test.py`, `phase8_report.py`

## 3. Files modified

- `final/results/phase8_prospective_llama/PHASE8_EXECUTION_MANIFEST.md` — updated **before any
  inference** to record the resolved model + SHA verification (was "PENDING HUMAN DECISION").
- **No Phase-5/6/7 file was modified.** No archived result, spec, or lock manifest was touched.
  Gate v2 remains byte-identical to its freeze (15/15 verified).

## 4. Files intentionally excluded (never commit)

| path | size | why |
|---|---|---|
| `/root/autodl-tmp/models/Llama-3.1-8B-Instruct/` | 15 GB | model weights (gated licence) |
| `/root/autodl-tmp/phase8_cache/acts_L*.npy` (12) | ~660 MB | activation caches (regenerable) |
| `/root/autodl-tmp/phase8_cache/*_baseline_details.jsonl` | ~2 MB | per-example details |
| `/root/autodl-tmp/phase8_cache/*.log`, `phase8_*_state.json` | small | run logs / resumable state |
| `/root/.cache/huggingface/token` | — | **your HF token — outside the repo, never committed** |

## 5. Model & cache locations

Weights: `/root/autodl-tmp/models/Llama-3.1-8B-Instruct` (outside repo),
`meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`,
**4/4 shards SHA256-verified byte-identical to official**.
Caches: `/root/autodl-tmp/phase8_cache` (687 MB, outside repo).

## 6. Large-file scan

**No file > 20 MB** in the Phase-8 set. Largest: `figures/fig2_llama_rho_curves.png` (112 KB).
No `.npy` / `.safetensors` / `.pt` / `.bin` in the repo. `git diff --check` clean.

## 7. Secret / privacy scan

- **HF token scan: CLEAN.** Both tokens you pasted were scanned for as literals *and* via
  `hf_[A-Za-z0-9]{30,}` across every Phase-8 file and script → **zero matches**. The token lives
  only at `/root/.cache/huggingface/token`, outside the repository.
- **⚠ ACTION FOR YOU: rotate/revoke both tokens.** They were pasted into a chat transcript, so
  they should be considered exposed regardless of what the repo contains.
- No other credentials, no proxy address, no personal email, no HF username in committed files.

## 8. Safety status

**PASS** — see `PHASE8_REPRODUCIBILITY_AND_SAFETY_CHECKLIST.md`.

## 9. Mechanism verdict

**B — PROSPECTIVE REJECTOR SUPPORTED, ADMISSION UNRESOLVED.**
Gate v2, frozen and hash-verified, admitted **0** of 3 Llama channels (all boundary-seeking).
The bounded, deliberately adversarial rejection audit tested the 2 rejections most likely to be
wrong: **both DENIED** → **TP 0 / FP 0 / TN 2 / FN 0**, rejection precision **1.0**, zero F-1 and
F-2 events. Pre-registered fragility predictions correct **2/2**. Rejected channels had Nets
+16…+23 and AUCs 0.84–0.96 — again showing Net and AUC do not establish mechanism.

## 10. Gate-v2 status

**Unchanged and still frozen.** Its *rejection* half now has prospective support on an untouched
architecture; its *admission* half remains **unvalidated** (no admission existed to test).
Report it as a **reliable rejector, not a validated admitter**. No Gate v3 was created.

## 11. Prospective protocol status

Executed in full for the pilot. **Multiseed: NOT RUN** (frozen prerequisite — an admitted,
confirmed channel — unmet). **OOD: NOT RUN** (gated behind multiseed); feasibility established
and design frozen in `OOD_PROTOCOL.md`.

## 12. Suggested commit message

```
Add Phase 8: prospective Actionability-Gate validation on Llama-3.1-8B

Frozen Gate v2 (lock manifest 15/15 verified) applied to an untouched
architecture. Llama-3.1-8B-Instruct @0e9e39f2, 4/4 shards SHA-verified
byte-identical to official. R0 PASS (acc .4403, +.086 over majority,
replay 40/40). Discovery: 3 eligible channels; obs L22 = 0.688 depth
(third architecture to converge). Gate: 0 ADMIT / 3 REJECT, all
boundary-seeking; decisions hashed before any test access.

Bounded adversarial rejection audit (slots given to the most likely
over-rejections): both DENIED -> TP0/FP0/TN2/FN0, rejection precision
1.0, zero F-1/F-2. Pre-registered fragility predictions correct 2/2.
Rejected channels had Net +16..+23, AUC .84-.96, reverse == real, and a
wrong-layer control 3x the real effect -- Net and AUC again fail to
establish mechanism. New taxonomy entry: direction-specific
redistribution (ca_tc val z=5.46 at interior rho, fails not_redirection;
audit confirms rejection, test z=0.879).

Verdict B: prospective rejector supported, admission unresolved.
Multiseed and OOD correctly not run (prerequisites unmet).
```

## 13. Commands for you to inspect and commit manually (assistant did NOT run these)

```bash
git status --short

# inspect first
git diff final/results/phase8_prospective_llama/PHASE8_EXECUTION_MANIFEST.md

git add final/results/phase8_prospective_llama
git add scripts/phase8_lib.py scripts/phase8_baseline.py scripts/phase8_discovery.py \
        scripts/phase8_extract_acts.py scripts/phase8_gate_eval.py \
        scripts/phase8_locked_test.py scripts/phase8_report.py

git diff --cached --name-status
git diff --cached --stat
git diff --cached --check

GIT_AUTHOR_NAME="the corresponding author" \
GIT_AUTHOR_EMAIL="<anonymised>" \
GIT_COMMITTER_NAME="the corresponding author" \
GIT_COMMITTER_EMAIL="<anonymised>" \
git commit -m "Add Phase 8 prospective Actionability-Gate validation on Llama-3.1-8B"

git push origin exp/sakiko-followup-archive
```

**Before pushing:** confirm no `*.npy`, `*.safetensors`, or token appears in
`git diff --cached --name-status`, and **revoke the two HF tokens** (§7).
