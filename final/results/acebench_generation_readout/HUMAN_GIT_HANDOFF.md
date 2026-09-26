# Phase 4 — Step 16 · Human Git Handoff

**I have executed no git write operation, and will not.** Only read-only git (`status`, `ls-files`,
`check-ignore`, `branch`, `rev-parse`) was used. The commands below are **shown for you to run**;
staging, committing, and pushing are yours.

## 0. Starting state (verified read-only, just now)

| | |
|---|---|
| Branch | `exp/sakiko-followup-archive` |
| HEAD | `b9f5aae` |
| Modified tracked files | **none** — Phase 4 changed nothing that already existed |
| New (untracked) | 1 directory (**27 files, 596 KB**, all text/PNG) + 8 scripts |

Safety scan: `PHASE4_REPRODUCIBILITY_AND_SAFETY_CHECKLIST.md` → **PASS**.

```bash
cd /root/autodl-tmp/sakiko-followup
git branch --show-current     # expect: exp/sakiko-followup-archive
git rev-parse --short HEAD    # expect: b9f5aae
git status --porcelain        # expect: only the "??" lines below, nothing " M"
```
```
?? final/results/acebench_generation_readout/
?? scripts/acebench_channel_discovery_feasibility.py
?? scripts/acebench_generation_baseline.py
?? scripts/acebench_inspect_schema.py
?? scripts/acebench_lock_manifest.py
?? scripts/acebench_paraphrase_check.py
?? scripts/acebench_parse_outputs.py
?? scripts/acebench_readout_audit.py
?? scripts/make_acebench_readout_figures.py
```

## 1. Stage the Phase-4 output package (one dir)
```bash
git add final/results/acebench_generation_readout
```

## 2. Stage the 8 Phase-4 scripts
```bash
git add scripts/acebench_inspect_schema.py \
        scripts/acebench_parse_outputs.py \
        scripts/acebench_readout_audit.py \
        scripts/acebench_generation_baseline.py \
        scripts/acebench_lock_manifest.py \
        scripts/acebench_channel_discovery_feasibility.py \
        scripts/acebench_paraphrase_check.py \
        scripts/make_acebench_readout_figures.py
```

## 3. Review BEFORE committing (do not skip — last checkpoint before history changes)
```bash
git diff --cached --name-status   # every line should be A under the two paths above — nothing else
git diff --cached --stat
git diff --cached --check         # expect: no output (no whitespace/CRLF errors)
```
**Guard.** Nothing matching `.cache/`, `*.safetensors`, `*.npy`, `*.pt`, or any raw-generation
`generation_details_*.jsonl` / `audit_sample.jsonl` may appear. All are gitignored
(`git check-ignore -v .cache/acebench_phase4/generation_details_full.jsonl` proves it). If one
somehow appears: `git reset HEAD <path>`.

## 4. Commit under your identity
```bash
GIT_AUTHOR_NAME="the corresponding author" \
GIT_AUTHOR_EMAIL="<anonymised>" \
GIT_COMMITTER_NAME="the corresponding author" \
GIT_COMMITTER_EMAIL="<anonymised>" \
git commit -m "Add Phase 4 ACEBench generation-readout validation study (outcome: DATASET NO-GO)"
```

<details>
<summary>Longer commit body, if you prefer</summary>

```
Add Phase 4 ACEBench generation-readout validation study

Redesigns the ACEBench readout as generation-based (the prior candidate-scoring
readout failed R0 by collapsing: acc 0.196, 74.5% one class) and validates it
under a protocol locked before the run (READOUT_LOCK_MANIFEST.json, 8 sha256).

Readout works: acc 0.756, macro-F1 0.687, bal-acc 0.621, max class share 0.675,
UNKNOWN 7.25%, replay determinism 100%, parser mislabel 0.98% (102-row audit).
136/137 native-label errors come from official/canonical parse rules, so the
errors are model-attributable, not parser artifacts.

R0 verdict: CONDITIONAL PASS -- 9 of 10 criteria pass; R0.9 (channel support)
fails with 0 of >=2 transitions clearing train>=30/val>=5/test>=5 (bootstrap
stability 0.31/0.215/0.0). Cause is structural: ACEBench special modes hold only
100 examples each, so per-split channel support is arithmetically out of reach --
as the protocol predicted pre-hoc.

Pre-registered paraphrase eligibility check FAILS (0.56 vs 0.80 gate; 0.79
format-insensitive; 0.65 latent-call). Restraint modes are wording-fragile
(ask_user 0.36) while calling is stable (0.84); a semantically equivalent
rewording moves the model +22 points toward calling.

Outcome D (DATASET NO-GO): retire ACEBench from the quantitative intervention
line; no transfer claim is made in either direction. Next step: second-model
architecture validation on W2C, which has adequate channel support.

No intervention was run. No threshold moved after results were seen.
```
</details>

## 5. Push
```bash
git push origin exp/sakiko-followup-archive
```

## 6. Verify after push
```bash
git log --oneline -1     # your new commit on top
git status --short       # clean
```

---

## 7. Things you should decide (I deliberately did not)

1. **`.gitignore` gap.** `*.pth`, `*.npy`, `*.pkl`, `clash/` are missing. Phase 4 produces **none** of
   these, and 44 `.npy` files are **already tracked** from earlier phases — so adding the patterns
   would be a no-op for them (gitignore never untracks tracked files) and would only *look* like
   protection. Left to you; it is pre-existing repo hygiene, not a Phase-4 change. Future-only fix:
   append those four patterns.
2. **Synthesis updates not made.** Unlike Phase 3, I did not touch
   `research_synthesis/{CURRENT_STATUS_AND_GAPS,CLAIM_EVIDENCE_BOUNDARY_MATRIX,NEXT_EXPERIMENT_ROADMAP}`
   — the Phase-4 brief did not ask for them, and this phase's result (a NO-GO with no new claim)
   changes the roadmap's *priority*, not its claim set. If you want the boundary matrix to record
   "ACEBench: readout valid, intervention not measurable", say so and I will draft it as a separate,
   reviewable change.
3. **Whether to keep `scripts/acebench_paraphrase_check.py`** in the commit. It is the evidence behind
   the eligibility failure, so I recommend keeping it.

## 8. What is deliberately NOT in this commit
- Raw generations (`generation_details_{full,paraphrase,smoke}.jsonl`), the 102-row
  `audit_sample.jsonl`, the regenerated 800-row dataset, and the model weights — all gitignored under
  `.cache/`, all regenerable from the documented commands. They embed third-party ACEBench dataset
  text; the locked artifact policy keeps them out of the repo.
- Any intervention artifact — **none exists**; no activations, routers, directions, or SAKIKO runs
  were produced this phase.
