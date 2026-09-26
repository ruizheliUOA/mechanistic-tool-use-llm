# Phase 8 — Reproducibility & Safety Checklist

## 1. Reproducibility

**Model.** `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`,
bf16, license-gated (access supplied by the researcher). **All 4 weight shards SHA256-verified
byte-identical to the official Meta LFS oids**; verified twice (after download, and again after
the disk cleanup below). Weights at `/root/autodl-tmp/models/Llama-3.1-8B-Instruct` — **outside
the repository**.

**Environment.** conda `sakiko-phase3`: Python 3.10.8, torch 2.1.2+cu121, transformers 4.49.0,
sklearn 1.7.2, numpy 1.26.3; GPU RTX 4090 D (24 GB), bf16.

**Data.** `nvidia/When2Call` test/mcq, N=3652, seed-42 order (local raw jsonl == HF order,
verified in Phase 5). Fixed split: train 2556 / val 548 / test 548, pairwise intersections 0.

**Determinism.** R0 replay bit-identical on 40 rows (predictions and avg_logp). Routers/PCA use
`random_state=42`. Random directions: fixed seed blocks (validation 2000+k, locked test
1000+k), disjoint and recorded in every row.

**Regeneration order:**
```
python scripts/phase8_baseline.py --replay 40     # R0 (~32 min GPU)
python scripts/phase8_discovery.py                # discovery + registered predictions (CPU)
python scripts/phase8_extract_acts.py             # 12 layers, train+val only (~12 min GPU)
python scripts/phase8_gate_eval.py                # Gate v2 Stage 1-5 -> frozen decisions (~5.5 h GPU)
python scripts/phase8_locked_test.py              # ONE-SHOT Stage-6 + bounded audit (~31 min GPU)
python scripts/phase8_report.py                   # atlas + figures (CPU)
```
Observed wall-clock: R0 32 min · extraction 12 min · gate eval 5 h 35 m · locked test 31 min.
Gate eval and locked test are resumable via out-of-repo state files.

## 2. Test-split firewall (the central integrity control)

- `assert_no_test()` guards **every** intervention index set (direction, router fit, router
  scoring, metrics, evaluation). Unit-tested: a single test index raises.
- Activations extracted for **train+val only** (3104 rows). Test activations were never cached.
- `phase8_locked_test.py` **refuses to run** unless `LLAMA_FROZEN_GATE_DECISIONS.json` exists
  **and** its SHA256 matches the recorded hash — self-tested before the decisions existed
  (`REFUSING: frozen gate decisions + hash must exist before any test access`) and hash-verified
  at run time (`45a3f8a7…`).
- Baseline predictions **were** computed on all 3652 rows: this is the R0 requirement and
  firewall rule #5's own first precondition. No configuration, threshold, or ρ was selected from
  a test metric.
- **No test arm was re-run after seeing its outcome. No configuration search followed the
  negative result.**

## 3. Secret / privacy scan

- **HF tokens:** the researcher supplied two tokens in chat. Scanned every Phase-8 repo file and
  script for both literals and for the generic `hf_[A-Za-z0-9]{30,}` pattern → **CLEAN, zero
  matches**. The active token exists only at `/root/.cache/huggingface/token`
  (**outside the repository**, mode 0600). **Recommendation to the researcher: rotate/revoke
  both tokens, since they were pasted into a chat transcript.**
- **Other credentials:** no `sk-…`, `ghp_…`, private keys, bearer headers, or passwords.
- **PII / infrastructure:** no personal email, no proxy host:port, no HF username in any
  committed file.
- **Model weights / activation caches / large logs:** none in the repo (see §4).

## 4. Large-file scan

No file > 20 MB in the Phase-8 staged set. Largest: `figures/fig2_llama_rho_curves.png`
(112 KB), `fig3_…png` (92 KB), `llama_rho_curves.json` (48 KB).
Out-of-repo (correctly excluded): `/root/autodl-tmp/models/Llama-3.1-8B-Instruct` (15 GB),
`/root/autodl-tmp/phase8_cache` (687 MB: 12 activation `.npy` + baseline details + logs).

**Disk incident (disclosed):** `/root/autodl-tmp` hit 100% during extraction. Cause: a **6.8 GB
redundant HF download blob cache** inside the model directory, duplicating the already-verified
shards. It was removed; the weights were then **re-verified byte-identical** (4/4). No
scientific artifact was affected.

## 5. Scientific-integrity checks

- Gate v2 **not modified**; no Gate v3; no threshold/grid/estimator/layer/outcome-rule change.
- Lock manifest verified **15/15 unchanged** before the model was touched.
- Pre-result predictions + execution manifest **hashed before any model output**
  (`phase8_part0_hashes.json`); per-channel fragility predictions **hashed before activation
  extraction**.
- Frozen decisions hashed before test access; **one shot per arm**.
- **Rejection audit was adversarial**: the 2 slots went to the rejections most likely to be
  *wrong* (highest interior z, highest Net), not the most obviously correct one. Disclosed in
  `LLAMA_PROSPECTIVE_RESULTS.md` §3; this selection can only make the gate look worse.
- Multiseed and OOD **not run** (pre-registered prerequisites unmet) — despite rejected
  channels showing Nets of +16…+23 that a Net-driven programme would have chased.
- Negative/rejection results reported in full; no Mistral rescue attempted.

## 6. Implementation corrections (none change a scientific choice)

1. **Model registration + Stage-6 driver** — mechanical extensions (Phase 7 was
   development-only and had no test driver). Every threshold copied verbatim from the frozen spec.
2. **Stage-4 scout** — the sweep evaluates real-only over method×inj×ρ, then runs the full
   6-ρ battery (real + reverse + 20 randoms) on the selected family. This implements the frozen
   spec's "cap: top-3 by val Net" compute bound; without it the run was ~12–18 GPU-h. The ρ
   grid, thresholds, and criteria are unchanged.
3. **Audit-selection rule** — the protocol says "up to 2"; the rule used is documented in §5
   and is adversarial to the gate.

---

**VERDICT: PASS**
