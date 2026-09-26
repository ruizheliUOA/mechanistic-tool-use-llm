# Phase 8 — Execution Manifest

**Written before any model output.** Records the exact frozen procedure to be executed, the
mechanical (non-scientific) implementation extensions, and the current pause point.

## 1. Frozen inputs (verified, hashed)

- Gate v2 spec + prospective protocol: `GATE_V2_LOCK_MANIFEST.json` → **15/15 artifacts
  unchanged**. HEAD `1ff1f6f` == freeze HEAD.
- Decision rules, verdict branches, and claim-withdrawal triggers: pre-registered in
  `PHASE8_PRE_RESULT_PREDICTIONS.json` (written before any model output).

## 2. Target model — **RESOLVED: the PRIMARY model, exactly as named** (recorded before any inference)

**`meta-llama/Llama-3.1-8B-Instruct`** @ revision `0e9e39f249a16976918f6564b8830bc894c89659`.
The human researcher accepted the licence / supplied access; the weights are local at
`/root/autodl-tmp/models/Llama-3.1-8B-Instruct` (outside the repo).

**The pre-registered OLMo fallback was NOT used** — no substitution occurred.

**Byte-identity verification (SHA256 vs the official `meta-llama` LFS oids fetched from the
official HF API):**

| file | size | official sha256 | verdict |
|---|---|---|---|
| model-00001-of-00004.safetensors | 4,976,698,672 | `2b1879f3…5607668` | ✅ OK |
| model-00002-of-00004.safetensors | 4,999,802,720 | `09d433f6…58f9af15` | ✅ OK |
| model-00003-of-00004.safetensors | 4,915,916,176 | `fc1cdddd…66ae9ea7fa` | ✅ OK |
| model-00004-of-00004.safetensors | 1,168,138,808 | `92ecfe1a…309d6d7b` | ✅ OK |

**RESULT: 4/4 verified, 0 mismatched, 0 missing → BYTE-IDENTICAL TO OFFICIAL.**

**Config (verified):** `model_type=llama`, **num_hidden_layers=32**, **hidden_size=4096**,
num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
max_position_embeddings=131072, torch_dtype=bfloat16.

**Derived frozen grids (computed from the model's own depth L=32 — not copied):**
obs = round({0.40,0.55,0.70,0.85}·32) = **{13, 18, 22, 27}**; inj offsets round({0,−0.06,−0.12}·32)
= {0, −2, −4}; wrong-layer = round(0.15·32) = **L5**.

**Credential handling:** the HF token was supplied by the researcher out-of-band and written
only to the standard HF store `/root/.cache/huggingface/token` (**outside the repository**).
It appears in **no** script, log, report, or committed artifact, and is excluded by the
Phase-8 safety scan.

## 3. Execution order (frozen; = protocol §2) and the driver for each step

| step | frozen procedure | driver (to write; mechanical) | firewall |
|---|---|---|---|
| R0 | avg_logp readout over the 4 W2C candidate texts on all 3652 seed-42 rows; validity gate | `phase8_baseline.py` (parameterized clone of `phase5_mistral_baseline.py`) | baseline predictions on full set allowed (Part-1 requirement) |
| discovery | frozen thresholds 50/15/20, 200-bootstrap, scope filter (train≥30, val/test≥8, ref≥80, own-val-err≥30) | `phase8_discovery.py` (reuses `discover_sakiko_channels.transition_discovery`) | uses baseline predictions only |
| registered predictions | F1=P(gold=runner-up), F2=median(top−gold) on train channel errors; declared expected outcome; **hashed before intervention** | inline in `phase8_discovery.py` | train only |
| extraction | MLP-output activations at obs+inj layers on the normalized-depth grid round({0.40,0.55,0.70,0.85}·L) | `phase8_extract_acts.py` (clone of `phase5_mistral_extract_acts.py`) | caches out of repo |
| geometry R1′/R2 + Stage-4/5 val sweep | ρ∈{0.25,0.5,1,2,4,8}; router LR matched-pred; real+reverse+20 random seed-block 2000+k; interior-ρ rule | `phase8_gate_eval.py` (reuses `phase7_lib` + `phase7_validation_specificity` logic, new model registered) | **`assert_no_test` on every intervention index set** |
| freeze decisions | write + hash `LLAMA_FROZEN_GATE_DECISIONS.json` (ADMIT/REJECT/INCONCLUSIVE per channel + machine-checkable reason) | `phase8_gate_eval.py` | — |
| **Stage 6 (Part 4)** | one-shot locked test on test rows; real/reverse/20-random (1000+k)/ungated/wrong-layer for admits; bounded rejection audit (≤2 specificity-rejects) | `phase8_locked_test.py` (**new**; runs ONLY after decisions hashed) | test intervention allowed **only after** the hash exists |
| multiseed (Part 5) | only if pilot admission rule holds | `phase8_multiseed.py` (conditional) | — |
| OOD (Part 6) | only after multiseed; feasibility check on W2C metadata first | `OOD_PROTOCOL.md` + driver (conditional) | — |

## 4. Implementation corrections / extensions (none change a scientific choice)

1. **Model registration** — add the chosen model to the `MODELS` registry (path, `num_hidden_layers`,
   `hidden_size`, argmax `label_order`). *Mechanical; the label vocabulary and scoring are the
   frozen W2C ones.*
2. **Stage-6 locked-test driver** (`phase8_locked_test.py`) — Phase 7 was development-only and
   did not include a test driver; this implements the frozen Stage-6 battery exactly (same
   controls, same z≥2/≤5%/real>reverse/Broke≤Fixed/2/residual−25% rules, seed block 1000+k).
   *New code, but every threshold and control is taken verbatim from the frozen spec.*
3. **Firewall reconciliation** — baseline predictions on the full 3652-row set are computed
   (required for R0 and discovery support counts); the intervention/steering evaluation on test
   is `assert_no_test`-guarded until `LLAMA_FROZEN_GATE_DECISIONS.json` is hashed. *Consistent
   with Phase 5 and with firewall rule #5 (baseline validity is its first precondition).*

No threshold, ρ value, estimator, layer rule, router family, or outcome rule is altered. Any
deviation discovered mid-run that would change a scientific choice ⇒ stop, verdict F
(APPARATUS FAILURE), report.

## 5. Current status

**PAUSED at the model-loading step**, pending the human model-access decision. Part-0
deliverables written and hashed below. No model loaded, downloaded, or inferred; no test row
touched; no git write.

## 6. Hashes of the Part-0 pre-result files (integrity anchor)

_(SHA256; recorded so the human can confirm these were written before any model output.)_
See `phase8_part0_hashes.json` (generated alongside this manifest).
