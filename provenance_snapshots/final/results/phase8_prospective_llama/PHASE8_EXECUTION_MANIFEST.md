# Phase 8 — Execution Manifest

**Written before any model output.** Records the exact frozen procedure to be executed, the
mechanical (non-scientific) implementation extensions, and the current pause point.

## 1. Frozen inputs (verified, hashed)

- Gate v2 spec + prospective protocol: `GATE_V2_LOCK_MANIFEST.json` → **15/15 artifacts
  unchanged**. HEAD `1ff1f6f` == freeze HEAD.
- Decision rules, verdict branches, and claim-withdrawal triggers: pre-registered in
  `PHASE8_PRE_RESULT_PREDICTIONS.json` (written before any model output).

## 2. Target model — **PENDING HUMAN DECISION** (see `PHASE8_INTEGRITY_AUDIT.md` §3)

- Primary `meta-llama/Llama-3.1-8B-Instruct`: **gated (manual), HTTP 401, no token → access
  failure.**
- Pre-registered fallback `allenai/OLMo-2-1124-7B-Instruct`: ungated, accessible, `olmo2`
  supported by transformers 4.49.0.

The chosen model + revision + SHA will be recorded here **before any inference**, and it will
be verified byte-identical to the official repo (SHA256) as in Phase 5.

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
