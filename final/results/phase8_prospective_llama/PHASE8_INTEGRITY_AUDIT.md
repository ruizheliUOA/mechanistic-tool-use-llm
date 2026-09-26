# Phase 8 — Integrity & Freeze Audit (Part 0)

**Date:** 2026-07-16 · Repo `/root/autodl-tmp/sakiko-followup` · branch
`exp/sakiko-followup-archive` · HEAD `1ff1f6f Add ACEBench generation-based readout
validation`. **No git write/network operation performed. No model loaded or downloaded yet.**

## 1. Repository & freeze verification

| check | result |
|---|---|
| repository path | `/root/autodl-tmp/sakiko-followup` ✓ |
| active branch | `exp/sakiko-followup-archive` ✓ (main untouched) |
| current HEAD | `1ff1f6f` ✓ (== `repo_head_at_freeze` in the lock manifest) |
| tracked/untracked status | 39 files staged (`A`) by the human researcher between phases; 19 untracked. HEAD unchanged (nothing committed). **I did not stage these and will not modify the index.** |
| Phase-7 deliverable count | 27 files present under `final/results/actionability_gate_development/` |
| **GATE_V2_LOCK_MANIFEST.json hashes** | **15 / 15 artifacts unchanged, 0 changed, 0 missing** — gate spec, protocol, all code, and dev evidence are bit-identical to the freeze |

**Documents read in full (Part 0 requirement):** `PHASE7_MECHANISM_DECISION.md`,
`PHASE7_MECHANISM_EXPLORATION.md`, `PHASE7_CLAIM_BOUNDARY.md`,
`ACTIONABILITY_GATE_V2_SPEC.md`, `GATE_V2_LOCK_MANIFEST.json`,
`PROSPECTIVE_LLAMA_PROTOCOL.md`, `NEXT_EXPERIMENT_DECISION_TREE.md`, and the Phase-7
`HUMAN_GIT_HANDOFF.md`. All are self-consistent; the operative decision rule is the frozen
Gate v2 (Stages 0–7) and the locked prospective protocol.

## 2. Is the frozen protocol internally executable?

**Yes — with mechanical model-registration plumbing only; no scientific choice changes.** The
Phase-7 code (`phase7_lib.py`, `phase7_validation_specificity.py`,
`phase7_gate_v2_evaluation.py`) already implements the frozen ρ definition, ρ grid, router,
Stage-4/5 logic, split firewall, and decision output. To run a new architecture it needs:

1. **Model registration** — add the target to the `MODELS` registry (path, layer count, hidden
   size, argmax label order). *Mechanical.*
2. **Baseline + discovery + activation-extraction** for the new model — the Phase-5 scripts
   (`phase5_mistral_baseline.py`, `phase5_mistral_discovery.py`, `phase5_mistral_extract_acts.py`)
   are already parameterized by model; they will be adapted to the new model via a thin driver.
   *Mechanical; reuses the frozen discovery thresholds and normalized-depth layer grid.*
3. **Stage-6 locked-test + rejection-audit driver** — a new script implementing the frozen
   Stage-6 battery (real/reverse/20-random/ungated/wrong-layer) on the test split, run once
   after decisions are frozen and hashed.

**Firewall clarification (important, and consistent with Phase 5).** The frozen Stage-1 scope
filter reports per-split *support counts* including test (`train≥30, val/test≥8`). Computing
those counts requires the model's **baseline predictions** on all 3652 rows — which is exactly
the R0 baseline that Part 1 requires and that firewall rule #5 lists as its *first
precondition* ("baseline validity is established"). The forbidden act is inspecting the
**locked test intervention outcome** (Fixed/Broke/Net under steering) before decisions are
frozen. Therefore: **baseline predictions on the full set are allowed and required; the
intervention/steering evaluation on test rows is firewalled** (enforced in code by
`assert_no_test` on every intervention index set, as in Phase 7) **until
`LLAMA_FROZEN_GATE_DECISIONS.json` is written and hashed.**

**Implementation corrections identified:** none that change a scientific choice. The only
change vs Phase 7 is adding a model to the registry and writing a Stage-6 test driver (which
Phase 7 deliberately did not include because it was development-only). Every such addition is
documented in `PHASE8_EXECUTION_MANIFEST.md` with old/new intent and justification, and none
alters a threshold, grid, estimator, layer rule, or outcome rule.

## 3. Model access — **BLOCKER requiring a human decision**

The frozen protocol names:
- **primary:** `meta-llama/Llama-3.1-8B-Instruct` (license-gated; human supplies access);
- **availability fallback:** `allenai/OLMo-2-1124-7B-Instruct` (ungated; "usable only for
  access failure — never for performance").

**Verified access state (read-only, this session):**

| model | HF API `gated` | config fetch | local copy | HF token available | usable now? |
|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | **manual** | **HTTP 401** (unauthorized) | none | **none** (env unset, `~/.cache/huggingface/token` absent, `HfFolder` empty) | **NO — access failure** |
| allenai/OLMo-2-1124-7B-Instruct | False | reachable via API (sha `470b1fba`) | none | n/a (ungated) | **YES** (transformers 4.49 supports `olmo2`) |

The archived `*_llama32.*` files are a **different** model (Llama-3.2, prior-phase transfer
work), **not** the protocol's Llama-3.1-8B, and are not usable as a substitute.

**This is precisely the "access failure" contingency the frozen protocol pre-registered.**
However, the Phase-8 instructions state explicitly: *"Use the exact Llama model … Do not
substitute a different checkpoint without stopping for human approval."* OLMo is a different
checkpoint. Both conditions point to the same action: **stop and obtain a human decision**
before loading any model. The access failure is a *missing input* (an HF token / accepted
Llama licence) that only the human can supply.

### Decision requested (see the question posed to the researcher)
1. **Provide Llama-3.1-8B access** (accept the licence and supply an HF token out-of-band), so
   the primary model is used exactly as named; **or**
2. **Authorize the pre-registered OLMo-2-1124-7B-Instruct fallback** for genuine access failure
   (scientifically valid: OLMo never influenced the gate's design, so it is a legitimate
   untouched prospective architecture; recorded before any inference, per the protocol).

**No model will be loaded or downloaded until this is resolved.** The choice will be recorded
in `PHASE8_EXECUTION_MANIFEST.md` before any inference, and switching for any *result-related*
reason remains prohibited.

## 4. Status

Part 0 model-independent deliverables (`PHASE8_INTEGRITY_AUDIT.md`,
`PHASE8_EXECUTION_MANIFEST.md`, `PHASE8_PRE_RESULT_PREDICTIONS.json`) are written and hashed
**before** any model output exists. The pipeline is **paused at the model-loading step**
pending the human decision above.
