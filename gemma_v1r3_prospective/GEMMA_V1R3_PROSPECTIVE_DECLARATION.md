# Gemma × V1R3 — prospective declaration (baseline-only feasibility screen)

**New and separate.** This declaration does not modify, reinterpret, or reference
the historical V1R3/Qwen3 declaration (`READY_FOR_QWEN3_BASELINE_ONLY_SCREEN`),
which stands unchanged. Historical Gemma × When2Call remains
`GEMMA_FORMAL_DECLINE` permanently and is not in scope here.

Declared: 2026-08-17. Status: **frozen, pending human gold gate.**

---

## 1. No prior Gemma outcome on V1R3 — VERIFIED

| Check | Result |
|---|---|
| `grep -rli "gemma"` over the package | 0 hits |
| `forbidden_access_confirmations.model_loaded` | `False` |
| `forbidden_access_confirmations.inference_run` | `False` |
| `forbidden_access_confirmations.gpu_used` | `False` |
| `forbidden_access_confirmations.weights_downloaded` | `False` |
| `forbidden_access_confirmations.activation_or_router_run` | `False` |
| `forbidden_access_confirmations.scientific_endpoint_evaluated` | `False` |
| `forbidden_access_confirmations.test_set_accessed` | `False` |

**Gemma has never observed a single row of V1R3.** The population is prospective
for Gemma in the strict sense.

## 2. Human gold / adjudication gate — OUTSTANDING, blocking

State of labelling, verified from `pilot_programmatic_gold.jsonl`:

- `gold_mode` **is populated**: 36 × `tool_call`, 36 × `request_for_info`,
  36 × `cannot_answer`, 36 × `direct_answer`
- `gold_label_source`: `programmatic_deterministic_derivation`
- `gold_engine_version`: `gold_engine_v1.2-v1r3`
- `annotator_1_label`: `None` × 144
- `annotator_2_label`: `None` × 144
- `adjudicated_label`: `None` × 144

Gold is computable and deterministic. **No human has ever labelled or adjudicated
a row.** This is the gate that made the previous NO_GO, and it cannot be
discharged by any automated process — including by me. Substituting an AI pass
here is precisely the defect being repaired.

Three routes are prepared in this directory; **the human team chooses one.**

| Route | File | Effort | Evidential strength |
|---|---|---|---|
| **A — blinded annotation** | `ADJUDICATION_A_BLINDED.csv` | ~3–4 h | **Strongest.** 144 rows in shuffled order, no gold shown. Agreement with programmatic gold is then a genuine measurement. |
| **B — anchored review** | `ADJUDICATION_B_REVIEW.csv` | ~1–1.5 h | **Weaker.** Row + gold + derivation shown; adjudicator confirms or overrides. Anchoring bias is real and must be disclosed in the paper. |
| **C — priority subset** | `ADJUDICATION_PRIORITY_ROWS.json` | ~20 min | **Weakest.** 19 rows that the superseded pass-1 audit either mode-disagreed on (8) or flagged as `another_mode_defensible` (19 total union). A targeted spot-check, not a gate. |

Route A is the only one that supports an unqualified "human-validated gold"
statement. B and C must be described in the paper as what they are.

Disclosure worth carrying forward regardless of route: the **pre-revision** audit
pass (`ai_audit_results_pass_1.jsonl`) disagreed with gold on 8 rows and flagged
19 as having a defensible alternative mode. After the V1R3 revision, pass A shows
`inferred_mode` matching gold 144/144 and `alternative_mode_defensible: False`
144/144. The repair worked, but it was a same-family audit repairing a
same-family draft.

## 3. Model revision and readout — FROZEN

| Item | Frozen value |
|---|---|
| Model | `google/gemma-2-9b-it` |
| Weights | present at `/root/autodl-tmp/hf_cache/models--google--gemma-2-9b-it` |
| Revision | to be pinned to the resolved snapshot hash at load; recorded in the run manifest before any scoring |
| Auth | `HF_TOKEN` passed explicitly — **not** via `HF_TOKEN_PATH`, which `HF_HOME` relocation breaks (this caused the earlier spurious 401) |
| Precision | `bfloat16` |
| Attention | `eager` |
| Generation | none — scored readout only |
| Determinism | `do_sample=False`, `use_cache=False`, TF32 off, `cudnn.deterministic=True` |
| Readout | native four-way candidate scoring over `A_D`, mean log-prob of the candidate continuation, argmax; identical in form to the frozen W2C readout |
| Replay | full re-score, byte-identical requirement on 144/144 |

## 4. Baseline metrics — FROZEN before execution

Declared now so none can be chosen after seeing results: accuracy; macro-F1;
predicted-class distribution; per-class recall and precision; confusion matrix;
every directed gold→prediction transition with counts; unparsed / tie / exclusion
counts; per-group (36 scenario groups) support; per-mode support; error support
and reference support per candidate channel; baseline-correct population size.

## 5. Automatic directed-channel discovery — FROZEN

All ordered pairs `(g → s)`, `g ≠ s`, over the four modes are enumerated —
**12 candidate channels**, exhaustively and automatically. Ranking is by observed
error count, descending, ties broken by lexicographic `(g, s)`.

**No channel is selected in advance.** `cannot_answer → tool_call` receives no
privilege of any kind. The population's own confusion topology determines what,
if anything, is eligible.

## 6. Support and stability criteria — FROZEN

At 144 rows this is a *feasibility screen*, not a powered study. Criteria:

- **Channel viability floor:** ≥ 12 errors on the channel, spanning ≥ 6 distinct
  scenario groups.
- **Stability:** the top-ranked channel must remain top-ranked in ≥ 80% of 1,000
  group-preserving bootstrap resamples.
- **Scale-up projection:** for any viable channel, project the N required to reach
  the frozen support tiers (TRAIN ≥ 60 err / ≥ 60 ref; DEV ≥ 30 / ≥ 30) at the
  observed rate, with a bootstrap interval.
- **Preservation feasibility:** report the baseline-correct count and note
  explicitly that certifying α = 0.05 preservation needs ≥ 59 exposed-correct rows
  with 0 breaks — a bar 144 rows cannot clear. This is stated now so it cannot be
  discovered later as a surprise.

## 7. Intervention — EXPLICITLY FORBIDDEN IN THIS PHASE

Forbidden until a separate declaration authorises it: any router fitting, any
direction estimation, any activation capture, any hook registration, any
intervention, any dose selection, any SEALED construct, any channel selection
beyond automatic enumeration and ranking.

This phase loads the model, scores 144 rows four ways, and writes metrics.
Nothing else.

## 8. Terminal outcomes

Exactly one of:

- **`GEMMA_V1R3_BASELINE_NO_GO`** — no channel meets the viability floor or
  stability criterion, or the readout fails.
- **`GEMMA_V1R3_READABILITY_ELIGIBLE`** — at least one channel is viable and
  stable; a readability phase becomes *arguable* under a future declaration.

`READABILITY_ELIGIBLE` authorises nothing by itself.

## 9. Scope ceiling

Even if every later stage succeeded, the strongest permitted claim is:

> **independent prospective cross-population replication on an internally
> constructed population.**

This is **not** an external benchmark replication. V1R3 is authored by this
project and audited within one model family. It can support a
*population-dependence* argument; it cannot discharge the outstanding external
transfer claim.

## 10. Execution readiness

| | |
|---|---|
| Gemma weights | present |
| GPU | RTX 4090 D, 24 GB, 1 MiB in use |
| Population | 144 rows, gold computable |
| Runner | not yet written; blocked behind the §2 gate by design |
| Estimated runtime | ~10–15 min for 144 rows × 4 candidate scorings |

**The only thing standing between here and a Gemma number is §2.**
