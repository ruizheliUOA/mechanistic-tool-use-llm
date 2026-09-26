# Human audit analysis plan — FROZEN

**Frozen 2026-08-17, before any human label has been seen.** Every metric,
threshold, and interpretation rule below is fixed now. Nothing here may be
changed after labels arrive.

## Inputs

- `ADJUDICATION_A1_BLINDED.csv`, `ADJUDICATION_A2_BLINDED.csv` — two independent
  annotators, independently shuffled, opaque per-file item tokens.
- `_SEALED_ANNOTATION_KEY.json` — token → `sample_id`. **Not opened until both
  files are filed and filed-state is recorded.**
- `pilot_programmatic_gold.jsonl` — the programmatic `gold_mode`.

## Thresholds are historical, not invented

The acceptance gates were declared in
`research_exploration/original_population_pilot_v1r3/compute_annotation_gates.py:16`
**before any annotation existed**, as part of the original package:

```
raw_agreement_min      0.90
cohens_kappa_min       0.80
per_class_agreement_min 0.85
no_unresolved_ambiguous
```

These are adopted unchanged. They govern **annotator-vs-annotator** agreement only.

## Metrics — all computed, all reported

1. **Primary agreement metric: raw pairwise agreement** between A1 and A2 over
   144 rows. Gate: ≥ 0.90.
2. **Cohen's kappa** between A1 and A2, chance-corrected with the empirical
   marginals of each annotator. Gate: ≥ 0.80.
3. **Per-class agreement** — for each of the four modes, agreement restricted to
   rows where either annotator used that mode. Gate: ≥ 0.85 for all four.
4. **Human-vs-programmatic-gold agreement** — A1 vs `gold_mode`, A2 vs
   `gold_mode`, and consensus vs `gold_mode`. **No threshold is declared for this
   quantity, because none was declared historically. It is reported
   descriptively.** See "Claim wording" below.
5. **Disagreement matrix** — full 4×4 of A1 label × A2 label, and 4×4 of
   consensus × programmatic gold. Reported in full, never summarised to a scalar.
6. **`another_mode_defensible` rate** per annotator and jointly.
7. **Per-group consistency** — of the 36 scenario groups, how many receive four
   distinct consensus labels.

## The previously flagged 19 rows

The superseded pre-revision audit pass flagged 19 rows (8 mode disagreements,
19 in union with `another_mode_defensible`). These are listed in
`ADJUDICATION_PRIORITY_ROWS.json`.

**They receive no special handling during annotation** — annotators do not know
which they are, and must not be told. After unsealing, they are analysed as a
**prespecified subgroup**: agreement on the 19 is reported separately from the
remaining 125. If agreement on the 19 is materially lower, that is reported as
evidence the V1R3 revision did not fully repair them, and the paper says so.
This subgroup comparison is descriptive; no gate attaches to it.

## Adjudication procedure

Applies only to rows where A1 ≠ A2, or where either marked
`another_mode_defensible: yes`.

1. The adjudicator sees both submissions **after both are filed**, plus the row.
2. Classify each disagreement's cause: `TOOL_AVAILABILITY`,
   `PARAMETER_SUFFICIENCY`, `DIRECT_ANSWERABILITY`, `SAFETY_VS_CAPABILITY`,
   `MULTI_STEP`, `FIRST_ACTION_UNDEFINED`, `WORDING`, `OTHER`.
3. If the ontology determines a unique label and one annotator misread the row,
   record `adjudicated_label` with a written operational reason.
4. **If both readings are defensible from the visible row, the row is defective.**
   It is removed from the analysed population and recorded as defective. It is
   *not* resolved by preferring the programmatic gold, and *not* by preferring the
   drafter's intended label. The intended label carries no evidential weight.
5. `adjudicated_label` is copied into a human gold column only after adjudication
   completes with no unresolved ambiguity.

## Pass / qualify / fail

Exactly one outcome, determined mechanically:

- **PASS** — all four historical gates hold (raw ≥ 0.90, kappa ≥ 0.80, every
  per-class ≥ 0.85, zero unresolved ambiguous rows after adjudication).
  → the human gold gate is satisfied; the Gemma baseline-only screen may run;
  gold may be described as **human-validated**.

- **QUALIFY** — gates 1–3 hold but defective rows were removed, or a per-class
  figure sits between 0.80 and 0.85.
  → the screen may run **on the retained population only**, with the removed rows
  and the reduced N disclosed in the paper.

- **FAIL** — raw < 0.90 or kappa < 0.80, or any per-class < 0.80, or unresolved
  ambiguity remains.
  → the human gate does **not** pass. The Gemma baseline screen is **not** run.
  The finding is reported as a property of the instrument, not of any model.

## Claim wording, predeclared as a function of the result

Because no threshold was ever declared for human-vs-programmatic-gold agreement,
that quantity cannot pass or fail anything. It instead fixes the wording:

| Human vs programmatic gold | Permitted description of V1R3 gold |
|---|---|
| ≥ 0.95 | "programmatically derived and human-confirmed" |
| 0.90 – 0.95 | "programmatically derived, human-confirmed on the large majority of rows, with N disagreements disclosed" |
| 0.80 – 0.90 | "programmatically derived; human agreement moderate — the label engine and human judgement diverge on a material minority, reported per class" |
| < 0.80 | "programmatically derived and **not** human-corroborated" — and any downstream Gemma result must be reported as conditional on a label engine humans did not reproduce |

In every band the reported number is the measured one. The band selects wording,
never whether a result is reported.

## Anti-gaming commitments

- The sealed key is not opened before both files are filed.
- No metric, threshold, or subgroup is added after unsealing.
- If a metric here turns out to be uncomputable, it is reported as uncomputable
  rather than replaced.
- Gemma is not loaded, and the Gemma runner is not written, until this gate is
  adjudicated.
