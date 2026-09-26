# Provenance path audit — `PAPER_NUMBER_PROVENANCE.csv`

62 claims, 23 distinct source artifacts. Three defects, all reproducibility
issues rather than numerical ones. **No claim value is wrong.**

## Defect A — abbreviated paths do not resolve as written

The CSV stores shortened paths. Example, carrying **8 Gemma claims**:

```
written:  panel_v1/gemma/formal_freeze/FORMAL_RESULTS.json
actual:   research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/FORMAL_RESULTS.json
```

Only **5 of 23** sources resolve by literal path. A reviewer following the table
as written finds nothing at 18 of them.

## Defect B — `FORMAL_RESULTS.json` is ambiguous across two models

Two files share that basename:

```
final/results/qwen3_4b_w2c_formal_v1/FORMAL_RESULTS.json
research_exploration/.../panel_v1/gemma/formal_freeze/FORMAL_RESULTS.json
```

Resolving the abbreviated Gemma path by basename lands on the **Qwen3-4B** file.
Anyone reconstructing the 8 Gemma claims that way silently reads the wrong
model's numbers, and every field name matches, so nothing errors.

**This is the most dangerous of the three** — it fails silently and produces
plausible wrong values.

## Defect C — one filename mismatch, one artifact absent

| written | status |
|---|---|
| `mistral7b_w2c_sakiko_ca/FINAL_REPORT.md` (2 claims) | file is actually `MISTRAL7B_W2C_SAKIKO_CA_FINAL_REPORT.md` — name mismatch only |
| `PHASE6_EVIDENCE_RECONSTRUCTION_AND_AUDIT.md` (1 claim) | **not present anywhere on the paper branch** |

So **1 of 62 claims** has no resolvable artifact on the submission branch.

## What is correct

The table's **evidence-class labelling is sound** and is the thing my own summary
document got wrong. It distinguishes `SEALED formal` (24), `retrospective` (15),
`prospective locked` (6), `simulation` (4), `DEV` (2), `development` (2),
`historical` (2) and others. Phi-3.5's three claims are labelled `retrospective`
**in the provenance table** — the label was lost only in the summary I wrote.

Model attribution is complete: Qwen3-8B 10, Gemma 11, Qwen3-4B 8, Qwen2.5-7B 7,
Qwen3.5-9B 6, Phi-3.5 3, Mistral 2, Llama 2, framework-level 13.

## Required fixes before submission

1. **Expand every path to repo-root-relative form.** Ambiguity is not acceptable
   in a table whose purpose is reproducibility.
2. **Disambiguate `FORMAL_RESULTS.json`** — always with its full directory.
3. **Correct the Mistral filename** to `MISTRAL7B_W2C_SAKIKO_CA_FINAL_REPORT.md`.
4. **Resolve `PHASE6_EVIDENCE_RECONSTRUCTION_AND_AUDIT.md`** — either restore it
   to the branch or re-source that claim. A submission that ships a provenance
   table with an unresolvable entry invites exactly the scrutiny it exists to
   prevent.
5. **Replace the `same` shorthand** with explicit repeated paths. It is a
   spreadsheet convenience that breaks any automated checker.

None of these change a number. All of them affect whether a reviewer can verify
one.
