# Figure role map — frozen

> **Superseded numbering (2026-09-11):** the manuscript uses Figure 1 = thesis
> (`fig1_thesis`), 2 = framework (`fig2_framework`), 3 = the four-panel chain, 4 =
> the ladder, 5 = mechanism, D1 = historical controls. Captions are in
> `manuscript/figure_captions.md`; the style is in `FIGURE_STYLE.md`. The role
> analysis below is kept for provenance.

Two visual objects, never conflated:

| | answers | figure |
|---|---|---|
| **What SAKIKO is** | what does it do, end to end? | **F1** |
| **What SAKIKO discovers** | why is correction alone insufficient? | **F0**, then **F3** with data |

| fig | question answered | main claim | redundancy | location | verdict |
|---|---|---|---|---|---|
| **F1 framework** | what does SAKIKO do? | the pipeline follows an intervention from the pre-execution decision through adjudication | none — the only figure showing the method | **§1, first figure** | **REDESIGNED** |
| **F0 dissociation map** | why is correction insufficient? | improving behaviour settles only the first of four questions | conceptual only; F3 carries the data | **§5 opener** | **KEEP, RELOCATED** |
| **F2 correction** | is the correction real and specific? | stable across seeds; corrupted variants do not reproduce it | none | §4 | KEEP |
| **F3 dissociation** | does improvement establish repair? | three measured dissociations | F0 is its conceptual preface, not a duplicate | §5 | KEEP |
| **F4 licensability** | what does the evidence license, per setting? | Correctable is not Licensable | F3c covers 3 modern settings; F4 covers all 7 and adds the property axis | §6 | KEEP |
| **F5 mechanism** | why separate the stages? | readability does not locate intervention | none | **appendix** | APPENDIX |

## Part 4 — the role of `fig0_dissociation_map`

**Recommendation: Option B — the Section 5 opening graphic.** Not Option A (a panel
inside F3, which is already three panels and full), not C (it is too central for an
appendix), and not D (F3 shows the measurements but never states the *structure* of
the argument — that four distinct questions exist and improvement answers one).

F0 and F3 are preface and evidence. F0 says *there are four questions*; F3 shows
*three of them coming apart*. Neither replaces the other.

## Part 5 — which figure comes first

**Option A: the framework figure.** A reader meeting F0 first learns that a method
they cannot yet picture fails three tests — which reads as an evaluation checklist,
the single misreading this paper must avoid. F1 first establishes the scientific
object, so the later finding lands as a property of a real method rather than a
critique of an unnamed one. F1 also carries the Verify taxonomy, so `OTHER_WRONG`
and `BROKEN` are already defined when §5 uses them.

**Decision: F1 in §1. F0 opens §5.**

## Part 8 — framework-figure quality test

| # | question | answerable from F1 alone? |
|---|---|---|
| 1 | what problem is studied | yes — band 1, the pre-execution decision before any tool executes |
| 2 | what is an error channel | yes — Discover: `g → s`, partially overlapping |
| 3 | how is an error detected | yes — Detect: per-input router |
| 4 | how is intervention selected | yes — Correct: estimator, site, dose, selective and gated |
| 5 | what can happen after intervention | yes — five outcomes in the Verify panel |
| 6 | why gold arrival ≠ leaving the source | yes — `OTHER_WRONG`: "left the source, landed elsewhere" |
| 7 | how are already-correct samples handled | yes — `CORRECT_RETAINED` / `BROKEN` |
| 8 | what does Licensable mean | yes — "the evidence supports the claim" |
| 9 | why more than activation steering | yes — the ADJUDICATION grouping and the License panel |
| 10 | why Verify and License exist | yes — "movement is not arrival; arrival is not preservation" |

**No v1/v2 split is visible:** CORRECTION and ADJUDICATION are brackets *within* one
framework, drawn in the same weight, with the pipeline arrow running unbroken
through all five stages.
