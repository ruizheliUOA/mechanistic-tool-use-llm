# Figure phase status

```
FIGURE_SYSTEM_DEFINED:            YES
FIGURE_PROVENANCE_COMPLETE:       YES   (21 elements mapped)
FIGURE_1_READY:                   YES
FIGURE_2_READY:                   YES
FIGURE_3_READY:                   YES
FIGURE_4_READY:                   YES
FIGURE_5_READY:                   NO    (not generated — see below)
MAIN_TEXT_FIGURES_RECOMMENDED:    F1, F2, F3, F4
APPENDIX_FIGURES_RECOMMENDED:     (F3 or F4 only if page budget forces it)
VISUAL_STYLE_CONSISTENT:          YES   (single shared module)
READY_FOR_MANUSCRIPT_INTEGRATION: YES
```

## F5 — why it is not done

The ACEBench transfer figure is a side-by-side comparison whose entire content is already carried
by prose and a table. Producing it under the current page overrun would add an asset that the
compression pass would immediately cut. It is specified in the storyboard and can be generated in
minutes if space appears. Marking it NO rather than shipping a weak figure.

## Claim discipline in the assets

- F2 carries a **†** on Qwen3-8B's preservation cell stating that only E1 passed and E2 is
  uncertified at 0/6, upper 0.393. The green tick alone would have implied safety certification.
- F1 exemplars are labelled as the rung a setting *reached*, with a footnote that a stop is a
  statement about evidence, not about a model.
- Llama does **not** appear in F1 as a steerable-but-not-correctable exemplar; in F2 it appears
  only as a specificity failure under a historical protocol generation.
- ACEBench appears in F2 as an adjudicability stop annotated "framework instantiation only."
- F4's caption states explicitly that the historical ADMIT is not revoked.

## Files

`main/fig1_sakiko_core.{pdf,png}` · `main/fig2_failure_localization_matrix.{pdf,png}` ·
`main/fig3_destination_accounting.{pdf,png}` · `main/fig4_preservation_exposure_audit.{pdf,png}`

Sources in `source/`: `sakiko_style.py` plus one generator per figure. Every figure regenerates
from frozen artifacts by running its script — no hand-entered numbers.
