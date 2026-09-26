# Figure style

The figures are ggplot2, written in R, on one shared theme (`R/theme_sakiko.R`) and
one palette. Colours are color.amfe.space palette 106 ("Lancet"). Type is the
manuscript's serif, so figure text matches body text.

## The look

`theme_bw()` at base size 8.5 in Times New Roman: a white panel inside a thin grey
frame, light grey major grid and a fainter minor grid, small outward ticks, grey tick
labels, and a centred "(a) Title" heading with a grey subtitle naming the model and
protocol. Panels are composed with patchwork (`pa | pb`) at exactly the ICLR text
width, so the sizes in the scripts are the sizes on the page.

**At most two panels per figure.** A figure that would need a third is split, and a
panel is never repeated in another figure.

## Colours: one meaning each, in every figure

Palette 106, in order: `#7b95c6 #49c2d9 #a1d8e8 #67a583 #a2c986 #d0e2c0 #fded95
#ffc1a6 #f59c7c #f47254 #c85e62`.

| role | colour | used for |
|---|---|---|
| `REAL` | `#7b95c6` blue | the real direction · the intervention · the activation arm |
| `ARRIVE` | `#67a583` green (`#d0e2c0` as a fill) | arrival at the required action · a property that holds · ADMIT |
| `COMPARE` | `#49c2d9` cyan (`#a1d8e8` as a fill) | a comparison arm: score-space, ungated · a partial property |
| `OTHER` | `#f59c7c` salmon (`#ffc1a6` as a fill) | a decision that moved but not to the required action · a property that fails |
| `BROKEN` | `#c85e62` brick | a correct decision broken |
| `DECLINE` | `#fded95` pale yellow | a formal DECLINE — insufficient evidence, deliberately not a failure colour |
| greys | `#8c8c8c` / `#bfbfbf` / `#dcdcdc` | control directions · a decision left as it was · one the Router never reached |

In the control battery each arm carries the colour of the component it corrupts, cool
to warm: real, direction, sign, site, channel. Small text on light fills uses the
darker inks defined beside the palette.

## Chart types

| figure · panel | chart | what it shows that another would not |
|---|---|---|
| 1a | stacked bars | the two populations one intervention touches, on one scale |
| 1b | forest plot | point estimate, interval and the one criterion (0.50) together |
| 3a | lollipop | the real direction against four corrupted variants on the same population |
| 3b | strip plot | all 59 random directions per setting, as points, against the real value |
| 4a | dumbbell | a paired comparison of two arms across outcomes |
| 4b | paired dot plot | collateral on both denominators against the 5% bound, so E2 ≥ E1 is visible |
| 5 | tile map | a status matrix across seven settings and six properties, faceted by protocol |
| D1a | lollipop against the bound | each seed's collateral and its distance above 5% |
| D1b | lollipop | the real direction against its controls |

## Data

Figures plot only what `figures_final/data/*.csv` contains. Those tables are written
by `export_figure_data.py` from `_panels.py:load_all()`, which reads the evidence
index and the frozen verdict artifacts and asserts every identity the captions state.
No R script computes a number.

## Rules carried into every figure

0.50 is the only destination boundary drawn; 0.6042 is never drawn or named. No
"admitted" for Qwen3-4B or Gemma-2-9b. Historical and sealed evidence never share a
scale: they sit in different panels or different facet rows, each labelled with its
protocol. No significance stars. British spelling.

## Regenerating

```
cd SAKIKO-paper
python figures_final/export_figure_data.py          # only if the evidence changed
for f in figures_final/R/fig*.R; do Rscript "$f"; done
```

Requires R (4.6.1 here) with ggplot2, patchwork and ggrepel. Output goes to
`figures/`; see `figures/MANUSCRIPT_FIGURES.md`.
