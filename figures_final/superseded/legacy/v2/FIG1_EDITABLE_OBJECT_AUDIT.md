# FIG1 editable-object audit

File: `figures/v2/final/fig1_sakiko_core_FINAL.pptx`
Slide canvas: **5.5 × 3.34 in** — native ICLR text width, no downscaling at insertion.

## Verdicts

| Check | Result |
|---|---|
| TEXT EDITABLE | **YES** — 58 shapes carry live text frames |
| BOXES EDITABLE | **YES** — 50 `AUTO_SHAPE` (rounded rectangles, ovals) |
| ARROWS EDITABLE | **YES** — 42 `LINE` connectors with DrawingML `tailEnd` arrowheads |
| TRANSFORMER MODULES EDITABLE | **YES** — 18 individual block shapes (9 per pass) |
| ENTIRE FIGURE FLATTENED | **NO** |

**Raster picture count: 0.** Programmatically verified via `python-pptx`: no shape
on the slide has `shape_type == PICTURE`. There is no embedded PNG anywhere in
the deck.

## Shape inventory

```
total shapes            128
  AUTO_SHAPE             50   rounded rects, ovals, dashed panels
  TEXT_BOX               36   free-standing labels and annotations
  LINE                   42   straight connectors, 40 with arrowheads
shapes with text         58
PICTURE (raster)          0
```

## What you can edit directly in PowerPoint

- **Move any module.** Every box is an independent shape; drag it and its text
  moves with it.
- **Recolour.** Fill and line colour are set per shape. The palette is six hexes:
  `1B1F26` ink, `33475B` slate, `3D6E9C` blue, `2F7D6A` teal, `A8453B` vermilion,
  `8B9199` grey, with matching 5%-saturation fills.
- **Resize type.** All text is Arial at explicit point sizes (5.0–7.0 pt). Select
  all and scale if you want a different floor.
- **Reroute arrows.** Connectors are straight segments; elbows are built from two
  or three separate connectors so each leg drags independently.
- **Adjust panel widths.** The three dashed grouping panels are plain rectangles
  with a `dash` preset; resize freely.
- **Edit equations.** `h′ = h + α d_c`, `p_c ≥ τ ?`, `â′ = w ∉ {g, s}` are Unicode
  text, not images — retypable in place.

## Known limitations, stated plainly

1. **No shape grouping.** `python-pptx` cannot create native groups. The nine
   transformer blocks per pass are nine separate shapes, not one group. To move a
   stack as a unit, rubber-band select it and group it manually (Ctrl+G) — a
   one-time action.
2. **Subscripts are flattened into the string.** `L_obs`, `h_obs`, `d_c`,
   `r_c(h_obs)` are written with underscores rather than true subscript runs. The
   matplotlib candidates render proper subscripts; the PPTX does not. If you want
   true subscripts, select the character and apply subscript formatting.
3. **Rounded-corner radius renders larger in LibreOffice than in PowerPoint.**
   The exported PNG/PDF here were produced by LibreOffice 7.3 headless, which
   interprets the `adj` value more generously. In PowerPoint the corners will look
   tighter and more academic. Judge the geometry from the PPTX, not from the PNG.
4. **`①`/`②` are Unicode glyphs**, so they depend on the viewing font having them.
   They render correctly in Arial on Windows and macOS.

## Exports

| File | Produced by | Use |
|---|---|---|
| `fig1_sakiko_core_FINAL.pptx` | `python-pptx` (source of truth) | editing |
| `fig1_sakiko_core_FINAL.pdf` | LibreOffice headless | LaTeX `\includegraphics` |
| `fig1_sakiko_core_FINAL.svg` | LibreOffice headless | vector editing elsewhere |
| `fig1_sakiko_core_FINAL.png` | LibreOffice headless | quick preview |

All three exports are regenerated from the PPTX by
`figures/v2/src/build_fig1_pptx.py` followed by three `soffice --convert-to`
calls, so the PPTX remains the single source of truth.
