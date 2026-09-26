# Visual style system

Single source: `source/sakiko_style.py`. Import and call `apply()` before plotting; save with
`save()` to emit PDF + 300 dpi PNG together.

## Palette — semantic, low-saturation, grayscale-separable

| role | hex | grayscale L* |
|---|---|---:|
| structure / framework | `#2E4057` muted navy | 0.34 |
| pass / licensed | `#3F7D6E` muted teal | 0.46 |
| fail / binding rung | `#A8433B` muted vermilion | 0.40 |
| inconclusive / vacuous | `#B8873A` muted amber | 0.58 |
| not reached / unavailable | `#8A8F98` neutral grey | 0.62 |
| rules, inert fills | `#DDE1E6` | — |
| ink | `#1C1F24` | — |

Teal/vermilion is distinguishable under deuteranopia and protanopia, and the five L* values are
spaced so the semantics survive grayscale printing. Colour never carries meaning alone — every
categorical cell also carries a glyph (✓ ✗ ~ · H).

## Typography and geometry

DejaVu Sans throughout. Body 8.5 pt, axis labels 8.5, ticks 7.8, legends 7.5, annotations 6.9–7.4,
panel letters 10 bold. Fonts embedded as TrueType (`pdf.fonttype 42`) so text stays selectable and
editable in the PDF.

Thin linework (0.7–1.1 pt), square or lightly rounded boxes, no shadows, no gradients, no 3D, no
icons. White background, top/right spines removed on data axes.

## Rules

1. Vector PDF is the deliverable; PNG is preview only.
2. Every figure answers exactly one question, stated in its panel title.
3. Every number is read from a frozen artifact at generation time — never typed by hand.
4. Retrospective quantities are labelled in the figure, not only in the caption.
5. Figure width 7.2 in (two-column full width); heights 2.5–4.2 in.
