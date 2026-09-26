# Top-tier figure style audit

## Scope and honesty note

**What this audit is based on:** the six reference figures supplied directly in
conversation (dense CV/diffusion method figures: a dehazing backbone with FGM
and haze-noise diffusion; an event-deblurring LDI/LDR pipeline in two versions,
one a TPAMI submission; a latent-compressor / IC-ControlNet figure; an
illumination-aware diffusion face-restoration figure with sub-panels (b)–(e); a
HyperAlign/LDM compression figure; and the GNVC-VD framework with panels (a)–(c)).
I inspected each of these images.

**What it is not based on:** I did **not** open the six arXiv PDFs named in the
brief (LayerNavigator, Representation Surgery 2402.09631, CAA 2312.06681, and
the three tool-use steering IDs 2607.05790 / 2605.05980 / 2602.04935). Web
fetching was not available in this session. The three tool-use IDs in particular
I could not verify exist. Every pattern recorded below is one I can point to in
an image I actually looked at; nothing here is inferred from an abstract or from
memory of a paper I did not open. The literature pass against those specific
PDFs remains outstanding and is listed at the end.

## Patterns visible in the supplied references

| # | Pattern | Where visible | Adopted for SAKIKO? |
|---|---|---|---|
| 1 | **Dashed grouping panels with a title tab riding the border.** Functional regions are enclosed and named; the title sits *on* the boundary with a white knockout, not inside as a heading row. | All six; clearest in the dehazing figure ("Dehazing Backbone", "FGM", "Haze-Noise Diffusion") | **Yes** — `S.panel()` implements exactly this. |
| 2 | **Pastel fills carry grouping; saturated strokes carry identity.** Fills are near-white tints (5–10% saturation); the border is the saturated hue. Nothing is filled with a strong colour. | Illumination figure; GNVC-VD panels (b)/(c) | **Yes** — every fill is a `*_F` tint of its stroke. |
| 3 | **Inline legend for repeated glyph semantics.** Where a symbol recurs (flame = trainable, snowflake = frozen), it is defined once in a boxed legend rather than re-annotated. | Illumination figure; GNVC-VD (a) | **Partly** — SAKIKO has only one recurring glyph (frozen), so it is annotated once at its first occurrence instead of paying for a legend box. |
| 4 | **Sub-panel letters (a)/(b)/(c) with a short descriptive phrase, not a sentence.** | GNVC-VD; illumination figure | **Yes** — candidates B and C. |
| 5 | **Math lives in the diagram; prose lives in the caption.** Labels are symbols (`z_0`, `Z_gt`, `E(m,T_i)`), not explanations. | LDI/LDR figure is almost pure notation | **Partly, and this is where the candidates currently fail** — see §Failure below. |
| 6 | **The backbone is drawn once and large; auxiliary modules are smaller and offset.** The reader's eye finds the main dataflow immediately. | Dehazing figure (backbone across the top); GNVC-VD (a) above (b)/(c) | **Yes** — the frozen transformer is the largest object; offline discovery is deliberately smaller. |
| 7 | **Thin uniform strokes (~0.5–0.9 pt) throughout.** No stroke-weight drama; emphasis comes from colour and size, not line thickness. | All six | **Yes** — `LW_BOX = 0.7`, `LW_SPINE = 0.9`. |
| 8 | **Rounded rectangles at small radius (~2–3 px), never pill-shaped.** | All six | **Yes** — `rounding_size` 0.3–1.2 data units. |
| 9 | **Repeated tensors drawn as offset stacked cards** to signal a batch/sequence dimension. | Dehazing (`Z_T`, `Z_t`); GNVC-VD latents | **No** — SAKIKO operates on a single hidden vector at one token position; stacked cards would misrepresent the object. |
| 10 | **No gradients, no shadows, no 3-D, no photographic chrome except genuine data samples.** | All six | **Yes.** |

## Patterns deliberately *not* copied

- **Colour count.** The illumination and HyperAlign figures run 8–12 hues. That is
  viable when hue encodes *module identity* in a system with many named
  components. SAKIKO has four semantic states, so more hues would encode nothing
  and would break the grayscale requirement.
- **Photographic insets.** The CV references embed images because the data *is*
  images. SAKIKO's data is a four-way categorical decision; a photo would be
  decoration.
- **Text inside large boxes as explanation.** The references keep this to a
  minimum; my current candidates do not (see below).

## Where the current candidates fail this audit

Rendered at ICLR single-column width (`\textwidth` = 5.5 in under
`iclr2027_conference.sty`, confirmed in `manuscript/06_latex/FULL_MANUSCRIPT.tex`),
all three candidates break:

- designed at **7.24 in**, so insertion scales them by **0.76×**;
- smallest type (`T_TINY` = 5.0 pt) lands at **3.8 pt** — unreadable;
- rendering at a true 5.5 in instead (`fig1_C_at_true_width.png`) keeps the type
  legible but the layout collapses: panel titles collide, and the LICENSABLE
  annotation overruns the panel-(c) title.

The root cause is pattern #5. The candidates carry explanatory sentences inside
the figure that the references would have pushed into the caption. At 7.24 in
that prose fits; at 5.5 in it does not. **This is a content problem wearing a
sizing problem's clothes** — shrinking the font is not the fix.

## Outstanding

1. Open the six named PDFs and re-run this audit against them; correct anything
   here that they contradict. Verify whether 2607.05790 / 2605.05980 / 2602.04935
   resolve at all.
2. Rebuild the chosen candidate at a native 5.5 in with roughly 40% less in-figure
   prose, moving it to the caption.
