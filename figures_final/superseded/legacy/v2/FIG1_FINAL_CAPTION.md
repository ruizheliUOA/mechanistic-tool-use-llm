# Figure 1 — final caption

## Caption (for the manuscript)

> **Figure 1: SAKIKO separates a behavioural change from a licensed correction
> claim.** *Top left:* channels are discovered offline from the model's own
> baseline errors — a directed channel $c=(g\rightarrow s)$ pairs a gold action
> $g$ with the specific wrong action $s$ the model emits — and supply both a
> channel-specific router $r_c$ and a candidate direction $d_c$. *Centre:* the
> model is frozen throughout. Because the injection site precedes the
> observation site in execution order ($L_{\mathrm{inj}}=21 < L_{\mathrm{obs}}=26$
> for the 36-layer backbone), diagnosis and intervention occupy two separate
> forward passes. Pass ① caches the MLP output at $L_{\mathrm{obs}}$ for the
> final prompt token; the router scores that cached activation offline, so no
> second network runs inside the model graph. If $p_c \geq \tau$, pass ② re-runs
> the same input while adding $h' = h + \alpha d_c$ at $L_{\mathrm{inj}}$, and
> the remaining frozen layers produce the action. Otherwise the forward pass is
> left untouched: no prompt edit, no weight update, no rewriting of the emitted
> answer. *Right:* every channel error resolves into exactly three mutually
> exclusive destinations. Conventional accuracy scores only GOLD_ARRIVAL as
> repaired and collapses SOURCE_RETAINED with OTHER_WRONG as still wrong, so it
> cannot distinguish an error that persists from one that has merely been
> redistributed; destination-resolved accounting separates all three, with
> source exit = gold arrival + other-wrong. Collateral risk is measured on a
> disjoint population of baseline-correct, router-exposed rows and is never
> netted against repairs. *Bottom:* the claim structure. Adjudicability is an
> entry condition on the study, not a property of the model. Readable →
> Steerable → Correctable are property claims about the tested setting and
> intervention. Licensability is a distinct object: an evidential decision,
> qualified by specificity, destination, preservation, yield and evidence
> precision, about whether the available evidence justifies asserting the
> correction claim. A DECLINE reports the state of that evidence; it does not
> assert that the setting is intrinsically uncorrectable.

## Length

~250 words. If the page budget bites, the deletable sentences in priority order
are: (1) the "Otherwise the forward pass is left untouched" sentence, which
Methods repeats; (2) the collateral-risk sentence, which §Preservation repeats.
Everything else is load-bearing for a reader who never reaches Methods.

## What deliberately does not appear in the figure

- Any model name (Qwen, Gemma, Mistral, Phi) or empirical count.
- Any CI, AUC, or licence verdict for a specific setting. Those are Figure 2.
- The five candidate correction surfaces (direct / score-space / activation /
  representation / decline). Compressing them into a legible box at 5.5 in cost
  more than it returned; the surface-adjudication point is made in Methods and
  in Figure 2's row structure instead. **This is a deliberate omission from the
  brief's §F, not an oversight.**

## Sentences that must not be used

The rejected v1 figure carried *"Aggregate metrics score SOURCE_RETAINED against
the other two."* This is backwards and is withdrawn. Verified against the frozen
artifacts: `fixed = gold_arrivals`, and on the Qwen3-8B primary channel
`source_exits = 52 = gold_arrivals 38 + wrong_to_wrong 14`. The correct statement
is that conventional aggregation collapses **{SOURCE_RETAINED, OTHER_WRONG}**.
