# External illustrator prompt — SAKIKO Figure 1

*Usable verbatim with another AI or a human designer. Pair it with
`SAKIKO_FIG1_SCIENTIFIC_SOURCE_OF_TRUTH.md`, `SAKIKO_FIG1_TOPOLOGY_LOCK.md`, and
your visual reference images.*

---

## SCIENTIFIC CONTENT

You are drawing the main method figure for a machine-learning paper. The system,
SAKIKO, decides whether a behavioural change caused by an internal intervention
may honestly be called a **correction**.

A frozen language model makes a pre-execution decision over four actions:
`tool_call`, `request_for_info`, `cannot_answer`, `direct_answer`. When it errs,
the error has a **direction**: `c = (g -> s)`, gold `g` and the specific wrong
action `s` the model emits.

Offline, from the model's own errors, the system discovers a supported channel,
fits a small per-channel logistic-regression **Router**, and estimates a
channel-specific direction `d_c`.

At evaluation there are **two forward passes over the same frozen model**:

- **Pass 1 (diagnose):** run the input, cache the hidden state `h_obs` at the
  MLP output of layer 26 (final prompt token). The Router then scores that
  *cached* vector offline in NumPy — it never runs inside the model.
- **Pass 2 (intervene):** only if `p_c >= tau`, re-run the same input while adding
  `h' = h + alpha * d_c` at the MLP output of layer 21, then continue through the
  remaining frozen layers to the action readout.

Layer 21 executes before layer 26, which is why two passes are required.

Every channel error then lands in exactly one of three places: `SOURCE_RETAINED`
(still `s`), `GOLD_ARRIVAL` (reaches `g`), `OTHER_WRONG` (some other wrong action
`w`). Conventional accuracy counts only `GOLD_ARRIVAL` as repaired and collapses
the other two as still wrong — so it cannot tell a persistent error from a
redistributed one.

A **separate, disjoint** population — rows the model already got right but on
which the Router fired — carries the collateral estimand. Breaks there are never
netted against repairs.

Finally, the claim structure: `Adjudicable` is an entry condition on the study;
`Readable -> Steerable -> Correctable` are property claims about the tested
setting; `Licensable` is an epistemic decision about whether the evidence
justifies asserting the Correctable claim, qualified by specificity, destination,
preservation, yield, and evidence precision.

## VISUAL PRIORITIES

Obvious in 5-10 seconds: the model is frozen; there are two passes; a small
Router gates the second one; one hidden state is edited; the output is a
multiclass action; there are three destinations; the ladder is ordered; and
**Correctable and Licensable are different kinds of thing.**

Present but secondary: offline discovery, direction estimation, the preservation
population, causal controls, adjudicability, the five licensing criteria.

Leave to the caption: K=59, all thresholds, dose grids, CI procedures, every
number.

## FORBIDDEN MISINTERPRETATIONS

Do **not**:

1. draw the two-pass implementation as one pass, or route `h_obs -> Router ->
   h_inj` inside a single forward pass;
2. make `Licensable` a fourth scientific property or a fourth node in the chain;
3. place any model name (Gemma, Qwen3-4B, Mistral, Phi) under a failing property
   — that reads as "Correctable = FAIL" when what failed was the licence;
4. write that aggregate correctness separates `SOURCE_RETAINED` from the other
   two — it collapses `{SOURCE_RETAINED, OTHER_WRONG}`;
5. imply activation space is the only correction surface;
6. imply `DECLINE` means intrinsically uncorrectable;
7. imply a historical `ADMIT` is a safety certification;
8. draw the Router as a neural network or a second LLM;
9. put any empirical number, CI, or per-model verdict in this figure.

## OPTIONAL VISUAL FREEDOMS

Yours entirely: composition and panel arrangement; whether the claim layer sits
below, beside, or wrapped around the mechanism; shape vocabulary; nesting versus
chaining for the ladder; colour palette; typography; arrow style; aspect ratio;
whether the transformer runs horizontally or vertically; how many blocks to draw;
whether the two passes are two lanes, one stack with a re-entry arc, or something
better you invent.

## REQUIRED OUTPUTS

- An editable source file (PPTX, SVG, AI, or Figma) with live text and shapes —
  not a flattened raster.
- Vector PDF for LaTeX, plus SVG and PNG.
- Native design width **5.5 inches** (ICLR single column). No text below ~6 pt at
  final size. Do not design large and shrink.
- White background; no gradients, shadows, 3-D, neon, or dashboard styling.
- Must remain interpretable in grayscale.
