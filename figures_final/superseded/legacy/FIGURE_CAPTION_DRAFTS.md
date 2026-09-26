# Figure captions

**Figure 1 — SAKIKO-Core.** *(A)* The framework separates properties of the **setting**
(Readable → Steerable → Correctable, inner panel) from properties of the **study**
(Adjudicable, Licensable, outer dashed region). This distinction is what allows a refusal to
state where evidence failed rather than asserting that a model cannot be corrected. Labels below
each node give the setting that terminated at that rung; red denotes a stop, teal the single
licensed setting. Preservation, yield and evidence constrain whether a setting reaches
Licensable. *(B)* Correctable resolves the fate of every channel error into three outcomes.
Aggregate metrics distinguish only SOURCE_RETAINED from the rest; separating GOLD_ARRIVAL from
OTHER_WRONG requires an action space with `|A_D| ≥ 3` and is the measurement this paper adds.
Exemplars mark the rung a setting reached under its own protocol generation and are not claims
about model capability.

**Figure 2 — Failure localisation across settings.** Rows are model–channel units; columns are
the rungs of the ladder. Green marks a rung passed, red the rung that bound the outcome, amber a
structurally vacuous test, grey an older protocol generation, and pale grey a rung never entered.
Read horizontally, each row terminates for a different reason: support, readability, specificity,
destination certification, or preservation. Read vertically, no single column explains the
outcomes. **†** Qwen3-8B passed the frozen preservation endpoint E1 (0/211); under the
retrospective exposure-conditional estimand E2 it is 0/6 with a one-sided 95% upper bound of
0.393 and is **not** preservation-certified. Rows marked with grey cells were adjudicated under
earlier protocol generations and are not directly comparable to the sealed formal trials.

**Figure 3 — Aggregate movement is not destination-correct repair.** *(A)* Under conventional
reporting all three settings appear positive, with Net between +16 and +40. *(B)* Resolving the
same errors by destination shows where the movement went. Qwen3-8B sends 38 of 52 departures to
gold; Qwen3-4B sends 27 of 64 to a third wrong action and Gemma 10 of 27. All three interventions
are direction-specific — none of 59 fresh matched-random directions reached the real effect in
any trial (*p* = 0.0167) — so the difference between the panels is not explained by intervention
quality. Fractions in (B) are of all channel errors, so bar widths are comparable across rows.

**Figure 4 — Preservation depends on which population is at risk.** *(A)* The frozen endpoint E1
counts damage over all baseline-correct rows, but a Router-gated intervention can only affect
rows it fires on. Exposure (dark) is a small fraction of the frozen denominator (pale) — six rows
of 211 for the licensed setting. *(B)* Point estimates under both estimands, with exact
Clopper–Pearson one-sided 95% upper bounds on E2. No setting attains the 0.05 bound, including
the licensed one, whose zero observed breakages over six exposed rows admit a true damage rate up
to 0.393. This does **not** revoke the historical ADMIT, which was adjudicated prospectively
against E1 and stands; it shows that E1 and E2 are different estimands and that passing the
former is not evidence of safety. Certifying 0.05 with zero breakages requires roughly 59 exposed
rows.
