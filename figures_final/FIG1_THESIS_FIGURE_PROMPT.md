# Prompt — Figure 1 (Introduction thesis figure)

Paste everything below the line into a design or coding assistant, or hand it to an
illustrator. It supersedes `fig1_header` and `fig0_dissociation_map`.

---

Design Figure 1 of an ICLR paper titled **"When Does Correction Become Repair?
Verifying Internal Interventions on Tool-Use Decisions."**

## The one message

A reader who looks for five seconds must leave with: **the correction is real, and
it is still not a repair.** An internal intervention changes a language model's
pre-execution tool decision and the headline number improves, yet the corrected
decisions do not all reach the required action, the intervention breaks most of the
correct decisions it touches, and favourable estimates can still fail the evidence
test. This is a figure about a scientific finding, not about a method.

## It must not be

A pipeline or architecture diagram · a Transformer schematic · boxes-and-arrows stages ·
a five-step process chart · a business infographic · a table · a rung ladder (the
ladder appears later, in Figure 4) · three equally sized panels.

## Layout

One row at full text width, about **7.0 × 2.3 in**. A single thin grey header runs
along the top, reading left to right, with every step in equal weight:

`correction → destination → preservation → evidence → licensed repair claim`

Below it sit three zones of **unequal** width, each aligned under its steps:

**Zone A — "what the aggregate says" (≈ 18% width, under *correction*).** One large
number, **+55**, in the main blue, with a small grey label *net gain · Phi-3.5*.
Nothing else. This is the claim the rest of the figure takes apart.

**Zone B — "where the corrections went, and what they cost" (≈ 50% width, under
*destination* and *preservation*). The dominant zone.** Two horizontal stacked bars,
or unit-dot rows, on a shared scale:
- *Row 1 (secondary):* **153** decisions moved off an error → **107** reach the
  required action (blue) · **46** land on another wrong action (light terracotta,
  deliberately lighter).
- *Row 2 (dominant — the visual centre of the whole figure):* **93** correct
  decisions the Router touched → **41** kept (light grey) · **52 broken** (strong
  terracotta, the thickest and most saturated mark on the page). Beside it, in bold
  terracotta: *52 of 93 broken*.
- Provenance tag beneath the zone, small italic grey: *Phi-3.5 · historical
  protocol · row-level records*.

**A thin vertical rule** separates Zone B from Zone C. The two use different
protocols and must never read as one comparison.

**Zone C — "does the evidence license a repair claim?" (≈ 32% width, under
*evidence* and *licensed repair claim*).** A three-row mini forest plot, horizontal
axis *target-hit* from 0.4 to 0.9, with **one** dashed vertical line at **0.50**, the
only tick labelled:
- Qwen3-8B: point 0.731, interval [0.604, 0.846] → small label *licensed* (blue)
- Qwen3-4B: point 0.578, interval [0.456, 0.697] → *declined* (olive)
- Gemma-2-9b: point 0.630, interval [0.444, 0.815] → *declined* (olive)
- Provenance tag: *sealed protocol · criteria fixed before evaluation*.

## Hierarchy — weight marks by evidence strength

1. **Strongest:** repair ≠ preservation — the 52 broken of 93 (row-level). Largest,
   most saturated.
2. **Strong:** correctable ≠ licensable — the two declined intervals crossing 0.50.
3. **Weakest:** movement ≠ destination — the 46 of 153. Visibly present but lighter;
   it is a partial, descriptive dissociation.

## Text budget

At most **25 words** in the figure besides numbers, model names and axis labels. No
sentence longer than six words inside the figure. No title inside the figure — the
caption carries the message.

## Style

Font DejaVu Sans (or the paper's serif, applied consistently); 6–9 pt at final size.
Colours (colour-blind safe, print safe): ink `#1c1c1c`, muted `#6f6f6f`, faint
`#c9c9c9`, main blue `#2b5f75` (arrival / licensed), terracotta `#a3503c` (broken /
wrong destination), olive `#8a7b52` (declined). White background, no gridlines, no
drop shadows, no gradients, no icons.

## Hard rules — any violation is an error

- Every number comes from `final_evidence/FINAL_PAPER_EVIDENCE.csv`: `phi_net`,
  `phi_source_exits`, `phi_gold_arrival`, `phi_other_wrong`, `phi_exposed_correct`,
  `phi_broken_e2`, and `{q8b,q4b,gemma}_target_hit`, `_target_hit_ci_lo`,
  `_target_hit_ci_hi`. No number is typed by hand. (Do not read the LFS row files;
  they are unavailable.)
- **0.50 is the only boundary.** Never draw, label or mention 0.6042; it is Qwen3-8B's
  own interval bound, not a criterion.
- Never write *admitted*, *failed* or *uncorrectable* for Qwen3-4B or Gemma-2-9b. Use
  *declined*.
- Never write *repaired*, *repair rate* or *fixed* for any count. The only place
  *repair* may appear is the header's *licensed repair claim*.
- No significance stars or p-values. British spelling.
- Historical (Phi-3.5) and sealed (Qwen3 / Gemma) evidence stay visually separated.

## Deliverable

`figures_final/make_fig1_thesis.py` writing `figures_final/fig1_thesis.{pdf,png}`,
reading the CSV through `_style.load()`, and asserting at render time that
107 + 46 = 153 and 41 + 52 = 93.

## Caption to pair with it

> **Correction is not repair.** *Left:* the aggregate reports a net gain of +55 for a
> channel-keyed intervention on Phi-3.5. *Middle:* resolving every outcome shows
> that 46 of the 153 decisions moved off an error land on another wrong action, and
> that the intervention breaks 52 of the 93 correct decisions its Router touches
> (E2; 52 of all 264 correct decisions, E1) — historical protocol, row-level records.
> *Right:* under criteria fixed before evaluation, the evidence licenses a repair
> claim for one of three sealed settings; two with favourable point estimates are
> declined because their 95% intervals cross 0.50. A decline denotes insufficient
> evidence under the evaluated protocol, not an uncorrectable setting.
