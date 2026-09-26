# Historical control mapping

Phi's 2026-03 placebo battery, and what each control does and does not correspond to.

| historical control | recorded result (Net) | modern analogue | equivalent? |
|---|---:|---|---|
| `REAL_LOCKED` | **+55** (Fixed 107 / Broke 52) | real `d_grad` arm | yes |
| `RANDOM_DIR` | +14 (71 / 57) | matched-random | **partially** — a single random draw, not K = 59 fresh matched-norm directions, and compared on Net rather than `target_gain_rate`. No add-one p-value is computable. |
| `REVERSE_DIR` | +14 (71 / 57) | reverse `−d` | **yes in kind**, Net-based rather than destination-based |
| `WRONG_LAYER` (L4) | +3 (66 / 63) | wrong-layer control | **yes in kind**; note L4 vs the modern practice of using the other frozen site |
| `MISMATCHED_CHANNEL` | −17 (38 / 55) | *no modern analogue* — the modern protocol labels mismatched directions `NON_ORTHOGONAL_MISMATCHED_CONTROL_NOT_INTERPRETABLE` | **no** |
| unrouted rows (0 of 255 changed) | — | zero control | **yes in effect**, though not run as a declared arm |
| — | absent | K = 59 fresh randoms with add-one p | **not recorded** |
| — | absent | frozen score-space comparator | **not recorded** |

## What this establishes

On its own Net-based definitions, Phi's direction **passed a strong specificity battery**:
the real arm's Net is roughly 4× the best placebo, wrong-layer retains 5% of the benefit,
and a mismatched direction is actively harmful.

## The nuance that must not be omitted

The placebos break almost as many rows as the real direction: `RANDOM_DIR` 57,
`REVERSE_DIR` 57, `MISMATCHED` 55, real **52**. Collateral is a property of the intervention
*regime* at α = 5–10, not of the real direction specifically. The real direction is
distinguished by how much it **fixes**, not by how little it breaks.

This is precisely why a Net-based specificity battery cannot substitute for a collateral
bound: every arm was damaging, and Net concealed it behind a larger repair count.

## The scientific contrast this licenses

> **Phi passed strong historical direction-specificity controls while failing a modern
> collateral licence.**
>
> Steerability evidence ≠ licensed correction.
