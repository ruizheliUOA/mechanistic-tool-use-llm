# The minimal benchmark-agnostic core

## Six constructs replace ten conditions

| # | construct | estimand | invariant across action spaces? |
|---|---|---|:-:|
| A | **Adjudicability** | support counts; `\|A_D\|≥3`; deterministic gold | **YES** — counting |
| B | **Specificity** | add-one empirical p vs matched-random null; reverse non-reproduction | **YES** — rank-based, unitless |
| C | **Destination** | `P(→g \| exit) ` and `P(→g) − P(→A_D\{g,s})` | **YES** — probabilities over the native action set |
| D | **Preservation** | `P(correct→wrong \| exposed to I)` | **YES** — conditional probability |
| E | **Evidence** | interval or risk-control bound on C and D at level δ | **YES** |
| F | **Yield** | `P(gold arrival \| eligible channel error)` | **YES** |

All six are **probabilities, ranks or counts** — dimensionless and defined for any `K`-class
action space. None references When2Call.

## Mapping the frozen 10 onto the 6

C1 support→A · C4 random null→B · C5, C2 destination point→C · C7 collateral point→D ·
C3, C6, C8 intervals→E · C11 non-vacuity→**precondition of D** · C9 zero exactness→**instrument
validity, part of A** · C10 structural→A.

**Ten conditions collapse to six constructs with no loss of observed coverage.** The pathology
battery confirms it: the three sole-catchers (random null, zero control, non-vacuity) map to B,
A and D respectively.

## The gap the audit found, fixed conceptually

The 10-condition licence admits a near-zero-yield intervention (8 exits from 200 errors, all
gold). **F closes it**, and F is not an arbitrary new threshold: it is the quantity the frozen
formal conjunction accidentally omitted when its support condition counted channel *errors*
while the DEV gate counted *exits*. F restores an existing DEV construct to the formal stage.

Note F ≠ C. Target-hit is `gold/exits` (precision when acting); yield is `gold/errors`
(coverage). The pathology passes C at 1.0 and fails F at 0.04. They are genuinely distinct.

## Verdict

```
SMALLER_CORE_SUFFICIENT AND MORE GENERAL
```

Six constructs, one of which (F) repairs a demonstrated gap. This is a simplification that
*increases* coverage — the strongest possible form of the argument for restructuring.
