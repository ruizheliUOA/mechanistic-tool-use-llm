# Pathology coverage — does each condition guard a real failure mode?

Nine synthetic intervention outcomes through the exact frozen conditions.
Full matrix: `LICENCE_PATHOLOGY_COVERAGE_MATRIX.csv`.

| pathology | ADMIT? | caught by |
|---|:-:|---|
| healthy correction (Qwen3-8B-like) | **YES** | — (correctly admitted) |
| high Net but real ≈ random | no | **C4 random null (sole)** |
| high target-hit but huge collateral | no | C7, C8 collateral |
| **high TG with almost no exits** | **YES** | **— nothing. Gap.** |
| many exits, wrong destination | no | C2, C3, C5, C6 destination |
| zero collateral because zero exposure | no | **C11 non-vacuity (sole)** |
| strong effect, tiny sample | no | C1 support, C8 |
| modest true effect, adequate n | no | C3, C6 CI (the Gemma/4B case) |
| broken zero control | no | **C9 zero exactness (sole)** |

**No condition is inert.** Every one of the ten catches at least one pathology, and three are
sole catchers: the random null, the zero control, and non-vacuity.

## The gap this found

**An intervention that fires on almost nothing but is perfect when it fires passes the formal
licence.** 200 channel errors, 8 exits, all 8 to gold: target-hit 1.0, Target Gain 0.04,
CI lower > 0, zero collateral. ADMIT.

The DEV gate has `source_exits_min = 10`, but the **formal** conjunction's support condition
counts channel *errors* (≥30), not *exits*. So a low-yield, high-precision intervention is
licensed as a correction. Whether that is wrong is arguable — it *is* correct when it acts — but
the licence currently makes no minimum-repair-rate demand, and the paper should not imply it does.
