# CI procedure — correction to my own reported numbers

## What I found

The frozen SAKIKO confidence intervals are **10,000-draw percentile bootstraps
resampled over channel errors**, not Clopper–Pearson intervals on source exits.
I have been quoting Clopper–Pearson values as though they were the frozen ones.

## Proof

Reproducing the frozen procedure for Qwen3-8B — bootstrap over the 72 routed
channel errors, outcome vector (38 gold / 14 other-wrong / 20 source-retained),
10,000 draws:

| | 2.5th pct | 97.5th pct |
|---|---:|---:|
| **my reproduction** | **0.604639** | **0.846154** |
| **frozen artifact** | **0.604157** | **0.846154** |
| Clopper–Pearson on 38/52 | 0.589757 | 0.844317 |

The reproduction matches the frozen value to Monte-Carlo noise. Clopper–Pearson
does not.

Corroborating detail: the frozen Qwen3-8B upper bound is exactly 44/52, but the
lower bound (0.604157 × 52 = 31.42) is **not** on the 1/52 lattice — impossible
for a bootstrap over exits, expected for a bootstrap over channel errors where
numerator and denominator both vary. Gemma's frozen endpoints (12/27, 22/27)
*are* on the lattice, which is consistent — with only 27 exits the resampled
values land on lattice points far more often.

## Corrected table — frozen values are authoritative

| model | target-hit | **frozen CI (bootstrap)** | my CP (superseded) | clears 0.50 |
|---|---:|---|---|---|
| Qwen3-8B | 38/52 = 0.7308 | **[0.6042, 0.8462]** | [0.5898, 0.8443] | **YES** |
| Gemma | 17/27 = 0.6296 | **[0.4444, 0.8148]** | [0.4237, 0.8148] | no |
| Qwen3-4B | 37/64 = 0.5781 | **[0.4559, 0.6970]** | [0.4482, 0.7006] | no |

## Impact on conclusions: none

Every verdict is unchanged. Clopper–Pearson is uniformly the more conservative
of the two here, so it moved every bound away from the threshold in the
direction that would only ever make a PASS harder — and Qwen3-8B passed under
both. Gemma and Qwen3-4B fail under both.

The one-sided binomial p-values I reported (0.0006 / 0.1239 / 0.1302) are exact
tests, independent of the interval procedure, and stand unchanged.

## Impact on my power calculations: one is wrong

`PREREGISTRATION_LEVEL2.md` characterises the historical rule ("Rule B") as
*"two-sided 95% CI lower > 0.50"* and derives **n = 199, k ≥ 114** from
Clopper–Pearson. That derivation used the wrong interval family. The historical
rule is a bootstrap over channel errors, which is **less conservative**, so the
true Rule-B requirement is **smaller than 199**.

This does **not** affect the prospective design, which uses Rule A — a one-sided
exact binomial, **n = 158, k ≥ 90, power 0.8057, type-I 0.0472** — verified
independently in this pass. Rule A was deliberately chosen as the prospective
rule and is unaffected.

**Action:** the Rule-B figure in `PREREGISTRATION_LEVEL2.md` is marked
superseded. It was never load-bearing — it appears only as the comparison
showing Rule A and Rule B differ — but a wrong number in a preregistration is
not acceptable, and the second-ADMIT power model (`n = 158`) does not depend on
it.

## Verified correct in the same pass

- adapter regression on Qwen3-8B transition-format records: gold 38, exits 52,
  other 14 — exact match
- preservation 0/6 at record level, all six baseline-correct rows Router-fired
- minimum n for 0 breaks at α=0.05: **59** (0/58 → 0.0503, 0/59 → 0.0495)
- Gemma McNemar 17:0 → p = 7.629e-06 = 0.5^17 exactly
- add-one p for 0/59 randoms = 1/60 = 0.016667, matches frozen 0.0166666…
- Q2 sizing: n=158, k≥90, power 0.8057, type-I 0.0472
