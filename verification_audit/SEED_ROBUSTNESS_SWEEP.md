# Seed-robustness sweep — do any verdicts depend on the bootstrap seed?

Prompted by finding that one reported CI bound was seed-dependent. If the
bootstrap seed can move a bound across its decision threshold, every historical
verdict is in question. So I tested all of them.

## Method

10 seeds (42, 1, 7, 123, 999, 20260819, 55555, 31337, 2024, 88) on the
authoritative bootstrap, with unrouted channel errors padded as
`SOURCE_RETAINED`. Both interval conditions, all three formal settings.

## Result — **no verdict is seed-dependent**

| setting | statistic | CI lower across 10 seeds | spread | threshold | all seeds |
|---|---|---|---:|---:|---|
| Qwen3-8B | target-hit | [0.6038, 0.6078] | 0.0041 | 0.50 | **PASS** |
| Qwen3-8B | Target Gain | [0.1149, 0.1264] | 0.0115 | 0.00 | **PASS** |
| Gemma | target-hit | [0.4375, 0.4444] | 0.0069 | 0.50 | **FAIL** |
| Gemma | Target Gain | [−0.0312, −0.0312] | 0.0000 | 0.00 | **FAIL** |
| Qwen3-4B | target-hit | [0.4545, 0.4583] | 0.0038 | 0.50 | **FAIL** |
| Qwen3-4B | Target Gain | [−0.0484, −0.0403] | 0.0081 | 0.00 | **FAIL** |

Every margin is far larger than its seed spread:

- Qwen3-8B target-hit clears 0.50 by **~0.104**, spread 0.004 — a **25×** margin
- Gemma misses 0.50 by **~0.058**, spread 0.007 — **8×**
- Qwen3-4B misses 0.50 by **~0.044**, spread 0.004 — **11×**

## Conclusion

The seed affects reported **decimals**, not **decisions**. The ADMIT is an ADMIT
under every seed tested; both DECLINEs are DECLINEs under every seed tested.

The seed-sensitivity defect found earlier was real and the reporting requirement
it produced stands — but it does **not** propagate into any historical verdict.

## What this does not cover

Only the two interval conditions. Conditions 1, 4, 5, 7, 8, 9, 10 are point,
count, or exactness tests with no bootstrap and therefore no seed dependence:

- specificity (condition 4) is `randoms_ge_real = 0`, a count
- collateral (7, 8) uses Clopper–Pearson, which is deterministic
- zero-exactness (9) is an identity check
- support (1) and point thresholds (2, 5) are direct comparisons

So the seed can only ever touch conditions 3 and 6, and it does not move either.
