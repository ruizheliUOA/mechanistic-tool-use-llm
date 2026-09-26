# Mistral Arm F (manual cascade) — cascade-level placebo

Matched to the archived Qwen cascade placebo (same protocol, n_random=10).

| variant | Fixed | Broke | Net |
|---|---|---|---|
| real | 88 | 6 | **+82** |
| reverse | 80 | 8 | **+72** |
| ungated | 109 | 130 | **-21** |

**Random (10 dirs):** mean 70.1 ± 19.9 (max 119, min 45); real 82; real ≥ all random: **False**; #random ≥ real: **1**; percentile 90; **marginal over random mean: +11.9**.
