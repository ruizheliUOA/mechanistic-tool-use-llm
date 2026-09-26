# Claim-by-claim language audit

Scanned: `ABSTRACT_V1`, `INTRODUCTION_V2`, `METHODS_V1`, `RESULTS_V1`, `DISCUSSION_V1`,
`CONCLUSION_V1`, `DRAFT_RELATED_WORK`, `LIMITATIONS_FINAL`.

| # | forbidden implication | occurrences | status |
|---|---|:-:|---|
| 1 | W2C gold is human-curated | 0 | Methods 2.5 states *Synthetic / Automated*; corrected in an earlier phase |
| 2 | W2C gold independently validated by SAKIKO | 0 | Methods 2.5 and Limitation 4 state explicitly that it was not |
| 3 | Qwen3.5 intrinsically non-actionable | 0 | Discussion 6.3 — "allocation-sensitive adjudicability, not intrinsic non-correctability" |
| 4 | Qwen3.5 `ca→*` robustly unsupported | 0 | 56.7% eligibility reported wherever the stop is mentioned |
| 5 | Gemma / Qwen3-4B intrinsically uncorrectable | 0 | Discussion 6.2 — DECLINE attaches to a tuple, not an ontology |
| 6 | Gemma / Qwen3-4B merely underpowered positives | 0 | Discussion 6.2 — "we do not, and the evidence does not permit it" |
| 7 | Qwen family more correctable | 0 | Discussion 6.7 |
| 8 | scale predicts correctability | 0 | Discussion 6.7 — ordering is non-monotonic |
| 9 | Llama cleanly shows steerable-not-correctable | 0 | Results 3.4 — specificity failed first; not used as an example |
| 10 | Qwen3-8B safety certified | 0 | Results 3.1 discloses 0/6 and 0.393 inline; Discussion 6.4; Conclusion |
| 11 | historical collateral = exposure-conditional collateral | 0 | E1/E2 labelled in every preservation sentence |
| 12 | ACEBench is a second formal benchmark | 0 | Methods 2.5, Results 3.7, Discussion 6.6, Abstract all say *instantiation* |
| 13 | the licence is calibrated | 0 | Methods 2.4; Limitation 7 |
| 14 | 0.05 universally optimal | 0 | Methods 2.4 — "we do not reconstruct provenance we do not have" |
| 15 | destination changed decisions beyond specificity | 0 | Discussion 6.1 and 6.5 state the opposite explicitly |
| 16 | activation generally superior to score-space | 0 | Limitation 11 |
| 17 | every rung has an isolated empirical example | 0 | Ladder audit records Correctable as definitional; Llama excluded |
| 18 | all modern evidence is multi-seed end-to-end | 0 | Methods 2.6 |
| 19 | 59 random directions = 59 pipeline seeds | 0 | Methods 2.6 states this in bold; Results 3.6 repeats it |

**No occurrences found. No corrections required in the current drafts.**

Two claims were corrected in earlier phases and do not appear: "W2C gold is human-curated" and
"restraint modes are wording-fragile."

## Standing wording risks for LaTeX assembly

1. "Framework transfer" must never become "validated on two benchmarks."
2. Never write a bare "collateral rate" — always E1 or E2, with exposure n.
3. Retain the `n_random = 1 for three of five seeds` caveat wherever the historical 5-seed result appears.
4. Keep the ACEBench parser figure as "0.98% on the audited subset."
5. Keep the 30-reallocation result labelled retrospective in every appearance.
