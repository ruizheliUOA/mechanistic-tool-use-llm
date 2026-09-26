# Manuscript consistency audit

Cross-checked METHODS_V1, RESULTS_V1, INTRODUCTION_V1, DRAFT_RELATED_WORK, FINAL_CLAIM_MATRIX,
LIMITATIONS_FINAL against PAPER_NUMBER_PROVENANCE.csv.

## Forbidden statements — scan result

| forbidden implication | found? | where handled |
|---|---|---|
| W2C human-curated gold | **NO** (previously present, corrected) | Methods 2.5 states Synthetic/Automated; Limitation 4 |
| Qwen3.5 intrinsically non-actionable | NO | Results 3.3 states allocation-sensitive explicitly |
| Gemma intrinsically uncorrectable | NO | Results 3.2 "not really correctable, not simple power failure" |
| Qwen family scaling | NO | absent from all sections |
| Llama clean formal steerable-not-correctable | NO | Results 3.4 states specificity failed first |
| historical ADMIT = safety certified | NO | Results 3.1 discloses 0/6, upper 0.393 inline |
| ACEBench = second licence benchmark | NO | Methods 2.5 and Results 3.7 both state framework-level only |
| destination changed decisions beyond specificity | NO | Results 3.5 states plainly it did not |
| licence is calibrated | NO | Methods 2.4; Limitation 7 |
| thresholds optimal | NO | Methods 2.4 "we do not reconstruct provenance we do not have" |
| activation beats score-space | NO | Limitation 11 |

**No contradictions found.** Two prior errors were corrected in earlier phases and do not appear
in the current drafts: the "human-curated" description of W2C gold, and the "restraint modes are
wording-fragile" paraphrase claim.

## Residual wording risks to watch during full drafting

1. Do not let "framework transfer" drift into "validated on two benchmarks."
2. Keep E1 and E2 labelled in every preservation sentence; never a bare "collateral rate."
3. Retain the `n_random=1` caveat wherever the 5/5 historical seed result appears.
4. Keep the ACEBench parser figure as "0.98% on the audited subset."
5. Keep 30-resplit results retrospective; never adjacent to a formal verdict without that label.
