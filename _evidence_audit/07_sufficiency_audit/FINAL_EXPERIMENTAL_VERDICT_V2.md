# Final experimental verdict

## The four sufficiency questions, answered separately

```
A. INTERNAL VALIDITY      STRONG
B. CLAIM SUFFICIENCY      ADEQUATE   (for the thesis as written)
C. EXTERNAL VALIDITY      PARTIAL
D. SAFETY CERTIFICATION   WEAK
```

A paper strong on A and adequate on B is publishable when C and D are scoped accordingly — which
they are, in fifteen explicit limitations.

## Scorecard

**6 STRONG · 6 ADEQUATE · 3 PARTIAL · 3 WEAK.**

The three WEAK dimensions are preservation, calibration and mechanism. **None threatens the
central thesis**, because the paper claims none of them: it reports that no setting certifies
preservation, that the licence is a conservative evidential rule rather than a calibrated
classifier, and that no mechanism was established. Each WEAK rating corresponds to a claim the
paper explicitly declines to make.

## Blockers

**None.** Of thirteen catalogued weaknesses: 5 `ALREADY_SUFFICIENT`, 5 `MUST_REMAIN_LIMITATION`,
1 `NOT_REQUIRED_FOR_CURRENT_CLAIM`, 1 `SAFE_TO_STRENGTHEN`, 1 `REQUIRES_NEW_PROSPECTIVE_STUDY`.
Zero marked as blocking.

## More end-to-end seeds?

```
MORE_FULL_PIPELINE_SEEDS_NEEDED: NO
```

Decoding is deterministic (byte-identical replay). The Router is conditionally deterministic —
coefficient difference exactly `0.000e+00` and pairwise cosine `1.000000000000` across 20
`random_state` values on identical data. Directions are deterministic functions of frozen TRAIN
populations. Channel, τ, dose and thresholds are frozen. Intervention-direction variance is
covered by K=59 matched randoms per trial; construction variance by 30 group-preserving
reallocations. **The only material stochastic source was allocation, and it has been quantified.**
Rerunning the formal pipelines under new seeds would re-execute a deterministic function and
consume a sealed population for no information.

## Counterfactual value of the three possible additions

| addition | value | why |
|---|---|---|
| **A. blinded W2C human gold audit, high agreement** | **MODERATE** | closes the last "we did not check"; cheap; addresses the only repairable weak dimension |
| B. prospectively preservation-certified setting | **MAJOR** — but requires a new study | needs ≥59 exposed correct rows; not reachable by re-analysis |
| C. second independent benchmark formal licence | **MAJOR** — but not currently possible | six screened populations, none eligible; would need a new instrument |

**Highest value-to-risk ratio: A.** B and C are both MAJOR in value and both require new
prospective data collection that cannot be completed on any near timeline; A is MODERATE in value
and completable in days by two human annotators.

## Decision

```
B. CURRENT EVIDENCE IS SUFFICIENT, BUT ONE OPTIONAL HIGH-VALUE VALIDATION
   SHOULD BE COMPLETED BEFORE SUBMISSION
```

**The one action: the blinded W2C human gold audit.** 150–200 rows stratified by gold class ×
domain × baseline correct/error, two annotators blind to model predictions and channel identity,
third-party adjudication, κ reported. Sample and analysis plan frozen before any label is seen.

It is **optional, not paper-critical.** If the team declines it, the paper stands on the current
limitation statement, which is accurate. If it is run, the team must accept in advance that a
result showing systematic disagreement on `cannot_answer` or `request_for_info` would require
re-scoping — commissioning it only if the answer is favourable would defeat its purpose.

I cannot execute it. It requires human annotators, and no simulated labels have been or will be
produced.

## Everything else

Stop validating. The record is sufficient for the thesis as written, and the correct response to
the remaining gaps is to keep the claims stopping exactly where the evidence stops — which the
current drafts already do.
