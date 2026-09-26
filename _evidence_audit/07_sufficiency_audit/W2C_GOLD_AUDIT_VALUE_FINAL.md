# The W2C gold audit — final value assessment

## The gap, precisely located

Model-side readout validity is **strong** and independent of this gap: 0 unparsed, 0 exact ties,
0 exclusions across 3098–3104 rows in four models, with deterministic replay. None of that
verifies that the benchmark's *gold labels* are correct.

Official metadata: **Data Collection Method: Synthetic. Labeling Method: Automated.** No
inter-annotator agreement, human verification or QC reported. We did not re-annotate.

## Which validity does this threaten?

| dimension | threatened? |
|---|---|
| A. internal intervention validity | **NO.** Every effect is measured as a *change* relative to the same labels. Label noise is common to baseline and intervened arms and cannot manufacture a destination shift or defeat a matched-random null. |
| B. construct validity | **PARTIALLY.** If gold is systematically wrong for a class, "GOLD_ARRIVAL" is misnamed for those rows. |
| C. external validity | **NO** beyond what is already conceded. |
| D. benchmark confidence | **YES** — this is where it bites. |

```
W2C_GOLD_VALIDATION_GAP: MEANINGFUL_BUT_NONFATAL
```

Nonfatal because the paper's claims are about *evidence for correction relative to a stated
gold*, not about the gold being philosophically correct. Meaningful because a reviewer aware of
the 2026 evaluator-validity literature will ask, and "we did not check" is a weaker answer than
a number.

## What an audit would and would not do

**Would address:** whether the automated labels agree with human judgement on the channels the
paper's claims rest on.

**Reassuring:** κ ≥ 0.8 overall with no class systematically below. **Qualifying:** κ 0.6–0.8, or
one class notably weaker — reportable with a scoped caveat. **Seriously damaging:** systematic
disagreement on `cannot_answer` or `request_for_info`, the gold classes of every channel in the
paper. That outcome would require re-scoping, and the team should be willing to accept it before
commissioning the audit.

## Protocol, if run

Blind annotators to: model predictions, baseline correctness, intervention outcomes, and which
channel a row belongs to. They see only the user query, the tool schemas, and the four candidate
actions.

Stratify by gold class × domain/scenario group × baseline correct-vs-error. Do **not** stratify by
"ambiguity" unless it can be defined without reference to model behaviour — otherwise it is
outcome shopping.

Two annotators is the minimum for κ and is adequate for a validity check; a third adjudicates
disagreements. 150–200 rows gives a κ standard error near 0.05, sufficient to distinguish 0.6
from 0.8. Freeze the sample and the analysis plan before any label is seen.

```
HUMAN_GOLD_AUDIT: HIGH_VALUE_PRE_SUBMISSION
```

Not paper-critical: the current limitation statement is accurate and the paper stands with it.
High value because it converts the single remaining "we did not check" into a measured number,
at low cost, addressing the weakest dimension in the scorecard that is *repairable at all*.

**Claude cannot perform this.** It requires real human annotators. No simulated labels were or
will be produced.
