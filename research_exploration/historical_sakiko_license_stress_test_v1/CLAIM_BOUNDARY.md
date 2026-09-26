# Claim boundary

| # | Statement | Classification |
|---|---|---|
| **A** | "Historical Phi was a failure." | **UNSUPPORTED** — it improved aggregate behaviour substantially and passed its own preregistered placebo battery. It fails one modern criterion that its protocol did not gate on. |
| **B** | "Historical Phi improved aggregate behaviour under the old protocol." | **SUPPORTED** — acc 48.18 → 58.21 (+10.04pp), Fixed 107 / Broke 52, Net +55, verified from committed artifacts. |
| **C** | "The modern SAKIKO licence would reject the frozen Phi result on at least one recoverable criterion." | **SUPPORTED** — collateral 52/264 = 0.1970, CP95 [0.1507, 0.2501], against a 0.05 point-and-interval bound; and 52/93 = 0.5591 on the eligible-at-risk denominator. |
| **D** | "The complete modern formal protocol has retrospectively been applied to Phi." | **MUST_NOT_CLAIM** — 6 of 18 registry criteria are `UNADJUDICABLE_FROM_HISTORY` (K = 59 randoms, add-one p, frozen bootstrap intervals, score-space comparator, sealed one-shot, modern Router metrics). |
| **E** | "Modern licensing changes which historical SAKIKO interventions count as successful correction." | **DEFENSIBLE_WITH_LIMITATION** — demonstrated on exactly one historical case; Qwen2.5, MetaTool, Mistral and Llama could not be re-adjudicated for stated structural reasons. |

## Mandatory labelling

This analysis is `RETROSPECTIVE_HISTORICAL_RE-ADJUDICATION`. It must never be described as
preregistered, prospective, confirmatory, sealed, or an independent replication. It shows that
a modern criterion **would have changed the interpretation** of a historical result; it does
not convert that result into a modern formal trial.

## Additional disclosures that travel with any use

- Phi's per-channel destination numbers come from a **three-router cascade** in which routers
  compete for rows (18 of 77 `ca→tc` errors were edited by the `ca→direct` direction), so they
  are `NOT_COMPARABLE` to a modern single-channel measurement.
- Phi's placebos broke 55–57 rows against the real arm's 52: collateral is a property of the
  intervention regime at α = 5–10, not of the real direction uniquely.
- No Phi checkpoint revision was recorded in 2026-03.
