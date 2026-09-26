# Paper impact decision

## Effect on the manuscript: none

No claim strengthens, no claim weakens, no claim changes. The manuscript proceeds on exactly
the evidence base the final pre-submission audit adjudicated as
`ICLR_READY_FREEZE_AND_WRITE`.

## Model breadth remains

Two Qwen sizes formally (8B ADMIT, 4B DECLINE) plus Phi-3.5 retrospectively. **Modern
cross-family evidence remains absent**, which is what this panel existed to obtain and did not.

## What this means for the known weakness

Breadth was already ranked risk 2 in `REJECTION_RISK_REGISTER.csv`, classified
`FIXABLE_BY_WRITING (claim scope)` with `requires_experiment: NO`. That classification is
unaffected. The mitigation was never this panel — it was making the claim explicitly
existential and setting-scoped, and never writing "generalises". That mitigation is still
available and still sufficient for the scoped claim.

## Should the panel be retried?

That is the operator's call, and it turns on provisioning rather than science. Retrying
requires, at minimum: Gemma license acceptance on the HuggingFace account owning the token;
bandwidth several orders of magnitude above the measured 18.2 KB/s; ~40 GB free disk or a
planned sequential download-hash-evict; and an unresolved VRAM question on a 24 GB card that
only the §12 smoke test can settle.

Against a 38-day runway to the ICLR abstract deadline, of which ~24 days would be download
alone at the observed rate, **the honest recommendation is not to retry for this submission.**
The panel's own §21 already froze the answer for the case where it yields nothing: the
Qwen3-8B existential ADMIT remains valid, and the paper's strength is selectivity — two
conclusion-changing refusals — not breadth of positive correctability.

## Stop rule

§25 applies as written. Model experimentation is finished for this manuscript. This outcome
does **not** authorise opening a third family, a larger Qwen, a larger Gemma, a new channel, or
another dataset.
