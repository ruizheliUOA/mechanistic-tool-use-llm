# When2Call gold provenance — audit

Source: official `nvidia/When2Call` dataset card, CC BY 4.0.

## Findings

| question | answer |
|---|---|
| data collection method | **Synthetic** (stated on the dataset card) |
| labeling method | **Automated** (stated on the dataset card) |
| gold action classes | tool call · follow-up question · cannot answer — **three**, confirming `direct_answer` is a distractor option, never gold |
| inter-annotator agreement | **not reported** |
| human verification of labels | **not reported** |
| QC procedure | **not reported** |
| test split human-verified | **not reported** |
| generation scripts published | yes, in the official GitHub repository |
| independent re-audit by SAKIKO | **NONE** |

## Correction to earlier project wording

An earlier draft in this project described When2Call gold as *"human-curated."* **That was
incorrect and is withdrawn.** The dataset card states the labels are synthetically generated and
automatically assigned.

## What this does and does not undermine

**Does not undermine:** the *readout* validation, which is extensive and model-side — 0 unparsed,
0 exact-tie margins, 0 exclusions across 3098–3104 rows in four independent models, with
deterministic replay. That evidence concerns our instrument, not the benchmark's labels, and it
stands.

**Does undermine:** any claim that the benchmark's gold is externally validated. It is not, by
its own documentation.

**Mitigating consideration, stated without overreach:** the labels are *programmatically*
generated from the construction pipeline, not produced by an LLM judge at evaluation time. The
failure mode arXiv 2607.02577 documents — evaluator error rates of 9.8–30.5% across BFCL v4,
τ²-Bench, MCP-Atlas and LiveMCPBench — concerns *evaluators*, which is a different object. That
paper does not audit When2Call, so it provides neither support nor evidence against it here.

## Required limitation wording

> When2Call gold labels are synthetically generated and automatically assigned, with no
> inter-annotator agreement or human verification reported by the dataset authors, and we did
> not independently re-annotate them. Our validation is of the model-side readout, not of the
> benchmark's labels.

## Optional future work — specified, not executed

A blinded stratified human audit (~150–200 rows stratified by gold action × domain × baseline
correct/error) with two annotators, adjudication protocol and Cohen's κ would close this. It is
**optional pre-submission validation, not a blocker**, and Claude cannot perform it — it requires
real human annotators. No such audit has been run and none is claimed.
