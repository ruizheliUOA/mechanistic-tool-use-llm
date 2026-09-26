# Qwen3.5-9B — scientific interpretation

## The four-axis classification

**MODEL.** Same lineage (Qwen), newer generation, 8.95B text parameters, hybrid 3:1 Gated
DeltaNet / full attention. Highest baseline accuracy in the panel (0.4533), least predicted-class
collapse (0.6633), errors closest to the decision boundary. On every naive indicator it is the
*most* favourable host model tested.

**CORRECTION DESIGN.** Untested. Router at L_obs 23 was fitted and declined; no estimator,
site, dose or surface was ever exercised. The estimator×site control that Qwen3.5 was meant to
resolve prospectively remains unresolved.

**BENCHMARK.** When2Call, held fixed. `direct` is never a gold label, so the action space is
effectively 3-way for gold with a 4-way candidate set. Channel support is entirely determined by
the model's own outcome distribution over three gold classes.

**LICENCE.** Never reached. No destination, collateral, specificity or precision quantity exists.

## The mechanism, stated at the level the evidence supports

Qwen3.5's accuracy gain is **unevenly distributed across gold classes**. `request_for_info`
recall roughly doubles relative to the other models (0.326 vs 0.170 / 0.132) while
`cannot_answer` recall *falls* to the lowest in the panel (0.087). Because reference and error
rows within a gold class are complementary and compete for a fixed number of rows, this
redistribution:

1. **starves** the reference side of every `cannot_answer → *` channel — DEV reference 29,
   against a frozen minimum of 30; and
2. **enriches** the one channel, `request_for_info → tool_call`, that three independent models
   now agree is not linearly readable at the observation site.

So the informative channels became untestable and the testable channel became uninformative.
Both halves follow from a single baseline property. Nothing about intervention is implicated.

## Answering the panel's motivating question

> Does the strong correctability observed in Qwen3-8B persist in a newer Qwen generation?

**Unresolved, and unresolvable from this experiment.** The Qwen3-8B ADMIT rests on
`cannot_answer → tool_call`. That channel is not admissible in Qwen3.5 — not because it is
hard to correct, but because Qwen3.5 answers `cannot_answer` correctly too rarely to supply a
reference population. The comparison the panel was designed to make cannot be made.

Against the listed candidate explanations:

| candidate | verdict |
|---|---|
| model-family / generation effect | **not evidenced** — the test never ran |
| representation / logit scaling | **not evidenced** — raw norms differ (L_obs MLP-output median 28.97 vs Gemma 3.75) but cross-architecture raw norms are not comparable and no intervention was normalised against them |
| **channel composition** | **SUPPORTED — this is the operative factor** |
| Router exposure | contributory: at τ=0.4 the Router fired on 119 channel errors against 121 baseline-correct and 93 off-channel rows |
| estimator compatibility | **untested** |
| observation/write-site mismatch | **untested** |
| intervention dose | **untested** |
| destination geometry | **untested** |
| statistical support / power | contributory at the support gate; never reached the licence |
| licensing thresholds | not implicated — no threshold was near-miss except the support gate, by one row |

## The one row

`cannot_answer → *` failed on DEV reference support of **29** against a frozen minimum of
**30**. It would have been trivial to justify 29 as "materially equivalent." It was not
adjusted. The gate exists to be binding when it is inconvenient, and this is the case that
tests whether it is. Recording the near-miss explicitly is the honest treatment; moving the
threshold would have invalidated every prior refusal in the project.

## What this changes in the paper

It adds a third prospective host-model instantiation and — more valuably — an unusually clean
**channel-intrinsic** result: across Qwen3-8B, Gemma-2-9B and Qwen3.5-9B, `cannot_answer → *`
is strongly readable (0.92–0.94) and `request_for_info → tool_call` is not (0.71–0.77). That is
a property of the error channel, replicated across two model families, and it is the most
reproducible quantity the panel produced.

## What it does not change

The formal evidence base is still a single benchmark. A Qwen3.5 ADMIT would not have fixed
that, and its absence does not worsen it. **The single-benchmark limitation stands exactly as
before**, and no result in this study may be used to soften it.
