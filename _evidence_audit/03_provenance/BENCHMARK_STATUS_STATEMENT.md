# What we can honestly say about each benchmark

I have been imprecise. "Not up to scratch" conflated three different judgements. Here they are
separated.

## The three questions, kept apart

1. **Is the benchmark itself sound?** (gold quality, construct validity)
2. **Can SAKIKO measure on it?** (readout validity)
3. **Can it carry a formal licence outcome?** (channel support + preservation exposure)

A benchmark can pass 1 and 2 and fail 3. That is what happened, and it is not the same as the
benchmark being poor.

## When2Call — the licence benchmark

| question | status |
|---|---|
| benchmark sound? | **PARTIALLY ESTABLISHED — CORRECTED.** NVIDIA-published, explicit multiclass first-action gold, deterministic readout. But the dataset card states **Data Collection Method: Synthetic; Labeling Method: Automated**, with no reported inter-annotator agreement, human verification or QC. My earlier description of it as *human-curated* was **wrong**. |
| SAKIKO measurement valid? | **Yes, demonstrated four times.** Across Qwen3-8B, Qwen3-4B, Gemma-2-9B and Qwen3.5-9B: **0 unparsed, 0 exact-tie margins, 0 exclusions** on 3098–3104 rows each, with deterministic replay. |
| carries licence outcomes? | **Yes — three prospective sealed trials.** |

**One honest gap to disclose:** we did not perform an independent audit of When2Call's *gold
labels*. Given [2607.02577] found 9.8–30.5% evaluator error across four other tool-calling
benchmarks, a reviewer may fairly ask. What we can say is that our readout is deterministic and tie-free on it, and that gold is
programmatically generated rather than LLM-judged at evaluation time. We can **not** say it is
human-verified. We should state this limitation rather than
imply a validation we did not run.

## ACEBench — the second instantiation

| question | status |
|---|---|
| benchmark sound? | **Yes.** Native 4-class ontology, benchmark-native gold. |
| SAKIKO measurement valid? | **Yes — `ACEBENCH_MEASUREMENT_PARTIAL`, 9/10 criteria locked in advance.** Accuracy 0.7562, macro-F1 0.6868, 100% replay determinism, 0.98% parser mislabel. PARTIAL only because the 102-row parser audit's stratification is not documented and the paraphrase test was format-confounded. |
| carries licence outcomes? | **No.** Support fails (0 of ≥2 channels qualify) and it is a prior development corpus. |

**ACEBench is an accurate benchmark on which we can measure accurately. It cannot carry a
licence.** Those are different sentences and the paper should use the precise one.

## The screened populations

None is "bad." Each fails a *specific structural requirement* of destination-resolved licensing:

| population | what it is | why it cannot carry a licence |
|---|---|---|
| MetaTool | sound binary benchmark | `\|A_D\{g,s}\| = 0` — the question is undefined, not unanswered |
| RUT-Bench | sound, well-built, 59 executable environments | `\|A_D\| = 1` — varies the *user*, not the action |
| AppWorld-UL | strong: programmatic assertions, open licence, ICML 2026 | interaction type is a *trajectory* property, not a first-action label; 34 root scenarios |
| UserToolBench | legitimate personalisation benchmark | ~300 rows, 10 persona groups |
| ToolDial | — | construct confound: 15 action sequences / 11,111 dialogues |
| WildToolBench | strong native labels | model-side interface ≠ native gold space |

## So — do we have more than one benchmark?

**Yes, and here is the exact wording:**

> Formal licence outcomes are established on one benchmark (When2Call), on which our readout is
> tie-free, exclusion-free and deterministic across four models. The measurement and
> channel-discovery apparatus is additionally instantiated on a second benchmark with an
> independent action ontology (ACEBench), where it satisfied 9 of 10 pre-locked criteria before
> terminating at adjudicability. We further characterise the eligibility of six additional
> populations and report the structural requirement each fails.

That is three distinct benchmark-level statements, all true, all backed by committed artifacts.
It is not "one benchmark," and it should not be written as if it were.

## What I should have said earlier

Not "the benchmarks aren't up to scratch." The accurate sentence is:

> **Pre-execution multiclass action gold with adequate per-channel support is rare in the
> published field, and we document six populations that fail one of the structural requirements.**

That is a finding, not an apology — and it is the sentence to put in the paper.
