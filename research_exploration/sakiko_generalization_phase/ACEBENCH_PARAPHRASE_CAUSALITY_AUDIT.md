# Paraphrase result — causality audit and self-correction

## The confound is real and was already documented

`PHASE4_DECISION_AND_NEXT_STEP.md:70`, citing `PARAPHRASE_ROBUSTNESS_REPORT.md §3`:

> *"the paraphrase's format directive is weaker than the official's, which inflates the raw gap.
> The licensed claim is only the one the gate required: **stability ≥0.80 was not demonstrated**
> — on any metric."*

The reformulated prompt did **not** hold formatting constraints fixed. The comparison therefore
does not isolate semantic paraphrase from prompt-interface effects.

## Correction to my own draft

`ACEBENCH_FRAMEWORK_TRANSFER_SECTION.md` as first drafted asserted:

> ~~"Restraint decisions are wording-fragile; calling is stable."~~

**That claim is not licensed by the archived evidence and is withdrawn.** The per-class
attribution (`ask_user` 0.36 vs calling 0.84–0.92) and the "+22 points toward calling" figure
are precisely the quantities the weaker format directive inflates.

## What survives, and it is not nothing

Three stability metrics were computed; **all three fail the 0.80 gate**:

| metric | value | gate |
|---|---:|---|
| raw stability | 0.56 | FAIL |
| **format-insensitive** | **0.79** | **FAIL** (narrowly) |
| latent-call | 0.65 | FAIL |

The **format-insensitive** metric is the one designed to neutralise exactly this confound, and it
still fails at 0.79. So the gate failure itself is robust to the format objection; what is *not*
robust is the mechanistic story about which modes are fragile and by how much.

## Licensed wording

> The preregistered reformulation-stability criterion failed (0.56 raw, 0.79 format-insensitive,
> 0.65 latent-call, against a 0.80 gate), with the largest instability observed on
> restraint/clarification modes. **The present test does not fully isolate semantic paraphrase
> from prompt-interface effects**, since the reformulated prompt carried a weaker format
> directive, so the per-mode magnitudes should be read as upper bounds on true semantic fragility.

## Consequence

The paraphrase result may be reported as a **prospectively-failed eligibility criterion** and as
a reason ACEBench was retired from the intervention line. It may **not** be reported as a
scientific finding about restraint behaviour in language models. Establishing that would require
the format-matched, multi-paraphrase study the archive itself specifies as future work
(`PHASE4_DECISION_AND_NEXT_STEP.md:120`).
