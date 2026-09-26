# Dataset scope and limits

## Design

`M` varies. `D` is held approximately fixed at the authoritative **When2Call** population and
ontology used by the modern Qwen3 protocol. This is deliberate: holding `D` fixed is what
isolates host-model variation and makes the cross-model question answerable at all.

## What this panel is

> **Prospective model-level replication on a fixed benchmark population.**

## What it is not

The When2Call evaluation population has already been used historically for other host models.
It is therefore **not** a new untouched population, and must never be described as one.

## What this panel cannot fix (stated before inference, per §22)

- single-dataset dependence
- cross-dataset generalisation
- natural-traffic generalisation
- deployment safety
- activation superiority over score-space
- mechanistic explanation

The manuscript must continue to state that **cross-dataset generalisation is unestablished**,
whatever this panel returns.

## Excluded populations

ToolDial, WildToolBench, FAIL-TaLMs, the original V1R3, and any further public Dataset-C search
are out of scope and may not be revived here.

## Per-model freezes

For each model, identity, readout, channel discovery, support rule, Router, estimator, layer
mapping, dose-selection rule, formal thresholds, seeds and controls are frozen **before** that
model's formal evaluation outputs are observed. No evaluation-side tuning, channel selection or
layer search.
