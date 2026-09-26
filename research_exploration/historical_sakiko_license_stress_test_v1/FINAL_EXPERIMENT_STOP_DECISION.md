# Final experiment stop decision

## `FREEZE_EXPERIMENTATION_AND_WRITE`

This was the final scientific analysis before paper freeze, and it completed without
mechanical failure. **No further experiment is proposed.**

## What was used

CPU only. Committed per-sample artifacts and committed aggregates. No model inference, no GPU,
no activation extraction, no Router training, no new directions, no Qwen3.5, no Dataset-C
search, no When2Call test access, no random-direction batteries.

## Deliberately left missing

- Qwen2.5 per-sample locked-test records (git-LFS pointers) — **not fetched**.
- Mistral and Llama intervened per-sample predictions — **not regenerated**.
- Phi checkpoint revision — **not reconstructed**.
- K = 59 randoms, bootstrap intervals, score-space arm for Phi — **not retrofitted**.

Each is recorded as `UNADJUDICABLE_FROM_HISTORY` or `NOT_RECORDED`. Missing was left missing.

## The state the project is now in

The evidence base is closed:

- Qwen3-8B × When2Call → **FORMAL ADMIT** (10/10, sealed one-shot)
- Qwen3-4B × When2Call → **FORMAL DECLINE** (destination intervals)
- Phi-3.5 historical → **WOULD-DECLINE** on recorded collateral (retrospective, partial)
- Mistral, Llama → older-protocol corroboration, clearly demarcated
- MetaTool, Qwen2.5 → breadth only, not re-adjudicable
- Dataset C → `NO_VIABLE_PUBLIC_DATASET_C`

Remaining risk is dominated by writing, positioning against the 2026 literature
(2605.07990, When2Tool, PRISMS), and claim discipline — not by missing measurements.

## Standing prohibition

Do not open Qwen3.5, another dataset search, another intervention branch, or another
re-analysis. If manuscript drafting exposes a claim unsupported by existing artifacts, the
correct response is to **narrow the claim**, not to run an experiment.
