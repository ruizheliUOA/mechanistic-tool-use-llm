# Benchmark evidence — side analysis

CPU-only. No dataset was re-audited, re-opened or modified; every row restates a prior
adjudication from its committed source. **No new benchmark or model is launched by this file.**

## Current standing

| population | multiclass first-action gold | wrong-to-wrong destinations | channel support | baseline-correct exposure | independent of prior development | formally eligible |
|---|---|---|---|---|---|---|
| **When2Call** | YES (3 gold classes; `direct` is a distractor only) | YES | YES | YES | **NO** — consumed by the Qwen3-8B ADMIT and the one-shot 4B / Gemma DECLINEs | **consumed** |
| MetaTool | binary — `\|A\{g,s}\| = 0` | **NO** | n/a | n/a | NO — prior SAKIKO development corpus | NO |
| ACEBench | partial | unclear | untested | untested | NO — prior readout-audit population | NO |
| ToolSandbox | partial | unclear | untested | untested | NO — prior development corpus | NO |
| WildToolBench | YES, native explicit label | plausible | untested | untested | YES | **blocked at G3** — native gold action space and shipped model-side interface are not the same object |
| ToolDial | YES nominally | n/a | n/a | n/a | YES | **NO — G4 fatal.** 11,111 dialogues span 15 distinct system-action sequences; a content-free lookup reaches 0.8632 where the paper reports LLMs below 70%. `DATASET_CONFOUND` |
| FAIL-TaLMs | construction-derived | n/a | n/a | n/a | YES | NO — label construction plus severe `cannot_answer` / tool-count confound |
| BFCL (as released) | partial | n/a | n/a | n/a | **NO** — upstream of When2Call, same lineage | NO |
| ToolFailBench | post-execution recovery, not a pre-execution decision | n/a | n/a | n/a | YES | NO — wrong construct |
| original / synthetic V1R3 | YES | YES | YES | YES | NO — internal construction | NO |

## The claim that still requires independent benchmark evidence

Exactly one, and it is unchanged by the entire model panel:

> **That destination-resolved correction licensing transfers to a pre-execution multiclass
> action population that is not When2Call.**

Every formal ADMIT and DECLINE in the project — Qwen3-8B, Qwen3-4B, Gemma-2-9B — sits on one
benchmark. Adding host models increases *model* breadth and cannot substitute for it.

## Why the panel could not have fixed this

The two-model panel varied `M` with `D` held fixed **by design**, precisely so host-model
variation could be isolated. That design choice makes it structurally incapable of producing
cross-dataset evidence. A Qwen3.5 ADMIT would have left the limitation exactly where a
`NO_ACTIONABLE_CHANNEL` leaves it.

## What would be required, stated so it is not mistaken for a plan

WildToolBench is the only candidate that reached G1–G7 with a native explicit first-action
label. It is blocked at G3 on interface non-equivalence, and resolving that would mean building
a new four-mode measurement instrument — a different study with its own prospective
declaration, not a continuation of this one. **No such work is authorised or started here.**
