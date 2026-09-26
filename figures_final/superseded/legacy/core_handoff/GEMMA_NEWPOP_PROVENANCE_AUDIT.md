# Gemma new-population provenance audit (A0)

## INDEPENDENT_FOR_GEMMA_PROSPECTIVE_USE: **NO**

Per the brief, the formal Gemma programme **STOPS here**. A1–A8 were not run. No
model was loaded, no GPU used, no SEALED access attempted.

## Location and identity

| Property | Verified value | Source |
|---|---|---|
| Location | `research_exploration/original_population_pilot_v1r3/` | filesystem — **note: in `sakiko-followup`, not on the `sakiko-paper` branch** |
| Payload | `ai_audit_pass_1.jsonl`, 144 rows | `wc -l` |
| Rows | 144 | `PILOT_VALIDATION_REPORT_V1R3.json` → `rows` |
| Groups | 36 | same → `groups` |
| Per class | `tool_call` 36, `request_for_info` 36, `cannot_answer` 36, `direct_answer` 36 | same → `per_mode` |
| Programmatic gold derivable | 144 / 144 | same → `programmatic_gold_derivable` |
| Structure gate | `V1R3_STRUCTURE_GATE_PASSED` | same |
| Leakage gate | `MULTIVIEW_LEAKAGE_GATE_PASSED` | same |
| Gold gate | `PROGRAMMATIC_GOLD_GATE_PASSED` | same |

**The commissioning properties the brief listed are confirmed.** 4 modes, 144
rows, 36 per class, unique programmatic gold, counterfactual groups — all verified
from the validation report, not from a summary. The brief was right to ask;
the numbers hold.

## Why the answer is still NO

Four independent grounds. **Any one of them alone is disqualifying.**

### 1. The package's own frozen verdict names a different model and a narrower scope

```
"verdict": "READY_FOR_QWEN3_BASELINE_ONLY_SCREEN"
```
— `PILOT_VALIDATION_REPORT_V1R3.json`

The authorisation is for **Qwen3**, and for a **baseline-only screen**. Gemma is
not the named model. Intervention is not the named scope. `PACKAGE_STATUS_V1R3.md:7`
states it again in prose: *"The audit clean result licenses only the named
baseline-only screen."*

Running a Gemma formal intervention programme on this population would consume it
outside its own frozen licence — the precise class of act the SAKIKO protocol
exists to prevent.

### 2. The authorising gate was never satisfied

`PILOT_BASELINE_SCREEN_PLAN.md:6`: *"Execution is authorised only after human
adjudication passes the annotation gates."*

`HUMAN_ANNOTATION_GUIDE.md:4-6`: *"The human-annotation route this document
describes was replaced by deterministic AI audit passes. **Nothing in this package
is human annotated.**"*

The gate that authorises execution has not been passed. It was substituted, not
satisfied.

### 3. The audit is not independent

```
"auditor_relationship": "SAME_MODEL_FAMILY_AS_DRAFTER"
```

`PACKAGE_STATUS_V1R3.md:7`: *"not human annotated, not natural benchmark data,
and **not independently AI-audited**."*

The instrument was drafted and audited within one model family. Gold correctness
is therefore not established independently of the systems being evaluated.

### 4. It is explicitly out of scope in a frozen scope document

`research_exploration/sakiko_final_model_breadth_panel_v1/DATASET_SCOPE_AND_LIMITS.md:32-33`:

> *"ToolDial, WildToolBench, FAIL-TaLMs, **the original V1R3**, and any further
> public Dataset-C search are out of scope and may not be revived here."*

## Has Gemma ever consumed it?

**No.** `grep -rn "gemma\|Gemma"` over the package returns zero hits, and
`forbidden_access_confirmations` records `model_loaded: False`, `inference_run:
False`, `gpu_used: False`, `activation_or_router_run: False`. The population is
pristine with respect to Gemma. That is the one thing that *would* have made it
attractive — and it is not sufficient.

## Independence from When2Call

**Lineage-independent: YES.** V1R3 shares no rows, no source corpus, and no
construction pipeline with When2Call, and it carries `direct_answer` as a real
gold class, which W2C never does.

**Independent as evidence: NO.** `BENCHMARK_EVIDENCE_SIDE_ANALYSIS.md:19` scores
it *"NO — internal construction"*. The outstanding claim requires transfer to a
pre-execution multiclass population that is **not authored by this project**. An
ADMIT on an internally authored, same-family-audited instrument would not
discharge that claim; it would restate it.

## What would change the answer

In order of cost:

1. **Human adjudication of the 144 rows** against `ADJUDICATION_GUIDE.md`, by an
   adjudicator independent of the drafting model. This satisfies ground 2 and
   partially ground 3.
2. **A scope amendment** to `DATASET_SCOPE_AND_LIMITS.md` explicitly re-admitting
   V1R3 for prospective Gemma use, authored by the human team. This addresses
   ground 4.
3. **A re-issued package verdict** naming Gemma and naming intervention. This
   addresses ground 1.
4. Ground 3 cannot be fully discharged without an external instrument. Even fully
   adjudicated, V1R3 remains internally authored — it can support a
   *population-dependence* argument, but not an *external replication* claim.

Note also that the prior Qwen3-8B screen on this population terminated at
`BASELINE_SCREEN_NO_VALID_MODEL_SIGNAL` because the model never emitted
`request_for_info`. That is a Qwen finding and does not predict Gemma. But it
means the population has never yet produced a usable channel topology from any
model, so even after re-authorisation the A2 adjudicability gate is a live risk
rather than a formality.

## Status

**GEMMA_PROSPECTIVE_STATUS: NO_GO**

Reason: `POPULATION_NOT_AUTHORISED_FOR_PROSPECTIVE_GEMMA_USE`.
Not a scientific failure of Gemma. Not a statement about the population's quality.
A scope and authorisation finding, reached before any compute was spent.
