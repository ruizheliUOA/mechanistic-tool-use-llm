# Related-work search protocol — frozen before any paper was added

## Frame chosen: **B, prespecified term search — but executed as CONVENIENCE SAMPLING**

Declared honestly up front: this session cannot execute an exhaustive venue sweep
of ICLR/ICML/NeurIPS/ACL/EMNLP/TMLR 2023–2026. Full-text retrieval is the binding
constraint, not search.

**This corpus is therefore CONVENIENCE-SELECTED and is labelled as such
throughout.** The only wording permitted from it is:

> "Among the closest works we verified..."

**Forbidden:** "the field generally", "most steering papers", "destination
evaluation is uncommon", or any proportion presented as a field rate.

## Parameters

```
SEARCH_FRAME    : convenience sample seeded from (i) the six papers named in the
                  Session A brief, (ii) works they cite as closest prior art
DATE_CUTOFF     : 2026-08-19
SEARCH_TERMS    : "activation steering", "representation engineering",
                  "conditional activation steering", "inference-time
                  intervention", "steering vector reliability",
                  "detector-gated intervention"
VENUES          : not restricted (convenience sample)
```

## Inclusion criteria (A2.2)

Directly studies at least one of: inference-time activation/representation
intervention; representation engineering; conditional or detector-gated
steering; steering reliability/evaluation; causal internal intervention;
multiclass intervention for action/tool decisions; abstention/refusal/
clarification intervention.

**Excluded:** papers that merely mention steering; prompting-only methods;
fine-tuning methods without inference-time intervention.

## Verification rules (A2.9, carried forward)

- Full text required for coding. **Abstract-only = NOT VERIFIED.**
- Unretrievable full text = NOT ACCESSED, with reason recorded.
- No N inferred from percentages, benchmark size, or related work.
- Every YES/PARTIAL in the novelty matrix cites a section, table, or figure.

## Estimand-compatibility rule (A2.5)

SAKIKO's design-sensitivity curve applies **only** to the Level-2 endpoint:
source exits -> gold arrivals vs non-gold arrivals, H0: p_target <= 0.50.

A paper receives an exchange-rate figure **only** if it reports a compatible
source-error population, source exits, gold/other-wrong destination structure,
and independent unit. Otherwise it is coded
**`NOT COMPARABLE UNDER SAKIKO LEVEL-2 ESTIMAND`**.

**This retracts the Session A table**, which applied the curve to four papers
that do not measure this estimand. That application was invalid and its numbers
are withdrawn.
