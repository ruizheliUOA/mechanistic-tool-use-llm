# Minimality — is a smaller licence sufficient?

## Leave-one-out on observed decisions

| removed | decisions changed |
|---|---|
| C4 random null | none observed — **but sole catcher of a real pathology, and it decided Mistral and Llama at earlier stages** |
| C7/C8 collateral | Phi would be admitted |
| C11 non-vacuity | Gemma `ca→direct` would be admitted on an untestable 0/471 |
| C3 **or** C6 (either alone) | **none** — they always failed jointly |
| C3 **and** C6 (both) | Qwen3-4B and Gemma would be admitted |
| C1, C2, C5, C9 | none observed |

## The redundancy question

C3 (Target Gain CI lower > 0) and C6 (target-hit CI lower > 0.50) share the `gold` numerator and
moved together in every observed case. Removing **either** changes no decision; removing **both**
changes two. They are near-redundant *on this data*.

They are not conceptually identical: C6 asks "of the decisions moved, are most moved correctly?"
and C3 asks "across all channel errors, is net repair positive?" A high-precision low-yield
intervention passes C6 and barely passes C3; a high-yield sloppy one does the reverse. The
simulation confirms both are triggered by "many exits, wrong destination." But on the evidence
the project actually has, **the licence is effectively 8 conditions plus one double-charged
destination-precision requirement.**

## Verdict

```
SMALLER_CORE_LICENCE_SUFFICIENT (on current evidence) —
but the full conjunction is not unjustified
```

The minimal set preserving every observed decision and every pathology catch is:

`support · specificity · destination point · one destination CI · collateral point ·
collateral CI · non-vacuity · zero exactness` — **eight conditions**.

Dropping the second destination CI is the only defensible simplification, and even that is
supported by *absence of divergence* in three formal trials rather than by a demonstration that
the two can never diverge. **No change to the frozen licence is proposed.** This is framework
analysis, and the sample is too small to justify amending a preregistered rule.
