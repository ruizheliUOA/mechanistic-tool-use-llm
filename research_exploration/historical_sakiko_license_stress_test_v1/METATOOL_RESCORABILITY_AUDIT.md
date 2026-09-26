# MetaTool rescorability audit

## Verdict: structurally non-equivalent — `NOT_APPLICABLE`

## The blocking fact

MetaTool-Binary is a **binary** decision task: 1,040 rows, native Yes/No readout, with the two
discovered channels `tool_call→no_tool` and `no_tool→tool_call`.

For any ordered channel `g → s` in a binary space:

```
|A \ {g, s}| = 2 − 2 = 0
```

`OTHER_WRONG` is **empty by construction**. Every source exit is necessarily a gold arrival,
so `target_hit ≡ 1` whenever any exit occurs, and Target Gain collapses to the exit rate.

## Consequence

The modern licence's central discriminating quantity — whether a steered decision reaches
gold or lands on a *third* wrong action — is undefined. MetaTool cannot distinguish
"steerable" from "correctable", which is the exact distinction the modern protocol exists to
draw.

Forcing MetaTool into a four-mode modern licence would be a category error and is not done.

## Residual value

MetaTool retains value as **breadth evidence** for the earlier rungs: channels are
discoverable and readable in a second action space and a second corpus. It is
`SUPPORTING_HISTORICAL` and must not be presented as a correctability result.

Per-sample availability was not pursued further, because the structural objection is decisive
regardless of what records exist.
