# Stage 2 formal run — VOID report

**This run is VOID. Its scientific endpoints must not be reported or reused.**

| field | value |
|---|---|
| UTC | _(fill)_ |
| void reason code | _(one of the frozen enum values)_ |
| void reason | _(precise description)_ |
| evaluation access had begun | _(true/false)_ |
| arms completed before void | _(list)_ |
| endpoints computed | **false** |
| automatic retry attempted | **false** |
| runner SHA256 | _(fill)_ |
| execution freeze SHA256 | _(fill)_ |

## What VOID means here

A VOID run consumed the one-shot look at the sealed evaluation population
without producing an admissible confirmatory result. Endpoints are not computed
and must not be recovered from partial records.

## What VOID does not mean

Natural scientific outcomes are **not** void. In particular, insufficient
channel-error support returns
`FORMAL_SUPPORT_INSUFFICIENT_NO_CONFIRMATORY_CLAIM` and the run remains valid.
A negative result — real not beating the random null — is a valid result, not a
void one.

## Required next step

A crash **before** evaluation access is repairable only through a new versioned
freeze with separate approval. A crash **after** evaluation access began must not
be automatically retried; any further evaluation contact requires explicit new
authorization and must be disclosed as a second look.
