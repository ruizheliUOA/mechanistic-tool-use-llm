# Final paper evidence database

75 canonical numbers. Machine-readable: `FINAL_PAPER_EVIDENCE.csv`.
Coverage: Phi-3.5 28 · Qwen3-8B 14 · Qwen2.5-7B 15 · Gemma-2-9b 10 · Qwen3-4B 8.

Every manuscript number must be checked against this file. Any number not present
here is an **orphan** and may not be written.

## Part 3 — destination regression gate: **PASS**

Recomputed from row-level records with no reference to stored summaries. The
channel is `cannot_answer → tool_call`, selected mechanically as `gold ==
'cannot_answer'` among routed rows (`pred_base == 'tool_call'`).

| model | routed ch. errors | SOURCE_RETAINED | GOLD_ARRIVAL | OTHER_WRONG | exits | target-hit |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-8B** | **72** ✓ | 20 | **38** ✓ | **14** ✓ | **52** ✓ | 0.7308 |
| **Qwen3-4B** | **118** ✓ | 54 | **37** ✓ | **27** ✓ | **64** ✓ | 0.5781 |
| **Gemma-2-9b** | **96** ✓ | 69 | **17** ✓ | **10** ✓ | **27** ✓ | 0.6296 |

All twelve frozen target values reproduce exactly.

## Part 4 — Phi turning point: **PASS**

```
548 rows · fired 293 · routed errors 200 · exposed-correct 93
SOURCE_RETAINED 47 · GOLD_ARRIVAL 107 · OTHER_WRONG 46 · exits 153
target-hit 107/153 = 0.6993 · BROKEN 52 · retained 41 · E2 52/93 = 0.5591
firing rule verified: |route != 'none'| = |route_prob >= 0.4| = 293
```

## Part 6 — router-confidence analysis: **PASS**

BROKEN n=52 mean 0.8305 median 0.8926 · retained n=41 mean 0.8432 median 0.9242 ·
Cliff's δ −0.0938 · Cohen's d −0.0716 · Mann-Whitney p 0.44. Break rate 55.9% /
56.8% / 51.0% at τ = 0.4 / 0.6 / 0.9 — exposure falls, rate does not.

## Part 7 — collateral denominator audit (**mandatory**)

| model | numerator | denominator | definition | rate | adjudication |
|---|---:|---:|---|---:|---|
| Phi-3.5 | 52 | 93 | **E2** exposed-correct | **55.9%** | fails budget, well-powered |
| Phi-3.5 | 52 | 264 | **E1** all baseline-correct | 19.7% | fails budget |
| Qwen3-8B | 0 | **6** | **E2** | — | **VACUOUS / UNADJUDICABLE** |
| Qwen3-8B | 0 | 211 | **E1** | 0% (CP upper **0.0141**) | passes; this is what the ADMIT used |
| Qwen3-4B | 1 | 50 | **E2** | 2.0% | passes |
| **Gemma-2-9b** | 1 | **11** | **E2** | **9.09%** | **exceeds 5% budget; weakly powered (n=11)** |
| Qwen3-4B dev q=0.5 | 6 | 455 / 106 | **E1 vs E2** | 1.3% vs **5.7%** | E1 passes, **E2 would fail** |
| Qwen3-4B `ca→direct` | 0 | **0** | zero exposure | — | **VACUOUS / UNADJUDICABLE** |

Never substitute one denominator for the other. Two rows are vacuous and must
never be reported as 0% collateral.

## Part 9 — structural specificity regression: **PASS**, with one narrowing

Recomputed across all 59 matched-random arms per model:

| model | REAL gold | random gold max | randoms ≥ real | reverse | wrong-layer | zero |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 38 | 8 | **0/59** ✓ | 0 | 13 | 0 |
| Gemma-2-9b | 17 | 0 | **0/59** ✓ | 0 | 1 | 0 |

**Narrowing — score-space.** The Qwen3-8B score-space comparator reaches **37 gold
arrivals against the activation arm's 38**. These are not meaningfully different.
S-6 is confirmed with force: no "activation beats score-space" claim is admissible,
and the comparator should be reported explicitly as near-equivalent on this
endpoint.

## Part 12 — mechanism regression: **PASS**, with one narrowing

**Gating necessity requires downgrading from `STRONGLY_SUPPORTED` to `SUPPORTED`.**
Recomputing gated vs ungated in the modern protocol:

| setting | gated Net | ungated Net | gated collateral | ungated collateral |
|---|---:|---:|---:|---:|
| Phi-3.5 / W2C | **+62** | **−32** | — | — |
| MetaTool nt_tc | +3 | **−39** | — | — |
| MetaTool tc_nt | +11 | **−34** | — | — |
| Qwen3-8B | **+38** | **+13** | 0/6 | 29/176 = 16.5% |
| **Gemma-2-9b** | **+16** | **+17** | 1/11 = 9.1% | 4/188 = 2.1% |

**Gemma is a genuine counterexample**: ungating did not reduce Net (+16 → +17) and
its collateral *rate* was lower. Gating necessity holds in four of five settings and
**fails in one**. It is model- and protocol-dependent, not universal.

(The Gemma rate comparison is itself unstable — a single break on n=11 is 9.1% —
which is why the counterexample is stated on Net, where it is unambiguous.)

All other MAIN_TEXT numbers traced to primary artifacts: channel share 78%,
common-layer cosines 0.4880 / 0.5687 / 0.7055, bootstrap stability 0.9199 healthy
min vs 0.8235 degenerate, estimator +50 vs +36, Broke-set Jaccard 1.000, alpha-sweep
exposure constant at 272. **No orphan mechanism number survives.**

## Part 13 — claim-by-claim

| claim | verdict | note |
|---|---|---|
| C2-1 correctable subset exists | **PASS** | regenerated on 3 models |
| C3-1 REAL beats every structural control | **PASS** | five-condition ordering intact |
| C3-2 0/59 matched randoms exceed real | **PASS** | verified independently for 2 models |
| C4-1 aggregate Net conceals preservation failure | **NARROW** | drop "≥1.6x"; thinnest E1 margin 6.1% vs 5% |
| C5-1 seven models across five rungs | **NARROW** | model-at-protocol-dose; conservative lower bounds |
| C5-2 evidence requirement scales with effect size | **PASS** | analytic |
| C6-1 MetaTool not global suppression | **NARROW** | 5/5 across two splits (524/1040 shared) |
| gating necessity | **NARROW** | `SUPPORTED`, not `STRONGLY_SUPPORTED` — Gemma counterexample |
| activation vs score-space | **PASS as SUPPORTING** | 37 vs 38; near-equivalent |
| Gemma/Qwen3-4B status | **CORRECTED** | S-5 falsified by primary artifacts; both are `FORMAL_DECLINE` — see S-20 |

No claim requires **REMOVE**.

## Part 0 — standing constraints checklist

S-1 … S-15 all present in `constraints/STANDING_CONSTRAINTS.md` and all respected
by the current evidence package. **S-16 and S-17 were specified in the E4 audit but
never written into the constraints file** — that gap is closed in this commit.
No manuscript, table or figure currently violates any S-number, because no
manuscript has been drafted yet.

## Part 14 — figures

**No figure is currently cleared for the paper.** The historical figure set predates
S-11, S-12, S-13, S-16, S-17, E4, the mechanism synthesis and the route-confidence
audit. Every figure must be regenerated from this database before use. Figure
regeneration is out of scope for this audit and is flagged as the first task of the
writing phase.
