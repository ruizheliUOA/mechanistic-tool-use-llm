# SAKIKO-Core v3

```
Adjudicable → [ Readable → Steerable → Correctable ] → Licensable
```

Scientific properties in brackets; `Adjudicable` and `Licensable` are epistemic status. Hence:

```
NOT LICENSED  ≠  INTRINSICALLY UNCORRECTABLE
NO OBSERVED COLLATERAL  ≠  CERTIFIED SAFE
```

The second line is not rhetoric: the ADMIT's `0/211` corresponds to a 95% one-sided upper bound
of **0.393** on the exposed population.

## Three-layer decomposition

**Benchmark-invariant constructs** — directed error channel `c=(g→s)`; adjudicability;
readability; specificity; destination; yield; exposure-conditional preservation; evidence.
All are counts, ranks or probabilities over a native action set. None references When2Call.

**Benchmark-specific adapters** — native action ontology; readout mechanism; parser; gold
derivation; **exposure-set definition**. Demonstrated twice: W2C (candidate scoring, 4 modes,
English) and ACEBench (free generation + parser, different 4 modes, bilingual).

**Application-specific decisions** — support floors; risk tolerance α; confidence δ; practical
yield requirement. These are *declared*, not universal. α = 0.05 is retained for historical
comparability and is a convention, not a derived optimum.

## The six constructs

| construct | estimand |
|---|---|
| 1 Adjudicability | support counts; `\|A_D\|≥3`; deterministic gold; defined exposure set |
| 2 Specificity | add-one empirical p vs matched-random null; reverse non-reproduction |
| 3 Destination | `P(→g \| exit)`; `P(→g) − P(→A_D\{g,s})` |
| 4 **Preservation** | **`P(correct→wrong \| exposed)`** with a finite-sample upper bound |
| 5 Yield | `P(gold arrival \| eligible channel error)` |
| 6 Evidence | interval or risk bound at δ, α |

Ten frozen conditions map onto these six with no loss of pathology coverage. Construct 5 repairs
the low-yield gap; construct 4 is materially changed — the denominator moves from all
baseline-correct to intervention-exposed, and a bound replaces a point comparison.

## Licence as an outer function

```
Licence = L( E_specificity, E_destination, E_preservation, E_yield ; δ, α )
```

`Correctable` (a property of the setting) stays separate from `licensed as correctable` (a
property of the evidence). This is the whole point of v3.

## Honest status

The v3 core is a **proposal**, validated retrospectively against 19 adjudicated units and one
pathology battery. It has never been run prospectively. Nothing here modifies a historical
verdict, threshold or preregistration.
