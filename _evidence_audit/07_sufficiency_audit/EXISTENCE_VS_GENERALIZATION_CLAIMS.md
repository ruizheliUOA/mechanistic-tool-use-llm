# What one ADMIT proves

The Qwen3-8B result supports the existential proposition

> ∃(M, D, c, I) : historical_licence(M, D, c, I) = ADMIT

and nothing stronger.

| proposition | verdict |
|---|---|
| **existence** — at least one tested configuration satisfies the complete frozen licence | **SUFFICIENT** |
| **prevalence** — correctable channels are common | **INSUFFICIENT** |
| **generalization** — transfers across families, scales, benchmarks | **INSUFFICIENT** |
| **safety** — the intervention is risk-controlled | **INSUFFICIENT** (0/6 exposed, upper 0.393) |

## Does the paper's thesis require more than existence?

**No.** The thesis is that behavioural movement is insufficient evidence of correction and that a
staged licence localises where evidence fails. Its load-bearing evidence is the **distribution of
stopping stages across nine settings**, not the single pass. The ADMIT's role is narrow but
necessary: it demonstrates the licence is *satisfiable* and therefore not a rule that refuses
everything by construction.

```
ONE_ADMIT_FOR_CURRENT_THESIS: SUFFICIENT
```

## Does the paper lean too heavily on it?

No — each setting answers a different question, and none is a redundant row:

| setting | distinct question answered |
|---|---|
| Qwen3-8B | Can the complete licence ever admit? |
| Qwen3-4B, Gemma | Can direction-specific, aggregate-positive effects still fail certification? |
| Qwen3.5 | Can the question terminate before intervention — and is that stop allocation-sensitive? |
| Phi-3.5 | Can aggregate historical gains conceal unacceptable preservation? |
| Mistral | Can readability exist without reliable intervention specificity? |
| ACEBench | Does the apparatus instantiate outside the originating ontology? |

Qwen3-4B and Gemma are the only pair answering the same question, and their agreement is
informative rather than redundant: two independent model families failing at the identical rung.

```
EVIDENCE_PORTFOLIO_DIVERSITY: STRONG
```
