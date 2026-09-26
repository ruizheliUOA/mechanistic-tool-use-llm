# Final causal synthesis

## 1. What SAKIKO is, most defensibly

> **A staged adjudication framework for pre-execution tool-decision correction.** Rather than
> reading every unsuccessful intervention as model uncorrectability, it localises where
> evidential legitimacy breaks: data support, internal readability, causal specificity,
> destination correctness, preservation, or confirmatory precision.

The audit supports this framing. **19 adjudicated units resolve into 9 distinct failure classes
at 6 distinct rungs**, and five of six rungs have a case where that rung and only that rung
decided the outcome.

## 2–4. What the verdicts mean

**ADMIT** — one `(M, D, c, I)` unit produced destination-correct, preservation-bounded,
direction-specific correction evidence that survived a prospective one-shot conjunction. It is
an existence claim about a setting, not a property of a model.

**DECLINE** — the tested proposal failed ≥1 frozen condition. It is **not** evidence that the
intervention did nothing: both formal DECLINEs had positive point effects, 0/59 randoms ≥ real,
p = 0.0167, and dead reverse arms.

**NO-GO** — the correction question was never validly posed. No intervention was run, so no
inference about correctability is available in either direction.

## 5–7. Sorting the failures

**Genuine evidence against the tested intervention (Class I–II):** Mistral `rfi_tc` and
`ca_direct` (reverse ≥ real; 17/20 and 7/20 randoms ≥ real) and Llama's locked TEST (5/20, 3/20;
z = 0.879, 1.256). Within the tested representation/estimator/dose family, no reliable
direction-specific write effect was identified. Phi is a separate genuine failure: real
destination behaviour (target-hit 0.69–0.74) with **real** collateral (52/264 = 19.7%,
52/93 = 55.9%).

**Primarily certification failures (Class III):** Qwen3-4B and Gemma. Point effects positive and
direction-specific; refused on conditions 3 and 6 alone.

**Never adjudicable (Class IV):** Qwen3.5 `ca→*` (reference 29 vs 30), Qwen3.5 and Gemma
`rfi→tc` (readability), Gemma and Qwen3-4B `ca→direct` (zero exposed correct rows), Qwen2.5
(LFS pointers), MetaTool (binary), ToolDial (confound), WildToolBench (interface).

## 8–10. The licence itself

Conservative, applied consistently, **not calibrated**. N=3 formal outcomes cannot estimate
`P(reliable correction | verdict)`, and no external criterion exists. Correctability,
preservation and certification should be **distinct axes** — the audit shows they fail
independently, and collapsing them into one word is what makes "DECLINE" ambiguous.

## 11. The expanded ladder

`Adjudicable → Readable → Steerable → Correctable → Preservable → Certifiable` is an
improvement, with one honest caveat: **Correctable is not yet empirically separated** from
Certifiable. The destination *point* conditions have never failed; only the *interval*
conditions have. Report five demonstrated rungs plus one definitional rung.

## 12–13. Why Qwen3-8B, and is it family or scale?

**Not family, not scale.** Within the Qwen lineage the outcomes are non-monotonic: 8B ADMIT,
4B precision-DECLINE, 3.5-9B no actionable channel. The simulation gives the sharper answer:
at its own effect size Qwen3-8B's licence-pass probability was **0.930**, versus 0.19–0.25 for
the DECLINEs. **Qwen3-8B's distinguishing feature is the size of its effect (Target Gain
+0.2759, ~3.5× the others), not its family or parameter count.** Whether that effect size is
itself caused by family remains **unresolved**.

## 14. What Qwen3.5 tells us

That improved aggregate accuracy need not produce a better correction substrate. Qwen3.5 has
the highest accuracy (0.4533), least collapse (0.6633) and errors nearest the boundary
(−0.4947), and terminated earliest. Its accuracy gain is class-asymmetric — `request_for_info`
recall roughly doubles while `cannot_answer` recall falls to 0.087 — which starves the reference
side of every `ca→*` channel and moves support onto the one channel that is not readable.

## 15. What Gemma tells us — and the correction to my earlier reading

Gemma is the central licence-audit case, and the simulation **overturns the "just under-powered"
story I reported earlier**. With exposure scaled realistically, P(ADMIT) rises to ≈0.46 near
n=400 then falls, as the binding condition migrates from precision to **collateral**: Gemma
broke 1 of 11 exposed correct rows, an exposure-conditional 9.1%, well above the 0.05 bound but
invisible behind the diluted 1/223 denominator.

So Gemma is best described as: *a direction-specific correction-like point effect that the
frozen population could not certify at the required destination precision, and whose
exposure-conditional collateral suggests a larger population would have bound on preservation
instead.* Not "really correctable."

## 16. What is still unknown

Exactly one major external-validity gap: **does destination-resolved correction licensing
transfer beyond When2Call?** The model panel varied `M` with `D` fixed by design and is
structurally incapable of answering it. Every alternative population failed eligibility on
independent grounds.

## 17. The one experiment that would change the paper most

**A second eligible multiclass pre-execution benchmark.** It is the only outstanding claim, and
it is the one a reviewer will press hardest. It is also **not currently runnable**: WildToolBench
is the sole candidate reaching G1–G7 and is blocked at G3 on interface non-equivalence, which
would require building a new four-mode measurement instrument — a different study with its own
prospective declaration.

Second-best and genuinely feasible: the **estimator × site control** (observation-site DiffMean,
write-site DiffMean, write-site gradient, all injected at the write site). The historical formal
comparison changed estimator and estimation site together, so those effects are confounded in
every existing result. It would not create a new ADMIT, but it would tell us whether any DECLINE
reflects intervention-design mismatch rather than model behaviour.

## Recommendation

```
NO FURTHER EXPERIMENT — FREEZE THE PAPER
```

Neither candidate is justified now. The benchmark experiment cannot be run without building a
new instrument. The estimator×site control is worth doing but changes no verdict and would not
alter the paper's claims — it belongs in future work.

The paper's contribution is not the single ADMIT. It is that **a preregistered staged licence,
applied to nine settings across four model families, refused eight of them at six distinguishable
rungs — including the authors' own prior headline result, a channel that missed support by one
row, and two interventions with positive direction-specific effects.** That is a working
adjudication framework, and it is defensible exactly as the evidence stands.
