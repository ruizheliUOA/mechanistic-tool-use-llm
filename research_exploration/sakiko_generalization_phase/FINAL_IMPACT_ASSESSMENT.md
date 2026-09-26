# Final impact assessment

## A. Does ACEBench reduce the single-benchmark objection?

```
YES — FRAMEWORK-LEVEL (not licence-level)
```

Four axes differ simultaneously: action ontology (`flag_param_error` has no W2C analogue),
measurement mechanism (free generation + parser vs candidate scoring), language (400 en / 400 zh)
and model (Qwen2.5-7B). Under criteria locked before execution, **9 of 10 passed** — accuracy
0.7562, macro-F1 0.6868, 4/4 modes ≥5%, 4/4 gold classes recall ≥0.10, 100% replay determinism,
0.98% parser mislabel — with max predicted-class share 0.675, better balanced than any W2C model
in the panel. Channel discovery ran automatically over all 10 observed transitions and the
support gate refused every one.

**Not licence-level:** ACEBench is a prior SAKIKO development corpus, no intervention was run,
and no destination or collateral result exists. The reply to Reviewer D is precise — *the licence
outcomes come from one benchmark; the apparatus has been instantiated on two* — and it must not
be stretched further.

## B. Does exposure-conditional preservation fix a real estimand problem?

**Yes, and the problem is larger than previously stated.** For Gemma, 212 of 223 rows in the
frozen denominator were never exposed. The frozen 0.45% and the causal 9.1% differ by 20×, and
they disagree about the verdict at scale.

## C. Can preservation currently be certified?

**No — in no setting, including the ADMIT.** One-sided 95% upper bounds on exposed populations:
Qwen3-8B **0.393** (0/6), Gemma **0.364** (1/11), Qwen3-4B 0.091 (1/50), Phi 0.647 (52/93); two
settings structurally vacuous at 0 exposure.

## D. Obstacle

**Both.** Too little exposure for Qwen3-8B and Gemma (6 and 11 rows against the 59 the rule of
three requires at α=0.05); high observed risk for Phi (55.9%) and Gemma (9.1%). Gemma is the
instructive case: even 1000 exposed rows at its observed rate reaches only 0.107, so it is a
**risk** problem the diluted denominator concealed, not a power problem.

## E. Does risk-controlled preservation strengthen SAKIKO or import standard statistics?

**Mostly imports standard statistics** — and that is still an improvement, because it replaces a
convention with a bound. But the honest accounting is that CRC is established machinery, and
arXiv 2607.24343 already applies distribution-free control to over-intervention on benign fields
in the tool-call domain. SAKIKO does not get novelty credit for the statistics.

## F. What remains novel

Three things, after conceding mechanism, method, and the risk-control machinery:
**destination-resolved accounting among K≥3 wrong actions**; **automatic directed-channel
discovery**; **adjudicating causal interventions rather than predictions**. Plus one empirical
contribution: a staged refusal record across nine settings and four families that localises
failure at six distinguishable rungs.

## G. Is the formulation still the strongest?

Yes. `Adjudicable → [Readable → Steerable → Correctable] → Licensable` resolves the DECLINE
ambiguity the adversarial audit identified, and this phase supplied its sharpest instance: the
ADMIT is `Correctable`-evidenced and **not** `Preservation`-certified. The old framing had no
way to say that.

## H. Reassessment

| axis | before | after |
|---|---|---|
| generalization | one benchmark | **framework instantiated on two**, licence still one |
| usefulness | moderate | **moderate-plus** — preservation estimand is a genuine repair |
| novelty | defensible but narrow | **unchanged and narrower than hoped** — CRC collision stands |
| ICLR readiness | borderline | **borderline, better defended** |

Not a rescue. The single-benchmark objection is answered at framework level, the weakest
construct is repaired, and one uncomfortable fact is now explicit: **the project's only ADMIT
carries a preservation claim its own data cannot certify.** Publishing that is more valuable than
hiding it, and it is the kind of thing Reviewer C rewards.

## Two self-corrections made in this phase

1. My drafted claim *"restraint decisions are wording-fragile; calling is stable"* is **withdrawn**
   — the archived report records that the paraphrase carried a weaker format directive. Only the
   gate failure (0.56 / 0.79 / 0.65 vs 0.80) is licensed.
2. My earlier framing of the ADMIT's collateral as a pass is now qualified: it passes E1 and is
   **uncertifiable** on E2.
