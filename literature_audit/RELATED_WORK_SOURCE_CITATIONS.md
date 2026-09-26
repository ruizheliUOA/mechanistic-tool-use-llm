# Source citations — every YES/PARTIAL

## CAST (Lee et al., ICLR 2025 Spotlight) — arxiv.org/html/2409.05907v3
- DETECTOR_OR_ROUTER_GATE = **YES**. §3.1: `h' <- h + f(sim(h, proj_c h)) * alpha * v`,
  with `f = 1 if sim(h, proj_c h) > theta else 0`; "when the condition is activated
  during text generation, the behavior vector is added to all subsequent forward passes."
- INTERNAL_CAUSAL_INTERVENTION = **YES**. Same mechanism, §3.1.
- PRESERVATION_ENDPOINT = **PARTIAL**. Table 2 reports "Harmful Refusal" and
  "Harmless Refusal"; the discrepancy serves as a proxy. No direct measurement of
  harm to already-compliant outputs.
- Condition vector computed by PCA on mean-centred hidden states of contrasting
  examples (§3.3).
- Threshold chosen by **post-hoc grid search** — hence PROSPECTIVE_FROZEN_CRITERIA = NO.

## RepE (Zou et al.) — arxiv.org/html/2310.01405v4
- INTERNAL_CAUSAL_INTERVENTION = **YES**. LAT reading vectors + contrast vectors +
  LoRRA control.
- GOLD_ARRIVAL_EXPLICIT = **PARTIAL**. Table 2 reports TruthfulQA accuracy 47.9%
  vs 31% baseline (7B) and §4.3.3 shows outputs flipping honest<->dishonest; but
  no proportion reaching a target destination.
- PRESERVATION_ENDPOINT = **PARTIAL**. Table 3 shows reward-score changes under
  power control; no systematic accuracy-preservation evaluation.
- Intervention is **unconditional** (§4.3.3) — hence DETECTOR_OR_ROUTER_GATE = NO.

## ITI (Li et al., NeurIPS 2023) — arxiv.org/html/2306.03341v6
- RANDOM_DIRECTION_CONTROL = **YES**. §4.3 Table 3: random directions score 31.2
  true*informative vs 42.3 for mass-mean shift.
- N: §4.1 "817 questions spanning 38 subcategories"; 2-fold CV (§4.3).

## Tan et al. 2024 — arxiv.org/html/2407.12404v8
- POWER_OR_DESIGN_SENSITIVITY = **PARTIAL**. The paper's subject is steering
  reliability and generalisation across 40 datasets; it does not compute an
  evidence requirement for a certification endpoint.
- N: §4.1, 40-10-50 train-val-test split of 1000 samples per dataset.

## CAA (Rimsky et al.) — arxiv.org/html/2312.06681v3
- N: §4.1 and Appendix F Table 9, 50 held-out MC questions per behaviour,
  7 behaviours. Format two options A/B (§3).

## AxBench (Wu et al. 2025) — arxiv.org/html/2501.17148v2
- §3.3: 10 instructions sampled per concept; 500 concepts (Concept500);
  open-ended generation scored by an LLM judge on a 0/1/2 scale.
