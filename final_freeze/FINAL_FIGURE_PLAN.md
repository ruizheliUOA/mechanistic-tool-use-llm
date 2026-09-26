# Final figure plan — planning only, nothing generated

**Every existing figure is STALE.** None may receive `PAPER_CLEARED` until
regenerated from `final_evidence/FINAL_PAPER_EVIDENCE.csv`. No figure may use
manually typed numbers where script-generated data exist.

## Figure 1 — Unified SAKIKO
`Discover → Detect → Correct → Verify → License`. Conceptual, no model names, no
numbers. Each stage annotated with the question it answers.
**Claim:** C1-1.

## Figure 2 — Can we correct it?
Act-1 correction across models with the structural-control battery beside it.
Historical and modern panels **visually separated**.
**Source:** `phi_e2_*`, `q8b_*`, `q4b_*`, `gemma_*`, `*_matched_random_exceed`.
**Population:** per-model routed channel errors. **Denominator:** stated per bar.
**Claim:** C2-1, C3-1, C3-2.

## Figure 3 — Why aggregate gain is not enough  ← **strongest results figure**

**Panel A — Phi:** Net **+55** shown beside exposed-correct collateral
**52/93 = 55.9%**. One bar goes up, the other goes up too.
*Source:* `phi_net`, `phi_collateral_e2`. *Denominator:* E2 n=93.

**Panel B — Qwen3-8B, activation vs score-space:** two arms with near-identical
aggregates and different destinations.

| | channel-level Net (S-3) | gold | OTHER_WRONG | target-hit |
|---|---:|---:|---:|---:|
| activation | +38 | 38 | **14** | **0.7308** |
| score-space | +35 | 37 | **21** | **0.6379** |

*Source:* `q8b_*`, `q8b_score_space_gold`. *Population:* 72 routed channel errors.
**Caption must state:** paired tests ns after Bonferroni; descriptive only; the
0/6 vs 2/6 collateral is **not** shown as preservation evidence (n=6).
**Claim:** C4-1, C4-2.

## Figure 4 — Evidence and formal verdicts  *(redesigned)*

**Not** a categorical rung plot. A **forest plot** — point estimate, 95% interval,
and the frozen decision boundary — with the formal verdict beside each row:

| model | target-hit | CI95 | boundary | verdict |
|---|---:|---|---:|---|
| Qwen3-8B | 0.7308 | [0.6042, 0.8462] | 0.50 | **ADMIT** |
| Qwen3-4B | 0.5781 | [0.4559, 0.6970] — **crosses 0.50** | 0.50 | **DECLINE** |
| Gemma-2-9b | 0.6296 | [0.4444, 0.8148] — **crosses 0.50** | 0.50 | **DECLINE** |

*Amended 2026-09-11: intervals are the values recorded in the frozen verdict
artifacts; the boundary is 0.50 for every setting (S-24).*

The reader must see *why* the declines happened: the point estimates are positive
and the intervals are wide. A rung-only plot hides exactly that.
Historical models (Phi, Qwen2.5-7B, Llama, Mistral) in a separate band.
**Caption must state:** DECLINE does not mean uncorrectable; it means the evidence
is insufficient under this protocol and sample support (64 and 27 exits).
**Claim:** C5-1. **Source:** `q8b_target_hit`, `q4b_target_hit`, `gemma_target_hit`.

## Figure 5 — Why the design works *(only if page budget permits)*
Readable ≠ Steerable: flat Router AUC ~0.96 across layers against DiffMean norm
2.12→14.26 and Net +13→+46, with bootstrap stability 0.8235 vs 0.92–0.995.
**Claim:** C6-1.

## Regeneration checklist — every figure must declare
canonical data source · population · denominator (E1/E2/none) · exact claim ·
provenance band (historical vs modern).
