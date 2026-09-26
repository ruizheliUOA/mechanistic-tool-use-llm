# Part 0 — coverage audit of the previous synthesis plan

**Question:** if we kept only PCA + margin + dose, what would disappear?

**Answer: most of the design science.** The previous plan was organised around the
*mechanism claims that failed*. That inverted the evidence: SAKIKO's mechanistic
contribution is largely a set of **design necessities that held**, and those were
being narrated as caveats rather than results.

| component | previous status | why it matters |
|---|---|---|
| error-channel decomposition | **UNDER-COVERED** | 3 channels = 78% of errors (622/513/465 of 2052); the empirical basis of the whole method |
| Router / readability | **UNDER-COVERED** | probe AUC 0.75–0.96; the detectability–geometry dissociation lives here |
| **gating necessity** | **OMITTED** | replicated in two settings, sign-inverting — the single strongest design finding |
| channel matching | partly covered | MISMATCHED_CHANNEL −17 vs REAL +55 |
| **estimator choice** | **OMITTED** | rule-driven PCA-1 vs DiffMean; locked test +50 vs +36 |
| observation vs injection layer | **OMITTED** | flat AUC ~0.96 across layers while Net moves +13→+46 |
| **direction stability** | **OMITTED** | bootstrap cos 0.92–0.995; the one unstable case (0.82) is the known-degenerate layer |
| random/reverse/wrong-layer controls | covered | retained |
| preservation | covered | now with E1/E2 discipline |
| finite-dose effects | covered (E4) | retained |
| destination redistribution | covered | the Verify motivation |
| score-space comparator | partly covered | SUPPORTING only; all paired tests ns after Bonferroni |
| **historical positive results** | **UNDER-COVERED** | multi-PC emergence; Jaccard=1.000 Broke set |
| failed hypotheses | over-covered | correctly retained, but was crowding out the above |

**Should remain outside the mechanism section:** licensing arithmetic (exchange
rate), benchmark commissioning, reviewer-objection bookkeeping, recovery/provenance
audits.

**Verdict: the previous plan was materially incomplete.** It would have shipped a
mechanism section that was a list of things SAKIKO cannot claim, while the
artifact-backed design findings — gating, estimator adaptivity, layer degeneracy,
direction stability — sat unreported.
