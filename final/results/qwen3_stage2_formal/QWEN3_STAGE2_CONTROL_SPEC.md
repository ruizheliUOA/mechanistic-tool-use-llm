# Formal control specification

| arm | budget | population | role |
|---|---|---|---|
| baseline | none | all 548 evaluation rows | defines both populations |
| zero | none | routed | exact-equality gate before any endpoint |
| real d_grad | q·s_c at L21 | routed | primary |
| K=59 fresh randoms | q·s_c at L21 | routed | **primary null** |
| frozen L26 DiffMean | q·s_c at L21 | routed | estimator comparator |
| reverse −d_grad | q·s_c at L21 | routed | sign specificity |
| wrong-layer d_grad | **same** q·s_c at L26 | routed | site specificity |
| ungated d_grad | q·s_c at L21 | all pred==source | Router necessity |
| score-space comparator | frozen b_c, no model arm | routed | reducibility test |

Every directional arm receives the **identical** absolute perturbation norm
`q·s_c = 41.5659848890` and the identical Router gate as the real arm.

Wrong-layer preserves the absolute budget and is **not** renormalized by L26's
own activation norm; renormalizing would change layer and budget together.

The score comparator uses the frozen `b_c = 1.673107087612152` exactly and loads no
additional model arm.

Excluded by construction: same-layer L21 DiffMean, another channel's d_grad, orthogonalized mismatch, pooled or shared direction, PCA or rank-2 direction, any new estimator, any additional dose.
