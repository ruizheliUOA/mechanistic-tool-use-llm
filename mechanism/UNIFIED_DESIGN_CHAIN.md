# The unified design chain

Seven links. Each cites its strongest artifact-backed evidence.

### 1. Structured errors exist — `STRONG`
Three directional transitions account for **622/513/465 of 2052 errors (~78%)**,
threshold- and split-stable. Tool-decision failure is not one homogeneous event.

### 2. Their internal states are readable — `STRONG`
Probe AUC 0.77–0.95 (W2C) and Router AUC 0.927–0.991 (MetaTool), with silhouette
≈ 0 — linearly separable directions in overlapping clouds.

### 3. Some of that structure is intervention-relevant — `STRONG`
REAL **+55** vs RANDOM +14, REVERSE +14, WRONG_LAYER +3, MISMATCHED **−17**;
**0/59** matched randoms exceed real in the sealed protocol; bootstrap direction
stability 0.92–0.995.

### 4. But intervention quality depends on design — `STRONG`
The densest link, and the one previously omitted:
- **gating**: sign inverts without it (+62→−32; +3/+11→−39/−34)
- **channel matching**: mismatched is worse than nothing (−17)
- **estimator**: PCA-1 vs DiffMean by `cos(DM,PC1)` → locked test +50 vs +36
- **site**: readout ≠ estimation ≠ injection; AUC flat while Net +13→+46
- **dose/exposure**: `target_hit` 0.53→0.91, non-monotone; `Broke/R` flat

### 5. Behavioural movement is not destination correctness — `STRONG`
153 exits → 107 gold, **46 other-wrong**. Equal-Net settings (+49) differ in
composition (F/B 2.96 vs 3.58).

### 6. Positive correction can coexist with preservation failure — `STRONG`
Phi Net **+55** alongside **52/93 = 55.9%** of Router-exposed baseline-correct rows
broken. Robust across every artifact-backed realization (thinnest margin 6.1% vs a
5% budget at τ=0.9). Damage is a fixed subpopulation (Jaccard 1.000), so it cannot
be engineered away by changing direction — only by not firing.

### 7. Correctability is protocol-conditional and evidence-graded — `SUPPORTED`
Exit-matched cross-model destination differences flip sign; dose selection is
frozen, scale-matched and takes the smallest admissible dose, so published rungs
are **conservative lower bounds**. Hence Verify + License.

**Links 1–6 are `STRONG`. Link 7 is `SUPPORTED`. No link is `OPEN`.**
