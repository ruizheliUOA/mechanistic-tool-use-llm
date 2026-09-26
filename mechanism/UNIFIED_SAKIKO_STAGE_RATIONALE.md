# Why each stage exists

# Discover → Detect → Correct → Verify → License

### DISCOVER
**Does:** finds recurring directional error transitions and selects channels.
**Exists because:** tool-decision errors are not homogeneous — three transitions
carry ~78% of errors, and channels differ in geometry (same-layer cosines
0.488–0.706) and in which estimator works.
**Strongest evidence:** error transition matrix; MetaTool's *different* dominant
channel on the same data (channel structure is model-dependent).
**Limitation:** channel structure is task- and label-space-dependent — 3 channels
on W2C, 1 on MetaTool-Binary, 0 quantifiable on ToolSandbox.

### DETECT
**Does:** trains a per-channel Router and fires only above τ.
**Exists because:** it is the only control on **exposure**, and exposure is the
dominant driver of damage (`Broke/R` flat 13.2%→10.3% while Broke falls 36→19).
**Strongest evidence:** ungated intervention inverts the sign — Phi +62→−32,
MetaTool +3/+11→−39/−34.
**Limitation:** detection does not locate intervention. AUC is flat at ~0.96 across
layers whose directions differ from usable to noise.

### CORRECT
**Does:** applies a channel-specific direction at a distinct injection site with
scale-normalised dose, on fired samples only.
**Exists because:** targeted intervention beats every corrupted variant — random
+14, reverse +14, wrong-layer +3, mismatched −17, against real +55; 0/59 matched
randoms exceed real.
**Strongest evidence:** the full control battery plus bootstrap stability 0.92–0.995.
**Limitation:** quality is design-conditional, not model-intrinsic; the damaged set
is fixed regardless of direction.

### VERIFY
**Does:** resolves every outcome into SOURCE_RETAINED / GOLD_ARRIVAL / OTHER_WRONG
and measures exposure-conditional preservation.
**Exists because:** exit is not arrival (46 of 153 exits land on a third wrong
action) and equal Net can hide different composition (+49 at F/B 2.96 vs 3.58).
**Strongest evidence:** Phi destination triple; the Phi turning point — Net +55
with 55.9% of exposed-correct rows broken.
**Limitation:** undefined in binary action spaces, where `OTHER_WRONG` is empty by
construction and `target_hit ≡ 1`.

### LICENSE
**Does:** grades evidence into rungs under prospectively frozen criteria and
declines what the evidence does not carry.
**Exists because:** correctability is protocol-conditional — exit-matched
cross-model destination differences flip sign, so a claim must name its protocol.
**Strongest evidence:** frozen dose grid, scale-matched injection, smallest-
admissible-dose rule ⇒ published rungs are **conservative lower bounds**.
**Limitation:** the gate adjudicates preservation on E1, which is permissive
relative to E2; and zero-exposure channels yield vacuous, not successful,
preservation.

**Each stage exists because the previous one is insufficient.** Discover without
Detect over-exposes; Detect without Correct only labels; Correct without Verify
mistakes movement for repair; Verify without License mistakes one result for a
general capability.
