# Phase 2 — Claim Boundary

Every claim below is tagged with its evidence level. Do not quote a claim without its tag.

## DEMONSTRATED (quantitative, controlled, committed)
1. **Fresh baseline validity (R0):** Qwen2.5-7B bf16 on MetaTool-Binary, native Yes/No readout —
   acc 0.7731, balanced predictions, deterministic re-score 50/50, 0 skips.
2. **Automatic channel discovery on a new (dataset × model) pair:** both binary transitions found;
   `tool_call→no_tool` stable, `no_tool→tool_call` weak-but-scope-eligible; **dominant channel flipped
   vs Phi** on the same data — channel structure is model-dependent.
3. **Target-native intervention effect, 5 seeds, locked tests:** B +8.6±4.3, C +13.0±2.1,
   **D +21.6±4.5**, 15/15 arm-runs positive; exact B+C=D additivity and zero eligibility overlap on
   every seed; Broke ≤ Fixed/2 in D on 5/5.
4. **R1 degeneracy gate necessity replicated:** L12/L16 DiffMean norms 0.66–1.26 (degenerate) vs
   L20/L24 13.6–33.5; the highest router AUC occurred at a degenerate layer (tc_nt L12, 0.988) —
   AUC-only selection would have failed.
5. **Rule-driven direction adaptivity:** R2 selected PCA-1 for the under-call channel on 5/5 seeds and
   DiffMean for the over-call channel on 5/5 seeds, decided on validation only.
6. **Gating necessity:** ungated −39/−34 vs gated +3/+11 (seed 42).

## DIRECTION-SPECIFIC (earned at seed-42 scope only)
- Reverse direction changes **zero** predictions; real Net ≥ 95th (over-call) and 100th (under-call)
  percentile of 20 matched-norm random draws (1/20 and 0/20 ≥ real). This is **stronger** than the
  archived Phi fp16 regime (random saturation, 11/20 ≥ real) — but it was measured **once, at the
  seed-42 locked configs**. Per-seed random distributions were not run. The 5-seed claim is therefore
  *robust behavioural + seed-42 direction-specific*, not "specificity proven on every seed".

## BEHAVIOURAL-ONLY
- None needed this phase — the specificity controls passed where run. (If later per-seed controls
  saturate, the multiseed Net claims remain valid at the behavioural level.)

## NOT DEMONSTRATED (do not claim)
- **Multi-class cross-dataset generalization** — ACEBench readout remains unsolved (Phase-1 NO-GO).
- **Wording invariance** — native Yes/No only; archived audit shows other candidate wordings collapse.
- **Cross-model direction transfer** — out of scope; the archived negative result stands.
- **Geometry-only channel discovery** — unchanged Phase-1 conclusion (transitions define, geometry
  validates).
- **Per-seed specificity** — see above.
- **Mechanism of the PCA-1 preference** on the under-call channel — observed, not explained.
- **Population-level robustness of the over-call channel** — its per-seed test support is small
  (13–22); per-channel test deltas carry wide uncertainty even though the sign is stable 5/5.

## Bookkeeping caveats (carry into any writeup)
- Arm B seed-42 damage profile is marginal (Broke 8 vs Fixed 11); other seeds are cleaner (3–6 broke).
- In arm D each channel's Broke reappears as the *other* channel's residual (e.g. seed 42: nt_tc 13→5
  where 5 = 2 own-residual + 3 inherited); Nets stay exactly additive.
- The regenerated seed-42 split does not byte-match the archived split files (524/1040 rows agree);
  the archived files were used for the pilot (authoritative), regenerated splits for 123/456/789/2024.
- bf16 single-GPU determinism was verified in-process (50/50); cross-hardware drift is expected at the
  W2C-documented ±small-counts level.
