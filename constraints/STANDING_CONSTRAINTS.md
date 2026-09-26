# SAKIKO STANDING CONSTRAINTS (cumulative, non-negotiable)

S-1 RETRACTED — Part A displacement test (post-treatment conditioning PROHIBITED
    project-wide). Literature MDE figures 0.6849/0.5440/0.5557/0.9167 WITHDRAWN.
    Literature audit CLOSED at six papers; field-prevalence claim dead by
    construction.
S-2 PRIOR ART — CAST (Lee et al., ICLR 2025) publishes detector-gated
    conditional intervention = SAKIKO's Router role. MUST cite. MUST NOT claim
    as novel. Permitted novelty sentence is composition-only.
S-3 POPULATION NAMING — Net-whole +40/+39 vs Net-channel +38/+35. Denominators
    87 total / 72 routed / 92 firing set / 86 errors-in-set. "32 of 72", never
    "32 of 87".
S-4 PRESERVATION TWO CLAIMS — SYSTEM 0/211 CP upper 0.0141 (well powered);
    CONDITIONAL 0/6 CP upper 0.393 (VACUOUS). 1.41% never alone. Router-only
    6/6: report LOWER bound 0.541.
S-5 [SUPERSEDED 2026-08-28 by S-20 — wording retained for provenance]
    OLD: "Qwen3-4B and Gemma are ADMITTED AT LEVEL-1, not DECLINE."
    FALSIFIED by primary artifacts FORMAL_PRINCIPAL_VERDICT.{txt,json}.
    The surviving true part: empirical content is the DISTRIBUTION, not a count.
S-6 FORBIDDEN CLAIMS — no "activation beats score-space"; no trend across three
    models; no margin-dependence explanation of rungs; no component-locus (T2
    never run, n=20); no second ADMIT; Router-only 1.000 never without its
    decomposition.
S-7 LABELLING — mechanism = PREREGISTERED FOLLOW-UP / EXPLORATORY. Phi-3.5
    POST-HOC block. Llama/Mistral flagged ESTIMATOR-SUSPECT + K=20.
S-8 ANTI-FABRICATION — every number traceable or declared MISSING.
S-9 OPEN BLOCKERS — anonymity; 52/93 provenance; gate calibration.

S-10 EVIDENCE BRANCH — `main` is NOT the source of truth for SAKIKO experimental
     evidence. Verified 2026-08-27: `main` (48 commits) contains ZERO row-level
     record files. Modern row-level licensing artifacts reside on `sakiko-paper`;
     historical provenance may require `exp/sakiko-followup-archive` (169 commits)
     and Git LFS. All `*.jsonl`, `*.npz`, `*.npy` are LFS-tracked — a checkout
     without `git lfs pull` yields ~132-byte pointers that silently read as empty.

S-11 E-2 SCOPE — E-2 is a FIVE-condition reduced historical calibration, never
     nine. Only REAL_LOCKED has surviving row-level destination records
     (`p0_final_test_details.jsonl`, 548 rows). Placebo destination outcomes are
     UNIDENTIFIABLE from surviving artifacts and must never be reconstructed
     from aggregate Fixed/Broke/Net. Ungated and remove_{rfi_tc,ca_tc,ca_direct}
     are unrecoverable; per-channel entries in the multiseed table are NOT
     substitutes.
S-12 PHI SEED — Phi seed 42 = Fixed 107 / Broke 52 / Net +55, artifact-backed by
     `p1_multiseed_table.csv` (five seeds: 42/123/456/789/2024; Broke range
     16-52). `p0_final_test_eval.json` IS seed 42. The prose claims "22-33 Broke
     across five seeds" and "23 Broke on seed 42" are unsupported and permanently
     excluded. Validation-sweep values (35, 33) are not test results.
S-13 PHI FIRING RULE — the firing decision is `route != 'none'` (equivalently
     `route_prob >= 0.4` at seed 42), giving 293 fired = 200 errors + 93 correct.
     The `route` field alone is the channel ASSIGNMENT and is populated for all
     548 rows; using it as the firing decision yields 264 and fails the gate.

S-14 CLAIM MATRIX FROZEN 2026-08-27 — `claim_matrix/FROZEN_CLAIM_MATRIX.{md,csv}`
     is the sole source of truth for drafting. 10 claims, 6 load-bearing.
     No claim enters the paper without a row. Verdict:
     CLAIM_MATRIX_FROZEN_READY_TO_DRAFT / NO_LOAD_BEARING_EXPERIMENT_REQUIRED.
     No further exploratory work without explicit authorization.
S-15 GIT IDENTITY — all commits use the corresponding author. Two commits
     (d3d4556, 9022b07) were authored as sakiko@local in error and were rewritten
     locally to 64216e6 / 95207f5. Correcting the remote requires a force-push.

S-16 COLLATERAL DENOMINATOR — `clean_collateral_rate` in every modern dose/licence
     artifact is the E1 estimand (all baseline-correct). It is NOT exposure-
     conditional. Never present it as E2. Where both exist, report both.
     Verified: Qwen3-4B dev q=0.5 gives E1 6/455 = 1.3% (passes) vs E2 6/106 = 5.7%
     (would fail). The frozen gate uses the permissive denominator.

S-17 VACUOUS PRESERVATION — a 0.0% collateral computed on zero Router-fired
     baseline-correct rows is VACUOUS / UNADJUDICABLE, never "preserved".
     Known cases: Qwen3-4B `cannot_answer__to__direct` (exposure 0 at every dose);
     Qwen3-8B E2 (n=6, design sensitivity 0.235).

S-18 GATING NECESSITY SCOPE — gating necessity is SUPPORTED, not STRONGLY_SUPPORTED.
     It holds for Phi-3.5/W2C (+62 vs -32), MetaTool both channels (+3/+11 vs
     -39/-34) and Qwen3-8B (+38 vs +13), and FAILS for Gemma-2-9b (+16 gated vs
     +17 ungated). Never state it as a universal property.

S-19 SCORE-SPACE NEAR-EQUIVALENCE — on Qwen3-8B the score-space comparator reaches
     37 gold arrivals vs the activation arm's 38. Report as near-equivalent on this
     endpoint. Reinforces S-6.

S-20 FORMAL VERDICT PRECEDENCE (supersedes S-5) — the frozen formal verdicts are
     authoritative and take precedence over any later rung label:
       Qwen3-8B   FORMAL_CONFIRMATORY_SUCCESS  (target_hit CI95 [0.60416, 0.84615])
       Qwen3-4B   QWEN3_4B_FORMAL_DECLINE
       Gemma-2-9b GEMMA_FORMAL_DECLINE  (8/10 pass; fails 3_target_gain_CI_lower_gt_0
                  and 6_target_hit_CI_lower_gt_0.50 — the destination-interval layer)
     Qwen3-4B and Gemma have POSITIVE destination point estimates (0.5781, 0.6296)
     but their intervals cross the frozen boundary. Never describe either as
     formally admitted. Never describe either as intrinsically uncorrectable —
     the archive already freezes `DECLINE != intrinsically uncorrectable`.

S-21 RUNG VOCABULARY — the authoritative, prospectively frozen ladder is
     Adjudicable -> Readable -> Steerable -> Correctable -> Preserving -> Licensable
     (research_exploration/sakiko_paper_freeze/FINAL_SAKIKO_DEFINITIONS.md), with
     outcomes ADMIT / DECLINE. "Level-1" and "Level-2" have NO prospectively frozen
     definition in any artifact; they are later shorthand. Use the frozen ladder.
     If "Level-1" is retained at all it is a POST_HOC_DESCRIPTIVE_RUNG and must be
     labelled as such, never as a licence.

S-22 CORRECTION / REPAIR VOCABULARY (2026-09-10; title "When Does Correction
     Become Repair?") — CORRECTION = a measurable movement toward the desired
     decision. REPAIR = the composite claim destination correctness + preservation
     + sufficient evidence; NOT a rung, NOT arrival. Only a frozen ADMIT licenses a
     repair claim; the only such setting is Qwen3-8B, always stated with both S-4
     preservation claims. Never "repair"/"repaired" for gold arrivals, target-hit or
     any historical setting; never "correctable" as a synonym for repaired; never
     "correction" for the evidential claim. S-20 verdicts and the S-21 ladder are
     unchanged. Record: writing/TERMINOLOGY_MIGRATION_RECORD.md.

S-23 NO GATING-NECESSITY CLAIMS (2026-09-11) — no text may say gating is
     necessary or essential, universally or in a particular setting. This
     supersedes the "essential in some" wording previously permitted under S-18;
     S-18's measured values stand. Describe gating only by its measured effects:
     removing it reverses the sign on Phi-3.5/W2C (+62 vs -32, within one run) and
     both MetaTool channels (+3/+11 vs -39/-34); lowers net on Qwen3-8B (+38 vs
     +13); gives no benefit on Gemma-2-9b (+16 vs +17); raises gold arrivals per
     broken decision in the three directly comparable settings (Figure 5c).

S-24 LICENSING CRITERIA, 0.6042, AND INTERVAL SOURCE (2026-09-11) — every sealed
     gate (Qwen3-8B, Qwen3-4B, Gemma-2-9b) used the same ten conditions, written in
     final/results/qwen3_stage2_formal/QWEN3_STAGE2_PREREGISTRATION.md before the
     first sealed evaluation and applied unchanged: destination = target-hit point
     > 0.50 and 95% bootstrap lower bound > 0.50, target-gain point > 0 and lower
     bound > 0; preservation = E1 collateral point <= 0.05 and 95% upper bound
     <= 0.05. 0.6042 is Qwen3-8B's own interval lower bound and was never a
     criterion; never present it as a threshold. The earlier "0.6042 disclosure"
     plan rested on the retired Level-2 rung and is withdrawn. Report intervals as
     recorded in the frozen verdict artifacts: Qwen3-8B [0.60416, 0.84615],
     Qwen3-4B [0.4559, 0.6970], Gemma-2-9b [0.4444, 0.8148]
     (final_evidence/EVIDENCE_CORRECTIONS.md).
