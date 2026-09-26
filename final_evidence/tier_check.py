"""Provenance-tier enforcement for the SAKIKO evidence chain.

Rule: no LOAD-BEARING claim may rest on Tier C (secondary-audit-only) evidence.

Coverage cannot drift: the claim list is read from FINAL_CLAIM_MATRIX.csv, and
every claim must be either mapped to evidence variables or explicitly EXEMPT.
An unmapped, non-exempt claim is a failure.

Run from repo root:  python3 final_evidence/tier_check.py
"""
import csv, sys

EV_CSV = 'final_evidence/FINAL_PAPER_EVIDENCE.csv'
CM_CSV = 'final_freeze/FINAL_CLAIM_MATRIX.csv'

# claims that legitimately cite no measured quantity
EXEMPT = {
    'C1-1': 'framework claim — no measured quantity',
    'C5-2': 'analytic — exact binomial design-sensitivity curve, no dataset',
}

DEP = {
 'C2-1': ['phi_net','q8b_gold_arrival','q4b_gold_arrival','gemma_gold_arrival','metatool_dual_mean_net'],
 'C2-2': ['metatool_overcall_mean_net','metatool_undercall_mean_net','metatool_partition_overlap','metatool_split_agreement'],
 'C3-1': ['phi_e2_real_locked','phi_e2_random_dir','phi_e2_reverse_dir','phi_e2_wrong_layer','phi_e2_mismatched'],
 'C3-2': ['q8b_matched_random_exceed','gemma_matched_random_exceed'],
 'C4-1': ['phi_net','phi_collateral_e2','phi_broken_e2','phi_exposed_correct'],
 'C4-2': ['q8b_target_hit','q8b_score_space_target_hit','q8b_score_space_other_wrong'],
 'C4-3': ['phi_broken_route_prob_mean','phi_retained_route_prob_mean','phi_confidence_cliffs_delta','phi_break_rate_tau09'],
 'C5-1': ['q8b_formal_verdict','q4b_formal_verdict','gemma_formal_verdict','phi_collateral_e2'],
 'C5-3': ['llama_randoms_ge_real','mistral_randoms_ge_real','llama_target_hit','mistral_real_net'],
 'C6-1': ['mech_channel_share','mech_cos_rfi_ca_tc','mech_broke_set_jaccard',
          'mech_bootstrap_stability_degenerate','mech_estimator_pca1_vs_diffmean'],
}

ev = {r['variable']: r for r in csv.DictReader(open(EV_CSV))}
claims = list(csv.DictReader(open(CM_CSV)))

fails, orphans, uncovered = [], [], []
print(f"{'claim':<7} {'LB':>3}  {'tiers':<8} status")
print('-' * 60)
for c in claims:
    cid, lb = c['id'], c['load_bearing'] == 'YES'
    if cid in EXEMPT:
        print(f"{cid:<7} {'Y' if lb else 'n':>3}  {'-':<8} EXEMPT ({EXEMPT[cid]})")
        continue
    if cid not in DEP:
        uncovered.append(cid)
        print(f"{cid:<7} {'Y' if lb else 'n':>3}  {'?':<8} ** UNMAPPED — add to DEP or EXEMPT **")
        continue
    vs = DEP[cid]
    miss = [v for v in vs if v not in ev]
    orphans += [f'{cid}:{v}' for v in miss]
    tiers = sorted({ev[v]['provenance_tier'] for v in vs if v in ev})
    bad = lb and 'C' in tiers
    if bad: fails.append(cid)
    status = '** LOAD-BEARING ON TIER C **' if bad else ('Tier B floor' if lb and 'B' in tiers else 'ok')
    if miss: status += f'  ORPHAN:{",".join(miss)}'
    print(f"{cid:<7} {'Y' if lb else 'n':>3}  {','.join(tiers) or '-':<8} {status}")

print()
ok = not (fails or orphans or uncovered)
if fails:     print("FAIL — load-bearing on Tier C:", ", ".join(fails))
if orphans:   print("FAIL — orphan variables:", ", ".join(orphans))
if uncovered: print("FAIL — unmapped claims:", ", ".join(uncovered))
print("TIER CHECK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
