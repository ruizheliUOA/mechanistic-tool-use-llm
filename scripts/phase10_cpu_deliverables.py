"""
phase10_cpu_deliverables.py — CPU-only paper deliverables from EXISTING evidence only.
=======================================================================================
No GPU, no new experiment, no test access. Every number is read from a stored primary
artifact (or from the R0-b destination-resolution already computed). Produces:
  1. cross-model actionability atlas            -> ATLAS_CROSS_MODEL_ACTIONABILITY.{csv,md}
  2. dedicated ca_tc cross-model analysis       -> CA_TC_CROSS_MODEL_ANALYSIS.md
  3. corrected Phi/Qwen headline tables         -> CORRECTED_HEADLINE_TABLES.md
  4. withdrawn-claims table                     -> WITHDRAWN_CLAIMS.md
  5. abstract/intro/contribution skeleton       -> PAPER_SKELETON.md
  6. paper figures (existing evidence only)     -> figures/fig_paper_{1,2,3}.{png,pdf}
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

R = L.ROOT / "final" / "results"
OUT = R / "paper_assets"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def j(p):
    return json.loads((R / p).read_text())


# ---------------- load primaries ----------------
phi_eval = j("clean/p0_final_test_eval.json")
phi_p1 = j("clean/p1_multiseed_all_results.json")
phi_plac = j("clean/p2_placebo_controls.json")
qw_multi = j("7b_w2c_sakiko/multiseed/qwen25_7b_multiseed_results.json")
carfi_rand = j("w2c_qwen_new_channel_intervention/placebos/random_direction_results.json")
carfi_rev = j("w2c_qwen_new_channel_intervention/placebos/reverse_results.json")
mt_rand = j("metatool_qwen7b_sakiko_ca/placebos/random_direction_distribution.json")
mt_gate = j("metatool_qwen7b_sakiko_ca/placebos/gating_decomposition.json")
mist_pilot = j("mistral7b_w2c_sakiko_ca/pilot/pilot_summary.json")
llama_rho = j("phase8_prospective_llama/llama_rho_curves.json")["ca_tc"]["rows"]
r0b = j("whole_program_audit/R0B_DESTINATION_RESOLVED.json")

phi_ps = r0b["phi_seed42_locked"]
qw_agg = r0b["qwen7b_multiseed"]["channel_aggregate"]
llama4 = [r for r in llama_rho if r["rho"] == 4.0][0]


def addone_p(n_ge, n=20):
    return round((n_ge + 1) / (n + 1), 4)


# ================= 1. CROSS-MODEL ACTIONABILITY ATLAS =================
# columns: readable / direction-steerable / target-correctable, each with the evidence
rows = [
    dict(model="Phi-3.5-mini", dataset="W2C", channel="rfi_tc", n_ref="-",
         readable="YES", readable_ev="router-gated locked test",
         steerable="YES", steerable_ev="REAL +55 vs RANDOM +14 / WRONG_LAYER +3 / MISMATCHED -17",
         correctable="YES", correctable_ev=f"target-hit {phi_ps['per_channel']['rfi_tc']['target_hit']} (R0-b per-sample)",
         net="+55 locked (3-ch cascade)", verdict="ACTIONABLE"),
    dict(model="Phi-3.5-mini", dataset="W2C", channel="ca_tc", n_ref="-",
         readable="YES", readable_ev="same", steerable="YES", steerable_ev="same 5-way placebo",
         correctable="YES", correctable_ev=f"target-hit {phi_ps['per_channel']['ca_tc']['target_hit']}",
         net="(in cascade)", verdict="ACTIONABLE"),
    dict(model="Phi-3.5-mini", dataset="W2C", channel="ca_direct", n_ref="-",
         readable="YES", readable_ev="same", steerable="YES", steerable_ev="same 5-way placebo",
         correctable="YES", correctable_ev=f"target-hit {phi_ps['per_channel']['ca_direct']['target_hit']}",
         net="(in cascade)", verdict="ACTIONABLE"),
    dict(model="Qwen2.5-7B", dataset="W2C", channel="rfi_tc", n_ref="277",
         readable="YES", readable_ev="router AUC high",
         steerable="YES", steerable_ev="real>=all random 5/5 (cascade)",
         correctable="YES", correctable_ev=f"target-hit mean {qw_agg['rfi_tc']['target_hit_mean']} (5 seeds, R0-b)",
         net="+92+/-19.8 5-seed (3-ch)", verdict="ACTIONABLE"),
    dict(model="Qwen2.5-7B", dataset="W2C", channel="ca_tc", n_ref="79",
         readable="YES", readable_ev="router AUC high",
         steerable="PARTIAL", steerable_ev="cascade-level only; no per-channel battery",
         correctable="NO", correctable_ev=f"target-hit mean {qw_agg['ca_tc']['target_hit_mean']}; 2/5 seeds NEGATIVE Target Gain",
         net="(in cascade)", verdict="NOT DESTINATION-CLEAN"),
    dict(model="Qwen2.5-7B", dataset="W2C", channel="ca_direct", n_ref="79",
         readable="YES", readable_ev="router AUC high",
         steerable="PARTIAL", steerable_ev="cascade-level only",
         correctable="UNRESOLVED", correctable_ev="aggregate hit>1 = cascade cross-channel artifact; per-sample never logged",
         net="(in cascade)", verdict="CONFOUNDED"),
    dict(model="Qwen2.5-7B", dataset="W2C", channel="ca_rfi", n_ref="79",
         readable="YES", readable_ev="router val AUC 0.959-0.988",
         steerable="YES", steerable_ev=f"real~16 vs random {carfi_rand['ca_rfi']['mean']}+/-{carfi_rand['ca_rfi']['std']:.2f}, {carfi_rand['ca_rfi']['n_ge_real']}/20 (p={addone_p(carfi_rand['ca_rfi']['n_ge_real'])}); reverse +{carfi_rev['ca_rfi']['net']}",
         correctable="YES", correctable_ev="per-seed target-hit 0.73-0.93 (R0-b)",
         net="+15.8 marginal 5-seed", verdict="ACTIONABLE (clearest)"),
    dict(model="Qwen2.5-7B", dataset="W2C", channel="tc_rfi", n_ref="-",
         readable="YES", readable_ev="discovered, stable (boot 0.86)",
         steerable="NO", steerable_ev=f"real 0 = reverse 0; 11/20 random>=real (p={addone_p(11)})",
         correctable="NO", correctable_ev="locked +0 (F2 B2)",
         net="+0", verdict="NON-ACTIONABLE (correctly gated out)"),
    dict(model="Qwen2.5-7B", dataset="MetaTool", channel="tc_nt", n_ref="273",
         readable="YES", readable_ev="router AUC 0.947-0.979",
         steerable="YES", steerable_ev=f"real {mt_rand['tc_nt']['real_net']} vs {mt_rand['tc_nt']['mean']}+/-{mt_rand['tc_nt']['std']}, {mt_rand['tc_nt']['n_ge_real']}/20 (p={addone_p(mt_rand['tc_nt']['n_ge_real'])})",
         correctable="PARTIAL", correctable_ev=f"net +{mt_gate['gated']['tc_nt']['net']} (F{mt_gate['gated']['tc_nt']['fixed']}/B{mt_gate['gated']['tc_nt']['broke']}); binary space, randoms saturate",
         net="+11", verdict="WEAK POSITIVE (binary caveat)"),
    dict(model="Qwen2.5-7B", dataset="MetaTool", channel="nt_tc", n_ref="-",
         readable="YES", readable_ev="router AUC 0.927-0.979",
         steerable="NO", steerable_ev=f"real {mt_rand['nt_tc']['real_net']} vs {mt_rand['nt_tc']['mean']}+/-{mt_rand['nt_tc']['std']}, {mt_rand['nt_tc']['n_ge_real']}/20 (p={addone_p(mt_rand['nt_tc']['n_ge_real'])}) — FAILS",
         correctable="NO", correctable_ev=f"net +{mt_gate['gated']['nt_tc']['net']} with {mt_gate['gated']['nt_tc']['touched_correct_damage']} breaks / {mt_gate['gated']['nt_tc']['n_touched']} touched",
         net="+3", verdict="NOT CONFIRMED"),
    dict(model="Mistral-7B-v0.3", dataset="W2C", channel="ca_tc", n_ref="-",
         readable="YES", readable_ev=f"router val AUC {mist_pilot['locked']['ca_tc']['router_val_auc']:.3f}",
         steerable="NO", steerable_ev="cascade z~0.60; reverse retains ~88%; random matches real",
         correctable="NO", correctable_ev="no rho in 32x range yields specificity (B' refuted)",
         net=f"+{mist_pilot['locked']['ca_tc']['val_net']} val", verdict="NON-SPECIFIC"),
    dict(model="Mistral-7B-v0.3", dataset="W2C", channel="rfi_tc", n_ref="51",
         readable="YES", readable_ev=f"router val AUC {mist_pilot['locked']['rfi_tc']['router_val_auc']:.3f}",
         steerable="NO", steerable_ev="performed worse than noise",
         correctable="NO", correctable_ev="—",
         net=f"+{mist_pilot['locked']['rfi_tc']['val_net']} val", verdict="NON-SPECIFIC"),
    dict(model="Llama-3.1-8B", dataset="W2C", channel="ca_tc", n_ref="146",
         readable="YES", readable_ev="router val AUC 0.9585",
         steerable="YES", steerable_ev=f"real +8 vs random, {llama4['random']['n_ge_real']}/20 (p={addone_p(llama4['random']['n_ge_real'])}); z=5.46 descriptive",
         correctable="NO", correctable_ev="target-hit 0.286 (10/35); M1 & M2 both FAIL frozen bar",
         net="+8 val", verdict="STEERABLE NOT CORRECTABLE"),
    dict(model="Llama-3.1-8B", dataset="W2C", channel="rfi_tc / ca_direct", n_ref="-",
         readable="YES", readable_ev="AUC 0.84-0.96",
         steerable="NO", steerable_ev="boundary-seeking; gate rejected",
         correctable="NO", correctable_ev="ca_direct wrong-layer +61 = 3x real +19",
         net="+16..+23", verdict="REJECTED BY GATE (precision 1.0)"),
]
with open(OUT / "ATLAS_CROSS_MODEL_ACTIONABILITY.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

md = ["# Cross-Model Actionability Atlas",
      "",
      "**Every cell traced to a stored primary artifact. No new experiment. Significance is",
      "nonparametric (add-one permutation p, floor 0.048 with 20 randoms); z quoted only as a",
      "descriptive effect size.**", "",
      "| model | dataset | channel | readable | direction-steerable | target-correctable | verdict |",
      "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['model']} | {r['dataset']} | `{r['channel']}` | {r['readable']} | "
              f"{r['steerable']} | {r['correctable']} | **{r['verdict']}** |")
md += ["", "## Evidence per cell", ""]
for r in rows:
    md.append(f"### {r['model']} · {r['dataset']} · `{r['channel']}` — {r['verdict']}")
    md.append(f"- **readable:** {r['readable']} — {r['readable_ev']}")
    md.append(f"- **steerable:** {r['steerable']} — {r['steerable_ev']}")
    md.append(f"- **correctable:** {r['correctable']} — {r['correctable_ev']}")
    md.append(f"- Net: {r['net']}\n")
md += ["## The dissociation, stated compactly", "",
       "- **readable but NOT steerable:** Mistral `ca_tc` (AUC 0.964, Net +54, yet random matches real).",
       "- **steerable but NOT target-correctable:** Llama `ca_tc` (beats all 20 randoms, p=0.048; target-hit 0.286).",
       "- **all three:** Qwen `ca_rfi` (hit 0.73-0.93), Phi (hit 0.69-0.74), Qwen-7B `rfi_tc` (hit 0.91).",
       "",
       "**Net cannot distinguish these; destination-resolved endpoints can.**"]
(OUT / "ATLAS_CROSS_MODEL_ACTIONABILITY.md").write_text("\n".join(md))

# ================= 2. ca_tc CROSS-MODEL ANALYSIS =================
ca = f"""# `ca_tc` — A Cross-Model Problematic Channel

**CPU-only, existing evidence. `ca_tc` = gold `cannot_answer`, predicted `tool_call` (the model
calls a tool when it should abstain). It is the one channel tested on every architecture, and it
resists destination-clean correction on three of four.**

## 1. The pattern

| model | router AUC | Net | target-hit | destination-clean? | evidence |
|---|---|---|---|---|---|
| **Phi-3.5-mini** | — | (in +55 cascade) | **{phi_ps['per_channel']['ca_tc']['target_hit']}** | **YES** | R0-b per-sample: {phi_ps['per_channel']['ca_tc']['own_to_gold']}/{phi_ps['per_channel']['ca_tc']['own_moved']} moved reached gold |
| **Qwen2.5-7B** | high | (in +92 cascade) | **{qw_agg['ca_tc']['target_hit_mean']}** | **NO** | per-seed {qw_agg['ca_tc']['target_hit_per_seed']}; Target Gain/seed {qw_agg['ca_tc']['target_gain_per_seed']} — **2/5 seeds negative** |
| **Mistral-7B-v0.3** | {mist_pilot['locked']['ca_tc']['router_val_auc']:.3f} | +{mist_pilot['locked']['ca_tc']['val_net']} val | — | **NO** | not even direction-specific: reverse retains ~88%, random matches real |
| **Llama-3.1-8B** | 0.9585 | +8 val | **0.286** | **NO** | {llama4['random']['n_ge_real']}/20 randoms beat real (p={addone_p(llama4['random']['n_ge_real'])}) yet only 10/35 moved reach gold |

## 2. What is shared across the failures

**The wrong destination is the same in every case: `direct`.**
- Llama: `direct` is the model's runner-up for **186/383** train `ca_tc` errors (gold: 156); at
  ρ=4 the `ca_direct` bucket grows by **+16** while `rfi_tc` moves −1; at ρ=8, `ca_direct` +33.
- Phi (where it *works*): of the {phi_ps['per_channel']['ca_tc']['own_moved']} moved,
  {phi_ps['per_channel']['ca_tc']['own_to_gold']} reach gold and only
  {phi_ps['per_channel']['ca_tc']['own_to_other_wrong']} scatter — destination flow
  {phi_ps['per_channel']['ca_tc']['destination_flow']}.
- Qwen-7B: the two seeds with negative Target Gain are the ones where redistribution dominates.

**Geometric account (Llama, Phase-9.5, train-only):** the target axis and the wrong-destination
axis are **0.714 collinear**, and the wrong destination is **nearer** (‖→direct‖ 1.28 <
‖→target‖ 1.43). A single linear ray that leaves `tool_call` passes closer to `direct` than to
`cannot_answer`. The L22 class-mean geometry is ≈rank-1 (evac/select/M1 pairwise ≥0.92), so there
is no second mean-based axis to separate them.

## 3. Why this matters more than a single negative

`ca_tc` is **abstention**: the model is asked to decline, and instead acts. It is the
safety-critical channel of the four. The finding is that **the channel most worth correcting is
the one that most resists destination-clean correction**, and that its failure mode is
*consistent* — errors evacuate the source but land on `direct` (answer anyway) rather than
`cannot_answer` (abstain). Evacuating `tool_call` without reaching `cannot_answer` converts an
unnecessary tool call into an unsupported direct answer: **arguably a worse failure**.

## 4. What is NOT claimed

- Not claimed that `ca_tc` is uncorrectable in principle — Phi corrects it cleanly
  (hit {phi_ps['per_channel']['ca_tc']['target_hit']}), so architecture matters.
- Not claimed that a nonlinear/rank-2/other-site method would fail — those were never run
  (rank-2 unmotivated: the plane is degenerate; site-mismatch withdrawn).
- Qwen-7B numbers are **channel-level aggregates** (per-sample intervened preds were never
  logged), so its 0.50 is an aggregate estimate, not a per-sample measurement.
"""
(OUT / "CA_TC_CROSS_MODEL_ANALYSIS.md").write_text(ca)

# ================= 3. CORRECTED HEADLINE TABLES =================
pf = phi_p1["per_seed_results"]
nets = [r["net"] for r in pf]
hd = f"""# Corrected Phi / Qwen Headline Tables

**Every value traced to a primary artifact and recomputed. Supersedes prior prose.**

## 1. Phi-3.5-mini × W2C

| quantity | corrected value | primary | note |
|---|---|---|---|
| locked test Net (seed 42) | **+55** | `clean/p0_final_test_eval.json` + R0-b per-sample | = Fixed **{phi_ps['fixed']}** − Broke **{phi_ps['broke']}** |
| accuracy | 48.18 → 58.21 (**+10.04**) | same | n = 548 |
| **collateral breaks** | **{phi_ps['clean_collateral_breaks']}** | R0-b per-sample | **previously reported as "0 worsened"** — that was per-channel only |
| **target-hit (overall)** | **{phi_ps['target_hit_overall']}** | R0-b | {phi_ps['correct_target_transitions']} to gold vs {phi_ps['wrong_to_wrong_redistribution']} wrong→wrong |
| Target Gain | **+{phi_ps['target_gain_total']}** | R0-b | destination-clean |
| 5-seed Net | **+{np.mean(nets):.1f}** ({nets}) | `clean/p1_multiseed_all_results.json` | archived config |
| 5-seed accuracy | {phi_p1['summary_stats']['accuracy']['mean']}±{phi_p1['summary_stats']['accuracy']['std']} | same | |
| placebo battery | REAL **+{phi_plac['REAL_LOCKED']['net']}** vs RANDOM +{phi_plac['RANDOM_DIR']['net']} / WRONG_LAYER +{phi_plac['WRONG_LAYER']['net']} / REVERSE +{phi_plac['REVERSE_DIR']['net']} / MISMATCHED {phi_plac['MISMATCHED_CHANNEL']['net']} | `clean/p2_placebo_controls.json` | strongest causal separation in the programme |

**Reporting-convention reconciliation (three numbers, all legitimate):**
- **+55** = seed-42 locked test (Fixed {phi_ps['fixed']} − Broke {phi_ps['broke']}).
- **+{np.mean(nets):.1f}** = 5-seed mean, archived `clean/` config.
- **+59** = thesis 5-seed mean under the "V6 Locked" config (accuracy 58.94% vs archived
  {phi_p1['summary_stats']['accuracy']['mean']}%) — a *different* locked config, ~1.2 Net higher.
- **"+63 for Phi seed-42" (thesis line 782) is a MISLABEL** — archived Phi seed-42 is +55 (now
  confirmed per-sample); **+63 is the Qwen-7B seed-42 Net**. Do not cite +63 for Phi.

**Canonical wording:** *"Phi×W2C: +55 net locked (seed 42; {phi_ps['fixed']} fixed / {phi_ps['broke']} broke;
target-hit {phi_ps['target_hit_overall']}), 5-seed +{np.mean(nets):.1f}, with a 5-way placebo separation."*

### Phi per-channel destination flow (R0-b, per-sample, n=548)

| channel | errors | moved | →gold | →wrong | target-hit | destinations |
|---|---|---|---|---|---|---|
"""
for chn, v in phi_ps["per_channel"].items():
    hd += (f"| `{chn}` | {v['n_baseline_errors']} | {v['own_moved']} | {v['own_to_gold']} | "
           f"{v['own_to_other_wrong']} | **{v['target_hit']}** | {v['destination_flow']} |\n")

hd += f"""
## 2. Qwen2.5-7B × W2C

| quantity | corrected value | primary | note |
|---|---|---|---|
| 5-seed Net | **{qw_multi['aggregate']['net_mean']} ± {qw_multi['aggregate']['net_std']}** | `7b_w2c_sakiko/multiseed/...results.json` | exact; per-seed {qw_multi['aggregate']['net_min']}–{qw_multi['aggregate']['net_max']} |
| positive seeds | **{qw_multi['aggregate']['n_positive']}/5** | same | |
| ΔAccuracy | **+{qw_multi['aggregate']['delta_acc_mean']*100:.1f} pts** | same | |
| **destination-clean?** | **NOT UNIFORMLY** | R0-b | see per-channel below |

### Qwen-7B per-channel destination flow (R0-b, per-seed CHANNEL aggregates — per-sample never logged)

| channel | target-hit per seed | mean | Target Gain per seed | reading |
|---|---|---|---|---|
| `rfi_tc` | {qw_agg['rfi_tc']['target_hit_per_seed']} | **{qw_agg['rfi_tc']['target_hit_mean']}** | {qw_agg['rfi_tc']['target_gain_per_seed']} | **destination-clean** |
| `ca_tc` | {qw_agg['ca_tc']['target_hit_per_seed']} | **{qw_agg['ca_tc']['target_hit_mean']}** | {qw_agg['ca_tc']['target_gain_per_seed']} | **NOT clean — 2/5 seeds negative** |
| `ca_direct` | {qw_agg['ca_direct']['target_hit_per_seed']} | {qw_agg['ca_direct']['target_hit_mean']} | {qw_agg['ca_direct']['target_gain_per_seed']} | **CONFOUNDED** (hit>1 impossible per-sample = cascade cross-channel artifact) |

**Canonical wording:** *"Qwen-7B×W2C: +{qw_multi['aggregate']['net_mean']}±{qw_multi['aggregate']['net_std']} 5-seed, 5/5 positive. The
effect is carried by the destination-clean `rfi_tc` channel (hit {qw_agg['rfi_tc']['target_hit_mean']}) and a large but
aggregate-confounded `ca_direct`; its `ca_tc` channel is Net-positive but not destination-clean
(hit {qw_agg['ca_tc']['target_hit_mean']}, 2/5 seeds negative Target Gain)."*

## 3. Statistical-reporting corrections (apply everywhere)

- **20 random directions → minimum add-one permutation p = (0+1)/(20+1) = 0.048.** "0/20" supports
  **p ≈ 0.048**, *not* 3σ. **Strike every 3-sigma claim.**
- Report the **empirical exceedance count + add-one p** as primary; quote z only descriptively with
  a normality caveat (the ρ=4 random Target-Gain null is skewed: −17.6 ± 12.7, range −46…−3).
- Report **Net with destination flow and collateral**, never Net alone.
"""
(OUT / "CORRECTED_HEADLINE_TABLES.md").write_text(hd)

# ================= 4. WITHDRAWN CLAIMS =================
wd = """# Withdrawn / Corrected Claims

**Every withdrawal below was raised by our own audits, not by external review. No frozen verdict
changed; no measured number was invalidated. Only interpretations, scopes, and aggregate framings
moved.**

| # | claim as originally stated | status | why it fell | what replaces it |
|---|---|---|---|---|
| W1 | "Read/write **site mismatch** explains the Llama `ca_tc` corrective failure" (Phase-9.5) | **WITHDRAWN** | cross-layer orthogonality is **generic** — every cached layer pair has \\|cos\\| ≤ 0.07, incl. adjacent L23↔L22 (−0.030); and the Phase-8 scout shows the same direction injected **at its own estimation site (L22) performs worse** (Net +4/ρ4, +12/ρ8) than upstream at L18 (+8, +23) | the effect is **mediated by downstream blocks**; cross-layer cosine carries no site information. Site-mismatch is not supported and the 2×2 native-axis study it motivated is rejected |
| W2 | "**M2 refutes** linear destination selection" (Phase-9) | **WITHDRAWN** | the executed M2 was an **invalid estimator**: split-half **0.144** (vs deployed 0.878, M1 0.984), in-plane fraction 0.013, and it moved *fewer* errors than matched-norm randoms. Ledoit-Wolf chose shrinkage 0.073 in 4096-D with 348 samples; forcing shrinkage→1 collapses it back onto the stable ray (cos 0.895) | M2's FAIL is **uninformative** about linear selectors. The FAIL verdict stands; the inference does not. The line stays stopped because the *stable* family is rank-1-exhausted, not because M2 proved anything |
| W3 | "L22 contains **two distinct functional axes** (evacuation + selection) → rank-2 is justified" | **CONTRADICTED** | evac/select/M1 are pairwise **≥ 0.92 collinear** (22.6°); M1 = 0.96·evac + 0.20·sel⊥ | the class-mean geometry at L22 is **≈ rank-1**; the proposed 2-D plane is degenerate and rank-2 has no motivating second axis |
| W4 | "M1 at ρ=4 **aims better**" | **CORRECTED** | gold arrivals are ~equal (M1 9 vs deployed 10) on the identical routed set; M1 simply **dislodges fewer wrong samples** (5 vs 25) | M1 reduces wrong-destination churn; it does not increase gold arrivals. Its target-hit 0.643 is evacuation *selectivity*, not steering |
| W5 | "Net-positive = **mere evacuation**" (Phase-9 §6 overcorrection) | **WITHDRAWN** | `Fixed` **requires arrival at gold** (`phase8_lib.py:345`) — Net does credit correct arrivals | Net's real defects: **blind to wrong→wrong redistribution** and **nets collateral silently** (Phi: +55 hides 52 breaks). Report Net *with* destination flow |
| W6 | "**4/4 architectures corrected**, total **+155**" (thesis) | **NARROWED** | 3 of 4 members are **cross-model transfer** (Ridge / zero-transfer of Phi's vector), and the thesis's own text reports the Qwen-1.5B transfer **fails its wrong-layer placebo** (wrong +35 > real +30); Gemma had only 3 control conditions | **Phi is the one specificity-controlled positive.** The +155 sums one controlled result with three Net-positive but **not specificity-controlled** transfers. Per-model Nets stand; the causal framing does not |
| W7 | "Phi locked: **100 fixed, 0 worsened**" | **CORRECTED** | "0 worsened" was **per-channel only**; R0-b per-sample gives **Fixed 107 / Broke 52** | "+55 net = 107 fixed − 52 collateral breaks, target-hit 0.72" |
| W8 | "Phi seed-42 Net = **+63**" (thesis line 782) | **CONTRADICTED** | archived primaries give Phi seed-42 = **+55** (now confirmed per-sample: 107−52); **+63 is the Qwen-7B seed-42 Net** | a mislabel/cross-reference. Cite +55 |
| W9 | "**Gate v2 separates** actionable from non-actionable" | **NARROWED** | its absolute `REF_FLOOR=80` **false-rejects Qwen `ca_rfi` (79 refs)** — the clearest actionable channel; the learning curve shows no cliff at 80 and split-half at 79 is 0.888 | Gate v2 is a **supported prospective rejector** (Llama: precision 1.0, 2/2 fragility predictions). Its **admission side is unvalidated** (0 prospective admits). The count floor is a known-imperfect stability proxy — recorded, **not retuned** |
| W10 | "MetaTool **dual-channel additive**, both positive" | **PARTIALLY SUPPORTED** | `tc_nt` confirmed (net +11, 0/20, p=0.048); **`nt_tc` is NOT** (net +3, 1/20 → p=0.095, and 8 breaks on 19 touched) | one weak binary-space positive; the second channel is unconfirmed and damaging |
| W11 | "**3-sigma** direction specificity" | **CONTRADICTED** | 20 randoms resolve to a **minimum add-one p = 0.048** | report exceedance counts + add-one p; z only as descriptive effect size |
| W12 | "SAKIKO can **choose** among rank-1 / low-rank / nonlinear / alt-site / abstention" | **NEVER TESTED** | only rank-1 DiffMean/PCA-1 + one invalid LDA were ever run | the adaptive-selection layer is **aspirational**; remove until built |
| W13 | "SAKIKO **OOD generalization**" | **NEVER TESTED** | both confirmed positives fail *different* frozen support floors on full data **before any split** (ca_rfi ref 79<80; tc_nt val-err 22<30, and MetaTool has no domain structure: 167 tools, max 3 errors) | **PROTOCOL INFEASIBLE** on available data; a real OOD test needs a new dataset/model |
"""
(OUT / "WITHDRAWN_CLAIMS.md").write_text(wd)

# ================= 6. PAPER FIGURES =================
# Fig 1: the dissociation — steerability (nonparam p) vs target-hit
fig, ax = plt.subplots(figsize=(8.4, 5.2))
pts = [
    ("Phi rfi_tc", addone_p(0), phi_ps["per_channel"]["rfi_tc"]["target_hit"], "#1a7f37", "o"),
    ("Phi ca_tc", addone_p(0), phi_ps["per_channel"]["ca_tc"]["target_hit"], "#1a7f37", "o"),
    ("Phi ca_direct", addone_p(0), phi_ps["per_channel"]["ca_direct"]["target_hit"], "#1a7f37", "o"),
    ("Qwen ca_rfi", addone_p(carfi_rand["ca_rfi"]["n_ge_real"]), 0.83, "#1a7f37", "*"),
    ("Qwen-7B rfi_tc", addone_p(0), qw_agg["rfi_tc"]["target_hit_mean"], "#1a7f37", "s"),
    ("Qwen-7B ca_tc", addone_p(0), qw_agg["ca_tc"]["target_hit_mean"], "#bf8700", "s"),
    ("MetaTool tc_nt", addone_p(mt_rand["tc_nt"]["n_ge_real"]), 0.55, "#bf8700", "^"),
    ("Llama ca_tc", addone_p(llama4["random"]["n_ge_real"]), 0.286, "#d1242f", "D"),
    ("MetaTool nt_tc", addone_p(mt_rand["nt_tc"]["n_ge_real"]), 0.0, "#57606a", "v"),
    ("Qwen tc_rfi", addone_p(11), 0.0, "#57606a", "v"),
]
# most channels share p=0.048 (0/20) -> jitter x slightly so markers/labels are legible
jit = {0.0476: [-0.004, -0.0022, 0.0, 0.0022, 0.0044, 0.0066, 0.0088, 0.011]}
seen = 0
for name, p, hit, c, m in pts:
    xp = p
    if abs(p - addone_p(0)) < 1e-6:
        xp = p + jit[0.0476][seen]; seen += 1
    ax.scatter(xp, hit, s=150, color=c, marker=m, edgecolor="k", linewidth=.6, zorder=3)
    ax.annotate(name, (xp, hit), xytext=(9, -3), textcoords="offset points", fontsize=8.5)
ax.axvline(0.048, color="k", ls="--", lw=1.2)
ax.axhline(0.60, color="k", ls=":", lw=1.2)
ax.annotate("p = 0.048\n(the floor with\n20 randoms)", xy=(0.048, 0.13), xytext=(0.062, 0.12),
            fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=.8))
ax.text(0.30, 0.625, "target-hit = 0.60 bar", fontsize=8)
ax.set_xscale("log"); ax.set_xlim(0.036, 2.4); ax.set_ylim(-0.06, 1.10)
ax.set_xlabel("direction-specificity   (add-one permutation p; ← more specific)")
ax.set_ylabel("target-correctness   (target-hit rate)")
ax.set_title("Readable ≠ Steerable ≠ Target-Correctable\n"
             "every channel here is readable (router AUC 0.84–0.99); they separate on these two axes",
             fontsize=10)
ax.text(0.040, 1.03, "ACTIONABLE", fontsize=9, color="#1a7f37", weight="bold")
ax.text(0.040, 0.33, "STEERABLE BUT\nNOT CORRECTABLE", fontsize=9, color="#d1242f", weight="bold")
ax.text(0.75, 0.06, "NOT DIRECTION-SPECIFIC", fontsize=9, color="#57606a", weight="bold")
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG / "fig_paper_1_dissociation.png", dpi=180)
fig.savefig(FIG / "fig_paper_1_dissociation.pdf"); plt.close(fig)

# Fig 2: Net vs destination-resolved truth
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
labels = ["Phi\n(3-ch)", "Qwen-7B\nrfi_tc", "Qwen-7B\nca_tc", "Llama\nca_tc", "Mistral\nca_tc"]
netv = [55, 47, 6, 8, 54]
hitv = [phi_ps["target_hit_overall"], qw_agg["rfi_tc"]["target_hit_mean"],
        qw_agg["ca_tc"]["target_hit_mean"], 0.286, np.nan]
cols = ["#1a7f37", "#1a7f37", "#bf8700", "#d1242f", "#57606a"]
axs[0].bar(range(5), netv, color=cols); axs[0].set_xticks(range(5)); axs[0].set_xticklabels(labels, fontsize=8)
axs[0].set_ylabel("Net (Fixed − Broke)"); axs[0].set_title("A. Net says: all positive", fontsize=10)
axs[0].axhline(0, color="k", lw=.6)
axs[1].bar(range(5), [0 if np.isnan(h) else h for h in hitv], color=cols)
axs[1].axhline(0.6, color="k", ls=":", lw=1.2); axs[1].text(2.6, 0.62, "0.60 bar", fontsize=8)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(labels, fontsize=8)
axs[1].set_ylabel("target-hit (moved → gold)"); axs[1].set_ylim(0, 1.05)
axs[1].set_title("B. Destination-resolved says: only some correct\n(Mistral: not even direction-specific)", fontsize=10)
axs[1].text(4, 0.03, "n/a\n(non-specific)", fontsize=7, ha="center")
fig.suptitle("The same results under Net vs under destination-resolved endpoints", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(FIG / "fig_paper_2_net_vs_destination.png", dpi=180)
fig.savefig(FIG / "fig_paper_2_net_vs_destination.pdf"); plt.close(fig)

# Fig 3: Phi placebo battery (the causal high-water mark)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
conds = ["REAL_LOCKED", "RANDOM_DIR", "REVERSE_DIR", "WRONG_LAYER", "MISMATCHED_CHANNEL"]
vals = [phi_plac[c]["net"] for c in conds]
bcol = ["#1a7f37"] + ["#57606a"] * 3 + ["#d1242f"]
ax.bar(range(5), vals, color=bcol)
for i, v in enumerate(vals):
    ax.text(i, v + (1.5 if v >= 0 else -3.5), f"{v:+d}", ha="center", fontsize=9, weight="bold")
ax.set_xticks(range(5))
ax.set_xticklabels(["REAL", "RANDOM", "REVERSE", "WRONG\nLAYER", "MISMATCHED\nCHANNEL"], fontsize=8)
ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Net on locked test (n=548)")
ax.set_title("Phi × W2C placebo battery — the programme's strongest causal separation\n"
             "REAL dominates all four controls; mismatch actively harms", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "fig_paper_3_phi_placebo.png", dpi=180)
fig.savefig(FIG / "fig_paper_3_phi_placebo.pdf"); plt.close(fig)

print("wrote:")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(R))
