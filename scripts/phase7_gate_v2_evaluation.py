"""
phase7_gate_v2_evaluation.py — response-shape analysis + retrospective Gate-v2 dev check.
==========================================================================================
CPU-only post-processing of `rho_specificity_results.json` (validation rows only).

(1) Classifies each development channel's rho-response shape per the pre-registered rules in
    PHASE7_PROTOCOL.md (interior-specific / boundary-seeking / flat / monotonic non-specific /
    unstable / negative).
(2) Applies the Gate-v2 Stage-4/Stage-5 logic retrospectively to the development channels and
    compares the decision to each channel's established development label.

This is NOT prospective evidence; it is an internal-consistency check.
No test data is read. Emits rho_specificity_results.csv, rho_response_summary.csv,
gate_v2_development_decisions.{json,csv}.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

RES = P.OUT / "rho_specificity_results.json"
RHO_MAX = 8.0

# ── pre-registered Gate-v2 Stage 4/5 thresholds (see PHASE7_PROTOCOL.md §17,18,21) ──
Z_MIN = 2.0
FRAC_GE_MAX = 0.05          # <= 5% of randoms may reach real
RESIDUAL_FRAC = 0.75        # own_after <= 0.75 * own_before
BROKE_RATIO = 0.5           # broke <= fixed/2
DAMAGE_FRAC = 0.25          # damage_on_correct <= 25% of touched_correct

# established development labels (from the Phase-7 evidence audit; test-side history)
DEV_LABEL = {
    ("qwen25_7b", "ca_rfi"): "positive_specific",
    ("qwen25_7b", "tc_rfi"): "negative",
    ("mistral7b_v03", "ca_tc"): "positive_specificity_limited",
    ("mistral7b_v03", "rfi_tc"): "negative",
    ("mistral7b_v03", "ca_direct"): "negative",
}
# FINAL actionability truth (the archived locked-test/gate decision), used for agreement:
# ca_tc had positive Net but was ultimately judged NON-actionable (failed the pre-registered
# gate; z=1.91<2). Agreement is scored against actionability, not Net sign.
FINAL_TRUTH = {
    ("qwen25_7b", "ca_rfi"): "actionable",
    ("qwen25_7b", "tc_rfi"): "non-actionable",
    ("mistral7b_v03", "ca_tc"): "non-actionable",
    ("mistral7b_v03", "rfi_tc"): "non-actionable",
    ("mistral7b_v03", "ca_direct"): "non-actionable",
}
STAGE1_POWER_FLOOR = 30   # own-channel validation errors required for an intervention claim


def point_passes(row):
    """Gate-v2 Stage-4 + Stage-5 conjunction at a single rho (validation only)."""
    real, rev, rnd = row["real"], row["reverse"], row["random"]
    z = row["spec_z"]
    n = rnd["n"]
    crit = {
        "net_pos": real["net"] > 0,
        "residual_down": (real["own_after"] <= RESIDUAL_FRAC * real["own_before"]
                          if real["own_before"] > 0 else False),
        "broke_controlled": (real["broke"] <= BROKE_RATIO * real["fixed"]
                             if real["fixed"] > 0 else real["broke"] == 0),
        "damage_controlled": (real["damage_on_correct"] <= DAMAGE_FRAC * real["touched_correct"]
                              if real["touched_correct"] > 0 else True),
        "not_redirection": real["own_to_gold"] > real["own_to_other_wrong"],
        "real_gt_reverse": real["net"] > rev["net"],
        "z_ok": (z is not None and z >= Z_MIN),
        "nonparam_ok": (rnd["n_ge_real"] / n) <= FRAC_GE_MAX,
        "interior": row["rho"] < RHO_MAX,
    }
    # zero-variance policy (pre-registered)
    if rnd.get("zero_variance"):
        crit["z_ok"] = (rnd["n_ge_real"] == 0 and real["net"] > 0 and real["net"] > rev["net"])
    return crit, all(crit.values())


def classify(rows):
    """Pre-registered response-shape classification."""
    rows = sorted(rows, key=lambda r: r["rho"])
    nets = [r["real"]["net"] for r in rows]
    passing = [r for r in rows if point_passes(r)[1]]
    interior_pass = [r for r in passing if r["rho"] < RHO_MAX]
    # stability: an interior pass must not be contradicted by BOTH neighbours
    stable_interior = []
    for r in interior_pass:
        i = [x["rho"] for x in rows].index(r["rho"])
        nb = [rows[j] for j in (i - 1, i + 1) if 0 <= j < len(rows)]
        # neighbour "supports" if it also has net>0 and z>=1 (weak coherence)
        sup = sum(1 for x in nb if x["real"]["net"] > 0
                  and (x["spec_z"] is not None and x["spec_z"] >= 1.0))
        if not nb or sup >= 1:
            stable_interior.append(r)
    if stable_interior:
        return "interior-specific", stable_interior
    if max(nets) <= 0:
        return "negative", []
    if max(abs(n) for n in nets) <= 3:
        return "flat", []
    # boundary-seeking: max Net at the maximum rho and no interior specificity
    if nets[-1] == max(nets):
        return "boundary-seeking", []
    # rising but never specific
    if nets[-1] > nets[0]:
        return "monotonic non-specific", []
    return "unstable", []


def main():
    res = json.loads(RES.read_text())
    flat_rows, summary, decisions = [], [], []
    for mk, chans in res.items():
        for ch, blob in chans.items():
            rows = blob["rows"]
            P.experiment_rows(rows)   # firewall: reject any split=="test" row
            assert all(r["split"] == "val" for r in rows), "non-val row in results"
            shape, passing = classify(rows)
            label = DEV_LABEL.get((mk, ch), "unknown")
            best = max(rows, key=lambda r: (r["real"]["net"]))
            # Stage-1 power floor: own-channel validation errors (constant across rho)
            own_val_errors = rows[0]["real"]["own_before"]
            stage1_pass = own_val_errors >= STAGE1_POWER_FLOOR
            gate_stage45_admit = len(passing) > 0
            # FULL gate = Stage-1 power floor AND Stage-4/5 screen
            gate_admit = stage1_pass and gate_stage45_admit
            sel = min(passing, key=lambda r: r["rho"]) if passing else None
            for r in rows:
                crit, ok = point_passes(r)
                flat_rows.append({
                    "model": mk, "channel": ch, "role": r["role"], "split": r["split"],
                    "rho": r["rho"], "alpha_equiv": r["alpha_equiv"], "delta_norm": r["delta_norm"],
                    "med_obs": r["med_obs"], "med_inj": r["med_inj"], "obs": r["obs"], "inj": r["inj"],
                    "method": r["method"], "thr": r["thr"], "router_val_auc": round(r["router_val_auc"] or 0, 4),
                    "n_routed": r["n_routed"], "fire_rate": r["fire_rate"],
                    "real_net": r["real"]["net"], "real_fixed": r["real"]["fixed"],
                    "real_broke": r["real"]["broke"],
                    "own_before": r["real"]["own_before"], "own_after": r["real"]["own_after"],
                    "own_to_gold": r["real"]["own_to_gold"],
                    "own_to_other_wrong": r["real"]["own_to_other_wrong"],
                    "damage_on_correct": r["real"]["damage_on_correct"],
                    "touched_correct": r["real"]["touched_correct"],
                    "reverse_net": r["reverse"]["net"],
                    "rand_mean": r["random"]["mean"], "rand_std": r["random"]["std"],
                    "rand_max": r["random"]["max"], "n_ge_real": r["random"]["n_ge_real"],
                    "n_random": r["random"]["n"], "percentile_of_real": r["random"]["percentile_of_real"],
                    "spec_z": r["spec_z"], "zero_variance": r["random"].get("zero_variance", False),
                    "gate_point_pass": ok, **{f"crit_{k}": v for k, v in crit.items()},
                })
            summary.append({
                "model": mk, "channel": ch, "role": rows[0]["role"], "dev_label": label,
                "archived_rho": blob["archived_rho"], "router_val_auc": round(blob["router_val_auc"] or 0, 4),
                "dm_norm": blob["dm_norm"], "n_err_train": blob["n_err_train"],
                "n_ref_train": blob["n_ref_train"],
                "response_shape": shape,
                "n_passing_rho": len(passing),
                "passing_rho": "|".join(str(r["rho"]) for r in passing),
                "selected_rho": (sel["rho"] if sel else None),
                "selected_net": (sel["real"]["net"] if sel else None),
                "selected_z": (sel["spec_z"] if sel else None),
                "best_net_any_rho": best["real"]["net"], "best_net_rho": best["rho"],
                "best_net_z": best["spec_z"],
                "gate_v2_decision": "ADMIT" if gate_admit else "REJECT",
            })
            truth = FINAL_TRUTH.get((mk, ch), "unknown")
            gate_label = "actionable" if gate_admit else "non-actionable"
            agree = (gate_label == truth)
            decisions.append({
                "model": mk, "channel": ch, "dev_label": label,
                "final_actionability_truth": truth,
                "own_val_errors": own_val_errors,
                "stage1_power_floor_pass": stage1_pass,
                "stage45_screen_admit": gate_stage45_admit,
                "gate_v2_full_decision": "ADMIT" if gate_admit else "REJECT",
                "response_shape": shape,
                "selected_rho": (sel["rho"] if sel else None),
                "selected_z": (sel["spec_z"] if sel else None),
                "agreement_vs_truth": ("consistent" if agree else "discrepant"),
                "reason": _reason(shape, passing, rows, label),
            })
    _w(P.OUT / "rho_specificity_results.csv", flat_rows)
    _w(P.OUT / "rho_response_summary.csv", summary)
    _w(P.OUT / "gate_v2_development_decisions.csv", decisions)
    json.dump(decisions, open(P.OUT / "gate_v2_development_decisions.json", "w"), indent=2)
    print("=== RESPONSE SHAPES ===")
    for s in summary:
        print(f"  {s['model']:14s}/{s['channel']:10s} [{s['dev_label']:28s}] shape={s['response_shape']:22s} "
              f"gate={s['gate_v2_decision']:6s} sel_rho={s['selected_rho']} sel_z={s['selected_z']} "
              f"best_net={s['best_net_any_rho']}@rho{s['best_net_rho']}(z={s['best_net_z']})")
    print("\n=== GATE-V2 DEVELOPMENT DECISIONS (full gate = Stage-1 power floor AND Stage-4/5) ===")
    for d in decisions:
        s1 = "S1ok" if d["stage1_power_floor_pass"] else f"S1FAIL(own_val={d['own_val_errors']})"
        print(f"  {d['model']:14s}/{d['channel']:10s} truth={d['final_actionability_truth']:14s} "
              f"screen={'ADMIT' if d['stage45_screen_admit'] else 'reject'} {s1:20s} "
              f"-> FULL {d['gate_v2_full_decision']:6s} [{d['agreement_vs_truth']}]")
    n_ok = sum(1 for d in decisions if d["agreement_vs_truth"] == "consistent")
    print(f"\n  full-gate agreement vs final actionability truth: {n_ok}/{len(decisions)}")
    scr = sum(1 for d in decisions if (("actionable" if d["stage45_screen_admit"] else "non-actionable")
                                       == d["final_actionability_truth"]))
    print(f"  Stage-4/5 screen ALONE agreement: {scr}/{len(decisions)} "
          f"(tc_rfi over-admits without the Stage-1 power floor)")


def _reason(shape, passing, rows, label):
    if passing:
        r = min(passing, key=lambda x: x["rho"])
        return (f"interior rho={r['rho']} net={r['real']['net']:+d} z={r['spec_z']} "
                f"rev={r['reverse']['net']:+d} own {r['real']['own_before']}->{r['real']['own_after']}")
    best = max(rows, key=lambda r: r["real"]["net"])
    fails = [k for k, v in point_passes(best)[0].items() if not v]
    return f"no rho passes; best net {best['real']['net']:+d}@rho{best['rho']} fails: {','.join(fails)}"


def _w(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
