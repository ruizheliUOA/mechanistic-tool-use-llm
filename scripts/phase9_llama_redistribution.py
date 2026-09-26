"""
phase9_llama_redistribution.py — Part B: why is Llama ca_tc direction-specific but not corrective?
==================================================================================================
CPU-ONLY. Uses train/validation artifacts only (cached activations + baseline details + the
Phase-8 validation rho curves). NO GPU, NO test rows (firewalled), NO new inference, NO Gate
modification.

B1. Post-intervention transition table for Llama ca_tc (validation, at the interior rho=4
    point where the direction IS specific: real +8, random -0.3+-1.5, 0/20, z=5.46).
    NOTE: per-sample intervened predictions are NOT stored by the Phase-8 sweep (it stores
    aggregates), so the transition table is reconstructed from the aggregate fields that ARE
    stored (own_before/own_after, own_to_gold, own_to_other_wrong, fixed/broke, nt_after per
    channel). Anything not recoverable is reported as UNAVAILABLE rather than invented.

B2/B3. Geometry of the DiffMean direction: does it point AWAY from the source class or TOWARD
    the target class? Decomposed on train activations:
      d_away   = unit(mean(correct_source_pred) - mean(err))      [leave the tool_call region]
      d_toward = unit(mean(correct_target_gold) - mean(err))      [enter the cannot_answer region]
    and the actual DiffMean d = unit(mean(ref_gold) - mean(err)).
    We report cos(d, d_away), cos(d, d_toward), and the class-mean geometry, plus a
    linear-separability check of target vs other-wrong destinations.

Outputs: phase9_llama_ca_tc_redistribution.{json,csv}
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

OUT = L.OUT.parent / "phase9_mechanism_and_ood"
OUT.mkdir(parents=True, exist_ok=True)

CH = "ca_tc"
GOLD, FROM_PRED = "cannot_answer", "tool_call"
ET = f"{GOLD}__{FROM_PRED}"
OBS, INJ = 22, 18


def domain_of(ds, i):
    t = ds[i]["tools"]
    if not t:
        return "NO_TOOL"
    x = t[0]
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return "RAW"
    n = (x.get("name") or "?")
    return n.split("_")[0] if "_" in n else n.split(".")[0]


def main():
    ds, meta = L.load_meta()
    tr, va = L.splits_train_val()
    L.assert_no_test(tr, "p9/tr"); L.assert_no_test(va, "p9/va")
    A = np.load(L.CACHE / f"acts_L{OBS}.npy")
    det = {d["uuid"]: d for d in (json.loads(l) for l in open(L.BASE_DETAILS))}
    curves = json.loads((L.OUT / "llama_rho_curves.json").read_text())["rows"] \
        if False else json.loads((L.OUT / "llama_rho_curves.json").read_text())[CH]["rows"]

    out = {"channel": CH, "model": L.MODELS[L.MODEL_KEY]["repo"], "obs": OBS, "inj": INJ,
           "note": "validation-only; per-sample intervened predictions were not stored by the "
                   "Phase-8 sweep, so the transition table is reconstructed from stored aggregates"}

    # ---------- B1: transition table from stored aggregates ----------
    tbl = []
    for r in sorted(curves, key=lambda x: x["rho"]):
        real = r["real"]
        own_moved = real["own_before"] - real["own_after"]      # errors that left the ca_tc bucket
        to_gold = real["own_to_gold"]
        to_other_wrong = real["own_to_other_wrong"]
        # where did the "other wrong" ones land? other_after gives per-channel residuals after
        tbl.append({
            "rho": r["rho"], "real_net": real["net"], "fixed": real["fixed"], "broke": real["broke"],
            "own_before": real["own_before"], "own_after": real["own_after"],
            "own_moved_off_source": own_moved,
            "own_to_gold(target-hit)": to_gold, "own_to_other_wrong": to_other_wrong,
            "target_hit_rate_of_moved": (round(to_gold / own_moved, 3) if own_moved > 0 else None),
            "redistribution_rate_of_moved": (round(to_other_wrong / own_moved, 3) if own_moved > 0 else None),
            "other_channel_residuals_after": real.get("other_after"),
            "spec_z": r["spec_z"], "rand_mean": r["random"]["mean"],
            "n_ge_real": r["random"]["n_ge_real"], "reverse_net": r["reverse"]["net"],
        })
    out["B1_transition_table_from_aggregates"] = tbl

    # ---------- B2/B3: away-from-source vs toward-target geometry (train only) ----------
    err = [i for i in tr if meta[i]["etype"] == ET]                                  # gold=CA, pred=TC
    ref_gold = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == GOLD]     # correct CA
    corr_src = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == FROM_PRED]  # correct TC
    # the two "other wrong" destinations observed for gold=CA
    ca_direct = [i for i in tr if meta[i]["etype"] == "cannot_answer__direct"]
    ca_rfi = [i for i in tr if meta[i]["etype"] == "cannot_answer__request_for_info"]

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-12)

    m_err = A[err].mean(0); m_ref = A[ref_gold].mean(0); m_src = A[corr_src].mean(0)
    d_actual = unit(m_ref - m_err)          # the deployed DiffMean
    d_away = unit(m_err - m_src)            # pointing AWAY from the correct-source region
    d_away_neg = unit(m_src - m_err)        # toward the correct-source (tool_call) region
    d_toward = unit(m_ref - m_err)          # identical to actual by construction (ref==target gold)

    # destination-class means (where redistributed samples end up, behaviourally)
    m_cad = A[ca_direct].mean(0) if ca_direct else None
    m_car = A[ca_rfi].mean(0) if ca_rfi else None

    geom = {
        "n_err_train": len(err), "n_ref_gold_train": len(ref_gold),
        "n_correct_source_train": len(corr_src),
        "n_ca_direct_train": len(ca_direct), "n_ca_rfi_train": len(ca_rfi),
        "cos(d_actual, unit(m_src - m_err))": float(np.dot(d_actual, d_away_neg)),
        "note_d_toward": "d_toward == d_actual by construction (reference pool == target gold class)",
        "||m_ref - m_err||": float(np.linalg.norm(m_ref - m_err)),
        "||m_src - m_err||": float(np.linalg.norm(m_src - m_err)),
        "||m_ref - m_src||": float(np.linalg.norm(m_ref - m_src)),
    }
    if m_cad is not None:
        geom["cos(d_actual, unit(m_ca_direct - m_err))"] = float(np.dot(d_actual, unit(m_cad - m_err)))
        geom["||m_ca_direct - m_err||"] = float(np.linalg.norm(m_cad - m_err))
    if m_car is not None:
        geom["cos(d_actual, unit(m_ca_rfi - m_err))"] = float(np.dot(d_actual, unit(m_car - m_err)))
        geom["||m_ca_rfi - m_err||"] = float(np.linalg.norm(m_car - m_err))

    # Is the target class separable from the other-wrong destination classes at all?
    # (if cannot_answer and direct/rfi regions are collinear from the error cloud, a single
    #  DiffMean cannot select among them)
    if m_cad is not None:
        geom["cos(unit(m_ref-m_err), unit(m_ca_direct-m_err))_TARGET_vs_WRONGDEST"] = \
            float(np.dot(unit(m_ref - m_err), unit(m_cad - m_err)))
    if m_car is not None:
        geom["cos(unit(m_ref-m_err), unit(m_ca_rfi-m_err))_TARGET_vs_WRONGDEST"] = \
            float(np.dot(unit(m_ref - m_err), unit(m_car - m_err)))
    out["B2_direction_geometry_train"] = geom

    # ---------- projection / margin / domain / cluster profile of the error cloud ----------
    proj = A[err] @ d_actual
    margins, p_gold_runner = [], 0
    for i in err:
        lp = det[meta[i]["uuid"]]["avg_logp"]
        order = sorted(lp.items(), key=lambda kv: -kv[1])
        margins.append(order[0][1] - lp[GOLD])
        if order[1][0] == GOLD:
            p_gold_runner += 1
    doms = Counter(domain_of(ds, i) for i in err)
    Xe = StandardScaler().fit_transform(A[err])
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xe)
    out["B3_error_cloud_profile_train"] = {
        "proj_on_d_actual": {"mean": float(proj.mean()), "std": float(proj.std()),
                             "min": float(proj.min()), "max": float(proj.max())},
        "margin_top_minus_gold": {"median": float(np.median(margins)),
                                  "p25": float(np.percentile(margins, 25)),
                                  "p75": float(np.percentile(margins, 75))},
        "P(gold is runner-up)": round(p_gold_runner / len(err), 3),
        "n_domains": len(doms), "top_domains": doms.most_common(8),
        "kmeans2_silhouette": float(silhouette_score(Xe, km.labels_)),
        "kmeans2_minor_frac": float(min(np.bincount(km.labels_)) / len(err)),
    }

    # what fraction of the gold=cannot_answer errors have RFI/direct as runner-up (i.e. the
    # "attractor" the redistribution goes to)?
    runner = Counter()
    for i in err:
        lp = det[meta[i]["uuid"]]["avg_logp"]
        order = sorted(lp.items(), key=lambda kv: -kv[1])
        runner[order[1][0]] += 1
    out["B3_runner_up_distribution_of_ca_tc_errors"] = dict(runner)

    (OUT / "phase9_llama_ca_tc_redistribution.json").write_text(json.dumps(out, indent=2, default=float))
    with open(OUT / "phase9_llama_ca_tc_redistribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in tbl[0].keys() if k != "other_channel_residuals_after"])
        w.writeheader()
        for r in tbl:
            w.writerow({k: v for k, v in r.items() if k != "other_channel_residuals_after"})

    print("=== B1 transition table (validation, from stored aggregates) ===")
    print(f"{'rho':>5} {'net':>5} {'own':>10} {'moved':>6} {'->gold':>7} {'->wrong':>8} {'hit%':>6} {'z':>6}")
    for r in tbl:
        print(f"{r['rho']:>5} {r['real_net']:>+5d} {str(r['own_before'])+'->'+str(r['own_after']):>10} "
              f"{r['own_moved_off_source']:>6d} {r['own_to_gold(target-hit)']:>7d} "
              f"{r['own_to_other_wrong']:>8d} {str(r['target_hit_rate_of_moved']):>6} {str(r['spec_z']):>6}")
    print("\n=== B2 geometry ===")
    for k, v in geom.items():
        print(f"  {k}: {v}")
    print("\n=== B3 runner-up distribution of ca_tc errors (train) ===", dict(runner))
    print("\nwrote", OUT / "phase9_llama_ca_tc_redistribution.json")


if __name__ == "__main__":
    main()
