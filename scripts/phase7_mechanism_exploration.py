"""
phase7_mechanism_exploration.py — CPU-ONLY mechanism study of the three representative cases.
==============================================================================================
Cases: qwen/ca_rfi (positive control), qwen/tc_rfi (false-admission), mistral/ca_tc
(non-actionable/saturation). Uses ONLY: cached activations (train/val rows), archived
summaries, and the completed Phase-7 sweep rows. NO model load, NO GPU, NO test rows
(firewalled), NO modification of the running sweep.

Outputs:
  unified_evidence_table.csv          (Part A)
  mechanism_geometry.json / .csv      (Part B1-B3 quantities)
  behavioral_response_summary.csv     (Part B4, from completed sweep rows)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

RHO_MAX = 8.0
def _load_res():
    res = json.loads((P.OUT / "rho_specificity_results.json").read_text()) \
        if (P.OUT / "rho_specificity_results.json").exists() else {}
    for mk in P.MODELS:
        sf = P.CACHE_OUT / f"phase7_state_{mk}.json"
        if sf.exists():
            st = json.loads(sf.read_text())
            for ch, blob in st.items():
                if len(blob.get("rows", [])) >= 6 and ch not in res.get(mk, {}):
                    res.setdefault(mk, {})[ch] = blob
    return res
RES = _load_res()

CASES = [
    # (model_key, channel, gold, from_pred, obs, inj, method, thr, role)
    ("qwen25_7b", "ca_rfi", "cannot_answer", "request_for_info", 20, 18, "diffmean", 0.8,
     "positive control"),
    ("qwen25_7b", "tc_rfi", "tool_call", "request_for_info", 24, 24, "diffmean", 0.4,
     "false-admission (val-pass, test-null)"),
    ("mistral7b_v03", "ca_tc", "cannot_answer", "tool_call", 22, 18, "diffmean", 0.8,
     "non-actionable / saturation"),
]

# archived context (historical; NOT Phase-7 inputs) — from committed prior-phase files
ARCHIVED = {
    ("qwen25_7b", "ca_rfi"): {"boot": 1.0, "test_net": 16, "test_z": 4.82, "test_nge": "0/20",
                              "locked_outcome": "CONFIRMED specific", "decision": "actionable"},
    ("qwen25_7b", "tc_rfi"): {"boot": 1.0, "test_net": 0, "test_z": 0.82, "test_nge": "11/20",
                              "locked_outcome": "NULL (failed gate 5/8)", "decision": "non-actionable"},
    ("mistral7b_v03", "ca_tc"): {"boot": 1.0, "test_net": 62, "test_z": 1.91, "test_nge": "1/20",
                                 "locked_outcome": "positive Net, failed pre-registered gate (crit 4; z<2)",
                                 "decision": "non-actionable (behavioral-only)"},
}


def eff_rank(eig):
    eig = np.asarray(eig); eig = eig[eig > 0]
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def analyze_case(mk, ch, gold, from_pred, obs, inj, method, thr):
    M = P.MODELS[mk]
    tr, va = P.splits()
    meta = P.load_meta(mk)
    et = f"{gold}__{from_pred}"
    A = np.load(M["cache"] / f"acts_L{obs}.npy")
    err = [i for i in tr if meta[i]["etype"] == et]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == gold]
    P.assert_no_test(err, "mech/err"); P.assert_no_test(ref, "mech/ref")

    med_obs = float(np.median(np.linalg.norm(A[tr], axis=1)))

    # ---- B1: heterogeneity of the error cloud ----
    Xe = StandardScaler().fit_transform(A[err])
    n_comp = min(50, len(err) - 1)
    pca_e = PCA(n_components=n_comp, random_state=42).fit(Xe)
    evr = pca_e.explained_variance_ratio_
    er = eff_rank(pca_e.explained_variance_)
    sil2 = sil3 = None; minor2 = None
    if len(err) >= 20:
        km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xe)
        sil2 = float(silhouette_score(Xe, km2.labels_))
        minor2 = float(min(np.bincount(km2.labels_)) / len(err))
        km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xe)
        sil3 = float(silhouette_score(Xe, km3.labels_))
    within_var = float(np.trace(np.cov(A[err].T)) / M["hidden"])
    ref_var = float(np.trace(np.cov(A[ref].T)) / M["hidden"]) if len(ref) > 2 else None

    # ---- B2: direction geometry ----
    dm, dm_norm = P.diffmean(A, err, ref)
    pc1, pc1_norm = P.pca_dir(A, err, ref, 1)
    cos_dm_pc1 = float(abs(np.dot(dm, pc1)))
    # split-half stability
    rng = np.random.RandomState(0); cos_sh = []
    for _ in range(20):
        pe, pr = rng.permutation(err), rng.permutation(ref)
        d1, _ = P.diffmean(A, list(pe[:len(pe)//2]), list(pr[:len(pr)//2]))
        d2, _ = P.diffmean(A, list(pe[len(pe)//2:]), list(pr[len(pr)//2:]))
        cos_sh.append(float(d1 @ d2))
    # alignment of DM with high-variance axes of the err+ref cloud (standardized)
    idx = err + ref
    scb = StandardScaler().fit(A[idx])
    pca_b = PCA(n_components=min(50, len(idx) - 1), random_state=42).fit(scb.transform(A[idx]))
    dm_std = dm / (scb.scale_ + 1e-12); dm_std /= (np.linalg.norm(dm_std) + 1e-12)
    proj = pca_b.components_ @ dm_std
    frac_top = {k: float((proj[:k] ** 2).sum()) for k in (1, 5, 10, 50) if k <= len(proj)}

    # ---- B3: router geometry (train fit, val margins) ----
    router = P.fit_router(A, tr, va, meta, et, from_pred)
    vpos = [i for i in va if meta[i]["etype"] == et]
    vneg = [i for i in va if meta[i]["pred"] == from_pred and meta[i]["etype"] != et]
    sc_pos = list(P.rscore(router, A, vpos).values()) if vpos else []
    sc_neg = list(P.rscore(router, A, vneg).values()) if vneg else []
    fp_at_thr = [i for i in vneg if P.rscore(router, A, [i])[i] >= thr]
    fp_correct = sum(1 for i in fp_at_thr if meta[i]["correct"])
    fp_othererr = len(fp_at_thr) - fp_correct
    w_raw = router["clf"].coef_[0] / router["scaler"].scale_
    w_raw = w_raw / (np.linalg.norm(w_raw) + 1e-12)
    cos_dm_router = float(np.dot(dm, -w_raw))

    return {
        "model": mk, "channel": ch, "obs": obs, "inj": inj, "method": method,
        "norm_depth": round(obs / M["layers"], 3),
        "n_err_train": len(err), "n_ref_train": len(ref),
        "med_act_norm_obs": round(med_obs, 4),
        # B1
        "pc1_evr": round(float(evr[0]), 4), "pc5_evr_cum": round(float(evr[:5].sum()), 4),
        "eff_rank_err_cloud": round(er, 1),
        "top1_eig_concentration": round(float(pca_e.explained_variance_[0]
                                              / pca_e.explained_variance_.sum()), 4),
        "silhouette_k2": (round(sil2, 3) if sil2 is not None else None),
        "silhouette_k3": (round(sil3, 3) if sil3 is not None else None),
        "minor_cluster_frac_k2": (round(minor2, 3) if minor2 is not None else None),
        "within_channel_var_per_dim": round(within_var, 6),
        "ref_var_per_dim": (round(ref_var, 6) if ref_var else None),
        # B2
        "dm_norm": round(dm_norm, 4), "pca1_norm": round(pc1_norm, 4),
        "cos_dm_pc1": round(cos_dm_pc1, 4),
        "dm_norm_over_actnorm": round(dm_norm / med_obs, 4),
        "dm_splithalf_cos_mean": round(float(np.mean(cos_sh)), 3),
        "dm_splithalf_cos_min": round(float(np.min(cos_sh)), 3),
        "dm_frac_in_top1_pc": round(frac_top.get(1, float("nan")), 4),
        "dm_frac_in_top5_pc": round(frac_top.get(5, float("nan")), 4),
        "dm_frac_in_top10_pc": round(frac_top.get(10, float("nan")), 4),
        "dm_frac_in_top50_pc": round(frac_top.get(50, float("nan")), 4),
        # B3
        "router_val_auc": round(router["val_auc"] or 0, 4),
        "val_pos_score_median": (round(float(np.median(sc_pos)), 3) if sc_pos else None),
        "val_neg_score_median": (round(float(np.median(sc_neg)), 3) if sc_neg else None),
        "val_margin_median_gap": (round(float(np.median(sc_pos) - np.median(sc_neg)), 3)
                                  if sc_pos and sc_neg else None),
        "val_fp_at_thr": len(fp_at_thr), "val_fp_correct": fp_correct,
        "val_fp_other_error": fp_othererr,
        "cos_dm_routerdir": round(cos_dm_router, 4),
    }


def behavioral_rows():
    out = []
    for mk, chans in RES.items():
        for ch, blob in chans.items():
            rows = sorted(blob["rows"], key=lambda r: r["rho"])
            P.experiment_rows(rows)
            best = max(rows, key=lambda r: r["real"]["net"])
            # best gate-relevant point = max z among interior points with net>0
            interior = [r for r in rows if r["rho"] < RHO_MAX and r["real"]["net"] > 0
                        and r["spec_z"] is not None]
            bz = max(interior, key=lambda r: r["spec_z"]) if interior else None
            for r in rows:
                real_net = r["real"]["net"]
                out.append({
                    "model": mk, "channel": ch, "rho": r["rho"], "split": "val",
                    "real_net": real_net, "reverse_net": r["reverse"]["net"],
                    "rand_mean": r["random"]["mean"], "rand_std": r["random"]["std"],
                    "rand_max": r["random"]["max"], "n_ge_real": r["random"]["n_ge_real"],
                    "spec_z": r["spec_z"],
                    "own_before": r["real"]["own_before"], "own_after": r["real"]["own_after"],
                    "own_to_gold": r["real"]["own_to_gold"],
                    "own_to_other_wrong": r["real"]["own_to_other_wrong"],
                    "damage_on_correct": r["real"]["damage_on_correct"],
                    "reverse_retention": (round(r["reverse"]["net"] / real_net, 3)
                                          if real_net > 0 else None),
                    "random_fraction_of_real": (round(r["random"]["mean"] / real_net, 3)
                                                if real_net > 0 else None),
                    "is_best_net": r["rho"] == best["rho"],
                    "is_best_interior_z": (bz is not None and r["rho"] == bz["rho"]),
                })
    return out


def main():
    geo = [analyze_case(*c[:8]) for c in CASES]
    json.dump(geo, open(P.OUT / "mechanism_geometry.json", "w"), indent=2)
    with open(P.OUT / "mechanism_geometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(geo[0].keys())); w.writeheader()
        for g in geo:
            w.writerow(g)

    beh = behavioral_rows()
    with open(P.OUT / "behavioral_response_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(beh[0].keys())); w.writeheader()
        for b in beh:
            w.writerow(b)

    # ---- Part A unified table (merging geometry, sweep, archived context) ----
    uni = []
    for (mk, ch, gold, fp_, obs, inj, method, thr, role), g in zip(CASES, geo):
        blob = RES.get(mk, {}).get(ch)
        arch = ARCHIVED[(mk, ch)]
        if blob:
            rows = sorted(blob["rows"], key=lambda r: r["rho"])
            interior = [r for r in rows if r["rho"] < RHO_MAX and r["real"]["net"] > 0
                        and r["spec_z"] is not None]
            bz = max(interior, key=lambda r: r["spec_z"]) if interior else None
            bnet = max(rows, key=lambda r: r["real"]["net"])
            pick = bz or bnet
        else:
            pick = None
        uni.append({
            "model": mk, "dataset": "When2Call/mcq", "channel": ch, "role": role,
            "support_train": g["n_err_train"], "support_ref_train": g["n_ref_train"],
            "bootstrap_stability": arch["boot"],
            "router_val_auc": g["router_val_auc"],
            "router_margin_median_gap_val": g["val_margin_median_gap"],
            "obs_layer": obs, "inj_layer": inj, "norm_depth": g["norm_depth"],
            "diffmean_norm": g["dm_norm"], "pca1_norm": g["pca1_norm"],
            "cos_dm_pc1": g["cos_dm_pc1"], "dir_norm_over_actnorm": g["dm_norm_over_actnorm"],
            "best_rho_interior_z": (pick["rho"] if pick else "unavailable"),
            "real_net_at_pick": (pick["real"]["net"] if pick else "unavailable"),
            "reverse_net_at_pick": (pick["reverse"]["net"] if pick else "unavailable"),
            "rand_mean_at_pick": (pick["random"]["mean"] if pick else "unavailable"),
            "rand_std_at_pick": (pick["random"]["std"] if pick else "unavailable"),
            "rand_max_at_pick": (pick["random"]["max"] if pick else "unavailable"),
            "frac_rand_ge_real_at_pick": (round(pick["random"]["n_ge_real"] / pick["random"]["n"], 3)
                                          if pick else "unavailable"),
            "spec_z_at_pick": (pick["spec_z"] if pick else "unavailable"),
            "own_residual_reduction_at_pick": (
                round(1 - pick["real"]["own_after"] / pick["real"]["own_before"], 3)
                if pick and pick["real"]["own_before"] else "unavailable"),
            "collateral_damage_at_pick": (pick["real"]["damage_on_correct"] if pick else "unavailable"),
            "validation_outcome_phase7": ("PASS-conjunction" if (mk, ch) == ("qwen25_7b", "ca_rfi")
                                          else "PASS-conjunction (later test-null)" if ch == "tc_rfi"
                                          else "FAIL at all rho"),
            "locked_test_outcome_archived": arch["locked_outcome"],
            "final_actionability_decision": arch["decision"],
        })
    with open(P.OUT / "unified_evidence_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(uni[0].keys())); w.writeheader()
        for u in uni:
            w.writerow(u)

    for g in geo:
        print(f"{g['model']}/{g['channel']}: effrank={g['eff_rank_err_cloud']} "
              f"pc1evr={g['pc1_evr']} sil2={g['silhouette_k2']} cos(dm,pc1)={g['cos_dm_pc1']} "
              f"splithalf={g['dm_splithalf_cos_mean']} top10frac={g['dm_frac_in_top10_pc']} "
              f"AUC={g['router_val_auc']} margin_gap={g['val_margin_median_gap']} "
              f"fp@thr={g['val_fp_at_thr']}({g['val_fp_correct']}corr/{g['val_fp_other_error']}err) "
              f"cos(dm,router)={g['cos_dm_routerdir']}")
    print("wrote unified_evidence_table.csv, mechanism_geometry.{json,csv}, "
          "behavioral_response_summary.csv")


if __name__ == "__main__":
    main()
