"""
phase9_5_m1_mechanism_audit.py — Phase 9.5: CPU-first M1 Mechanism Audit (Llama ca_tc).
=======================================================================================
EXISTING ARTIFACTS ONLY. No GPU, no test rows (assert_no_test everywhere), no Gate change,
no rerun, no reinterpretation of the Phase-9 FAIL verdicts.

Honest data constraint stated up front: neither the Phase-8 sweep nor the Phase-9 M1/M2 probe
stored PER-SAMPLE post-intervention predictions — only aggregates. Therefore:
  - actual responder identity (which routed error moved to gold) is UNAVAILABLE on CPU;
  - post-intervention margins are UNAVAILABLE on CPU.
Everything below is computed from what IS stored: per-sample BASELINE 4-class avg_logp
(baseline details), cached activations at 12 layers, the deterministic reconstruction of the
routed set / router / directions, and the stored aggregate cells (M1/M2 rho in {1,2,4};
deployed DiffMean rho in {0.25..8}). Anything else is labelled DEFERRED-GPU.

Sections
  A  per-sample table for the routed set (router conf, baseline margins, runner-up, domain,
     neighborhoods, projections on 6 axes at L22 and L18)
  B  functional-axis decomposition (evac / select / deployed / M1 / M2 / M2-logistic):
     cosine matrices at L22 and L18, raw norms, split-half stability, 2-D (evac, sel-perp)
     coordinates of each estimator, cross-layer transport cosines (H5)
  C  trajectory analysis from stored aggregates: movement counts + destination mix vs rho,
     nesting/monotonicity checks, generic-large-perturbation comparison vs randoms,
     perturbation size vs class-mean separations
  D  heterogeneity audit: gold-adjacency structure of the routed set; kmeans stability
     (split-half ARI) on the train error cloud; cluster profiles on preregistered metadata
Outputs: phase9_5_axes.json, phase9_5_per_sample_routed.csv/.json,
         phase9_5_trajectory.json, phase9_5_heterogeneity.json  (figures in a separate script)
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L
import phase9_m1m2_probe as P                      # reuse build_M1 / build_M2 VERBATIM
from phase9_llama_redistribution import domain_of  # same domain rule as Part B

OUT = L.ROOT / "final" / "results" / "phase9_mechanism_and_ood"
CH, GOLD, FROM_PRED, ET = P.CH, P.GOLD, P.FROM_PRED, P.ET
OBS, INJ, THR = P.OBS, P.INJ, P.THR
CLS = ["direct", "tool_call", "request_for_info", "cannot_answer"]
WRONG_DESTS = ["direct", "request_for_info"]       # the non-gold, non-source destinations


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def build_axes(A, tr, meta):
    """All six axes in one activation space, train-only. Returns dict name->(unit, raw_norm)."""
    err = [i for i in tr if meta[i]["etype"] == ET]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == GOLD]      # correct CA
    src = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == FROM_PRED]  # correct TC
    wrong = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] in WRONG_DESTS]
    L.assert_no_test(err + ref + src + wrong, "axes")
    m_err, m_ref, m_src = A[err].mean(0), A[ref].mean(0), A[src].mean(0)
    m_wrong = A[wrong].mean(0)
    ax = {}
    ax["deployed_dm"] = (unit(m_ref - m_err), float(np.linalg.norm(m_ref - m_err)))
    ax["evac"] = (unit(m_err - m_src), float(np.linalg.norm(m_err - m_src)))     # away-from-source (Part B d_away)
    ax["select"] = (unit(m_ref - m_wrong), float(np.linalg.norm(m_ref - m_wrong)))  # target vs pooled wrong-dest
    d1, i1 = P.build_M1(A, tr, meta)
    ax["M1"] = (d1, i1["raw_norm"])
    d2, i2, d2log = P.build_M2(A, tr, meta)
    ax["M2_lda"] = (d2, 1.0)
    ax["M2_logistic"] = (d2log, 1.0)
    pools = {"err": err, "ref": ref, "src": src, "wrong": wrong}
    return ax, pools, {"m_err": m_err, "m_ref": m_ref, "m_src": m_src, "m_wrong": m_wrong}


def splithalf_axis(A, tr, meta, name, n=20, seed=0):
    """Split-half cosine of each axis under halving of its contributing pools."""
    rng = np.random.RandomState(seed)
    cos = []
    for _ in range(n):
        h1, h2 = {}, {}
        for split, pool in (("tr1", None),):
            pass
        perm = {k: rng.permutation(v) for k, v in _pools_for(tr, meta).items()}
        halves = [{k: list(v[: len(v) // 2]) for k, v in perm.items()},
                  {k: list(v[len(v) // 2:]) for k, v in perm.items()}]
        ds = []
        for h in halves:
            ds.append(_axis_from_pools(A, h, name))
        if ds[0] is None or ds[1] is None:
            return None
        cos.append(float(ds[0] @ ds[1]))
    return {"mean": float(np.mean(cos)), "p25": float(np.percentile(cos, 25)),
            "p75": float(np.percentile(cos, 75)), "n": n}


def _pools_for(tr, meta):
    return {"err": [i for i in tr if meta[i]["etype"] == ET],
            "ref": [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == GOLD],
            "src": [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == FROM_PRED],
            "wrong": [i for i in tr if meta[i]["correct"] and meta[i]["gold"] in WRONG_DESTS]}


def _axis_from_pools(A, pools, name):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    e, r, s, w = (pools[k] for k in ("err", "ref", "src", "wrong"))
    if min(len(e), len(r), len(s), len(w)) < 3:
        return None
    if name == "deployed_dm":
        return unit(A[r].mean(0) - A[e].mean(0))
    if name == "evac":
        return unit(A[e].mean(0) - A[s].mean(0))
    if name == "select":
        return unit(A[r].mean(0) - A[w].mean(0))
    if name == "M1":
        return unit(A[r].mean(0) - A[s].mean(0))
    if name in ("M2_lda", "M2_logistic"):
        X = A[w + r]; y = np.array([0] * len(w) + [1] * len(r))
        sc = StandardScaler().fit(X)
        if name == "M2_lda":
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(X), y)
            return unit(lda.coef_[0] / sc.scale_)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42
                                 ).fit(sc.transform(X), y)
        return unit(clf.coef_[0] / sc.scale_)
    raise ValueError(name)


def decompose_on_plane(d, evac, sel):
    """Coordinates of unit d in the (evac, sel_perp) plane + in-plane fraction."""
    e = unit(evac)
    sp = unit(sel - (sel @ e) * e)          # sel component orthogonal to evac
    a, b = float(d @ e), float(d @ sp)
    return {"coef_evac": a, "coef_sel_perp": b,
            "in_plane_frac": a * a + b * b,          # ||projection||^2 of a unit vector
            "resid_frac": 1.0 - (a * a + b * b),
            "cos_evac_sel": float(unit(evac) @ unit(sel))}


def main():
    ds, meta = L.load_meta()
    tr, va = L.splits_train_val()
    L.assert_no_test(tr, "9.5/tr"); L.assert_no_test(va, "9.5/va")
    det = {d["uuid"]: d for d in (json.loads(l) for l in open(L.BASE_DETAILS))}
    A22 = np.load(L.CACHE / f"acts_L{OBS}.npy")
    A18 = np.load(L.CACHE / f"acts_L{INJ}.npy")

    # ---- routed set, reconstructed exactly as executed ----
    base_pred = {i: meta[i]["pred"] for i in list(tr) + list(va)}
    router = L.fit_router(A22, tr, va, meta, ET, FROM_PRED)
    vsc = L.rscore(router, A22, va)
    routed = [i for i in va if base_pred[i] == FROM_PRED and vsc[i] >= THR]
    L.assert_no_test(routed, "9.5/routed")
    own = [i for i in routed if meta[i]["etype"] == ET]
    assert len(routed) == 74, f"routed reconstruction mismatch: {len(routed)}"
    assert len(own) == 79 - (79 - len(own))  # own errors among routed
    print(f"[reconstruction] router AUC={router['val_auc']:.4f} routed={len(routed)} "
          f"own-channel among routed={len(own)} (own_before in cells = 79 counts ALL val "
          f"ca_tc errors; routed own = those the router fires on)")

    # =================================================================== B: AXES
    axes22, pools22, means22 = build_axes(A22, tr, meta)
    axes18, pools18, means18 = build_axes(A18, tr, meta)
    names = list(axes22.keys())
    C22 = [[float(axes22[a][0] @ axes22[b][0]) for b in names] for a in names]
    C18 = [[float(axes18[a][0] @ axes18[b][0]) for b in names] for a in names]
    cross = {n: float(axes22[n][0] @ axes18[n][0]) for n in names}   # H5: transportability
    # how the EXECUTED (L22-built) axes read in L18 functional coordinates
    exec_in_L18 = {n: {m: float(axes22[n][0] @ axes18[m][0]) for m in ("evac", "select", "deployed_dm")}
                   for n in ("deployed_dm", "M1", "M2_lda")}
    sh = {n: splithalf_axis(A22, tr, meta, n) for n in names}
    dec = {n: decompose_on_plane(axes22[n][0], axes22["evac"][0], axes22["select"][0])
           for n in ("deployed_dm", "M1", "M2_lda", "M2_logistic")}
    axes_out = {
        "space_note": "all executed directions were ESTIMATED at L22 and INJECTED at L18",
        "names": names,
        "raw_norms_L22": {n: axes22[n][1] for n in names},
        "cosine_L22": C22, "cosine_L18": C18,
        "cross_layer_same_axis_cos(L22,L18)": cross,
        "executed_L22_axes_read_in_L18_coords": exec_in_L18,
        "splithalf_L22": sh,
        "decomposition_on_(evac,sel_perp)_L22": dec,
        "pool_sizes": {k: len(v) for k, v in pools22.items()},
    }
    (OUT / "phase9_5_axes.json").write_text(json.dumps(axes_out, indent=2, default=float))
    print("\n=== B: cosine matrix at L22 (estimation space) ===")
    print("        " + " ".join(f"{n[:9]:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:8]:>8}" + " ".join(f"{C22[i][j]:>10.3f}" for j in range(len(names))))
    print("cross-layer cos(L22,L18) per axis:", {k: round(v, 3) for k, v in cross.items()})
    print("decomposition on (evac, sel_perp):")
    for n, d in dec.items():
        print(f"  {n:>12}: evac={d['coef_evac']:+.3f} sel_perp={d['coef_sel_perp']:+.3f} "
              f"in-plane={d['in_plane_frac']:.3f} resid={d['resid_frac']:.3f}")

    # =================================================================== A: PER-SAMPLE
    rows = []
    # neighborhood pools (train)
    lab = {}
    for i in tr:
        if meta[i]["etype"] == ET: lab[i] = "err_ca_tc"
        elif meta[i]["correct"]: lab[i] = "corr_" + meta[i]["gold"]
    nb_idx = list(lab.keys())
    NB = A22[nb_idx]; NBn = NB / (np.linalg.norm(NB, axis=1, keepdims=True) + 1e-12)
    for i in routed:
        d = det[meta[i]["uuid"]]
        lp = d["avg_logp"]
        order = sorted(lp.items(), key=lambda kv: -kv[1])
        gap_gold = lp[order[0][0]] - lp[GOLD]                      # top minus gold (>=0 on errors)
        best_wrong = max(lp[w] for w in WRONG_DESTS)
        h22, h18 = A22[i], A18[i]
        v = h22 / (np.linalg.norm(h22) + 1e-12)
        sims = NBn @ v
        nn = np.argsort(-sims)[:10]
        nbc = Counter(lab[nb_idx[j]] for j in nn)
        rows.append({
            "idx": i, "uuid": meta[i]["uuid"], "is_own_channel_err": meta[i]["etype"] == ET,
            "gold": meta[i]["gold"], "baseline_pred": meta[i]["pred"],
            "router_score": round(float(vsc[i]), 4),
            "n_prompt_tok": d["n_prompt_tok"], "domain": domain_of(ds, i),
            "lp_direct": lp["direct"], "lp_tool_call": lp["tool_call"],
            "lp_rfi": lp["request_for_info"], "lp_ca": lp["cannot_answer"],
            "runner_up": order[1][0], "gold_rank": [k for k, _ in order].index(GOLD) + 1,
            "gap_top_minus_gold": round(gap_gold, 4),
            "gap_gold_minus_bestwrong": round(lp[GOLD] - best_wrong, 4),
            "proj22_deployed": round(float((h22 - means22["m_err"]) @ axes22["deployed_dm"][0]), 4),
            "proj22_M1": round(float((h22 - means22["m_err"]) @ axes22["M1"][0]), 4),
            "proj22_M2": round(float((h22 - means22["m_err"]) @ axes22["M2_lda"][0]), 4),
            "proj22_evac": round(float((h22 - means22["m_err"]) @ axes22["evac"][0]), 4),
            "proj22_select": round(float((h22 - means22["m_err"]) @ axes22["select"][0]), 4),
            "proj18_M1_L22axis": round(float((h18 - means18["m_err"]) @ axes22["M1"][0]), 4),
            "nn10_corr_ca": nbc.get("corr_cannot_answer", 0),
            "nn10_corr_tc": nbc.get("corr_tool_call", 0),
            "nn10_corr_direct": nbc.get("corr_direct", 0),
            "nn10_corr_rfi": nbc.get("corr_request_for_info", 0),
            "nn10_err_ca_tc": nbc.get("err_ca_tc", 0),
            "actual_movement_at_rho4": "UNAVAILABLE (per-sample outcomes not stored; DEFERRED-GPU)",
        })
    with open(OUT / "phase9_5_per_sample_routed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    (OUT / "phase9_5_per_sample_routed.json").write_text(json.dumps(rows, indent=2, default=float))

    own_rows = [r for r in rows if r["is_own_channel_err"]]
    ru = Counter(r["runner_up"] for r in own_rows)
    gold_adj = [r for r in own_rows if r["runner_up"] == GOLD]
    print(f"\n=== A: routed set = {len(rows)} ({len(own_rows)} own-channel errors) ===")
    print("runner-up distribution of routed own errors:", dict(ru))
    print(f"gold-adjacent (runner_up==gold): {len(gold_adj)}  "
          f"median gap_top_minus_gold={np.median([r['gap_top_minus_gold'] for r in own_rows]):.3f}")

    # =================================================================== C: TRAJECTORY
    m1m2 = json.loads((OUT / "phase9_m1m2_results.json").read_text())
    curves = json.loads((L.OUT / "llama_rho_curves.json").read_text())[CH]["rows"]
    base_nt = L.metrics(va, meta, base_pred, base_pred,
                        P.CHMAP)["nt_before"]
    traj = {"base_nt": base_nt, "methods": {}}
    for mname, cells in (("deployed", sorted(curves, key=lambda r: r["rho"])),
                         ("M1", sorted(m1m2["M1"]["rows"], key=lambda r: r["rho"])),
                         ("M2", sorted(m1m2["M2"]["rows"], key=lambda r: r["rho"]))):
        seq = []
        for r in cells:
            real = r["real"]
            moved = real["own_before"] - real["own_after"]
            dest = {k: real["other_after"][k] - base_nt[k] for k in real["other_after"]}
            seq.append({"rho": r["rho"], "moved": moved, "to_gold": real["own_to_gold"],
                        "to_wrong": real["own_to_other_wrong"], "dest_delta": dest,
                        "net": real["net"], "val_acc": real.get("val_acc"),
                        "rand_moved_proxy": None})
        # monotonicity of the moved-set size (nesting under a 1-D crossing model implies monotone)
        mv = [s["moved"] for s in seq]
        traj["methods"][mname] = {
            "cells": seq, "moved_sequence": mv,
            "moved_monotone_nondecreasing": all(mv[i] <= mv[i + 1] for i in range(len(mv) - 1)),
        }
    # generic-movement scale of randoms at rho=4 (from stored random TG lists: TG = g - w and
    # moved >= g + w >= |TG|, so |TG| lower-bounds random movement)
    for mname in ("M1", "M2"):
        r4 = [r for r in m1m2[mname]["rows"] if r["rho"] == 4.0][0]
        tg = r4["random_target_gain"]["values"]
        traj["methods"][mname]["random_rho4_TG_values"] = tg
        traj["methods"][mname]["random_rho4_min_movement_lower_bound"] = \
            {"mean_|TG|": float(np.mean(np.abs(tg))), "max_|TG|": int(np.max(np.abs(tg)))}
    # perturbation size vs geometry
    med_inj = float(np.median(np.linalg.norm(A18[tr], axis=1)))
    med_obs = float(np.median(np.linalg.norm(A22[tr], axis=1)))
    traj["scale"] = {
        "med_inj_L18": med_inj, "med_obs_L22": med_obs,
        "delta_norm_at_rho": {str(r): r * med_inj for r in (1.0, 2.0, 4.0)},
        "class_separations_L22": {"||m_ref-m_err||": axes22["deployed_dm"][1],
                                  "||m_err-m_src||": axes22["evac"][1],
                                  "||m_ref-m_wrongpool||": axes22["select"][1],
                                  "||m_ref-m_src|| (M1 raw)": axes22["M1"][1]},
        "class_separations_L18": {"||m_ref-m_err||": axes18["deployed_dm"][1],
                                  "||m_err-m_src||": axes18["evac"][1],
                                  "||m_ref-m_wrongpool||": axes18["select"][1],
                                  "||m_ref-m_src|| (M1 raw)": axes18["M1"][1]},
    }
    (OUT / "phase9_5_trajectory.json").write_text(json.dumps(traj, indent=2, default=float))
    print("\n=== C: movement vs rho (from stored aggregates) ===")
    for mname, d in traj["methods"].items():
        print(f"  {mname:>9}: moved={d['moved_sequence']} monotone={d['moved_monotone_nondecreasing']}")
        for s in d["cells"]:
            print(f"        rho={s['rho']:<5} moved={s['moved']:>3} gold={s['to_gold']:>2} "
                  f"wrong={s['to_wrong']:>2} dest_delta={s['dest_delta']} val_acc={s['val_acc']}")
    print(f"  scale: ||dh||(rho=4)={4*med_inj:.2f} vs L18 class separations "
          f"{traj['scale']['class_separations_L18']}")

    # =================================================================== D: HETEROGENEITY
    err_tr = pools22["err"]
    Xe = StandardScaler().fit_transform(A22[err_tr])
    het = {"n_err_train": len(err_tr)}
    # split-half ARI stability of kmeans-2/3 (nearest-centroid transfer between halves)
    rng = np.random.RandomState(0)
    for k in (2, 3):
        aris, sils = [], []
        for _ in range(20):
            perm = rng.permutation(len(err_tr))
            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
            km1 = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xe[h1])
            km2 = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xe[h2])
            l1 = km1.predict(Xe); l2 = km2.predict(Xe)
            aris.append(adjusted_rand_score(l1, l2))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xe)
        sil = float(silhouette_score(Xe, km.labels_))
        het[f"kmeans{k}"] = {"splithalf_ARI_mean": float(np.mean(aris)),
                             "splithalf_ARI_p25": float(np.percentile(aris, 25)),
                             "splithalf_ARI_p75": float(np.percentile(aris, 75)),
                             "silhouette_full": sil,
                             "sizes": np.bincount(km.labels_).tolist()}
        # profile the full-fit clusters on preregistered metadata only
        prof = []
        for c in range(k):
            members = [err_tr[j] for j in range(len(err_tr)) if km.labels_[j] == c]
            rus = Counter(); gaps = []
            for i in members:
                lp = det[meta[i]["uuid"]]["avg_logp"]
                o = sorted(lp.items(), key=lambda kv: -kv[1])
                rus[o[1][0]] += 1; gaps.append(o[0][1] - lp[GOLD])
            doms = Counter(domain_of(ds, i) for i in members)
            prof.append({"cluster": c, "n": len(members),
                         "P(runner_up==gold)": round(rus.get(GOLD, 0) / max(len(members), 1), 3),
                         "runner_up": dict(rus), "median_gap_top_minus_gold": float(np.median(gaps)),
                         "top_domains": doms.most_common(5),
                         "median_ntok": float(np.median([det[meta[i]['uuid']]["n_prompt_tok"] for i in members]))})
        het[f"kmeans{k}_profiles"] = prof
    # gold-adjacency of the routed own errors: the candidate responder pool
    het["routed_own"] = {
        "n": len(own_rows),
        "runner_up_distribution": dict(ru),
        "n_gold_adjacent": len(gold_adj),
        "gold_adjacent_median_gap": (float(np.median([r["gap_top_minus_gold"] for r in gold_adj]))
                                     if gold_adj else None),
        "non_adjacent_median_gap": float(np.median([r["gap_top_minus_gold"] for r in own_rows
                                                    if r["runner_up"] != GOLD])),
        "observed_to_gold_counts": {"deployed@4": 10, "deployed@8": 28, "M1@4": 9, "M2@4": 0},
        "note": "if responders ~= gold-adjacent pool, to_gold should saturate near n_gold_adjacent",
    }
    # do gold-adjacent errors differ on projections / router conf? (preregistered features only)
    ga = np.array([r["runner_up"] == GOLD for r in own_rows])
    for feat in ("router_score", "proj22_M1", "proj22_select", "proj22_evac",
                 "gap_top_minus_gold", "n_prompt_tok", "nn10_corr_ca"):
        x = np.array([r[feat] for r in own_rows], dtype=float)
        het["routed_own"][f"{feat}: gold_adj_vs_rest_median"] = \
            [float(np.median(x[ga])), float(np.median(x[~ga]))]
    (OUT / "phase9_5_heterogeneity.json").write_text(json.dumps(het, indent=2, default=float))
    print("\n=== D: heterogeneity ===")
    for k in (2, 3):
        d = het[f"kmeans{k}"]
        print(f"  kmeans{k}: split-half ARI={d['splithalf_ARI_mean']:.3f} "
              f"[{d['splithalf_ARI_p25']:.3f},{d['splithalf_ARI_p75']:.3f}] "
              f"silhouette={d['silhouette_full']:.3f} sizes={d['sizes']}")
        for p in het[f"kmeans{k}_profiles"]:
            print(f"    c{p['cluster']}: n={p['n']} P(ru==gold)={p['P(runner_up==gold)']} "
                  f"gap_med={p['median_gap_top_minus_gold']:.3f} doms={p['top_domains'][:3]}")
    print(f"  routed own: n={len(own_rows)} gold-adjacent={len(gold_adj)} "
          f"(observed to_gold: deployed@4=10, M1@4=9, deployed@8=28)")

    print("\nwrote phase9_5_axes.json, phase9_5_per_sample_routed.{csv,json}, "
          "phase9_5_trajectory.json, phase9_5_heterogeneity.json")


if __name__ == "__main__":
    main()
