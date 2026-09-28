"""
phase3_support_geometry.py — Steps 3 & 4 (CPU part): channel support + geometry search.
=========================================================================================
Consumes the regenerated baseline + activation caches. Produces:

  channel_support/channel_support_summary.json + channel_support_table.csv
  geometry/layer_direction_router_summary.json + layer_direction_router_table.csv
  geometry/GEOMETRY_ANALYSIS.md

R1/R2 decisions are made here (train/val only; no model, no test intervention).
"""
from __future__ import annotations
import json, logging, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L

log = logging.getLogger("p3.geom")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

SUP = L.OUT / "channel_support"; SUP.mkdir(parents=True, exist_ok=True)
GEO = L.OUT / "geometry"; GEO.mkdir(parents=True, exist_ok=True)


def bootstrap_stability(meta, tr, va, te, etype, n_boot=200, seed=42):
    """Archived default-threshold stability: train>=50 AND min(val,test)>=15 per resample."""
    rng = np.random.RandomState(seed)
    tr_et = np.array([meta[i]["etype"] == etype for i in tr])
    va_et = np.array([meta[i]["etype"] == etype for i in va])
    te_et = np.array([meta[i]["etype"] == etype for i in te])
    ok = 0
    for _ in range(n_boot):
        t = tr_et[rng.randint(0, len(tr), len(tr))].sum()
        v = va_et[rng.randint(0, len(va), len(va))].sum()
        s = te_et[rng.randint(0, len(te), len(te))].sum()
        if t >= 50 and min(v, s) >= 15:
            ok += 1
    return ok / n_boot


def main():
    ds, meta = L.load_meta_and_dataset()
    tr, va, te = L.load_splits()
    acts = {Lx: np.load(L.CACHE_DIR / f"acts_L{Lx}.npy") for Lx in L.OBS_LAYERS}

    # ---------- Step 3: channel support ----------
    support, rows = {}, []
    for ch, cfg in L.NEW_CHANNELS.items():
        et = cfg["etype"]
        cnt = {"all": sum(1 for m in meta if m["etype"] == et),
               "train": sum(1 for i in tr if meta[i]["etype"] == et),
               "val": sum(1 for i in va if meta[i]["etype"] == et),
               "test": sum(1 for i in te if meta[i]["etype"] == et)}
        ref = sum(1 for i in tr if meta[i]["correct"] and meta[i]["gold"] == cfg["ref_gold"])
        elig = {s: sum(1 for i in idxs if meta[i]["pred"] == cfg["from_pred"])
                for s, idxs in [("train", tr), ("val", va), ("test", te)]}
        boot = bootstrap_stability(meta, tr, va, te, et)
        scope_counts = cnt["train"] >= 30 and cnt["val"] >= 5 and ref >= 30
        support[ch] = {"etype": et, "counts": cnt, "ref_pool_train": ref,
                       "gate_eligible_pred_side": elig,
                       "bootstrap_stable_freq_default_thr": boot,
                       "scope_filter_counts_pass": bool(scope_counts)}
        rows.append([ch, et, cnt["all"], cnt["train"], cnt["val"], cnt["test"], ref,
                     elig["val"], elig["test"], round(boot, 3), scope_counts])
        log.info("support %s: %s ref=%d boot=%.2f counts_pass=%s", ch, cnt, ref, boot, scope_counts)

    # ---------- Step 4 (CPU): per-layer geometry + routers ----------
    geom, table = {}, []
    for ch, cfg in L.NEW_CHANNELS.items():
        et, fp = cfg["etype"], cfg["from_pred"]
        per_layer = {}
        for Lx in L.OBS_LAYERS:
            A = acts[Lx]
            u_dm, nrm, mn, ne, nr = L.direction(A, tr, meta, et, cfg["ref_gold"], "diffmean")
            u_p1, p1n, _, _, _ = L.direction(A, tr, meta, et, cfg["ref_gold"], "pca1")
            cos = abs(float(u_dm @ u_p1))
            r_m = L.fit_router(A, tr, va, meta, et, fp, "matched_pred")
            r_c = L.fit_router(A, tr, va, meta, et, fp, "all_correct")
            per_layer[Lx] = {"dm_norm": round(nrm, 3), "median_norm": round(mn, 3),
                             "ratio": round(nrm / mn, 4), "pca1_norm": round(p1n, 3),
                             "cos_dm_pca1": round(cos, 4),
                             "router_val_auc_matched": (round(r_m["val_auc"], 4)
                                                        if r_m["val_auc"] else None),
                             "router_val_auc_allcorrect": (round(r_c["val_auc"], 4)
                                                           if r_c["val_auc"] else None),
                             "n_err_train": ne, "n_ref_train": nr,
                             "n_router_neg_matched": r_m["n_neg"]}
            table.append([ch, Lx, per_layer[Lx]["dm_norm"], per_layer[Lx]["median_norm"],
                          per_layer[Lx]["ratio"], per_layer[Lx]["cos_dm_pca1"],
                          per_layer[Lx]["router_val_auc_matched"],
                          per_layer[Lx]["router_val_auc_allcorrect"], ne, nr])
        survivors = [Lx for Lx in L.OBS_LAYERS if per_layer[Lx]["dm_norm"] >= L.DEGEN_NORM]
        if not survivors:
            geom[ch] = {"per_layer": per_layer, "R1_obs": None,
                        "R1_verdict": "NO-GO (all layers degenerate, norm < 5)"}
            log.warning("%s: NO layer survives R1", ch)
            continue
        r1 = max(survivors, key=lambda Lx: (per_layer[Lx]["ratio"],
                                            per_layer[Lx]["router_val_auc_matched"] or 0))
        cos_r1 = per_layer[r1]["cos_dm_pca1"]
        methods = ["diffmean", "pca1"] if cos_r1 < L.COS_THR else ["diffmean"]
        geom[ch] = {"per_layer": per_layer, "R1_obs": r1,
                    "R1_survivors": survivors,
                    "R1_gated_out": [Lx for Lx in L.OBS_LAYERS if Lx not in survivors],
                    "cos_dm_pca1_at_R1": cos_r1, "R2_method_candidates": methods,
                    "inj_candidates": sorted({max(0, r1 + o) for o in L.INJ_OFFSETS})}
        log.info("%s: R1 obs=L%d (survivors %s) cos=%.3f methods=%s",
                 ch, r1, survivors, cos_r1, methods)

    # cross-channel direction geometry at shared layers (interference)
    inter = {}
    for Lx in L.OBS_LAYERS:
        A = acts[Lx]
        uA, _, _, _, _ = L.direction(A, tr, meta, L.NEW_CHANNELS["ca_rfi"]["etype"],
                                     "cannot_answer", "diffmean")
        uB, _, _, _, _ = L.direction(A, tr, meta, L.NEW_CHANNELS["tc_rfi"]["etype"],
                                     "tool_call", "diffmean")
        inter[f"L{Lx}"] = round(float(uA @ uB), 4)

    json.dump(support, open(SUP / "channel_support_summary.json", "w"), indent=2)
    with open(SUP / "channel_support_table.csv", "w") as f:
        f.write("channel,etype,count_all,train,val,test,ref_pool_train,"
                "gate_eligible_val,gate_eligible_test,bootstrap_stable_freq,scope_counts_pass\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    json.dump({"geometry": geom, "cos_between_new_channel_dirs_by_layer": inter},
              open(GEO / "layer_direction_router_summary.json", "w"), indent=2)
    with open(GEO / "layer_direction_router_table.csv", "w") as f:
        f.write("channel,obs_layer,dm_norm,median_act_norm,ratio,cos_dm_pca1,"
                "router_val_auc_matched,router_val_auc_allcorrect,n_err_train,n_ref_train\n")
        for r in table:
            f.write(",".join(map(str, r)) + "\n")
    log.info("support + geometry written; GEOMETRY_ANALYSIS.md is written after the val sweep")


if __name__ == "__main__":
    main()
