"""
phase5_mistral_pilot.py — Phase-5 seed-42 pilot (geometry, val sweep, locked tests, controls, gate).
=====================================================================================================
Two stages (resumable):

  --stage geo : CPU, no model. Per scope-passing channel, per obs layer in OBS_LAYERS,
                compute DiffMean norm, median-norm, norm-ratio, router val-AUC, cos(DM,PC1).
                R1 gate (norm >= DEGEN_NORM) -> pick obs by max norm-ratio (tie-break AUC).
                R2: DiffMean default; add PCA-1 iff cos(DM,PC1) < COS_THR. Write geometry table.

  --stage run : GPU. Per channel: fit router @R1 obs (matched_pred), val sweep
                (method x inj x alpha x thr) -> lock by val Net. Locked-test arms:
                  A baseline, C_i single channel, D auto-cascade (gate passers), F manual diag.
                Controls per selected channel: real / reverse / random x20 / ungated / wrong-layer.
                8-criteria utility gate per channel + composition check.

All tuning on val; each locked-test arm once. Outputs: geometry/, pilot/, placebos/.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L

GEO = L.OUT / "geometry"; GEO.mkdir(parents=True, exist_ok=True)
PIL = L.OUT / "pilot"; PIL.mkdir(parents=True, exist_ok=True)
PLA = L.OUT / "placebos"; PLA.mkdir(parents=True, exist_ok=True)
STATE = PIL / "_state.json"
DISC = L.OUT / "channel_discovery" / "mistral7b_channel_discovery.json"

MANUAL = ["rfi_tc", "ca_tc", "ca_direct"]   # diagnostic Arm F set

# channel_id_short -> (gold, from_pred) reverse map from discovery SHORT table
from phase5_mistral_discovery import SHORT
ID2GP = {v: k for k, v in SHORT.items()}


def chan_def(cid):
    g, p = ID2GP[cid]
    return {"gold": g, "from_pred": p, "ref_gold": g, "etype": f"{g}__{p}"}


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2, default=float))


# ── STAGE geo ────────────────────────────────────────────────────────────────
def stage_geo():
    ds, meta = L.load_meta()
    tr, va, te = L.load_splits()
    acts = {lyr: np.load(L.CACHE_DIR / f"acts_L{lyr}.npy") for lyr in L.OBS_LAYERS}
    disc = json.loads(DISC.read_text())
    scope = disc["scope_filter"]["scope_pass_channels"]
    # include manual channels for diagnostic even if not scope-passing (if they have any support)
    cand_ids = list(dict.fromkeys(scope + [m for m in MANUAL]))

    geom = {}
    for cid in cand_ids:
        d = chan_def(cid)
        et = d["etype"]
        n_tr = sum(1 for i in tr if meta[i]["etype"] == et)
        per_layer = {}
        for lyr in L.OBS_LAYERS:
            A = acts[lyr]
            err = [i for i in tr if meta[i]["etype"] == et]
            ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == d["ref_gold"]]
            if len(err) < 10 or len(ref) < 5:
                per_layer[lyr] = {"insufficient": True, "n_err": len(err), "n_ref": len(ref)}
                continue
            dm, dm_norm = L.diffmean(A, err, ref)
            med = float(np.median(np.linalg.norm(A[tr], axis=1)))
            router = L.fit_router(A, tr, va, meta, et, d["from_pred"], "matched_pred")
            cos = L.cos_dm_pc1(A, tr, meta, et, d["ref_gold"])
            per_layer[lyr] = {"dm_norm": round(dm_norm, 3), "median_norm": round(med, 3),
                              "norm_ratio": round(dm_norm / (med + 1e-9), 4),
                              "router_val_auc": (round(router["val_auc"], 4)
                                                 if router and router["val_auc"] else None),
                              "cos_dm_pc1": (round(cos, 4) if cos is not None else None),
                              "n_err": len(err), "n_ref": len(ref)}
        # R1 (relative, model-agnostic): keep layers with norm_ratio >= DEGEN_RATIO_FLOOR,
        # pick obs by max norm_ratio, tie-break router val-AUC.
        surv = [lyr for lyr in L.OBS_LAYERS
                if per_layer.get(lyr, {}).get("norm_ratio", 0) >= L.DEGEN_RATIO_FLOOR]
        r1 = None
        if surv:
            r1 = max(surv, key=lambda lyr: (per_layer[lyr]["norm_ratio"],
                                            per_layer[lyr]["router_val_auc"] or 0))
        # R2: DiffMean default, add PCA-1 if cos < COS_THR at R1
        methods = ["diffmean"]
        if r1 is not None and per_layer[r1]["cos_dm_pc1"] is not None \
                and per_layer[r1]["cos_dm_pc1"] < L.COS_THR:
            methods.append("pca1")
        inj = sorted({max(0, r1 + o) for o in L.INJ_OFFSETS}) if r1 is not None else []
        geom[cid] = {"etype": et, "gold": d["gold"], "from_pred": d["from_pred"],
                     "n_train_err": n_tr, "per_layer": per_layer, "R1_obs": r1,
                     "R2_method_candidates": methods, "inj_candidates": inj,
                     "scope_pass": cid in scope, "is_manual": cid in MANUAL}
    json.dump({"geometry": geom}, open(GEO / "mistral7b_geometry.json", "w"), indent=2, default=float)
    with open(GEO / "mistral7b_geometry_table.csv", "w") as f:
        f.write("channel,obs_layer,norm_depth,dm_norm,median_norm,norm_ratio,router_val_auc,cos_dm_pc1,n_err,n_ref\n")
        for cid, g in geom.items():
            for lyr, d in g["per_layer"].items():
                if d.get("insufficient"):
                    continue
                f.write(f"{cid},{lyr},{round(lyr/L.N_LAYERS,3)},{d['dm_norm']},{d['median_norm']},"
                        f"{d['norm_ratio']},{d['router_val_auc']},{d['cos_dm_pc1']},{d['n_err']},{d['n_ref']}\n")
    with open(GEO / "MISTRAL7B_GEOMETRY.md", "w") as f:
        f.write("# Mistral-7B — per-channel geometry & R1/R2 selection (seed-42 split)\n\n")
        f.write("| channel | scope | R1 obs (depth) | norm@R1 | ratio@R1 | AUC@R1 | cos(DM,PC1)@R1 | methods | inj |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for cid, g in geom.items():
            r1 = g["R1_obs"]
            pl = g["per_layer"].get(r1, {}) if r1 is not None else {}
            f.write(f"| {cid} | {'Y' if g['scope_pass'] else '(manual)' if g['is_manual'] else 'N'} | "
                    f"{('L%d (%.2f)' % (r1, r1/L.N_LAYERS)) if r1 is not None else 'REJECTED(R1)'} | "
                    f"{pl.get('dm_norm','—')} | {pl.get('norm_ratio','—')} | {pl.get('router_val_auc','—')} | "
                    f"{pl.get('cos_dm_pc1','—')} | {g['R2_method_candidates']} | {g['inj_candidates']} |\n")
    print("geometry done; R1 obs:", {c: geom[c]["R1_obs"] for c in geom})


# ── STAGE run helpers ────────────────────────────────────────────────────────
def apply_routed(idxs, base_pred, routed, preds):
    ip = {i: base_pred[i] for i in idxs}
    for i in routed:
        ip[i] = preds[i]
    return ip


def channels_for_metrics(geom):
    """All discovered/manual channels (for residual reporting)."""
    return {cid: {"gold": g["gold"], "from_pred": g["from_pred"], "etype": g["etype"]}
            for cid, g in geom.items()}


def stage_run():
    ds, meta = L.load_meta()
    tr, va, te = L.load_splits()
    acts = {lyr: np.load(L.CACHE_DIR / f"acts_L{lyr}.npy") for lyr in L.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    geom = json.loads((GEO / "mistral7b_geometry.json").read_text())["geometry"]
    CH = channels_for_metrics(geom)
    st = load_state()

    # channels eligible for the val sweep: scope-passing with an R1 obs; plus manual (diagnostic)
    sweep_ids = [cid for cid, g in geom.items()
                 if g["R1_obs"] is not None and (g["scope_pass"] or g["is_manual"])]
    print("sweep channels:", sweep_ids)

    model, tok, dev = L.load_model(assert_bf16=True)

    def router_at(cid):
        g = geom[cid]; A = acts[g["R1_obs"]]
        return L.fit_router(A, tr, va, meta, g["etype"], g["from_pred"], "matched_pred"), A

    # ---- val sweep ----
    st.setdefault("sweep", {})
    for cid in sweep_ids:
        if cid in st["sweep"]:
            continue
        g = geom[cid]; d = chan_def(cid)
        router, A = router_at(cid)
        vsc = L.rscore(router, A, va)
        firing = [i for i in va if base_pred[i] == g["from_pred"] and vsc[i] >= min(L.THRESHOLDS)]
        rows = []
        t0 = time.time()
        for method in g["R2_method_candidates"]:
            unit, nrm, mn, ne, nr = L.direction(A, tr, meta, g["etype"], d["ref_gold"], method)
            for inj in g["inj_candidates"]:
                for alpha in L.ALPHAS:
                    corr = (alpha * mn * unit).astype(np.float32)
                    preds = {i: L.predict(model, tok, dev, ds[i], corr, inj)[0] for i in firing}
                    gc.collect(); torch.cuda.empty_cache()
                    for thr in L.THRESHOLDS:
                        routed = [i for i in firing if vsc[i] >= thr]
                        ip = apply_routed(va, base_pred, routed, preds)
                        m = L.metrics(va, meta, base_pred, ip, CH)
                        rows.append({"method": method, "obs": g["R1_obs"], "inj": inj,
                                     "alpha": alpha, "thr": thr, "net": m["net"],
                                     "fixed": m["fixed"], "broke": m["broke"],
                                     "ch_fixed": m["chan_fixed"][cid], "touched": len(routed),
                                     "dir_norm": round(nrm, 3)})
        st["sweep"][cid] = {"rows": rows, "router_val_auc": router["val_auc"],
                            "n_firing": len(firing)}
        save_state(st)
        print(f"[{cid}] swept in {(time.time()-t0)/60:.1f} min ({len(rows)} rows)")

    # ---- lock (val only) ----
    st.setdefault("locked", {})
    for cid in sweep_ids:
        if cid in st["locked"]:
            continue
        rows = st["sweep"][cid]["rows"]
        best = max(rows, key=lambda r: (r["net"], -r["broke"], r["ch_fixed"]))
        st["locked"][cid] = {**{k: best[k] for k in ("method", "obs", "inj", "alpha", "thr", "dir_norm")},
                             "val_net": best["net"], "val_fixed": best["fixed"],
                             "val_broke": best["broke"], "val_ch_fixed": best["ch_fixed"],
                             "val_touched": best["touched"],
                             "router_val_auc": st["sweep"][cid]["router_val_auc"]}
    save_state(st)
    json.dump(st["locked"], open(PIL / "locked_configs.json", "w"), indent=2, default=float)
    locked = st["locked"]

    # ---- single-channel locked test (Arm C_i) + record test preds ----
    st.setdefault("arms", {})
    st.setdefault("test_preds", {})
    if "A_baseline" not in st["arms"]:
        m = L.metrics(te, meta, base_pred, base_pred, CH)
        st["arms"]["A_baseline"] = {"arm": "A_baseline", **m, "n_touched": 0,
                                    "transition_matrix": L.transition_matrix(te, meta, base_pred)}
        save_state(st)
    test_routed = {}
    for cid in sweep_ids:
        g = geom[cid]; d = chan_def(cid); c = locked[cid]
        A = acts[c["obs"]]
        router, _ = router_at(cid)
        tsc = L.rscore(router, A, te)
        routed = [i for i in te if base_pred[i] == g["from_pred"] and tsc[i] >= c["thr"]]
        test_routed[cid] = (routed, tsc)
        arm_name = f"C_{cid}"
        if arm_name in st["arms"]:
            continue
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, g["etype"], d["ref_gold"], c["method"])
        corr = (c["alpha"] * mn * unit).astype(np.float32)
        t0 = time.time()
        preds = {i: L.predict(model, tok, dev, ds[i], corr, c["inj"])[0] for i in routed}
        st["test_preds"][cid] = {str(k): v for k, v in preds.items()}
        ip = apply_routed(te, base_pred, routed, preds)
        m = L.metrics(te, meta, base_pred, ip, CH)
        touched = routed
        own_touch = [i for i in touched if meta[i]["etype"] == g["etype"]]
        to_gold = sum(1 for i in own_touch if ip[i] == meta[i]["gold"])
        to_other_wrong = sum(1 for i in own_touch if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i])
        dmg = sum(1 for i in touched if meta[i]["correct"] and ip[i] != meta[i]["gold"])
        st["arms"][arm_name] = {"arm": arm_name, "channel": cid, **m,
                                "n_touched": len(touched),
                                "touched_correct": sum(1 for i in touched if meta[i]["correct"]),
                                "damage_on_correct": dmg,
                                "own_errors_touched": len(own_touch), "own_to_gold": to_gold,
                                "own_to_other_wrong": to_other_wrong,
                                "destination": L.destination_table(te, meta, base_pred, ip, touched),
                                "transition_matrix": L.transition_matrix(te, meta, ip)}
        save_state(st)
        print(f"[C_{cid}] locked test net={m['net']:+d} ({(time.time()-t0)/60:.1f} min)")

    # ---- controls per scope-passing channel (real/reverse/random/ungated/wrong-layer) ----
    st.setdefault("controls", {})
    for cid in sweep_ids:
        if not geom[cid]["scope_pass"] or cid in st["controls"]:
            continue
        g = geom[cid]; d = chan_def(cid); c = locked[cid]
        A = acts[c["obs"]]
        routed, tsc = test_routed[cid]
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, g["etype"], d["ref_gold"], c["method"])
        D = A.shape[1]

        def eval_dir(uvec, inj, ss):
            preds = {i: L.predict(model, tok, dev, ds[i], (c["alpha"] * mn * uvec).astype(np.float32), inj)[0]
                     for i in ss}
            ip = apply_routed(te, base_pred, ss, preds)
            m = L.metrics(te, meta, base_pred, ip, CH)
            gc.collect(); torch.cuda.empty_cache()
            dmg = sum(1 for i in ss if meta[i]["correct"] and ip[i] != meta[i]["gold"])
            return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
                    "n_touched": len(ss), "nt_after_own": m["nt_after"][cid], "damage_on_correct": dmg}

        real_preds = {int(k): v for k, v in st["test_preds"][cid].items()}
        ip_real = apply_routed(te, base_pred, routed, real_preds)
        m_real = L.metrics(te, meta, base_pred, ip_real, CH)
        out = {"real": {"fixed": m_real["fixed"], "broke": m_real["broke"], "net": m_real["net"],
                        "n_touched": len(routed), "nt_after_own": m_real["nt_after"][cid]}}
        out["reverse"] = eval_dir(-unit, c["inj"], routed)
        rnd = []
        for k in range(L.N_RANDOM):
            rng = np.random.RandomState(1000 + k)
            v = rng.randn(D).astype(np.float32); v /= (np.linalg.norm(v) + 1e-12)
            rnd.append(eval_dir(v, c["inj"], routed))
        nets = [r["net"] for r in rnd]
        rn = out["real"]["net"]
        out["random"] = {"n": len(rnd), "nets": nets, "mean": float(np.mean(nets)),
                         "std": float(np.std(nets)), "max": int(np.max(nets)), "min": int(np.min(nets)),
                         "n_ge_real": int(sum(1 for x in nets if x >= rn)),
                         "percentile_of_real": float(100.0 * sum(1 for x in nets if x < rn) / len(nets))}
        out["ungated"] = eval_dir(unit, c["inj"], [i for i in te if base_pred[i] == g["from_pred"]])
        out["wrong_layer"] = eval_dir(unit, L.WRONG_LAYER, routed)
        st["controls"][cid] = out
        save_state(st)
        print(f"[{cid}] controls: real={rn:+d} rev={out['reverse']['net']:+d} "
              f"rand_mean={out['random']['mean']:.1f} pct={out['random']['percentile_of_real']:.0f}")

    # ---- 8-criteria utility gate per scope channel ----
    base_nt = st["arms"]["A_baseline"]["nt_before"]
    gate = {}
    for cid in sweep_ids:
        if not geom[cid]["scope_pass"]:
            continue
        arm = st["arms"][f"C_{cid}"]; ctrl = st["controls"][cid]
        own_b, own_a = arm["nt_before"][cid], arm["nt_after"][cid]
        crit = {
            "1_net_pos": arm["net"] > 0,
            "2_residual_down_25pct": own_a <= 0.75 * own_b if own_b else False,
            "3_broke_controlled": arm["broke"] <= arm["fixed"] / 2 if arm["fixed"] else arm["broke"] == 0,
            "4_other_channels_ok": all(arm["nt_after"][o] <= base_nt[o] + 2 for o in CH if o != cid),
            "5_not_mere_redirection": arm["own_to_gold"] > arm["own_to_other_wrong"],
            "6_real_gt_reverse": ctrl["real"]["net"] > ctrl["reverse"]["net"],
            "7_random_advantage": (ctrl["random"]["percentile_of_real"] >= 75.0
                                   and ctrl["real"]["net"] > ctrl["random"]["mean"] + ctrl["random"]["std"]),
            "8_no_test_tuning": True,
        }
        gate[cid] = {"criteria": crit, "pass": all(crit.values()), "net": arm["net"],
                     "fixed": arm["fixed"], "broke": arm["broke"], "own_residual": [own_b, own_a]}
        print(f"GATE [{cid}] pass={gate[cid]['pass']} {crit}")
    st["gate"] = gate
    save_state(st)

    # ---- Arm D: auto-cascade over gate-passing channels ----
    passers = [cid for cid in gate if gate[cid]["pass"]]
    st["cascade_members"] = passers
    if passers and "D_auto_cascade" not in st["arms"]:
        _run_cascade(st, "D_auto_cascade", passers, ds, meta, tr, va, te, acts,
                     base_pred, geom, locked, CH, model, tok, dev, test_routed)
    # ---- Arm F: manual channels diagnostic (only those with a locked config) ----
    manual_avail = [c for c in MANUAL if c in locked]
    if manual_avail and "F_manual_diag" not in st["arms"]:
        _run_cascade(st, "F_manual_diag", manual_avail, ds, meta, tr, va, te, acts,
                     base_pred, geom, locked, CH, model, tok, dev, test_routed)
    save_state(st)

    _finalize(st, geom, gate, passers, manual_avail)
    print("PILOT DONE. gate:", {c: gate[c]["pass"] for c in gate}, "| cascade:", passers)


def _priority(members, locked):
    # over-call channels (from_pred == tool_call) first, then by val_net desc
    return sorted(members, key=lambda c: (0 if ID2GP[c][1] == "tool_call" else 1,
                                          -locked[c]["val_net"]))


def _run_cascade(st, arm_name, members, ds, meta, tr, va, te, acts, base_pred, geom,
                 locked, CH, model, tok, dev, test_routed):
    order = _priority(members, locked)
    routers = {}
    for cid in members:
        g = geom[cid]; A = acts[g["R1_obs"]]
        routers[cid] = (L.fit_router(A, tr, va, meta, g["etype"], g["from_pred"], "matched_pred"), A)
    tsc = {cid: L.rscore(routers[cid][0], routers[cid][1], te) for cid in members}
    dirs = {}
    for cid in members:
        g = geom[cid]; d = chan_def(cid); c = locked[cid]; A = acts[c["obs"]]
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, g["etype"], d["ref_gold"], c["method"])
        dirs[cid] = (unit, mn)
    ip = dict(base_pred); routed_counts = Counter(); touched = []
    t0 = time.time()
    for i in te:
        chosen = None
        for cid in order:
            g = geom[cid]; c = locked[cid]
            if base_pred[i] == g["from_pred"] and tsc[cid][i] >= c["thr"]:
                chosen = cid; break
        if chosen is None:
            continue
        c = locked[chosen]; unit, mn = dirs[chosen]
        corr = (c["alpha"] * mn * unit).astype(np.float32)
        ip[i] = L.predict(model, tok, dev, ds[i], corr, c["inj"])[0]
        routed_counts[chosen] += 1; touched.append(i)
    m = L.metrics(te, meta, base_pred, ip, CH)
    dmg = sum(1 for i in touched if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    st["arms"][arm_name] = {"arm": arm_name, "members": order, **m,
                            "n_touched": len(touched),
                            "touched_correct": sum(1 for i in touched if meta[i]["correct"]),
                            "damage_on_correct": dmg, "routed": dict(routed_counts),
                            "destination": L.destination_table(te, meta, base_pred, ip, touched),
                            "transition_matrix": L.transition_matrix(te, meta, ip)}
    save_state(st)
    print(f"[{arm_name}] cascade net={m['net']:+d} routed={dict(routed_counts)} ({(time.time()-t0)/60:.1f} min)")


def _finalize(st, geom, gate, passers, manual_avail):
    summary = {"locked": st["locked"], "arms": st["arms"], "gate": gate,
               "cascade_members": passers, "manual_diag_members": manual_avail}
    json.dump(summary, open(PIL / "pilot_summary.json", "w"), indent=2, default=float)
    json.dump({c: st["controls"].get(c) for c in st.get("controls", {})},
              open(PLA / "placebo_all_controls.json", "w"), indent=2, default=float)
    # key table
    with open(PIL / "pilot_key_table.csv", "w") as f:
        f.write("arm,channel_or_members,acc,macro_f1_3cls,fixed,broke,net,n_touched,damage_on_correct\n")
        for name, a in st["arms"].items():
            who = a.get("channel") or ",".join(a.get("members", [])) or "-"
            f.write(f"{name},\"{who}\",{a.get('acc')},{a.get('macro_f1_3cls')},{a.get('fixed','')},"
                    f"{a.get('broke','')},{a.get('net','')},{a.get('n_touched','')},{a.get('damage_on_correct','')}\n")
    # placebo summary
    with open(PLA / "PLACEBO_AND_SPECIFICITY_SUMMARY.md", "w") as f:
        f.write("# Mistral-7B — Placebo & Specificity Controls (seed-42, locked config)\n\n")
        f.write("| channel | real Net | reverse Net | random mean±std (max) | real pct | #rand≥real | ungated Net(broke) | wrong-layer Net |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for cid, o in st.get("controls", {}).items():
            r = o["random"]
            f.write(f"| {cid} | **{o['real']['net']:+d}** | {o['reverse']['net']:+d} | "
                    f"{r['mean']:.1f}±{r['std']:.1f} ({r['max']}) | {r['percentile_of_real']:.0f} | "
                    f"{r['n_ge_real']} | {o['ungated']['net']:+d}({o['ungated']['broke']}) | "
                    f"{o['wrong_layer']['net']:+d} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["geo", "run"])
    args = ap.parse_args()
    if args.stage == "geo":
        stage_geo()
    else:
        stage_run()


if __name__ == "__main__":
    main()
