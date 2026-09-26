"""
phase8_gate_eval.py — Part 3: VALIDATION-ONLY Actionability Gate v2 evaluation.
================================================================================
Applies the FROZEN Gate v2 (Stages 1-5) to every Stage-1-eligible Llama channel using
validation rows only, then writes LLAMA_FROZEN_GATE_DECISIONS.json (+ sha256).

Per channel:
  Stage 3 geometry: R1' (norm-ratio >= 0.05, pick max ratio over the frozen obs grid,
                    tie-break router AUC); R2 (cos(DM,PC1) < 0.6 -> add PCA-1);
                    split-half cosine >= 0.5.
  Stage 2 detectability: router val AUC >= 0.75 and some threshold with val precision >= 0.5.
  Stage 4/5: full 6-rho response curve per (method, inj, thr-locked) with
             real + reverse + 20 matched-norm randoms (seed block 2000+k, paired across rho).
             Interior-rho requirement; z>=2; n_ge/N<=0.05; real>reverse; residual; broke.

NO test row is touched (assert_no_test everywhere). Decisions are frozen BEFORE Stage 6.

Usage: python scripts/phase8_gate_eval.py
"""
from __future__ import annotations
import gc, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

STATE = L.CACHE / "phase8_gate_state.json"
SHORT = {("request_for_info", "tool_call"): "rfi_tc", ("cannot_answer", "tool_call"): "ca_tc",
         ("cannot_answer", "direct"): "ca_direct", ("cannot_answer", "request_for_info"): "ca_rfi",
         ("tool_call", "request_for_info"): "tc_rfi", ("tool_call", "direct"): "tc_direct",
         ("request_for_info", "direct"): "rfi_direct", ("request_for_info", "cannot_answer"): "rfi_ca",
         ("tool_call", "cannot_answer"): "tc_ca"}
ID2GP = {v: k for k, v in SHORT.items()}


def chan_map(cids):
    return {c: {"gold": ID2GP[c][0], "from_pred": ID2GP[c][1],
                "etype": f"{ID2GP[c][0]}__{ID2GP[c][1]}"} for c in cids}


def evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed, CH, unit,
                       delta_norm, inj, ch):
    L.assert_no_test(routed, "eval_dir")
    corr = (delta_norm * unit).astype(np.float32)
    preds = {i: L.predict(model, tok, dev, ds[i], corr, inj)[0] for i in routed}
    ip = {i: base_pred[i] for i in va}
    for i in routed:
        ip[i] = preds[i]
    m = L.metrics(va, meta, base_pred, ip, CH)
    et = CH[ch]["etype"]
    own = [i for i in routed if meta[i]["etype"] == et]
    to_gold = sum(1 for i in own if ip[i] == meta[i]["gold"])
    to_other = sum(1 for i in own if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i])
    dmg = sum(1 for i in routed if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    tc = sum(1 for i in routed if meta[i]["correct"])
    gc.collect(); torch.cuda.empty_cache()
    return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"], "val_acc": m["acc"],
            "own_before": m["nt_before"][ch], "own_after": m["nt_after"][ch],
            "other_after": {k: v for k, v in m["nt_after"].items() if k != ch},
            "own_touched": len(own), "own_to_gold": to_gold, "own_to_other_wrong": to_other,
            "n_touched": len(routed), "touched_correct": tc, "damage_on_correct": dmg}


def stage45_point_pass(row):
    """Frozen Stage-4 + Stage-5 conjunction at one rho (validation)."""
    real, rev, rnd = row["real"], row["reverse"], row["random"]
    z = row["spec_z"]
    crit = {
        "net_pos": real["net"] > 0,
        "residual_down": (real["own_after"] <= L.RESIDUAL_FRAC * real["own_before"]
                          if real["own_before"] > 0 else False),
        "broke_controlled": (real["broke"] <= L.BROKE_RATIO * real["fixed"]
                             if real["fixed"] > 0 else real["broke"] == 0),
        "not_redirection": real["own_to_gold"] > real["own_to_other_wrong"],
        "real_gt_reverse": real["net"] > rev["net"],
        "z_ok": (z is not None and z >= L.Z_MIN),
        "nonparam_ok": (rnd["n_ge_real"] / rnd["n"]) <= L.FRAC_GE_MAX,
        "interior_rho": row["rho"] < L.RHO_MAX,
    }
    if rnd.get("zero_variance"):
        crit["z_ok"] = (rnd["n_ge_real"] == 0 and real["net"] > 0 and real["net"] > rev["net"])
    return crit, all(crit.values())


def main():
    disc = json.loads((L.OUT / "llama_channel_discovery.json").read_text())
    eligible = disc["stage1_eligible_channels"]
    if not eligible:
        L.log.warning("no Stage-1-eligible channels -> verdict D (NO ELIGIBLE CHANNELS)")
    ds, meta = L.load_meta()
    tr, va = L.splits_train_val()
    L.assert_no_test(tr, "main/tr"); L.assert_no_test(va, "main/va")
    base_pred = {i: meta[i]["pred"] for i in list(tr) + list(va)}
    CH = chan_map(eligible)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    acts = {lyr: np.load(L.CACHE / f"acts_L{lyr}.npy") for lyr in
            sorted(set(L.obs_grid()) | {x for o in L.obs_grid() for x in L.inj_candidates(o)})}
    D = L.hidden_size()
    rand_units = [L.random_unit(D, L.VAL_SEED_BLOCK + k) for k in range(L.N_RANDOM)]

    model = tok = dev = None
    geometry = state.get("geometry", {})

    # ---------- Stage 2 + Stage 3 (CPU) ----------
    for ch in eligible:
        if ch in geometry:
            continue
        g, p = ID2GP[ch]; et = CH[ch]["etype"]
        per_layer = {}
        for obs in L.obs_grid():
            A = acts[obs]
            err = [i for i in tr if meta[i]["etype"] == et]
            ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == g]
            if len(err) < 10 or len(ref) < 5:
                continue
            dm, dm_norm = L.diffmean(A, err, ref)
            med = float(np.median(np.linalg.norm(A[tr], axis=1)))
            r = L.fit_router(A, tr, va, meta, et, p)
            per_layer[obs] = {"dm_norm": round(dm_norm, 4), "med_obs": round(med, 4),
                              "norm_ratio": round(dm_norm / (med + 1e-9), 4),
                              "router_val_auc": (round(r["val_auc"], 4) if r and r["val_auc"] else None),
                              "router_pr": (r["pr"] if r else None),
                              "margin_gap": (round(r["margin_gap"], 3) if r and r["margin_gap"] is not None else None),
                              "fp_on_correct": (r["fp_on_correct"] if r else None),
                              "cos_dm_pc1": (lambda c: round(c, 4) if c is not None else None)(
                                  L.cos_dm_pc1(A, tr, meta, et, g)),
                              "splithalf": (lambda s: round(s, 3) if s is not None else None)(
                                  L.splithalf_cos(A, tr, meta, et, g))}
        surv = [o for o, d in per_layer.items() if d["norm_ratio"] >= L.DEGEN_RATIO_FLOOR]
        r1 = max(surv, key=lambda o: (per_layer[o]["norm_ratio"], per_layer[o]["router_val_auc"] or 0)) if surv else None
        methods, inj, stage2, stage3 = [], [], None, None
        if r1 is not None:
            d = per_layer[r1]
            auc_ok = (d["router_val_auc"] or 0) >= L.AUC_FLOOR
            prec_ok = any(v["precision"] >= L.PREC_FLOOR for v in (d["router_pr"] or {}).values())
            stage2 = {"auc": d["router_val_auc"], "auc_ok": bool(auc_ok), "precision_ok": bool(prec_ok),
                      "leaky_router_flag": bool((d["margin_gap"] or 1) < 0.9)}
            sh_ok = (d["splithalf"] is not None and d["splithalf"] >= L.SPLITHALF_FLOOR)
            stage3 = {"R1_obs": r1, "norm_ratio": d["norm_ratio"], "splithalf": d["splithalf"],
                      "splithalf_ok": bool(sh_ok), "cos_dm_pc1": d["cos_dm_pc1"]}
            methods = ["diffmean"] + (["pca1"] if (d["cos_dm_pc1"] is not None
                                                   and d["cos_dm_pc1"] < L.COS_THR) else [])
            inj = L.inj_candidates(r1)
        geometry[ch] = {"per_layer": {str(k): v for k, v in per_layer.items()}, "R1_obs": r1,
                        "stage2": stage2, "stage3": stage3, "methods": methods, "inj_candidates": inj}
        L.log.info("[%s] R1 obs=%s ratio=%s AUC=%s cos=%s splithalf=%s methods=%s inj=%s", ch, r1,
                   (per_layer.get(r1) or {}).get("norm_ratio"), (stage2 or {}).get("auc"),
                   (stage3 or {}).get("cos_dm_pc1"), (stage3 or {}).get("splithalf"), methods, inj)
    state["geometry"] = geometry
    STATE.write_text(json.dumps(state, indent=2, default=float))

    # channels that reach Stage 4 (geometry+detectability OK)
    to_sweep = [c for c in eligible
                if geometry[c]["R1_obs"] is not None
                and geometry[c]["stage2"]["auc_ok"] and geometry[c]["stage2"]["precision_ok"]
                and geometry[c]["stage3"]["splithalf_ok"]]
    L.log.info("channels reaching Stage-4 sweep: %s", to_sweep)

    # ---------- Stage 4/5 sweep (GPU) ----------
    state.setdefault("sweep", {})
    if to_sweep:
        model, tok, dev = L.load_model()
    for ch in to_sweep:
        if ch in state["sweep"]:
            continue
        g, p = ID2GP[ch]; et = CH[ch]["etype"]
        G = geometry[ch]; obs = G["R1_obs"]
        A = acts[obs]
        med_obs = float(np.median(np.linalg.norm(A[tr], axis=1)))
        router = L.fit_router(A, tr, va, meta, et, p)
        vsc = L.rscore(router, A, va)
        # threshold: frozen rule = smallest grid thr with val precision >= 0.5 (Stage-2 floor)
        thr = next((float(t) for t in L.THRESHOLDS
                    if (router["pr"][str(t)]["precision"] >= L.PREC_FLOOR)), None)
        routed = [i for i in va if base_pred[i] == p and vsc[i] >= thr]
        L.assert_no_test(routed, f"{ch}/routed")
        t0 = time.time()

        # ---- Stage 4a: real-only sweep over method x inj x rho (cheap) ----
        scout = []
        for method in G["methods"]:
            unit_m, _, _, _ = L.direction(A, tr, meta, et, g, method)
            for inj in G["inj_candidates"]:
                med_inj = float(np.median(np.linalg.norm(acts[inj][tr], axis=1)))
                for rho in L.RHO_GRID:
                    r = evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed, CH,
                                           unit_m, rho * med_inj, inj, ch)
                    scout.append({"method": method, "inj": inj, "rho": rho,
                                  "med_inj": med_inj, "net": r["net"]})
                    L.log.info("   scout [%s] %s inj%d rho=%.2f -> net=%+d", ch, method, inj, rho, r["net"])
        # frozen selection: the (method,inj) family whose best val Net over rho is highest
        best_family = max({(s["method"], s["inj"]) for s in scout},
                          key=lambda k: max(s["net"] for s in scout
                                            if (s["method"], s["inj"]) == k))
        method, inj = best_family
        med_inj = next(s["med_inj"] for s in scout if (s["method"], s["inj"]) == best_family)
        unit, nrm, ne, nr = L.direction(A, tr, meta, et, g, method)
        L.log.info("[%s] Stage-4 family selected: method=%s inj=L%d (best val Net over rho=%+d)",
                   ch, method, inj, max(s["net"] for s in scout if (s["method"], s["inj"]) == best_family))

        # ---- Stage 4b/5: full 6-rho battery (real + reverse + 20 randoms) on that family ----
        rows = []
        if True:
            if True:
                for rho in L.RHO_GRID:
                    dn = rho * med_inj
                    real = evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed,
                                              CH, unit, dn, inj, ch)
                    rev = evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed,
                                             CH, -unit, dn, inj, ch)
                    rnds = [evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed,
                                               CH, ru, dn, inj, ch) for ru in rand_units]
                    nets = [r["net"] for r in rnds]
                    mu, sd = float(np.mean(nets)), float(np.std(nets))
                    zv = None if sd < 1e-6 else (real["net"] - mu) / sd
                    row = {"channel": ch, "split": "val", "method": method, "obs": obs, "inj": inj,
                           "thr": thr, "rho": rho, "delta_norm": round(dn, 4),
                           "alpha_equiv": round(dn / med_obs, 4), "med_obs": round(med_obs, 4),
                           "med_inj": round(med_inj, 4), "router_val_auc": router["val_auc"],
                           "n_routed": len(routed),
                           "fire_rate": round(len(routed) / max(sum(1 for i in va if base_pred[i] == p), 1), 4),
                           "real": real, "reverse": rev,
                           "random": {"n": len(nets), "nets": nets, "mean": round(mu, 3),
                                      "std": round(sd, 3), "max": int(np.max(nets)),
                                      "min": int(np.min(nets)),
                                      "n_ge_real": int(sum(1 for x in nets if x >= real["net"])),
                                      "percentile_of_real": round(100.0 * sum(1 for x in nets if x < real["net"]) / len(nets), 1),
                                      "zero_variance": bool(sd < 1e-6)},
                           "spec_z": (None if zv is None else round(zv, 3)),
                           "seed_block": L.VAL_SEED_BLOCK}
                    crit, ok = stage45_point_pass(row)
                    row["stage45_pass"] = ok; row["criteria"] = crit
                    rows.append(row)
                    L.log.info("  [%s] %s inj%d rho=%.2f: real=%+d rev=%+d rand=%.1f±%.1f nge=%d/%d z=%s own %d->%d pass=%s",
                               ch, method, inj, rho, real["net"], rev["net"], mu, sd,
                               row["random"]["n_ge_real"], len(nets),
                               ("NA" if zv is None else f"{zv:.2f}"),
                               real["own_before"], real["own_after"], ok)
                    state["sweep"][ch] = {"rows": rows, "thr": thr, "router_val_auc": router["val_auc"],
                                          "scout": scout, "selected_family": {"method": method, "inj": inj}}
                    STATE.write_text(json.dumps(state, indent=2, default=float))
        L.log.info("[%s] swept in %.1f min", ch, (time.time() - t0) / 60)

    # ---------- frozen decisions ----------
    decisions = {}
    for ch in eligible:
        G = geometry[ch]
        d = {"channel": ch, "gold": ID2GP[ch][0], "from_pred": ID2GP[ch][1]}
        if G["R1_obs"] is None:
            d.update(decision="REJECT", reason="stage3_degenerate_geometry_all_layers_below_norm_ratio_floor")
        elif not G["stage2"]["auc_ok"]:
            d.update(decision="REJECT", reason=f"stage2_undetectable_auc={G['stage2']['auc']}<{L.AUC_FLOOR}")
        elif not G["stage2"]["precision_ok"]:
            d.update(decision="REJECT", reason="stage2_no_threshold_with_val_precision>=0.5")
        elif not G["stage3"]["splithalf_ok"]:
            d.update(decision="REJECT", reason=f"stage3_splithalf={G['stage3']['splithalf']}<{L.SPLITHALF_FLOOR}")
        else:
            rows = state["sweep"].get(ch, {}).get("rows", [])
            passing = [r for r in rows if r["stage45_pass"]]
            if not rows:
                d.update(decision="INCONCLUSIVE", reason="sweep_missing")
            elif passing:
                sel = min(passing, key=lambda r: r["rho"])
                d.update(decision="ADMIT",
                         reason=f"stage45_pass at interior rho={sel['rho']} z={sel['spec_z']} "
                                f"net={sel['real']['net']} rev={sel['reverse']['net']} "
                                f"own {sel['real']['own_before']}->{sel['real']['own_after']}",
                         locked_config={k: sel[k] for k in ("method", "obs", "inj", "thr", "rho",
                                                            "delta_norm", "alpha_equiv", "med_inj")})
            else:
                best = max(rows, key=lambda r: r["real"]["net"])
                fails = [k for k, v in best["criteria"].items() if not v]
                nets = [r["real"]["net"] for r in sorted(rows, key=lambda r: r["rho"])]
                shape = ("negative" if max(nets) <= 0 else
                         "flat" if max(abs(n) for n in nets) <= 3 else
                         "boundary-seeking" if nets[-1] == max(nets) else
                         "monotonic non-specific" if nets[-1] > nets[0] else "unstable")
                d.update(decision="REJECT",
                         reason=f"stage45_no_passing_rho; shape={shape}; best net={best['real']['net']} "
                                f"@rho{best['rho']} fails={','.join(fails)}",
                         response_shape=shape,
                         audit_candidate=bool("z_ok" in fails or "nonparam_ok" in fails
                                              or "real_gt_reverse" in fails))
        decisions[ch] = d
        L.log.info("DECISION [%s]: %s — %s", ch, d["decision"], d["reason"])

    out = {"model": L.MODELS[L.MODEL_KEY]["repo"], "revision": L.MODELS[L.MODEL_KEY]["revision"],
           "gate": "Actionability Gate v2 (frozen; GATE_V2_LOCK_MANIFEST.json)",
           "split_used": "validation only (test firewalled)",
           "stage1_eligible": eligible, "geometry": geometry,
           "decisions": decisions,
           "n_admit": sum(1 for d in decisions.values() if d["decision"] == "ADMIT"),
           "n_reject": sum(1 for d in decisions.values() if d["decision"] == "REJECT"),
           "n_inconclusive": sum(1 for d in decisions.values() if d["decision"] == "INCONCLUSIVE")}
    p = L.OUT / "LLAMA_FROZEN_GATE_DECISIONS.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (L.OUT / "LLAMA_FROZEN_GATE_DECISIONS.sha256").write_text(h + "\n")
    # full curves for reporting
    json.dump({c: state["sweep"].get(c, {}) for c in eligible},
              open(L.OUT / "llama_rho_curves.json", "w"), indent=2, default=float)
    L.log.info("FROZEN DECISIONS written; sha256=%s", h)
    print("ADMIT:", [c for c, d in decisions.items() if d["decision"] == "ADMIT"])
    print("REJECT:", [c for c, d in decisions.items() if d["decision"] == "REJECT"])
    print("sha256:", h)


if __name__ == "__main__":
    main()
