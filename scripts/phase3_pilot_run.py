"""
phase3_pilot_run.py — Phase 3 seed-42 pilot: val sweep -> lock -> locked tests -> controls -> gate.
====================================================================================================
Implements PHASE3_PROTOCOL.md §6–§11 for the two new channels (ca_rfi, tc_rfi).
All tuning on validation only; each locked test runs once; controls at the locked config.

Outputs:
  geometry/  val-sweep rows appended to layer_direction_router_table/summary + GEOMETRY_ANALYSIS.md
  pilot/     locked_configs.json (written BEFORE test), pilot_summary.json, pilot_key_table.csv,
             pilot_details.jsonl, PHASE3_NEW_CHANNEL_PILOT_SUMMARY.md
  placebos/  random/reverse/ungated (+wrong-layer) results + PLACEBO_AND_SPECIFICITY_SUMMARY.md
"""
from __future__ import annotations
import gc, json, logging, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L
import qwen7b_native_sakiko as P

log = logging.getLogger("p3.pilot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

GEO = L.OUT / "geometry"
PIL = L.OUT / "pilot"; PIL.mkdir(parents=True, exist_ok=True)
PLA = L.OUT / "placebos"; PLA.mkdir(parents=True, exist_ok=True)
STATE = PIL / "_state.json"          # resumable intermediate state (small, committable-safe)


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2, default=float))


def apply_channel(idxs, meta, base_pred, routed, preds):
    ip = {i: base_pred[i] for i in idxs}
    for i in routed:
        ip[i] = preds[i]
    return ip


def run_arm(te, meta, base_pred, routed, preds, ch, name):
    ip = apply_channel(te, meta, base_pred, routed, preds)
    m = L.metrics5(te, meta, base_pred, ip)
    touched = list(routed)
    dmg = sum(1 for i in touched if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    n_corr_touched = sum(1 for i in touched if meta[i]["correct"])
    dest = L.destination_table(te, meta, base_pred, ip, touched)
    et = L.NEW_CHANNELS[ch]["etype"] if ch in L.NEW_CHANNELS else ch
    own_touch = [i for i in touched if meta[i]["etype"] == et]
    to_gold = sum(1 for i in own_touch if ip[i] == meta[i]["gold"])
    to_other_wrong = sum(1 for i in own_touch
                         if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i])
    return {"arm": name, "channel": ch, **m, "n_touched": len(touched),
            "touched_correct": n_corr_touched, "damage_on_correct": dmg,
            "own_errors_touched": len(own_touch), "own_to_gold": to_gold,
            "own_to_other_wrong": to_other_wrong,
            "destination_redistribution": dest,
            "transition_matrix_after": L.transition_matrix(te, meta, ip)}, ip


def main():
    ds, meta = L.load_meta_and_dataset()
    tr, va, te = L.load_splits()
    acts = {Lx: np.load(L.CACHE_DIR / f"acts_L{Lx}.npy") for Lx in L.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    geom = json.loads((GEO / "layer_direction_router_summary.json").read_text())["geometry"]
    support = json.loads((L.OUT / "channel_support" /
                          "channel_support_summary.json").read_text())
    st = load_state()

    eligible = [ch for ch in L.NEW_CHANNELS
                if support[ch]["scope_filter_counts_pass"] and geom[ch]["R1_obs"] is not None]
    log.info("eligible channels: %s", eligible)
    if not eligible:
        log.warning("no eligible channel — NO-GO; writing empty pilot summary")
    model, tok, dev = P.load_model()

    # ---------- val sweep (per channel, R1 layer, R2 methods, inj x alpha x thr) ----------
    if "sweep" not in st:
        st["sweep"] = {}
    for ch in eligible:
        if ch in st["sweep"]:
            continue
        cfg = L.NEW_CHANNELS[ch]
        r1 = geom[ch]["R1_obs"]; A = acts[r1]
        router = L.fit_router(A, tr, va, meta, cfg["etype"], cfg["from_pred"], "matched_pred")
        vsc = L.rscore(router, A, va)
        elig = [i for i in va if base_pred[i] == cfg["from_pred"]]
        firing = [i for i in elig if vsc[i] >= min(L.THRESHOLDS)]
        log.info("[%s] obs=L%d val gate-eligible=%d firing>=%.1f: %d",
                 ch, r1, len(elig), min(L.THRESHOLDS), len(firing))
        rows = []
        t0 = time.time()
        for method in geom[ch]["R2_method_candidates"]:
            unit, nrm, mn, ne, nr = L.direction(A, tr, meta, cfg["etype"],
                                                cfg["ref_gold"], method)
            for inj in geom[ch]["inj_candidates"]:
                for alpha in L.ALPHAS:
                    corr = (alpha * mn * unit).astype(np.float32)
                    preds = {i: P.predict_hooked(model, tok, dev, ds[i], corr, inj)
                             for i in firing}
                    gc.collect(); torch.cuda.empty_cache()
                    for thr in L.THRESHOLDS:
                        routed = [i for i in firing if vsc[i] >= thr]
                        ip = apply_channel(va, meta, base_pred, routed, preds)
                        m = L.metrics5(va, meta, base_pred, ip)
                        rows.append({"method": method, "obs": r1, "inj": inj,
                                     "alpha": alpha, "thr": thr, "net": m["net"],
                                     "fixed": m["fixed"], "broke": m["broke"],
                                     "ch_fixed": m["chan_fixed"][ch],
                                     "nt_after_own": m["nt_after"][ch],
                                     "touched": len(routed), "dir_norm": round(nrm, 3)})
            log.info("[%s] method=%s swept in %.1f min", ch, method, (time.time() - t0) / 60)
        st["sweep"][ch] = {"rows": rows, "router_val_auc": router["val_auc"],
                           "n_firing": len(firing), "n_eligible": len(elig)}
        save_state(st)

    # ---------- lock (val only) ----------
    if "locked" not in st:
        locked = {}
        for ch in eligible:
            rows = st["sweep"][ch]["rows"]
            best = max(rows, key=lambda r: (r["net"], -r["broke"], r["ch_fixed"]))
            locked[ch] = {**{k: best[k] for k in
                             ("method", "obs", "inj", "alpha", "thr", "dir_norm")},
                          "val_net": best["net"], "val_fixed": best["fixed"],
                          "val_broke": best["broke"], "val_ch_fixed": best["ch_fixed"],
                          "val_touched": best["touched"],
                          "router_val_auc": st["sweep"][ch]["router_val_auc"]}
            log.info("LOCK [%s]: %s", ch, locked[ch])
        st["locked"] = locked
        save_state(st)
        json.dump(locked, open(PIL / "locked_configs.json", "w"), indent=2, default=float)
    locked = st["locked"]

    # ---------- locked tests (once per arm) ----------
    if "arms" not in st:
        st["arms"] = {}
    if "A_baseline" not in st["arms"]:
        m = L.metrics5(te, meta, base_pred, base_pred)
        st["arms"]["A_baseline"] = {"arm": "A_baseline", "channel": None, **m,
                                    "n_touched": 0,
                                    "transition_matrix_after":
                                        L.transition_matrix(te, meta, base_pred)}
        save_state(st)
    test_routed, test_preds_real = {}, {}
    for ch, arm_name in [("ca_rfi", "B_ca_rfi_only"), ("tc_rfi", "C_tc_rfi_only")]:
        if ch not in eligible:
            st["arms"][arm_name] = {"arm": arm_name, "channel": ch, "status": "NOT_RUN",
                                    "reason": "failed scope filter / R1"}
            save_state(st); continue
        c = locked[ch]; cfg = L.NEW_CHANNELS[ch]
        A = acts[c["obs"]]
        router = L.fit_router(A, tr, va, meta, cfg["etype"], cfg["from_pred"], "matched_pred")
        tsc = L.rscore(router, A, te)
        routed = [i for i in te if base_pred[i] == cfg["from_pred"] and tsc[i] >= c["thr"]]
        test_routed[ch] = (routed, tsc, router)
        if arm_name in st["arms"]:
            test_preds_real[ch] = {int(k): v for k, v in
                                   st.get("test_preds", {}).get(ch, {}).items()}
            continue
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, cfg["etype"], cfg["ref_gold"],
                                          c["method"])
        corr = (c["alpha"] * mn * unit).astype(np.float32)
        t0 = time.time()
        preds = {i: P.predict_hooked(model, tok, dev, ds[i], corr, c["inj"]) for i in routed}
        log.info("[%s] locked test: %d routed, %.1f min", ch, len(routed),
                 (time.time() - t0) / 60)
        res, ip = run_arm(te, meta, base_pred, routed, preds, ch, arm_name)
        st["arms"][arm_name] = res
        st.setdefault("test_preds", {})[ch] = {str(k): v for k, v in preds.items()}
        test_preds_real[ch] = preds
        save_state(st)
        gc.collect(); torch.cuda.empty_cache()

    # ---------- controls (per eligible channel at locked config) ----------
    if "controls" not in st:
        st["controls"] = {}
    for ch in eligible:
        if ch in st["controls"]:
            continue
        c = locked[ch]; cfg = L.NEW_CHANNELS[ch]
        A = acts[c["obs"]]
        routed, tsc, router = test_routed[ch]
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, cfg["etype"], cfg["ref_gold"],
                                          c["method"])
        D = A.shape[1]
        out = {}
        real_res, _ = run_arm(te, meta, base_pred, routed, test_preds_real[ch], ch, "real")
        out["real"] = {k: real_res[k] for k in ("fixed", "broke", "net", "n_touched")}
        out["real"]["nt_after_own"] = real_res["nt_after"][ch]

        def eval_dir(unit_vec, inj, tag, sample_set=None):
            ss = routed if sample_set is None else sample_set
            corr = (c["alpha"] * mn * unit_vec).astype(np.float32)
            preds = {i: P.predict_hooked(model, tok, dev, ds[i], corr, inj) for i in ss}
            res, _ = run_arm(te, meta, base_pred, ss, preds, ch, tag)
            gc.collect(); torch.cuda.empty_cache()
            return {"fixed": res["fixed"], "broke": res["broke"], "net": res["net"],
                    "n_touched": res["n_touched"], "nt_after_own": res["nt_after"][ch],
                    "damage_on_correct": res["damage_on_correct"]}

        t0 = time.time()
        out["reverse"] = eval_dir(-unit, c["inj"], "reverse")
        log.info("[%s] reverse: net=%+d (%.1f min)", ch, out["reverse"]["net"],
                 (time.time() - t0) / 60)
        rnd = []
        for k in range(L.N_RANDOM):
            rng = np.random.RandomState(1000 + k)
            v = rng.randn(D).astype(np.float32); v /= (np.linalg.norm(v) + 1e-12)
            rnd.append(eval_dir(v, c["inj"], f"random_{k}"))
            log.info("[%s] random %d/%d: net=%+d", ch, k + 1, L.N_RANDOM, rnd[-1]["net"])
        nets = [r["net"] for r in rnd]
        real_net = out["real"]["net"]
        out["random"] = {"n": len(rnd), "nets": nets,
                         "mean": float(np.mean(nets)), "std": float(np.std(nets)),
                         "max": int(np.max(nets)), "min": int(np.min(nets)),
                         "n_ge_real": int(sum(1 for x in nets if x >= real_net)),
                         "percentile_of_real":
                             float(100.0 * sum(1 for x in nets if x < real_net) / len(nets))}
        ungated_set = [i for i in te if base_pred[i] == cfg["from_pred"]]
        out["ungated"] = eval_dir(unit, c["inj"], "ungated", sample_set=ungated_set)
        out["wrong_layer"] = eval_dir(unit, L.WRONG_LAYER, "wrong_layer")
        st["controls"][ch] = out
        save_state(st)

    # ---------- pilot gate ----------
    gate = {}
    base_nt = st["arms"]["A_baseline"]["nt_before"]
    for ch in L.NEW_CHANNELS:
        arm_name = "B_ca_rfi_only" if ch == "ca_rfi" else "C_tc_rfi_only"
        arm = st["arms"].get(arm_name, {})
        if ch not in eligible or arm.get("status") == "NOT_RUN":
            gate[ch] = {"eligible": False, "pass": False, "reason": "NOT_RUN"}
            continue
        ctrl = st["controls"][ch]
        own_b, own_a = arm["nt_before"][ch], arm["nt_after"][ch]
        crit = {
            "1_net_pos": arm["net"] > 0,
            "2_residual_down_25pct": own_a <= 0.75 * own_b,
            "3_broke_controlled": arm["broke"] <= arm["fixed"] / 2,
            "4_archived_channels_ok": all(
                arm["nt_after"][o] <= base_nt[o] + 2
                for o in ("rfi_tc", "ca_tc", "ca_direct")),
            "5_not_mere_redirection": arm["own_to_gold"] > arm["own_to_other_wrong"],
            "6_real_gt_reverse": ctrl["real"]["net"] > ctrl["reverse"]["net"],
            "7_random_advantage": (ctrl["random"]["percentile_of_real"] >= 75.0 and
                                   ctrl["real"]["net"] > ctrl["random"]["mean"]
                                   + ctrl["random"]["std"]),
            "8_no_test_tuning": True,
        }
        gate[ch] = {"eligible": True, "criteria": crit, "pass": all(crit.values()),
                    "own_residual": [own_b, own_a],
                    "net": arm["net"], "fixed": arm["fixed"], "broke": arm["broke"]}
        log.info("GATE [%s]: pass=%s %s", ch, gate[ch]["pass"], crit)
    st["gate"] = gate
    save_state(st)
    json.dump({"locked": locked, "arms": {k: v for k, v in st["arms"].items()},
               "gate": gate},
              open(PIL / "pilot_summary.json", "w"), indent=2, default=float)
    json.dump({ch: st["controls"].get(ch) for ch in eligible},
              open(PLA / "placebo_all_controls.json", "w"), indent=2, default=float)
    log.info("pilot complete; gate: %s",
             {ch: gate[ch].get("pass") for ch in gate})


if __name__ == "__main__":
    main()
