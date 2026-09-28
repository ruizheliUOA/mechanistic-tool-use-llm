"""
phase7_validation_specificity.py — VALIDATION-ONLY rho x specificity screen (Phase 7).
=======================================================================================
For each development channel, holds the ARCHIVED validation-selected configuration fixed
(obs layer, inj layer, direction method, router, threshold) and sweeps ONLY the
architecture-normalized perturbation ratio rho. At each rho it evaluates, on the VALIDATION
firing set:

    real direction | reverse direction | N matched-norm random directions (paired across rho)

FIREWALL: train rows are used only to fit the router / estimate the direction / estimate the
activation scale; ALL intervention measurement is on validation rows. assert_no_test() guards
every index set. No test row is ever loaded, scored, or used for any selection.

Magnitude identity:  ||delta_h|| = rho * median_train(||h_inj||),  alpha = rho * med_inj/med_obs
(the archived code multiplies alpha by med_obs, so this reproduces it exactly).

Usage:
  python scripts/phase7_validation_specificity.py --model qwen25_7b
  python scripts/phase7_validation_specificity.py --model mistral7b_v03
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

STATE_DIR = P.CACHE_OUT
RESULTS = P.OUT / "rho_specificity_results.json"

# ── FROZEN Phase-7 configuration (see PHASE7_PROTOCOL.md; do not edit after the run) ──
RHO_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_RANDOM = 20
RANDOM_SEED_BLOCK = 2000          # disjoint from the archived locked-test block (1000+k)

# development channels: archived validation-selected configs (magnitude replaced by rho)
DEV_CHANNELS = {
    "qwen25_7b": {
        "ca_rfi": {"gold": "cannot_answer", "from_pred": "request_for_info",
                   "obs": 20, "inj": 18, "method": "diffmean", "thr": 0.8,
                   "archived_alpha": 2.0, "role": "positive_control"},
        "tc_rfi": {"gold": "tool_call", "from_pred": "request_for_info",
                   "obs": 24, "inj": 24, "method": "diffmean", "thr": 0.4,
                   "archived_alpha": 1.0, "role": "negative_control"},
    },
    "mistral7b_v03": {
        "ca_tc": {"gold": "cannot_answer", "from_pred": "tool_call",
                  "obs": 22, "inj": 18, "method": "diffmean", "thr": 0.8,
                  "archived_alpha": 6.0, "role": "target_discriminator"},
        "rfi_tc": {"gold": "request_for_info", "from_pred": "tool_call",
                   "obs": 22, "inj": 20, "method": "diffmean", "thr": 0.6,
                   "archived_alpha": 6.0, "role": "negative_control"},
        "ca_direct": {"gold": "cannot_answer", "from_pred": "direct",
                      "obs": 22, "inj": 18, "method": "pca1", "thr": 0.4,
                      "archived_alpha": 6.0, "role": "negative_control"},
    },
}


def chan_map(model_key):
    return {ch: {"gold": c["gold"], "from_pred": c["from_pred"],
                 "etype": f"{c['gold']}__{c['from_pred']}"}
            for ch, c in DEV_CHANNELS[model_key].items()}


def evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed, CH,
                       unit, delta_norm, inj, label_order, ch):
    """Score the validation firing set under one direction; return validation metrics."""
    P.assert_no_test(routed, "evaluate_direction/routed")
    corr = (delta_norm * unit).astype(np.float32)
    preds = {}
    for i in routed:
        preds[i] = P.predict(model, tok, dev, ds[i], label_order, corr, inj)[0]
    ip = {i: base_pred[i] for i in va}
    for i in routed:
        ip[i] = preds[i]
    m = P.metrics(va, meta, base_pred, ip, CH)
    own_et = CH[ch]["etype"]
    own_touched = [i for i in routed if meta[i]["etype"] == own_et]
    own_to_gold = sum(1 for i in own_touched if ip[i] == meta[i]["gold"])
    own_to_other_wrong = sum(1 for i in own_touched
                             if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i])
    dmg = sum(1 for i in routed if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    n_touched_correct = sum(1 for i in routed if meta[i]["correct"])
    gc.collect(); torch.cuda.empty_cache()
    return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
            "val_acc": m["acc"], "own_before": m["nt_before"][ch], "own_after": m["nt_after"][ch],
            "other_after": {k: v for k, v in m["nt_after"].items() if k != ch},
            "own_touched": len(own_touched), "own_to_gold": own_to_gold,
            "own_to_other_wrong": own_to_other_wrong,
            "n_touched": len(routed), "touched_correct": n_touched_correct,
            "damage_on_correct": dmg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEV_CHANNELS.keys()))
    args = ap.parse_args()
    mk = args.model
    M = P.MODELS[mk]
    calib = json.loads((P.OUT / "rho_calibration.json").read_text())["models"][mk]["layer_scale"]

    tr, va = P.splits()
    P.assert_no_test(tr, "main/train"); P.assert_no_test(va, "main/val")
    ds = P.load_ds_raw()
    meta = P.load_meta(mk)
    base_pred = {i: meta[i]["pred"] for i in list(tr) + list(va)}
    CH = chan_map(mk)
    state_f = STATE_DIR / f"phase7_state_{mk}.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}

    model, tok, dev = P.load_model(mk)
    label_order = M["label_order"]
    D = M["hidden"]
    # paired random unit vectors: SAME 20 directions reused at every rho (isolates magnitude)
    rand_units = [P.random_unit(D, RANDOM_SEED_BLOCK + k) for k in range(N_RANDOM)]

    for ch, cfg in DEV_CHANNELS[mk].items():
        if ch in state:
            P.log.info("[%s/%s] cached, skipping", mk, ch); continue
        obs, inj = cfg["obs"], cfg["inj"]
        A = np.load(M["cache"] / f"acts_L{obs}.npy")
        med_obs = float(np.median(np.linalg.norm(A[tr], axis=1)))
        med_inj = float(calib[str(inj)]["median_norm_train"])
        router = P.fit_router(A, tr, va, meta, CH[ch]["etype"], cfg["from_pred"])
        vsc = P.rscore(router, A, va)
        routed = [i for i in va if base_pred[i] == cfg["from_pred"] and vsc[i] >= cfg["thr"]]
        P.assert_no_test(routed, f"{ch}/routed")
        eligible = [i for i in va if base_pred[i] == cfg["from_pred"]]
        unit, dm_norm, n_err, n_ref = P.direction(A, tr, meta, CH[ch]["etype"], cfg["gold"],
                                                  cfg["method"])
        archived_rho = cfg["archived_alpha"] * med_obs / med_inj
        P.log.info("[%s/%s] obs L%d inj L%d %s thr%.2f | routed=%d/%d fire=%.3f | "
                   "med_obs=%.3f med_inj=%.3f | archived alpha=%.1f -> rho=%.3f | AUC=%.3f "
                   "n_err=%d n_ref=%d", mk, ch, obs, inj, cfg["method"], cfg["thr"],
                   len(routed), len(eligible), len(routed) / max(len(eligible), 1),
                   med_obs, med_inj, cfg["archived_alpha"], archived_rho,
                   router["val_auc"] or 0, n_err, n_ref)

        rows = []
        t0 = time.time()
        for rho in RHO_GRID:
            delta_norm = rho * med_inj            # ||delta_h||
            alpha_equiv = delta_norm / med_obs    # archived-code alpha that yields this rho
            real = evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed, CH,
                                      unit, delta_norm, inj, label_order, ch)
            rev = evaluate_direction(model, tok, dev, ds, va, meta, base_pred, routed, CH,
                                     -unit, delta_norm, inj, label_order, ch)
            rnds = []
            for k, ru in enumerate(rand_units):
                rnds.append(evaluate_direction(model, tok, dev, ds, va, meta, base_pred,
                                               routed, CH, ru, delta_norm, inj, label_order, ch))
            rnets = [r["net"] for r in rnds]
            mean_r, std_r = float(np.mean(rnets)), float(np.std(rnets))
            zero_var = std_r < 1e-6
            z = (float("nan") if zero_var else (real["net"] - mean_r) / std_r)
            n_ge = int(sum(1 for x in rnets if x >= real["net"]))
            rows.append({
                "model": mk, "channel": ch, "role": cfg["role"], "split": "val",
                "rho": rho, "alpha_equiv": round(alpha_equiv, 4),
                "delta_norm": round(delta_norm, 4),
                "med_obs": round(med_obs, 4), "med_inj": round(med_inj, 4),
                "obs": obs, "inj": inj, "method": cfg["method"], "thr": cfg["thr"],
                "router_val_auc": router["val_auc"], "n_routed": len(routed),
                "n_eligible": len(eligible),
                "fire_rate": round(len(routed) / max(len(eligible), 1), 4),
                "real": real, "reverse": rev,
                "random": {"n": len(rnets), "nets": rnets, "mean": round(mean_r, 3),
                           "std": round(std_r, 3), "max": int(np.max(rnets)),
                           "min": int(np.min(rnets)), "n_ge_real": n_ge,
                           "percentile_of_real": round(100.0 * sum(1 for x in rnets
                                                                   if x < real["net"]) / len(rnets), 1),
                           "zero_variance": bool(zero_var)},
                "spec_z": (None if zero_var else round(z, 3)),
                "seed_block": RANDOM_SEED_BLOCK,
            })
            P.log.info("  [%s] rho=%.2f (a=%.2f, |d|=%.2f): real=%+d rev=%+d rand=%.1f±%.1f "
                       "max=%d nge=%d/%d z=%s own %d->%d dmg=%d",
                       ch, rho, alpha_equiv, delta_norm, real["net"], rev["net"], mean_r, std_r,
                       int(np.max(rnets)), n_ge, len(rnets),
                       ("NA" if zero_var else f"{z:.2f}"),
                       real["own_before"], real["own_after"], real["damage_on_correct"])
            state[ch] = {"rows": rows, "archived_rho": round(archived_rho, 4),
                         "dm_norm": round(dm_norm, 4), "n_err_train": n_err,
                         "n_ref_train": n_ref, "router_val_auc": router["val_auc"]}
            state_f.write_text(json.dumps(state, indent=2, default=float))
        P.log.info("[%s/%s] done in %.1f min", mk, ch, (time.time() - t0) / 60)

    # merge into the shared results file
    allres = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allres[mk] = state
    RESULTS.write_text(json.dumps(allres, indent=2, default=float))
    P.log.info("wrote %s", RESULTS)


if __name__ == "__main__":
    main()
