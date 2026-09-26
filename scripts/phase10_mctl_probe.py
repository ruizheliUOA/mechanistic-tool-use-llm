"""
phase10_mctl_probe.py — Phase-10 Margin-Conditioned Transition Logging (MCTL).
==============================================================================
Implements PHASE10_MCTL_PREREGISTRATION.md (sha256 fe292f97...) VERBATIM.
Adjudication rules: PHASE10_MCTL_ADJUDICATION_ADDENDUM.md (sha256 61734192...).

FROZEN (nothing here may deviate):
  channel ca_tc | obs L22 | inj L18 (MLP-out, all positions) | Phase-8 router thr 0.4 (reloaded)
  routed n=74 | med_inj 4.6157 | directions: deployed DiffMean + M1 ONLY (M2 excluded, split-half
  0.144 < 0.5) | rho {1,2,3,4} | controls real/reverse/zero + 20 matched-norm randoms per rho
  (seed block 5000+k, SHARED across directions) | validation only, assert_no_test everywhere.

VALIDITY GATE (addendum §2): stored deployed/M1 aggregates at rho {1,2,4} must reproduce EXACTLY
(net, own_to_gold, own_to_other_wrong, own_moved) + routed n / med_inj / router AUC, and the zero
control must reproduce the stored baseline. Any mismatch -> VOID, stop, change nothing.

PER-SAMPLE LOGGING (the whole point): for every routed sample x every cell we store all four
candidate avg_logp, argmax pred, transition class, post-intervention margins, plus the FROZEN
baseline margins / router score / runner-up / gold-adjacency computed once.

Usage: python scripts/phase10_mctl_probe.py            # runs gate then all cells (resumable)
       python scripts/phase10_mctl_probe.py --gate-only
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L
import phase9_m1m2_probe as P9          # reuse build_M1 VERBATIM (frozen estimator)

OUT = L.ROOT / "final" / "results" / "phase10_mctl"
OUT.mkdir(parents=True, exist_ok=True)
STATE = L.CACHE / "phase10_mctl_state.json"

# ---- FROZEN CONFIG (prereg §2) ----
CH, GOLD, FROM_PRED = "ca_tc", "cannot_answer", "tool_call"
ET = f"{GOLD}__{FROM_PRED}"
OBS, INJ, THR = 22, 18, 0.4
RHOS = [1.0, 2.0, 3.0, 4.0]
N_RANDOM = 20
SEED_BLOCK = 5000                      # disjoint from val 2000+k, test 1000+k, OOD 3000+k, P9 4000+k
WRONG_DESTS = ("direct", "request_for_info")
CHMAP = P9.CHMAP

# ---- VALIDITY GATE expected values (addendum §2; from stored Phase-8/9 artifacts) ----
EXPECT = {
    ("deployed", 1.0): {"net": 0, "own_to_gold": 0, "own_to_other_wrong": 3},
    ("deployed", 2.0): {"net": 0, "own_to_gold": 0, "own_to_other_wrong": 2},
    ("deployed", 4.0): {"net": 8, "own_to_gold": 10, "own_to_other_wrong": 25},
    ("M1", 1.0): {"net": 2, "own_to_gold": 2, "own_to_other_wrong": 3, "own_moved": 5},
    ("M1", 2.0): {"net": 0, "own_to_gold": 0, "own_to_other_wrong": 2, "own_moved": 2},
    ("M1", 4.0): {"net": 6, "own_to_gold": 9, "own_to_other_wrong": 5, "own_moved": 14},
}
EXPECT_ROUTED, EXPECT_MED_INJ, EXPECT_AUC = 74, 4.6157, 0.9585


def unit(v):
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def build_deployed(A, tr, meta):
    """The deployed Phase-8 DiffMean: unit(mean(correct CA) - mean(ca_tc errors)). Train only."""
    err = [i for i in tr if meta[i]["etype"] == ET]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == GOLD]
    L.assert_no_test(err + ref, "deployed")
    d, n = L.diffmean(A, err, ref)
    return d, {"n_err": len(err), "n_ref": len(ref), "raw_norm": float(n)}


def score_all(model, tok, dev, ds, idxs, corr, inj):
    """Per-sample 4-class avg_logp + argmax. corr=None -> clean."""
    out = {}
    for i in idxs:
        pred, sc = L.predict(model, tok, dev, ds[i], corr, inj)
        out[i] = {"pred": pred, "logp": {k: round(float(v), 5) for k, v in sc.items()}}
    gc.collect(); torch.cuda.empty_cache()
    return out


def cell_metrics(va, meta, base_pred, per_sample, routed):
    """Aggregate exactly as Phase-8/9 did (so the gate can compare), from per-sample preds."""
    ip = {i: base_pred[i] for i in va}
    for i in routed:
        ip[i] = per_sample[i]["pred"]
    m = L.metrics(va, meta, base_pred, ip, CHMAP)
    own = [i for i in routed if meta[i]["etype"] == ET]
    to_gold = sum(1 for i in own if ip[i] == meta[i]["gold"])
    to_other = sum(1 for i in own if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i])
    moved = m["nt_before"][CH] - m["nt_after"][CH]
    dmg = sum(1 for i in routed if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"], "val_acc": m["acc"],
            "own_before": m["nt_before"][CH], "own_after": m["nt_after"][CH],
            "own_moved": moved, "own_to_gold": to_gold, "own_to_other_wrong": to_other,
            "target_gain": to_gold - to_other,
            "target_hit_rate": (round(to_gold / moved, 4) if moved > 0 else None),
            "redistribution_rate": (round(to_other / moved, 4) if moved > 0 else None),
            "other_after": {k: v for k, v in m["nt_after"].items() if k != CH},
            "damage_on_correct": dmg}


def transition_class(i, meta, base_pred, pred):
    if pred == base_pred[i]:
        return "stay"
    if pred == meta[i]["gold"]:
        return "to_gold"
    return f"to_wrong:{pred}"


def margins(logp, gold):
    """(top - gold) and (gold - best wrong-destination)."""
    top = max(logp.values())
    best_wrong = max(logp[w] for w in WRONG_DESTS)
    return round(top - logp[gold], 5), round(logp[gold] - best_wrong, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    ds, meta = L.load_meta()
    tr, va = L.splits_train_val()
    L.assert_no_test(tr, "p10/tr"); L.assert_no_test(va, "p10/va")
    base_pred = {i: meta[i]["pred"] for i in list(tr) + list(va)}
    A_obs = np.load(L.CACHE / f"acts_L{OBS}.npy")
    A_inj = np.load(L.CACHE / f"acts_L{INJ}.npy")
    med_inj = float(np.median(np.linalg.norm(A_inj[tr], axis=1)))

    router = L.fit_router(A_obs, tr, va, meta, ET, FROM_PRED)
    vsc = L.rscore(router, A_obs, va)
    routed = [i for i in va if base_pred[i] == FROM_PRED and vsc[i] >= THR]
    L.assert_no_test(routed, "p10/routed")

    # ---- config gate (before any GPU) ----
    cfg_fail = []
    if len(routed) != EXPECT_ROUTED: cfg_fail.append(f"routed {len(routed)} != {EXPECT_ROUTED}")
    if round(med_inj, 4) != EXPECT_MED_INJ: cfg_fail.append(f"med_inj {med_inj:.4f} != {EXPECT_MED_INJ}")
    if round(router["val_auc"], 4) != EXPECT_AUC: cfg_fail.append(f"AUC {router['val_auc']:.4f} != {EXPECT_AUC}")
    if cfg_fail:
        L.log.error("VOID (config gate): %s", "; ".join(cfg_fail))
        json.dump({"verdict": "VOID", "stage": "config_gate", "failures": cfg_fail},
                  open(OUT / "phase10_mctl_VOID.json", "w"), indent=2)
        sys.exit(2)
    L.log.info("config gate OK: routed=%d med_inj=%.4f AUC=%.4f", len(routed), med_inj, router["val_auc"])

    dirs = {}
    d_dep, i_dep = build_deployed(A_obs, tr, meta); dirs["deployed"] = (d_dep, i_dep)
    d_m1, i_m1 = P9.build_M1(A_obs, tr, meta);      dirs["M1"] = (unit(d_m1), i_m1)
    L.log.info("directions: deployed %s | M1 %s | M2 EXCLUDED (split-half 0.144 < 0.5)", i_dep, i_m1)

    rand_units = {r: [L.random_unit(L.hidden_size(), SEED_BLOCK + k) for k in range(N_RANDOM)]
                  for r in RHOS}

    model, tok, dev = L.load_model()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"cells": {}}

    def run_cell(key, corr, inj):
        if key in state["cells"]:
            return state["cells"][key]
        ps = score_all(model, tok, dev, ds, routed, corr, inj)
        agg = cell_metrics(va, meta, base_pred, ps, routed)
        rec = {"per_sample": {str(i): ps[i] for i in routed}, "agg": agg}
        state["cells"][key] = rec
        STATE.write_text(json.dumps(state, default=float))
        return rec

    t0 = time.time()

    # ---------- ZERO control (rho=0, no injection) : must reproduce baseline ----------
    z = run_cell("zero", None, None)
    zero_mismatch = [i for i in routed if z["per_sample"][str(i)]["pred"] != base_pred[i]]
    if zero_mismatch:
        L.log.error("VOID (zero control): %d/%d routed preds differ from stored baseline",
                    len(zero_mismatch), len(routed))
        json.dump({"verdict": "VOID", "stage": "zero_control",
                   "n_mismatch": len(zero_mismatch), "idx": zero_mismatch[:20]},
                  open(OUT / "phase10_mctl_VOID.json", "w"), indent=2)
        sys.exit(2)
    L.log.info("zero control OK: all %d routed preds reproduce the stored baseline", len(routed))

    # ---------- VALIDITY GATE: real cells at rho {1,2,4} for both directions ----------
    gate_rows, gate_fail = [], []
    for dname in ("deployed", "M1"):
        d, _ = dirs[dname]
        for rho in (1.0, 2.0, 4.0):
            r = run_cell(f"{dname}|real|{rho}", (rho * med_inj * d).astype(np.float32), INJ)
            a, e = r["agg"], EXPECT[(dname, rho)]
            diffs = {k: (a[k], v) for k, v in e.items() if a[k] != v}
            gate_rows.append({"direction": dname, "rho": rho,
                              "got": {k: a[k] for k in e}, "expected": e, "match": not diffs})
            if diffs:
                gate_fail.append(f"{dname} rho={rho}: " +
                                 ", ".join(f"{k} got {g} expected {x}" for k, (g, x) in diffs.items()))
            L.log.info("  gate %s rho=%.0f: net=%+d gold=%d wrong=%d moved=%d -> %s",
                       dname, rho, a["net"], a["own_to_gold"], a["own_to_other_wrong"],
                       a["own_moved"], "MATCH" if not diffs else "MISMATCH")
    if gate_fail:
        L.log.error("VOID (validity gate): %s", " | ".join(gate_fail))
        json.dump({"verdict": "VOID", "stage": "validity_gate", "failures": gate_fail,
                   "rows": gate_rows}, open(OUT / "phase10_mctl_VOID.json", "w"), indent=2)
        sys.exit(2)
    L.log.info("VALIDITY GATE PASSED: 6/6 stored aggregates reproduced exactly")
    json.dump({"passed": True, "rows": gate_rows}, open(OUT / "phase10_validity_gate.json", "w"),
              indent=2, default=float)
    if args.gate_only:
        L.log.info("--gate-only: stopping after gate (%.1f min)", (time.time() - t0) / 60)
        return

    # ---------- remaining cells: real@rho3, reverse@all rho, randoms@all rho ----------
    for dname in ("deployed", "M1"):
        d, _ = dirs[dname]
        for rho in RHOS:
            run_cell(f"{dname}|real|{rho}", (rho * med_inj * d).astype(np.float32), INJ)
            run_cell(f"{dname}|reverse|{rho}", (rho * med_inj * -d).astype(np.float32), INJ)
            L.log.info("  %s rho=%.0f real/reverse done (%.1f min)", dname, rho, (time.time() - t0) / 60)
    for rho in RHOS:   # randoms are SHARED across directions (depend only on rho)
        for k, ru in enumerate(rand_units[rho]):
            run_cell(f"random{k}|{rho}", (rho * med_inj * ru).astype(np.float32), INJ)
        L.log.info("  randoms rho=%.0f done (%d) (%.1f min)", rho, N_RANDOM, (time.time() - t0) / 60)

    # ---------- frozen baseline features (computed once, from stored baseline only) ----------
    det = {d["uuid"]: d for d in (json.loads(l) for l in open(L.BASE_DETAILS))}
    frozen = {}
    for i in routed:
        lp = det[meta[i]["uuid"]]["avg_logp"]
        order = sorted(lp.items(), key=lambda kv: -kv[1])
        gap_gold, gap_bw = margins(lp, GOLD)
        frozen[str(i)] = {
            "idx": i, "uuid": meta[i]["uuid"], "gold": meta[i]["gold"],
            "baseline_pred": base_pred[i], "is_own_channel_err": meta[i]["etype"] == ET,
            "router_score": round(float(vsc[i]), 4),
            "runner_up": order[1][0], "gold_adjacent": order[1][0] == GOLD,
            "gold_rank": [k for k, _ in order].index(GOLD) + 1,
            "baseline_gap_top_minus_gold": gap_gold,
            "baseline_gap_gold_minus_bestwrong": gap_bw,
            "baseline_logp": {k: round(float(v), 5) for k, v in lp.items()},
        }

    out = {"config": {"channel": CH, "obs": OBS, "inj": INJ, "thr": THR, "rhos": RHOS,
                      "n_routed": len(routed), "med_inj": round(med_inj, 4),
                      "router_val_auc": round(router["val_auc"], 4),
                      "seed_block": SEED_BLOCK, "n_random": N_RANDOM,
                      "directions": {k: v[1] for k, v in dirs.items()},
                      "M2": "EXCLUDED (split-half 0.144 < 0.5)"},
           "validity_gate": {"passed": True, "rows": gate_rows},
           "frozen_baseline": frozen,
           "cells": state["cells"]}
    (OUT / "phase10_mctl_raw.json").write_text(json.dumps(out, default=float))
    L.log.info("DONE in %.1f min | cells=%d | wrote %s",
               (time.time() - t0) / 60, len(state["cells"]), OUT / "phase10_mctl_raw.json")


if __name__ == "__main__":
    main()
