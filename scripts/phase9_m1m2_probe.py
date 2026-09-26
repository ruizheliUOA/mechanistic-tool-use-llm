"""
phase9_m1m2_probe.py — Phase-9 Part C: compact Llama ca_tc M1/M2 probe.
========================================================================
Implements PHASE9_LLAMA_M1M2_PROTOCOL.md (hash 443bebe4499a8753...) VERBATIM.
Nothing here may deviate from the frozen protocol: obs L22, inj L18, ORIGINAL Phase-8 router
(thr 0.4, unchanged/not refit-for-selection), rho in {1,2,4}, M1 and M2 only, 20 matched-norm
randoms at seed block 4000+k, validation rows only.

FIREWALL: assert_no_test() on every index set. The Llama locked test is permanently
unavailable and is never referenced here.

M1: source->target pairwise DiffMean = unit(mean(correct CA) - mean(correct TC))
M2: shrinkage-LDA target-vs-wrong-destination selector
      positives = correct cannot_answer ; negatives = correct direct U correct request_for_info
      direction = unit(w / sigma), Ledoit-Wolf shrunk LDA
      + logistic-probe weight vector reported alongside as a pre-declared tie-break VARIANT
        (not a third method)

Primary endpoint: Target Gain = own_to_gold - own_to_other_wrong.

Usage: python scripts/phase9_m1m2_probe.py --method M2
       python scripts/phase9_m1m2_probe.py --method M1
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

OUT = L.ROOT / "final" / "results" / "phase9_mechanism_and_ood"
OUT.mkdir(parents=True, exist_ok=True)
STATE = L.CACHE / "phase9_m1m2_state.json"

# ---- FROZEN (protocol §2) ----
CH = "ca_tc"
GOLD, FROM_PRED = "cannot_answer", "tool_call"
ET = f"{GOLD}__{FROM_PRED}"
OBS, INJ = 22, 18
THR = 0.4
RHOS = [1.0, 2.0, 4.0]
N_RANDOM = 20
SEED_BLOCK = 4000            # disjoint from val 2000+k, test 1000+k, reserved OOD 3000+k
RHO_BOUNDARY = 4.0           # grid max here -> "no boundary-only success"
# success thresholds (frozen §4)
TARGET_HIT_MIN = 0.60
REDIST_MAX = 0.40
Z_MIN = 3.0                  # stricter than the gate's 2 (family-wise over 6 cells)
FRAC_GE_MAX = 0.05
BROKE_RATIO = 0.5

CHMAP = {"ca_tc": {"gold": "cannot_answer", "from_pred": "tool_call", "etype": "cannot_answer__tool_call"},
         "ca_direct": {"gold": "cannot_answer", "from_pred": "direct", "etype": "cannot_answer__direct"},
         "rfi_tc": {"gold": "request_for_info", "from_pred": "tool_call", "etype": "request_for_info__tool_call"}}


def unit(v):
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def build_M1(A, tr, meta):
    """source->target pairwise DiffMean (train only)."""
    tgt = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == GOLD]        # correct CA
    src = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == FROM_PRED]   # correct TC
    L.assert_no_test(tgt + src, "M1")
    d = unit(A[tgt].mean(0) - A[src].mean(0))
    return d, {"n_target": len(tgt), "n_source": len(src),
               "raw_norm": float(np.linalg.norm(A[tgt].mean(0) - A[src].mean(0)))}


def build_M2(A, tr, meta):
    """shrinkage-LDA target vs wrong-destinations (+ logistic variant), train only."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pos = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == GOLD]                    # correct CA
    neg = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] in ("direct", "request_for_info")]
    L.assert_no_test(pos + neg, "M2")
    X = A[neg + pos]; y = np.array([0] * len(neg) + [1] * len(pos))
    sc = StandardScaler().fit(X)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(X), y)
    d_lda = unit(lda.coef_[0] / sc.scale_)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                             random_state=42).fit(sc.transform(X), y)
    d_log = unit(clf.coef_[0] / sc.scale_)
    return d_lda, {"n_pos_correct_CA": len(pos), "n_neg_correct_wrongdest": len(neg),
                   "cos_lda_vs_logistic": float(np.dot(d_lda, d_log)),
                   "variant_logistic_available": True}, d_log


def evaluate(model, tok, dev, ds, va, meta, base_pred, routed, u, dn, ch=CH):
    L.assert_no_test(routed, "eval")
    corr = (dn * u).astype(np.float32)
    preds = {i: L.predict(model, tok, dev, ds[i], corr, INJ)[0] for i in routed}
    ip = {i: base_pred[i] for i in va}
    for i in routed:
        ip[i] = preds[i]
    m = L.metrics(va, meta, base_pred, ip, CHMAP)
    own = [i for i in routed if meta[i]["etype"] == ET]
    to_gold = sum(1 for i in own if ip[i] == meta[i]["gold"])
    to_other = sum(1 for i in own if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i])
    moved = m["nt_before"][ch] - m["nt_after"][ch]
    dmg = sum(1 for i in routed if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    tc = sum(1 for i in routed if meta[i]["correct"])
    gc.collect(); torch.cuda.empty_cache()
    return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"], "val_acc": m["acc"],
            "own_before": m["nt_before"][ch], "own_after": m["nt_after"][ch],
            "own_moved": moved, "own_to_gold": to_gold, "own_to_other_wrong": to_other,
            "target_gain": to_gold - to_other,
            "target_hit_rate": (round(to_gold / moved, 4) if moved > 0 else None),
            "redistribution_rate": (round(to_other / moved, 4) if moved > 0 else None),
            "other_after": {k: v for k, v in m["nt_after"].items() if k != ch},
            "n_touched": len(routed), "touched_correct": tc, "damage_on_correct": dmg}


def collateral_ok(real, base_nt):
    for k, v in real["other_after"].items():
        b = base_nt[k]
        if v > max(b + 2, b * 1.25):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["M1", "M2"])
    args = ap.parse_args()

    ds, meta = L.load_meta()
    tr, va = L.splits_train_val()
    L.assert_no_test(tr, "main/tr"); L.assert_no_test(va, "main/va")
    base_pred = {i: meta[i]["pred"] for i in list(tr) + list(va)}
    A_obs = np.load(L.CACHE / f"acts_L{OBS}.npy")
    A_inj = np.load(L.CACHE / f"acts_L{INJ}.npy")
    med_inj = float(np.median(np.linalg.norm(A_inj[tr], axis=1)))

    # ORIGINAL Phase-8 router, unchanged (same construction, same threshold)
    router = L.fit_router(A_obs, tr, va, meta, ET, FROM_PRED)
    vsc = L.rscore(router, A_obs, va)
    routed = [i for i in va if base_pred[i] == FROM_PRED and vsc[i] >= THR]
    L.assert_no_test(routed, "routed")
    base_nt = L.metrics(va, meta, base_pred, base_pred, CHMAP)["nt_before"]

    if args.method == "M1":
        d, info = build_M1(A_obs, tr, meta); d_var = None
    else:
        d, info, d_var = build_M2(A_obs, tr, meta)
    L.log.info("[%s] router AUC=%.4f thr=%.2f routed=%d | med_inj(L%d)=%.4f | info=%s",
               args.method, router["val_auc"], THR, len(routed), INJ, med_inj, info)

    rand_units = [L.random_unit(L.hidden_size(), SEED_BLOCK + k) for k in range(N_RANDOM)]
    model, tok, dev = L.load_model()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    rows = state.get(args.method, {}).get("rows", [])
    done = {r["rho"] for r in rows}

    t0 = time.time()
    for rho in RHOS:
        if rho in done:
            continue
        dn = rho * med_inj
        real = evaluate(model, tok, dev, ds, va, meta, base_pred, routed, d, dn)
        rev = evaluate(model, tok, dev, ds, va, meta, base_pred, routed, -d, dn)
        rnds = [evaluate(model, tok, dev, ds, va, meta, base_pred, routed, ru, dn)
                for ru in rand_units]
        # z on the PRIMARY endpoint (Target Gain) and on Net, both reported
        tg = [r["target_gain"] for r in rnds]; nets = [r["net"] for r in rnds]
        mu_t, sd_t = float(np.mean(tg)), float(np.std(tg))
        mu_n, sd_n = float(np.mean(nets)), float(np.std(nets))
        z_tg = None if sd_t < 1e-6 else (real["target_gain"] - mu_t) / sd_t
        z_net = None if sd_n < 1e-6 else (real["net"] - mu_n) / sd_n
        nge_tg = int(sum(1 for x in tg if x >= real["target_gain"]))
        crit = {
            "target_hit>=0.60": (real["target_hit_rate"] is not None and real["target_hit_rate"] >= TARGET_HIT_MIN),
            "redistribution<=0.40": (real["redistribution_rate"] is not None and real["redistribution_rate"] <= REDIST_MAX),
            "z_targetgain>=3": (z_tg is not None and z_tg >= Z_MIN),
            "nonparam<=0.05": (nge_tg / N_RANDOM) <= FRAC_GE_MAX,
            "real>reverse(target_gain)": real["target_gain"] > rev["target_gain"],
            "broke_controlled": (real["broke"] <= BROKE_RATIO * real["fixed"] if real["fixed"] > 0 else real["broke"] == 0),
            "collateral_ok": collateral_ok(real, base_nt),
        }
        row = {"method": args.method, "channel": CH, "split": "val", "rho": rho,
               "delta_norm": round(dn, 4), "med_inj": round(med_inj, 4), "obs": OBS, "inj": INJ,
               "thr": THR, "router_val_auc": router["val_auc"], "n_routed": len(routed),
               "seed_block": SEED_BLOCK, "direction_info": info,
               "real": real, "reverse": rev,
               "random_target_gain": {"n": N_RANDOM, "values": tg, "mean": round(mu_t, 3),
                                      "std": round(sd_t, 3), "max": int(np.max(tg)),
                                      "n_ge_real": nge_tg},
               "random_net": {"mean": round(mu_n, 3), "std": round(sd_n, 3), "max": int(np.max(nets)),
                              "n_ge_real": int(sum(1 for x in nets if x >= real["net"]))},
               "z_target_gain": (None if z_tg is None else round(z_tg, 3)),
               "z_net": (None if z_net is None else round(z_net, 3)),
               "criteria": crit, "cell_pass": all(crit.values())}
        rows.append(row)
        L.log.info("  [%s] rho=%.1f: TargetGain=%+d (rand %.1f±%.1f, nge=%d/%d, z=%s) | "
                   "hit=%s redist=%s | net=%+d rev_tg=%+d | own %d->%d moved=%d gold=%d wrong=%d | pass=%s",
                   args.method, rho, real["target_gain"], mu_t, sd_t, nge_tg, N_RANDOM,
                   ("NA" if z_tg is None else f"{z_tg:.2f}"),
                   real["target_hit_rate"], real["redistribution_rate"], real["net"],
                   rev["target_gain"], real["own_before"], real["own_after"],
                   real["own_moved"], real["own_to_gold"], real["own_to_other_wrong"],
                   row["cell_pass"])
        state[args.method] = {"rows": rows, "med_inj": med_inj, "router_val_auc": router["val_auc"],
                              "n_routed": len(routed), "direction_info": info}
        STATE.write_text(json.dumps(state, indent=2, default=float))

    # method-level verdict (frozen §4: incl. "no boundary-only success")
    interior_pass = [r for r in rows if r["cell_pass"] and r["rho"] < RHO_BOUNDARY]
    verdict = "PASS" if interior_pass else "FAIL"
    state[args.method]["verdict"] = verdict
    state[args.method]["interior_pass_rhos"] = [r["rho"] for r in interior_pass]
    STATE.write_text(json.dumps(state, indent=2, default=float))
    json.dump(state, open(OUT / "phase9_m1m2_results.json", "w"), indent=2, default=float)
    L.log.info("[%s] VERDICT=%s (interior passes at rho=%s) in %.1f min", args.method, verdict,
               [r["rho"] for r in interior_pass], (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
