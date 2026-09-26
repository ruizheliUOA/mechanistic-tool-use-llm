"""
phase8_locked_test.py — Part 4: ONE-SHOT locked-test confirmation + bounded rejection audit.
=============================================================================================
Runs ONLY after LLAMA_FROZEN_GATE_DECISIONS.json exists and is hashed. Refuses to run
otherwise. Executes the frozen Stage-6 battery exactly once per arm:

  admitted channel  -> real / reverse / 20 matched-norm randoms (seed block 1000+k)
                       / ungated / wrong-layer (~0.15L)
  rejected-for-specificity channels (<=2, bounded audit) -> the same battery ONCE at their
                       best Stage-4 config, solely to SCORE the gate (never fed back).

Confirmation rule (frozen): z>=2 AND n_ge/N<=0.05 AND real>reverse AND Broke<=Fixed/2
AND own residual <= 0.75*before.

Usage: python scripts/phase8_locked_test.py
"""
from __future__ import annotations
import gc, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L
from phase8_gate_eval import ID2GP, chan_map

DEC_P = L.OUT / "LLAMA_FROZEN_GATE_DECISIONS.json"
HASH_P = L.OUT / "LLAMA_FROZEN_GATE_DECISIONS.sha256"
STATE = L.CACHE / "phase8_locked_state.json"
MAX_AUDIT = 2


def guard():
    if not DEC_P.exists() or not HASH_P.exists():
        raise SystemExit("REFUSING: frozen gate decisions + hash must exist before any test access.")
    cur = hashlib.sha256(DEC_P.read_bytes()).hexdigest()
    rec = HASH_P.read_text().strip()
    if cur != rec:
        raise SystemExit(f"REFUSING: decisions file changed after hashing (got {cur[:12]}, "
                         f"recorded {rec[:12]}). The firewall requires an immutable decision file.")
    L.log.info("decision-file hash verified (%s) — test access unlocked", cur[:16])
    return json.loads(DEC_P.read_text())


def eval_dir(model, tok, dev, ds, te, meta, base_pred, sample_set, CH, unit, dn, inj, ch):
    corr = (dn * unit).astype(np.float32)
    preds = {i: L.predict(model, tok, dev, ds[i], corr, inj)[0] for i in sample_set}
    ip = {i: base_pred[i] for i in te}
    for i in sample_set:
        ip[i] = preds[i]
    m = L.metrics(te, meta, base_pred, ip, CH)
    et = CH[ch]["etype"]
    own = [i for i in sample_set if meta[i]["etype"] == et]
    dmg = sum(1 for i in sample_set if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    gc.collect(); torch.cuda.empty_cache()
    return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"], "acc": m["acc"],
            "own_before": m["nt_before"][ch], "own_after": m["nt_after"][ch],
            "own_to_gold": sum(1 for i in own if ip[i] == meta[i]["gold"]),
            "own_to_other_wrong": sum(1 for i in own if ip[i] != meta[i]["gold"] and ip[i] != base_pred[i]),
            "n_touched": len(sample_set), "damage_on_correct": dmg,
            "other_after": {k: v for k, v in m["nt_after"].items() if k != ch}}


def confirm(real, rev, rnd):
    nets = rnd["nets"]; mu, sd = float(np.mean(nets)), float(np.std(nets))
    z = None if sd < 1e-6 else (real["net"] - mu) / sd
    nge = int(sum(1 for x in nets if x >= real["net"]))
    crit = {"z_ok": (z is not None and z >= L.Z_MIN),
            "nonparam_ok": (nge / len(nets)) <= L.FRAC_GE_MAX,
            "real_gt_reverse": real["net"] > rev["net"],
            "broke_controlled": (real["broke"] <= L.BROKE_RATIO * real["fixed"]
                                 if real["fixed"] > 0 else real["broke"] == 0),
            "residual_down": (real["own_after"] <= L.RESIDUAL_FRAC * real["own_before"]
                              if real["own_before"] > 0 else False)}
    if sd < 1e-6:
        crit["z_ok"] = (nge == 0 and real["net"] > 0 and real["net"] > rev["net"])
    return {"z": (None if z is None else round(z, 3)), "rand_mean": round(mu, 3),
            "rand_std": round(sd, 3), "rand_max": int(np.max(nets)), "n_ge_real": nge,
            "n_random": len(nets), "criteria": crit, "CONFIRMED": all(crit.values())}


def run_arm(model, tok, dev, ds, te, meta, base_pred, CH, ch, cfg, tag):
    g, p = ID2GP[ch]; et = CH[ch]["etype"]
    obs, inj, method, thr = cfg["obs"], cfg["inj"], cfg["method"], cfg["thr"]
    tr, va = L.splits_train_val()
    A = np.load(L.CACHE / f"acts_L{obs}.npy")
    unit, nrm, ne, nr = L.direction(A, tr, meta, et, g, method)   # train-only direction
    router = L.fit_router(A, tr, va, meta, et, p)                 # train/val-only router
    # score the TEST rows' router activations live (test acts are not cached)
    tsc = live_router_scores(model, tok, dev, ds, te, obs, router)
    routed = [i for i in te if base_pred[i] == p and tsc[i] >= thr]
    dn = cfg["rho"] * cfg["med_inj"]
    D = L.hidden_size()
    L.log.info("[%s/%s] locked test: routed=%d rho=%.2f |dh|=%.3f inj=L%d", tag, ch, len(routed),
               cfg["rho"], dn, inj)
    real = eval_dir(model, tok, dev, ds, te, meta, base_pred, routed, CH, unit, dn, inj, ch)
    rev = eval_dir(model, tok, dev, ds, te, meta, base_pred, routed, CH, -unit, dn, inj, ch)
    rnds = []
    for k in range(L.N_RANDOM):
        rnds.append(eval_dir(model, tok, dev, ds, te, meta, base_pred, routed, CH,
                             L.random_unit(D, L.TEST_SEED_BLOCK + k), dn, inj, ch))
    rnd = {"nets": [r["net"] for r in rnds]}
    ungated_set = [i for i in te if base_pred[i] == p]
    ungated = eval_dir(model, tok, dev, ds, te, meta, base_pred, ungated_set, CH, unit, dn, inj, ch)
    wrong = eval_dir(model, tok, dev, ds, te, meta, base_pred, routed, CH, unit, dn,
                     L.wrong_layer(), ch)
    conf = confirm(real, rev, rnd)
    L.log.info("[%s/%s] real=%+d rev=%+d rand=%.1f (max %d, nge %d/%d) z=%s ungated=%+d wrongL=%+d -> %s",
               tag, ch, real["net"], rev["net"], conf["rand_mean"], conf["rand_max"],
               conf["n_ge_real"], conf["n_random"], conf["z"], ungated["net"], wrong["net"],
               "CONFIRMED" if conf["CONFIRMED"] else "DENIED")
    return {"arm": tag, "channel": ch, "locked_config": cfg, "n_routed": len(routed),
            "real": real, "reverse": rev, "random": {**rnd, **{k: conf[k] for k in
                                                               ("rand_mean", "rand_std", "rand_max",
                                                                "n_ge_real", "n_random")}},
            "ungated": ungated, "wrong_layer": wrong, "spec_z": conf["z"],
            "confirmation": conf, "CONFIRMED": conf["CONFIRMED"]}


def live_router_scores(model, tok, dev, ds, idxs, obs, router):
    """Router scores for TEST rows: capture obs-layer MLP output live (test acts uncached)."""
    out = {}
    mlp = model.model.layers[obs].mlp
    for i in idxs:
        ids = tok.encode(L.make_prompt_text(tok, ds[i]), add_special_tokens=False)
        last = len(ids) - 1
        cap = {}

        def hook(m, _in, o):
            oo = o[0] if isinstance(o, tuple) else o
            cap["v"] = oo[0, last, :].detach().float().cpu().numpy()
        h = mlp.register_forward_hook(hook)
        with torch.no_grad():
            model(torch.tensor([ids], device=dev))
        h.remove()
        v = cap["v"].reshape(1, -1)
        out[i] = float(router["clf"].predict_proba(router["scaler"].transform(v))[0, 1])
    return out


def main():
    dec = guard()
    ds, meta = L.load_meta()
    tr, va, te = L.splits_all()
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    eligible = dec["stage1_eligible"]
    CH = chan_map(eligible)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    admits = [c for c, d in dec["decisions"].items() if d["decision"] == "ADMIT"]
    # ---- bounded rejection audit selection (protocol allows "up to 2"; disclosed rule) ----
    # The audit exists to FALSIFY the gate, so the 2 slots go to the rejections most likely to
    # be WRONG (i.e. adversarial to the gate), not to the most obviously-correct rejections:
    #   (a) the channel with the strongest INTERIOR specificity signal (max interior z with
    #       Net>0) — the best candidate for an F-2 over-rejection;
    #   (b) the channel with the highest best-validation Net — the largest behavioural effect
    #       the gate threw away;
    #   then any narrowly "specificity-rejected" channels (audit_candidate) fill remaining slots.
    # This selection can only make the gate look WORSE, never better, so it cannot bias the
    # verdict in the gate's favour. Made once, before any test access.
    curves_sel = json.loads((L.OUT / "llama_rho_curves.json").read_text())

    def interior_z(ch):
        rows = [r for r in curves_sel.get(ch, {}).get("rows", [])
                if r["rho"] < L.RHO_MAX and r["real"]["net"] > 0 and r["spec_z"] is not None]
        return max((r["spec_z"] for r in rows), default=-99)

    def best_net(ch):
        rows = curves_sel.get(ch, {}).get("rows", [])
        return max((r["real"]["net"] for r in rows), default=-99)

    rejected = [c for c, d in dec["decisions"].items() if d["decision"] == "REJECT"]
    ranked, seen = [], set()
    for ch in sorted(rejected, key=interior_z, reverse=True)[:1]:      # (a)
        ranked.append(ch); seen.add(ch)
    for ch in sorted(rejected, key=best_net, reverse=True):            # (b)
        if ch not in seen:
            ranked.append(ch); seen.add(ch); break
    for ch in rejected:                                               # (c) fill
        if ch not in seen and dec["decisions"][ch].get("audit_candidate"):
            ranked.append(ch); seen.add(ch)
    audits = ranked[:MAX_AUDIT]
    L.log.info("audit selection: interior_z=%s best_net=%s -> audits=%s",
               {c: interior_z(c) for c in rejected}, {c: best_net(c) for c in rejected}, audits)
    L.log.info("ADMIT arms: %s | bounded rejection audit: %s (of %d rejected)",
               admits, audits, len(rejected))
    if not admits and not audits:
        L.log.warning("no admitted channels and no specificity-rejections to audit")
    model, tok, dev = L.load_model()
    curves = json.loads((L.OUT / "llama_rho_curves.json").read_text())

    results = state.get("arms", {})
    for ch in admits:
        if ch in results:
            continue
        cfg = dec["decisions"][ch]["locked_config"]
        results[ch] = run_arm(model, tok, dev, ds, te, meta, base_pred, CH, ch, cfg, "ADMIT")
        state["arms"] = results; STATE.write_text(json.dumps(state, indent=2, default=float))
    for ch in audits:
        if ch in results:
            continue
        rows = curves.get(ch, {}).get("rows", [])
        best = max(rows, key=lambda r: r["real"]["net"])
        cfg = {k: best[k] for k in ("method", "obs", "inj", "thr", "rho", "med_inj")}
        results[ch] = run_arm(model, tok, dev, ds, te, meta, base_pred, CH, ch, cfg, "REJECT_AUDIT")
        state["arms"] = results; STATE.write_text(json.dumps(state, indent=2, default=float))

    # ---- score the gate ----
    tp = fp = tn = fn = 0
    scored = []
    for ch, r in results.items():
        pre = dec["decisions"][ch]["decision"]
        conf = r["CONFIRMED"]
        if pre == "ADMIT" and conf: tp += 1; k = "TP_admit_confirmed"
        elif pre == "ADMIT" and not conf: fp += 1; k = "FP_admit_denied (F-1 OVER-ADMISSION)"
        elif pre == "REJECT" and not conf: tn += 1; k = "TN_reject_denied"
        else: fn += 1; k = "FN_reject_confirmed (F-2 OVER-REJECTION)"
        scored.append({"channel": ch, "gate_decision": pre, "test_confirmed": conf, "outcome": k,
                       "net": r["real"]["net"], "z": r["spec_z"]})
        L.log.info("SCORE [%s]: gate=%s test_confirmed=%s -> %s", ch, pre, conf, k)
    out = {"model": L.MODELS[L.MODEL_KEY]["repo"], "revision": L.MODELS[L.MODEL_KEY]["revision"],
           "decisions_sha256": HASH_P.read_text().strip(),
           "arms": results, "scoring": scored,
           "confusion": {"TP": tp, "FP_over_admission": fp, "TN": tn, "FN_over_rejection": fn},
           "admission_precision": (tp / (tp + fp)) if (tp + fp) else None,
           "rejection_precision": (tn / (tn + fn)) if (tn + fn) else None,
           "F1_over_admission_events": fp, "F2_over_rejection_events": fn}
    (L.OUT / "LLAMA_PROSPECTIVE_RESULTS.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["confusion"], indent=2))


if __name__ == "__main__":
    main()
