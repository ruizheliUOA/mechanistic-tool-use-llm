"""
qwen7b_native_sakiko_multiseed.py — Qwen2.5-7B native SAKIKO multi-seed robustness
==================================================================================
Reuses the seed-independent cached activations (data/processed/qwen25_7b_w2c/cache/
acts_L{12,16,20}.npy) — only the train/val/test partition changes per seed, so NO
re-extraction is needed. Model loaded once; all 5 seeds run sequentially with
incremental writes.

Per seed: deterministic stratified 70/15/15 split -> LR routers + DiffMean directions
(3 channels) -> reduced val sweep -> lock -> locked test once -> controls.

Reduced grid (covers seed-42 winners): inj offsets {0,-2,-4}, alpha {2,4,6},
threshold {0.4..0.8}. Controls: REAL/REVERSE/UNGATED all seeds; RANDOM x10 for
seeds {42,123}, x1 (labeled limited) for others.

Outputs under final/results/7b_w2c_sakiko/multiseed/  (does NOT touch the pilot files).

Usage:
  /path/to/project/.venv/bin/python scripts/qwen7b_native_sakiko_multiseed.py
"""
from __future__ import annotations
import json, pickle, gc, time, logging, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen7b_native_sakiko as P
from qwen7b_native_sakiko_run import _metrics

SEEDS = [42, 123, 456, 789, 2024]
RANDOM_FULL_SEEDS = {42, 123}      # 10 random dirs; others get 1 (limited)
INJ_OFFSETS = [0, -2, -4]
ALPHAS = [2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]

OUT = P.OUT / "multiseed"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("ms")


def gen_split(meta, seed):
    """Deterministic stratified 70/15/15 on qwen error_type (rare classes pooled)."""
    n = len(meta)
    raw = [m["etype"] for m in meta]
    cnt = Counter(raw)
    strat = [e if cnt[e] >= 5 else "_rare" for e in raw]
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trv, te = next(s1.split(range(n), strat))
    strat_trv = [strat[i] for i in trv]
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=seed)
    tr_rel, va_rel = next(s2.split(range(len(trv)), strat_trv))
    tr = sorted(int(trv[i]) for i in tr_rel)
    va = sorted(int(trv[i]) for i in va_rel)
    te = sorted(int(i) for i in te)
    return tr, va, te


def fit_channel(ch, cfg, tr, va, acts, meta):
    """Choose obs layer by val AUC; fit router + DiffMean direction at that layer."""
    best = None
    for L in P.OBS_LAYERS:
        A = acts[L]
        pos = [i for i in tr if meta[i]["etype"] == ch]
        neg = [i for i in tr if meta[i]["correct"]]
        if len(pos) < 10:
            return None
        sc = StandardScaler(); Xs = sc.fit_transform(A[neg + pos])
        y = np.array([0]*len(neg) + [1]*len(pos))
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42)
        clf.fit(Xs, y)
        vpos = [i for i in va if meta[i]["etype"] == ch]
        vneg = [i for i in va if meta[i]["correct"]]
        vauc = None
        if vpos and vneg:
            Xv = sc.transform(A[vneg + vpos]); yv = np.array([0]*len(vneg)+[1]*len(vpos))
            vauc = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
        cand = {"obs_layer": L, "scaler": sc, "clf": clf, "val_auc": vauc,
                "n_pos": len(pos), "n_neg": len(neg)}
        if best is None or (vauc or 0) > (best["val_auc"] or 0):
            best = cand
    # direction at chosen obs
    L = best["obs_layer"]; A = acts[L]
    err = [i for i in tr if meta[i]["etype"] == ch]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == cfg["ref_gold"]]
    if len(err) < 10 or len(ref) < 5:
        best["direction"] = None; return best
    emean = A[err].mean(0); rmean = A[ref].mean(0)
    raw = rmean - emean; nrm = float(np.linalg.norm(raw))
    best["direction"] = {"unit": (raw/(nrm+1e-12)).astype(np.float32),
                         "median_norm": float(np.median(np.linalg.norm(A[tr], axis=1))),
                         "norm": nrm, "n_err": len(err), "n_ref": len(ref)}
    return best


def rscore(chinfo, idxs, acts):
    A = acts[chinfo["obs_layer"]]
    return dict(zip(idxs, chinfo["clf"].predict_proba(chinfo["scaler"].transform(A[idxs]))[:, 1]))


def run_seed(seed, model, tok, dev, acts, meta, ds, base_pred):
    tr, va, te = gen_split(meta, seed)
    chans = {}
    for ch, cfg in P.CHANNELS.items():
        ci = fit_channel(ch, cfg, tr, va, acts, meta)
        if ci and ci["direction"]:
            chans[ch] = ci
    # ── per-channel val sweep ──
    locked = {}; cache = {}
    va_score = {ch: rscore(chans[ch], va, acts) for ch in chans}
    for ch, ci in chans.items():
        cfg = P.CHANNELS[ch]; unit = ci["direction"]["unit"]; mn = ci["direction"]["median_norm"]
        obsL = ci["obs_layer"]
        firing = [i for i in va if base_pred[i] == cfg["from_pred"] and va_score[ch][i] >= min(THRESHOLDS)]
        inj_layers = sorted({max(0, obsL+o) for o in INJ_OFFSETS})
        rows = []
        for inj in inj_layers:
            for a in ALPHAS:
                corr = (a*mn*unit).astype(np.float32); preds = {}
                for i in firing:
                    preds[i] = P.predict_hooked(model, tok, dev, ds[i], corr, inj)
                cache[(ch, inj, a)] = preds
                for thr in THRESHOLDS:
                    ip = dict(base_pred); t = 0
                    for i in firing:
                        if va_score[ch][i] >= thr: ip[i] = preds[i]; t += 1
                    m = _metrics(va, meta, base_pred, ip)
                    rows.append({"inj": inj, "alpha": a, "thr": thr, "net": m["net"],
                                 "fixed": m["fixed"], "broke": m["broke"],
                                 "ch_fixed": m["chan_fixed"][ch], "touched": t})
                gc.collect(); torch.cuda.empty_cache()
        b = max(rows, key=lambda r: (r["net"], -r["broke"], r["ch_fixed"]))
        locked[ch] = {"obs_layer": obsL, "inj_layer": b["inj"], "alpha": b["alpha"],
                      "threshold": b["thr"], "val_net": b["net"], "router_val_auc": ci["val_auc"],
                      "dir_norm": round(ci["direction"]["norm"], 3),
                      "n_ref": ci["direction"]["n_ref"], "n_err_train": ci["direction"]["n_err"]}
        log.info("  seed %d [%s] obs L%d inj L%d a=%.1f thr=%.2f val_net=%+d AUC=%.3f",
                 seed, ch, obsL, b["inj"], b["alpha"], b["thr"], b["net"], ci["val_auc"] or 0)

    def cascade(idxs, override=None, gated=True):
        vs = {ch: rscore(chans[ch], idxs, acts) for ch in locked}
        ip = dict(base_pred); routed = Counter(); touched = []
        for i in idxs:
            chosen = None
            for ch in P.PRIORITY:
                if ch not in locked: continue
                gate = (base_pred[i] == P.CHANNELS[ch]["from_pred"]) if gated else True
                if gate and vs[ch][i] >= locked[ch]["threshold"]:
                    chosen = ch; break
            if not chosen: continue
            inj = locked[chosen]["inj_layer"]; a = locked[chosen]["alpha"]; mn = chans[chosen]["direction"]["median_norm"]
            unit = override[chosen] if override else chans[chosen]["direction"]["unit"]
            key = (chosen, inj, a)
            if override is None and i in cache.get(key, {}):
                ip[i] = cache[key][i]
            else:
                ip[i] = P.predict_hooked(model, tok, dev, ds[i], (a*mn*unit).astype(np.float32), inj)
            routed[chosen] += 1; touched.append(i)
        return ip, routed, touched

    # ── locked test ──
    base_acc = sum(1 for i in te if base_pred[i] == meta[i]["gold"])/len(te)
    tip, troute, ttouch = cascade(te)
    tm = _metrics(te, meta, base_pred, tip)
    dmg = sum(1 for i in ttouch if meta[i]["correct"] and tip[i] != meta[i]["gold"])
    ntc = sum(1 for i in ttouch if meta[i]["correct"])
    three_before = sum(1 for i in te if meta[i]["etype"] in P.CHANNELS)
    three_after = sum(1 for i in te if (
        (meta[i]["gold"]=="request_for_info" and tip[i]=="tool_call") or
        (meta[i]["gold"]=="cannot_answer" and tip[i]=="tool_call") or
        (meta[i]["gold"]=="cannot_answer" and tip[i]=="direct")))

    # ── controls ──
    D = acts[P.OBS_LAYERS[0]].shape[1]
    def variant(kind, rs=None, gated=True):
        if kind == "reverse": ov = {ch: -chans[ch]["direction"]["unit"] for ch in locked}
        elif kind == "random":
            rng = np.random.RandomState(rs); ov = {}
            for ch in locked:
                v = rng.randn(D).astype(np.float32); ov[ch] = v/(np.linalg.norm(v)+1e-12)
        else: ov = None
        ip, _, tch = cascade(te, override=ov, gated=gated)
        m = _metrics(te, meta, base_pred, ip)
        return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"], "n_touched": len(tch)}

    ctrl = {"real": {"fixed": tm["fixed"], "broke": tm["broke"], "net": tm["net"], "n_touched": len(ttouch)}}
    ctrl["reverse"] = variant("reverse")
    ctrl["ungated"] = variant("real", gated=False)
    nrand = 10 if seed in RANDOM_FULL_SEEDS else 1
    rnets = [variant("random", rs=1000+k)["net"] for k in range(nrand)]
    ctrl["random"] = {"n": nrand, "nets": rnets, "limited": nrand < 10,
                      "mean": round(float(np.mean(rnets)), 2), "std": round(float(np.std(rnets)), 2),
                      "max": int(np.max(rnets)), "min": int(np.min(rnets)),
                      "real_net": tm["net"], "real_ge_all": bool(tm["net"] > max(rnets)),
                      "n_random_ge_real": int(sum(1 for x in rnets if x >= tm["net"]))}
    log.info("  seed %d TEST: acc %.4f->%.4f net=%+d (fix%d/broke%d) 3ch %d->%d | real>=allrand=%s(max%d)",
             seed, base_acc, tm["acc"], tm["net"], tm["fixed"], tm["broke"], three_before, three_after,
             ctrl["random"]["real_ge_all"], ctrl["random"]["max"])

    row = {
        "seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "base_acc": round(base_acc, 5), "sakiko_acc": round(tm["acc"], 5),
        "delta_acc": round(tm["acc"]-base_acc, 5), "fixed": tm["fixed"], "broke": tm["broke"],
        "net": tm["net"], "three_before": three_before, "three_after": three_after,
        "nt_before": tm["nt_before"], "nt_after": tm["nt_after"], "chan_fixed": tm["chan_fixed"],
        "routed": dict(troute), "n_touched": len(ttouch), "touched_correct": ntc, "damage_on_correct": dmg,
        "locked": locked,
        "router_auc": {ch: round(locked[ch]["router_val_auc"], 4) if locked[ch]["router_val_auc"] else None for ch in locked},
    }
    return row, ctrl


def main():
    ds, meta = P.load_meta_and_dataset()
    acts = {L: np.load(P.CACHE_DIR / f"acts_L{L}.npy") for L in P.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    model, tok, dev = P.load_model()

    rows = []; controls = {}
    details_f = open(OUT / "qwen25_7b_multiseed_details.jsonl", "w")
    t0 = time.time()
    for seed in SEEDS:
        log.info("================ SEED %d ================", seed)
        row, ctrl = run_seed(seed, model, tok, dev, acts, meta, ds, base_pred)
        rows.append(row); controls[str(seed)] = ctrl
        details_f.write(json.dumps({"seed": seed, **row}, default=float, ensure_ascii=False) + "\n")
        details_f.flush()
        # incremental dump
        json.dump({"per_seed": rows}, open(OUT / "qwen25_7b_multiseed_results.json", "w"),
                  indent=2, default=float)
        json.dump(controls, open(OUT / "qwen25_7b_controls.json", "w"), indent=2, default=float)
    details_f.close()
    log.info("All seeds done in %.1f min", (time.time()-t0)/60)
    _write_reports(rows, controls)


def _write_reports(rows, controls):
    nets = [r["net"] for r in rows]; deltas = [r["delta_acc"] for r in rows]
    agg = {"n_seeds": len(rows), "net_mean": round(float(np.mean(nets)), 2),
           "net_std": round(float(np.std(nets)), 2), "net_min": int(np.min(nets)),
           "net_max": int(np.max(nets)), "n_positive": int(sum(1 for x in nets if x > 0)),
           "delta_acc_mean": round(float(np.mean(deltas)), 4)}
    json.dump({"aggregate": agg, "per_seed": rows}, open(OUT / "qwen25_7b_multiseed_results.json", "w"),
              indent=2, default=float)

    with open(OUT / "qwen25_7b_multiseed_summary.md", "w") as f:
        f.write("# Qwen2.5-7B Native SAKIKO — Multi-Seed Robustness (W2C)\n\n")
        f.write("> Reduced val grid (inj {0,-2,-4}, alpha {2,4,6}, thr {0.4..0.8}); per-seed "
                "stratified 70/15/15 on Qwen error types; cached seed-independent activations; "
                "locked test once per seed. Pilot seed=42 outputs untouched.\n\n")
        f.write("## Per-seed locked test\n\n")
        f.write("| seed | base→SAKIKO acc | Δ | Fixed | Broke | Net | 3ch before→after | rfi_tc | ca_tc | ca_direct |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            nb, na = r["nt_before"], r["nt_after"]
            f.write(f"| {r['seed']} | {r['base_acc']:.4f}→{r['sakiko_acc']:.4f} | {r['delta_acc']:+.4f} "
                    f"| {r['fixed']} | {r['broke']} | **{r['net']:+d}** | {r['three_before']}→{r['three_after']} "
                    f"| {nb['rfi_tc']}→{na['rfi_tc']} | {nb['ca_tc']}→{na['ca_tc']} | {nb['ca_direct']}→{na['ca_direct']} |\n")
        f.write(f"\n**Aggregate:** Net = {agg['net_mean']} ± {agg['net_std']} "
                f"(min {agg['net_min']}, max {agg['net_max']}); positive {agg['n_positive']}/{agg['n_seeds']}; "
                f"mean Δacc = {agg['delta_acc_mean']:+.4f}.\n\n")
        f.write("## Router AUC & directions per seed\n\n")
        f.write("| seed | rfi_tc AUC | ca_tc AUC | ca_direct AUC | ca_direct dir_norm | ca ref pool |\n|---|---|---|---|---|---|\n")
        for r in rows:
            lc = r["locked"]
            cad = lc.get("ca_direct", {})
            f.write(f"| {r['seed']} | {r['router_auc'].get('rfi_tc')} | {r['router_auc'].get('ca_tc')} "
                    f"| {r['router_auc'].get('ca_direct')} | {cad.get('dir_norm','—')} | {cad.get('n_ref','—')} |\n")

    with open(OUT / "qwen25_7b_controls_summary.md", "w") as f:
        f.write("# Qwen2.5-7B Native SAKIKO — Controls (multi-seed)\n\n")
        f.write("| seed | real net | reverse net | ungated net (broke) | gated broke | random mean (max) | real≥all rand |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            c = controls[str(r['seed'])]; rd = c["random"]
            lim = " *(1, limited)*" if rd["limited"] else ""
            f.write(f"| {r['seed']} | **{c['real']['net']:+d}** | {c['reverse']['net']:+d} "
                    f"| {c['ungated']['net']:+d} ({c['ungated']['broke']}) | {c['real']['broke']} "
                    f"| {rd['mean']} ({rd['max']}){lim} | {rd['real_ge_all']} |\n")


if __name__ == "__main__":
    main()
