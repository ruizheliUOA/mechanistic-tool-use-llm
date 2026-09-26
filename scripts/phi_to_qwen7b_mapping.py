"""
phi_to_qwen7b_mapping.py — Phi-3.5 → Qwen2.5-7B direction mapping pilot (seed=42)
=================================================================================
Maps Phi-derived DiffMean correction directions (Phi L18) into Qwen2.5-7B L16 space
via four mapping methods, injects them in Qwen-7B gated by Qwen-NATIVE routers, and
compares to the archived Qwen-7B target-native reference.

Stages:
  quality   : (CPU) build Phi & Qwen DiffMean dirs, train mappings (Ridge / PCA-Ridge /
              Procrustes / PLS) on train, map dirs, report val R² + cosine-to-native.
  intervene : (GPU) val α/threshold sweep per method (cascade, native routers, inject L16),
              select best on val, locked test once, placebo (real/reverse/random/wrong-layer).

Usage:
  python scripts/phi_to_qwen7b_mapping.py --stage quality
  python scripts/phi_to_qwen7b_mapping.py --stage intervene
"""
from __future__ import annotations
import argparse, json, pickle, gc, sys, logging, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen7b_native_sakiko as P
from qwen7b_native_sakiko_run import _metrics

ROOT = P.ROOT
PHI_CACHE = ROOT / "sakiko_v3" / "cache" / "acts_L18.npy"          # Phi source (3072)
QWEN_L16 = P.CACHE_DIR / "acts_L16.npy"                            # Qwen target/inject (3584)
PHI_TRACE = ROOT / "trace_db" / "w2c_phi35.jsonl"
SPLIT_DIR = ROOT / "sakiko_v3" / "results" / "splits"
OUT = ROOT / "final" / "results" / "mapping_phi_to_qwen7b"
MAPCACHE = OUT / "_cache"
OUT.mkdir(parents=True, exist_ok=True); MAPCACHE.mkdir(exist_ok=True)

PHI_OBS_LAYER = 18
QWEN_INJ_LAYER = 16
ALPHAS = [2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
NATIVE_SEED42_NET = 79     # archived reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("map")


def etype(gold, pred):
    if gold == pred: return "correct"
    if gold == "request_for_info" and pred == "tool_call": return "rfi_tc"
    if gold == "cannot_answer" and pred == "tool_call": return "ca_tc"
    if gold == "cannot_answer" and pred == "direct": return "ca_direct"
    return "other"


def load_phi_meta():
    rows = [json.loads(l) for l in open(PHI_TRACE)]
    return [{"idx": i, "gold": r["gold"], "pred": r["pred"],
             "correct": r["gold"] == r["pred"], "etype": etype(r["gold"], r["pred"])}
            for i, r in enumerate(rows)]


def splits():
    return (list(map(int, json.load(open(SPLIT_DIR/"train_idx.json")))),
            list(map(int, json.load(open(SPLIT_DIR/"val_idx.json")))),
            list(map(int, json.load(open(SPLIT_DIR/"test_idx.json")))))


def diffmean(acts, meta, idxs, ch, ref_gold):
    err = [i for i in idxs if meta[i]["etype"] == ch]
    ref = [i for i in idxs if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    if len(err) < 10 or len(ref) < 5: return None
    raw = acts[ref].mean(0) - acts[err].mean(0)
    nrm = float(np.linalg.norm(raw))
    return {"unit": (raw/(nrm+1e-12)).astype(np.float32), "norm": nrm,
            "n_err": len(err), "n_ref": len(ref)}


# ── mapping methods ───────────────────────────────────────────────────────────

def fit_mappers(Xtr, Ytr, Xval, Yval):
    """Return list of mapper dicts, each with f(X)->Yhat, hyperparam tuned on val R²."""
    sx, sy = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
    Xs, Ys = sx.transform(Xtr), sy.transform(Ytr)
    Xvs, Yvs = sx.transform(Xval), sy.transform(Yval)

    def r2(Yhat_std):  # in standardized space
        ss_res = np.sum((Yvs - Yhat_std)**2); ss_tot = np.sum((Yvs - Yvs.mean(0))**2)
        return 1.0 - ss_res/(ss_tot+1e-12)

    def make_f(fwd_std):
        def f(X):
            return fwd_std(sx.transform(X)) * sy.scale_ + sy.mean_
        return f

    mappers = {}

    # 1) Ridge — tune alpha
    best = None
    for a in [1.0, 10.0, 100.0, 1000.0]:
        r = Ridge(alpha=a, fit_intercept=True, random_state=42).fit(Xs, Ys)
        v = r2(r.predict(Xvs))
        if best is None or v > best[0]: best = (v, a, r)
    vr2, a, r = best
    mappers["ridge"] = {"val_r2": float(vr2), "hp": f"alpha={a:g}",
                        "f": make_f(lambda Z, r=r: r.predict(Z))}

    # 2) PCA-Ridge — PCA phi to k, ridge to full qwen_std
    best = None
    for k in [64, 128, 256]:
        pk = PCA(n_components=k, random_state=42).fit(Xs)
        r = Ridge(alpha=10.0, fit_intercept=True, random_state=42).fit(pk.transform(Xs), Ys)
        v = r2(r.predict(pk.transform(Xvs)))
        if best is None or v > best[0]: best = (v, k, pk, r)
    vr2, k, pk, r = best
    mappers["pca_ridge"] = {"val_r2": float(vr2), "hp": f"k={k}",
                            "f": make_f(lambda Z, pk=pk, r=r: r.predict(pk.transform(Z)))}

    # 3) Procrustes in shared PCA space (rank r)
    best = None
    for rk in [64, 128, 256]:
        pxa = PCA(n_components=rk, random_state=42).fit(Xs)
        pya = PCA(n_components=rk, random_state=42).fit(Ys)
        Xp, Yp = pxa.transform(Xs), pya.transform(Ys)
        U, _, Vt = np.linalg.svd(Xp.T @ Yp, full_matrices=False)
        R = (U @ Vt).astype(np.float32)   # rk×rk orthogonal
        Yvp_pred = pxa.transform(Xvs) @ R
        v = r2(pya.inverse_transform(Yvp_pred))
        if best is None or v > best[0]: best = (v, rk, pxa, pya, R)
    vr2, rk, pxa, pya, R = best
    mappers["procrustes"] = {"val_r2": float(vr2), "hp": f"rank={rk}",
                             "f": make_f(lambda Z, pxa=pxa, pya=pya, R=R: pya.inverse_transform(pxa.transform(Z) @ R))}

    # 4) PLS — tune n_components
    best = None
    for n in [16, 32, 64]:
        pls = PLSRegression(n_components=n, scale=False).fit(Xs, Ys)
        v = r2(pls.predict(Xvs))
        if best is None or v > best[0]: best = (v, n, pls)
    vr2, n, pls = best
    mappers["pls"] = {"val_r2": float(vr2), "hp": f"n={n}",
                      "f": make_f(lambda Z, pls=pls: pls.predict(Z))}

    return mappers, sx


def map_dir(f, mu_phi, unit):
    return (f((mu_phi + unit)[None])[0] - f(mu_phi[None])[0]).astype(np.float32)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


# ── STAGE quality ─────────────────────────────────────────────────────────────

def stage_quality():
    phi_acts = np.load(PHI_CACHE); qwen_acts = np.load(QWEN_L16)
    phi_meta = load_phi_meta()
    _, qmeta = P.load_meta_and_dataset()
    tr, va, te = splits()
    log.info("Phi %s | Qwen L16 %s | splits %d/%d/%d", phi_acts.shape, qwen_acts.shape, len(tr), len(va), len(te))

    # Phi DiffMean dirs @ L18 (from Phi errors); Qwen native DiffMean @ L16 (from Qwen errors)
    refg = {"rfi_tc": "request_for_info", "ca_tc": "cannot_answer", "ca_direct": "cannot_answer"}
    phi_dirs = {ch: diffmean(phi_acts, phi_meta, tr, ch, refg[ch]) for ch in P.CHANNELS}
    qwen_dirs = {ch: diffmean(qwen_acts, qmeta, tr, ch, refg[ch]) for ch in P.CHANNELS}

    mappers, sx = fit_mappers(phi_acts[tr], qwen_acts[tr], phi_acts[va], qwen_acts[va])
    mu_phi = phi_acts[tr].mean(0).astype(np.float32)

    report = {"phi_obs_layer": PHI_OBS_LAYER, "qwen_inj_layer": QWEN_INJ_LAYER,
              "phi_dirs": {ch: {k: v for k, v in d.items() if k != "unit"} for ch, d in phi_dirs.items() if d},
              "qwen_native_L16_dirs": {ch: {k: v for k, v in d.items() if k != "unit"} for ch, d in qwen_dirs.items() if d},
              "methods": {}}
    mapped_all = {}
    for name, m in mappers.items():
        mapped = {ch: map_dir(m["f"], mu_phi, phi_dirs[ch]["unit"]) for ch in phi_dirs if phi_dirs[ch]}
        mapped_all[name] = mapped
        coss = {ch: round(cos(mapped[ch], qwen_dirs[ch]["unit"]), 4) for ch in mapped if qwen_dirs[ch]}
        report["methods"][name] = {"val_r2": round(m["val_r2"], 4), "hp": m["hp"],
                                   "cosine_to_native_L16": coss,
                                   "mapped_norm": {ch: round(float(np.linalg.norm(mapped[ch])), 3) for ch in mapped}}
        log.info("[%s] val_R2=%.4f (%s) cos_to_native=%s", name, m["val_r2"], m["hp"], coss)

    # save mapped dirs + native L16 dirs for the intervene stage
    np.savez(MAPCACHE / "mapped_dirs.npz",
             **{f"{name}__{ch}": mapped_all[name][ch] for name in mapped_all for ch in mapped_all[name]},
             **{f"native__{ch}": qwen_dirs[ch]["unit"] for ch in qwen_dirs if qwen_dirs[ch]},
             median_norm_qwen_L16=np.array([np.median(np.linalg.norm(qwen_acts[tr], axis=1))], dtype=np.float32))
    json.dump(report, open(OUT / "mapping_quality.json", "w"), indent=2)
    _write_quality_md(report)
    log.info("quality stage done")


def _write_quality_md(rep):
    with open(OUT / "mapping_quality_summary.md", "w") as f:
        f.write("# Phi→Qwen2.5-7B Mapping Quality (val, seed=42)\n\n")
        f.write(f"Source Phi L{rep['phi_obs_layer']} (3072) → target Qwen L{rep['qwen_inj_layer']} (3584).\n\n")
        f.write("| method | val R² | hp | cos→native rfi_tc | ca_tc | ca_direct |\n|---|---|---|---|---|---|\n")
        for nm, m in rep["methods"].items():
            c = m["cosine_to_native_L16"]
            f.write(f"| {nm} | {m['val_r2']} | {m['hp']} | {c.get('rfi_tc')} | {c.get('ca_tc')} | {c.get('ca_direct')} |\n")
        f.write("\nCosine is between the **mapped Phi direction** and the **Qwen-native L16 DiffMean** "
                "direction per channel (1.0 = perfect alignment).\n")


# ── STAGE intervene ───────────────────────────────────────────────────────────

def stage_intervene():
    qwen_acts = np.load(QWEN_L16)
    acts_router = {12: np.load(P.CACHE_DIR/"acts_L12.npy"), 16: qwen_acts, 20: np.load(P.CACHE_DIR/"acts_L20.npy")}
    ds, qmeta = P.load_meta_and_dataset()
    tr, va, te = splits()
    routers = pickle.load(open(P.CACHE_DIR/"routers.pkl", "rb"))   # Qwen-native routers
    base_pred = {i: qmeta[i]["pred"] for i in range(len(qmeta))}
    md = np.load(MAPCACHE/"mapped_dirs.npz")
    median_norm = float(md["median_norm_qwen_L16"][0])
    methods = sorted({k.split("__")[0] for k in md.files if "__" in k and not k.startswith("native")})
    mapped = {nm: {ch: md[f"{nm}__{ch}"] for ch in P.CHANNELS if f"{nm}__{ch}" in md.files} for nm in methods}
    native = {ch: md[f"native__{ch}"] for ch in P.CHANNELS if f"native__{ch}" in md.files}

    model, tok, dev = P.load_model()

    def rscore(ch, idxs):
        r = routers[ch]; A = acts_router[r["obs_layer"]]
        return dict(zip(idxs, r["clf"].predict_proba(r["scaler"].transform(A[idxs]))[:, 1]))
    vsc = {ch: rscore(ch, va) for ch in routers}

    def corr_for(unit, alpha): return (alpha*median_norm*unit).astype(np.float32)

    def cascade(dirs, idxs, vscore, alpha, thr_map, gated=True, inj=QWEN_INJ_LAYER):
        ip = dict(base_pred); routed = Counter(); touched = []
        for i in idxs:
            chosen = None
            for ch in P.PRIORITY:
                if ch not in dirs: continue
                gate = (base_pred[i] == P.CHANNELS[ch]["from_pred"]) if gated else True
                if gate and vscore[ch][i] >= thr_map[ch]:
                    chosen = ch; break
            if not chosen: continue
            ip[i] = P.predict_hooked(model, tok, dev, ds[i], corr_for(dirs[chosen], alpha), inj)
            routed[chosen] += 1; touched.append(i)
        return ip, routed, touched

    # ── val sweep: precompute corrected preds per (alpha, channel, sample) to avoid
    #     redundant forwards across thresholds ──
    log.info("=== val sweep over methods (precomputed) ===")
    minthr = min(THRESHOLDS)
    def sweep_method(dirs):
        rows = []
        # precompute: pre[(alpha, ch)][i] = corrected pred for gate-eligible firing-at-minthr i
        pre = {}
        for alpha in ALPHAS:
            for ch in dirs:
                cvec = corr_for(dirs[ch], alpha)
                elig = [i for i in va if base_pred[i] == P.CHANNELS[ch]["from_pred"] and vsc[ch][i] >= minthr]
                pre[(alpha, ch)] = {i: P.predict_hooked(model, tok, dev, ds[i], cvec, QWEN_INJ_LAYER) for i in elig}
            gc.collect(); torch.cuda.empty_cache()
        for alpha in ALPHAS:
            for thr in THRESHOLDS:
                ip = dict(base_pred); touched = 0
                for i in va:
                    for ch in P.PRIORITY:
                        if ch not in dirs: continue
                        if base_pred[i] == P.CHANNELS[ch]["from_pred"] and vsc[ch][i] >= thr:
                            ip[i] = pre[(alpha, ch)].get(i, base_pred[i]); touched += 1; break
                m = _metrics(va, qmeta, base_pred, ip)
                rows.append({"alpha": alpha, "thr": thr, "net": m["net"], "fixed": m["fixed"],
                             "broke": m["broke"], "touched": touched, "acc": m["acc"]})
        return rows
    sweep = {}
    for nm in methods + ["native_L16"]:
        dirs = native if nm == "native_L16" else mapped[nm]
        rows = sweep_method(dirs)
        best = max(rows, key=lambda r: (r["net"], -r["broke"]))
        sweep[nm] = {"rows": rows, "best": best}
        log.info("[%s] best val: alpha=%.1f thr=%.2f net=%+d (fix%d/broke%d touched%d)",
                 nm, best["alpha"], best["thr"], best["net"], best["fixed"], best["broke"], best["touched"])
    json.dump(sweep, open(OUT/"mapping_val_sweep.json", "w"), indent=2, default=float)

    # select best MAPPING method (exclude native_L16) by val net
    best_method = max(methods, key=lambda nm: (sweep[nm]["best"]["net"], -sweep[nm]["best"]["broke"]))
    bc = sweep[best_method]["best"]
    log.info("SELECTED best mapping = %s (val net %+d)", best_method, bc["net"])

    # ── locked test for best mapping + native_L16 same-grid reference ──
    tsc = {ch: rscore(ch, te) for ch in routers}
    test_base_acc = sum(1 for i in te if base_pred[i] == qmeta[i]["gold"])/len(te)

    def run_test(dirs, alpha, thr, gated=True, inj=QWEN_INJ_LAYER):
        ip, routed, touched = cascade(dirs, te, tsc, alpha, {ch: thr for ch in dirs}, gated=gated, inj=inj)
        m = _metrics(te, qmeta, base_pred, ip)
        dmg = sum(1 for i in touched if qmeta[i]["correct"] and ip[i] != qmeta[i]["gold"])
        return ip, m, dict(routed), len(touched), dmg

    ip_map, m_map, routed_map, touch_map, dmg_map = run_test(mapped[best_method], bc["alpha"], bc["thr"])
    nb = sweep["native_L16"]["best"]
    _, m_nat, _, _, _ = run_test(native, nb["alpha"], nb["thr"])
    three_before = sum(1 for i in te if qmeta[i]["etype"] in P.CHANNELS)
    three_after = sum(1 for i in te if (
        (qmeta[i]["gold"]=="request_for_info" and ip_map[i]=="tool_call") or
        (qmeta[i]["gold"]=="cannot_answer" and ip_map[i]=="tool_call") or
        (qmeta[i]["gold"]=="cannot_answer" and ip_map[i]=="direct")))

    test_summary = {
        "best_method": best_method, "alpha": bc["alpha"], "threshold": bc["thr"],
        "base_acc": round(test_base_acc, 5), "mapped_acc": round(m_map["acc"], 5),
        "delta_acc": round(m_map["acc"]-test_base_acc, 5),
        "fixed": m_map["fixed"], "broke": m_map["broke"], "net": m_map["net"],
        "nt_before": m_map["nt_before"], "nt_after": m_map["nt_after"], "chan_fixed": m_map["chan_fixed"],
        "three_before": three_before, "three_after": three_after,
        "routed": routed_map, "touched": touch_map, "damage_on_correct": dmg_map,
        "native_L16_test_net": m_nat["net"], "native_archived_net": NATIVE_SEED42_NET,
        "pct_of_native_L16": round(100*m_map["net"]/max(m_nat["net"], 1), 1),
        "pct_of_native_archived": round(100*m_map["net"]/NATIVE_SEED42_NET, 1),
    }
    with open(OUT/"mapping_locked_test_details.jsonl", "w") as f:
        for i in te:
            f.write(json.dumps({"idx": i, "uuid": qmeta[i]["uuid"], "gold": qmeta[i]["gold"],
                "base_pred": base_pred[i], "int_pred": ip_map[i], "etype": qmeta[i]["etype"]}, ensure_ascii=False)+"\n")
    json.dump(test_summary, open(OUT/"mapping_locked_test_summary.json", "w"), indent=2, default=float)
    log.info("LOCKED TEST [%s]: acc %.4f->%.4f net=%+d (fix%d/broke%d) | native_L16 net=%+d | %%native=%.0f%%",
             best_method, test_base_acc, m_map["acc"], m_map["net"], m_map["fixed"], m_map["broke"],
             m_nat["net"], test_summary["pct_of_native_L16"])

    # ── placebo (best mapping) ──
    D = qwen_acts.shape[1]
    def variant(kind, rs=None, gated=True, inj=QWEN_INJ_LAYER):
        if kind == "reverse": dirs = {ch: -mapped[best_method][ch] for ch in mapped[best_method]}
        elif kind == "random":
            rng = np.random.RandomState(rs); dirs = {}
            for ch in mapped[best_method]:
                v = rng.randn(D).astype(np.float32); dirs[ch] = v/(np.linalg.norm(v)+1e-12)
        else: dirs = mapped[best_method]
        _, m, _, tch, _ = run_test(dirs, bc["alpha"], bc["thr"], gated=gated, inj=inj)
        return {"net": m["net"], "fixed": m["fixed"], "broke": m["broke"], "touched": tch}
    placebo = {"real": {"net": m_map["net"], "fixed": m_map["fixed"], "broke": m_map["broke"], "touched": touch_map}}
    placebo["reverse"] = variant("reverse")
    rnets = [variant("random", rs=2000+k)["net"] for k in range(10)]
    placebo["random"] = {"n": 10, "nets": rnets, "mean": round(float(np.mean(rnets)), 2),
                         "max": int(np.max(rnets)), "min": int(np.min(rnets)),
                         "real_ge_all": bool(m_map["net"] > max(rnets)),
                         "n_random_ge_real": int(sum(1 for x in rnets if x >= m_map["net"]))}
    placebo["wrong_layer"] = variant("real", inj=12)   # inject at L12 instead of L16
    json.dump(placebo, open(OUT/"mapping_placebo_controls.json", "w"), indent=2, default=float)
    log.info("placebo: real=%+d reverse=%+d random(mean%.1f max%d real>=all=%s) wrong_layer=%+d",
             m_map["net"], placebo["reverse"]["net"], placebo["random"]["mean"], placebo["random"]["max"],
             placebo["random"]["real_ge_all"], placebo["wrong_layer"]["net"])

    _write_test_md(test_summary, sweep, placebo, methods)
    log.info("intervene stage done")


def _write_test_md(ts, sweep, placebo, methods):
    with open(OUT/"mapping_locked_test_summary.md", "w") as f:
        f.write("# Phi→Qwen2.5-7B Mapping — Locked Test (seed=42)\n\n")
        f.write("## Method selection (val net)\n\n| method | val R²? | best α | thr | val net (fix/broke) |\n|---|---|---|---|---|\n")
        for nm in methods + ["native_L16"]:
            b = sweep[nm]["best"]
            f.write(f"| {nm} | — | {b['alpha']} | {b['thr']} | {b['net']:+d} ({b['fixed']}/{b['broke']}) |\n")
        f.write(f"\n**Selected mapping:** `{ts['best_method']}` (α={ts['alpha']}, thr={ts['threshold']}).\n\n")
        f.write("## Locked test\n\n| metric | value |\n|---|---|\n")
        f.write(f"| baseline acc | {ts['base_acc']:.4f} |\n| mapped-SAKIKO acc | {ts['mapped_acc']:.4f} |\n")
        f.write(f"| Δ acc | {ts['delta_acc']:+.4f} |\n| Fixed | {ts['fixed']} |\n| Broke | {ts['broke']} |\n| **Net** | **{ts['net']:+d}** |\n")
        f.write(f"| 3-channel before→after | {ts['three_before']}→{ts['three_after']} |\n")
        f.write(f"| touched | {ts['touched']} | \n| damage on correct | {ts['damage_on_correct']} |\n")
        f.write(f"| native-L16 test net (same grid) | {ts['native_L16_test_net']:+d} |\n")
        f.write(f"| archived native seed=42 net | +{ts['native_archived_net']} |\n")
        f.write(f"| **% of native-L16 recovered** | **{ts['pct_of_native_L16']}%** |\n")
        f.write(f"| % of archived native recovered | {ts['pct_of_native_archived']}% |\n\n")
        f.write("## Per-channel (before→after)\n\n| channel | before | after | fixed |\n|---|---|---|---|\n")
        for ch in P.CHANNELS:
            f.write(f"| {ch} | {ts['nt_before'][ch]} | {ts['nt_after'][ch]} | {ts['chan_fixed'][ch]} |\n")
        rd = placebo["random"]
        f.write(f"\n## Placebo\n\n| variant | Net |\n|---|---|\n")
        f.write(f"| real | **{placebo['real']['net']:+d}** |\n| reverse | {placebo['reverse']['net']:+d} |\n")
        f.write(f"| wrong_layer (L12) | {placebo['wrong_layer']['net']:+d} |\n")
        f.write(f"| random mean (max) | {rd['mean']} ({rd['max']}) |\n\n")
        f.write(f"Real ≥ all 10 random: **{rd['real_ge_all']}**; #random ≥ real: {rd['n_random_ge_real']}.\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["quality", "intervene"])
    a = ap.parse_args()
    (stage_quality if a.stage == "quality" else stage_intervene)()


if __name__ == "__main__":
    main()
