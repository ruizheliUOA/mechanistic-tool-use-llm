"""
qwen7b_method_diagnostics.py — focused method diagnostics for Qwen2.5-7B native SAKIKO
=====================================================================================
Diagnoses the three weakest parts (seed=42), using cached/extracted Qwen activations:
  Part A — ca_direct layer sensitivity (obs L16/18/20/22 × inj L16/18/20 × alpha)
  Part B — rfi_tc router variants (LR / balanced / margin-augmented / obs sweep)
  Part C — DiffMean vs PCA direction extraction per channel

Stages:
  extract : extract Qwen MLP acts at L18, L22 (one multi-hook pass; L12/16/20 already cached)
  diag    : run Parts A/B/C and write reports

Usage:
  python scripts/qwen7b_method_diagnostics.py --stage extract
  python scripts/qwen7b_method_diagnostics.py --stage diag
"""
from __future__ import annotations
import argparse, json, pickle, gc, sys, logging, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen7b_native_sakiko as P
from qwen7b_native_sakiko_run import _metrics

OUT = P.ROOT / "final" / "results" / "7b_w2c_sakiko" / "method_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
EXTRA_LAYERS = [18, 22]
ALL_OBS = [16, 18, 20, 22]
BASE_DETAILS = P.ROOT / "final" / "results" / "7b_w2c_baseline" / "qwen25_7b_baseline_details.jsonl"
REFG = {"rfi_tc": "request_for_info", "ca_tc": "cannot_answer", "ca_direct": "cannot_answer"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("diag")


def acts_path(L): return P.CACHE_DIR / f"acts_L{L}.npy"


def stage_extract():
    need = [L for L in EXTRA_LAYERS if not acts_path(L).exists()]
    if not need:
        log.info("L18/L22 already cached."); return
    ds, meta = P.load_meta_and_dataset()
    model, tok, dev = P.load_model()
    n = len(ds); D = model.config.hidden_size
    acts = {L: np.zeros((n, D), dtype=np.float32) for L in need}
    mlps = {L: model.model.layers[L].mlp for L in need}
    for i in range(n):
        prompt = P.make_prompt_text(tok, ds[i])
        ids = tok.encode(prompt, add_special_tokens=False); last = len(ids)-1
        inp = torch.tensor([ids], device=dev); cap = {}
        hs = []
        for L in need:
            def mk(L):
                def h(m,_in,out,_L=L):
                    o = out[0] if isinstance(out, tuple) else out
                    cap[_L] = o[0, last, :].detach().float().cpu().numpy()
                return h
            hs.append(mlps[L].register_forward_hook(mk(L)))
        with torch.no_grad(): model(inp)
        for h in hs: h.remove()
        for L in need: acts[L][i] = cap[L]
        del inp
        if (i+1) % 400 == 0: log.info("  extract %d/%d", i+1, n); gc.collect(); torch.cuda.empty_cache()
    for L in need: np.save(acts_path(L), acts[L])
    log.info("Extracted layers %s", need)


# ── direction methods ─────────────────────────────────────────────────────────

def get_pools(meta, idxs, ch, ref_gold):
    err = [i for i in idxs if meta[i]["etype"] == ch]
    ref = [i for i in idxs if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    return err, ref


def diffmean(A, err, ref):
    raw = A[ref].mean(0) - A[err].mean(0)
    nrm = float(np.linalg.norm(raw))
    return (raw/(nrm+1e-12)).astype(np.float32), nrm


def pca_dir(A, err, ref, k=1):
    """PCA on combined err+ref (standardized). k=1: first PC sign-aligned to (ref-err) mean.
       k>1: DiffMean delta projected onto top-k PC subspace."""
    idx = err + ref
    sc = StandardScaler().fit(A[idx]); X = sc.transform(A[idx])
    pca = PCA(n_components=min(k, X.shape[1], len(idx)-1), random_state=42).fit(X)
    delta_std = sc.transform(A[ref].mean(0, keepdims=True))[0] - sc.transform(A[err].mean(0, keepdims=True))[0]
    if k == 1:
        comp = pca.components_[0]
        if np.dot(comp, delta_std) < 0: comp = -comp
        raw = (comp / sc.scale_)
    else:
        proj = np.zeros_like(delta_std)
        for c in pca.components_:
            proj += np.dot(delta_std, c) * c
        raw = (proj / sc.scale_)
    nrm = float(np.linalg.norm(raw))
    return (raw/(nrm+1e-12)).astype(np.float32), nrm


def fit_router(A, meta, tr, va, ch, balanced=False, extra_feat=None):
    pos = [i for i in tr if meta[i]["etype"] == ch]
    neg = [i for i in tr if meta[i]["correct"]]
    if balanced and len(neg) > len(pos):
        rng = np.random.RandomState(42); neg = list(rng.choice(neg, len(pos), replace=False))
    def feat(idxs):
        X = A[idxs]
        if extra_feat is not None:
            X = np.concatenate([X, np.array([[extra_feat[i]] for i in idxs])], axis=1)
        return X
    sc = StandardScaler().fit(feat(tr)) if False else StandardScaler().fit(feat(neg+pos))
    Xtr = sc.transform(feat(neg+pos)); ytr = np.array([0]*len(neg)+[1]*len(pos))
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42)
    cvk = min(5, len(pos))
    cv = float(cross_val_score(clf, Xtr, ytr, cv=cvk, scoring="roc_auc").mean()) if cvk >= 2 else None
    clf.fit(Xtr, ytr)
    vpos = [i for i in va if meta[i]["etype"] == ch]; vneg = [i for i in va if meta[i]["correct"]]
    vauc = None; pr = {}
    if vpos and vneg:
        Xv = sc.transform(feat(vneg+vpos)); yv = np.array([0]*len(vneg)+[1]*len(vpos))
        prob = clf.predict_proba(Xv)[:, 1]; vauc = float(roc_auc_score(yv, prob))
        # full-val scores for firing rate & P/R
        allv = sc.transform(feat(va)); vall = clf.predict_proba(allv)[:, 1]
        yall = np.array([1 if meta[i]["etype"] == ch else 0 for i in va])
        for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
            fire = vall >= thr
            tp = int(((fire) & (yall == 1)).sum()); fp = int(((fire) & (yall == 0)).sum()); fn = int(((~fire) & (yall == 1)).sum())
            pr[str(thr)] = {"precision": round(tp/(tp+fp), 3) if tp+fp else 0.0,
                            "recall": round(tp/(tp+fn), 3) if tp+fn else 0.0, "fires": int(fire.sum())}
    return {"sc": sc, "clf": clf, "feat": feat, "cv_auc": cv, "val_auc": vauc,
            "n_pos": len(pos), "n_neg": len(neg), "pr": pr}


def router_scores(router, A, idxs):
    return dict(zip(idxs, router["clf"].predict_proba(router["sc"].transform(router["feat"](idxs)))[:, 1]))


# ── intervention helpers (model) ──────────────────────────────────────────────

def precompute_corrected(model, tok, dev, ds, idxs, unit, median_norm, inj, alpha):
    cvec = (alpha*median_norm*unit).astype(np.float32)
    return {i: P.predict_hooked(model, tok, dev, ds[i], cvec, inj) for i in idxs}


def channel_net(va, meta, base_pred, ch, gate_pred, vscore, corrected, thr):
    ip = dict(base_pred); touched = 0
    for i in va:
        if base_pred[i] == gate_pred and vscore[i] >= thr and i in corrected:
            ip[i] = corrected[i]; touched += 1
    m = _metrics(va, meta, base_pred, ip)
    after = sum(1 for i in va if meta[i]["gold"] == P.CHANNELS[ch]["gold"] and ip[i] == P.CHANNELS[ch]["from_pred"])
    before = sum(1 for i in va if meta[i]["etype"] == ch)
    return {"net": m["net"], "fixed": m["fixed"], "broke": m["broke"], "touched": touched,
            "ch_before": before, "ch_after": after}


# ── DIAG ──────────────────────────────────────────────────────────────────────

def stage_diag():
    ds, meta = P.load_meta_and_dataset()
    tr = list(map(int, json.load(open(P.ROOT/"sakiko_v3/results/splits/train_idx.json"))))
    va = list(map(int, json.load(open(P.ROOT/"sakiko_v3/results/splits/val_idx.json"))))
    te = list(map(int, json.load(open(P.ROOT/"sakiko_v3/results/splits/test_idx.json"))))
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    A = {L: np.load(acts_path(L)) for L in ALL_OBS}
    # baseline score margins (for margin-augmented router): tool_call - request_for_info
    det = [json.loads(l) for l in open(BASE_DETAILS)]
    margin_tc_rfi = {i: det[i]["avg_logp"]["tool_call"] - det[i]["avg_logp"]["request_for_info"] for i in range(len(det))}

    model, tok, dev = P.load_model()

    # ════════ PART A: ca_direct layer sensitivity ════════
    log.info("===== PART A: ca_direct layer sensitivity =====")
    ch = "ca_direct"; gate = P.CHANNELS[ch]["from_pred"]  # "direct"
    err_ref = {L: get_pools(meta, tr, ch, REFG[ch]) for L in ALL_OBS}
    # direction + router per obs layer (cheap)
    obs_info = {}
    for L in ALL_OBS:
        err, ref = err_ref[L]
        unit, nrm = diffmean(A[L], err, ref)
        rt = fit_router(A[L], meta, tr, va, ch)
        median_norm = float(np.median(np.linalg.norm(A[L][tr], axis=1)))
        obs_info[L] = {"unit": unit, "dir_norm": round(nrm, 3), "median_norm": median_norm,
                       "router_val_auc": round(rt["val_auc"], 4) if rt["val_auc"] else None,
                       "n_err": len(err), "n_ref": len(ref), "router": rt}
        log.info("  ca_direct obs L%d: dir_norm=%.3f router_auc=%s n_err=%d n_ref=%d",
                 L, nrm, obs_info[L]["router_val_auc"], len(err), len(ref))
    partA = {"obs_layers": {L: {k: v for k, v in obs_info[L].items() if k not in ("unit", "router")} for L in ALL_OBS}, "scan": []}
    # intervention scan: obs × inj × alpha (threshold post-hoc)
    for L in ALL_OBS:
        rt = obs_info[L]["router"]; vscore = router_scores(rt, A[L], va)
        elig = [i for i in va if base_pred[i] == gate and vscore[i] >= 0.4]
        for inj in [16, 18, 20]:
            for alpha in [1.0, 2.0, 4.0, 6.0]:
                corrected = precompute_corrected(model, tok, dev, ds, elig, obs_info[L]["unit"],
                                                  obs_info[L]["median_norm"], inj, alpha)
                for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
                    r = channel_net(va, meta, base_pred, ch, gate, vscore, corrected, thr)
                    partA["scan"].append({"obs": L, "inj": inj, "alpha": alpha, "thr": thr, **r})
                gc.collect(); torch.cuda.empty_cache()
        log.info("  scanned obs L%d", L)
    bestA = max(partA["scan"], key=lambda r: (r["net"], -r["broke"]))
    partA["best_val"] = bestA
    log.info("  PART A best val: obs L%d inj L%d a=%.1f thr=%.2f net=%+d (ch %d->%d)",
             bestA["obs"], bestA["inj"], bestA["alpha"], bestA["thr"], bestA["net"], bestA["ch_before"], bestA["ch_after"])
    # locked test for best ca_direct config
    Lb = bestA["obs"]; rt = obs_info[Lb]["router"]; tsc = router_scores(rt, A[Lb], te)
    elig_te = [i for i in te if base_pred[i] == gate and tsc[i] >= bestA["thr"]]
    corr_te = precompute_corrected(model, tok, dev, ds, elig_te, obs_info[Lb]["unit"], obs_info[Lb]["median_norm"], bestA["inj"], bestA["alpha"])
    ip = dict(base_pred)
    for i in elig_te: ip[i] = corr_te[i]
    cad_before = sum(1 for i in te if meta[i]["etype"] == ch)
    cad_after = sum(1 for i in te if meta[i]["gold"] == "cannot_answer" and ip[i] == "direct")
    partA["locked_test"] = {"obs": Lb, "inj": bestA["inj"], "alpha": bestA["alpha"], "thr": bestA["thr"],
                            "ca_direct_before": cad_before, "ca_direct_after": cad_after,
                            "reduction": cad_before - cad_after}
    json.dump(partA, open(OUT/"partA_ca_direct_layer_scan.json", "w"), indent=2, default=float)
    log.info("  PART A locked test: ca_direct %d->%d (best obs L%d)", cad_before, cad_after, Lb)

    # ════════ PART B: rfi_tc router variants ════════
    log.info("===== PART B: rfi_tc router variants =====")
    ch = "rfi_tc"; gate = "tool_call"
    # fixed rfi_tc direction + native config (obs L20, inj L18, alpha 2.0 from pilot) for Net comparison
    err20, ref20 = get_pools(meta, tr, ch, REFG[ch])
    rfi_unit, rfi_norm = diffmean(A[20], err20, ref20)
    rfi_mednorm = float(np.median(np.linalg.norm(A[20][tr], axis=1)))
    RFI_INJ, RFI_ALPHA = 18, 2.0
    # precompute corrected preds for ALL gate-eligible (pred==tool_call) val samples (union)
    elig_all = [i for i in va if base_pred[i] == gate]
    corr_rfi = precompute_corrected(model, tok, dev, ds, elig_all, rfi_unit, rfi_mednorm, RFI_INJ, RFI_ALPHA)
    variants = {}
    def eval_variant(name, router, A_obs):
        vscore = router_scores(router, A_obs, va)
        best = None
        for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
            r = channel_net(va, meta, base_pred, ch, gate, vscore, corr_rfi, thr)
            if best is None or (r["net"], -r["broke"]) > (best["net"], -best["broke"]): best = {"thr": thr, **r}
        variants[name] = {"cv_auc": round(router["cv_auc"], 4) if router["cv_auc"] else None,
                          "val_auc": round(router["val_auc"], 4) if router["val_auc"] else None,
                          "pr": router["pr"], "best_thr": best["thr"], "net": best["net"],
                          "fixed": best["fixed"], "broke": best["broke"], "ch_before": best["ch_before"],
                          "ch_after": best["ch_after"], "touched": best["touched"]}
        log.info("  rfi_tc [%s]: cv=%s val_auc=%s best_net=%+d (thr%.2f)", name,
                 variants[name]["cv_auc"], variants[name]["val_auc"], best["net"], best["thr"])
    eval_variant("original_LR_L20", fit_router(A[20], meta, tr, va, ch), A[20])
    eval_variant("balanced_LR_L20", fit_router(A[20], meta, tr, va, ch, balanced=True), A[20])
    eval_variant("margin_aug_L20", fit_router(A[20], meta, tr, va, ch, extra_feat=margin_tc_rfi), A[20])
    for L in ALL_OBS:
        eval_variant(f"LR_L{L}", fit_router(A[L], meta, tr, va, ch), A[L])
    partB = {"fixed_direction": {"obs": 20, "inj": RFI_INJ, "alpha": RFI_ALPHA, "dir_norm": round(rfi_norm, 3)},
             "variants": variants}
    json.dump(partB, open(OUT/"partB_rfi_router_variants.json", "w"), indent=2, default=float)

    # ════════ PART C: DiffMean vs PCA per channel ════════
    log.info("===== PART C: direction extraction comparison =====")
    # native per-channel config (obs/inj/alpha) from pilot
    NATIVE = {"rfi_tc": (20, 18, 2.0), "ca_tc": (20, 16, 4.0), "ca_direct": (16, 16, 6.0)}
    routers_native = pickle.load(open(P.CACHE_DIR/"routers.pkl", "rb"))
    partC = {}
    for ch, (obsL, inj, alpha) in NATIVE.items():
        gate = P.CHANNELS[ch]["from_pred"]
        err, ref = get_pools(meta, tr, ch, REFG[ch])
        dm_unit, dm_norm = diffmean(A[obsL], err, ref)
        p1_unit, p1_norm = pca_dir(A[obsL], err, ref, k=1)
        pk_unit, pk_norm = pca_dir(A[obsL], err, ref, k=10)
        mednorm = float(np.median(np.linalg.norm(A[obsL][tr], axis=1)))
        rt = routers_native[ch]; Arouter = np.load(acts_path(rt["obs_layer"]))
        vscore = dict(zip(va, rt["clf"].predict_proba(rt["scaler"].transform(Arouter[va]))[:, 1]))
        elig = [i for i in va if base_pred[i] == gate and vscore[i] >= 0.4]
        res = {}
        for name, unit in [("diffmean", dm_unit), ("pca1", p1_unit), ("pca_top10", pk_unit)]:
            corr = precompute_corrected(model, tok, dev, ds, elig, unit, mednorm, inj, alpha)
            best = None
            for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
                r = channel_net(va, meta, base_pred, ch, gate, vscore, corr, thr)
                if best is None or (r["net"], -r["broke"]) > (best["net"], -best["broke"]): best = {"thr": thr, **r}
            res[name] = best
            gc.collect(); torch.cuda.empty_cache()
        partC[ch] = {"obs": obsL, "inj": inj, "alpha": alpha,
                     "dir_norm": {"diffmean": round(dm_norm, 3), "pca1": round(p1_norm, 3), "pca_top10": round(pk_norm, 3)},
                     "cos_diffmean_pca1": round(float(dm_unit @ p1_unit), 4),
                     "cos_diffmean_pca10": round(float(dm_unit @ pk_unit), 4),
                     "net": {k: res[k]["net"] for k in res}, "detail": res}
        log.info("  %s: net diffmean=%+d pca1=%+d pca10=%+d | cos(dm,pca1)=%.3f",
                 ch, res["diffmean"]["net"], res["pca1"]["net"], res["pca_top10"]["net"],
                 partC[ch]["cos_diffmean_pca1"])
    json.dump(partC, open(OUT/"partC_direction_comparison.json", "w"), indent=2, default=float)
    log.info("===== diagnostics done =====")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["extract", "diag"])
    a = ap.parse_args()
    (stage_extract if a.stage == "extract" else stage_diag)()


if __name__ == "__main__":
    main()
