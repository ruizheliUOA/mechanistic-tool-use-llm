"""
phase6_dev_diagnostics.py — CPU-only retrospective geometry diagnostics (Phase 6).
==================================================================================
NO model, NO GPU, NO tuning, NO test-set decisions. Reads the existing activation caches
(gitignored / out-of-repo) and per-sample baseline details for Qwen2.5-7B and
Mistral-7B-v0.3, and computes matched per-channel geometry statistics at each channel's
archived/locked observation layer. Purpose: discriminate the Phase-6 competing hypotheses
(direction quality vs saturation vs heterogeneity vs scale vs router/steering mismatch)
on DEVELOPMENT evidence only.

Outputs: final/results/actionability_gate_planning/dev_geometry_diagnostics.csv (+ stdout).
"""
from __future__ import annotations
import json, csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "actionability_gate_planning"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen25_7b": {
        "cache": ROOT / "data/processed/qwen25_7b_w2c/cache",
        "details": ROOT / "data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl",
        "hidden": 3584, "layers": 28,
        # channel -> (gold, pred, obs_layer, inj_layer, locked_alpha, spec_z_lockedtest)
        "channels": {
            "rfi_tc":    ("request_for_info", "tool_call", 20, 18, 2.0, None),
            "ca_tc":     ("cannot_answer", "tool_call", 20, 16, 4.0, None),
            "ca_direct": ("cannot_answer", "direct", 20, 16, 6.0, None),  # archived taskA cfg
            "ca_rfi":    ("cannot_answer", "request_for_info", 20, 18, 2.0, 4.82),
            "tc_rfi":    ("tool_call", "request_for_info", 24, 24, 1.0, 0.82),
        },
    },
    "mistral7b_v03": {
        "cache": Path("/root/autodl-tmp/phase5_mistral_cache"),
        "details": Path("/root/autodl-tmp/phase5_mistral_cache/mistral7b_w2c_baseline_details.jsonl"),
        "hidden": 4096, "layers": 32,
        "channels": {
            "rfi_tc":    ("request_for_info", "tool_call", 22, 20, 6.0, -1.22),
            "ca_tc":     ("cannot_answer", "tool_call", 22, 18, 6.0, 1.91),
            "ca_direct": ("cannot_answer", "direct", 22, 18, 6.0, 0.32),
        },
    },
}

SPLIT_DIR = ROOT / "final" / "results" / "splits"


def load_meta(details):
    det = [json.loads(l) for l in open(details)]
    return [{"gold": d["gold"], "pred": d["pred"], "correct": d["pred"] == d["gold"]}
            for d in det]


def diffmean(A, err, ref):
    raw = A[ref].mean(0) - A[err].mean(0)
    n = float(np.linalg.norm(raw))
    return raw / (n + 1e-12), n


def main():
    tr = list(map(int, json.load(open(SPLIT_DIR / "train_idx.json"))))
    rows = []
    for mname, M in MODELS.items():
        meta = load_meta(M["details"])
        cache = {}
        for ch, (gold, pred, obs, inj, alpha, specz) in M["channels"].items():
            for L in {obs, inj}:
                if L not in cache:
                    p = M["cache"] / f"acts_L{L}.npy"
                    cache[L] = np.load(p) if p.exists() else None
            A = cache[obs]
            if A is None:
                print(f"  !! {mname}/{ch}: missing cache L{obs}, skipped"); continue
            err = [i for i in tr if meta[i]["gold"] == gold and meta[i]["pred"] == pred]
            ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == gold]
            neg = [i for i in tr if meta[i]["pred"] == pred
                   and not (meta[i]["gold"] == gold and meta[i]["pred"] == pred)]
            if len(err) < 10 or len(ref) < 5:
                print(f"  !! {mname}/{ch}: err={len(err)} ref={len(ref)}, skipped"); continue

            dm, dm_norm = diffmean(A, err, ref)
            med_obs = float(np.median(np.linalg.norm(A[tr], axis=1)))
            med_inj = (float(np.median(np.linalg.norm(cache[inj][tr], axis=1)))
                       if cache.get(inj) is not None else float("nan"))
            # split-half DiffMean estimation stability
            rng = np.random.RandomState(0)
            cos_sh = []
            for _ in range(20):
                pe = rng.permutation(err); pr = rng.permutation(ref)
                d1, _ = diffmean(A, list(pe[:len(pe)//2]), list(pr[:len(pr)//2]))
                d2, _ = diffmean(A, list(pe[len(pe)//2:]), list(pr[len(pr)//2:]))
                cos_sh.append(float(d1 @ d2))
            # PCA on err+ref (standardized) — cos(DM, PC1), PC1 EVR
            idx = err + ref
            sc = StandardScaler().fit(A[idx]); X = sc.transform(A[idx])
            pca = PCA(n_components=5, random_state=42).fit(X)
            delta = (sc.transform(A[ref].mean(0, keepdims=True))[0]
                     - sc.transform(A[err].mean(0, keepdims=True))[0])
            c1 = pca.components_[0]
            cos_dm_pc1 = abs(float(np.dot(c1, delta / (np.linalg.norm(delta) + 1e-12))))
            # error-cloud heterogeneity: 2-means silhouette + PC1 EVR of err-only cloud
            Xe = StandardScaler().fit_transform(A[err])
            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xe)
            sil = float(silhouette_score(Xe, km.labels_)) if len(set(km.labels_)) > 1 else 0.0
            frac_minor = float(min(np.bincount(km.labels_)) / len(err))
            pca_e = PCA(n_components=3, random_state=42).fit(Xe)
            evr1 = float(pca_e.explained_variance_ratio_[0])
            # router weight alignment (raw-space): LR on standardized acts, w/scale -> unit
            scr = StandardScaler().fit(A[neg + err])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                                     random_state=42).fit(
                scr.transform(A[neg + err]), [0] * len(neg) + [1] * len(err))
            w = clf.coef_[0] / scr.scale_
            w = w / (np.linalg.norm(w) + 1e-12)
            cos_dm_router = float(np.dot(dm, -w))  # -w: router points TOWARD error class
            rows.append({
                "model": mname, "channel": ch, "obs": obs, "obs_depth": round(obs / M["layers"], 3),
                "n_err_train": len(err), "n_ref_train": len(ref),
                "med_act_norm_obs": round(med_obs, 3), "med_act_norm_inj": round(med_inj, 3),
                "dm_norm": round(dm_norm, 4), "norm_ratio": round(dm_norm / med_obs, 4),
                "dm_norm_per_sqrtD": round(dm_norm / np.sqrt(M["hidden"]), 5),
                "locked_alpha": alpha,
                "pert_ratio_at_inj": round(alpha * med_obs / med_inj, 2) if med_inj == med_inj else None,
                "cos_dm_pc1": round(cos_dm_pc1, 3),
                "splithalf_dm_cos_mean": round(float(np.mean(cos_sh)), 3),
                "err_cloud_2means_silhouette": round(sil, 3),
                "err_cloud_minor_frac": round(frac_minor, 3),
                "err_cloud_pc1_evr": round(evr1, 3),
                "cos_dm_routerdir": round(cos_dm_router, 3),
                "lockedtest_spec_z": specz,
            })
            print(f"{mname}/{ch}: dmnorm={dm_norm:.3f} ratio={dm_norm/med_obs:.3f} "
                  f"splithalf={np.mean(cos_sh):.3f} sil={sil:.3f} evr1={evr1:.3f} "
                  f"cos(dm,router)={cos_dm_router:.3f} pert_ratio={alpha*med_obs/med_inj:.2f} "
                  f"z={specz}")
    with open(OUT / "dev_geometry_diagnostics.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)
    print("\nwrote", OUT / "dev_geometry_diagnostics.csv")


if __name__ == "__main__":
    main()
