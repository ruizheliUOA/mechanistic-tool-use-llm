#!/usr/bin/env python3
"""
P2-Extra: Independent Baselines for Reviewer Concern
====================================================

Adds two independently re-run baselines on the same locked split:
1) GLOBAL_STEERING_INDEP:
   - Single global correction direction (no error-type-specific vectors)
   - Single global binary router (error vs correct)
   - One correction channel only
2) UNGATED_TOP1_INDEP:
   - Keeps 3 learned correction vectors
   - Removes threshold gating + target gate
   - Always applies top-1 router channel per sample

Outputs:
  - sakiko/results/p2_independent_baselines.json
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phi_mainline.run_v31 import (
    RESULT_DIR,
    SEED,
    RFI2TC,
    CA2TC,
    CA2DIRECT,
    load_data,
    build_meta,
    load_model,
    collect_activations,
    cascade_route,
    compute_correction_vec,
    predict_with_correction,
    compute_metrics,
)
from phi_mainline.p0_run_locked_eval import (
    evaluate_on_split,
    _train_pipeline,
)
from phi_placebo.p2_placebo_controls import (
    load_splits,
    load_locked_config,
    build_locked_cfg,
    extract_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sakiko_p2_indep")

TARGET_ERRORS = {RFI2TC, CA2TC, CA2DIRECT}


def train_global_basis(acts_obs, meta, train_idx, n_comp=64, n_pcs=20, alpha=5.0):
    err_idx = [i for i in train_idx if meta[i]["error_type"] in TARGET_ERRORS]
    ref_idx = [i for i in train_idx if meta[i]["correct"]]

    if len(err_idx) < 20 or len(ref_idx) < 20:
        raise RuntimeError(f"Insufficient data for global basis: err={len(err_idx)}, ref={len(ref_idx)}")

    combined = np.concatenate([acts_obs[err_idx], acts_obs[ref_idx]], axis=0)
    scaler = StandardScaler()
    combined_s = scaler.fit_transform(combined)

    k = min(n_comp, combined_s.shape[0] - 1, combined_s.shape[1])
    pca = PCA(n_components=k, random_state=SEED)
    pca.fit(combined_s)

    err_pcs = pca.transform(scaler.transform(acts_obs[err_idx]))
    ref_pcs = pca.transform(scaler.transform(acts_obs[ref_idx]))

    basis = {
        "components": pca.components_.astype(np.float32),
        "mean_": scaler.mean_.astype(np.float32),
        "scale_": scaler.scale_.astype(np.float32),
        "err_mean_pc": err_pcs.mean(axis=0).astype(np.float32),
        "ref_mean_pc": ref_pcs.mean(axis=0).astype(np.float32),
    }

    vec = compute_correction_vec(basis, n_pcs=n_pcs, alpha=alpha)
    return vec, len(err_idx), len(ref_idx)


def train_global_router(acts_obs, meta, train_idx):
    idx_pos = [i for i in train_idx if meta[i]["error_type"] in TARGET_ERRORS]
    idx_neg = [i for i in train_idx if meta[i]["correct"]]
    all_idx = idx_neg + idx_pos
    X = acts_obs[all_idx]
    y = np.array([0] * len(idx_neg) + [1] * len(idx_pos))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)
    clf.fit(X_s, y)

    return {"scaler": scaler, "clf": clf, "n_pos": len(idx_pos), "n_neg": len(idx_neg)}


def evaluate_global_steering_indep(model, tok, rows, meta, test_idx, acts_obs, cfg, global_router, global_corr):
    threshold = cfg["threshold_rfi"]
    layer = cfg["ca_tc_inj_layer"] if cfg.get("ca_tc_inj_layer") is not None else cfg["rfi_inj_layer"]
    mode = cfg["ca_tc_inj_mode"] if cfg.get("ca_tc_inj_mode") is not None else cfg["rfi_inj_mode"]

    results = []
    for idx in tqdm(test_idx, desc="GLOBAL_STEERING_INDEP", leave=False):
        row = rows[idx]
        m = meta[idx]
        clean_pred = m["pred"]

        X_s = global_router["scaler"].transform(acts_obs[idx].reshape(1, -1))
        prob = global_router["clf"].predict_proba(X_s)[0, 1]

        int_pred = clean_pred
        route = "none"
        if prob >= threshold:
            int_pred, _ = predict_with_correction(model, tok, row, global_corr, layer, mode)
            route = "global"

        results.append({
            "idx": int(idx),
            "gold": m["gold"],
            "clean_pred": clean_pred,
            "int_pred": int_pred,
            "route": route,
            "route_prob": float(round(prob, 4)),
            "error_type": m["error_type"],
        })

    metrics = compute_metrics(results, cfg)
    return extract_metrics(metrics)


def evaluate_ungated_top1_indep(model, tok, rows, meta, test_idx, acts_obs, routers, cfg, corrections):
    cfg0 = dict(cfg)
    cfg0["threshold_rfi"] = 0.0
    cfg0["threshold_ca_tc"] = 0.0
    cfg0["threshold_ca_direct"] = 0.0

    results = []
    for idx in tqdm(test_idx, desc="UNGATED_TOP1_INDEP", leave=False):
        row = rows[idx]
        m = meta[idx]
        clean_pred = m["pred"]

        ch_name, ch_prob = cascade_route(routers, acts_obs[idx], cfg0)
        int_pred = clean_pred
        route = "none"

        if ch_name is not None and ch_name in corrections:
            corr_info = corrections[ch_name]
            int_pred, _ = predict_with_correction(
                model, tok, row,
                corr_info["vec"], corr_info["layer"], corr_info["mode"]
            )
            route = ch_name + "_ungated"

        results.append({
            "idx": int(idx),
            "gold": m["gold"],
            "clean_pred": clean_pred,
            "int_pred": int_pred,
            "route": route,
            "route_prob": float(round(ch_prob, 4)),
            "error_type": m["error_type"],
        })

        if len(results) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    metrics = compute_metrics(results, cfg)
    return extract_metrics(metrics)


def main():
    t0 = time.time()

    rows = load_data()
    meta = build_meta(rows)
    model, tok = load_model()
    locked = load_locked_config()
    cfg = build_locked_cfg(locked)
    train_idx, _, test_idx = load_splits()

    acts_obs = collect_activations(model, tok, rows, cfg["obs_layer"])

    routers, real_corrections = _train_pipeline(acts_obs, meta, train_idx, cfg)

    _, real_metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, real_corrections
    )
    real_summary = extract_metrics(real_metrics)

    global_vec, n_err, n_ref = train_global_basis(
        acts_obs, meta, train_idx,
        n_comp=cfg["n_comp"], n_pcs=cfg["n_pcs"], alpha=cfg["rfi_alpha"]
    )
    global_router = train_global_router(acts_obs, meta, train_idx)
    global_summary = evaluate_global_steering_indep(
        model, tok, rows, meta, test_idx, acts_obs, cfg, global_router, global_vec
    )

    ungated_summary = evaluate_ungated_top1_indep(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, real_corrections
    )

    out = {
        "meta": {
            "script": "p2_independent_baselines.py",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_n": len(test_idx),
            "train_n": len(train_idx),
            "global_basis": {"n_err": n_err, "n_ref": n_ref},
            "definitions": {
                "GLOBAL_STEERING_INDEP": "single global correction direction + global binary router",
                "UNGATED_TOP1_INDEP": "threshold=0 + no TARGET_GATE; always apply top1 router channel",
            },
        },
        "REAL_LOCKED": real_summary,
        "GLOBAL_STEERING_INDEP": global_summary,
        "UNGATED_TOP1_INDEP": ungated_summary,
    }

    out_path = RESULT_DIR / "p2_independent_baselines.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    elapsed = (time.time() - t0) / 60.0
    log.info("Saved: %s", out_path)
    log.info("REAL_LOCKED: acc=%.2f net=%+d", real_summary["accuracy"], real_summary["net"])
    log.info("GLOBAL_STEERING_INDEP: acc=%.2f net=%+d", global_summary["accuracy"], global_summary["net"])
    log.info("UNGATED_TOP1_INDEP: acc=%.2f net=%+d", ungated_summary["accuracy"], ungated_summary["net"])
    log.info("Done in %.1f min", elapsed)


if __name__ == "__main__":
    main()
