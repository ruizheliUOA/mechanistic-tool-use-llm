#!/usr/bin/env python3
"""
P0: Val-Selected + Locked Test Evaluation
==========================================

目标: 把 V3.1 cascade_opt 从 test-tuned → val-selected + locked test.

流程:
  Step 1: 三分 split (70/15/15, StratifiedShuffleSplit, seed=42)
  Step 2: val threshold sweep (0.40 ~ 0.90)
  Step 3: val alpha sweep for CA→direct (2 ~ 12)
  Step 4: 锁定最佳配置 → p0_locked_config.json
  Step 5: locked test eval (一次性, 不再调参)

使用方法:
    cd /path/to/project
    source .venv/bin/activate
    python sakiko/scripts/p0_run_locked_eval.py
"""

import gc, hashlib, json, logging, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Import from V3.1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phi_mainline.run_v31 import (
    ROOT, CACHE_DIR, RESULT_DIR, MODEL_PATH, DEVICE, SEED,
    LABEL_VOCAB, RFI2TC, CA2TC, CA2DIRECT, TARGET_GATE,
    load_data, build_meta, load_model,
    collect_activations, train_pca_basis, compute_correction_vec,
    train_binary_router, cascade_route,
    score_candidate_hooked, predict_with_correction,
    compute_metrics, print_report,
)
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sakiko_p0")

SPLIT_DIR = RESULT_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def three_way_split(meta, seed=SEED):
    n = len(meta)
    raw_labels = [m["error_type"] for m in meta]
    counts = Counter(raw_labels)
    strat_labels = [l if counts[l] >= 5 else "_rare" for l in raw_labels]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(range(n), strat_labels))

    trainval_labels = [strat_labels[i] for i in trainval_idx]
    val_frac = 0.15 / 0.85
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_rel, val_rel = next(sss2.split(range(len(trainval_idx)), trainval_labels))

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    train_idx = sorted(train_idx.astype(int).tolist())
    val_idx = sorted(val_idx.astype(int).tolist())
    test_idx = sorted(test_idx.astype(int).tolist())

    log.info("Three-way split: train=%d, val=%d, test=%d (total=%d)",
             len(train_idx), len(val_idx), len(test_idx), n)
    return train_idx, val_idx, test_idx


def save_split_info(meta, train_idx, val_idx, test_idx):
    def error_counts(idxs):
        c = Counter(meta[i]["error_type"] for i in idxs)
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    def idx_hash(idxs):
        return hashlib.sha256(json.dumps(idxs).encode()).hexdigest()[:16]

    info = {
        "seed": SEED,
        "split_ratio": "70/15/15",
        "n_total": len(meta),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_error_counts": error_counts(train_idx),
        "val_error_counts": error_counts(val_idx),
        "test_error_counts": error_counts(test_idx),
        "train_hash": idx_hash(train_idx),
        "val_hash": idx_hash(val_idx),
        "test_hash": idx_hash(test_idx),
    }

    with open(RESULT_DIR / "p0_split_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(SPLIT_DIR / f"{name}_idx.json", "w") as f:
            json.dump(idxs, f)

    log.info("Split info saved to %s", RESULT_DIR / "p0_split_info.json")
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ec = error_counts(idxs)
        log.info("  %s (n=%d): %s", name, len(idxs),
                 {k: v for k, v in list(ec.items())[:6]})
    return info


def evaluate_on_split(model, tok, rows, meta, eval_idx, acts_obs,
                      routers, cfg, corrections):
    results = []
    route_counts = Counter()

    for idx in tqdm(eval_idx, desc="Eval", leave=False):
        row = rows[idx]
        m = meta[idx]
        gold = m["gold"]
        clean_pred = m["pred"]

        ch_name, ch_prob = cascade_route(routers, acts_obs[idx], cfg)

        int_pred = clean_pred
        route = "none"

        if ch_name is not None and ch_name in corrections:
            gate_blocked = clean_pred in TARGET_GATE.get(ch_name, set())
            if not gate_blocked:
                corr_info = corrections[ch_name]
                int_pred, _ = predict_with_correction(
                    model, tok, row, corr_info["vec"],
                    corr_info["layer"], corr_info["mode"])
                route = ch_name
                route_counts[ch_name] += 1
            else:
                route = ch_name + "_gated"

        results.append({
            "idx": int(idx), "gold": gold,
            "clean_pred": clean_pred, "int_pred": int_pred,
            "route": route, "route_prob": round(ch_prob, 4),
            "error_type": m["error_type"],
        })

        if len(results) % 100 == 0:
            gc.collect(); torch.cuda.empty_cache()

    metrics = compute_metrics(results, cfg)
    return results, metrics


def _train_pipeline(acts_obs, meta, train_idx, cfg):
    log.info("Training PCA bases on train set...")
    basis_rfi = train_pca_basis(acts_obs, meta, train_idx, RFI2TC,
                                cfg["n_comp"], ref_gold="tool_call")
    basis_ca_tc = train_pca_basis(acts_obs, meta, train_idx, CA2TC,
                                  cfg["n_comp"], ref_gold="tool_call")
    basis_ca_dir = None
    if cfg.get("ca_direct_inj_layer") is not None:
        basis_ca_dir = train_pca_basis(acts_obs, meta, train_idx, CA2DIRECT,
                                       cfg["n_comp"], ref_gold="cannot_answer")

    corrections = {}
    if basis_rfi is not None and cfg.get("rfi_inj_layer") is not None:
        corrections["rfi_tc"] = {
            "vec": compute_correction_vec(basis_rfi, cfg["n_pcs"], cfg["rfi_alpha"]),
            "layer": cfg["rfi_inj_layer"],
            "mode": cfg["rfi_inj_mode"],
        }
    if basis_ca_tc is not None and cfg.get("ca_tc_inj_layer") is not None:
        corrections["ca_tc"] = {
            "vec": compute_correction_vec(basis_ca_tc, cfg["n_pcs"], cfg["ca_tc_alpha"]),
            "layer": cfg["ca_tc_inj_layer"],
            "mode": cfg["ca_tc_inj_mode"],
        }
    if basis_ca_dir is not None and cfg.get("ca_direct_inj_layer") is not None:
        corrections["ca_direct"] = {
            "vec": compute_correction_vec(basis_ca_dir, cfg["n_pcs"], cfg["ca_direct_alpha"]),
            "layer": cfg["ca_direct_inj_layer"],
            "mode": cfg["ca_direct_inj_mode"],
        }

    log.info("Training cascade binary routers on train set...")
    routers = {}
    routers["rfi_tc"] = train_binary_router(
        acts_obs, meta, train_idx, RFI2TC, name="rfi_tc")
    routers["ca_tc"] = train_binary_router(
        acts_obs, meta, train_idx, CA2TC, name="ca_tc")
    if cfg.get("ca_direct_inj_layer") is not None:
        routers["ca_direct"] = train_binary_router(
            acts_obs, meta, train_idx, CA2DIRECT, name="ca_direct")
    else:
        routers["ca_direct"] = None

    return routers, corrections


def val_threshold_sweep(model, tok, rows, meta, val_idx, acts_obs,
                        train_idx, base_cfg):
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    routers, corrections_base = _train_pipeline(acts_obs, meta, train_idx, base_cfg)

    sweep_results = []
    for thr in thresholds:
        log.info("=== Threshold sweep: thr=%.2f ===", thr)
        cfg = dict(base_cfg)
        cfg["threshold_rfi"] = thr
        cfg["threshold_ca_tc"] = thr
        cfg["threshold_ca_direct"] = thr

        _, metrics = evaluate_on_split(
            model, tok, rows, meta, val_idx, acts_obs,
            routers, cfg, corrections_base)

        entry = {
            "threshold": thr,
            "accuracy": metrics["intervened_accuracy"],
            "delta_acc": metrics["delta_accuracy"],
            "macro_f1": metrics["intervened_f1"],
            "delta_f1": metrics["delta_f1"],
            "fixed": metrics["total_corrected"],
            "broke": metrics["total_broken"],
            "net": metrics["net_benefit"],
            "channel_stats": metrics.get("channel_stats", {}),
            "baseline_accuracy": metrics["baseline_accuracy"],
        }
        sweep_results.append(entry)
        log.info("  thr=%.2f -> acc=%.2f%%, net=%+d, fixed=%d, broke=%d",
                 thr, entry["accuracy"], entry["net"],
                 entry["fixed"], entry["broke"])

    out = {"sweep_on": "val", "n_val": len(val_idx),
           "ca_direct_alpha": base_cfg["ca_direct_alpha"],
           "selection_criterion": "max net_benefit, tiebreak by accuracy",
           "results": sweep_results}
    with open(RESULT_DIR / "p0_val_threshold_sweep.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("Threshold sweep saved to p0_val_threshold_sweep.json")

    best = max(sweep_results, key=lambda x: (x["net"], x["accuracy"]))
    log.info("Best threshold on val: %.2f (net=%+d, acc=%.2f%%)",
             best["threshold"], best["net"], best["accuracy"])
    return best["threshold"], routers, sweep_results


def val_alpha_sweep(model, tok, rows, meta, val_idx, acts_obs,
                    train_idx, base_cfg, locked_threshold, routers):
    alphas = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    basis_rfi = train_pca_basis(acts_obs, meta, train_idx, RFI2TC,
                                base_cfg["n_comp"], ref_gold="tool_call")
    basis_ca_tc = train_pca_basis(acts_obs, meta, train_idx, CA2TC,
                                  base_cfg["n_comp"], ref_gold="tool_call")
    basis_ca_dir = train_pca_basis(acts_obs, meta, train_idx, CA2DIRECT,
                                   base_cfg["n_comp"], ref_gold="cannot_answer")

    sweep_results = []
    for alpha in alphas:
        log.info("=== Alpha sweep: a=%.1f ===", alpha)
        cfg = dict(base_cfg)
        cfg["threshold_rfi"] = locked_threshold
        cfg["threshold_ca_tc"] = locked_threshold
        cfg["threshold_ca_direct"] = locked_threshold
        cfg["ca_direct_alpha"] = alpha

        corrections = {}
        if basis_rfi is not None and cfg.get("rfi_inj_layer") is not None:
            corrections["rfi_tc"] = {
                "vec": compute_correction_vec(basis_rfi, cfg["n_pcs"], cfg["rfi_alpha"]),
                "layer": cfg["rfi_inj_layer"],
                "mode": cfg["rfi_inj_mode"],
            }
        if basis_ca_tc is not None and cfg.get("ca_tc_inj_layer") is not None:
            corrections["ca_tc"] = {
                "vec": compute_correction_vec(basis_ca_tc, cfg["n_pcs"], cfg["ca_tc_alpha"]),
                "layer": cfg["ca_tc_inj_layer"],
                "mode": cfg["ca_tc_inj_mode"],
            }
        if basis_ca_dir is not None and cfg.get("ca_direct_inj_layer") is not None:
            corrections["ca_direct"] = {
                "vec": compute_correction_vec(basis_ca_dir, cfg["n_pcs"], alpha),
                "layer": cfg["ca_direct_inj_layer"],
                "mode": cfg["ca_direct_inj_mode"],
            }

        _, metrics = evaluate_on_split(
            model, tok, rows, meta, val_idx, acts_obs,
            routers, cfg, corrections)

        entry = {
            "ca_direct_alpha": alpha,
            "accuracy": metrics["intervened_accuracy"],
            "delta_acc": metrics["delta_accuracy"],
            "macro_f1": metrics["intervened_f1"],
            "delta_f1": metrics["delta_f1"],
            "fixed": metrics["total_corrected"],
            "broke": metrics["total_broken"],
            "net": metrics["net_benefit"],
            "channel_stats": metrics.get("channel_stats", {}),
            "baseline_accuracy": metrics["baseline_accuracy"],
        }
        sweep_results.append(entry)
        log.info("  a=%.1f -> acc=%.2f%%, net=%+d, fixed=%d, broke=%d",
                 alpha, entry["accuracy"], entry["net"],
                 entry["fixed"], entry["broke"])

    out = {"sweep_on": "val", "n_val": len(val_idx),
           "locked_threshold": locked_threshold,
           "selection_criterion": "max net_benefit, tiebreak by accuracy",
           "results": sweep_results}
    with open(RESULT_DIR / "p0_val_alpha_sweep.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("Alpha sweep saved to p0_val_alpha_sweep.json")

    best = max(sweep_results, key=lambda x: (x["net"], x["accuracy"]))
    log.info("Best alpha on val: %.1f (net=%+d, acc=%.2f%%)",
             best["ca_direct_alpha"], best["net"], best["accuracy"])
    return best["ca_direct_alpha"], sweep_results


def save_locked_config(locked_threshold, locked_alpha, base_cfg):
    locked = {
        "description": "Val-selected locked config for P0 final test",
        "selection_method": "threshold and alpha selected on val set only",
        "threshold_rfi": locked_threshold,
        "threshold_ca_tc": locked_threshold,
        "threshold_ca_direct": locked_threshold,
        "ca_direct_alpha": locked_alpha,
        "rfi_alpha": base_cfg["rfi_alpha"],
        "ca_tc_alpha": base_cfg["ca_tc_alpha"],
        "obs_layer": base_cfg["obs_layer"],
        "rfi_inj_layer": base_cfg["rfi_inj_layer"],
        "rfi_inj_mode": base_cfg["rfi_inj_mode"],
        "ca_tc_inj_layer": base_cfg["ca_tc_inj_layer"],
        "ca_tc_inj_mode": base_cfg["ca_tc_inj_mode"],
        "ca_direct_inj_layer": base_cfg["ca_direct_inj_layer"],
        "ca_direct_inj_mode": base_cfg["ca_direct_inj_mode"],
        "n_comp": base_cfg["n_comp"],
        "n_pcs": base_cfg["n_pcs"],
    }
    with open(RESULT_DIR / "p0_locked_config.json", "w") as f:
        json.dump(locked, f, indent=2, ensure_ascii=False)
    log.info("Locked config saved: threshold=%.2f, ca_direct_alpha=%.1f",
             locked_threshold, locked_alpha)
    return locked


def final_locked_test(model, tok, rows, meta, test_idx, acts_obs,
                      train_idx, locked_cfg):
    log.info("=" * 60)
    log.info("FINAL LOCKED TEST -- test set only, no further tuning")
    log.info("=" * 60)

    cfg = {
        "obs_layer": locked_cfg["obs_layer"],
        "rfi_inj_layer": locked_cfg["rfi_inj_layer"],
        "rfi_inj_mode": locked_cfg["rfi_inj_mode"],
        "rfi_alpha": locked_cfg["rfi_alpha"],
        "threshold_rfi": locked_cfg["threshold_rfi"],
        "ca_tc_inj_layer": locked_cfg["ca_tc_inj_layer"],
        "ca_tc_inj_mode": locked_cfg["ca_tc_inj_mode"],
        "ca_tc_alpha": locked_cfg["ca_tc_alpha"],
        "threshold_ca_tc": locked_cfg["threshold_ca_tc"],
        "ca_direct_inj_layer": locked_cfg["ca_direct_inj_layer"],
        "ca_direct_inj_mode": locked_cfg["ca_direct_inj_mode"],
        "ca_direct_alpha": locked_cfg["ca_direct_alpha"],
        "threshold_ca_direct": locked_cfg["threshold_ca_direct"],
        "n_comp": locked_cfg["n_comp"],
        "n_pcs": locked_cfg["n_pcs"],
    }

    routers, corrections = _train_pipeline(acts_obs, meta, train_idx, cfg)

    results, metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs,
        routers, cfg, corrections)

    metrics["router_cv"] = {}
    for ch_name, r in routers.items():
        if r is not None:
            metrics["router_cv"][ch_name] = {
                "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                "n_pos": r["n_pos"], "n_neg": r["n_neg"],
            }

    metrics["config_name"] = "p0_locked_test"
    metrics["locked_config"] = locked_cfg
    metrics["note"] = "Val-selected config. Test set used only for this final evaluation."

    with open(RESULT_DIR / "p0_final_test_eval.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False,
                  default=lambda o: int(o) if hasattr(o, "item") else o)

    with open(RESULT_DIR / "p0_final_test_details.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False,
                               default=lambda o: int(o) if hasattr(o, "item") else o) + "\n")

    log.info("Final test results saved.")
    return metrics


def write_comparison_md(metrics_new, old_file="v31_v31_cascade_opt.json"):
    old_path = RESULT_DIR / old_file
    old = {}
    if old_path.exists():
        with open(old_path) as f:
            old = json.load(f)

    lines = []
    lines.append("# P0: Locked Test vs Old Test-Tuned Comparison\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d')}\n")
    lines.append("## Key Notes\n")
    lines.append("- **Old (v31_cascade_opt)**: test-tuned exploratory result")
    lines.append("- **New (p0_locked_test)**: val-selected + locked test\n")
    lines.append("## Results Comparison\n")
    lines.append("| Metric | Old (test-tuned) | New (val-selected) | Change |")
    lines.append("|---|---|---|---|")

    old_n = old.get('n_test', 'N/A')
    new_n = metrics_new['n_test']
    lines.append(f"| Test samples | {old_n} | {new_n} | split changed |")

    old_ba = old.get('baseline_accuracy', 0)
    new_ba = metrics_new['baseline_accuracy']
    lines.append(f"| Baseline accuracy | {old_ba}% | {new_ba}% | — |")

    old_ia = old.get('intervened_accuracy', 0)
    new_ia = metrics_new['intervened_accuracy']
    lines.append(f"| Intervened accuracy | {old_ia}% | {new_ia}% | {new_ia - old_ia:+.2f}% |")

    old_da = old.get('delta_accuracy', 0)
    new_da = metrics_new['delta_accuracy']
    lines.append(f"| Delta accuracy | +{old_da}% | +{new_da}% | — |")

    old_f1 = old.get('intervened_f1', 0)
    new_f1 = metrics_new['intervened_f1']
    lines.append(f"| Macro-F1 | {old_f1}% | {new_f1}% | {new_f1 - old_f1:+.2f}% |")

    old_fix = old.get('total_corrected', 'N/A')
    new_fix = metrics_new['total_corrected']
    lines.append(f"| Fixed | {old_fix} | {new_fix} | — |")

    old_brk = old.get('total_broken', 'N/A')
    new_brk = metrics_new['total_broken']
    lines.append(f"| Broke | {old_brk} | {new_brk} | — |")

    old_net = old.get('net_benefit', 'N/A')
    new_net = metrics_new['net_benefit']
    lines.append(f"| Net benefit | {old_net} | {new_net} | — |")

    lines.append("\n## Per-channel (new)\n")
    for ch, cs in metrics_new.get("channel_stats", {}).items():
        lines.append(f"- **{ch}**: routed={cs['n_routed']}, fixed={cs['fixed']}, "
                     f"broke={cs['broke']}, net={cs['net']}, precision={cs['precision']}%")

    locked = metrics_new.get('locked_config', {})
    lines.append("\n## Config\n")
    lines.append("| Param | Old | New (val-selected) |")
    lines.append("|---|---|---|")
    lines.append(f"| threshold | 0.60 (test-tuned) | {locked.get('threshold_rfi', 'N/A')} (val-selected) |")
    lines.append(f"| ca_direct_alpha | 8.0 (test-tuned) | {locked.get('ca_direct_alpha', 'N/A')} (val-selected) |")

    lines.append("\n## Conclusion\n")
    lines.append(f"Old v31_cascade_opt (acc={old_ia}%, net={old_net}) = **test-tuned exploratory result**.")
    lines.append(f"New p0_locked_test (acc={new_ia}%, net={new_net}) = **clean evaluation result**.")

    with open(RESULT_DIR / "p0_locked_vs_old_comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Comparison saved to p0_locked_vs_old_comparison.md")


def main():
    t0 = time.time()

    log.info("=" * 60)
    log.info("P0: Val-Selected + Locked Test Evaluation")
    log.info("=" * 60)

    rows = load_data()
    meta = build_meta(rows)
    model, tok = load_model()

    base_cfg = {
        "obs_layer": 18,
        "rfi_inj_layer": 14,
        "rfi_inj_mode": "mlp_all",
        "rfi_alpha": 5.0,
        "ca_tc_inj_layer": 16,
        "ca_tc_inj_mode": "mlp_prompt",
        "ca_tc_alpha": 2.0,
        "ca_direct_inj_layer": 16,
        "ca_direct_inj_mode": "mlp_prompt",
        "ca_direct_alpha": 2.0,
        "n_comp": 64,
        "n_pcs": 20,
        "threshold_rfi": 0.60,
        "threshold_ca_tc": 0.60,
        "threshold_ca_direct": 0.60,
    }

    acts_obs = collect_activations(model, tok, rows, base_cfg["obs_layer"])

    log.info("Step 1: Three-way split")
    train_idx, val_idx, test_idx = three_way_split(meta)
    split_info = save_split_info(meta, train_idx, val_idx, test_idx)

    log.info("Step 2: Val threshold sweep")
    locked_threshold, routers, thr_results = val_threshold_sweep(
        model, tok, rows, meta, val_idx, acts_obs, train_idx, base_cfg)

    log.info("Step 3: Val alpha sweep (CA->direct)")
    locked_alpha, alpha_results = val_alpha_sweep(
        model, tok, rows, meta, val_idx, acts_obs, train_idx,
        base_cfg, locked_threshold, routers)

    log.info("Step 4: Lock config")
    locked_cfg = save_locked_config(locked_threshold, locked_alpha, base_cfg)

    log.info("Step 5: Final locked test eval")
    final_metrics = final_locked_test(
        model, tok, rows, meta, test_idx, acts_obs, train_idx, locked_cfg)

    print_report(final_metrics, locked_cfg, "P0 Locked Test")
    write_comparison_md(final_metrics)

    elapsed = time.time() - t0
    log.info("P0 complete. Total time: %.1f min", elapsed / 60)

    print("\n" + "=" * 70)
    print("  P0 SUMMARY")
    print("=" * 70)
    print(f"  Split: train={split_info['n_train']}, val={split_info['n_val']}, test={split_info['n_test']}")
    print(f"  Val-selected threshold: {locked_threshold:.2f}")
    print(f"  Val-selected CA->direct alpha: {locked_alpha:.1f}")
    print(f"  Final test accuracy: {final_metrics['intervened_accuracy']:.2f}% (d={final_metrics['delta_accuracy']:+.2f}%)")
    print(f"  Final test net benefit: {final_metrics['net_benefit']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
