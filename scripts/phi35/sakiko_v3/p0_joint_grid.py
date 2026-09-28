#!/usr/bin/env python3
"""P0 Joint Grid Search: threshold × ca_direct_alpha on val set only."""

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sakiko_v3.run_v31 import (
    RESULT_DIR, load_data, build_meta, load_model, collect_activations,
    train_pca_basis, compute_correction_vec, train_binary_router,
    RFI2TC, CA2TC, CA2DIRECT,
)
from sakiko_v3.p0_run_locked_eval import (
    three_way_split, evaluate_on_split, _train_pipeline,
)
import logging, sys as _sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    handlers=[logging.StreamHandler(_sys.stdout)])
log = logging.getLogger("p0_grid")

THRESHOLDS = [0.40, 0.50, 0.60, 0.70]
ALPHAS = [6.0, 8.0, 10.0, 12.0]

def main():
    t0 = time.time()
    rows = load_data()
    meta = build_meta(rows)
    model, tok = load_model()

    base_cfg = {
        "obs_layer": 18,
        "rfi_inj_layer": 14, "rfi_inj_mode": "mlp_all", "rfi_alpha": 5.0,
        "ca_tc_inj_layer": 16, "ca_tc_inj_mode": "mlp_prompt", "ca_tc_alpha": 2.0,
        "ca_direct_inj_layer": 16, "ca_direct_inj_mode": "mlp_prompt", "ca_direct_alpha": 2.0,
        "n_comp": 64, "n_pcs": 20,
        "threshold_rfi": 0.60, "threshold_ca_tc": 0.60, "threshold_ca_direct": 0.60,
    }

    acts_obs = collect_activations(model, tok, rows, base_cfg["obs_layer"])
    train_idx, val_idx, _ = three_way_split(meta)

    # Train routers once (threshold-independent)
    routers, _ = _train_pipeline(acts_obs, meta, train_idx, base_cfg)

    # PCA bases once (alpha-independent)
    basis_rfi = train_pca_basis(acts_obs, meta, train_idx, RFI2TC,
                                base_cfg["n_comp"], ref_gold="tool_call")
    basis_ca_tc = train_pca_basis(acts_obs, meta, train_idx, CA2TC,
                                  base_cfg["n_comp"], ref_gold="tool_call")
    basis_ca_dir = train_pca_basis(acts_obs, meta, train_idx, CA2DIRECT,
                                   base_cfg["n_comp"], ref_gold="cannot_answer")

    # Fixed corrections for rfi_tc and ca_tc
    corr_rfi = {
        "vec": compute_correction_vec(basis_rfi, base_cfg["n_pcs"], base_cfg["rfi_alpha"]),
        "layer": base_cfg["rfi_inj_layer"], "mode": base_cfg["rfi_inj_mode"],
    }
    corr_ca_tc = {
        "vec": compute_correction_vec(basis_ca_tc, base_cfg["n_pcs"], base_cfg["ca_tc_alpha"]),
        "layer": base_cfg["ca_tc_inj_layer"], "mode": base_cfg["ca_tc_inj_mode"],
    }

    grid_results = []
    for alpha in ALPHAS:
        corr_ca_dir = {
            "vec": compute_correction_vec(basis_ca_dir, base_cfg["n_pcs"], alpha),
            "layer": base_cfg["ca_direct_inj_layer"], "mode": base_cfg["ca_direct_inj_mode"],
        }
        corrections = {"rfi_tc": corr_rfi, "ca_tc": corr_ca_tc, "ca_direct": corr_ca_dir}

        for thr in THRESHOLDS:
            cfg = dict(base_cfg)
            cfg["threshold_rfi"] = thr
            cfg["threshold_ca_tc"] = thr
            cfg["threshold_ca_direct"] = thr
            cfg["ca_direct_alpha"] = alpha

            log.info("Grid: thr=%.2f, alpha=%.1f", thr, alpha)
            _, metrics = evaluate_on_split(
                model, tok, rows, meta, val_idx, acts_obs,
                routers, cfg, corrections)

            entry = {
                "threshold": thr, "ca_direct_alpha": alpha,
                "accuracy": metrics["intervened_accuracy"],
                "delta_acc": metrics["delta_accuracy"],
                "macro_f1": metrics["intervened_f1"],
                "delta_f1": metrics["delta_f1"],
                "fixed": metrics["total_corrected"],
                "broke": metrics["total_broken"],
                "net": metrics["net_benefit"],
            }
            grid_results.append(entry)
            log.info("  -> acc=%.2f%%, f1=%.2f%%, net=%+d", 
                     entry["accuracy"], entry["macro_f1"], entry["net"])

    # Find best
    best = max(grid_results, key=lambda x: (x["net"], x["accuracy"]))
    
    out = {
        "sweep_on": "val", "n_val": len(val_idx),
        "grid": "threshold × ca_direct_alpha",
        "thresholds": THRESHOLDS, "alphas": ALPHAS,
        "selection_criterion": "max net_benefit, tiebreak by accuracy",
        "best": best,
        "results": grid_results,
    }
    with open(RESULT_DIR / "p0_joint_grid_val.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    log.info("Joint grid saved. Best: thr=%.2f, alpha=%.1f (net=%+d, acc=%.2f%%)",
             best["threshold"], best["ca_direct_alpha"], best["net"], best["accuracy"])

    elapsed = time.time() - t0
    log.info("Total time: %.1f min", elapsed / 60)

    # Print grid table
    print("\n" + "=" * 80)
    print("  JOINT GRID (val): threshold × ca_direct_alpha")
    print("=" * 80)
    print(f"  {'thr':>6} {'alpha':>6} {'acc%':>8} {'f1%':>8} {'net':>6} {'fixed':>6} {'broke':>6}")
    print("  " + "-" * 52)
    for r in sorted(grid_results, key=lambda x: (-x["net"], -x["accuracy"])):
        mark = " <-- BEST" if r == best else ""
        print(f"  {r['threshold']:6.2f} {r['ca_direct_alpha']:6.1f} {r['accuracy']:8.2f} "
              f"{r['macro_f1']:8.2f} {r['net']:+6d} {r['fixed']:6d} {r['broke']:6d}{mark}")
    print("=" * 80)

    current = (0.40, 10.0)
    if (best["threshold"], best["ca_direct_alpha"]) == current:
        print("\n  ✓ P0 locked config (0.40, 10.0) CONFIRMED by joint grid search.")
    else:
        print(f"\n  ⚠ Best combo changed to ({best['threshold']}, {best['ca_direct_alpha']})!")
        print("    Need to update locked config and re-run final test.")

if __name__ == "__main__":
    main()
