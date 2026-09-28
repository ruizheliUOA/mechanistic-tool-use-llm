#!/usr/bin/env python3
"""
P1: Multi-Seed Stability
=========================

对 5 个不同 seed，各自完整执行 P0 clean pipeline：
  1. 三分 split (70/15/15)
  2. val threshold sweep
  3. val alpha sweep
  4. lock config
  5. final locked test (一次性)

然后汇总 mean / std / min / max，判断方法稳定性。

Seeds: 42, 123, 456, 789, 2024
"""

import csv, gc, json, logging, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sakiko_v3.run_v31 import (
    RESULT_DIR, SEED,
    LABEL_VOCAB, RFI2TC, CA2TC, CA2DIRECT,
    load_data, build_meta, load_model,
    collect_activations, train_pca_basis, compute_correction_vec,
    train_binary_router, cascade_route,
    predict_with_correction, compute_metrics, print_report,
)
from sakiko_v3.p0_run_locked_eval import (
    three_way_split, evaluate_on_split, _train_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sakiko_p1")

SEEDS = [42, 123, 456, 789, 2024]

BASE_CFG = {
    "obs_layer": 18,
    "rfi_inj_layer": 14, "rfi_inj_mode": "mlp_all", "rfi_alpha": 5.0,
    "ca_tc_inj_layer": 16, "ca_tc_inj_mode": "mlp_prompt", "ca_tc_alpha": 2.0,
    "ca_direct_inj_layer": 16, "ca_direct_inj_mode": "mlp_prompt", "ca_direct_alpha": 2.0,
    "n_comp": 64, "n_pcs": 20,
    "threshold_rfi": 0.60, "threshold_ca_tc": 0.60, "threshold_ca_direct": 0.60,
}

THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
ALPHAS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def run_threshold_sweep_silent(model, tok, rows, meta, val_idx, acts_obs,
                               train_idx, base_cfg):
    """Threshold sweep on val, returns (best_threshold, routers)."""
    routers, corrections = _train_pipeline(acts_obs, meta, train_idx, base_cfg)
    best_thr, best_net, best_acc = None, -9999, -9999
    for thr in THRESHOLDS:
        cfg = dict(base_cfg)
        cfg["threshold_rfi"] = thr
        cfg["threshold_ca_tc"] = thr
        cfg["threshold_ca_direct"] = thr
        _, metrics = evaluate_on_split(
            model, tok, rows, meta, val_idx, acts_obs, routers, cfg, corrections)
        net = metrics["net_benefit"]
        acc = metrics["intervened_accuracy"]
        log.info("    thr=%.2f → val net=%+d, acc=%.2f%%", thr, net, acc)
        if (net, acc) > (best_net, best_acc):
            best_net, best_acc, best_thr = net, acc, thr
    log.info("  Best threshold: %.2f (val net=%+d)", best_thr, best_net)
    return best_thr, routers


def run_alpha_sweep_silent(model, tok, rows, meta, val_idx, acts_obs,
                           train_idx, base_cfg, locked_threshold, routers):
    """Alpha sweep on val, returns best_alpha."""
    basis_rfi = train_pca_basis(acts_obs, meta, train_idx, RFI2TC,
                                base_cfg["n_comp"], ref_gold="tool_call")
    basis_ca_tc = train_pca_basis(acts_obs, meta, train_idx, CA2TC,
                                  base_cfg["n_comp"], ref_gold="tool_call")
    basis_ca_dir = train_pca_basis(acts_obs, meta, train_idx, CA2DIRECT,
                                   base_cfg["n_comp"], ref_gold="cannot_answer")

    best_alpha, best_net, best_acc = None, -9999, -9999
    for alpha in ALPHAS:
        cfg = dict(base_cfg)
        cfg["threshold_rfi"] = locked_threshold
        cfg["threshold_ca_tc"] = locked_threshold
        cfg["threshold_ca_direct"] = locked_threshold
        cfg["ca_direct_alpha"] = alpha

        corrections = {}
        if basis_rfi is not None:
            corrections["rfi_tc"] = {
                "vec": compute_correction_vec(basis_rfi, cfg["n_pcs"], cfg["rfi_alpha"]),
                "layer": cfg["rfi_inj_layer"], "mode": cfg["rfi_inj_mode"],
            }
        if basis_ca_tc is not None:
            corrections["ca_tc"] = {
                "vec": compute_correction_vec(basis_ca_tc, cfg["n_pcs"], cfg["ca_tc_alpha"]),
                "layer": cfg["ca_tc_inj_layer"], "mode": cfg["ca_tc_inj_mode"],
            }
        if basis_ca_dir is not None:
            corrections["ca_direct"] = {
                "vec": compute_correction_vec(basis_ca_dir, cfg["n_pcs"], alpha),
                "layer": cfg["ca_direct_inj_layer"], "mode": cfg["ca_direct_inj_mode"],
            }

        _, metrics = evaluate_on_split(
            model, tok, rows, meta, val_idx, acts_obs, routers, cfg, corrections)
        net = metrics["net_benefit"]
        acc = metrics["intervened_accuracy"]
        log.info("    alpha=%.1f → val net=%+d, acc=%.2f%%", alpha, net, acc)
        if (net, acc) > (best_net, best_acc):
            best_net, best_acc, best_alpha = net, acc, alpha

    log.info("  Best alpha: %.1f (val net=%+d)", best_alpha, best_net)
    return best_alpha


def run_final_test(model, tok, rows, meta, test_idx, acts_obs,
                   train_idx, locked_thr, locked_alpha):
    """Final locked test for one seed. Returns metrics dict."""
    cfg = dict(BASE_CFG)
    cfg["threshold_rfi"] = locked_thr
    cfg["threshold_ca_tc"] = locked_thr
    cfg["threshold_ca_direct"] = locked_thr
    cfg["ca_direct_alpha"] = locked_alpha

    routers, corrections = _train_pipeline(acts_obs, meta, train_idx, cfg)

    results, metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, corrections)
    return metrics


def run_one_seed(seed, model, tok, rows, meta, acts_obs):
    """Full clean pipeline for one seed. Returns result dict."""
    log.info("=" * 70)
    log.info("SEED %d: Starting full clean pipeline", seed)
    log.info("=" * 70)

    t0 = time.time()

    # Step 1: Split
    train_idx, val_idx, test_idx = three_way_split(meta, seed=seed)
    log.info("  Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Step 2: Val threshold sweep
    log.info("  Step 2: Threshold sweep on val...")
    locked_thr, routers = run_threshold_sweep_silent(
        model, tok, rows, meta, val_idx, acts_obs, train_idx, BASE_CFG)

    # Step 3: Val alpha sweep
    log.info("  Step 3: Alpha sweep on val...")
    locked_alpha = run_alpha_sweep_silent(
        model, tok, rows, meta, val_idx, acts_obs, train_idx,
        BASE_CFG, locked_thr, routers)

    # Step 4+5: Final locked test
    log.info("  Step 5: Final locked test (thr=%.2f, alpha=%.1f)...", locked_thr, locked_alpha)
    metrics = run_final_test(
        model, tok, rows, meta, test_idx, acts_obs, train_idx,
        locked_thr, locked_alpha)

    elapsed = time.time() - t0

    result = {
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "locked_threshold": locked_thr,
        "locked_alpha": locked_alpha,
        "baseline_accuracy": metrics["baseline_accuracy"],
        "baseline_f1": metrics["baseline_f1"],
        "accuracy": metrics["intervened_accuracy"],
        "delta_acc": metrics["delta_accuracy"],
        "macro_f1": metrics["intervened_f1"],
        "delta_f1": metrics["delta_f1"],
        "fixed": metrics["total_corrected"],
        "broke": metrics["total_broken"],
        "net": metrics["net_benefit"],
        "channel_stats": metrics.get("channel_stats", {}),
        "clean_damage": metrics.get("clean_damage", {}),
        "elapsed_min": round(elapsed / 60, 1),
    }

    # Flag negative per-channel net
    for ch, cs in result["channel_stats"].items():
        if cs["net"] < 0:
            log.warning("  ⚠ SEED %d: channel %s has NEGATIVE net=%d", seed, ch, cs["net"])

    log.info("  SEED %d DONE: acc=%.2f%% (Δ%+.2f%%), net=%+d, thr=%.2f, α=%.1f [%.1f min]",
             seed, result["accuracy"], result["delta_acc"], result["net"],
             locked_thr, locked_alpha, result["elapsed_min"])
    return result


def compute_summary_stats(all_results):
    """Compute mean/std/min/max across seeds."""
    keys = ["accuracy", "delta_acc", "macro_f1", "delta_f1", "fixed", "broke", "net",
            "baseline_accuracy", "baseline_f1"]
    stats = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        stats[k] = {
            "mean": round(np.mean(vals), 2),
            "std": round(np.std(vals, ddof=1), 2) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "values": [round(v, 2) for v in vals],
        }

    # Per-channel stats
    channels = ["rfi_tc", "ca_tc", "ca_direct"]
    ch_stats = {}
    for ch in channels:
        nets = [r["channel_stats"].get(ch, {}).get("net", 0) for r in all_results]
        fixeds = [r["channel_stats"].get(ch, {}).get("fixed", 0) for r in all_results]
        brokes = [r["channel_stats"].get(ch, {}).get("broke", 0) for r in all_results]
        n_positive = sum(1 for n in nets if n > 0)
        ch_stats[ch] = {
            "net_mean": round(np.mean(nets), 2),
            "net_std": round(np.std(nets, ddof=1), 2) if len(nets) > 1 else 0.0,
            "net_min": min(nets),
            "net_max": max(nets),
            "net_values": nets,
            "fixed_mean": round(np.mean(fixeds), 2),
            "broke_mean": round(np.mean(brokes), 2),
            "n_positive": n_positive,
            "n_seeds": len(all_results),
        }
    stats["channel_stats"] = ch_stats
    return stats


def write_csv(all_results):
    """Write p1_multiseed_table.csv."""
    fieldnames = [
        "seed", "n_train", "n_val", "n_test",
        "locked_threshold", "locked_alpha",
        "baseline_accuracy", "accuracy", "delta_acc",
        "baseline_f1", "macro_f1", "delta_f1",
        "fixed", "broke", "net",
        "rfi_tc_net", "ca_tc_net", "ca_direct_net",
    ]
    rows_out = []
    for r in all_results:
        row = {k: r.get(k, "") for k in fieldnames[:15]}
        row["rfi_tc_net"] = r["channel_stats"].get("rfi_tc", {}).get("net", "")
        row["ca_tc_net"] = r["channel_stats"].get("ca_tc", {}).get("net", "")
        row["ca_direct_net"] = r["channel_stats"].get("ca_direct", {}).get("net", "")
        rows_out.append(row)

    with open(RESULT_DIR / "p1_multiseed_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    log.info("CSV saved to p1_multiseed_table.csv")


def write_summary_md(all_results, stats):
    """Write p1_multiseed_summary.md."""
    lines = []
    lines.append("# P1: Multi-Seed Stability Report\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d')}")
    lines.append(f"**Seeds**: {', '.join(str(s) for s in SEEDS)}")
    lines.append(f"**Method**: Full clean pipeline per seed (split → val sweep → lock → test)\n")

    # 1. Per-seed table
    lines.append("## 1. Per-Seed Results\n")
    lines.append("| seed | thr | α | baseline | acc% | Δacc | F1% | ΔF1 | fixed | broke | net | rfi_tc | ca_tc | ca_dir |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        cs = r["channel_stats"]
        rfi_net = cs.get("rfi_tc", {}).get("net", "—")
        ca_tc_net = cs.get("ca_tc", {}).get("net", "—")
        ca_dir_net = cs.get("ca_direct", {}).get("net", "—")
        # Flag negative channels
        rfi_s = f"**{rfi_net}**⚠" if isinstance(rfi_net, int) and rfi_net < 0 else str(rfi_net)
        cat_s = f"**{ca_tc_net}**⚠" if isinstance(ca_tc_net, int) and ca_tc_net < 0 else str(ca_tc_net)
        cad_s = f"**{ca_dir_net}**⚠" if isinstance(ca_dir_net, int) and ca_dir_net < 0 else str(ca_dir_net)
        lines.append(f"| {r['seed']} | {r['locked_threshold']:.2f} | {r['locked_alpha']:.1f} | "
                     f"{r['baseline_accuracy']:.2f} | {r['accuracy']:.2f} | {r['delta_acc']:+.2f} | "
                     f"{r['macro_f1']:.2f} | {r['delta_f1']:+.2f} | "
                     f"{r['fixed']} | {r['broke']} | {r['net']:+d} | "
                     f"{rfi_s} | {cat_s} | {cad_s} |")

    # 2. Summary statistics
    lines.append("\n## 2. Summary Statistics\n")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    for k in ["accuracy", "delta_acc", "macro_f1", "delta_f1", "net", "fixed", "broke"]:
        s = stats[k]
        lines.append(f"| {k} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | {s['max']:.2f} |")

    # 3. Channel stability
    lines.append("\n## 3. Channel-Level Stability\n")
    lines.append("| Channel | Mean net | Std | Min | Max | #Positive / #Seeds |")
    lines.append("|---|---|---|---|---|---|")
    for ch in ["rfi_tc", "ca_tc", "ca_direct"]:
        cs = stats["channel_stats"][ch]
        lines.append(f"| {ch} | {cs['net_mean']:+.1f} | {cs['net_std']:.1f} | "
                     f"{cs['net_min']:+d} | {cs['net_max']:+d} | "
                     f"{cs['n_positive']}/{cs['n_seeds']} |")

    # 4. Conclusion
    lines.append("\n## 4. Conclusion\n")

    # Check pass criteria
    all_net_positive = all(r["net"] > 0 for r in all_results)
    all_delta_positive = all(r["delta_acc"] > 0 for r in all_results)
    mean_net = stats["net"]["mean"]
    std_net = stats["net"]["std"]
    ch_positive_counts = {ch: stats["channel_stats"][ch]["n_positive"] for ch in ["rfi_tc", "ca_tc", "ca_direct"]}
    channels_mostly_positive = sum(1 for v in ch_positive_counts.values() if v >= 3)

    lines.append("### Pass Criteria Check\n")
    lines.append(f"- 5/5 seeds overall net > 0: {'✅' if all_net_positive else '❌'} "
                 f"({sum(1 for r in all_results if r['net'] > 0)}/5)")
    lines.append(f"- 5/5 seeds delta_acc > 0: {'✅' if all_delta_positive else '❌'} "
                 f"({sum(1 for r in all_results if r['delta_acc'] > 0)}/5)")
    lines.append(f"- Mean net clearly positive: {'✅' if mean_net > 2 * std_net else '⚠'} "
                 f"(mean={mean_net:+.1f}, std={std_net:.1f})")
    lines.append(f"- ≥2 channels mostly positive: {'✅' if channels_mostly_positive >= 2 else '❌'} "
                 f"({channels_mostly_positive}/3 channels)")

    # Stability ranking
    ch_stability = sorted(stats["channel_stats"].items(),
                          key=lambda x: x[1]["net_mean"] / max(x[1]["net_std"], 0.01),
                          reverse=True)
    lines.append(f"\n### Stability Ranking (mean/std ratio)\n")
    for i, (ch, cs) in enumerate(ch_stability):
        ratio = cs["net_mean"] / max(cs["net_std"], 0.01)
        label = "most stable" if i == 0 else ("least stable" if i == len(ch_stability) - 1 else "")
        lines.append(f"- **{ch}**: mean={cs['net_mean']:+.1f}, std={cs['net_std']:.1f}, "
                     f"ratio={ratio:.2f} {f'← {label}' if label else ''}")

    # Overall verdict
    passed = all_net_positive and all_delta_positive and channels_mostly_positive >= 2
    lines.append(f"\n### Overall Verdict\n")
    if passed:
        lines.append(f"**✅ P1 PASSED.** The method is stable across {len(SEEDS)} seeds.")
        lines.append(f"\nStrongest claim upgrade: "
                     f"\"SAKIKO V3.1 achieves Δacc={stats['delta_acc']['mean']:+.2f}% "
                     f"± {stats['delta_acc']['std']:.2f}% and net={stats['net']['mean']:+.1f} "
                     f"± {stats['net']['std']:.1f} across {len(SEEDS)} independent splits "
                     f"(seeds: {', '.join(str(s) for s in SEEDS)}).\"")
        lines.append(f"\n**Recommendation**: Ready to proceed to P3 cross-model.")
    else:
        lines.append(f"**⚠ P1 PARTIAL.** Some stability criteria not met.")
        if not all_net_positive:
            lines.append(f"- Some seeds have net ≤ 0")
        if not all_delta_positive:
            lines.append(f"- Some seeds have delta_acc ≤ 0")
        lines.append(f"\n**Recommendation**: Investigate unstable seeds before proceeding.")

    with open(RESULT_DIR / "p1_multiseed_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Summary saved to p1_multiseed_summary.md")


def main():
    t0 = time.time()

    log.info("=" * 70)
    log.info("P1: Multi-Seed Stability (%d seeds)", len(SEEDS))
    log.info("=" * 70)

    rows = load_data()
    meta = build_meta(rows)
    model, tok = load_model()
    acts_obs = collect_activations(model, tok, rows, BASE_CFG["obs_layer"])

    all_results = []
    for seed in SEEDS:
        result = run_one_seed(seed, model, tok, rows, meta, acts_obs)
        all_results.append(result)
        gc.collect(); torch.cuda.empty_cache()

    # Compute summary
    stats = compute_summary_stats(all_results)

    # Save all results JSON
    out = {
        "experiment": "P1 multi-seed stability",
        "seeds": SEEDS,
        "base_config": BASE_CFG,
        "threshold_grid": THRESHOLDS,
        "alpha_grid": ALPHAS,
        "per_seed_results": all_results,
        "summary_stats": stats,
    }
    with open(RESULT_DIR / "p1_multiseed_all_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False,
                  default=lambda o: int(o) if hasattr(o, "item") else float(o))
    log.info("All results saved to p1_multiseed_all_results.json")

    # Write CSV
    write_csv(all_results)

    # Write summary MD
    write_summary_md(all_results, stats)

    elapsed = time.time() - t0
    log.info("P1 complete. Total time: %.1f min", elapsed / 60)

    # Print final table
    print("\n" + "=" * 90)
    print("  P1 MULTI-SEED STABILITY — FINAL SUMMARY")
    print("=" * 90)
    print(f"  {'seed':>6} {'thr':>5} {'α':>5} {'acc%':>8} {'Δacc':>8} {'F1%':>8} {'net':>6} "
          f"{'rfi':>5} {'ca_tc':>6} {'ca_d':>5}")
    print("  " + "-" * 76)
    for r in all_results:
        cs = r["channel_stats"]
        print(f"  {r['seed']:6d} {r['locked_threshold']:5.2f} {r['locked_alpha']:5.1f} "
              f"{r['accuracy']:8.2f} {r['delta_acc']:+8.2f} {r['macro_f1']:8.2f} "
              f"{r['net']:+6d} {cs.get('rfi_tc',{}).get('net',0):+5d} "
              f"{cs.get('ca_tc',{}).get('net',0):+6d} "
              f"{cs.get('ca_direct',{}).get('net',0):+5d}")
    print("  " + "-" * 76)
    s = stats
    print(f"  {'MEAN':>6} {'':>5} {'':>5} {s['accuracy']['mean']:8.2f} "
          f"{s['delta_acc']['mean']:+8.2f} {s['macro_f1']['mean']:8.2f} "
          f"{s['net']['mean']:+6.0f} "
          f"{s['channel_stats']['rfi_tc']['net_mean']:+5.0f} "
          f"{s['channel_stats']['ca_tc']['net_mean']:+6.0f} "
          f"{s['channel_stats']['ca_direct']['net_mean']:+5.0f}")
    print(f"  {'STD':>6} {'':>5} {'':>5} {s['accuracy']['std']:8.2f} "
          f"{s['delta_acc']['std']:+8.2f} {s['macro_f1']['std']:8.2f} "
          f"{s['net']['std']:6.0f} "
          f"{s['channel_stats']['rfi_tc']['net_std']:5.0f} "
          f"{s['channel_stats']['ca_tc']['net_std']:6.0f} "
          f"{s['channel_stats']['ca_direct']['net_std']:5.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
