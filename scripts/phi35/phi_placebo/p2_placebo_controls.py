#!/usr/bin/env python3
"""
P2: Placebo Controls
====================

4 placebo conditions evaluated on P0 locked test split:
  1. RANDOM_DIR  — random Gaussian vectors, same L2 norm as real
  2. WRONG_LAYER — real vectors injected at wrong layers
  3. MISMATCHED_CHANNEL — real vectors cross-assigned to wrong channels
  4. REVERSE_DIR — real vectors with negated alpha (opposite direction)

All use the same train/test split and routers from P0.
"""

import gc, json, logging, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phi_mainline.run_v31 import (
    RESULT_DIR, SEED,
    LABEL_VOCAB, RFI2TC, CA2TC, CA2DIRECT,
    load_data, build_meta, load_model,
    collect_activations, train_pca_basis, compute_correction_vec,
    train_binary_router, cascade_route,
    predict_with_correction, compute_metrics,
)
from phi_mainline.p0_run_locked_eval import (
    evaluate_on_split, _train_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sakiko_p2")

SPLIT_DIR = RESULT_DIR / "splits"


def load_splits():
    """Load P0 locked splits from disk."""
    train_idx = json.loads((SPLIT_DIR / "train_idx.json").read_text())
    val_idx = json.loads((SPLIT_DIR / "val_idx.json").read_text())
    test_idx = json.loads((SPLIT_DIR / "test_idx.json").read_text())
    return train_idx, val_idx, test_idx


def load_locked_config():
    """Load P0 locked config from disk."""
    return json.loads((RESULT_DIR / "p0_locked_config.json").read_text())


def build_locked_cfg(locked):
    """Convert locked config dict to the cfg dict format expected by pipeline."""
    return {
        "obs_layer": locked["obs_layer"],
        "rfi_inj_layer": locked["rfi_inj_layer"],
        "rfi_inj_mode": locked["rfi_inj_mode"],
        "rfi_alpha": locked["rfi_alpha"],
        "threshold_rfi": locked["threshold_rfi"],
        "ca_tc_inj_layer": locked["ca_tc_inj_layer"],
        "ca_tc_inj_mode": locked["ca_tc_inj_mode"],
        "ca_tc_alpha": locked["ca_tc_alpha"],
        "threshold_ca_tc": locked["threshold_ca_tc"],
        "ca_direct_inj_layer": locked["ca_direct_inj_layer"],
        "ca_direct_inj_mode": locked["ca_direct_inj_mode"],
        "ca_direct_alpha": locked["ca_direct_alpha"],
        "threshold_ca_direct": locked["threshold_ca_direct"],
        "n_comp": locked["n_comp"],
        "n_pcs": locked["n_pcs"],
    }


def extract_metrics(metrics):
    """Extract the key metrics we report for each placebo."""
    return {
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


def make_random_vec_like(real_vec, rng):
    """Create a random Gaussian vector with the same L2 norm as real_vec."""
    rand = rng.standard_normal(real_vec.shape).astype(np.float32)
    rand_norm = np.linalg.norm(rand)
    real_norm = np.linalg.norm(real_vec)
    if rand_norm < 1e-12:
        return rand
    return rand * (real_norm / rand_norm)


# ═══════════════════════════════════════════════════════════════════════════
# Placebo 1: RANDOM_DIR
# ═══════════════════════════════════════════════════════════════════════════

def placebo_random_dir(model, tok, rows, meta, test_idx, acts_obs,
                       routers, cfg, real_corrections):
    """Replace each correction vector with a random direction of same L2 norm."""
    log.info("=" * 60)
    log.info("PLACEBO: RANDOM_DIR")
    log.info("=" * 60)

    rng = np.random.RandomState(SEED + 100)
    placebo_corrections = {}
    for ch_name, corr_info in real_corrections.items():
        real_vec = corr_info["vec"]
        rand_vec = make_random_vec_like(real_vec, rng)
        log.info("  %s: real L2=%.4f, random L2=%.4f",
                 ch_name, np.linalg.norm(real_vec), np.linalg.norm(rand_vec))
        placebo_corrections[ch_name] = {
            "vec": rand_vec,
            "layer": corr_info["layer"],
            "mode": corr_info["mode"],
        }

    results, metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs,
        routers, cfg, placebo_corrections)
    return extract_metrics(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# Placebo 2: WRONG_LAYER
# ═══════════════════════════════════════════════════════════════════════════

def placebo_wrong_layer(model, tok, rows, meta, test_idx, acts_obs,
                        routers, cfg, real_corrections):
    """Inject real correction vectors at wrong layers (far from optimal)."""
    log.info("=" * 60)
    log.info("PLACEBO: WRONG_LAYER")
    log.info("=" * 60)

    # Real layers: rfi_tc=14, ca_tc=16, ca_direct=16
    # Use layer 4 (very early) for all — should be ineffective
    wrong_layer = 4
    placebo_corrections = {}
    for ch_name, corr_info in real_corrections.items():
        log.info("  %s: real layer=%d, wrong layer=%d",
                 ch_name, corr_info["layer"], wrong_layer)
        placebo_corrections[ch_name] = {
            "vec": corr_info["vec"],
            "layer": wrong_layer,
            "mode": corr_info["mode"],
        }

    results, metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs,
        routers, cfg, placebo_corrections)
    return extract_metrics(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# Placebo 3: MISMATCHED_CHANNEL
# ═══════════════════════════════════════════════════════════════════════════

def placebo_mismatched_channel(model, tok, rows, meta, test_idx, acts_obs,
                               routers, cfg, real_corrections):
    """Cross-assign correction vectors to wrong channels."""
    log.info("=" * 60)
    log.info("PLACEBO: MISMATCHED_CHANNEL")
    log.info("=" * 60)

    # Rotation: rfi_tc gets ca_direct vec, ca_tc gets rfi_tc vec, ca_direct gets ca_tc vec
    mapping = {
        "rfi_tc": "ca_direct",
        "ca_tc": "rfi_tc",
        "ca_direct": "ca_tc",
    }

    placebo_corrections = {}
    for ch_name, corr_info in real_corrections.items():
        donor = mapping[ch_name]
        donor_info = real_corrections[donor]
        log.info("  %s: using vec from %s (layer=%d→%d)",
                 ch_name, donor, donor_info["layer"], corr_info["layer"])
        placebo_corrections[ch_name] = {
            "vec": donor_info["vec"],        # wrong vector
            "layer": corr_info["layer"],      # keep correct layer
            "mode": corr_info["mode"],        # keep correct mode
        }

    results, metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs,
        routers, cfg, placebo_corrections)
    return extract_metrics(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# Placebo 4: REVERSE_DIR
# ═══════════════════════════════════════════════════════════════════════════

def placebo_reverse_dir(model, tok, rows, meta, test_idx, acts_obs,
                        routers, cfg, real_corrections):
    """Negate all correction vectors (equivalent to negative alpha)."""
    log.info("=" * 60)
    log.info("PLACEBO: REVERSE_DIR")
    log.info("=" * 60)

    placebo_corrections = {}
    for ch_name, corr_info in real_corrections.items():
        log.info("  %s: reversing direction (L2=%.4f)",
                 ch_name, np.linalg.norm(corr_info["vec"]))
        placebo_corrections[ch_name] = {
            "vec": -corr_info["vec"],
            "layer": corr_info["layer"],
            "mode": corr_info["mode"],
        }

    results, metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs,
        routers, cfg, placebo_corrections)
    return extract_metrics(metrics)


# ═══════════════════════════════════════════════════════════════════════════
# Summary generation
# ═══════════════════════════════════════════════════════════════════════════

def write_summary_md(all_results, locked_config):
    """Write p2_placebo_summary.md."""
    real = all_results["REAL_LOCKED"]
    lines = []
    lines.append("# P2: Placebo Controls Summary\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d')}")
    lines.append(f"**Baseline config**: P0 locked (threshold=0.40, ca_direct_alpha=10.0)")
    lines.append(f"**Test set**: n={real.get('n_test', 548)} (P0 locked split)\n")

    lines.append("## Overview\n")
    lines.append("| Condition | acc% | Δacc | F1% | ΔF1 | fixed | broke | net |")
    lines.append("|---|---|---|---|---|---|---|---|")

    conditions = ["REAL_LOCKED", "RANDOM_DIR", "WRONG_LAYER", "MISMATCHED_CHANNEL", "REVERSE_DIR"]
    for cond in conditions:
        r = all_results[cond]
        mark = " **←baseline**" if cond == "REAL_LOCKED" else ""
        lines.append(f"| {cond} | {r['accuracy']:.2f} | {r['delta_acc']:+.2f} | "
                     f"{r['macro_f1']:.2f} | {r['delta_f1']:+.2f} | "
                     f"{r['fixed']} | {r['broke']} | {r['net']:+d} |{mark}")

    # Per-channel detail for each placebo
    lines.append("\n## Per-Channel Stats\n")
    for cond in conditions:
        r = all_results[cond]
        cs = r.get("channel_stats", {})
        if cs:
            lines.append(f"### {cond}\n")
            lines.append("| Channel | Routed | Fixed | Broke | Net | Precision |")
            lines.append("|---|---|---|---|---|---|")
            for ch, s in cs.items():
                lines.append(f"| {ch} | {s['n_routed']} | {s['fixed']} | "
                             f"{s['broke']} | {s['net']:+d} | {s['precision']:.1f}% |")
            lines.append("")

    # Analysis
    lines.append("## Analysis\n")

    real_net = real["net"]
    lines.append("### 1. Which placebos are near zero or harmful?\n")
    for cond in ["RANDOM_DIR", "WRONG_LAYER", "MISMATCHED_CHANNEL", "REVERSE_DIR"]:
        r = all_results[cond]
        pnet = r["net"]
        if pnet <= 0:
            verdict = f"**harmful** (net={pnet:+d})"
        elif pnet <= real_net * 0.3:
            verdict = f"**near zero** (net={pnet:+d}, only {pnet/real_net*100:.0f}% of real)"
        else:
            verdict = f"**unexpectedly effective** (net={pnet:+d}, {pnet/real_net*100:.0f}% of real)"
        lines.append(f"- **{cond}**: {verdict}")

    lines.append(f"\n### 2. Does correction direction show specificity?\n")
    random_net = all_results["RANDOM_DIR"]["net"]
    reverse_net = all_results["REVERSE_DIR"]["net"]
    mismatch_net = all_results["MISMATCHED_CHANNEL"]["net"]
    if real_net > random_net and real_net > reverse_net and real_net > mismatch_net:
        lines.append("**Yes.** The real correction direction is strictly superior to:")
        lines.append(f"- Random direction (net {real_net:+d} vs {random_net:+d})")
        lines.append(f"- Reversed direction (net {real_net:+d} vs {reverse_net:+d})")
        lines.append(f"- Mismatched channels (net {real_net:+d} vs {mismatch_net:+d})")
    else:
        lines.append("**Partial.** Not all placebos are clearly defeated:")
        for cond, pnet in [("RANDOM_DIR", random_net), ("REVERSE_DIR", reverse_net),
                           ("MISMATCHED_CHANNEL", mismatch_net)]:
            cmp = ">" if real_net > pnet else "≤"
            lines.append(f"- vs {cond}: real {real_net:+d} {cmp} placebo {pnet:+d}")

    lines.append(f"\n### 3. Does the method pass the basic causal control check?\n")
    wrong_net = all_results["WRONG_LAYER"]["net"]
    all_placebos_weaker = all(all_results[c]["net"] < real_net
                              for c in ["RANDOM_DIR", "WRONG_LAYER",
                                        "MISMATCHED_CHANNEL", "REVERSE_DIR"])
    any_placebo_harmful = any(all_results[c]["net"] <= 0
                              for c in ["RANDOM_DIR", "WRONG_LAYER",
                                        "MISMATCHED_CHANNEL", "REVERSE_DIR"])

    if all_placebos_weaker:
        lines.append(f"**✅ YES.** All 4 placebos produce strictly lower net benefit than the "
                     f"real locked config (net={real_net:+d}).")
        if any_placebo_harmful:
            lines.append("Some placebos are actively harmful (net ≤ 0), further confirming "
                         "that the real correction direction is non-trivial.")
        lines.append("\nThe correction vectors are not replaceable by random noise, wrong-layer "
                     "injection, cross-channel assignment, or reversed direction. "
                     "This provides evidence that the learned direction captures a meaningful "
                     "error-type-specific signal.")
    else:
        lines.append(f"**⚠ PARTIAL.** Not all placebos are strictly weaker. ")
        lines.append("Further investigation is needed to understand which controls fail.")

    lines.append("\n## Conclusion\n")
    lines.append(f"- Real locked config: **acc={real['accuracy']:.2f}%, net={real_net:+d}**")
    best_placebo = max(
        [(c, all_results[c]["net"]) for c in
         ["RANDOM_DIR", "WRONG_LAYER", "MISMATCHED_CHANNEL", "REVERSE_DIR"]],
        key=lambda x: x[1])
    lines.append(f"- Best placebo ({best_placebo[0]}): **net={best_placebo[1]:+d}**")
    gap = real_net - best_placebo[1]
    lines.append(f"- Gap: **{gap:+d}** net benefit")

    with open(RESULT_DIR / "p2_placebo_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Summary saved to p2_placebo_summary.md")


def main():
    t0 = time.time()

    log.info("=" * 60)
    log.info("P2: Placebo Controls")
    log.info("=" * 60)

    # Load everything
    rows = load_data()
    meta = build_meta(rows)
    model, tok = load_model()
    locked = load_locked_config()
    cfg = build_locked_cfg(locked)
    train_idx, _, test_idx = load_splits()

    acts_obs = collect_activations(model, tok, rows, cfg["obs_layer"])

    # Train real pipeline on train set
    log.info("Training real pipeline on train set...")
    routers, real_corrections = _train_pipeline(acts_obs, meta, train_idx, cfg)

    # Log real correction norms
    for ch, c in real_corrections.items():
        log.info("  Real %s: L2=%.4f, layer=%d, mode=%s",
                 ch, np.linalg.norm(c["vec"]), c["layer"], c["mode"])

    # First: evaluate real locked config (sanity check — should match P0)
    log.info("Evaluating REAL locked config on test...")
    real_results, real_metrics = evaluate_on_split(
        model, tok, rows, meta, test_idx, acts_obs,
        routers, cfg, real_corrections)
    real_summary = extract_metrics(real_metrics)
    real_summary["n_test"] = len(test_idx)
    log.info("REAL: acc=%.2f%%, net=%+d", real_summary["accuracy"], real_summary["net"])

    # Sanity check vs P0
    p0 = json.loads((RESULT_DIR / "p0_final_test_eval.json").read_text())
    assert abs(real_summary["accuracy"] - p0["intervened_accuracy"]) < 0.01, \
        f"Sanity fail: real={real_summary['accuracy']}, p0={p0['intervened_accuracy']}"
    log.info("✓ Sanity check passed: matches P0 final test eval")

    all_results = {"REAL_LOCKED": real_summary}

    # Placebo 1: RANDOM_DIR
    all_results["RANDOM_DIR"] = placebo_random_dir(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, real_corrections)
    log.info("RANDOM_DIR: acc=%.2f%%, net=%+d",
             all_results["RANDOM_DIR"]["accuracy"], all_results["RANDOM_DIR"]["net"])

    # Placebo 2: WRONG_LAYER
    all_results["WRONG_LAYER"] = placebo_wrong_layer(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, real_corrections)
    log.info("WRONG_LAYER: acc=%.2f%%, net=%+d",
             all_results["WRONG_LAYER"]["accuracy"], all_results["WRONG_LAYER"]["net"])

    # Placebo 3: MISMATCHED_CHANNEL
    all_results["MISMATCHED_CHANNEL"] = placebo_mismatched_channel(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, real_corrections)
    log.info("MISMATCHED_CHANNEL: acc=%.2f%%, net=%+d",
             all_results["MISMATCHED_CHANNEL"]["accuracy"], all_results["MISMATCHED_CHANNEL"]["net"])

    # Placebo 4: REVERSE_DIR
    all_results["REVERSE_DIR"] = placebo_reverse_dir(
        model, tok, rows, meta, test_idx, acts_obs, routers, cfg, real_corrections)
    log.info("REVERSE_DIR: acc=%.2f%%, net=%+d",
             all_results["REVERSE_DIR"]["accuracy"], all_results["REVERSE_DIR"]["net"])

    # Save JSON
    with open(RESULT_DIR / "p2_placebo_controls.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: int(o) if hasattr(o, "item") else o)
    log.info("Results saved to p2_placebo_controls.json")

    # Write summary MD
    write_summary_md(all_results, locked)

    elapsed = time.time() - t0
    log.info("P2 complete. Total time: %.1f min", elapsed / 60)

    # Print summary table
    print("\n" + "=" * 80)
    print("  P2 PLACEBO CONTROLS — SUMMARY")
    print("=" * 80)
    print(f"  {'Condition':<22} {'acc%':>8} {'Δacc':>8} {'F1%':>8} {'net':>6}")
    print("  " + "-" * 56)
    for cond in ["REAL_LOCKED", "RANDOM_DIR", "WRONG_LAYER",
                 "MISMATCHED_CHANNEL", "REVERSE_DIR"]:
        r = all_results[cond]
        mark = " ← real" if cond == "REAL_LOCKED" else ""
        print(f"  {cond:<22} {r['accuracy']:8.2f} {r['delta_acc']:+8.2f} "
              f"{r['macro_f1']:8.2f} {r['net']:+6d}{mark}")
    print("=" * 80)


if __name__ == "__main__":
    main()
