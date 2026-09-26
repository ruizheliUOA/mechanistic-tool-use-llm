"""
make_cross_dataset_figures.py — CPU-only figures for the Phase-1 cross-dataset package.
Every figure states a claim backed by a committed table; no decorative plots.
Usage: python scripts/make_cross_dataset_figures.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "final/results/cross_dataset_channel_discovery"
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PILOT_SUM = D / "new_benchmark_pilot/acebench_pilot_baseline_summary.json"
PILOT_DISC = D / "new_benchmark_pilot/acebench_channel_discovery.json"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def heatmap(conf, gold_order, pred_order, title, name, n_note=""):
    M = np.array([[conf[g].get(p, 0) for p in pred_order] for g in gold_order], dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(pred_order) + 2, 1.0 * len(gold_order) + 1.5))
    im = ax.imshow(M, cmap="Blues")
    for i in range(len(gold_order)):
        for j in range(len(pred_order)):
            v = int(M[i, j])
            err = gold_order[i] != pred_order[j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color=("crimson" if err and v > 0 else
                           ("white" if M[i, j] > M.max() * 0.6 else "black")),
                    fontweight="bold" if err and v > 0 else "normal")
    ax.set_xticks(range(len(pred_order)), pred_order, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(gold_order)), gold_order, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title(f"{title}\n(error cells in red{n_note})", fontsize=10)
    fig.colorbar(im, shrink=0.8)
    save(fig, name)


def main():
    qwen = json.load(open(D / "w2c_qwen_channel_discovery.json"))
    phi = json.load(open(D / "w2c_phi_channel_discovery.json"))
    meta = json.load(open(D / "metatool_channel_discovery.json"))

    # ── fig 2/3/4 (+ ACEBench): error-transition heatmaps ──
    heatmap(qwen["confusion_matrix_gold_by_pred"], qwen["native_gold_labels"],
            qwen["native_pred_labels"], "W2C / Qwen2.5-7B baseline confusion (n=3652)",
            "fig_confusion_w2c_qwen")
    heatmap(phi["confusion_matrix_gold_by_pred"], phi["native_gold_labels"],
            phi["native_pred_labels"], "W2C / Phi-3.5 baseline confusion (test split, n=548)",
            "fig_confusion_w2c_phi")
    heatmap(meta["confusion_matrix_gold_by_pred"], meta["native_gold_labels"],
            meta["native_pred_labels"], "MetaTool-Binary / Phi-3.5 baseline confusion (n=1040)",
            "fig_confusion_metatool")

    datasets = [("W2C\nQwen2.5-7B", qwen), ("W2C\nPhi-3.5*", phi), ("MetaTool\nPhi-3.5", meta)]
    ace = None
    if PILOT_SUM.exists() and PILOT_DISC.exists():
        ace = json.load(open(PILOT_DISC))
        s = json.load(open(PILOT_SUM))
        heatmap(ace["confusion_matrix_gold_by_pred"], ace["native_gold_labels"],
                ace["native_pred_labels"],
                f"ACEBench-decision / Qwen2.5-7B baseline confusion (n={ace['n_samples']})",
                "fig_confusion_acebench")
        datasets.append(("ACEBench\nQwen2.5-7B", ace))

    # ── fig 1: label-space size vs stable channel count ──
    # respect the R0 baseline-validity gate: an invalid readout claims no channels
    def valid(r):
        return not r.get("baseline_validity", {}).get("readout_collapse_flag", False)
    names = [(n if valid(r) else n + "\n(readout invalid)") for n, r in datasets] + \
            ["ToolSandbox\n(no labels)"]
    n_labels = [len(r["native_gold_labels"]) for _, r in datasets] + [0]
    n_stable = [(len(r["stable_channels_default"]) if valid(r) else 0)
                for _, r in datasets] + [0]
    n_arch = [sum(1 for c in r["discovered_channels"]
                  if c["status_default"] == "stable" and c["in_archived_set"]) if valid(r) else 0
              for _, r in datasets] + [0]
    x = np.arange(len(names)); w = 0.28
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(x - w, n_labels, w, label="native gold classes", color="#888")
    ax.bar(x, n_stable, w, label="stable channels (default thresholds)", color="#2b6cb0")
    ax.bar(x + w, n_arch, w, label="of which archived/hand-selected", color="#c05621")
    for xi, v in zip(x, n_stable):
        ax.text(xi, v + 0.05, str(v), ha="center", fontsize=9)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Channel count is dataset-derived, not fixed\n"
                  "(*Phi = test-split-exact; ToolSandbox = zero-channel boundary case)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig_labelspace_vs_channels")

    # ── fig 5: channel-count stability under threshold settings ──
    fig, ax = plt.subplots(figsize=(7.5, 4))
    settings = ["strict", "default", "lenient"]
    for label, r in datasets:
        ns = [r["n_stable_by_threshold_setting"][s] for s in settings]
        ax.plot(settings, ns, marker="o", label=label.replace("\n", " "))
    ax.set_ylabel("# stable channels"); ax.set_xlabel("threshold setting")
    ax.set_title("Channel-count sensitivity to selection thresholds", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig_threshold_stability")

    # ── fig 6: candidate benchmark comparison ──
    rows = list(csv.DictReader(open(D / "NEW_BENCHMARK_CANDIDATE_MATRIX.csv")))
    cats, labs, colors = [], [], []
    cmap = {"SELECTED": "#2f855a", "reject": "#c53030", "reject_for_phase2": "#dd6b20",
            "watch_list": "#888"}
    for r in rows:
        rec = r["recommendation"].split()[0].split("_(")[0]
        key = ("SELECTED" if rec.startswith("SELECTED") else
               "watch_list" if rec.startswith("watch") else
               "reject_for_phase2" if "phase2" in rec else "reject")
        cats.append(r["dataset"].replace("_", "\n", 1))
        try:
            labs.append(int(r["number_of_labels"].split()[0]))
        except ValueError:
            labs.append(0)
        colors.append(cmap[key])
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.bar(range(len(cats)), labs, color=colors)
    ax.set_xticks(range(len(cats)), cats, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("# native decision-level labels")
    ax.set_title("Benchmark candidates: native decision-label count\n"
                  "green=selected · orange=reject-for-phase2 · red=reject · grey=unverified",
                  fontsize=10)
    save(fig, "fig_benchmark_candidates")

    # manifest
    with open(FIG / "figure_manifest.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["figure", "claim", "data_source"])
        wr.writerow(["fig_confusion_w2c_qwen", "W2C/Qwen dominant error transitions = archived 3 channels",
                     "w2c_qwen_channel_discovery.json"])
        wr.writerow(["fig_confusion_w2c_phi", "Phi shows the same dominant W2C transitions (test-exact)",
                     "w2c_phi_channel_discovery.json"])
        wr.writerow(["fig_confusion_metatool", "binary label space -> 2 error transitions only",
                     "metatool_channel_discovery.json"])
        if ace:
            wr.writerow(["fig_confusion_acebench", "new 4-mode label space -> new transition structure",
                         "new_benchmark_pilot/acebench_channel_discovery.json"])
        wr.writerow(["fig_labelspace_vs_channels", "channel count is dataset-derived (5/3/2/0 pattern)",
                     "channel_discovery_comparison.csv"])
        wr.writerow(["fig_threshold_stability", "stable-channel count under strict/default/lenient",
                     "threshold_sensitivity.csv"])
        wr.writerow(["fig_benchmark_candidates", "candidate ranking by native label count",
                     "NEW_BENCHMARK_CANDIDATE_MATRIX.csv"])


if __name__ == "__main__":
    main()
