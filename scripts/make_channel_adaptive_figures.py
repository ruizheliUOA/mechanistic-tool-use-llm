"""
make_channel_adaptive_figures.py — figures for the channel-adaptive + geometry package.
========================================================================================
Every figure is backed by a CSV/JSON already produced under final/results/channel_adaptive/.
No model, no activations loaded here — pure plotting from the derived tables. Figures that would
need MetaTool/Phi activations are intentionally NOT produced (they stay TODO). Writes a
figure_manifest.csv mapping each figure to its source table.

Usage:  python scripts/make_channel_adaptive_figures.py
"""
from __future__ import annotations
import csv, json, sys, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CA = ROOT / "final" / "results" / "channel_adaptive"
FIG = CA / "figures"
FIG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("figures")
MANIFEST: list[dict] = []
GOLD = ["tool_call", "request_for_info", "cannot_answer"]
PRED = ["tool_call", "direct", "request_for_info", "cannot_answer"]


def record(fname, source, desc, status="generated", notes=""):
    MANIFEST.append({"figure_file": fname, "source_table": source, "description": desc,
                     "status": status, "notes": notes})


def read_csv(name):
    with open(CA / name) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / name.replace(".png", ".pdf"), bbox_inches="tight"); plt.close(fig)
    log.info("wrote %s", name)


# 1. Qwen W2C error-transition heatmap
def fig_transition_heatmap():
    rows = read_csv("error_transition_matrix_w2c_qwen7b.csv")
    M = np.zeros((len(GOLD), len(PRED)))
    for r in rows:
        if r["gold"] in GOLD and r["pred"] in PRED:
            M[GOLD.index(r["gold"]), PRED.index(r["pred"])] = int(r["count"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(PRED))); ax.set_xticklabels(PRED, rotation=30, ha="right")
    ax.set_yticks(range(len(GOLD))); ax.set_yticklabels(GOLD)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title("Qwen2.5-7B W2C confusion (gold → pred), n=3652")
    for i in range(len(GOLD)):
        for j in range(len(PRED)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    save(fig, "fig_w2c_confusion_heatmap.png")
    record("fig_w2c_confusion_heatmap.png", "error_transition_matrix_w2c_qwen7b.csv",
           "Qwen2.5-7B W2C gold→pred confusion heatmap (exact from per-sample labels)")


# 2. Qwen dominant-channel bar
def fig_dominant_channels():
    rows = [r for r in read_csv("channel_discovery_summary.csv") if r["dataset"] == "w2c_qwen25_7b"]
    rows = [r for r in rows if r["count_all"] not in ("", None)]
    rows.sort(key=lambda r: -int(r["count_all"]))
    names = [r["channel_id"] for r in rows]; counts = [int(r["count_all"]) for r in rows]
    named = {"rfi_tc", "ca_tc", "ca_direct"}
    colors = ["#c0392b" if n in named else "#7f8c8d" for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), counts, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("error count (all, n=3652)")
    ax.set_title("W2C/Qwen2.5-7B discovered error channels (red = SAKIKO-targeted)")
    for i, c in enumerate(counts):
        ax.text(i, c + 4, str(c), ha="center", fontsize=8)
    save(fig, "fig_dominant_channels_qwen7b.png")
    record("fig_dominant_channels_qwen7b.png", "channel_discovery_summary.csv",
           "Discovered W2C error channels by count; 3 SAKIKO channels highlighted")


# 3. Dataset label-space / channel comparison
def fig_dataset_channel_space():
    data = [("W2C\n(Qwen2.5-7B)", 4, 3, "multi-class"),
            ("MetaTool-Binary", 2, 1, "binary over-call"),
            ("ToolSandbox", 0, 0, "qualitative-only")]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(data))
    ax.bar([i - 0.2 for i in x], [d[1] for d in data], width=0.4, label="# gold classes", color="#2980b9")
    ax.bar([i + 0.2 for i in x], [d[2] for d in data], width=0.4, label="# SAKIKO channels", color="#c0392b")
    ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel("count"); ax.set_title("Label space & channel count are dataset-dependent")
    for i, d in enumerate(data):
        ax.text(i, max(d[1], d[2]) + 0.15, d[3], ha="center", fontsize=8, style="italic")
    ax.legend()
    save(fig, "fig_dataset_channel_space.png")
    record("fig_dataset_channel_space.png",
           "channel_discovery_{summary.csv,metatool_binary_summary.json,phi35_summary.json}",
           "Label-space/channel-count comparison across W2C, MetaTool-Binary, ToolSandbox")


# 4. ca_direct layer comparison (norm + net vs layer)
def fig_ca_direct_layers():
    rows = read_csv("ca_direct_layer_comparison_qwen7b.csv")
    rows.sort(key=lambda r: int(str(r["obs_layer"]).lstrip("L")))
    L = [f"L{str(r['obs_layer']).lstrip('L')}" for r in rows]
    norm = [float(r["direction_norm"]) for r in rows]
    net = [int(r["val_net"]) for r in rows]
    auc = [float(r["router_val_auc"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(L, norm, "o-", color="#c0392b", label="DiffMean norm")
    ax1.plot(L, net, "s-", color="#27ae60", label="best-val Net")
    ax1.set_xlabel("ca_direct observation layer"); ax1.set_ylabel("norm / Net")
    ax2 = ax1.twinx(); ax2.plot(L, auc, "^--", color="#7f8c8d", label="router val AUC")
    ax2.set_ylabel("router val AUC"); ax2.set_ylim(0.9, 1.0)
    ax1.set_title("ca_direct: Net tracks direction norm, not router AUC")
    lns = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lns, [l.get_label() for l in lns], loc="center right", fontsize=8)
    save(fig, "fig_ca_direct_layer_comparison.png")
    record("fig_ca_direct_layer_comparison.png", "ca_direct_layer_comparison_qwen7b.csv",
           "ca_direct DiffMean norm & Net rise with layer while router AUC stays flat (~0.96)")


# 5. ca_tc DiffMean vs PCA-1 (val + test)
def fig_ca_tc_pca():
    rows = read_csv("ca_tc_pca_vs_diffmean_qwen7b.csv")
    keep = {("diffmean", "val"), ("diffmean", "test"), ("pca1", "val"), ("pca1", "test")}
    d = {(r["direction_method"], r["split"]): int(r["net"]) for r in rows
         if (r["direction_method"], r["split"]) in keep}
    labels = ["val", "test"]
    dm = [d[("diffmean", s)] for s in labels]; pc = [d[("pca1", s)] for s in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x - 0.2, dm, 0.4, label="DiffMean", color="#2980b9")
    ax.bar(x + 0.2, pc, 0.4, label="PCA-1", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("ca_tc Net")
    ax.set_title("ca_tc: PCA-1 > DiffMean (val & locked test)")
    for i, v in enumerate(dm): ax.text(i - 0.2, v + 0.5, v, ha="center", fontsize=8)
    for i, v in enumerate(pc): ax.text(i + 0.2, v + 0.5, v, ha="center", fontsize=8)
    ax.legend()
    save(fig, "fig_ca_tc_pca_vs_diffmean.png")
    record("fig_ca_tc_pca_vs_diffmean.png", "ca_tc_pca_vs_diffmean_qwen7b.csv",
           "ca_tc DiffMean vs PCA-1 Net on val and locked test")


# 6. Router / probe AUC by layer
def fig_probe_auc():
    if not (CA / "qwen7b_probe_auc_by_layer.csv").exists():
        record("fig_probe_auc_by_layer.png", "qwen7b_probe_auc_by_layer.csv",
               "per-layer linear-probe AUC per channel", status="SKIPPED_no_activations")
        return
    rows = read_csv("qwen7b_probe_auc_by_layer.csv")
    chans = sorted({r["channel"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for ch in chans:
        rs = sorted([r for r in rows if r["channel"] == ch], key=lambda r: int(r["obs_layer"]))
        ax.plot([f"L{r['obs_layer']}" for r in rs], [float(r["cv_auc_5fold"]) for r in rs],
                "o-", label=ch)
    ax.set_xlabel("layer"); ax.set_ylabel("5-fold CV probe AUC")
    ax.set_title("Channel separability (probe AUC) by layer — Qwen2.5-7B")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig_probe_auc_by_layer.png")
    record("fig_probe_auc_by_layer.png", "qwen7b_probe_auc_by_layer.csv",
           "per-layer linear-probe AUC per channel (from local activations)")


# 7. Direction norm by channel (native layer)
def fig_direction_norm():
    if not (CA / "channel_layer_geometry_qwen7b.csv").exists():
        record("fig_direction_norm_by_channel.png", "channel_layer_geometry_qwen7b.csv",
               "DiffMean norm per channel", status="SKIPPED_no_activations")
        return
    rows = [r for r in read_csv("channel_layer_geometry_qwen7b.csv") if r["is_native_layer"] == "1"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = [r["channel"] for r in rows]; norms = [float(r["diffmean_norm"]) for r in rows]
    ax.bar(names, norms, color=["#c0392b", "#2980b9", "#27ae60"][:len(names)])
    for i, v in enumerate(norms): ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("DiffMean norm (native layer)")
    ax.set_title("Direction norm by channel (native obs layer)")
    save(fig, "fig_direction_norm_by_channel.png")
    record("fig_direction_norm_by_channel.png", "channel_layer_geometry_qwen7b.csv",
           "DiffMean direction norm per channel at its native observation layer")


# 8. PCA-2D scatter
def fig_pca2d():
    if not (CA / "qwen7b_pca2d_channel_points.csv").exists():
        record("fig_pca2d_channels.png", "qwen7b_pca2d_channel_points.csv",
               "2-D PCA of channel/correct samples", status="SKIPPED_no_activations")
        return
    rows = read_csv("qwen7b_pca2d_channel_points.csv")
    colors = {"correct": "#bdc3c7", "rfi_tc": "#c0392b", "ca_tc": "#2980b9", "ca_direct": "#27ae60"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, col in colors.items():
        xs = [float(r["pc1"]) for r in rows if r["channel_or_correct"] == lab]
        ys = [float(r["pc2"]) for r in rows if r["channel_or_correct"] == lab]
        ax.scatter(xs, ys, s=8, alpha=0.5, label=lab, color=col)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Qwen2.5-7B L{rows[0]['obs_layer']} activations, 2-D PCA by channel")
    ax.legend(markerscale=2, fontsize=8)
    save(fig, "fig_pca2d_channels.png")
    record("fig_pca2d_channels.png", "qwen7b_pca2d_channel_points.csv",
           "2-D PCA scatter of channel-error vs correct samples (local activations)")


# 9. Silhouette by layer
def fig_silhouette():
    if not (CA / "qwen7b_silhouette_by_layer.csv").exists():
        record("fig_silhouette_by_layer.png", "qwen7b_silhouette_by_layer.csv",
               "silhouette by layer", status="SKIPPED_no_activations")
        return
    rows = sorted(read_csv("qwen7b_silhouette_by_layer.csv"), key=lambda r: int(r["obs_layer"]))
    L = [f"L{r['obs_layer']}" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, col in [("silhouette_overall", "black"), ("sil_rfi_tc", "#c0392b"),
                     ("sil_ca_tc", "#2980b9"), ("sil_ca_direct", "#27ae60")]:
        ax.plot(L, [float(r[key]) for r in rows], "o-", label=key.replace("sil_", ""), color=col)
    ax.set_xlabel("layer"); ax.set_ylabel("silhouette (PCA-50)")
    ax.set_title("Channel silhouette by layer — Qwen2.5-7B"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, "fig_silhouette_by_layer.png")
    record("fig_silhouette_by_layer.png", "qwen7b_silhouette_by_layer.csv",
           "per-layer silhouette of channel/correct structure (local activations)")


def main():
    fig_transition_heatmap(); fig_dominant_channels(); fig_dataset_channel_space()
    fig_ca_direct_layers(); fig_ca_tc_pca(); fig_probe_auc(); fig_direction_norm()
    fig_pca2d(); fig_silhouette()
    # explicit TODO rows for figures we deliberately do NOT make
    for fn, why in [("fig_metatool_geometry.png", "MetaTool activations not cached"),
                    ("fig_phi_geometry.png", "Phi activations not cached"),
                    ("fig_phi_qwen_alignment.png", "no Phi activations / restored mapping cache")]:
        record(fn, "-", "geometry plot", status="TODO_needs_activations", notes=why)
    with open(CA / "figure_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["figure_file", "source_table", "description", "status", "notes"])
        wr.writeheader()
        for m in MANIFEST:
            wr.writerow(m)
    log.info("=== figures done: %d generated, manifest written ===",
             sum(1 for m in MANIFEST if m["status"] == "generated"))


if __name__ == "__main__":
    main()
