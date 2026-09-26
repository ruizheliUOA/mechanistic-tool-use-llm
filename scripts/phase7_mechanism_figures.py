"""
phase7_mechanism_figures.py — Part-E figures from real Phase-7 data (CPU only, val only).
figA: detectability-actionability map.  figD: static-geometry comparison (3 cases).
figE: final taxonomy. (figB/figC = fig1/fig2 from phase7_make_figures.py, real curves.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

FIG = P.OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
GEO = json.loads((P.OUT / "mechanism_geometry.json").read_text())
RHO_MAX = 8.0
N_RHO_COMPLETE = 6


def load_results_merged():
    """RESULTS + COMPLETE channels (6/6 rho) from interim state files (read-only).
    Incomplete channels of the still-running pre-registered sweep are excluded."""
    res = json.loads((P.OUT / "rho_specificity_results.json").read_text()) \
        if (P.OUT / "rho_specificity_results.json").exists() else {}
    for mk in P.MODELS:
        sf = P.CACHE_OUT / f"phase7_state_{mk}.json"
        if sf.exists():
            state = json.loads(sf.read_text())
            for ch, blob in state.items():
                if len(blob.get("rows", [])) >= N_RHO_COMPLETE and ch not in res.get(mk, {}):
                    res.setdefault(mk, {})[ch] = blob
    return res


RES = load_results_merged()


def best_interior_z(mk, ch):
    blob = RES.get(mk, {}).get(ch)
    if not blob:
        return None
    rows = [r for r in blob["rows"] if r["rho"] < RHO_MAX and r["spec_z"] is not None
            and r["real"]["net"] > 0]
    return max((r["spec_z"] for r in rows), default=None)


def figA():
    pts = [
        ("Qwen ca_rfi", 0.9176, best_interior_z("qwen25_7b", "ca_rfi"), "#1a7f37", "o",
         "actionable (test-confirmed)"),
        ("Qwen tc_rfi", 0.8486, best_interior_z("qwen25_7b", "tc_rfi"), "#8250df", "^",
         "val-pass / test-null"),
        ("Mistral ca_tc", 0.9636, best_interior_z("mistral7b_v03", "ca_tc"), "#d1242f", "s",
         "non-actionable"),
        ("Mistral rfi_tc", 0.8777, None, "#d1242f", "x", "pending (neg-control)"),
        ("Mistral ca_direct", 0.9380, None, "#d1242f", "x", "pending (neg-control)"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, auc, z, c, m, note in pts:
        if z is None:
            ax.scatter([auc], [-0.6], c="#aaaaaa", marker=m, s=70)
            ax.annotate(f"{name}\n({note})", (auc, -0.6), textcoords="offset points",
                        xytext=(6, -14), fontsize=7, color="#777")
        else:
            ax.scatter([auc], [z], c=c, marker=m, s=110, zorder=4)
            ax.annotate(f"{name}\nz={z:.2f}", (auc, z), textcoords="offset points",
                        xytext=(7, 4), fontsize=8, color=c)
    ax.axhline(2.0, color="k", ls=":", lw=1.4)
    ax.annotate("actionability bar (interior val z = 2)", (0.845, 2.1), fontsize=8)
    ax.axvline(0.75, color="k", ls="--", lw=0.9)
    ax.annotate("detectability floor\nAUC = 0.75", (0.752, ax.get_ylim()[0] + 0.4), fontsize=7)
    ax.set_xlabel("router detectability (validation AUC)")
    ax.set_ylabel("direction-specific evidence\n(best interior validation z, Net>0)")
    ax.set_title("Detectability does not imply actionability\n"
                 "(validation-only, 20 matched-norm randoms; grey = sweep pending)", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "figA_detectability_vs_actionability.png", dpi=180)
    fig.savefig(FIG / "figA_detectability_vs_actionability.pdf")
    plt.close(fig)


def figD():
    cases = [(g, {"qwen25_7b/ca_rfi": ("Qwen ca_rfi\n(actionable)", "#1a7f37"),
                  "qwen25_7b/tc_rfi": ("Qwen tc_rfi\n(false-admission)", "#8250df"),
                  "mistral7b_v03/ca_tc": ("Mistral ca_tc\n(non-actionable)", "#d1242f")}
              [f"{g['model']}/{g['channel']}"]) for g in GEO]
    metrics = [("eff_rank_err_cloud", "effective rank\n(error cloud)"),
               ("pc1_evr", "PC1 explained\nvariance ratio"),
               ("silhouette_k2", "2-means\nsilhouette"),
               ("dm_splithalf_cos_mean", "DiffMean split-half\ncosine (stability)"),
               ("cos_dm_pc1", "cos(DiffMean, PC1)"),
               ("dm_frac_in_top10_pc", "DiffMean energy in\ntop-10 variance axes")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.1 * len(metrics), 3.4))
    for ax, (key, title) in zip(axes, metrics):
        for i, (g, (label, color)) in enumerate(cases):
            ax.bar(i, g[key], color=color, width=0.65)
            ax.text(i, g[key], f"{g[key]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([c[1][0].split("\n")[0].split(" ")[1] for c in cases], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Static geometry at the readout site does NOT separate the three outcomes\n"
                 "(green=actionable, purple=false-admission, red=non-actionable; train rows only)",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG / "figD_geometry_comparison.png", dpi=180)
    fig.savefig(FIG / "figD_geometry_comparison.pdf")
    plt.close(fig)


def figE():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.62, "DETECTABLE **AND** ACTIONABLE",
         "Qwen ca_rfi\ninterior val z=6.96 (ρ=4); reverse→0;\ntest z=4.82 (0/20); multiseed 5/5",
         "#d3f2d9", "#1a7f37"),
        (0.52, 0.62, "DETECTABLE, **NON-ACTIONABLE**",
         "Mistral ca_tc (AUC 0.964)\nno ρ in 0.25–8 passes conjunction;\nrandom mean +27–33 at high ρ\n"
         "Mistral rfi_tc / ca_direct: pending sweep;\narchived test: non-specific",
         "#ffe1e1", "#d1242f"),
        (0.02, 0.16, "VALIDATION-POSITIVE, **TEST-NEGATIVE**",
         "Qwen tc_rfi\nval Net +6, z=3.29 (0/20) at ρ=1\ntest Net 0, z=0.82 (11/20)\n→ Stage-5 admits ONLY to Stage-6",
         "#efe3ff", "#8250df"),
        (0.52, 0.16, "INSUFFICIENT EVIDENCE",
         "Qwen v1 rfi_tc/ca_tc/ca_direct:\nper-channel randoms never run (cascade n=10)\n"
         "3/5 multiseed seeds: n_random=1\nACEBench: no channel met support",
         "#f0f0f0", "#57606a"),
    ]
    for x, y, title, body, fc, ec in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.46, 0.36, facecolor=fc, edgecolor=ec, lw=2))
        ax.text(x + 0.23, y + 0.315, title.replace("**", ""), ha="center", fontsize=9.5,
                fontweight="bold", color=ec)
        ax.text(x + 0.23, y + 0.15, body, ha="center", va="center", fontsize=7.6)
    ax.set_title("Channel taxonomy after Phase 7 (development evidence; validation-only screen "
                 "+ archived locked tests)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figE_channel_taxonomy.png", dpi=180)
    fig.savefig(FIG / "figE_channel_taxonomy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figA(); figD(); figE()
    print("wrote figA/figD/figE to", FIG)
