"""phase9_5_figures.py — Phase 9.5 figures (CPU, from the audit JSONs only)."""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

OUT = L.ROOT / "final" / "results" / "phase9_mechanism_and_ood"
FIG = OUT / "figures"
GOLD = "cannot_answer"

ax_j = json.loads((OUT / "phase9_5_axes.json").read_text())
tr_j = json.loads((OUT / "phase9_5_trajectory.json").read_text())
ps = json.loads((OUT / "phase9_5_per_sample_routed.json").read_text())
own = [r for r in ps if r["is_own_channel_err"]]

# ---------------- Figure 1: axis decomposition ----------------
names = ax_j["names"]
C = np.array(ax_j["cosine_L22"])
fig, axs = plt.subplots(1, 4, figsize=(16.5, 3.9),
                        gridspec_kw={"width_ratios": [1.25, 1.1, 1, 1]})
# A: heatmap
im = axs[0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
axs[0].set_xticks(range(len(names))); axs[0].set_yticks(range(len(names)))
short = ["deploy", "evac", "select", "M1", "M2lda", "M2log"]
axs[0].set_xticklabels(short, rotation=45, ha="right", fontsize=8)
axs[0].set_yticklabels(short, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        axs[0].text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(C[i, j]) > 0.6 else "black")
axs[0].set_title("A. cos matrix at L22 (estimation space)", fontsize=9)
fig.colorbar(im, ax=axs[0], fraction=0.045)
# B: (evac, sel_perp) plane
dec = ax_j["decomposition_on_(evac,sel_perp)_L22"]
cols = {"deployed_dm": "#0969da", "M1": "#1a7f37", "M2_lda": "#d1242f", "M2_logistic": "#bf8700"}
th = np.linspace(0, 2 * np.pi, 200)
axs[1].plot(np.cos(th), np.sin(th), color="#ccc", lw=0.8)
for n, d in dec.items():
    axs[1].annotate("", xy=(d["coef_evac"], d["coef_sel_perp"]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=cols[n], lw=2))
    axs[1].annotate(f"{n.replace('_lda','').replace('_dm','')}\n(in-plane {d['in_plane_frac']:.2f})",
                    xy=(d["coef_evac"], d["coef_sel_perp"]),
                    xytext=(d["coef_evac"] * 1.15 + 0.02, d["coef_sel_perp"] * 1.15 + 0.04),
                    fontsize=7.5, color=cols[n])
axs[1].axhline(0, color="k", lw=0.5); axs[1].axvline(0, color="k", lw=0.5)
axs[1].set_xlim(-0.35, 1.25); axs[1].set_ylim(-0.35, 1.25); axs[1].set_aspect("equal")
axs[1].set_xlabel("evacuation axis coefficient", fontsize=8)
axs[1].set_ylabel("selection⊥ coefficient", fontsize=8)
axs[1].set_title(f"B. estimators in the (evac, sel⊥) plane\ncos(evac,select)={ax_j['cosine_L22'][1][2]:.3f} → plane nearly degenerate",
                 fontsize=9)
# C: split-half stability
sh = ax_j["splithalf_L22"]
vals = [sh[n]["mean"] for n in names]
bc = ["#d1242f" if v < 0.5 else "#1a7f37" for v in vals]
axs[2].bar(range(len(names)), vals, color=bc)
axs[2].axhline(0.5, color="k", ls="--", lw=1)
axs[2].text(0.05, 0.515, "Gate v2 Stage-3 floor (0.5)", fontsize=7)
axs[2].set_xticks(range(len(names))); axs[2].set_xticklabels(short, rotation=45, ha="right", fontsize=8)
axs[2].set_ylim(0, 1.05); axs[2].set_title("C. split-half stability of each axis\n(M2 = 0.14: resampling noise)", fontsize=9)
# D: cross-layer transport
cl = ax_j["cross_layer_same_axis_cos(L22,L18)"]
axs[3].bar(range(len(names)), [cl[n] for n in names], color="#57606a")
axs[3].set_xticks(range(len(names))); axs[3].set_xticklabels(short, rotation=45, ha="right", fontsize=8)
axs[3].set_ylim(-0.15, 1.0)
axs[3].axhline(0, color="k", lw=0.6)
axs[3].set_title("D. cos(axis@L22, axis@L18) — same axis,\nboth layers: ≈0 (injected space ⊥ local geometry)", fontsize=9)
fig.suptitle("Phase 9.5 — functional-axis decomposition of the executed Llama ca_tc directions (train-only, CPU)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIG / "fig_phase9_5_axis_decomposition.png", dpi=180)
fig.savefig(FIG / "fig_phase9_5_axis_decomposition.pdf")
plt.close(fig)

# ---------------- Figure 2: movement / margin structure ----------------
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
# A: moved vs rho
meth = tr_j["methods"]
for mname, col, mk in (("deployed", "#0969da", "d"), ("M1", "#1a7f37", "o"), ("M2", "#d1242f", "s")):
    cells = meth[mname]["cells"]
    axs[0].plot([c["rho"] for c in cells], [c["moved"] for c in cells],
                marker=mk, color=col, label=mname, lw=2, ms=6)
lb = meth["M1"]["random_rho4_min_movement_lower_bound"]
axs[0].scatter([4.0], [lb["mean_|TG|"]], marker="*", s=140, color="#8250df", zorder=5,
               label="randoms @ρ=4 (LOWER bound,\nmean |TG|)")
axs[0].scatter([4.0], [lb["max_|TG|"]], marker="*", s=80, color="#8250df", alpha=0.5)
axs[0].axhline(68, color="k", ls=":", lw=1)
axs[0].text(0.27, 69.5, "all 68 routed own errors", fontsize=7)
axs[0].set_xscale("log", base=2); axs[0].set_xticks([0.25, 0.5, 1, 2, 4, 8])
axs[0].set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
axs[0].set_xlabel("ρ"); axs[0].set_ylabel("errors moved off source")
axs[0].set_title("A. movement is generic at large ρ\n(randoms ≥ deployed > M1 ≫ M2)", fontsize=9)
axs[0].legend(fontsize=7); axs[0].grid(alpha=0.25)
# B: destination split at rho=4 (+deployed rho 8), vs gold-adjacent pool
bars = [("deployed\nρ=4", 10, 25), ("M1\nρ=4", 9, 5), ("M2\nρ=4", 0, 4), ("deployed\nρ=8", 28, 39)]
x = np.arange(len(bars))
axs[1].bar(x, [b[1] for b in bars], color="#1a7f37", label="→ gold")
axs[1].bar(x, [b[2] for b in bars], bottom=[b[1] for b in bars], color="#d1242f", label="→ wrong dest")
axs[1].axhline(22, color="k", ls="--", lw=1.2)
axs[1].text(-0.45, 23, "gold-adjacent routed pool = 22", fontsize=7.5)
axs[1].set_xticks(x); axs[1].set_xticklabels([b[0] for b in bars], fontsize=8)
axs[1].set_ylabel("moved errors")
axs[1].set_title("B. gold arrivals ≈ equal at ρ=4 (10 vs 9);\nestimators differ in WRONG moves (25 vs 5)", fontsize=9)
axs[1].legend(fontsize=8)
# C: baseline structure of routed own errors along the injected axis
ru_col = {"cannot_answer": "#1a7f37", "direct": "#d1242f", "request_for_info": "#bf8700"}
for ru, col in ru_col.items():
    pts = [r for r in own if r["runner_up"] == ru]
    axs[2].scatter([r["proj22_M1"] for r in pts], [r["gap_top_minus_gold"] for r in pts],
                   s=26, color=col, label=f"runner-up = {ru} (n={len(pts)})", alpha=0.85)
axs[2].axvline(0, color="k", lw=0.5)
axs[2].set_xlabel("baseline projection on M1 axis (centered on error mean), L22")
axs[2].set_ylabel("baseline gap: top logp − gold logp")
axs[2].set_title("C. gold-adjacent errors already sit high on the\ninjected axis and close to gold (predicted responders)", fontsize=9)
axs[2].legend(fontsize=7); axs[2].grid(alpha=0.25)
fig.suptitle("Phase 9.5 — movement and margin structure from stored aggregates + baseline details "
             "(per-sample post-intervention outcomes were not stored: responder identity is DEFERRED-GPU)",
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "fig_phase9_5_margin_trajectories.png", dpi=180)
fig.savefig(FIG / "fig_phase9_5_margin_trajectories.pdf")
plt.close(fig)
print("wrote", FIG / "fig_phase9_5_axis_decomposition.png")
print("wrote", FIG / "fig_phase9_5_margin_trajectories.png")
