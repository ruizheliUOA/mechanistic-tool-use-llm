"""
make_sakiko_figures.py — generate the 5 SAKIKO paper figures from archived CSVs.
CPU-only, no model, no GPU. Reads final/results/figures/*.csv, writes plots/ (PNG+PDF).
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[1] / "final" / "results" / "figures"
OUT = FIG / "plots"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
CH = ["rfi_tc", "ca_tc", "ca_direct"]
COL = {"rfi_tc": "#4C72B0", "ca_tc": "#DD8452", "ca_direct": "#C44E52"}


def rows(name):
    with open(FIG / name) as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


# ── Fig 1: channel error reduction (multiseed mean before/after + per-seed points) ──
def fig1():
    r = rows("fig1_channel_error_reduction.csv")
    mean = {x["channel"]: (float(x["before"]), float(x["after"])) for x in r if x["source"] == "multiseed_mean"}
    perseed = defaultdict(list)
    for x in r:
        if x["source"] == "multiseed" and x["seed"] != "mean":
            perseed[x["channel"]].append((float(x["before"]), float(x["after"])))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xpos = np.arange(len(CH)); w = 0.38
    before = [mean[c][0] for c in CH]; after = [mean[c][1] for c in CH]
    ax.bar(xpos - w/2, before, w, label="before SAKIKO", color="#B0B0B0")
    ax.bar(xpos + w/2, after, w, label="after SAKIKO", color=[COL[c] for c in CH])
    for i, c in enumerate(CH):
        ys = [a for (_, a) in perseed[c]]
        ax.scatter([xpos[i] + w/2]*len(ys), ys, color="black", s=18, zorder=5, alpha=0.7)
        red = 100*(mean[c][0]-mean[c][1])/mean[c][0]
        ax.text(xpos[i], max(before)*0.92, f"−{red:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(CH); ax.set_ylabel("error count (test)")
    ax.set_title("Fig 1 · Qwen2.5-7B channel error reduction (multi-seed mean; dots = per-seed after)")
    ax.legend(); save(fig, "fig1_channel_error_reduction")


# ── Fig 2: placebo distribution (real vs reverse vs random) per seed, multiseed ──
def fig2():
    r = [x for x in rows("fig2_placebo_distribution.csv") if x["source"] == "multiseed"]
    seeds = sorted({x["seed"] for x in r}, key=int)
    real = {s: None for s in seeds}; rev = {s: None for s in seeds}; rnd = defaultdict(list)
    for x in r:
        v = float(x["net"])
        if x["variant"] == "real": real[x["seed"]] = v
        elif x["variant"] == "reverse": rev[x["seed"]] = v
        elif x["variant"].startswith("random"): rnd[x["seed"]].append(v)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xpos = np.arange(len(seeds))
    for i, s in enumerate(seeds):
        ys = rnd[s]
        ax.scatter([i]*len(ys), ys, color="#999999", s=22, alpha=0.7,
                   label="random dirs" if i == 0 else None)
        if len(ys) > 1:
            ax.plot([i-0.18, i+0.18], [np.mean(ys)]*2, color="#555555", lw=2)
        ax.scatter([i], [real[s]], color="#C44E52", marker="*", s=220, zorder=6,
                   label="real" if i == 0 else None, edgecolor="black", linewidth=0.5)
        ax.scatter([i], [rev[s]], color="#4C72B0", marker="v", s=80, zorder=6,
                   label="reverse" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(seeds); ax.set_xlabel("seed"); ax.set_ylabel("Net (test)")
    ax.set_title("Fig 2 · Placebo distribution: real ≫ random, reverse collapses (Qwen2.5-7B, multi-seed)")
    ax.legend(loc="upper left"); save(fig, "fig2_placebo_distribution")


# ── Fig 3: global penalty vs SAKIKO (Net + Δacc) ──
def fig3():
    r = rows("fig3_penalty_vs_sakiko.csv")
    order = ["baseline", "global_penalty_l0.15", "SAKIKO_seed42", "SAKIKO_multiseed_mean"]
    labels = ["baseline", "global\npenalty", "SAKIKO\nseed=42", "SAKIKO\nmulti-seed"]
    d = {x["method"]: x for x in r}
    net = [float(d[m]["net"]) for m in order]
    dacc = [float(d[m]["delta_acc"])*100 for m in order]
    cols = ["#B0B0B0", "#DD8452", "#4C72B0", "#55A868"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.bar(labels, net, color=cols)
    for i, v in enumerate(net): a1.text(i, v+1, f"{v:.0f}", ha="center", fontweight="bold")
    a1.set_ylabel("Net (corrected preds)"); a1.set_title("Net")
    a2.bar(labels, dacc, color=cols)
    for i, v in enumerate(dacc): a2.text(i, v+0.2, f"{v:.1f}", ha="center", fontweight="bold")
    a2.set_ylabel("Δ accuracy (pts)"); a2.set_title("Δ accuracy")
    fig.suptitle("Fig 3 · Global tool_call penalty vs SAKIKO — penalty ≈ 4% of SAKIKO", fontweight="bold")
    save(fig, "fig3_penalty_vs_sakiko")


# ── Fig 4: mapping R² vs intervention Net ──
def fig4():
    r = rows("fig4_mapping_r2_vs_net.csv")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for x in r:
        if x["val_r2"] == "n/a":  # native reference
            ax.axhline(float(x["val_net"]), color="#55A868", ls="--", lw=1.5,
                       label=f"native L16 (val net {x['val_net']})")
            continue
        r2 = float(x["val_r2"]); net = float(x["val_net"])
        ax.scatter(r2, net, s=130, zorder=5)
        ax.annotate(x["method"], (r2, net), xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("mapping val R² (activation reconstruction)")
    ax.set_ylabel("intervention val Net")
    ax.set_title("Fig 4 · Mapping R² vs Net — high R² (Ridge) ≠ high Net; Procrustes wins")
    ax.legend(); save(fig, "fig4_mapping_r2_vs_net")


# ── Fig 5: ca_direct layer sensitivity (dir norm vs reduction, L16 vs L20) ──
def fig5():
    r = rows("fig5_ca_direct_layer_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for x in r:
        L = int(x["ca_direct_obs_layer"]); norm = float(x["ca_direct_dir_norm"]); red = float(x["ca_direct_reduction"])
        c = "#C44E52" if L == 16 else "#55A868"
        ax.scatter(norm, red, s=150, color=c, zorder=5, edgecolor="black", linewidth=0.5)
        ax.annotate(f"seed {x['seed']} (L{L})", (norm, red), xytext=(6, 4), textcoords="offset points", fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#C44E52',markersize=11,label='obs L16 (degenerate)'),
                       Line2D([0],[0],marker='o',color='w',markerfacecolor='#55A868',markersize=11,label='obs L20 (well-conditioned)')])
    ax.set_xlabel("ca_direct DiffMean direction norm")
    ax.set_ylabel("ca_direct error reduction (test)")
    ax.set_title("Fig 5 · ca_direct layer sensitivity — small direction norm (L16) → weak repair")
    save(fig, "fig5_ca_direct_layer_sensitivity")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("All figures written to", OUT)
