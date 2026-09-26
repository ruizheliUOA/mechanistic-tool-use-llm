"""
phase7_make_figures.py — claim-bearing figures for Phase 7 (CPU only, no test data).
====================================================================================
1. fig1: validation Net vs rho (real / reverse / random mean±sd band) per channel.
2. fig2: specificity z vs rho per channel, with the z>=2 bar and the boundary marker.
3. fig3: raw alpha vs normalized rho across Qwen and Mistral (why alpha is not portable).
Figure manifest documents source data and the claim each figure supports.
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
from phase7_mechanism_figures import load_results_merged
RES = load_results_merged()
CAL = json.loads((P.OUT / "rho_calibration.json").read_text())

STYLE = {"real": ("#1a7f37", "o", "real (learned)"),
         "reverse": ("#d1242f", "v", "reverse"),
         "random": ("#888888", "s", "random mean ± sd")}
ORDER = [("qwen25_7b", "ca_rfi"), ("qwen25_7b", "tc_rfi"),
         ("mistral7b_v03", "ca_tc"), ("mistral7b_v03", "rfi_tc"),
         ("mistral7b_v03", "ca_direct")]
TITLE = {"ca_rfi": "Qwen ca_rfi\n(positive control)", "tc_rfi": "Qwen tc_rfi\n(negative control)",
         "ca_tc": "Mistral ca_tc\n(DISCRIMINATOR)", "rfi_tc": "Mistral rfi_tc\n(negative control)",
         "ca_direct": "Mistral ca_direct\n(negative control)"}


def panels():
    return [(mk, ch) for mk, ch in ORDER if mk in RES and ch in RES[mk]]


def fig1():
    ps = panels()
    fig, axes = plt.subplots(1, len(ps), figsize=(3.4 * len(ps), 3.6), sharex=True)
    if len(ps) == 1:
        axes = [axes]
    for ax, (mk, ch) in zip(axes, ps):
        rows = sorted(RES[mk][ch]["rows"], key=lambda r: r["rho"])
        rho = [r["rho"] for r in rows]
        real = [r["real"]["net"] for r in rows]
        rev = [r["reverse"]["net"] for r in rows]
        rm = np.array([r["random"]["mean"] for r in rows])
        rs = np.array([r["random"]["std"] for r in rows])
        ax.fill_between(rho, rm - rs, rm + rs, color="#bbbbbb", alpha=0.45, label="random ±1sd", zorder=1)
        ax.plot(rho, rm, "s--", color="#888888", ms=4, lw=1.2, label="random mean", zorder=2)
        ax.plot(rho, rev, "v--", color="#d1242f", ms=5, lw=1.2, label="reverse", zorder=3)
        ax.plot(rho, real, "o-", color="#1a7f37", ms=6, lw=2, label="real", zorder=4)
        ar = RES[mk][ch]["archived_rho"]
        ax.axvline(ar, color="#0969da", ls=":", lw=1.4, zorder=0)
        ax.annotate(f"archived\nρ={ar:.2f}", (ar, ax.get_ylim()[1]), fontsize=7, color="#0969da",
                    ha="center", va="top")
        ax.set_xscale("log", base=2); ax.set_xticks(rho); ax.set_xticklabels([str(r) for r in rho])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(TITLE.get(ch, ch), fontsize=9)
        ax.set_xlabel("ρ  (‖Δh‖ / median‖h_inj‖)", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation Net (Fixed − Broke)", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Validation Net vs architecture-normalized perturbation ρ  "
                 "(development only; no test data)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig1_val_net_vs_rho.png", dpi=180)
    fig.savefig(FIG / "fig1_val_net_vs_rho.pdf")
    plt.close(fig)


def fig2():
    ps = panels()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    cmap = {"ca_rfi": "#1a7f37", "tc_rfi": "#8250df", "ca_tc": "#0969da",
            "rfi_tc": "#d1242f", "ca_direct": "#bf8700"}
    for mk, ch in ps:
        rows = sorted(RES[mk][ch]["rows"], key=lambda r: r["rho"])
        rho = [r["rho"] for r in rows]
        z = [(r["spec_z"] if r["spec_z"] is not None else np.nan) for r in rows]
        ls = "-" if mk == "mistral7b_v03" else "--"
        mkr = "o" if ch in ("ca_rfi", "ca_tc") else "x"
        ax.plot(rho, z, ls, marker=mkr, color=cmap.get(ch, "#333"), lw=2 if ch == "ca_tc" else 1.4,
                ms=7 if ch == "ca_tc" else 5,
                label=f"{'Mistral' if mk.startswith('mistral') else 'Qwen'} {ch}")
    ax.axhline(2.0, color="k", ls=":", lw=1.5)
    ax.annotate("specificity bar  z = 2", (0.26, 2.12), fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(6.0, 9.0, color="#ffdddd", alpha=0.5, zorder=0)
    ax.annotate("boundary\nρ_max", (8.0, ax.get_ylim()[0]), fontsize=7, color="#d1242f",
                ha="center", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    ax.set_xlabel("ρ  (‖Δh‖ / median‖h_inj‖, train-estimated)")
    ax.set_ylabel("specificity  z = (Net_real − mean Net_random) / sd")
    ax.set_title("Specificity vs normalized perturbation (validation only, 20 matched-norm randoms)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_specificity_z_vs_rho.png", dpi=180)
    fig.savefig(FIG / "fig2_specificity_z_vs_rho.pdf")
    plt.close(fig)


def fig3():
    """Why raw alpha is not portable: same alpha -> different rho per channel/architecture."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = {"ca_rfi": "#1a7f37", "tc_rfi": "#8250df", "ca_tc": "#0969da",
            "rfi_tc": "#d1242f", "ca_direct": "#bf8700"}
    for mk, ch in panels():
        rows = sorted(RES[mk][ch]["rows"], key=lambda r: r["rho"])
        a = [r["alpha_equiv"] for r in rows]; rho = [r["rho"] for r in rows]
        ls = "-" if mk == "mistral7b_v03" else "--"
        ax.plot(a, rho, ls, marker="o", ms=4, color=cmap.get(ch, "#333"),
                label=f"{'Mistral' if mk.startswith('mistral') else 'Qwen'} {ch} "
                      f"(×{rows[0]['med_obs']/rows[0]['med_inj']:.2f})")
    ax.plot([0.1, 10], [0.1, 10], color="k", lw=0.8, ls=":", label="ρ = α (identity)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("raw α  (archived parameterization)")
    ax.set_ylabel("normalized ρ = ‖Δh‖ / median‖h_inj‖")
    ax.set_title("Raw α is not architecture-portable\n(legend shows med_obs/med_inj per channel)",
                 fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_alpha_vs_rho.png", dpi=180)
    fig.savefig(FIG / "fig3_alpha_vs_rho.pdf")
    plt.close(fig)


def manifest():
    lines = ["# Phase-7 Figure Manifest\n",
             "All figures use **validation rows only**. No test data enters any panel.\n",
             "Source: `rho_specificity_results.json` (this phase), `rho_calibration.json`.\n"]
    lines += [
        "\n## fig1_val_net_vs_rho.{png,pdf}\n",
        "**Source:** `rho_specificity_results.json` → per channel, per ρ: real/reverse Net and the "
        "20-random Net distribution (mean ± sd band). Blue dotted line = the archived operating ρ.\n",
        "**Claim:** whether validation utility has an *interior* optimum in normalized magnitude, "
        "and whether the real direction separates from matched-norm randoms there.\n",
        "\n## fig2_specificity_z_vs_rho.{png,pdf}\n",
        "**Source:** same; z = (Net_real − mean Net_random)/sd(Net_random), 20 randoms, seed block 2000+k.\n",
        "**Claim:** the central Phase-7 discriminator — does Mistral ca_tc reach z ≥ 2 at a "
        "non-boundary ρ (B′) or at no ρ (A-global)? Shaded band marks the ρ-grid boundary, where a "
        "peak cannot count as interior-specific.\n",
        "\n## fig3_alpha_vs_rho.{png,pdf}\n",
        "**Source:** `rho_calibration.json` train medians + the archived per-channel obs/inj layers.\n",
        "**Claim:** the archived raw α is not a physical magnitude — identical α maps to ρ values "
        "differing by up to ~2.5× across channels (med_obs/med_inj ∈ [0.84, 2.06]), which motivates "
        "the ρ re-parameterization independently of any outcome.\n",
    ]
    (FIG / "FIGURE_MANIFEST.md").write_text("".join(lines))


if __name__ == "__main__":
    fig1(); fig2(); fig3(); manifest()
    print("wrote figures to", FIG)
