"""
make_acebench_readout_figures.py — figures for the Phase-4 ACEBench generation-readout package.
================================================================================================
Every figure is backed by a CSV/JSON already produced under
final/results/acebench_generation_readout/. No model, no activations, no generation here — pure
plotting from the derived tables. Writes a figure_manifest.csv mapping each figure to its source.

Usage:  python scripts/make_acebench_readout_figures.py
"""
from __future__ import annotations
import csv, json, sys, logging
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "final" / "results" / "acebench_generation_readout"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("figures")
MANIFEST: list[dict] = []
GOLD = ["tool_call", "ask_user", "flag_param_error", "cannot_comply"]
PRED = GOLD + ["UNKNOWN"]
SCOPE = {"train": 30, "val": 5, "test": 5}


def record(fname, source, desc, status="generated", notes=""):
    MANIFEST.append({"figure_file": fname, "source_table": source, "description": desc,
                     "status": status, "notes": notes})


def save(fig, fname):
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)
    log.info("wrote %s", fname)


def fig1_confusion(summary):
    """Gold x pred confusion of the locked 800-row baseline."""
    conf = summary["confusion_gold_by_pred"]
    m = np.array([[conf[g][p] for p in PRED] for g in GOLD], float)
    row = m.sum(1, keepdims=True)
    frac = m / np.clip(row, 1, None)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(PRED)), PRED, rotation=30, ha="right")
    ax.set_yticks(range(len(GOLD)), GOLD)
    for i in range(len(GOLD)):
        for j in range(len(PRED)):
            ax.text(j, i, f"{int(m[i, j])}\n{frac[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted (parser v1.0, gold-blind)")
    ax.set_ylabel("gold mode")
    ax.set_title("ACEBench generation readout — confusion (n=800)\n"
                 f"acc {summary['accuracy']:.3f} · macro-F1 {summary['macro_f1']:.3f} · "
                 f"bal-acc {summary['balanced_accuracy']:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, label="row-normalised share")
    save(fig, "fig1_confusion_matrix.png")
    record("fig1_confusion_matrix.png", "baseline/acebench_generation_baseline_summary.json",
           "Gold x predicted confusion for the locked 800-row generation baseline.")


def fig2_pred_dist(summary):
    """Prediction distribution vs the R0.3 collapse thresholds, with the prior failed readout."""
    pdist = summary["prediction_distribution"]
    labs = [p for p in PRED if p in pdist]
    vals = [pdist[p] / summary["n"] for p in labs]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labs, vals, color=["#3b6ea5" if l != "UNKNOWN" else "#999999" for l in labs])
    ax.axhline(0.70, ls="--", c="crimson", lw=1.2, label="R0.3 collapse ceiling (0.70)")
    ax.axhline(0.745, ls=":", c="darkred", lw=1.2,
               label="prior candidate-scoring readout (0.745 — FAILED)")
    ax.axhline(0.05, ls="--", c="grey", lw=0.9, label="R0.3 floor: modes >=5%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_ylabel("share of predictions")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(range(len(labs)), labs, rotation=20, ha="right")
    ax.set_title("R0.3 — no collapse: max share 0.675, all 4 modes >=5%", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "fig2_prediction_distribution.png")
    record("fig2_prediction_distribution.png",
           "baseline/acebench_generation_baseline_summary.json",
           "Predicted-mode distribution against the locked R0.3 collapse thresholds.")


def fig3_parser_breakdown(details):
    """Parser outcome breakdown: how each label was reached (rule tier) + UNKNOWN composition."""
    tiers = Counter()
    for r in details:
        rp = r["rule_path"]
        tier = ("official template" if rp.startswith("bracket.official")
                else "canonical call list" if rp == "bracket.call_list"
                else "fallback cue (flagged)" if ".fallback." in rp
                else "UNKNOWN — abstained")
        tiers[tier] += 1
    unk = [r for r in details if r["pred"] == "UNKNOWN"]
    comp = Counter("truncated at 256 tok" if r["finish"] == "length"
                   else "quoted-call form" if (r["output"] or "").lstrip().startswith('["')
                   else "malformed / other" for r in unk)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    order = ["canonical call list", "official template", "fallback cue (flagged)",
             "UNKNOWN — abstained"]
    cols = ["#3b6ea5", "#4c9f70", "#e0a458", "#999999"]
    axes[0].barh(order, [tiers[o] for o in order], color=cols)
    for i, o in enumerate(order):
        axes[0].text(tiers[o] + 6, i, str(tiers[o]), va="center", fontsize=8)
    axes[0].set_xlabel("rows")
    axes[0].set_title("how the parser reached each label (n=800)\n"
                      "99.3% of native-label errors via official/canonical rules", fontsize=9)
    axes[0].invert_yaxis()
    ck = list(comp.keys())
    axes[1].barh(ck, [comp[k] for k in ck], color="#999999")
    for i, k in enumerate(ck):
        axes[1].text(comp[k] + 0.4, i, str(comp[k]), va="center", fontsize=8)
    axes[1].set_xlabel("rows")
    axes[1].set_title(f"UNKNOWN composition (n={len(unk)}, 7.25%)\n"
                      "abstentions on malformed/truncated call attempts", fontsize=9)
    axes[1].invert_yaxis()
    save(fig, "fig3_parser_breakdown.png")
    record("fig3_parser_breakdown.png", ".cache/acebench_phase4/generation_details_full.jsonl",
           "Parser rule-tier breakdown and UNKNOWN composition (source is gitignored raw detail).")


def fig4_transitions(cd):
    """R0.9: per-split transition support against the intervention-scope thresholds."""
    ts = [t for t in cd["transitions"] if t["count_all"] >= 5]
    names = [f"{t['gold']}→{t['pred']}" for t in ts]
    x = np.arange(len(ts))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for k, (split, col) in enumerate([("train", "#3b6ea5"), ("val", "#4c9f70"),
                                      ("test", "#e0a458")]):
        v = [t[f"count_{split}"] for t in ts]
        ax.bar(x + (k - 1) * w, v, w, label=f"{split} (need >={SCOPE[split]})", color=col)
        ax.axhline(SCOPE[split], ls="--", lw=0.8, c=col, alpha=0.8)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 0.5, str(vi), ha="center", fontsize=7)
    ax.set_xticks(x, names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("rows in channel")
    ax.set_title("R0.9 FAIL — 0 of >=2 transitions clear train>=30 & val>=5 & test>=5\n"
                 "bootstrap scope-stability: 0.31 / 0.215 / 0.0 (need >=0.80)", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, "fig4_transition_support.png")
    record("fig4_transition_support.png", "channel_discovery/acebench_channel_discovery.json",
           "Per-split error-transition support vs the locked intervention-scope thresholds (R0.9).")


def fig5_paraphrase(para):
    """The eligibility gate: per-mode agreement under a semantically equivalent prompt."""
    modes = GOLD
    agree = [para["per_mode_agreement"][m]["agreement"] for m in modes]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    cols = ["#4c9f70" if a >= 0.8 else "#c0504d" for a in agree]
    axes[0].bar(modes, agree, color=cols)
    axes[0].axhline(0.80, ls="--", c="crimson", lw=1.2, label="gate >=0.80")
    axes[0].axhline(para["agreement_all"], ls=":", c="black", lw=1.1,
                    label=f"overall {para['agreement_all']:.2f} (FAIL)")
    for i, a in enumerate(agree):
        axes[0].text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("agreement with locked baseline")
    axes[0].set_xticks(range(len(modes)), modes, rotation=20, ha="right")
    axes[0].set_title("paraphrase check FAIL — restraint modes are the fragile ones", fontsize=9)
    axes[0].legend(fontsize=7)
    bd, pdd = para["base_pred_dist"], para["para_pred_dist"]
    x = np.arange(len(PRED))
    axes[1].bar(x - 0.2, [bd.get(p, 0) for p in PRED], 0.4, label="official prompt",
                color="#3b6ea5")
    axes[1].bar(x + 0.2, [pdd.get(p, 0) for p in PRED], 0.4, label="paraphrased prompt",
                color="#e0a458")
    axes[1].set_xticks(x, PRED, rotation=20, ha="right")
    axes[1].set_ylabel("predictions (n=100)")
    axes[1].set_title("ask_user collapses 20→4; UNKNOWN 2→32\n"
                      "(29/30 are quoted-call attempts → +22pts toward calling)", fontsize=9)
    axes[1].legend(fontsize=7)
    save(fig, "fig5_paraphrase_agreement.png")
    record("fig5_paraphrase_agreement.png", "paraphrase_check/paraphrase_agreement.json",
           "Pre-registered paraphrase eligibility gate: per-mode agreement and distribution shift.")


def main():
    summary = json.loads((RES / "baseline/acebench_generation_baseline_summary.json").read_text())
    cd = json.loads((RES / "channel_discovery/acebench_channel_discovery.json").read_text())
    para = json.loads((RES / "paraphrase_check/paraphrase_agreement.json").read_text())
    det_path = ROOT / ".cache/acebench_phase4/generation_details_full.jsonl"
    details = [json.loads(l) for l in open(det_path, encoding="utf-8")]

    fig1_confusion(summary)
    fig2_pred_dist(summary)
    fig3_parser_breakdown(details)
    fig4_transitions(cd)
    fig5_paraphrase(para)

    with open(FIG / "figure_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure_file", "source_table", "description", "status",
                                          "notes"])
        w.writeheader()
        w.writerows(MANIFEST)
    log.info("figure_manifest.csv: %d figures", len(MANIFEST))


if __name__ == "__main__":
    main()
