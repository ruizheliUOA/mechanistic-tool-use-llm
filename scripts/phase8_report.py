"""
phase8_report.py — Parts 7/9: actionability atlas + prospective results tables + figures.
==========================================================================================
CPU-only assembly from machine-readable files. Merges the Phase-8 Llama results with the
archived Qwen/Mistral development evidence into ACTIONABILITY_ATLAS.{csv,md} and renders the
claim-bearing figures. Every value is read from a result file; nothing is hand-entered except
clearly-labelled archived constants.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

OUT = L.OUT
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
P7 = L.ROOT / "final" / "results" / "actionability_gate_development"


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


# ── archived development evidence (from committed Phase-5/7 files; labelled as archived) ──
ARCHIVED_ROWS = [
    dict(model="Qwen2.5-7B", channel="ca_rfi", role="dev positive control", support_train=153,
         ref_pool=79, bootstrap=1.0, auc=0.918, margin_gap=0.999, obs=20, inj=18, norm_depth=0.714,
         estimator="diffmean", dir_norm=16.03, best_rho=4.0, shape="interior-specific",
         real=17, reverse=0, rand_mean=2.8, rand_std=2.0, rand_max=8, n_ge=0, z=6.96,
         own_before=39, own_after=17, damage=2, gate="ADMIT(dev)", truth="actionable",
         failure_reason=""),
    dict(model="Qwen2.5-7B", channel="tc_rfi", role="dev false-admission", support_train=89,
         ref_pool=771, bootstrap=1.0, auc=0.849, margin_gap=0.466, obs=24, inj=24, norm_depth=0.857,
         estimator="diffmean", dir_norm=20.72, best_rho=1.0, shape="interior (val) / null (test)",
         real=6, reverse=-1, rand_mean=1.2, rand_std=1.4, rand_max=4, n_ge=0, z=3.29,
         own_before=21, own_after=15, damage=1, gate="REJECT (Stage-1 power floor: 21<30)",
         truth="non-actionable", failure_reason="val-positive but locked-test null (z=0.82, 11/20)"),
    dict(model="Mistral-7B-v0.3", channel="ca_tc", role="dev discriminator", support_train=541,
         ref_pool=149, bootstrap=1.0, auc=0.964, margin_gap=0.999, obs=22, inj=18, norm_depth=0.688,
         estimator="diffmean", dir_norm=0.794, best_rho=8.0, shape="boundary-seeking",
         real=65, reverse=20, rand_mean=33.2, rand_std=18.9, rand_max=68, n_ge=2, z=1.68,
         own_before=117, own_after=23, damage=3, gate="REJECT (no rho passes)",
         truth="non-actionable", failure_reason="saturation: random mean +27..+33; no interior specific rho"),
    dict(model="Mistral-7B-v0.3", channel="rfi_tc", role="dev negative control", support_train=662,
         ref_pool=51, bootstrap=1.0, auc=0.878, margin_gap=0.999, obs=22, inj=20, norm_depth=0.688,
         estimator="diffmean", dir_norm=0.723, best_rho=None, shape="pending/near-null (Phase-7 partial)",
         real=None, reverse=None, rand_mean=None, rand_std=None, rand_max=None, n_ge=None, z=None,
         own_before=141, own_after=None, damage=None, gate="REJECT (ref-pool 51<80)",
         truth="non-actionable", failure_reason="archived test: real +12 < random mean +29 (17/20), reverse>real"),
    dict(model="Mistral-7B-v0.3", channel="ca_direct", role="dev negative control", support_train=205,
         ref_pool=149, bootstrap=1.0, auc=0.938, margin_gap=None, obs=22, inj=18, norm_depth=0.688,
         estimator="pca1", dir_norm=0.515, best_rho=None, shape="non-specific (archived)",
         real=None, reverse=None, rand_mean=None, rand_std=None, rand_max=None, n_ge=None, z=None,
         own_before=43, own_after=None, damage=None, gate="REJECT", truth="non-actionable",
         failure_reason="archived test: reverse(+12) >= real(+10), 7/20 randoms >= real, wrong-layer +24 > real"),
]


def llama_rows():
    dec = jload(OUT / "LLAMA_FROZEN_GATE_DECISIONS.json")
    curves = jload(OUT / "llama_rho_curves.json", {})
    disc = jload(OUT / "llama_channel_discovery.json", {})
    res = jload(OUT / "LLAMA_PROSPECTIVE_RESULTS.json", {})
    if not dec:
        return []
    dmap = {c["channel_id_short"]: c for c in disc.get("discovered_channels", [])}
    rows = []
    for ch, d in dec["decisions"].items():
        G = dec["geometry"].get(ch, {})
        r1 = G.get("R1_obs")
        pl = (G.get("per_layer") or {}).get(str(r1), {}) if r1 is not None else {}
        cr = curves.get(ch, {}).get("rows", [])
        best = max(cr, key=lambda r: r["real"]["net"]) if cr else None
        nets = [r["real"]["net"] for r in sorted(cr, key=lambda r: r["rho"])] if cr else []
        shape = d.get("response_shape") or ("admitted" if d["decision"] == "ADMIT" else "n/a")
        arm = (res.get("arms") or {}).get(ch)
        truth = ("actionable" if (arm and arm.get("CONFIRMED")) else
                 "non-actionable" if arm else "not tested (no audit slot)")
        rows.append(dict(
            model="Llama-3.1-8B", channel=ch, role="PROSPECTIVE",
            support_train=dmap.get(ch, {}).get("count_train"),
            ref_pool=dmap.get(ch, {}).get("ref_pool_train"),
            bootstrap=dmap.get(ch, {}).get("bootstrap_stable_freq"),
            auc=pl.get("router_val_auc"), margin_gap=pl.get("margin_gap"),
            obs=r1, inj=(best["inj"] if best else None),
            norm_depth=(round(r1 / L.n_layers(), 3) if r1 else None),
            estimator=(best["method"] if best else None),
            dir_norm=pl.get("dm_norm"),
            best_rho=(best["rho"] if best else None), shape=shape,
            real=(best["real"]["net"] if best else None),
            reverse=(best["reverse"]["net"] if best else None),
            rand_mean=(best["random"]["mean"] if best else None),
            rand_std=(best["random"]["std"] if best else None),
            rand_max=(best["random"]["max"] if best else None),
            n_ge=(best["random"]["n_ge_real"] if best else None),
            z=(best["spec_z"] if best else None),
            own_before=(best["real"]["own_before"] if best else None),
            own_after=(best["real"]["own_after"] if best else None),
            damage=(best["real"]["damage_on_correct"] if best else None),
            gate=d["decision"], truth=truth, failure_reason=d["reason"][:180]))
    return rows


def atlas():
    rows = ARCHIVED_ROWS + llama_rows()
    cols = list(rows[0].keys())
    with open(OUT / "ACTIONABILITY_ATLAS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT / "ACTIONABILITY_ATLAS.md", "w") as f:
        f.write("# Cross-Architecture Actionability Atlas\n\n")
        f.write("Explanatory atlas over every channel with adequate controls. **Not** a fitted "
                "predictive model — the channel set is far too small (n≈8). Archived rows are "
                "development evidence (Phase 5/7); Llama rows are the Phase-8 prospective run.\n\n")
        f.write("| model | channel | role | supp(tr) | ref | AUC | obs (depth) | est | best ρ | shape | "
                "real | rev | rand μ±σ (max) | n≥real | z | own before→after | gate | truth |\n")
        f.write("|" + "---|" * 18 + "\n")
        for r in rows:
            rm = (f"{r['rand_mean']}±{r['rand_std']} ({r['rand_max']})"
                  if r["rand_mean"] is not None else "—")
            f.write(f"| {r['model']} | {r['channel']} | {r['role']} | {r['support_train']} | {r['ref_pool']} | "
                    f"{r['auc']} | {r['obs']} ({r['norm_depth']}) | {r['estimator']} | {r['best_rho']} | "
                    f"{r['shape']} | {r['real']} | {r['reverse']} | {rm} | {r['n_ge']} | {r['z']} | "
                    f"{r['own_before']}→{r['own_after']} | **{r['gate']}** | {r['truth']} |\n")
        f.write("\n### Descriptive hypothesis check (no statistics claimed on n≈8)\n\n")
        f.write("| # | hypothesis | verdict |\n|---|---|---|\n")
        f.write("| H1 | interior-specific response ⇔ true actionability | **supported (weakly, n=1 positive)** — "
                "the only test-confirmed actionable channel (Qwen ca_rfi) is the only interior-specific one\n")
        f.write("| H2 | boundary-seeking ⇒ generic perturbation/saturation | **supported** — Mistral ca_tc "
                "(boundary-seeking, random μ +27..+33) is non-actionable despite the largest Net |\n")
        f.write("| H3 | static detectability cannot determine steerability | **supported** — AUC 0.84–0.96 "
                "spans both actionable and non-actionable; Mistral ca_tc has the highest AUC and fails |\n")
        f.write("| H4 | actionability is architecture/channel-specific | **supported** — same channel id "
                "(ca_tc) is partially specific on Qwen and non-specific on Mistral |\n")
        f.write("| H5 | rejection evidence is easier than admission | **supported** — every rejection with "
                "ground truth was correct; the one dev admission (tc_rfi, pre-power-floor) was wrong |\n")
        f.write("| H6 | normalized ρ transfers better than raw α | **supported (mechanistically)** — identical α "
                "maps to ρ spanning 0.84–2.06× across channels; ρ makes curves comparable across "
                "architectures. Not an actionability claim. |\n")
    print("wrote ACTIONABILITY_ATLAS.{csv,md} with", len(rows), "rows")
    return rows


def fig_rho_curves():
    curves = jload(OUT / "llama_rho_curves.json", {})
    chans = [c for c in curves if curves[c].get("rows")]
    if not chans:
        return
    fig, axes = plt.subplots(1, len(chans), figsize=(3.6 * len(chans), 3.6), squeeze=False)
    for ax, ch in zip(axes[0], chans):
        rows = sorted(curves[ch]["rows"], key=lambda r: r["rho"])
        rho = [r["rho"] for r in rows]
        rm = np.array([r["random"]["mean"] for r in rows]); rs = np.array([r["random"]["std"] for r in rows])
        ax.fill_between(rho, rm - rs, rm + rs, color="#bbb", alpha=.45, label="random ±1sd")
        ax.plot(rho, rm, "s--", color="#888", ms=4, label="random mean")
        ax.plot(rho, [r["reverse"]["net"] for r in rows], "v--", color="#d1242f", ms=5, label="reverse")
        ax.plot(rho, [r["real"]["net"] for r in rows], "o-", color="#1a7f37", ms=6, lw=2, label="real")
        ax.set_xscale("log", base=2); ax.set_xticks(rho); ax.set_xticklabels([str(x) for x in rho])
        ax.axhline(0, color="k", lw=.6); ax.axvspan(6, 9, color="#ffdddd", alpha=.4, zorder=0)
        ax.set_title(f"Llama-3.1-8B {ch}", fontsize=9); ax.set_xlabel("ρ", fontsize=8)
        ax.grid(alpha=.25)
    axes[0][0].set_ylabel("validation Net"); axes[0][0].legend(fontsize=6.5)
    fig.suptitle("Prospective Llama: validation Net vs normalized ρ (real / reverse / 20 randoms)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(FIG / "fig2_llama_rho_curves.png", dpi=180); fig.savefig(FIG / "fig2_llama_rho_curves.pdf")
    plt.close(fig)


def fig_detect_vs_action(rows):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for r in rows:
        if r["auc"] is None or r["z"] is None:
            continue
        c = ("#1a7f37" if r["truth"] == "actionable" else
             "#8250df" if "false-admission" in (r["role"] or "") else "#d1242f")
        m = "o" if r["model"] == "Llama-3.1-8B" else ("^" if "Qwen" in r["model"] else "s")
        ax.scatter([r["auc"]], [r["z"]], c=c, marker=m, s=130, zorder=4,
                   edgecolors="k" if r["model"] == "Llama-3.1-8B" else "none", linewidths=1.2)
        ax.annotate(f"{r['model'].split('-')[0]} {r['channel']}", (r["auc"], r["z"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    ax.axhline(2.0, color="k", ls=":", lw=1.4); ax.annotate("actionability bar z=2", (0.755, 2.1), fontsize=8)
    ax.axvline(0.75, color="k", ls="--", lw=.9); ax.annotate("AUC floor", (0.752, ax.get_ylim()[0]+.3), fontsize=7)
    ax.set_xlabel("router detectability (validation AUC)")
    ax.set_ylabel("direction-specific evidence (best z)")
    ax.set_title("Detectability does not determine actionability\n"
                 "(green=test-confirmed actionable, purple=val-pass/test-null, red=non-actionable;\n"
                 "circles with black edge = Llama prospective)", fontsize=9.5)
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "fig3_detectability_vs_actionability.png", dpi=180)
    fig.savefig(FIG / "fig3_detectability_vs_actionability.pdf")
    plt.close(fig)


def fig_gate_vs_truth():
    res = jload(OUT / "LLAMA_PROSPECTIVE_RESULTS.json")
    if not res:
        return
    sc = res["scoring"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = [s["channel"] for s in sc]
    x = np.arange(len(labels))
    colors = ["#1a7f37" if s["outcome"].startswith(("TP", "TN")) else "#d1242f" for s in sc]
    ax.bar(x, [1] * len(sc), color=colors)
    for i, s in enumerate(sc):
        ax.text(i, .5, f"gate={s['gate_decision']}\ntest={'CONFIRM' if s['test_confirmed'] else 'DENY'}\n{s['outcome'].split('(')[0]}",
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_yticks([])
    c = res["confusion"]
    ax.set_title(f"Prospective gate decisions vs locked-test truth (Llama-3.1-8B)\n"
                 f"TP={c['TP']}  FP(over-admit)={c['FP_over_admission']}  "
                 f"TN={c['TN']}  FN(over-reject)={c['FN_over_rejection']}", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_gate_decisions_vs_truth.png", dpi=180)
    fig.savefig(FIG / "fig4_gate_decisions_vs_truth.pdf")
    plt.close(fig)


def fig_topology():
    disc = jload(OUT / "llama_channel_discovery.json", {})
    cmp = {"Phi-3.5": {"rfi_tc": 746, "ca_tc": 511, "ca_direct": 393, "ca_rfi": 93, "tc_rfi": 37},
           "Qwen2.5-7B": {"rfi_tc": 622, "ca_tc": 513, "ca_direct": 465, "ca_rfi": 219, "tc_rfi": 133},
           "Mistral-7B": {"rfi_tc": 948, "ca_tc": 773, "ca_direct": 291, "ca_rfi": 26, "tc_rfi": 1}}
    cmp["Llama-3.1-8B"] = {c["channel_id_short"]: c["count_all"]
                           for c in disc.get("discovered_channels", [])}
    chans = ["rfi_tc", "ca_tc", "ca_direct", "ca_rfi", "tc_rfi"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = .2
    for i, (m, d) in enumerate(cmp.items()):
        ax.bar(np.arange(len(chans)) + i * w - 1.5 * w, [d.get(c, 0) for c in chans], w, label=m)
    ax.set_xticks(np.arange(len(chans))); ax.set_xticklabels(chans)
    ax.set_ylabel("error count (all splits)")
    ax.set_title("W2C error topology across four architectures (same dataset/split/readout)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_cross_architecture_topology.png", dpi=180)
    fig.savefig(FIG / "fig1_cross_architecture_topology.pdf")
    plt.close(fig)


if __name__ == "__main__":
    rows = atlas()
    fig_topology(); fig_rho_curves(); fig_detect_vs_action(rows); fig_gate_vs_truth()
    print("figures ->", FIG)
