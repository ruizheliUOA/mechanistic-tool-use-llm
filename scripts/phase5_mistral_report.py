"""
phase5_mistral_report.py — assemble cross-model W2C synthesis + machine-readable summaries.
============================================================================================
Reads Phase-5 outputs (baseline, discovery, geometry, pilot, placebos, multiseed if present)
plus archived Phi-3.5 / Qwen2.5-7B W2C numbers, and writes:
  final/results/mistral7b_w2c_sakiko_ca/cross_model_w2c_summary.csv
  final/results/mistral7b_w2c_sakiko_ca/CROSS_MODEL_W2C_SYNTHESIS.md

Pure post-processing; no model, no test-set decisions.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L

OUT = L.OUT


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def main():
    base = jload(OUT / "baseline" / "mistral7b_w2c_baseline_summary.json", {})
    disc = jload(OUT / "channel_discovery" / "mistral7b_channel_discovery.json", {})
    geom = (jload(OUT / "geometry" / "mistral7b_geometry.json", {}) or {}).get("geometry", {})
    pilot = jload(OUT / "pilot" / "pilot_summary.json", {})
    controls = jload(OUT / "placebos" / "placebo_all_controls.json", {})
    ms = jload(OUT / "multiseed" / "mistral7b_multiseed_results.json", {})

    # archived baselines / channel structure (from committed files; see PHASE5 audit)
    ARCH = {
        "Phi-3.5-mini": {"acc": 0.4822, "macro_f1_3": 0.4526, "n": 3652,
                         "pred_tool_call": 2470, "channels": {"rfi_tc": 746, "ca_tc": 511, "ca_direct": 393},
                         "selected": ["rfi_tc", "ca_tc", "ca_direct"], "headline_net": None},
        "Qwen2.5-7B": {"acc": 0.4381, "macro_f1_3": 0.4006, "n": 3652,
                       "pred_tool_call": 2244, "channels": {"rfi_tc": 622, "ca_tc": 513, "ca_direct": 465,
                                                            "ca_rfi": 219, "tc_rfi": 133},
                       "selected": ["rfi_tc", "ca_tc", "ca_direct", "ca_rfi"],
                       "rejected_utility": ["tc_rfi"], "headline_net": "+92±20 (5/5)"},
    }
    mis_channels = {c["channel_id_short"]: c["count_all"] for c in disc.get("discovered_channels", [])
                    if c.get("status_default") == "stable"}
    gate = pilot.get("gate", {})
    mis_selected = [c for c in gate if gate[c].get("pass")]
    mis_rejected = [c for c in gate if not gate[c].get("pass")]

    # cross-model CSV
    rows = [("model", "n", "acc", "macro_f1_3", "pred_tool_call_share", "n_stable_channels",
             "selected_channels", "utility_rejected", "headline_net")]
    for name, a in ARCH.items():
        rows.append((name, a["n"], a["acc"], a["macro_f1_3"],
                     round(a["pred_tool_call"] / a["n"], 3), len(a["channels"]),
                     "|".join(a["selected"]), "|".join(a.get("rejected_utility", [])),
                     a["headline_net"] or ""))
    mis_ptc = (base.get("pred_distribution", {}).get("tool_call", 0) / max(base.get("n_eval", 1), 1))
    ms_agg = ms.get("aggregate", {}) if ms else {}
    headline = (f"+{ms_agg['net_mean']:.0f}±{ms_agg['net_std']:.0f} ({ms_agg['n_positive']}/{ms_agg['n_seeds']})"
                if ms_agg else (f"pilot Net {pilot.get('arms', {}).get('D_auto_cascade', {}).get('net', 'NA')}"))
    rows.append(("Mistral-7B-v0.3", base.get("n_eval"), base.get("accuracy"),
                 base.get("macro_f1_3cls"), round(mis_ptc, 3), len(mis_channels),
                 "|".join(mis_selected), "|".join(mis_rejected), headline))
    with open(OUT / "cross_model_w2c_summary.csv", "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # synthesis MD
    with open(OUT / "CROSS_MODEL_W2C_SYNTHESIS.md", "w") as f:
        f.write("# Cross-Model W2C Synthesis — Phi-3.5 vs Qwen2.5-7B vs Mistral-7B-v0.3\n\n")
        f.write("> Normalized layer depth used throughout (Phi 32L, Qwen 28L, Mistral 32L). "
                "Net is reported only alongside baseline error opportunity, test size, channel "
                "support, Fixed and Broke.\n\n")
        f.write("## 1. Baseline error structure\n\n")
        f.write("| model | layers | acc | macro-F1(3) | pred tool_call share | dominant channels (count) |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| Phi-3.5-mini | 32 | 0.4822 | 0.4526 | {2470/3652:.3f} | rfi_tc 746 / ca_tc 511 / ca_direct 393 |\n")
        f.write(f"| Qwen2.5-7B | 28 | 0.4381 | 0.4006 | {2244/3652:.3f} | rfi_tc 622 / ca_tc 513 / ca_direct 465 / ca_rfi 219 |\n")
        ch_str = " / ".join(f"{k} {v}" for k, v in sorted(mis_channels.items(), key=lambda x: -x[1])[:5])
        f.write(f"| Mistral-7B-v0.3 | 32 | {base.get('accuracy')} | {base.get('macro_f1_3cls')} | "
                f"{mis_ptc:.3f} | {ch_str} |\n\n")
        f.write("## 2. Discovered vs selected channels\n\n")
        f.write(f"- Phi selected: {ARCH['Phi-3.5-mini']['selected']}\n")
        f.write(f"- Qwen selected: {ARCH['Qwen2.5-7B']['selected']}; utility-rejected: "
                f"{ARCH['Qwen2.5-7B'].get('rejected_utility')}\n")
        f.write(f"- Mistral stable channels: {sorted(mis_channels)}\n")
        f.write(f"- Mistral scope-passing: {disc.get('scope_filter', {}).get('scope_pass_channels')}\n")
        f.write(f"- Mistral utility-passing (gate): {mis_selected}; utility-rejected: {mis_rejected}\n\n")
        f.write("## 3. Geometry (Mistral, normalized depth)\n\n")
        f.write("| channel | R1 obs (depth) | method | router val-AUC |\n|---|---|---|---|\n")
        for cid, g in geom.items():
            r1 = g.get("R1_obs")
            lk = pilot.get("locked", {}).get(cid, {})
            f.write(f"| {cid} | {('L%d (%.2f)' % (r1, r1/32)) if r1 is not None else 'REJECTED'} | "
                    f"{lk.get('method','—')} | {lk.get('router_val_auc','—')} |\n")
        f.write("\n## 4. Intervention outcome (Mistral pilot, seed-42)\n\n")
        f.write("| arm | channel(s) | Fixed | Broke | Net | Δacc-context |\n|---|---|---|---|---|---|\n")
        for name, arm in pilot.get("arms", {}).items():
            who = arm.get("channel") or ",".join(arm.get("members", [])) or "-"
            f.write(f"| {name} | {who} | {arm.get('fixed','')} | {arm.get('broke','')} | "
                    f"{arm.get('net','')} | acc {arm.get('acc','')} |\n")
        if controls:
            f.write("\n## 5. Placebo specificity (Mistral)\n\n")
            f.write("| channel | real Net | reverse Net | random mean(max) | real pct | verdict |\n|---|---|---|---|---|---|\n")
            for cid, o in controls.items():
                if not o:
                    continue
                r = o["random"]
                verdict = ("direction-specific" if (r["percentile_of_real"] >= 75 and
                           o["real"]["net"] > r["mean"] + r["std"] and o["real"]["net"] > o["reverse"]["net"])
                           else "non-specific/behavioral")
                f.write(f"| {cid} | {o['real']['net']:+d} | {o['reverse']['net']:+d} | "
                        f"{r['mean']:.1f}({r['max']}) | {r['percentile_of_real']:.0f} | {verdict} |\n")
        if ms_agg:
            f.write(f"\n## 6. Mistral multi-seed\n\nNet = {ms_agg['net_mean']:.1f} ± {ms_agg['net_std']:.1f} "
                    f"(min {ms_agg['net_min']}, max {ms_agg['net_max']}); positive "
                    f"{ms_agg['n_positive']}/{ms_agg['n_seeds']}; mean Δacc {ms_agg['delta_acc_mean']:+.4f}.\n")
    print("wrote cross_model_w2c_summary.csv and CROSS_MODEL_W2C_SYNTHESIS.md")


if __name__ == "__main__":
    main()
