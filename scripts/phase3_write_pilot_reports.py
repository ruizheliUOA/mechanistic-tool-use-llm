"""
phase3_write_pilot_reports.py — CPU-only: emit the pilot + placebo deliverable files from _state.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L

PIL = L.OUT / "pilot"; PLA = L.OUT / "placebos"
st = json.loads((PIL / "_state.json").read_text())
summary = json.loads((PIL / "pilot_summary.json").read_text())
controls = json.loads((PLA / "placebo_all_controls.json").read_text())
locked = summary["locked"]; gate = summary["gate"]; arms = st["arms"]
ARMS = {"ca_rfi": "B_ca_rfi_only", "tc_rfi": "C_tc_rfi_only"}
NAMES = {"ca_rfi": "cannot_answer→request_for_info", "tc_rfi": "tool_call→request_for_info"}

# ---- pilot_details.jsonl : per-arm per-metric flat rows ----
with open(PIL / "pilot_details.jsonl", "w") as f:
    f.write(json.dumps({"arm": "A_baseline", **{k: arms["A_baseline"][k]
            for k in ("acc", "macro_f1_3cls", "nt_before")}}) + "\n")
    for ch, an in ARMS.items():
        a = arms.get(an, {})
        if a.get("status") == "NOT_RUN":
            f.write(json.dumps({"arm": an, "channel": ch, "status": "NOT_RUN"}) + "\n"); continue
        f.write(json.dumps({"arm": an, "channel": ch,
                "locked": locked.get(ch), **{k: a[k] for k in
                ("acc", "macro_f1_3cls", "fixed", "broke", "net", "n_touched",
                 "touched_correct", "damage_on_correct", "own_errors_touched",
                 "own_to_gold", "own_to_other_wrong", "nt_before", "nt_after",
                 "chan_fixed", "destination_redistribution", "transition_matrix_after")}}) + "\n")

# ---- pilot_key_table.csv ----
with open(PIL / "pilot_key_table.csv", "w") as f:
    f.write("arm,channel,net,fixed,broke,acc,own_before,own_after,own_fixed,touched,"
            "damage_on_correct,val_net,gate_pass\n")
    base = arms["A_baseline"]
    f.write(f"A_baseline,,0,0,0,{base['acc']},,,,0,0,,\n")
    for ch, an in ARMS.items():
        a = arms.get(an, {})
        if a.get("status") == "NOT_RUN":
            f.write(f"{an},{ch},NOT_RUN\n"); continue
        f.write(f"{an},{ch},{a['net']},{a['fixed']},{a['broke']},{a['acc']},"
                f"{a['nt_before'][ch]},{a['nt_after'][ch]},{a['chan_fixed'][ch]},"
                f"{a['n_touched']},{a['damage_on_correct']},{locked[ch]['val_net']},"
                f"{gate[ch]['pass']}\n")

# ---- separated placebo jsons ----
json.dump({ch: controls[ch]["random"] for ch in controls},
          open(PLA / "random_direction_results.json", "w"), indent=2)
json.dump({ch: controls[ch]["reverse"] for ch in controls},
          open(PLA / "reverse_results.json", "w"), indent=2)
json.dump({ch: controls[ch]["ungated"] for ch in controls},
          open(PLA / "ungated_results.json", "w"), indent=2)
json.dump({ch: controls[ch].get("wrong_layer") for ch in controls},
          open(PLA / "wrong_layer_results.json", "w"), indent=2)

# ---- PHASE3_NEW_CHANNEL_PILOT_SUMMARY.md ----
with open(PIL / "PHASE3_NEW_CHANNEL_PILOT_SUMMARY.md", "w") as f:
    f.write("# Phase 3 — New-Channel Seed-42 Pilot Summary\n\n")
    f.write("Baseline (Arm A, test n=548): acc **{:.4f}**, macro-F1(3cls) {:.4f}. "
            "Regenerated baseline (bf16, this session; see baseline_regen/ adjudication).\n\n"
            .format(arms["A_baseline"]["acc"], arms["A_baseline"]["macro_f1_3cls"]))
    f.write("## Locked configs (validation-selected, frozen before test)\n\n")
    f.write("| channel | obs | inj | α | thr | method | router val AUC | val Net |\n|---|---|---|---|---|---|---|---|\n")
    for ch in ARMS:
        c = locked.get(ch)
        if c:
            f.write(f"| {ch} | L{c['obs']} | L{c['inj']} | {c['alpha']} | {c['thr']} | "
                    f"{c['method']} | {c['router_val_auc']:.3f} | {c['val_net']:+d} |\n")
    f.write("\n## Locked-test results (once per arm)\n\n")
    f.write("| arm | channel | Net | Fixed | Broke | acc | own before→after | touched | dmg/correct |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    for ch, an in ARMS.items():
        a = arms.get(an, {})
        if a.get("status") == "NOT_RUN":
            f.write(f"| {an} | {ch} | NOT_RUN | | | | | | |\n"); continue
        f.write(f"| {an} | {ch} ({NAMES[ch]}) | **{a['net']:+d}** | {a['fixed']} | {a['broke']} | "
                f"{a['acc']:.4f} | {a['nt_before'][ch]}→{a['nt_after'][ch]} | {a['n_touched']} "
                f"| {a['damage_on_correct']}/{a['touched_correct']} |\n")
    f.write("\n## Error-moving check (own touched errors: →gold vs →other wrong)\n\n")
    f.write("| channel | →gold | →other-wrong | verdict |\n|---|---|---|---|\n")
    for ch, an in ARMS.items():
        a = arms.get(an, {})
        if a.get("status") == "NOT_RUN":
            continue
        v = "fixes" if a["own_to_gold"] > a["own_to_other_wrong"] else "redirects"
        f.write(f"| {ch} | {a['own_to_gold']} | {a['own_to_other_wrong']} | {v} |\n")
    f.write("\n## Pilot gate\n\n| channel | gate | failing criteria |\n|---|---|---|\n")
    for ch in ARMS:
        g = gate[ch]
        if not g.get("eligible", True) or "criteria" not in g:
            f.write(f"| {ch} | {g.get('reason','—')} | — |\n"); continue
        fails = [k for k, v in g["criteria"].items() if not v]
        f.write(f"| {ch} | **{'PASS' if g['pass'] else 'FAIL'}** | {', '.join(fails) or 'none'} |\n")
    passed = [ch for ch in ARMS if gate[ch].get("pass")]
    f.write(f"\n**Outcome:** {len(passed)} channel(s) pass → "
            f"{'4-channel' if len(passed)==1 else '5-channel' if len(passed)==2 else 'no'} "
            f"cascade extension. Passing: {passed}.\n")

# ---- PLACEBO_AND_SPECIFICITY_SUMMARY.md ----
with open(PLA / "PLACEBO_AND_SPECIFICITY_SUMMARY.md", "w") as f:
    f.write("# Phase 3 — Placebo & Specificity Controls (test, locked configs)\n\n")
    f.write("| channel | real Net | reverse Net | random mean±std (max) | real pctile | #rand≥real "
            "| ungated Net (broke) | wrong-layer Net | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    for ch in controls:
        c = controls[ch]; r = c["random"]
        real = c["real"]["net"]
        if r["percentile_of_real"] >= 75 and real > r["mean"] + r["std"] and \
           c["real"]["net"] > c["reverse"]["net"]:
            verdict = "direction-specific"
        elif real > 0 and r["n_ge_real"] > r["n"] * 0.25:
            verdict = "behavioural/non-specific"
        else:
            verdict = "null/non-specific"
        f.write(f"| {ch} | **{real:+d}** | {c['reverse']['net']:+d} | "
                f"{r['mean']:.2f}±{r['std']:.2f} ({r['max']}) | {r['percentile_of_real']:.0f} "
                f"| {r['n_ge_real']}/{r['n']} | {c['ungated']['net']:+d} ({c['ungated']['broke']}) "
                f"| {c['wrong_layer']['net']:+d} | **{verdict}** |\n")
    f.write("\n**Interpretation grid (PHASE3_PROTOCOL §10/§14):** direction-specific = real ≫ random, "
            "reverse collapses, correct layer needed. non-specific = random matches real.\n")

print("pilot + placebo deliverables written")
