"""
phase8_discovery.py — Part 2: automatic channel discovery + registered predictions.
====================================================================================
Reuses the frozen discovery engine (discover_sakiko_channels.transition_discovery) on the
target model's baseline predictions, applies the frozen scope filter + ref-pool/power floors,
compares topology to Phi/Qwen/Mistral, and writes the pre-intervention REGISTERED PREDICTIONS
(fragility F1/F2 + declared expected outcome) — all before any activation extraction.

Outputs: llama_channel_discovery.{json,csv}, LLAMA_CHANNEL_TOPOLOGY_REPORT.md,
registered_predictions.json (+ hash).
"""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L
from discover_sakiko_channels import transition_discovery

OUT = L.OUT
SHORT = {("request_for_info", "tool_call"): "rfi_tc", ("cannot_answer", "tool_call"): "ca_tc",
         ("cannot_answer", "direct"): "ca_direct", ("cannot_answer", "request_for_info"): "ca_rfi",
         ("tool_call", "request_for_info"): "tc_rfi", ("tool_call", "direct"): "tc_direct",
         ("request_for_info", "direct"): "rfi_direct", ("request_for_info", "cannot_answer"): "rfi_ca",
         ("tool_call", "cannot_answer"): "tc_ca", ("direct", "tool_call"): "d_tc"}
SCOPE_TRAIN_MIN, SCOPE_EVAL_MIN = 30, 8
GOLD = set(L.GOLD_CLASSES)


def sid(g, p):
    return SHORT.get((g, p), f"{g}__{p}")


def main():
    ds, meta = L.load_meta()
    tr, va, te = L.splits_all()
    split_of = {}
    for nm, idxs in [("train", tr), ("val", va), ("test", te)]:
        for i in idxs:
            split_of[i] = nm
    rows = [{"gold": m["gold"], "pred": m["pred"], "split": split_of[m["idx"]]} for m in meta]
    res, _ = transition_discovery(rows, dataset="w2c_llama31_8b", model="Llama-3.1-8B-Instruct",
                                  archived=["rfi_tc", "ca_tc", "ca_direct"], note="Phase-8 prospective")

    # train correct-gold reference pools (for ref-pool floor)
    ref_pool = Counter(meta[i]["gold"] for i in tr if meta[i]["correct"])
    # own-channel VALIDATION error counts (for power floor) — val predictions from baseline
    val_err = Counter((meta[i]["gold"], meta[i]["pred"]) for i in va if not meta[i]["correct"])

    chans = []
    for c in res["discovered_channels"]:
        g, p = c["gold"], c["pred"]; cid = sid(g, p)
        n_ref = int(ref_pool.get(g, 0)); n_val_err = int(val_err.get((g, p), 0))
        scope_ok = (g in GOLD and p != g and (c["count_train"] or 0) >= SCOPE_TRAIN_MIN
                    and (c["count_val"] or 0) >= SCOPE_EVAL_MIN and (c["count_test"] or 0) >= SCOPE_EVAL_MIN
                    and c["status_default"] in ("stable", "weak"))
        ref_ok = n_ref >= L.REF_FLOOR
        power_ok = n_val_err >= L.POWER_FLOOR
        eligible = scope_ok and ref_ok and power_ok
        reason = []
        if g not in GOLD: reason.append("gold_not_intervenable")
        if (c["count_train"] or 0) < SCOPE_TRAIN_MIN: reason.append(f"train<{SCOPE_TRAIN_MIN}")
        if (c["count_val"] or 0) < SCOPE_EVAL_MIN or (c["count_test"] or 0) < SCOPE_EVAL_MIN: reason.append("eval<8")
        if c["status_default"] not in ("stable", "weak"): reason.append(f"status={c['status_default']}")
        if not ref_ok: reason.append(f"ref_pool={n_ref}<{L.REF_FLOOR}")
        if not power_ok: reason.append(f"own_val_err={n_val_err}<{L.POWER_FLOOR}")
        chans.append({**c, "channel_id_short": cid, "ref_pool_train": n_ref,
                      "own_val_errors": n_val_err, "scope_pass": bool(scope_ok),
                      "ref_pool_pass": bool(ref_ok), "power_pass": bool(power_ok),
                      "stage1_eligible": bool(eligible),
                      "eligibility_reason": ("eligible" if eligible else "; ".join(reason))})

    eligible_ids = [c["channel_id_short"] for c in chans if c["stage1_eligible"]]
    out = {"model": "Llama-3.1-8B-Instruct", "revision": L.MODELS[L.MODEL_KEY]["revision"],
           "baseline_accuracy": res["baseline_accuracy"], "total_errors": res["total_errors"],
           "baseline_validity_R0": res["baseline_validity"],
           "confusion_matrix_gold_by_pred": res["confusion_matrix_gold_by_pred"],
           "discovered_channels": chans, "n_bootstrap": res["n_bootstrap"],
           "stage1_eligible_channels": eligible_ids,
           "floors": {"scope_train_min": SCOPE_TRAIN_MIN, "scope_eval_min": SCOPE_EVAL_MIN,
                      "ref_pool_floor": L.REF_FLOOR, "power_floor": L.POWER_FLOOR}}
    json.dump(out, open(OUT / "llama_channel_discovery.json", "w"), indent=2)
    with open(OUT / "llama_channel_discovery.csv", "w") as f:
        f.write("channel,gold,pred,count_all,count_train,count_val,count_test,pct_err,status,"
                "bootstrap,ref_pool,own_val_err,scope_pass,ref_pass,power_pass,stage1_eligible,reason\n")
        for c in chans:
            f.write(f"{c['channel_id_short']},{c['gold']},{c['pred']},{c['count_all']},{c['count_train']},"
                    f"{c['count_val']},{c['count_test']},{c['pct_of_errors']},{c['status_default']},"
                    f"{c['bootstrap_stable_freq']},{c['ref_pool_train']},{c['own_val_errors']},"
                    f"{c['scope_pass']},{c['ref_pool_pass']},{c['power_pass']},{c['stage1_eligible']},"
                    f"\"{c['eligibility_reason']}\"\n")

    # ── registered predictions (fragility, BEFORE intervention) ──
    reg = {"note": "Registered per-channel predictions written before activation extraction/intervention.",
           "fragility_rule": "F1 = P(gold is runner-up) on TRAIN channel errors; F2 = median(top-gold) avg_logp/token. "
                             "FRAGILE if F1 >= 0.35 (Phase-6 dev separation). Declared expected outcome uses the "
                             "Phase-7 finding: fragile/high-F1 channels are predicted NON-actionable.",
           "channels": {}}
    for c in chans:
        if not c["stage1_eligible"]:
            continue
        g, p = c["gold"], c["pred"]; et = f"{g}__{p}"
        errs = [i for i in tr if meta[i]["etype"] == et]
        f1 = f2 = None
        if errs:
            det = {r["uuid"]: r for r in [json.loads(l) for l in open(L.BASE_DETAILS)]}
            runner = topgold = []
            runner_hits = 0; gold_gaps = []
            for i in errs:
                lp = det[meta[i]["uuid"]]["avg_logp"]
                order = sorted(lp.items(), key=lambda kv: -kv[1])
                if order[1][0] == g:
                    runner_hits += 1
                gold_gaps.append(order[0][1] - lp[g])
            f1 = round(runner_hits / len(errs), 3); f2 = round(float(np.median(gold_gaps)), 4)
        fragile = (f1 is not None and f1 >= 0.35)
        reg["channels"][c["channel_id_short"]] = {
            "gold": g, "pred": p, "F1_P_gold_runnerup": f1, "F2_median_top_minus_gold": f2,
            "fragile_flag": bool(fragile),
            "declared_expected_outcome": ("non-actionable" if fragile else "actionable-candidate"),
            "n_train_errors": len(errs)}
    (OUT / "registered_predictions.json").write_text(json.dumps(reg, indent=2))
    h = hashlib.sha256((OUT / "registered_predictions.json").read_bytes()).hexdigest()
    (OUT / "registered_predictions.sha256").write_text(h + "\n")

    _report(out, chans)
    print("R0 validity:", out["baseline_validity_R0"].get("readout_collapse_flag"))
    print("stage1-eligible channels:", eligible_ids)
    print("registered_predictions sha256:", h[:16], "...")


def _cmp_topology():
    return {"phi35": {"rfi_tc": 746, "ca_tc": 511, "ca_direct": 393},
            "qwen25_7b": {"rfi_tc": 622, "ca_tc": 513, "ca_direct": 465, "ca_rfi": 219, "tc_rfi": 133},
            "mistral7b_v03": {"rfi_tc": 948, "ca_tc": 773, "ca_direct": 291}}


def _report(out, chans):
    cmp = _cmp_topology()
    mis = {c["channel_id_short"]: c["count_all"] for c in chans if c["status_default"] == "stable"}
    with open(OUT / "LLAMA_CHANNEL_TOPOLOGY_REPORT.md", "w") as f:
        f.write("# Llama-3.1-8B — W2C Channel Topology (Phase 8)\n\n")
        v = out["baseline_validity_R0"]
        f.write(f"- baseline acc {out['baseline_accuracy']:.4f}; total errors {out['total_errors']}; "
                f"collapse flag {v['readout_collapse_flag']} (majority pred {v['majority_pred_share']:.3f} "
                f"class {v['majority_pred_class']})\n\n")
        f.write("## Discovered transitions\n\n| channel | gold→pred | all | tr | val | te | status | boot | ref | own_val_err | Stage-1 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for c in chans:
            f.write(f"| {c['channel_id_short']} | {c['gold']}→{c['pred']} | {c['count_all']} | {c['count_train']} | "
                    f"{c['count_val']} | {c['count_test']} | {c['status_default']} | {c['bootstrap_stable_freq']} | "
                    f"{c['ref_pool_train']} | {c['own_val_errors']} | "
                    f"{'ELIGIBLE' if c['stage1_eligible'] else c['eligibility_reason']} |\n")
        f.write(f"\n**Stage-1-eligible (proceed to gate):** {out['stage1_eligible_channels']}\n\n")
        f.write("## Cross-architecture topology (stable channel counts)\n\n| channel | Phi | Qwen | Mistral | Llama |\n|---|---|---|---|---|\n")
        allk = sorted(set().union(*[set(d) for d in cmp.values()], set(mis)))
        for k in allk:
            f.write(f"| {k} | {cmp['phi35'].get(k,'')} | {cmp['qwen25_7b'].get(k,'')} | "
                    f"{cmp['mistral7b_v03'].get(k,'')} | {mis.get(k,'')} |\n")
        f.write("\n*(Cross-model counts are context only; they do not alter the Llama rules.)*\n")


if __name__ == "__main__":
    main()
