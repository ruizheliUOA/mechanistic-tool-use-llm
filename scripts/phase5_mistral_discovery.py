"""
phase5_mistral_discovery.py — automatic W2C channel discovery for Mistral (CPU, no model).
==========================================================================================
Reuses the archived, tested discovery engine (`discover_sakiko_channels.transition_discovery`,
threshold settings, 200-bootstrap stability, R0 validity gate) on Mistral's own baseline
labels, then applies the Phase-5 scope filter and compares the discovered structure to
Phi-3.5 / Qwen2.5-7B / MetaTool-Qwen.

Does NOT hard-code any channel count or the Qwen/Phi channel set.

Usage:  python scripts/phase5_mistral_discovery.py
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L
from discover_sakiko_channels import transition_discovery  # tested engine (reuse)

OUTD = L.OUT / "channel_discovery"; OUTD.mkdir(parents=True, exist_ok=True)

SHORT = {("request_for_info", "tool_call"): "rfi_tc",
         ("cannot_answer", "tool_call"): "ca_tc",
         ("cannot_answer", "direct"): "ca_direct",
         ("cannot_answer", "request_for_info"): "ca_rfi",
         ("tool_call", "request_for_info"): "tc_rfi",
         ("tool_call", "direct"): "tc_direct",
         ("request_for_info", "direct"): "rfi_direct",
         ("request_for_info", "cannot_answer"): "rfi_ca",
         ("tool_call", "cannot_answer"): "tc_ca",
         ("direct", "tool_call"): "d_tc"}


def sid(g, p):
    return SHORT.get((g, p), f"{g}__{p}")


# scope filter thresholds (Phase-5 §12)
SCOPE_TRAIN_MIN = 30
SCOPE_EVAL_MIN = 8
GOLD_CLASSES = set(L.GOLD_CLASSES)


def main():
    ds, meta = L.load_meta()
    tr, va, te = L.load_splits()
    split_of = {}
    for nm, idxs in [("train", tr), ("val", va), ("test", te)]:
        for i in idxs:
            split_of[i] = nm
    rows = [{"gold": m["gold"], "pred": m["pred"], "split": split_of[m["idx"]]} for m in meta]

    res, _sens = transition_discovery(rows, dataset="w2c_mistral7b", model="Mistral-7B-Instruct-v0.3",
                                      archived=["rfi_tc", "ca_tc", "ca_direct"], note="Phase-5 native")

    # attach short ids + scope-filter decision
    chans = []
    for c in res["discovered_channels"]:
        g, p = c["gold"], c["pred"]
        cid = sid(g, p)
        scope_pass = (g in GOLD_CLASSES and p != g
                      and (c["count_train"] or 0) >= SCOPE_TRAIN_MIN
                      and (c["count_val"] or 0) >= SCOPE_EVAL_MIN
                      and (c["count_test"] or 0) >= SCOPE_EVAL_MIN
                      and c["status_default"] in ("stable", "weak"))
        reason = []
        if g not in GOLD_CLASSES:
            reason.append("gold_not_intervenable")
        if (c["count_train"] or 0) < SCOPE_TRAIN_MIN:
            reason.append(f"train<{SCOPE_TRAIN_MIN}")
        if (c["count_val"] or 0) < SCOPE_EVAL_MIN or (c["count_test"] or 0) < SCOPE_EVAL_MIN:
            reason.append(f"eval<{SCOPE_EVAL_MIN}")
        if c["status_default"] not in ("stable", "weak"):
            reason.append(f"status={c['status_default']}")
        chans.append({**c, "channel_id_short": cid, "scope_pass": bool(scope_pass),
                      "scope_reject_reason": ("" if scope_pass else "; ".join(reason))})

    scope_pass_ids = [c["channel_id_short"] for c in chans if c["scope_pass"]]

    out = {
        "model": "Mistral-7B-Instruct-v0.3", "revision": L.MODEL_REVISION,
        "dataset": "nvidia/When2Call test/mcq", "n_samples": res["n_samples"],
        "baseline_accuracy": res["baseline_accuracy"], "total_errors": res["total_errors"],
        "baseline_validity_R0": res["baseline_validity"],
        "confusion_matrix_gold_by_pred": res["confusion_matrix_gold_by_pred"],
        "threshold_settings": res["threshold_settings"], "n_bootstrap": res["n_bootstrap"],
        "discovered_channels": chans,
        "stable_channels_default": [sid(*_k(c)) for c in res["discovered_channels"]
                                    if c["status_default"] == "stable"],
        "n_stable_by_threshold_setting": res["n_stable_by_threshold_setting"],
        "scope_filter": {"train_min": SCOPE_TRAIN_MIN, "eval_min": SCOPE_EVAL_MIN,
                         "gold_classes": sorted(GOLD_CLASSES),
                         "scope_pass_channels": scope_pass_ids},
        "vs_archived_manual_set": {"manual": ["rfi_tc", "ca_tc", "ca_direct"],
                                   "auto_stable_extra": [c["channel_id_short"] for c in chans
                                                         if c["status_default"] == "stable"
                                                         and c["channel_id_short"] not in
                                                         ("rfi_tc", "ca_tc", "ca_direct")]},
    }
    json.dump(out, open(OUTD / "mistral7b_channel_discovery.json", "w"), indent=2)

    # CSV
    with open(OUTD / "mistral7b_channel_discovery.csv", "w") as f:
        f.write("channel_id,gold,pred,count_all,count_train,count_val,count_test,"
                "pct_of_errors,status_default,status_strict,status_lenient,"
                "bootstrap_stable_freq,in_manual_set,scope_pass,scope_reject_reason\n")
        for c in chans:
            f.write(f"{c['channel_id_short']},{c['gold']},{c['pred']},{c['count_all']},"
                    f"{c['count_train']},{c['count_val']},{c['count_test']},{c['pct_of_errors']},"
                    f"{c['status_default']},{c['status_strict']},{c['status_lenient']},"
                    f"{c['bootstrap_stable_freq']},{c['in_archived_set']},{c['scope_pass']},"
                    f"\"{c['scope_reject_reason']}\"\n")

    _write_md(out, chans)
    _cross_model(chans)
    print("R0 validity:", out["baseline_validity_R0"])
    print("scope-pass channels:", scope_pass_ids)


def _k(c):
    return (c["gold"], c["pred"])


def _write_md(out, chans):
    v = out["baseline_validity_R0"]
    with open(OUTD / "MISTRAL7B_CHANNEL_DISCOVERY.md", "w") as f:
        f.write("# Mistral-7B-Instruct-v0.3 — Automatic W2C Channel Discovery\n\n")
        f.write(f"- baseline acc {out['baseline_accuracy']:.4f}; total errors "
                f"{out['total_errors']}; N={out['n_samples']}\n")
        f.write(f"- R0 validity: readout_collapse_flag={v['readout_collapse_flag']} "
                f"(majority_pred {v['majority_pred_share']:.3f} class {v['majority_pred_class']}, "
                f"acc−majority_gold {v['acc_minus_majority_gold']:+.4f})\n\n")
        f.write("## Discovered transitions (gold→pred)\n\n")
        f.write("| channel | gold→pred | all | tr | val | te | %err | status | boot | in manual | scope |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for c in chans:
            f.write(f"| {c['channel_id_short']} | {c['gold']}→{c['pred']} | {c['count_all']} | "
                    f"{c['count_train']} | {c['count_val']} | {c['count_test']} | {c['pct_of_errors']} | "
                    f"{c['status_default']} | {c['bootstrap_stable_freq']} | {c['in_archived_set']} | "
                    f"{'PASS' if c['scope_pass'] else '—'} |\n")
        f.write(f"\n**Scope-passing channels (intervention-eligible):** "
                f"{out['scope_filter']['scope_pass_channels']}\n\n")
        f.write(f"**Auto-discovered stable channels beyond the manual "
                f"{{rfi_tc, ca_tc, ca_direct}}:** {out['vs_archived_manual_set']['auto_stable_extra']}\n")


def _cross_model(mistral_chans):
    """Compare discovered stable-channel structure across Phi / Qwen / MetaTool-Qwen / Mistral."""
    qwen = {"rfi_tc": 622, "ca_tc": 513, "ca_direct": 465, "ca_rfi": 219, "tc_rfi": 133,
            "tc_direct": 53, "rfi_direct": 42}
    phi = {"rfi_tc": 746, "ca_tc": 511, "ca_direct": 393, "ca_rfi": 93, "rfi_direct": 47,
           "tc_rfi": 37, "tc_direct": 31}
    mis = {c["channel_id_short"]: c["count_all"] for c in mistral_chans
           if c["status_default"] == "stable"}
    all_ids = sorted(set(qwen) | set(phi) | set(mis),
                     key=lambda k: -(mis.get(k, 0) + qwen.get(k, 0)))
    with open(L.OUT / "channel_discovery" / "cross_model_channel_comparison.csv", "w") as f:
        f.write("channel_id,phi35_count,qwen7b_count,mistral7b_count,"
                "in_phi_stable,in_qwen_stable,in_mistral_stable\n")
        for k in all_ids:
            f.write(f"{k},{phi.get(k,'')},{qwen.get(k,'')},{mis.get(k,'')},"
                    f"{k in phi},{k in qwen},{k in mis}\n")


if __name__ == "__main__":
    main()
