"""
qwen25_7b_tool_call_penalty_baseline.py
=======================================
Score-level control baseline: does a GLOBAL tool_call penalty explain Qwen2.5-7B
SAKIKO's W2C gains?  For every sample:

    score'(tool_call) = score(tool_call) - lambda   (other candidates unchanged)
    pred' = argmax over adjusted avg_logp

This is NOT activation intervention — no model is loaded. Reuses the archived
Qwen2.5-7B baseline details (per-candidate avg_logp). lambda is swept on the seed=42
validation split only; the best lambda is applied once to test.

Outputs under final/results/7b_w2c_sakiko/tool_call_penalty_baseline/.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "final" / "results" / "7b_w2c_baseline" / "qwen25_7b_baseline_details.jsonl"
SPLITS = ROOT / "sakiko_v3" / "results" / "splits"
OUT = ROOT / "final" / "results" / "7b_w2c_sakiko" / "tool_call_penalty_baseline"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["tool_call", "direct", "request_for_info", "cannot_answer"]
LAMBDAS = [round(x, 2) for x in np.arange(0.0, 5.0001, 0.05)]


def load():
    det = [json.loads(l) for l in open(DET)]
    gold = {i: d["gold"] for i, d in enumerate(det)}
    base = {i: d["pred"] for i, d in enumerate(det)}
    scores = {i: d["avg_logp"] for i, d in enumerate(det)}
    tr = list(map(int, json.load(open(SPLITS / "train_idx.json"))))
    va = list(map(int, json.load(open(SPLITS / "val_idx.json"))))
    te = list(map(int, json.load(open(SPLITS / "test_idx.json"))))
    return det, gold, base, scores, tr, va, te


def predict(scores_i: dict, lam: float) -> str:
    adj = dict(scores_i)
    adj["tool_call"] = scores_i["tool_call"] - lam
    return max(LABELS, key=lambda k: adj.get(k, -1e9))


def channel(gold, pred):
    if gold == pred: return "correct"
    if gold == "request_for_info" and pred == "tool_call": return "rfi_tc"
    if gold == "cannot_answer" and pred == "tool_call": return "ca_tc"
    if gold == "cannot_answer" and pred == "direct": return "ca_direct"
    return "other"


def metrics(idxs, gold, base, scores, lam):
    new = {i: predict(scores[i], lam) for i in idxs}
    n = len(idxs)
    acc = sum(new[i] == gold[i] for i in idxs) / n
    fixed = sum(1 for i in idxs if base[i] != gold[i] and new[i] == gold[i])
    broke = sum(1 for i in idxs if base[i] == gold[i] and new[i] != gold[i])
    def ch_before(ch):
        if ch == "rfi_tc": return sum(1 for i in idxs if gold[i] == "request_for_info" and base[i] == "tool_call")
        if ch == "ca_tc": return sum(1 for i in idxs if gold[i] == "cannot_answer" and base[i] == "tool_call")
        if ch == "ca_direct": return sum(1 for i in idxs if gold[i] == "cannot_answer" and base[i] == "direct")
    def ch_after(ch):
        if ch == "rfi_tc": return sum(1 for i in idxs if gold[i] == "request_for_info" and new[i] == "tool_call")
        if ch == "ca_tc": return sum(1 for i in idxs if gold[i] == "cannot_answer" and new[i] == "tool_call")
        if ch == "ca_direct": return sum(1 for i in idxs if gold[i] == "cannot_answer" and new[i] == "direct")
    nt_b = {ch: ch_before(ch) for ch in ["rfi_tc", "ca_tc", "ca_direct"]}
    nt_a = {ch: ch_after(ch) for ch in ["rfi_tc", "ca_tc", "ca_direct"]}
    # gold tool_call recall before/after
    gtc = [i for i in idxs if gold[i] == "tool_call"]
    rec_b = sum(1 for i in gtc if base[i] == "tool_call") / len(gtc) if gtc else 0.0
    rec_a = sum(1 for i in gtc if new[i] == "tool_call") / len(gtc) if gtc else 0.0
    # originally-correct tool_call broken
    broke_gtc = sum(1 for i in gtc if base[i] == "tool_call" and new[i] != "tool_call")
    return {"lambda": lam, "n": n, "accuracy": round(acc, 5),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "nt_before": nt_b, "nt_after": nt_a,
            "gold_tc_recall_before": round(rec_b, 4), "gold_tc_recall_after": round(rec_a, 4),
            "broken_correct_tool_call": broke_gtc,
            "n_gold_tool_call": len(gtc)}, new


def confmat(idxs, gold, preds):
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    for i in idxs:
        cm[gold[i]][preds[i]] += 1
    return cm


def main():
    det, gold, base, scores, tr, va, te = load()
    print(f"Loaded {len(det)} details; splits {len(tr)}/{len(va)}/{len(te)}")

    # ── Step 2-3: val sweep ──
    val_rows = []
    for lam in LAMBDAS:
        m, _ = metrics(va, gold, base, scores, lam)
        val_rows.append(m)
    json.dump(val_rows, open(OUT / "penalty_val_sweep.json", "w"), indent=2, default=float)

    # selection: best Net, tie-break low broke, then accuracy; guard tool_call damage
    # require gold_tc_recall_after >= 0.80 * recall_before (no excessive true-tool damage)
    base_rec = val_rows[0]["gold_tc_recall_before"]
    eligible = [r for r in val_rows if r["gold_tc_recall_after"] >= 0.80 * base_rec]
    pool = eligible if eligible else val_rows
    best = max(pool, key=lambda r: (r["net"], -r["broke"], r["accuracy"]))
    best_lambda = best["lambda"]
    # also the unconstrained best-Net lambda (for reporting)
    best_net_uncon = max(val_rows, key=lambda r: (r["net"], -r["broke"]))
    print(f"val baseline acc={val_rows[0]['accuracy']:.4f} | selected lambda={best_lambda} "
          f"(net {best['net']:+d}, broke {best['broke']}, acc {best['accuracy']:.4f}, "
          f"gold_tc_recall {best['gold_tc_recall_before']:.3f}->{best['gold_tc_recall_after']:.3f})")
    print(f"unconstrained best-net lambda={best_net_uncon['lambda']} net={best_net_uncon['net']:+d} "
          f"but gold_tc_recall->{best_net_uncon['gold_tc_recall_after']:.3f}")

    # ── Step 4: locked test ──
    tm, test_new = metrics(te, gold, base, scores, best_lambda)
    tm0, test_base = metrics(te, gold, base, scores, 0.0)
    cm_before = confmat(te, gold, test_base)
    cm_after = confmat(te, gold, test_new)
    with open(OUT / "penalty_locked_test_details.jsonl", "w") as f:
        for i in te:
            f.write(json.dumps({"idx": i, "uuid": det[i]["uuid"], "gold": gold[i],
                "base_pred": base[i], "penalty_pred": test_new[i],
                "etype_before": channel(gold[i], base[i]),
                "etype_after": channel(gold[i], test_new[i])}, ensure_ascii=False) + "\n")
    test_summary = {"selected_lambda": best_lambda,
                    "base_acc": tm0["accuracy"], "penalty_acc": tm["accuracy"],
                    "delta_acc": round(tm["accuracy"] - tm0["accuracy"], 5),
                    "fixed": tm["fixed"], "broke": tm["broke"], "net": tm["net"],
                    "nt_before": tm["nt_before"], "nt_after": tm["nt_after"],
                    "gold_tc_recall_before": tm["gold_tc_recall_before"],
                    "gold_tc_recall_after": tm["gold_tc_recall_after"],
                    "broken_correct_tool_call": tm["broken_correct_tool_call"],
                    "n_gold_tool_call": tm["n_gold_tool_call"],
                    "confusion_before": cm_before, "confusion_after": cm_after}
    json.dump(test_summary, open(OUT / "penalty_locked_test_summary.json", "w"), indent=2, default=float)
    print(f"LOCKED TEST lambda={best_lambda}: acc {tm0['accuracy']:.4f}->{tm['accuracy']:.4f} "
          f"net={tm['net']:+d} (fix{tm['fixed']}/broke{tm['broke']}) | "
          f"gold_tc_recall {tm['gold_tc_recall_before']:.3f}->{tm['gold_tc_recall_after']:.3f} "
          f"broke_true_tc={tm['broken_correct_tool_call']}")
    print("  channels:", {ch: (tm["nt_before"][ch], tm["nt_after"][ch]) for ch in tm["nt_before"]})

    _write_reports(val_rows, best, best_net_uncon, test_summary)


def _write_reports(val_rows, best, best_net_uncon, ts):
    with open(OUT / "penalty_val_sweep_summary.md", "w") as f:
        f.write("# Tool_call Penalty — Validation Sweep (Qwen2.5-7B, seed=42)\n\n")
        f.write(f"λ from 0.0 to 5.0 step 0.05 (avg_logp scale ~unit; tool_call−best_other margin "
                f"mean 0.14, std 1.23). Val baseline acc = {val_rows[0]['accuracy']:.4f}.\n\n")
        f.write("| λ | acc | Fixed | Broke | Net | rfi_tc→ | ca_tc→ | ca_direct→ | gold_tc recall→ | broke_true_tc |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        show = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, best["lambda"]]
        seen = set()
        for lam in sorted(set(show)):
            r = next((x for x in val_rows if abs(x["lambda"] - lam) < 1e-9), None)
            if not r or lam in seen: continue
            seen.add(lam)
            star = " ⭐" if abs(lam - best["lambda"]) < 1e-9 else ""
            f.write(f"| {lam}{star} | {r['accuracy']:.4f} | {r['fixed']} | {r['broke']} | {r['net']:+d} "
                    f"| {r['nt_before']['rfi_tc']}→{r['nt_after']['rfi_tc']} "
                    f"| {r['nt_before']['ca_tc']}→{r['nt_after']['ca_tc']} "
                    f"| {r['nt_before']['ca_direct']}→{r['nt_after']['ca_direct']} "
                    f"| {r['gold_tc_recall_before']:.3f}→{r['gold_tc_recall_after']:.3f} "
                    f"| {r['broken_correct_tool_call']} |\n")
        f.write(f"\n**Selected λ = {best['lambda']}** (best Net with gold_tool_call recall ≥ 80% of "
                f"baseline). Unconstrained best-Net λ = {best_net_uncon['lambda']} "
                f"(Net {best_net_uncon['net']:+d}) but drops gold_tc recall to "
                f"{best_net_uncon['gold_tc_recall_after']:.3f}.\n")

    with open(OUT / "penalty_locked_test_summary.md", "w") as f:
        f.write("# Tool_call Penalty — Locked Test (Qwen2.5-7B, seed=42)\n\n")
        f.write(f"Selected λ = **{ts['selected_lambda']}** (val-selected).\n\n")
        f.write("| metric | value |\n|---|---|\n")
        f.write(f"| baseline test acc | {ts['base_acc']:.4f} |\n")
        f.write(f"| penalty test acc | {ts['penalty_acc']:.4f} |\n")
        f.write(f"| Δ acc | {ts['delta_acc']:+.4f} |\n")
        f.write(f"| Fixed | {ts['fixed']} |\n| Broke | {ts['broke']} |\n| **Net** | **{ts['net']:+d}** |\n")
        f.write(f"| gold tool_call recall before→after | {ts['gold_tc_recall_before']:.3f}→{ts['gold_tc_recall_after']:.3f} |\n")
        f.write(f"| originally-correct tool_call broken | {ts['broken_correct_tool_call']} / {ts['n_gold_tool_call']} gold tool_call |\n\n")
        f.write("## Per-channel (before→after)\n\n| channel | before | after |\n|---|---|---|\n")
        for ch in ts["nt_before"]:
            f.write(f"| {ch} | {ts['nt_before'][ch]} | {ts['nt_after'][ch]} |\n")
        f.write("\n## Confusion matrix (rows=gold, cols=pred)\n\n")
        for label, cm in [("Before (λ=0)", ts["confusion_before"]), (f"After (λ={ts['selected_lambda']})", ts["confusion_after"])]:
            f.write(f"**{label}**\n\n| gold↓ | tool_call | direct | request_for_info | cannot_answer |\n|---|---|---|---|---|\n")
            for g in LABELS:
                f.write(f"| {g} | " + " | ".join(str(cm[g][p]) for p in LABELS) + " |\n")
            f.write("\n")


if __name__ == "__main__":
    main()
