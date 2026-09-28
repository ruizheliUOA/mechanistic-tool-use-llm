"""
phase5_mistral_baseline.py — R0: Mistral-7B-Instruct-v0.3 deterministic W2C MCQ baseline.
=========================================================================================
avg_logp response-mode scoring (identical protocol to Phi/Qwen); NO steering, NO router.
Writes per-example details to the OUT-OF-REPO cache and R0 summaries to results/baseline/.

Usage:
  python scripts/phase5_mistral_baseline.py --smoke 5          # preflight smoke test
  python scripts/phase5_mistral_baseline.py                    # full R0 baseline
  python scripts/phase5_mistral_baseline.py --replay 40        # + deterministic replay check
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report, precision_recall_fscore_support)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L

LABELS4 = ["tool_call", "direct", "request_for_info", "cannot_answer"]  # report order
COLLAPSE_PRED_SHARE = 0.60  # archived rule: collapse iff pred_share>0.60 AND acc<majority_gold


def score_all(model, tok, dev, ds, indices):
    records, skipped = [], []
    for i in indices:
        s = ds[i]
        pt = L.make_prompt_text(tok, s)
        pids_list = tok.encode(pt, add_special_tokens=False)
        if len(pids_list) > L.MAXTOK:
            skipped.append({"idx": i, "uuid": s["uuid"], "reason": "prompt>MAXTOK",
                            "n_tok": len(pids_list)})
            continue
        pids = torch.tensor([pids_list], device=dev)
        try:
            sc = {}
            for lb in L.LABEL_ORDER:
                cand = s["answers"].get(lb, "")
                sc[lb] = (L.score_candidate(model, tok, dev, pids, cand) if cand else -1e9)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped.append({"idx": i, "uuid": s["uuid"], "reason": "OOM",
                                "n_tok": len(pids_list)})
                gc.collect(); torch.cuda.empty_cache(); continue
            raise
        del pids
        pred = max(L.LABEL_ORDER, key=lambda lb: sc[lb])
        vals = sorted(sc.values(), reverse=True)
        margin = float(vals[0] - vals[1])
        records.append({"idx": i, "uuid": s["uuid"], "gold": s["correct_answer"],
                        "pred": pred, "correct": bool(pred == s["correct_answer"]),
                        "tools_empty": (len(s["tools"] or []) == 0),
                        "n_prompt_tok": len(pids_list), "margin": round(margin, 5),
                        "avg_logp": {k: round(v, 5) for k, v in sc.items()}})
        if len(records) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    return records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run only first N samples (preflight)")
    ap.add_argument("--replay", type=int, default=0, help="re-score first K to check determinism")
    args = ap.parse_args()

    OUTB = L.OUT / "baseline"; OUTB.mkdir(parents=True, exist_ok=True)
    ds = L.load_ds_raw()
    n = len(ds)
    tr, va, te = L.load_splits()
    split_of = {}
    for name, idxs in [("train", tr), ("val", va), ("test", te)]:
        for i in idxs:
            split_of[i] = name

    model, tok, dev = L.load_model(assert_bf16=True)

    indices = list(range(min(args.smoke, n))) if args.smoke else list(range(n))
    t0 = time.time()
    records, skipped = score_all(model, tok, dev, ds, tqdm(indices, ncols=90, desc="R0"))
    elapsed = time.time() - t0
    L.log.info("scored %d, skipped %d in %.1f min", len(records), len(skipped), elapsed / 60)

    # deterministic replay check
    replay_report = None
    if args.replay:
        rec2, _ = score_all(model, tok, dev, ds, list(range(min(args.replay, n))))
        base_by_idx = {r["idx"]: r for r in records}
        n_pred_match = sum(1 for r in rec2 if base_by_idx[r["idx"]]["pred"] == r["pred"])
        n_score_match = sum(1 for r in rec2
                            if base_by_idx[r["idx"]]["avg_logp"] == r["avg_logp"])
        replay_report = {"n": len(rec2), "pred_agreement": n_pred_match,
                         "score_bitmatch": n_score_match,
                         "pred_identical": n_pred_match == len(rec2),
                         "score_identical": n_score_match == len(rec2)}
        L.log.info("REPLAY: pred %d/%d identical, score %d/%d identical",
                   n_pred_match, len(rec2), n_score_match, len(rec2))

    if args.smoke:
        L.log.info("SMOKE OK: %d scored, %d skipped; finite=%s; preds=%s",
                   len(records), len(skipped),
                   all(np.isfinite(list(r["avg_logp"].values())).all() for r in records),
                   Counter(r["pred"] for r in records))
        json.dump({"n": len(records), "skipped": len(skipped),
                   "records": records, "replay": replay_report},
                  open(OUTB / "mistral7b_w2c_smoke.json", "w"), indent=2)
        return

    # ── write per-example details (OUT OF REPO) + copy into results/baseline ──
    with open(L.BASE_DETAILS, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUTB / "mistral7b_w2c_baseline_details.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    golds = [r["gold"] for r in records]; preds = [r["pred"] for r in records]
    acc = accuracy_score(golds, preds)
    bal_acc = balanced_accuracy_score(golds, preds)
    active = [c for c in L.GOLD_CLASSES if c in set(golds)]
    macro_f1_3 = f1_score(golds, preds, labels=active, average="macro", zero_division=0)
    macro_f1_4 = f1_score(golds, preds, labels=LABELS4, average="macro", zero_division=0)
    cm = confusion_matrix(golds, preds, labels=LABELS4)
    cm_dict = {g: {p: int(cm[i][j]) for j, p in enumerate(LABELS4)}
               for i, g in enumerate(LABELS4)}
    gold_dist = dict(Counter(golds)); pred_dist = dict(Counter(preds))
    majority_gold = max(gold_dist.values()) / len(records)
    majority_pred_share = max(pred_dist.values()) / len(records)
    majority_pred_class = max(pred_dist, key=pred_dist.get)
    prec, rec, f1c, sup = precision_recall_fscore_support(
        golds, preds, labels=LABELS4, zero_division=0)
    per_class = {LABELS4[k]: {"precision": round(float(prec[k]), 4),
                              "recall": round(float(rec[k]), 4),
                              "f1": round(float(f1c[k]), 4), "support": int(sup[k])}
                 for k in range(len(LABELS4))}

    # transitions overall + per split
    def transitions(idxs_set=None):
        c = Counter()
        for r in records:
            if idxs_set is not None and r["idx"] not in idxs_set:
                continue
            if r["gold"] != r["pred"]:
                c[(r["gold"], r["pred"])] += 1
        return c
    trans_all = transitions()
    trans_by_split = {nm: transitions(set(idxs))
                      for nm, idxs in [("train", tr), ("val", va), ("test", te)]}
    n_err = sum(trans_all.values())

    # R0 validity gate
    collapsed = (majority_pred_share > COLLAPSE_PRED_SHARE) and (acc < majority_gold)
    margins = [r["margin"] for r in records]
    r0 = {
        "meaningful_above_majority": bool(acc > majority_gold + 0.01),
        "not_collapsed": bool(not collapsed),
        "finite_scores": bool(all(np.isfinite(list(r["avg_logp"].values())).all()
                                  for r in records)),
        "deterministic_replay": (replay_report or {}).get("pred_identical"),
        "adequate_transition_support": bool(
            sum(1 for t, c in trans_all.items()
                if trans_by_split["train"].get(t, 0) >= 30) >= 1),
        "n_pred_classes": len(pred_dist),
    }
    r0_pass = (r0["meaningful_above_majority"] and r0["not_collapsed"]
               and r0["finite_scores"] and r0["adequate_transition_support"]
               and (r0["deterministic_replay"] in (True, None)))

    summary = {
        "model_repo": L.MODEL_REPO, "model_revision": L.MODEL_REVISION,
        "model_path": L.MODEL_PATH, "dtype": "bfloat16", "seed": L.SEED,
        "n_total": n, "n_eval": len(records), "n_skipped": len(skipped),
        "skipped_detail": skipped,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "accuracy": round(acc, 5), "balanced_accuracy": round(bal_acc, 5),
        "macro_f1_3cls": round(macro_f1_3, 5), "macro_f1_4opt": round(macro_f1_4, 5),
        "majority_gold_share": round(majority_gold, 5),
        "majority_pred_share": round(majority_pred_share, 5),
        "majority_pred_class": majority_pred_class,
        "acc_minus_majority_gold": round(acc - majority_gold, 5),
        "gold_distribution": gold_dist, "pred_distribution": pred_dist,
        "confusion_matrix_gold_by_pred": cm_dict, "per_class": per_class,
        "total_errors": n_err,
        "score_margin": {"mean": round(float(np.mean(margins)), 5),
                         "median": round(float(np.median(margins)), 5),
                         "p05": round(float(np.percentile(margins, 5)), 5),
                         "p95": round(float(np.percentile(margins, 95)), 5),
                         "min": round(float(np.min(margins)), 5)},
        "all_transitions_overall": {f"{g}->{p}": c for (g, p), c in
                                    sorted(trans_all.items(), key=lambda x: -x[1])},
        "transitions_by_split": {nm: {f"{g}->{p}": c for (g, p), c in
                                      sorted(cc.items(), key=lambda x: -x[1])}
                                 for nm, cc in trans_by_split.items()},
        "replay": replay_report,
        "R0_checks": r0, "R0_PASS": bool(r0_pass),
        "elapsed_min": round(elapsed / 60, 1),
    }
    json.dump(summary, open(OUTB / "mistral7b_w2c_baseline_summary.json", "w"), indent=2)

    # confusion CSV
    with open(OUTB / "mistral7b_w2c_confusion_matrix.csv", "w") as f:
        f.write("gold\\pred," + ",".join(LABELS4) + "\n")
        for g in LABELS4:
            f.write(g + "," + ",".join(str(cm_dict[g][p]) for p in LABELS4) + "\n")
    # transition counts CSV (overall + per split)
    with open(OUTB / "mistral7b_w2c_transition_counts.csv", "w") as f:
        f.write("gold,pred,count_all,count_train,count_val,count_test\n")
        for (g, p), c in sorted(trans_all.items(), key=lambda x: -x[1]):
            f.write(f"{g},{p},{c},{trans_by_split['train'].get((g,p),0)},"
                    f"{trans_by_split['val'].get((g,p),0)},{trans_by_split['test'].get((g,p),0)}\n")

    # markdown summary
    with open(OUTB / "MISTRAL7B_W2C_BASELINE_SUMMARY.md", "w") as f:
        f.write("# Mistral-7B-Instruct-v0.3 — R0 W2C MCQ Baseline\n\n")
        f.write(f"- Model: `{L.MODEL_REPO}` @ `{L.MODEL_REVISION}` (bf16), 32 layers, hidden 4096\n")
        f.write(f"- Protocol: avg_logp response-mode scoring, seed-42 W2C test/mcq, N={len(records)} "
                f"(skipped {len(skipped)})\n")
        f.write(f"- Split: train {len(tr)} / val {len(va)} / test {len(te)}\n\n")
        f.write("## Headline\n\n| metric | value |\n|---|---|\n")
        f.write(f"| accuracy | {acc:.4f} |\n| balanced accuracy | {bal_acc:.4f} |\n")
        f.write(f"| macro-F1 (3-cls) | {macro_f1_3:.4f} |\n| macro-F1 (4-opt) | {macro_f1_4:.4f} |\n")
        f.write(f"| majority-gold baseline | {majority_gold:.4f} |\n")
        f.write(f"| acc − majority | {acc-majority_gold:+.4f} |\n")
        f.write(f"| total errors | {n_err} |\n")
        f.write(f"| **R0 verdict** | **{'PASS' if r0_pass else 'FAIL'}** |\n\n")
        f.write("## Gold / prediction distribution\n\n")
        f.write(f"- gold: {gold_dist}\n- pred: {pred_dist}  (majority pred class: "
                f"{majority_pred_class}, share {majority_pred_share:.3f})\n\n")
        f.write("## Confusion matrix (rows=gold, cols=pred)\n\n| gold\\pred | "
                + " | ".join(LABELS4) + " |\n|" + "---|" * (len(LABELS4) + 1) + "\n")
        for g in LABELS4:
            f.write(f"| {g} | " + " | ".join(str(cm_dict[g][p]) for p in LABELS4) + " |\n")
        f.write("\n## Per-class precision / recall / F1\n\n| class | P | R | F1 | support |\n|---|---|---|---|---|\n")
        for c in LABELS4:
            pc = per_class[c]
            f.write(f"| {c} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |\n")
        f.write("\n## Top error transitions (candidate channels; count_all / train / val / test)\n\n")
        f.write("| gold → pred | all | train | val | test |\n|---|---|---|---|---|\n")
        for (g, p), c in sorted(trans_all.items(), key=lambda x: -x[1])[:12]:
            f.write(f"| {g} → {p} | {c} | {trans_by_split['train'].get((g,p),0)} | "
                    f"{trans_by_split['val'].get((g,p),0)} | {trans_by_split['test'].get((g,p),0)} |\n")
        f.write(f"\n## R0 gate\n\n```\n{json.dumps(r0, indent=2)}\nR0_PASS = {r0_pass}\n```\n")
        if replay_report:
            f.write(f"\nDeterministic replay: {replay_report}\n")

    print("\n" + classification_report(golds, preds, labels=LABELS4, zero_division=0))
    L.log.info("R0 %s | acc=%.4f macroF1_3=%.4f | majority=%.4f | errors=%d",
               "PASS" if r0_pass else "FAIL", acc, macro_f1_3, majority_gold, n_err)


if __name__ == "__main__":
    main()
