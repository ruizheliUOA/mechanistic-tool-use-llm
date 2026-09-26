"""
phase8_baseline.py — Part 1: R0 readout validity on the untouched target model.
================================================================================
avg_logp W2C readout on ALL 3652 seed-42 rows (baseline predictions are the R0 requirement and
feed discovery support counts). NO steering. Deterministic-replay check on 40 rows.

Usage:
  python scripts/phase8_baseline.py --smoke 5
  python scripts/phase8_baseline.py --replay 40
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

LABELS4 = ["tool_call", "direct", "request_for_info", "cannot_answer"]
COLLAPSE_PRED_SHARE = 0.60


def score_all(model, tok, dev, ds, indices):
    records, skipped = [], []
    for i in indices:
        s = ds[i]
        pt = L.make_prompt_text(tok, s)
        ids = tok.encode(pt, add_special_tokens=False)
        if len(ids) > L.MAXTOK:
            skipped.append({"idx": i, "uuid": s["uuid"], "reason": "prompt>MAXTOK", "n_tok": len(ids)})
            continue
        pids = torch.tensor([ids], device=dev)
        try:
            sc = {lb: (L.score_candidate(model, tok, dev, pids, s["answers"].get(lb, ""))
                       if s["answers"].get(lb, "") else -1e9) for lb in L.LABEL_ORDER}
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped.append({"idx": i, "uuid": s["uuid"], "reason": "OOM"})
                gc.collect(); torch.cuda.empty_cache(); continue
            raise
        del pids
        pred = max(L.LABEL_ORDER, key=lambda lb: sc[lb])
        vals = sorted(sc.values(), reverse=True)
        records.append({"idx": i, "uuid": s["uuid"], "gold": s["correct_answer"], "pred": pred,
                        "correct": bool(pred == s["correct_answer"]),
                        "n_prompt_tok": len(ids), "margin": round(float(vals[0] - vals[1]), 5),
                        "avg_logp": {k: round(v, 5) for k, v in sc.items()}})
        if len(records) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    return records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--replay", type=int, default=0)
    args = ap.parse_args()

    OUTB = L.OUT; OUTB.mkdir(parents=True, exist_ok=True)
    ds = L.load_ds_raw(); n = len(ds)
    tr, va, te = L.splits_all()
    split_of = {}
    for nm, idxs in [("train", tr), ("val", va), ("test", te)]:
        for i in idxs:
            split_of[i] = nm
    model, tok, dev = L.load_model()

    indices = list(range(min(args.smoke, n))) if args.smoke else list(range(n))
    t0 = time.time()
    records, skipped = score_all(model, tok, dev, ds, tqdm(indices, ncols=90, desc="R0"))
    elapsed = time.time() - t0
    L.log.info("scored %d skipped %d in %.1f min", len(records), len(skipped), elapsed / 60)

    replay = None
    if args.replay:
        rec2, _ = score_all(model, tok, dev, ds, list(range(min(args.replay, n))))
        by = {r["idx"]: r for r in records}
        pm = sum(1 for r in rec2 if by[r["idx"]]["pred"] == r["pred"])
        sm = sum(1 for r in rec2 if by[r["idx"]]["avg_logp"] == r["avg_logp"])
        replay = {"n": len(rec2), "pred_identical": pm == len(rec2), "score_identical": sm == len(rec2)}
        L.log.info("REPLAY pred %d/%d score %d/%d", pm, len(rec2), sm, len(rec2))

    if args.smoke:
        L.log.info("SMOKE OK scored=%d skipped=%d finite=%s preds=%s", len(records), len(skipped),
                   all(np.isfinite(list(r["avg_logp"].values())).all() for r in records),
                   Counter(r["pred"] for r in records))
        return

    with open(L.BASE_DETAILS, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    golds = [r["gold"] for r in records]; preds = [r["pred"] for r in records]
    acc = accuracy_score(golds, preds); bal = balanced_accuracy_score(golds, preds)
    active = [c for c in L.GOLD_CLASSES if c in set(golds)]
    mf3 = f1_score(golds, preds, labels=active, average="macro", zero_division=0)
    cm = confusion_matrix(golds, preds, labels=LABELS4)
    cmd = {g: {p: int(cm[i][j]) for j, p in enumerate(LABELS4)} for i, g in enumerate(LABELS4)}
    gd = dict(Counter(golds)); pd = dict(Counter(preds))
    maj_gold = max(gd.values()) / len(records); maj_pred = max(pd.values()) / len(records)
    maj_pred_cls = max(pd, key=pd.get)
    prec, rec, f1c, sup = precision_recall_fscore_support(golds, preds, labels=LABELS4, zero_division=0)
    per_class = {LABELS4[k]: {"precision": round(float(prec[k]), 4), "recall": round(float(rec[k]), 4),
                              "f1": round(float(f1c[k]), 4), "support": int(sup[k])} for k in range(4)}

    def trans(idxs_set=None):
        c = Counter()
        for r in records:
            if idxs_set is not None and r["idx"] not in idxs_set:
                continue
            if r["gold"] != r["pred"]:
                c[(r["gold"], r["pred"])] += 1
        return c
    trans_all = trans()
    tbs = {nm: trans(set(idxs)) for nm, idxs in [("train", tr), ("val", va), ("test", te)]}
    n_err = sum(trans_all.values())
    collapsed = (maj_pred > COLLAPSE_PRED_SHARE) and (acc < maj_gold)
    margins = [r["margin"] for r in records]
    r0 = {"meaningful_above_majority": bool(acc > maj_gold + 0.01), "not_collapsed": bool(not collapsed),
          "finite_scores": bool(all(np.isfinite(list(r["avg_logp"].values())).all() for r in records)),
          "deterministic_replay": (replay or {}).get("pred_identical"),
          "adequate_transition_support": bool(sum(1 for t in trans_all if tbs["train"].get(t, 0) >= 30) >= 1),
          "n_pred_classes": len(pd)}
    r0_pass = (r0["meaningful_above_majority"] and r0["not_collapsed"] and r0["finite_scores"]
               and r0["adequate_transition_support"] and (r0["deterministic_replay"] in (True, None)))

    summary = {"model_repo": L.MODELS[L.MODEL_KEY]["repo"], "model_revision": L.MODELS[L.MODEL_KEY]["revision"],
               "dtype": "bfloat16", "seed": L.SEED, "n_total": n, "n_eval": len(records),
               "n_skipped": len(skipped), "skipped_detail": skipped,
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "accuracy": round(acc, 5), "balanced_accuracy": round(bal, 5),
               "macro_f1_3cls": round(mf3, 5), "majority_gold_share": round(maj_gold, 5),
               "majority_pred_share": round(maj_pred, 5), "majority_pred_class": maj_pred_cls,
               "acc_minus_majority_gold": round(acc - maj_gold, 5),
               "gold_distribution": gd, "pred_distribution": pd, "confusion_matrix_gold_by_pred": cmd,
               "per_class": per_class, "total_errors": n_err,
               "score_margin": {"mean": round(float(np.mean(margins)), 5), "median": round(float(np.median(margins)), 5),
                                "p05": round(float(np.percentile(margins, 5)), 5), "min": round(float(np.min(margins)), 5)},
               "all_transitions_overall": {f"{g}->{p}": c for (g, p), c in sorted(trans_all.items(), key=lambda x: -x[1])},
               "transitions_by_split": {nm: {f"{g}->{p}": c for (g, p), c in sorted(cc.items(), key=lambda x: -x[1])}
                                        for nm, cc in tbs.items()},
               "replay": replay, "R0_checks": r0, "R0_PASS": bool(r0_pass), "elapsed_min": round(elapsed / 60, 1)}
    json.dump(summary, open(OUTB / "llama_r0_summary.json", "w"), indent=2)
    with open(OUTB / "llama_confusion_matrix.csv", "w") as f:
        f.write("gold\\pred," + ",".join(LABELS4) + "\n")
        for g in LABELS4:
            f.write(g + "," + ",".join(str(cmd[g][p]) for p in LABELS4) + "\n")
    with open(OUTB / "llama_error_transition_matrix.csv", "w") as f:
        f.write("gold,pred,count_all,count_train,count_val,count_test\n")
        for (g, p), c in sorted(trans_all.items(), key=lambda x: -x[1]):
            f.write(f"{g},{p},{c},{tbs['train'].get((g,p),0)},{tbs['val'].get((g,p),0)},{tbs['test'].get((g,p),0)}\n")
    L.log.info("R0 %s acc=%.4f maj=%.4f errs=%d", "PASS" if r0_pass else "FAIL", acc, maj_gold, n_err)


if __name__ == "__main__":
    main()
