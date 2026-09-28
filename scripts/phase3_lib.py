"""
phase3_lib.py — shared helpers for Phase 3 (new-channel W2C × Qwen2.5-7B intervention).
========================================================================================
Reuses the archived modules (`qwen7b_native_sakiko` = P, `qwen7b_method_diagnostics` for
diffmean/pca_dir) and replaces only what the LFS-less clone cannot provide:

  - loader: raw local w2c_test_mcq.jsonl -> Dataset.from_list(union-normalized) ->
    .shuffle(seed=42)  (byte-identical convention to the archived GPU-confirmation shim);
  - baseline details: regenerated per-sample file in the gitignored cache dir
    (data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl).

Locked in final/results/w2c_qwen_new_channel_intervention/PHASE3_PROTOCOL.md before any run.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import qwen7b_native_sakiko as P  # noqa: E402  (archived shared module)

ROOT = P.ROOT
RAW_MCQ = ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"
CACHE_DIR = P.CACHE_DIR                      # data/processed/qwen25_7b_w2c/cache (gitignored)
BASE_DETAILS_REGEN = CACHE_DIR / "baseline_details_regen.jsonl"
OUT = ROOT / "final" / "results" / "w2c_qwen_new_channel_intervention"

OBS_LAYERS = [12, 16, 20, 24]                # Phase-3 locked observation grid
WRONG_LAYER = 4                              # locked wrong-layer control injection
SEED = 42
ALPHAS = [0.5, 1.0, 2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
INJ_OFFSETS = [0, -2, -4]
DEGEN_NORM = 5.0
COS_THR = 0.6
N_RANDOM = 20

LABELS = ["tool_call", "direct", "request_for_info", "cannot_answer"]

# the two Phase-3 candidate channels (aliases -> archived etype ids)
NEW_CHANNELS = {
    "ca_rfi": {"gold": "cannot_answer", "from_pred": "request_for_info",
               "ref_gold": "cannot_answer", "etype": "cannot_answer__request_for_info"},
    "tc_rfi": {"gold": "tool_call", "from_pred": "request_for_info",
               "ref_gold": "tool_call", "etype": "tool_call__request_for_info"},
}
# archived channels (frozen reference cascade R = taskA updated config)
REF_LOCKED = {
    "rfi_tc":    {"obs": 20, "inj": 18, "alpha": 2.0, "thr": 0.70, "method": "diffmean"},
    "ca_tc":     {"obs": 20, "inj": 16, "alpha": 4.0, "thr": 0.80, "method": "diffmean"},
    "ca_direct": {"obs": 20, "inj": 16, "alpha": 6.0, "thr": 0.40, "method": "diffmean"},
}
REF_PRIORITY = ["rfi_tc", "ca_tc", "ca_direct"]
ALL_CH_DEFS = {**{ch: {"gold": c["gold"], "from_pred": c["from_pred"], "etype": ch}
                  for ch, c in P.CHANNELS.items()},
               **{ch: {"gold": c["gold"], "from_pred": c["from_pred"], "etype": c["etype"]}
                  for ch, c in NEW_CHANNELS.items()}}


def load_ds_raw():
    """Raw jsonl -> union-normalized rows -> Dataset -> shuffle(seed=42). Archived shim recipe."""
    from datasets import Dataset
    rows = [json.loads(l) for l in open(RAW_MCQ)]
    keys = sorted({k for r in rows for k in r.keys()})
    norm = [{k: r.get(k, None) for k in keys} for r in rows]
    return Dataset.from_list(norm).shuffle(seed=SEED)


def load_meta_and_dataset():
    """Archived meta derivation, driven by the REGENERATED baseline details."""
    ds = load_ds_raw()
    det = [json.loads(l) for l in open(BASE_DETAILS_REGEN)]
    assert len(ds) == len(det), (len(ds), len(det))
    meta = []
    for i, (s, d) in enumerate(zip(ds, det)):
        assert s["uuid"] == d["uuid"], i
        gold, pred = d["gold"], d["pred"]
        correct = (pred == gold)
        if correct:
            etype = "correct"
        elif gold == "request_for_info" and pred == "tool_call":
            etype = "rfi_tc"
        elif gold == "cannot_answer" and pred == "tool_call":
            etype = "ca_tc"
        elif gold == "cannot_answer" and pred == "direct":
            etype = "ca_direct"
        else:
            etype = f"{gold}__{pred}"
        meta.append({"idx": i, "uuid": s["uuid"], "gold": gold, "pred": pred,
                     "correct": correct, "etype": etype})
    return ds, meta


# install on the archived module so any archived helper resolves to the regen loader
P.load_meta_and_dataset = load_meta_and_dataset


def load_splits():
    d = ROOT / "final" / "results" / "splits"
    return (list(map(int, json.load(open(d / "train_idx.json")))),
            list(map(int, json.load(open(d / "val_idx.json")))),
            list(map(int, json.load(open(d / "test_idx.json")))))


def fit_router(A, tr, va, meta, etype, from_pred, negatives="matched_pred"):
    """LR router. positives = channel train errors; negatives per protocol §7."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    pos = [i for i in tr if meta[i]["etype"] == etype]
    if negatives == "matched_pred":
        neg = [i for i in tr if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    else:  # archived all-correct convention (old channels / comparability AUC)
        neg = [i for i in tr if meta[i]["correct"]]
    sc = StandardScaler().fit(A[neg + pos])
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42).fit(
        sc.transform(A[neg + pos]), np.array([0] * len(neg) + [1] * len(pos)))
    if negatives == "matched_pred":
        vpos = [i for i in va if meta[i]["etype"] == etype]
        vneg = [i for i in va if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    else:
        vpos = [i for i in va if meta[i]["etype"] == etype]
        vneg = [i for i in va if meta[i]["correct"]]
    vauc = None
    if vpos and vneg:
        yv = np.array([0] * len(vneg) + [1] * len(vpos))
        vauc = float(roc_auc_score(yv, clf.predict_proba(sc.transform(A[vneg + vpos]))[:, 1]))
    return {"scaler": sc, "clf": clf, "val_auc": vauc, "n_pos": len(pos), "n_neg": len(neg)}


def rscore(router, A, idxs):
    return dict(zip(idxs, router["clf"].predict_proba(
        router["scaler"].transform(A[idxs]))[:, 1]))


def direction(A, tr, meta, etype, ref_gold, method="diffmean"):
    import qwen7b_method_diagnostics as D
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    unit, nrm = (D.diffmean(A, err, ref) if method == "diffmean"
                 else D.pca_dir(A, err, ref, k=1))
    mn = float(np.median(np.linalg.norm(A[tr], axis=1)))
    return unit.astype(np.float32), float(nrm), mn, len(err), len(ref)


def metrics5(idxs, meta, base_pred, int_pred):
    """Fixed/Broke/Net + residuals over ALL five channels + destination redistribution."""
    fixed = broke = 0
    for i in idxs:
        g = meta[i]["gold"]
        if base_pred[i] != g and int_pred[i] == g:
            fixed += 1
        elif base_pred[i] == g and int_pred[i] != g:
            broke += 1
    nt_before = {ch: sum(1 for i in idxs if meta[i]["etype"] == d["etype"])
                 for ch, d in ALL_CH_DEFS.items()}
    nt_after = {ch: sum(1 for i in idxs
                        if meta[i]["gold"] == d["gold"] and int_pred[i] == d["from_pred"])
                for ch, d in ALL_CH_DEFS.items()}
    chan_fixed = {ch: sum(1 for i in idxs if meta[i]["etype"] == d["etype"]
                          and int_pred[i] == meta[i]["gold"])
                  for ch, d in ALL_CH_DEFS.items()}
    acc = sum(1 for i in idxs if int_pred[i] == meta[i]["gold"]) / len(idxs)
    golds = [meta[i]["gold"] for i in idxs]
    preds = [int_pred[i] for i in idxs]
    from sklearn.metrics import f1_score
    active = [c for c in ["tool_call", "request_for_info", "cannot_answer"] if c in set(golds)]
    f1 = float(f1_score(golds, preds, labels=active, average="macro", zero_division=0))
    return {"acc": round(acc, 5), "macro_f1_3cls": round(f1, 5),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "nt_before": nt_before, "nt_after": nt_after, "chan_fixed": chan_fixed}


def destination_table(idxs, meta, base_pred, int_pred, touched):
    """For touched samples: gold x (base_pred -> int_pred) movement table."""
    tset = set(touched)
    moves = Counter()
    for i in idxs:
        if i not in tset:
            continue
        moves[(meta[i]["gold"], base_pred[i], int_pred[i])] += 1
    return [{"gold": g, "from": bp, "to": ip, "n": n}
            for (g, bp, ip), n in sorted(moves.items(), key=lambda x: -x[1])]


def transition_matrix(idxs, meta, int_pred):
    m = Counter((meta[i]["gold"], int_pred[i]) for i in idxs)
    return {g: {p: int(m.get((g, p), 0)) for p in LABELS}
            for g in ["tool_call", "request_for_info", "cannot_answer"]}
