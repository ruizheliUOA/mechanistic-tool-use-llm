"""
metatool_binary_robustness_audit.py
====================================
Robustness audit for the MetaTool-native binary SAKIKO pilot.

Reuses the pilot's low-level inference helpers (build_prompt_text,
score_candidate_hooked) and the convert script's deterministic
stratified_split. Activations are extracted ONCE for all 1040 samples
(in original_id order) and indexed per split — they are seed- and
candidate-independent (they come from the prompt forward pass, which
never contains the candidate string).

Parts:
  1. Multi-seed split robustness   (seeds 42, 123, 456, 789, 2024)
  2. Random placebo distribution   (>=20 random dirs on seed=42 locked cfg)
  3. Candidate-string sensitivity  (Yes/No vs verbose vs literal, seed=42)
  4. Quantization note

Channel: nt_tc (gold=no_tool, baseline_pred=tool_call) — over-calling.

Intervention:  correction = alpha * median_norm * unit(DiffMean direction)
               h' = h + correction   (MLP output at inj_layer, mlp_all)

Decision gate: apply only when router fires AND baseline_pred == "tool_call".

Usage:
  cd /path/to/project
  /path/to/project/.venv/bin/python scripts/metatool_binary_robustness_audit.py \\
      --model_key phi35

IMPORTANT: MetaTool-Binary, NOT W2C-4way. 4-bit NF4 quantization.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Import reusable pieces from existing scripts ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ on path
from convert_metatool_binary import stratified_split
from metatool_binary_sakiko_pilot import (
    build_prompt_text,
    score_candidate_hooked,
    load_model_cfg,
    load_tokenizer,
    load_model,
    ALPHAS, INJ_LAYERS, THRESHOLDS, OBS_LAYER, INJ_MODE,
)

# ── Paths / constants ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "processed" / "metatool_binary"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR   = ROOT / "final" / "results" / "dataset_extension"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

CANDIDATE_SETS = {
    "yes_no":   {"tool_call": "Yes",         "no_tool": "No"},
    "verbose":  {"tool_call": "Use a tool",  "no_tool": "Do not use a tool"},
    "literal":  {"tool_call": "tool_call",   "no_tool": "no_tool"},
}
LABEL_ORDER = ["tool_call", "no_tool"]

N_RANDOM_PLACEBO = 25   # >= 20 required

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mt_audit")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_all_records() -> tuple[list[dict], dict, dict]:
    """Return (records, golds_by_oid, thought_by_oid). oid == original_id == position."""
    records = load_jsonl(DATA_DIR / "metatool_binary_all.jsonl")
    # sanity: original_id == index
    for i, r in enumerate(records):
        assert r["original_id"] == i, "original_id must equal position"
    golds   = {r["original_id"]: r["gold_response_mode"] for r in records}
    thought = {r["original_id"]: r["thought_prompt"]      for r in records}
    return records, golds, thought


def load_yesno_baseline() -> dict:
    """baseline_pred[oid] for the Yes/No candidate set, from the baseline run."""
    details = load_jsonl(OUT_DIR / "metatool_binary_baseline_details.jsonl")
    bp = {}
    for d in details:
        oid = int(d["sample_id"].split("_")[-1])
        bp[oid] = d["pred_response_mode"]
    return bp


# ── Activation extraction (once, all 1040, oid order) ────────────────────────

def extract_acts_all(model, tok, golds, thought, obs_layer, device, force=False):
    cache = CACHE_DIR / f"acts_all_L{obs_layer}.npy"
    if cache.exists() and not force:
        acts = np.load(cache)
        log.info("Loaded cached acts_all: %s", acts.shape)
        return acts

    n = len(golds)
    D = model.config.hidden_size
    acts = np.zeros((n, D), dtype=np.float32)
    mlp = model.model.layers[obs_layer].mlp

    for oid in tqdm(range(n), desc=f"Acts-all-L{obs_layer}"):
        prompt_text = build_prompt_text(thought[oid], tok)
        token_ids   = tok.encode(prompt_text, add_special_tokens=False)
        last_pos    = len(token_ids) - 1
        input_ids   = torch.tensor([token_ids], device=device)

        cap = {}
        def _hook(m, i, o, _c=cap, _p=last_pos):
            _c["a"] = o[0, _p, :].detach().cpu().float().numpy()
        h = mlp.register_forward_hook(_hook)
        with torch.no_grad():
            model(input_ids)
        h.remove()
        acts[oid] = cap["a"]
        del input_ids
        if (oid + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()

    np.save(cache, acts)
    log.info("Saved acts_all: %s → %s", acts.shape, cache)
    return acts


# ── Baseline re-scoring (for alternative candidate sets) ─────────────────────

def rescore_baseline(model, tok, oids, thought, candidates, device) -> dict:
    """Baseline prediction (no correction) for each oid under given candidates."""
    D_dummy = None
    bp = {}
    zero_vec = None
    for oid in tqdm(oids, desc="Rescore-baseline", leave=False):
        prompt_text = build_prompt_text(thought[oid], tok)
        if zero_vec is None:
            zero_vec = np.zeros(model.config.hidden_size, dtype=np.float32)
        scores = {}
        for lbl, cand in candidates.items():
            # zero correction at L0 = plain logprob scoring
            scores[lbl] = score_candidate_hooked(
                model, tok, prompt_text, cand, zero_vec, 0, device, INJ_MODE)
        bp[oid] = max(LABEL_ORDER, key=lambda lb: scores[lb])
    return bp


# ── Channel labels ────────────────────────────────────────────────────────────

def error_type(oid, golds, base_pred):
    g, p = golds[oid], base_pred[oid]
    return "correct" if p == g else f"{g}_to_{p}"


# ── DiffMean direction (index-based) ─────────────────────────────────────────

def difmean(train_oids, acts_all, golds, base_pred):
    nt_tc = [o for o in train_oids
             if golds[o] == "no_tool" and base_pred[o] == "tool_call"]
    ref   = [o for o in train_oids
             if base_pred[o] == golds[o] and golds[o] == "no_tool"]
    if len(nt_tc) < 5 or len(ref) < 5:
        return None
    err_mean = acts_all[nt_tc].mean(axis=0)
    ref_mean = acts_all[ref].mean(axis=0)
    raw   = ref_mean - err_mean
    nrm   = float(np.linalg.norm(raw))
    unit  = raw / (nrm + 1e-12)
    median_norm = float(np.median(np.linalg.norm(acts_all[train_oids], axis=1)))
    return {
        "unit": unit.astype(np.float32),
        "direction_norm": nrm,
        "median_norm": median_norm,
        "n_err": len(nt_tc),
        "n_ref": len(ref),
    }


# ── LR router (index-based) ──────────────────────────────────────────────────

def train_router(train_oids, val_oids, acts_all, golds, base_pred):
    pos = [o for o in train_oids
           if golds[o] == "no_tool" and base_pred[o] == "tool_call"]
    neg = [o for o in train_oids if base_pred[o] == golds[o]]
    if len(pos) < 5:
        return None

    X = acts_all[neg + pos]
    y = np.array([0] * len(neg) + [1] * len(pos))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42)

    cv_k = min(5, len(pos))
    cv_auc = None
    if cv_k >= 2:
        try:
            cv_auc = float(cross_val_score(clf, Xs, y, cv=cv_k, scoring="roc_auc").mean())
        except Exception:
            cv_auc = None
    clf.fit(Xs, y)

    # val AUC over val pos/neg
    vpos = [o for o in val_oids if golds[o] == "no_tool" and base_pred[o] == "tool_call"]
    vneg = [o for o in val_oids if base_pred[o] == golds[o]]
    val_auc = None
    if vpos and vneg:
        Xv = scaler.transform(acts_all[vneg + vpos])
        yv = np.array([0] * len(vneg) + [1] * len(vpos))
        probs = clf.predict_proba(Xv)[:, 1]
        val_auc = float(roc_auc_score(yv, probs)) if len(set(yv)) > 1 else None

    return {"scaler": scaler, "clf": clf, "val_auc": val_auc,
            "cv_auc": cv_auc, "n_pos": len(pos), "n_neg": len(neg)}


def router_scores(router, oids, acts_all):
    Xs = router["scaler"].transform(acts_all[oids])
    return router["clf"].predict_proba(Xs)[:, 1]


# ── Hooked prediction with arbitrary candidates ──────────────────────────────

def predict_corrected_c(model, tok, thought_prompt, corr_vec, inj_layer,
                        device, candidates):
    prompt_text = build_prompt_text(thought_prompt, tok)
    scores = {}
    for lbl, cand in candidates.items():
        scores[lbl] = score_candidate_hooked(
            model, tok, prompt_text, cand, corr_vec, inj_layer, device, INJ_MODE)
    return max(LABEL_ORDER, key=lambda lb: scores[lb])


# ── Metrics ───────────────────────────────────────────────────────────────────

def eval_split(oids, golds, base_pred, int_pred):
    """int_pred: dict oid->pred. Returns metric dict."""
    n = len(oids)
    base_acc = sum(base_pred[o] == golds[o] for o in oids) / n
    int_acc  = sum(int_pred[o]  == golds[o] for o in oids) / n
    fixed = sum(1 for o in oids
                if base_pred[o] != golds[o] and int_pred[o] == golds[o])
    broke = sum(1 for o in oids
                if base_pred[o] == golds[o] and int_pred[o] != golds[o])
    nt_tc_before = sum(1 for o in oids
                       if golds[o] == "no_tool" and base_pred[o] == "tool_call")
    nt_tc_after  = sum(1 for o in oids
                       if golds[o] == "no_tool" and int_pred[o] == "tool_call")
    n_gold_nt = sum(1 for o in oids if golds[o] == "no_tool")
    return {
        "n": n,
        "baseline_acc": round(base_acc, 5),
        "sakiko_acc":   round(int_acc, 5),
        "delta_acc":    round(int_acc - base_acc, 5),
        "fixed": fixed, "broke": broke, "net": fixed - broke,
        "nt_tc_before": nt_tc_before, "nt_tc_after": nt_tc_after,
        "false_toolcall_rate_before": round(nt_tc_before / n_gold_nt, 5) if n_gold_nt else 0.0,
        "false_toolcall_rate_after":  round(nt_tc_after  / n_gold_nt, 5) if n_gold_nt else 0.0,
    }


# ── Val sweep + locked config selection ──────────────────────────────────────

def val_sweep_and_lock(model, tok, val_oids, golds, base_pred, thought,
                       direction, router, acts_all, device):
    unit        = direction["unit"]
    median_norm = direction["median_norm"]
    vscores = router_scores(router, val_oids, acts_all)
    vscore_by_oid = {o: float(vscores[k]) for k, o in enumerate(val_oids)}

    grid = []
    for inj_layer in INJ_LAYERS:
        for alpha in ALPHAS:
            corr = (alpha * median_norm * unit).astype(np.float32)
            corrected = {}
            for o in val_oids:
                corrected[o] = predict_corrected_c(
                    model, tok, thought[o], corr, inj_layer, device,
                    CANDIDATE_SETS["yes_no"])
            for thr in THRESHOLDS:
                int_pred = {}
                touched = 0
                for o in val_oids:
                    if vscore_by_oid[o] >= thr and base_pred[o] == "tool_call":
                        int_pred[o] = corrected[o]; touched += 1
                    else:
                        int_pred[o] = base_pred[o]
                m = eval_split(val_oids, golds, base_pred, int_pred)
                grid.append({"inj_layer": inj_layer, "alpha": alpha,
                             "threshold": thr, "n_touched": touched, **m})
            gc.collect(); torch.cuda.empty_cache()

    best = max(grid, key=lambda x: (x["net"], -x["broke"], x["sakiko_acc"]))
    locked = {"inj_layer": best["inj_layer"], "alpha": best["alpha"],
              "threshold": best["threshold"],
              "val_net": best["net"], "val_fixed": best["fixed"],
              "val_broke": best["broke"],
              "val_acc_before": best["baseline_acc"],
              "val_acc_after": best["sakiko_acc"]}
    return locked, grid


def run_locked_test(model, tok, test_oids, golds, base_pred, thought,
                    direction, router, acts_all, locked, device,
                    candidates=None, gated=True):
    if candidates is None:
        candidates = CANDIDATE_SETS["yes_no"]
    unit        = direction["unit"]
    median_norm = direction["median_norm"]
    corr = (locked["alpha"] * median_norm * unit).astype(np.float32)
    tscores = router_scores(router, test_oids, acts_all)
    tscore_by_oid = {o: float(tscores[k]) for k, o in enumerate(test_oids)}

    int_pred = {}
    touched = 0
    for o in test_oids:
        fires = tscore_by_oid[o] >= locked["threshold"]
        gate_ok = (base_pred[o] == "tool_call") if gated else True
        if fires and gate_ok:
            int_pred[o] = predict_corrected_c(
                model, tok, thought[o], corr, locked["inj_layer"], device, candidates)
            touched += 1
        else:
            int_pred[o] = base_pred[o]
    m = eval_split(test_oids, golds, base_pred, int_pred)
    m["n_touched"] = touched
    m["router_fire_rate"] = round(sum(1 for o in test_oids
                                       if tscore_by_oid[o] >= locked["threshold"]) / len(test_oids), 5)
    return m, int_pred, tscore_by_oid


# ── PART 1: Multi-seed ────────────────────────────────────────────────────────

def part1_multiseed(model, tok, records, golds, thought, yesno_bp, acts_all, device):
    log.info("==================  PART 1: Multi-seed  ==================")
    results = []
    seed42_cache = {}

    for seed in SEEDS:
        log.info("---- seed %d ----", seed)
        train, val, test = stratified_split(records, seed=seed)
        train_oids = [r["original_id"] for r in train]
        val_oids   = [r["original_id"] for r in val]
        test_oids  = [r["original_id"] for r in test]

        direction = difmean(train_oids, acts_all, golds, yesno_bp)
        router    = train_router(train_oids, val_oids, acts_all, golds, yesno_bp)
        if direction is None or router is None:
            log.warning("seed %d: insufficient data, skipping", seed)
            continue

        locked, grid = val_sweep_and_lock(
            model, tok, val_oids, golds, yesno_bp, thought,
            direction, router, acts_all, device)
        test_m, int_pred, tscores = run_locked_test(
            model, tok, test_oids, golds, yesno_bp, thought,
            direction, router, acts_all, locked, device)

        row = {
            "seed": seed,
            "n_train": len(train_oids), "n_val": len(val_oids), "n_test": len(test_oids),
            "router_val_auc": round(router["val_auc"], 5) if router["val_auc"] else None,
            "router_cv_auc":  round(router["cv_auc"], 5) if router["cv_auc"] else None,
            "direction_n_err": direction["n_err"], "direction_n_ref": direction["n_ref"],
            "selected_inj_layer": locked["inj_layer"],
            "selected_alpha": locked["alpha"],
            "selected_threshold": locked["threshold"],
            "val_net": locked["val_net"],
            "test_baseline_acc": test_m["baseline_acc"],
            "test_sakiko_acc":   test_m["sakiko_acc"],
            "test_delta_acc":    test_m["delta_acc"],
            "test_fixed": test_m["fixed"], "test_broke": test_m["broke"],
            "test_net": test_m["net"],
            "nt_tc_before": test_m["nt_tc_before"], "nt_tc_after": test_m["nt_tc_after"],
            "false_toolcall_rate_before": test_m["false_toolcall_rate_before"],
            "false_toolcall_rate_after":  test_m["false_toolcall_rate_after"],
            "n_touched": test_m["n_touched"],
            "router_fire_rate": test_m["router_fire_rate"],
        }
        results.append(row)
        log.info("seed %d: acc %.4f→%.4f  fixed=%d broke=%d net=%+d  nt_tc %d→%d  "
                 "cfg=L%d a=%.1f thr=%.2f  valAUC=%s",
                 seed, test_m["baseline_acc"], test_m["sakiko_acc"],
                 test_m["fixed"], test_m["broke"], test_m["net"],
                 test_m["nt_tc_before"], test_m["nt_tc_after"],
                 locked["inj_layer"], locked["alpha"], locked["threshold"],
                 row["router_val_auc"])

        if seed == 42:
            seed42_cache = {
                "train_oids": train_oids, "val_oids": val_oids, "test_oids": test_oids,
                "direction": direction, "router": router, "locked": locked,
                "test_metrics": test_m,
            }

    return results, seed42_cache


# ── PART 2: Random placebo distribution ──────────────────────────────────────

def part2_random_placebo(model, tok, golds, yesno_bp, thought, acts_all,
                         seed42, device):
    log.info("==================  PART 2: Random placebo  ==================")
    test_oids   = seed42["test_oids"]
    direction   = seed42["direction"]
    router      = seed42["router"]
    locked      = seed42["locked"]
    real_net    = seed42["test_metrics"]["net"]
    median_norm = direction["median_norm"]
    D           = acts_all.shape[1]

    tscores = router_scores(router, test_oids, acts_all)
    tscore_by_oid = {o: float(tscores[k]) for k, o in enumerate(test_oids)}
    # which samples are touched (same router → same fired+gated set for all dirs)
    touched_oids = [o for o in test_oids
                    if tscore_by_oid[o] >= locked["threshold"]
                    and yesno_bp[o] == "tool_call"]

    random_nets = []
    random_details = []
    for i in range(N_RANDOM_PLACEBO):
        rng = np.random.RandomState(1000 + i)
        rdir = rng.randn(D).astype(np.float32)
        rdir = rdir / (np.linalg.norm(rdir) + 1e-12)
        corr = (locked["alpha"] * median_norm * rdir).astype(np.float32)

        int_pred = dict(yesno_bp)  # start from baseline
        for o in touched_oids:
            int_pred[o] = predict_corrected_c(
                model, tok, thought[o], corr, locked["inj_layer"], device,
                CANDIDATE_SETS["yes_no"])
        m = eval_split(test_oids, golds, yesno_bp, int_pred)
        random_nets.append(m["net"])
        random_details.append({"i": i, "net": m["net"], "fixed": m["fixed"],
                               "broke": m["broke"], "nt_tc_after": m["nt_tc_after"]})
        gc.collect(); torch.cuda.empty_cache()

    arr = np.array(random_nets, dtype=float)
    n_ge = int((arr >= real_net).sum())
    n_gt = int((arr > real_net).sum())
    summary = {
        "real_net": real_net,
        "n_random": N_RANDOM_PLACEBO,
        "random_mean": round(float(arr.mean()), 4),
        "random_std":  round(float(arr.std()), 4),
        "random_max":  int(arr.max()),
        "random_min":  int(arr.min()),
        "random_median": float(np.median(arr)),
        "n_random_ge_real": n_ge,
        "n_random_gt_real": n_gt,
        "real_rank_among_random": n_gt + 1,   # 1 = strictly best
        "real_percentile": round(100.0 * (1 - n_ge / N_RANDOM_PLACEBO), 1),
        "n_touched": len(touched_oids),
        "locked_cfg": {k: locked[k] for k in ("inj_layer", "alpha", "threshold")},
        "random_nets": random_nets,
    }
    log.info("Real net=%d | random mean=%.2f std=%.2f max=%d min=%d | "
             "real rank=%d/%d (%.0f%%ile) | n_ge=%d",
             real_net, summary["random_mean"], summary["random_std"],
             summary["random_max"], summary["random_min"],
             summary["real_rank_among_random"], N_RANDOM_PLACEBO,
             summary["real_percentile"], n_ge)
    return summary, random_details


# ── PART 3: Candidate-string sensitivity ─────────────────────────────────────

def part3_candidate_sensitivity(model, tok, records, golds, thought,
                                yesno_bp, acts_all, seed42, device):
    log.info("==================  PART 3: Candidate sensitivity  ==================")
    # Use the seed=42 split throughout
    train_oids = seed42["train_oids"]
    val_oids   = seed42["val_oids"]
    test_oids  = seed42["test_oids"]
    locked     = seed42["locked"]     # reuse already-selected (layer, alpha, threshold)
    all_oids   = train_oids + val_oids + test_oids

    out = {}
    for name, cands in CANDIDATE_SETS.items():
        log.info("---- candidate set: %s = %s ----", name, cands)
        if name == "yes_no":
            base_pred = yesno_bp
        else:
            base_pred = rescore_baseline(model, tok, all_oids, thought, cands, device)

        # Build direction + router from THIS candidate set's baseline labels
        direction = difmean(train_oids, acts_all, golds, base_pred)
        router    = train_router(train_oids, val_oids, acts_all, golds, base_pred)

        test_base_acc = sum(base_pred[o] == golds[o] for o in test_oids) / len(test_oids)

        entry = {
            "candidates": cands,
            "test_baseline_acc": round(test_base_acc, 5),
            "train_nt_tc": direction["n_err"] if direction else None,
            "router_val_auc": round(router["val_auc"], 5) if (router and router["val_auc"]) else None,
        }

        if direction is not None and router is not None:
            test_m, _, _ = run_locked_test(
                model, tok, test_oids, golds, base_pred, thought,
                direction, router, acts_all, locked, device, candidates=cands)
            entry.update({
                "sakiko_net": test_m["net"],
                "sakiko_fixed": test_m["fixed"],
                "sakiko_broke": test_m["broke"],
                "test_sakiko_acc": test_m["sakiko_acc"],
                "test_delta_acc": test_m["delta_acc"],
                "nt_tc_before": test_m["nt_tc_before"],
                "nt_tc_after": test_m["nt_tc_after"],
                "n_touched": test_m["n_touched"],
                "locked_cfg_reused": {k: locked[k] for k in ("inj_layer", "alpha", "threshold")},
            })
            log.info("%s: base_acc=%.4f  sakiko_acc=%.4f  net=%+d (fixed=%d broke=%d)  nt_tc %d→%d",
                     name, test_base_acc, test_m["sakiko_acc"], test_m["net"],
                     test_m["fixed"], test_m["broke"],
                     test_m["nt_tc_before"], test_m["nt_tc_after"])
        else:
            log.warning("%s: insufficient nt_tc structure to build direction/router", name)
        out[name] = entry
    return out


# ── Report writers ────────────────────────────────────────────────────────────

QUANT_NOTE = (
    "All MetaTool-Binary experiments (baseline, pilot, and this audit) use "
    "**4-bit NF4 quantization, float16 compute, device_map=auto** on Phi-3.5-mini-instruct. "
    "The W2C main experiments may have used a different precision setting "
    "(the W2C V3.1 pipeline loads the model in float16 without 4-bit quantization). "
    "Therefore MetaTool-Binary results should be reported as a **binary external pilot "
    "under this quantized model setting**, and are NOT numerically comparable to W2C "
    "accuracy. An fp16 confirmation of the MetaTool-Binary pilot would be required before "
    "claiming precision-independent equivalence."
)


def write_outputs(multiseed, placebo, placebo_details, cand_sens):
    # multiseed json
    (OUT_DIR / "metatool_binary_multiseed_results.json").write_text(
        json.dumps(multiseed, indent=2, ensure_ascii=False), encoding="utf-8")
    # placebo json
    (OUT_DIR / "metatool_binary_random_placebo.json").write_text(
        json.dumps({"summary": placebo, "per_direction": placebo_details},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    # candidate sensitivity json
    (OUT_DIR / "metatool_binary_candidate_sensitivity.json").write_text(
        json.dumps(cand_sens, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── aggregate stats ──
    nets   = [r["test_net"] for r in multiseed]
    deltas = [r["test_delta_acc"] for r in multiseed]
    aucs   = [r["router_val_auc"] for r in multiseed if r["router_val_auc"] is not None]
    n_pos  = sum(1 for x in nets if x > 0)

    agg = {
        "n_seeds": len(multiseed),
        "net_mean": round(float(np.mean(nets)), 3),
        "net_std":  round(float(np.std(nets)), 3),
        "net_min":  int(np.min(nets)),
        "net_max":  int(np.max(nets)),
        "n_positive_net": n_pos,
        "delta_acc_mean": round(float(np.mean(deltas)), 5),
        "router_val_auc_mean": round(float(np.mean(aucs)), 5) if aucs else None,
    }

    summary_json = {
        "dataset": "MetaTool-Binary",
        "model": "microsoft/Phi-3.5-mini-instruct",
        "quantization": "4-bit NF4 float16 device_map=auto",
        "channel": "nt_tc (gold=no_tool, baseline_pred=tool_call)",
        "candidate_strings_primary": CANDIDATE_SETS["yes_no"],
        "part1_aggregate": agg,
        "part1_per_seed": multiseed,
        "part2_random_placebo": placebo,
        "part3_candidate_sensitivity": cand_sens,
        "part4_quantization_note": QUANT_NOTE,
    }
    (OUT_DIR / "metatool_binary_robustness_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    write_md(multiseed, agg, placebo, cand_sens)
    return agg


def write_md(multiseed, agg, placebo, cand_sens):
    L = []
    A = L.append
    A("# MetaTool-Binary SAKIKO — Robustness Audit\n")
    A("> **CAVEATS**")
    A("> - MetaTool-Binary, NOT W2C-4way. Binary labels only (`tool_call` / `no_tool`).")
    A("> - Tests the external binary over-calling channel (nt_tc) only.")
    A("> - NOT numerically comparable to W2C accuracy.")
    A("> - 4-bit NF4 quantization throughout (see Part 4).\n")

    # Part 1
    A("## Part 1 — Multi-seed split robustness\n")
    A("Per seed: independent 70/15/15 stratified split of the same 1,040 samples; "
      "DiffMean direction + LR router fit on train; (layer, alpha, threshold) tuned on "
      "val only; locked test evaluated once.\n")
    A("| Seed | Base acc | SAKIKO acc | Δacc | Fixed | Broke | Net | nt_tc→ | FTC rate→ | Layer | α | Thr | val AUC |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in multiseed:
        A(f"| {r['seed']} | {r['test_baseline_acc']:.4f} | {r['test_sakiko_acc']:.4f} | "
          f"{r['test_delta_acc']:+.4f} | {r['test_fixed']} | {r['test_broke']} | "
          f"**{r['test_net']:+d}** | {r['nt_tc_before']}→{r['nt_tc_after']} | "
          f"{r['false_toolcall_rate_before']:.3f}→{r['false_toolcall_rate_after']:.3f} | "
          f"L{r['selected_inj_layer']} | {r['selected_alpha']} | {r['selected_threshold']} | "
          f"{r['router_val_auc']} |")
    A("")
    A(f"**Aggregate**: Net mean = {agg['net_mean']} ± {agg['net_std']} "
      f"(min {agg['net_min']}, max {agg['net_max']}); "
      f"positive-net seeds = {agg['n_positive_net']}/{agg['n_seeds']}; "
      f"mean Δacc = {agg['delta_acc_mean']:+.4f}; "
      f"mean router val AUC = {agg['router_val_auc_mean']}.\n")

    # Part 2
    A("## Part 2 — Random placebo distribution (seed=42 locked cfg)\n")
    p = placebo
    A(f"Locked cfg: L{p['locked_cfg']['inj_layer']}, α={p['locked_cfg']['alpha']}, "
      f"thr={p['locked_cfg']['threshold']}; {p['n_touched']} samples touched (same router "
      f"→ same fired+gated set for every direction).\n")
    A("| Quantity | Value |")
    A("|---|---|")
    A(f"| Real net | **{p['real_net']}** |")
    A(f"| Random directions | {p['n_random']} |")
    A(f"| Random net mean | {p['random_mean']} |")
    A(f"| Random net std | {p['random_std']} |")
    A(f"| Random net max | {p['random_max']} |")
    A(f"| Random net min | {p['random_min']} |")
    A(f"| Random net median | {p['random_median']} |")
    A(f"| # random ≥ real | {p['n_random_ge_real']} |")
    A(f"| Real rank among random | {p['real_rank_among_random']} / {p['n_random']} |")
    A(f"| Real percentile | {p['real_percentile']}%ile |")
    A("")

    # Part 3
    A("## Part 3 — Candidate-string sensitivity (seed=42)\n")
    A("Direction + router rebuilt per candidate set (activations are candidate-independent); "
      "locked (layer, α, threshold) reused from the seed=42 selection.\n")
    A("| Candidate set | tool_call / no_tool | Base acc | SAKIKO acc | Net | nt_tc→ | val AUC |")
    A("|---|---|---|---|---|---|---|")
    for name, e in cand_sens.items():
        c = e["candidates"]
        sa  = e.get("test_sakiko_acc", None)
        net = e.get("sakiko_net", None)
        ntb = e.get("nt_tc_before", None); nta = e.get("nt_tc_after", None)
        nt_str = f"{ntb}→{nta}" if ntb is not None else "—"
        A(f"| {name} | `{c['tool_call']}` / `{c['no_tool']}` | "
          f"{e['test_baseline_acc']:.4f} | {sa if sa is None else f'{sa:.4f}'} | "
          f"{net if net is None else f'{net:+d}'} | {nt_str} | {e.get('router_val_auc')} |")
    A("")

    # Part 4
    A("## Part 4 — Quantization note\n")
    A(QUANT_NOTE + "\n")

    # Conclusions
    A("## Conclusions\n")
    pos_all = agg["n_positive_net"] == agg["n_seeds"]
    q1 = ("**Yes** — every seed yields positive Net." if pos_all
          else f"**Mostly** — {agg['n_positive_net']}/{agg['n_seeds']} seeds positive.")
    A(f"1. **Positive across seeds?** {q1} "
      f"Net = {agg['net_mean']} ± {agg['net_std']} (min {agg['net_min']}).\n")

    rank = placebo["real_rank_among_random"]
    if rank == 1 and placebo["real_net"] > placebo["random_max"]:
        q2 = (f"**Yes** — Real net ({placebo['real_net']}) strictly exceeds all "
              f"{placebo['n_random']} random directions "
              f"(random max = {placebo['random_max']}, mean = {placebo['random_mean']}).")
    elif placebo["real_net"] > placebo["random_mean"] + 2 * placebo["random_std"]:
        q2 = (f"**Yes** — Real net ({placebo['real_net']}) is >2σ above the random mean "
              f"({placebo['random_mean']} ± {placebo['random_std']}); rank {rank}/{placebo['n_random']}.")
    else:
        q2 = (f"**Partially** — Real net ({placebo['real_net']}) ranks {rank}/{placebo['n_random']} "
              f"vs random (mean {placebo['random_mean']}, max {placebo['random_max']}).")
    A(f"2. **Real clearly stronger than random?** {q2}\n")

    base_accs = [e["test_baseline_acc"] for e in cand_sens.values()]
    nets3 = [e.get("sakiko_net") for e in cand_sens.values() if e.get("sakiko_net") is not None]
    spread = max(base_accs) - min(base_accs)
    if nets3 and all(x > 0 for x in nets3):
        q3 = (f"Baseline accuracy varies by {spread:.3f} across wordings, but SAKIKO net "
              f"stays positive for every candidate set (nets: {nets3}). "
              f"The correction is **not an artifact of Yes/No tokenization**.")
    else:
        q3 = (f"Baseline accuracy varies by {spread:.3f}; SAKIKO net = {nets3}. "
              f"Result shows some wording dependence — interpret with care.")
    A(f"3. **Sensitive to candidate wording?** {q3}\n")

    if pos_all and rank <= 2 and nets3 and all(x > 0 for x in nets3):
        q4 = ("**Robust external binary validation** (within this quantized setting): "
              "consistent across seeds, direction-specific vs random, and stable across wordings.")
    elif agg["n_positive_net"] >= agg["n_seeds"] - 1:
        q4 = ("**Strong promising pilot** — consistent and direction-specific, but report as a "
              "pilot pending fp16 confirmation.")
    else:
        q4 = "**Promising pilot only** — further work needed before a robustness claim."
    A(f"4. **Robust validation or promising pilot?** {q4}\n")

    A("5. **Recommended next step:**")
    if pos_all and rank == 1 and nets3 and all(x > 0 for x in nets3):
        A("   - Write this up as a **cross-dataset binary extension result** (primary).")
        A("   - Run an **fp16 confirmation** of seed=42 to close the quantization gap (cheap, high value).")
        A("   - Then proceed to **ToolSandbox-derived** multi-class annotation.")
    else:
        A("   - Run **fp16 confirmation** before any external claim.")
        A("   - Keep framing as a pilot; defer ToolSandbox until confirmed.")
    A("")

    (OUT_DIR / "metatool_binary_robustness_summary.md").write_text("\n".join(L), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="phi35")
    ap.add_argument("--obs_layer", type=int, default=OBS_LAYER)
    ap.add_argument("--force_acts", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    records, golds, thought = load_all_records()
    yesno_bp = load_yesno_baseline()
    log.info("Loaded %d records; %d Yes/No baseline preds", len(records), len(yesno_bp))

    mcfg, defaults = load_model_cfg(args.model_key)
    model_path = mcfg.get("local_path", mcfg["model_id"])
    tok = load_tokenizer(model_path, mcfg, defaults)
    model, device = load_model(model_path, mcfg, defaults)

    acts_all = extract_acts_all(model, tok, golds, thought,
                                args.obs_layer, device, force=args.force_acts)

    multiseed, seed42 = part1_multiseed(
        model, tok, records, golds, thought, yesno_bp, acts_all, device)

    placebo, placebo_details = part2_random_placebo(
        model, tok, golds, yesno_bp, thought, acts_all, seed42, device)

    cand_sens = part3_candidate_sensitivity(
        model, tok, records, golds, thought, yesno_bp, acts_all, seed42, device)

    agg = write_outputs(multiseed, placebo, placebo_details, cand_sens)

    log.info("==================  DONE in %.1f min  ==================",
             (time.time() - t0) / 60)
    log.info("Part1 net mean=%.2f (%d/%d positive) | Part2 real rank=%d/%d | "
             "Part3 nets=%s",
             agg["net_mean"], agg["n_positive_net"], agg["n_seeds"],
             placebo["real_rank_among_random"], placebo["n_random"],
             [cand_sens[k].get("sakiko_net") for k in cand_sens])


if __name__ == "__main__":
    main()
