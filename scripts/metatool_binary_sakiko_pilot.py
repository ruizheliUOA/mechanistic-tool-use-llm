"""
metatool_binary_sakiko_pilot.py
================================
MetaTool-native binary SAKIKO pilot — nt_tc over-calling channel.

Error channel:
  nt_tc: gold=no_tool, baseline_pred=tool_call  (false tool-call / over-calling)

Stages:
  1. Load baseline details + assign to splits
  2. Extract L18 MLP activations for all 1040 samples (cached)
  3. DiffMean direction on train: mean(correct_nt) - mean(nt_tc_errors)
  4. LR router for nt_tc (obs=L18), evaluate on val
  5. Val sweep: alpha=[0.5,1.0,2.0,4.0,6.0] × inj_layer=[14,16,18] × threshold
  6. Lock best val config → locked test eval
  7. Placebo controls (real / random / reverse / ungated)
  8. Write pilot report

Decision gate:
  Only apply correction when router fires AND baseline_pred == "tool_call".
  Block if baseline_pred already "no_tool" (no correction needed).

Candidate strings (matching baseline evaluator):
  tool_call → "Yes"   (single token)
  no_tool   → "No"    (single token)

Intervention formula:
  correction = alpha * median_norm * unit(direction)
  h' = h + correction    (added to MLP output at inj_layer)

Usage:
  cd /path/to/project
  /path/to/project/.venv/bin/python scripts/metatool_binary_sakiko_pilot.py \\
      --model_key phi35

IMPORTANT:
  - Uses MetaTool-Binary, NOT W2C-4way.
  - Binary labels only: tool_call / no_tool.
  - Not directly comparable to W2C accuracy.
  - 4-bit NF4 quantization (same as baseline evaluator).
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
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
CFG_MODELS  = ROOT / "configs" / "models.yaml"
DATA_DIR    = ROOT / "data" / "processed" / "metatool_binary"
CACHE_DIR   = DATA_DIR / "cache"
OUT_DIR     = ROOT / "final" / "results" / "dataset_extension"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {"tool_call": "Yes", "no_tool": "No"}
LABEL_ORDER = ["tool_call", "no_tool"]

# Scan grid
ALPHAS      = [0.5, 1.0, 2.0, 4.0, 6.0]
INJ_LAYERS  = [14, 16, 18]
THRESHOLDS  = [0.40, 0.50, 0.60, 0.70, 0.80]
OBS_LAYER   = 18
INJ_MODE    = "mlp_all"   # add correction to all positions, matching W2C rfi_tc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mt_sakiko")


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_model_cfg(model_key: str) -> dict:
    with open(CFG_MODELS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if model_key not in cfg["models"]:
        log.error("model_key '%s' not in models.yaml", model_key)
        sys.exit(1)
    return cfg["models"][model_key], cfg["defaults"]


# ─── Tokenizer & model ────────────────────────────────────────────────────────

def build_prompt_text(thought_prompt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": thought_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = ""
        for m in messages:
            text += f"<|{m['role']}|>\n{m['content']}\n"
        return text + "<|assistant|>\n"


def load_tokenizer(model_path: str, mcfg: dict, defaults: dict):
    log.info("Loading tokenizer from %s", model_path)
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=mcfg.get("trust_remote_code",
                                    defaults.get("trust_remote_code", True)),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_path: str, mcfg: dict, defaults: dict):
    use_4bit  = defaults.get("load_in_4bit", False)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(defaults.get("torch_dtype", "float16"), torch.float16)

    quant_kwargs: dict = {}
    if use_4bit and torch.cuda.is_available():
        log.info("4-bit NF4 quantization enabled")
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
    attn = mcfg.get("attn_implementation")
    if attn:
        quant_kwargs["attn_implementation"] = attn

    log.info("Loading model (4bit=%s, dtype=%s) …", use_4bit, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code",
                                    defaults.get("trust_remote_code", True)),
        **quant_kwargs,
    )
    model.eval()
    device = next(model.parameters()).device
    log.info("Model on device: %s  |  layers: %d  |  hidden: %d",
             device, len(model.model.layers), model.config.hidden_size)
    return model, device


# ─── Stage 1: Load data and split baseline ────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stage1_split_baseline() -> tuple[list, list, list, dict, dict]:
    """
    Load train/val/test records and map baseline details onto each split.
    Returns: train_recs, val_recs, test_recs, baseline_by_id, split_stats
    """
    log.info("=== Stage 1: Load data and baseline details ===")

    train_recs = load_jsonl(DATA_DIR / "metatool_binary_train.jsonl")
    val_recs   = load_jsonl(DATA_DIR / "metatool_binary_val.jsonl")
    test_recs  = load_jsonl(DATA_DIR / "metatool_binary_test.jsonl")

    log.info("Split sizes: train=%d val=%d test=%d",
             len(train_recs), len(val_recs), len(test_recs))

    details_path = OUT_DIR / "metatool_binary_baseline_details.jsonl"
    if not details_path.exists():
        log.error("Baseline details not found: %s", details_path)
        log.error("Run eval_metatool_binary_baseline.py first on the full set.")
        sys.exit(1)

    raw_details = load_jsonl(details_path)
    baseline_by_id = {r["sample_id"]: r for r in raw_details}
    log.info("Loaded %d baseline detail records", len(raw_details))

    def attach_baseline(recs: list[dict], split_name: str) -> list[dict]:
        enriched = []
        missing = 0
        for r in recs:
            sid = r["sample_id"]
            if sid not in baseline_by_id:
                missing += 1
                continue
            b = baseline_by_id[sid]
            r2 = dict(r)
            r2["baseline_pred"]  = b["pred_response_mode"]
            r2["baseline_gold"]  = b["gold_response_mode"]
            r2["baseline_correct"] = b["correct"]
            r2["logp_tc"] = b["scored"]["tool_call"]["avg_logp"]
            r2["logp_nt"] = b["scored"]["no_tool"]["avg_logp"]
            r2["margin"]  = r2["logp_tc"] - r2["logp_nt"]
            gold = r["gold_response_mode"]
            pred = r2["baseline_pred"]
            r2["error_type"] = "correct" if pred == gold else f"{gold}_to_{pred}"
            enriched.append(r2)
        if missing:
            log.warning("Split %s: %d samples missing from baseline details",
                        split_name, missing)
        return enriched

    train_recs = attach_baseline(train_recs, "train")
    val_recs   = attach_baseline(val_recs,   "val")
    test_recs  = attach_baseline(test_recs,  "test")

    def split_stats(recs, name):
        n = len(recs)
        ec = Counter(r["error_type"] for r in recs)
        acc = sum(r["baseline_correct"] for r in recs) / n if n else 0
        nt_tc = ec.get("no_tool_to_tool_call", 0)
        tc_nt = ec.get("tool_call_to_no_tool", 0)
        log.info("  %s (n=%d): acc=%.3f  nt_tc=%d  tc_nt=%d  correct=%d",
                 name, n, acc, nt_tc, tc_nt, ec.get("correct", 0))
        return {"n": n, "accuracy": round(acc, 4),
                "nt_tc": nt_tc, "tc_nt": tc_nt,
                "error_counts": dict(ec)}

    stats = {
        "train": split_stats(train_recs, "train"),
        "val":   split_stats(val_recs,   "val"),
        "test":  split_stats(test_recs,  "test"),
    }
    return train_recs, val_recs, test_recs, baseline_by_id, stats


# ─── Stage 2: Extract activations ────────────────────────────────────────────

def stage2_extract_activations(
    model, tok, train_recs, val_recs, test_recs,
    obs_layer: int, device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract MLP output at last prompt position for obs_layer.
    Uses deterministic ordering: all = train + val + test (by original_id).
    Caches per-split arrays.
    """
    log.info("=== Stage 2: Extract L%d activations ===", obs_layer)

    train_cache = CACHE_DIR / f"acts_train_L{obs_layer}.npy"
    val_cache   = CACHE_DIR / f"acts_val_L{obs_layer}.npy"
    test_cache  = CACHE_DIR / f"acts_test_L{obs_layer}.npy"

    if train_cache.exists() and val_cache.exists() and test_cache.exists():
        train_acts = np.load(train_cache)
        val_acts   = np.load(val_cache)
        test_acts  = np.load(test_cache)
        log.info("Loaded cached activations: train=%s val=%s test=%s",
                 train_acts.shape, val_acts.shape, test_acts.shape)
        return train_acts, val_acts, test_acts

    def extract_split(recs: list[dict], split_name: str) -> np.ndarray:
        D = model.config.hidden_size
        n = len(recs)
        acts = np.zeros((n, D), dtype=np.float32)
        mlp = model.model.layers[obs_layer].mlp

        for i, rec in enumerate(tqdm(recs, desc=f"Acts-L{obs_layer}-{split_name}")):
            prompt_text = build_prompt_text(rec["thought_prompt"], tok)
            token_ids = tok.encode(prompt_text, add_special_tokens=False)
            last_pos  = len(token_ids) - 1
            input_ids = torch.tensor([token_ids], device=device)

            captured = {}
            def _hook(module, inp, out, _c=captured, _p=last_pos):
                _c["act"] = out[0, _p, :].detach().cpu().float().numpy()

            h = mlp.register_forward_hook(_hook)
            with torch.no_grad():
                model(input_ids)
            h.remove()
            acts[i] = captured["act"]
            del input_ids

            if (i + 1) % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        return acts

    train_acts = extract_split(train_recs, "train")
    np.save(train_cache, train_acts)
    log.info("Saved train acts: %s → %s", train_acts.shape, train_cache)

    val_acts = extract_split(val_recs, "val")
    np.save(val_cache, val_acts)
    log.info("Saved val acts: %s → %s", val_acts.shape, val_cache)

    test_acts = extract_split(test_recs, "test")
    np.save(test_cache, test_acts)
    log.info("Saved test acts: %s → %s", test_acts.shape, test_cache)

    return train_acts, val_acts, test_acts


# ─── Stage 3: DiffMean direction ─────────────────────────────────────────────

def stage3_difmean_direction(
    train_recs: list[dict], train_acts: np.ndarray
) -> dict:
    """
    DiffMean for nt_tc channel:
      direction = mean(correct_no_tool_acts) - mean(nt_tc_error_acts)
    """
    log.info("=== Stage 3: DiffMean direction (nt_tc) ===")

    nt_tc_idx = [i for i, r in enumerate(train_recs)
                 if r["error_type"] == "no_tool_to_tool_call"]
    ref_idx   = [i for i, r in enumerate(train_recs)
                 if r["baseline_correct"] and r["gold_response_mode"] == "no_tool"]

    log.info("nt_tc errors in train: %d", len(nt_tc_idx))
    log.info("correct no_tool in train: %d", len(ref_idx))

    if len(nt_tc_idx) < 5:
        log.error("Insufficient nt_tc errors in train (%d). Aborting.", len(nt_tc_idx))
        sys.exit(1)
    if len(ref_idx) < 5:
        log.error("Insufficient correct no_tool in train (%d). Aborting.", len(ref_idx))
        sys.exit(1)

    err_mean = train_acts[nt_tc_idx].mean(axis=0)     # (D,)
    ref_mean = train_acts[ref_idx].mean(axis=0)       # (D,)

    raw_direction = ref_mean - err_mean               # (D,)
    direction_norm = float(np.linalg.norm(raw_direction))
    unit_direction = raw_direction / (direction_norm + 1e-12)

    # Median L2 norm of train activations (reference scale)
    all_norms = np.linalg.norm(train_acts, axis=1)
    median_norm = float(np.median(all_norms))

    log.info("DiffMean direction: norm=%.4f  median_train_act_norm=%.4f",
             direction_norm, median_norm)

    direction_info = {
        "unit_direction": unit_direction.astype(np.float32),
        "direction_norm": direction_norm,
        "median_norm":    median_norm,
        "n_err":          len(nt_tc_idx),
        "n_ref":          len(ref_idx),
        "err_mean_norm":  float(np.linalg.norm(err_mean)),
        "ref_mean_norm":  float(np.linalg.norm(ref_mean)),
    }
    return direction_info


# ─── Stage 4: LR router ───────────────────────────────────────────────────────

def stage4_train_router(
    train_recs: list[dict], train_acts: np.ndarray,
    val_recs: list[dict], val_acts: np.ndarray,
) -> dict:
    """
    Logistic Regression router for nt_tc channel.
    Positive: nt_tc errors in train
    Negative: all correct train samples
    """
    log.info("=== Stage 4: LR router for nt_tc ===")

    pos_idx = [i for i, r in enumerate(train_recs)
               if r["error_type"] == "no_tool_to_tool_call"]
    neg_idx = [i for i, r in enumerate(train_recs)
               if r["baseline_correct"]]

    log.info("Router train: pos=%d (nt_tc), neg=%d (correct)", len(pos_idx), len(neg_idx))

    X_pos = train_acts[pos_idx]
    X_neg = train_acts[neg_idx]
    X = np.concatenate([X_neg, X_pos])
    y = np.array([0] * len(neg_idx) + [1] * len(pos_idx))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="liblinear", random_state=42)
    cv_k = min(5, len(pos_idx))
    if cv_k >= 2:
        cv_scores = cross_val_score(clf, X_s, y, cv=cv_k, scoring="roc_auc")
        log.info("Router CV AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    clf.fit(X_s, y)

    # Evaluate on val
    val_pos_idx = [i for i, r in enumerate(val_recs)
                   if r["error_type"] == "no_tool_to_tool_call"]
    val_neg_idx = [i for i, r in enumerate(val_recs)
                   if r["baseline_correct"]]
    log.info("Router val: pos=%d (nt_tc), neg=%d (correct)",
             len(val_pos_idx), len(val_neg_idx))

    val_all_idx = val_neg_idx + val_pos_idx
    X_val = val_acts[val_all_idx]
    y_val = np.array([0] * len(val_neg_idx) + [1] * len(val_pos_idx))
    X_val_s = scaler.transform(X_val)
    val_probs = clf.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
    log.info("Router val AUC: %.4f", val_auc)

    # Full val scores (for all val samples, needed for sweep)
    val_scores_all = clf.predict_proba(
        scaler.transform(val_acts))[:, 1]

    # Full test scores (precomputed for locked test)
    router_info = {
        "scaler": scaler,
        "clf": clf,
        "val_auc": round(val_auc, 5),
        "n_train_pos": len(pos_idx),
        "n_train_neg": len(neg_idx),
        "val_scores_all": val_scores_all,  # shape (len(val_recs),)
    }

    # Print precision-recall at each threshold
    if len(set(y_val)) > 1:
        prec, rec, thr = precision_recall_curve(y_val, val_probs)
        log.info("Router val P/R at thresholds:")
        for t in THRESHOLDS:
            idx_ = np.searchsorted(thr, t)
            p_ = prec[idx_] if idx_ < len(prec) else 0.0
            r_ = rec[idx_]  if idx_ < len(rec)  else 0.0
            fire_ = int((val_scores_all >= t).sum())
            log.info("  thr=%.2f  prec=%.3f  rec=%.3f  fires_on_val=%d",
                     t, p_, r_, fire_)

    return router_info


# ─── Hooked scoring ───────────────────────────────────────────────────────────

def score_candidate_hooked(
    model, tok,
    prompt_text: str,
    candidate_text: str,
    correction_vec: np.ndarray,
    inj_layer: int,
    device,
    mode: str = "mlp_all",
) -> float:
    """
    Score candidate_text continuation of prompt_text,
    with correction_vec added to MLP output at inj_layer.
    """
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    cand_ids   = tok.encode(candidate_text, add_special_tokens=False)
    if not cand_ids:
        return -1e9

    input_ids  = torch.tensor([prompt_ids + cand_ids], device=device)
    prompt_len = len(prompt_ids)
    cand_len   = len(cand_ids)
    corr_t     = torch.from_numpy(correction_vec).to(torch.float32).to(device)

    def _hook(module, inp, out):
        if mode == "mlp_all":
            out[:, :, :] += corr_t.to(out.dtype)
        elif mode == "mlp_prompt":
            out[:, :prompt_len, :] += corr_t.to(out.dtype)
        elif mode == "mlp_last":
            out[0, prompt_len - 1, :] += corr_t.to(out.dtype)
        return out

    mlp = model.model.layers[inj_layer].mlp
    h   = mlp.register_forward_hook(_hook)
    with torch.no_grad():
        logits = model(input_ids).logits
    h.remove()
    del input_ids

    # avg logp over candidate tokens
    logits_slice = logits[0, prompt_len - 1: prompt_len + cand_len - 1, :]
    log_probs    = torch.log_softmax(logits_slice.float(), dim=-1)
    token_logps  = [log_probs[i, cand_ids[i]].item() for i in range(cand_len)]
    return sum(token_logps) / cand_len


def predict_corrected(
    model, tok, rec: dict,
    correction_vec: np.ndarray,
    inj_layer: int,
    device,
    mode: str = "mlp_all",
) -> str:
    prompt_text = build_prompt_text(rec["thought_prompt"], tok)
    scores = {}
    for lbl, cand in CANDIDATES.items():
        scores[lbl] = score_candidate_hooked(
            model, tok, prompt_text, cand, correction_vec,
            inj_layer, device, mode)
    return max(LABEL_ORDER, key=lambda lb: scores[lb])


# ─── Stage 5: Val sweep ───────────────────────────────────────────────────────

def sweep_val(
    model, tok,
    val_recs: list[dict],
    val_acts: np.ndarray,
    router_info: dict,
    direction_info: dict,
    device,
) -> list[dict]:
    """
    For each (inj_layer, alpha): precompute corrected predictions for all val.
    Then for each threshold: apply gate and compute metrics.
    """
    log.info("=== Stage 5: Val sweep ===")

    unit_dir    = direction_info["unit_direction"]
    median_norm = direction_info["median_norm"]
    val_scores  = router_info["val_scores_all"]

    # Baseline metrics on val
    n_val    = len(val_recs)
    golds_val  = [r["gold_response_mode"] for r in val_recs]
    preds_base = [r["baseline_pred"] for r in val_recs]
    base_acc   = sum(p == g for p, g in zip(preds_base, golds_val)) / n_val
    base_nt_tc = sum(1 for r in val_recs if r["error_type"] == "no_tool_to_tool_call")

    log.info("Val baseline: acc=%.4f  nt_tc_errors=%d", base_acc, base_nt_tc)

    results_grid = []

    for inj_layer in INJ_LAYERS:
        for alpha in ALPHAS:
            corr_vec = (alpha * median_norm * unit_dir).astype(np.float32)

            log.info("  Precomputing corrected preds: layer=L%d alpha=%.1f ...",
                     inj_layer, alpha)
            corrected_preds = []
            for i, rec in enumerate(tqdm(val_recs,
                                         desc=f"L{inj_layer} α={alpha}",
                                         leave=False)):
                cp = predict_corrected(
                    model, tok, rec, corr_vec, inj_layer, device, INJ_MODE)
                corrected_preds.append(cp)
                if (i + 1) % 50 == 0:
                    gc.collect(); torch.cuda.empty_cache()

            # Now apply different thresholds without re-running model
            for thr in THRESHOLDS:
                int_preds = []
                n_touched = 0
                for i, rec in enumerate(val_recs):
                    fires = (val_scores[i] >= thr)
                    gate_ok = (rec["baseline_pred"] == "tool_call")
                    if fires and gate_ok:
                        int_preds.append(corrected_preds[i])
                        n_touched += 1
                    else:
                        int_preds.append(rec["baseline_pred"])

                int_acc  = sum(p == g for p, g in zip(int_preds, golds_val)) / n_val
                fixed    = sum(1 for r, ip in zip(val_recs, int_preds)
                               if r["baseline_pred"] != r["gold_response_mode"]
                               and ip == r["gold_response_mode"])
                broke    = sum(1 for r, ip in zip(val_recs, int_preds)
                               if r["baseline_pred"] == r["gold_response_mode"]
                               and ip != r["gold_response_mode"])
                net      = fixed - broke

                # nt_tc after intervention
                nt_tc_after = sum(1 for r, ip in zip(val_recs, int_preds)
                                  if r["gold_response_mode"] == "no_tool"
                                  and ip == "tool_call")
                # touched-correct damage
                touched_correct = sum(1 for r, ip in zip(val_recs, int_preds)
                                      if r["baseline_pred"] == r["gold_response_mode"]
                                      and val_scores[val_recs.index(r)] >= thr
                                      and r["baseline_pred"] == "tool_call"
                                      and ip != r["gold_response_mode"])

                results_grid.append({
                    "inj_layer": inj_layer,
                    "alpha": alpha,
                    "threshold": thr,
                    "val_acc_before": round(base_acc, 5),
                    "val_acc_after":  round(int_acc, 5),
                    "fixed": fixed,
                    "broke": broke,
                    "net":   net,
                    "n_touched": n_touched,
                    "nt_tc_before": base_nt_tc,
                    "nt_tc_after":  nt_tc_after,
                    "touched_correct_broke": touched_correct,
                })

            # Log best threshold for this (layer, alpha)
            sub = [r for r in results_grid
                   if r["inj_layer"] == inj_layer and r["alpha"] == alpha]
            best = max(sub, key=lambda x: (x["net"], -x["broke"]))
            log.info("  L%d α=%.1f  best thr=%.2f  net=%+d (fixed=%d broke=%d)"
                     "  acc %.4f→%.4f  nt_tc %d→%d  touched=%d",
                     inj_layer, alpha,
                     best["threshold"], best["net"],
                     best["fixed"], best["broke"],
                     best["val_acc_before"], best["val_acc_after"],
                     best["nt_tc_before"], best["nt_tc_after"],
                     best["n_touched"])

    return results_grid


# ─── Stage 6: Lock config and test ───────────────────────────────────────────

def stage6_locked_test(
    model, tok,
    test_recs: list[dict],
    test_acts: np.ndarray,
    router_info: dict,
    direction_info: dict,
    locked_cfg: dict,
    device,
) -> list[dict]:
    """Run locked configuration on test split. One shot, no re-tuning."""
    log.info("=== Stage 6: Locked test eval ===")
    log.info("Locked config: %s", {k: v for k, v in locked_cfg.items()
                                    if not isinstance(v, np.ndarray)})

    unit_dir    = direction_info["unit_direction"]
    median_norm = direction_info["median_norm"]
    inj_layer   = locked_cfg["inj_layer"]
    alpha       = locked_cfg["alpha"]
    threshold   = locked_cfg["threshold"]

    corr_vec = (alpha * median_norm * unit_dir).astype(np.float32)

    # Precompute router scores for test
    test_scores = router_info["clf"].predict_proba(
        router_info["scaler"].transform(test_acts))[:, 1]

    detail_records = []
    for i, rec in enumerate(tqdm(test_recs, desc="LockedTest")):
        gold  = rec["gold_response_mode"]
        pred0 = rec["baseline_pred"]
        fires = (test_scores[i] >= threshold)
        gate_ok = (pred0 == "tool_call")

        int_pred = pred0
        routed   = False
        if fires and gate_ok:
            int_pred = predict_corrected(
                model, tok, rec, corr_vec, inj_layer, device, INJ_MODE)
            routed = True

        detail_records.append({
            "sample_id":      rec["sample_id"],
            "query":          rec["query"][:120],
            "gold":           gold,
            "baseline_pred":  pred0,
            "int_pred":       int_pred,
            "correct_before": bool(pred0 == gold),
            "correct_after":  bool(int_pred == gold),
            "error_type":     rec["error_type"],
            "router_score":   round(float(test_scores[i]), 5),
            "fired":          bool(fires),
            "gated":          bool(fires and not gate_ok),
            "intervened":     bool(routed),
        })

        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    return detail_records


# ─── Stage 7: Placebo controls ───────────────────────────────────────────────

def run_placebo(
    model, tok,
    test_recs: list[dict],
    test_acts: np.ndarray,
    router_info: dict,
    direction_info: dict,
    locked_cfg: dict,
    device,
    placebo_type: str,  # "real", "random", "reverse", "ungated"
) -> dict:
    """
    Run one placebo variant and return metrics dict.
    """
    unit_dir    = direction_info["unit_direction"]
    median_norm = direction_info["median_norm"]
    inj_layer   = locked_cfg["inj_layer"]
    alpha       = locked_cfg["alpha"]
    threshold   = locked_cfg["threshold"]

    if placebo_type == "real":
        direction = unit_dir
    elif placebo_type == "random":
        rng = np.random.RandomState(42)
        rand_dir = rng.randn(len(unit_dir)).astype(np.float32)
        direction = rand_dir / (np.linalg.norm(rand_dir) + 1e-12)
    elif placebo_type == "reverse":
        direction = -unit_dir
    elif placebo_type == "ungated":
        direction = unit_dir
    else:
        raise ValueError(f"Unknown placebo_type: {placebo_type}")

    corr_vec = (alpha * median_norm * direction).astype(np.float32)
    test_scores = router_info["clf"].predict_proba(
        router_info["scaler"].transform(test_acts))[:, 1]

    golds = [r["gold_response_mode"] for r in test_recs]
    base_preds = [r["baseline_pred"] for r in test_recs]

    int_preds = []
    n_touched = 0
    for i, rec in enumerate(tqdm(test_recs, desc=f"Placebo-{placebo_type}",
                                  leave=False)):
        fires   = (test_scores[i] >= threshold)
        gate_ok = (rec["baseline_pred"] == "tool_call") if placebo_type != "ungated" else True
        if fires and gate_ok:
            ip = predict_corrected(
                model, tok, rec, corr_vec, inj_layer, device, INJ_MODE)
            int_preds.append(ip)
            n_touched += 1
        else:
            int_preds.append(rec["baseline_pred"])
        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    n = len(test_recs)
    base_acc = sum(p == g for p, g in zip(base_preds, golds)) / n
    int_acc  = sum(p == g for p, g in zip(int_preds, golds)) / n
    fixed    = sum(1 for r, ip in zip(test_recs, int_preds)
                   if r["baseline_pred"] != r["gold_response_mode"]
                   and ip == r["gold_response_mode"])
    broke    = sum(1 for r, ip in zip(test_recs, int_preds)
                   if r["baseline_pred"] == r["gold_response_mode"]
                   and ip != r["gold_response_mode"])
    nt_tc_after = sum(1 for r, ip in zip(test_recs, int_preds)
                      if r["gold_response_mode"] == "no_tool"
                      and ip == "tool_call")
    nt_tc_before = sum(1 for r in test_recs
                       if r["error_type"] == "no_tool_to_tool_call")

    log.info("Placebo [%s]: acc %.4f→%.4f  fixed=%d  broke=%d  net=%+d"
             "  nt_tc %d→%d  touched=%d",
             placebo_type, base_acc, int_acc, fixed, broke, fixed - broke,
             nt_tc_before, nt_tc_after, n_touched)

    return {
        "type": placebo_type,
        "baseline_acc": round(base_acc, 5),
        "int_acc":      round(int_acc, 5),
        "delta_acc":    round(int_acc - base_acc, 5),
        "fixed":        fixed,
        "broke":        broke,
        "net":          fixed - broke,
        "nt_tc_before": nt_tc_before,
        "nt_tc_after":  nt_tc_after,
        "n_touched":    n_touched,
    }


def stage7_placebo_controls(
    model, tok,
    test_recs, test_acts,
    router_info, direction_info, locked_cfg,
    device,
) -> list[dict]:
    log.info("=== Stage 7: Placebo controls ===")
    placebo_results = []
    for ptype in ["real", "random", "reverse", "ungated"]:
        pr = run_placebo(
            model, tok, test_recs, test_acts,
            router_info, direction_info, locked_cfg, device, ptype)
        placebo_results.append(pr)
    return placebo_results


# ─── Metrics helper ──────────────────────────────────────────────────────────

def compute_test_metrics(test_recs: list[dict], detail_records: list[dict]) -> dict:
    n = len(detail_records)
    golds      = [d["gold"]          for d in detail_records]
    base_preds = [d["baseline_pred"] for d in detail_records]
    int_preds  = [d["int_pred"]      for d in detail_records]

    base_acc = sum(p == g for p, g in zip(base_preds, golds)) / n
    int_acc  = sum(p == g for p, g in zip(int_preds, golds)) / n
    fixed    = sum(1 for d in detail_records
                   if d["baseline_pred"] != d["gold"] and d["int_pred"] == d["gold"])
    broke    = sum(1 for d in detail_records
                   if d["baseline_pred"] == d["gold"] and d["int_pred"] != d["gold"])
    net      = fixed - broke

    nt_tc_before = sum(1 for d in detail_records if d["error_type"] == "no_tool_to_tool_call")
    nt_tc_after  = sum(1 for d in detail_records
                       if d["gold"] == "no_tool" and d["int_pred"] == "tool_call")
    n_gold_nt    = sum(1 for d in detail_records if d["gold"] == "no_tool")

    n_touched    = sum(1 for d in detail_records if d["intervened"])
    n_fired      = sum(1 for d in detail_records if d["fired"])

    # Touched-correct damage: intervened on sample that was baseline-correct
    touched_and_broke = sum(1 for d in detail_records
                            if d["intervened"] and d["correct_before"]
                            and not d["correct_after"])
    n_intervened_correct_before = sum(1 for d in detail_records
                                      if d["intervened"] and d["correct_before"])

    return {
        "n_test":              n,
        "baseline_acc":        round(base_acc, 5),
        "int_acc":             round(int_acc, 5),
        "delta_acc":           round(int_acc - base_acc, 5),
        "fixed":               fixed,
        "broke":               broke,
        "net":                 net,
        "nt_tc_before":        nt_tc_before,
        "nt_tc_after":         nt_tc_after,
        "false_toolcall_rate_before": round(nt_tc_before / n_gold_nt, 5) if n_gold_nt else 0,
        "false_toolcall_rate_after":  round(nt_tc_after  / n_gold_nt, 5) if n_gold_nt else 0,
        "n_touched":           n_touched,
        "router_fire_rate":    round(n_fired / n, 5),
        "touched_correct_broke": touched_and_broke,
        "touched_correct_n":   n_intervened_correct_before,
        "touched_correct_damage_rate": round(touched_and_broke / n_intervened_correct_before, 5)
                                       if n_intervened_correct_before else 0.0,
    }


# ─── Stage 8: Write report ────────────────────────────────────────────────────

def stage8_write_report(
    split_stats: dict,
    direction_info: dict,
    router_info: dict,
    sweep_grid: list[dict],
    locked_cfg: dict,
    test_metrics: dict,
    placebo_results: list[dict],
    detail_records: list[dict],
):
    log.info("=== Stage 8: Write report ===")

    # Save per-sample test details
    details_out = OUT_DIR / "metatool_binary_sakiko_pilot_details.jsonl"
    with open(details_out, "w", encoding="utf-8") as f:
        for d in detail_records:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log.info("Details → %s (%d records)", details_out, len(detail_records))

    # Save sweep grid
    sweep_out = OUT_DIR / "metatool_binary_sakiko_val_sweep.json"
    with open(sweep_out, "w", encoding="utf-8") as f:
        json.dump(sweep_grid, f, indent=2, ensure_ascii=False)
    log.info("Val sweep → %s (%d configs)", sweep_out, len(sweep_grid))

    # Save JSON summary
    summary_json = OUT_DIR / "metatool_binary_sakiko_pilot_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "dataset":          "MetaTool-Binary",
            "model":            "microsoft/Phi-3.5-mini-instruct",
            "quantization":     "4-bit NF4 float16",
            "split_stats":      split_stats,
            "direction": {
                "method":      "DiffMean",
                "n_err":       direction_info["n_err"],
                "n_ref":       direction_info["n_ref"],
                "direction_norm": round(direction_info["direction_norm"], 5),
                "median_norm":    round(direction_info["median_norm"], 5),
            },
            "router": {
                "method":    "LogisticRegression (liblinear, C=1.0)",
                "obs_layer": OBS_LAYER,
                "n_pos":     router_info["n_train_pos"],
                "n_neg":     router_info["n_train_neg"],
                "val_auc":   router_info["val_auc"],
            },
            "locked_cfg": {k: v for k, v in locked_cfg.items()
                           if not isinstance(v, np.ndarray)},
            "test_metrics":  test_metrics,
            "placebo":       placebo_results,
        }, f, indent=2, ensure_ascii=False)
    log.info("JSON summary → %s", summary_json)

    # Markdown report
    md_out = OUT_DIR / "metatool_binary_sakiko_pilot_summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        _write_md(f, split_stats, direction_info, router_info,
                  locked_cfg, test_metrics, placebo_results)
    log.info("Markdown report → %s", md_out)


def _write_md(f, split_stats, direction_info, router_info,
              locked_cfg, test_metrics, placebo_results):
    tm = test_metrics

    # ── Interpretation helper ──
    net = tm["net"]
    fixed = tm["fixed"]
    broke = tm["broke"]
    nt_tc_reduced = tm["nt_tc_before"] - tm["nt_tc_after"]

    if net > 0 and broke <= fixed // 2:
        interpret = ("**POSITIVE**: Net improvement with acceptable collateral damage. "
                     "Supports cross-dataset generalization of SAKIKO over-calling correction.")
    elif net > 0:
        interpret = ("**MARGINAL**: Net positive but Broke is high relative to Fixed. "
                     "Direction works but router precision needs improvement.")
    elif net == 0:
        interpret = ("**NEUTRAL**: Fixed == Broke. "
                     "Direction may be correct but router is too noisy.")
    else:
        interpret = ("**NEGATIVE**: Broke > Fixed. "
                     "Diagnose: check router OOD, direction weakness, or insufficient channel structure.")

    # Placebo interpretation
    real_r   = next((p for p in placebo_results if p["type"] == "real"),   {})
    rand_r   = next((p for p in placebo_results if p["type"] == "random"), {})
    rev_r    = next((p for p in placebo_results if p["type"] == "reverse"),{})
    ung_r    = next((p for p in placebo_results if p["type"] == "ungated"),{})

    real_net = real_r.get("net", 0)
    rand_net = rand_r.get("net", 0)
    rev_net  = rev_r.get("net", 0)

    if real_net > 0 and rand_net <= 0 and rev_net <= 0:
        causal = "**CAUSAL**: Real direction is uniquely beneficial. Random/Reverse do not help."
    elif real_net > 0 and rand_net > 0:
        causal = "**MIXED**: Both real and random improve — direction specificity uncertain."
    elif real_net <= 0:
        causal = "**NON-CAUSAL**: Real direction does not help. Direction may be wrong."
    else:
        causal = "**PARTIAL**: Further ablation needed."

    f.write("# MetaTool-Binary SAKIKO Pilot — Summary Report\n\n")
    f.write("> **IMPORTANT CAVEATS**\n")
    f.write("> - This is MetaTool-Binary, NOT W2C-4way.\n")
    f.write("> - Binary labels only: `tool_call` / `no_tool`.\n")
    f.write("> - Tests external binary over-calling correction only (nt_tc channel).\n")
    f.write("> - Does NOT test request_for_info / cannot_answer / direct separation.\n")
    f.write("> - Results are NOT directly comparable to W2C accuracy.\n")
    f.write("> - 4-bit NF4 quantization used throughout (same as baseline evaluator).\n\n")

    f.write("## 1. Dataset & Baseline\n\n")
    f.write("| Split | N | Accuracy | nt_tc errors | tc_nt errors |\n|---|---|---|---|---|\n")
    for sname in ["train", "val", "test"]:
        s = split_stats[sname]
        f.write(f"| {sname} | {s['n']} | {s['accuracy']:.4f} | {s['nt_tc']} | {s['tc_nt']} |\n")

    f.write("\n## 2. Direction (DiffMean)\n\n")
    f.write("```\n")
    f.write(f"Channel:          nt_tc (gold=no_tool, pred=tool_call)\n")
    f.write(f"Direction:        mean(correct_no_tool) - mean(nt_tc_errors)  [train]\n")
    f.write(f"n_err (nt_tc):    {direction_info['n_err']}\n")
    f.write(f"n_ref (correct):  {direction_info['n_ref']}\n")
    f.write(f"||direction||:    {direction_info['direction_norm']:.4f}\n")
    f.write(f"median_norm:      {direction_info['median_norm']:.4f}\n")
    f.write("```\n\n")

    f.write("## 3. Router\n\n")
    f.write("```\n")
    f.write(f"Model:      LogisticRegression (liblinear, C=1.0, StandardScaler)\n")
    f.write(f"Obs layer:  L{OBS_LAYER} (MLP output, last prompt position)\n")
    f.write(f"Positive:   nt_tc errors in train  (n={router_info['n_train_pos']})\n")
    f.write(f"Negative:   all correct in train   (n={router_info['n_train_neg']})\n")
    f.write(f"Val AUC:    {router_info['val_auc']:.5f}\n")
    f.write("```\n\n")

    f.write("## 4. Locked Configuration\n\n")
    f.write("| Parameter | Value |\n|---|---|\n")
    f.write(f"| obs_layer | L{OBS_LAYER} |\n")
    f.write(f"| inj_layer | L{locked_cfg['inj_layer']} |\n")
    f.write(f"| inj_mode  | {INJ_MODE} |\n")
    f.write(f"| alpha     | {locked_cfg['alpha']} |\n")
    f.write(f"| threshold | {locked_cfg['threshold']} |\n")
    f.write(f"| direction | DiffMean (nt_tc) |\n\n")
    f.write("*Configuration locked on val only. Test was evaluated once.*\n\n")

    f.write("## 5. Locked Test Results\n\n")
    f.write("| Metric | Value |\n|---|---|\n")
    f.write(f"| N test | {tm['n_test']} |\n")
    f.write(f"| Baseline accuracy | {tm['baseline_acc']:.4f} |\n")
    f.write(f"| SAKIKO accuracy   | {tm['int_acc']:.4f} |\n")
    f.write(f"| Δ accuracy        | {tm['delta_acc']:+.4f} |\n")
    f.write(f"| Fixed             | {tm['fixed']} |\n")
    f.write(f"| Broke             | {tm['broke']} |\n")
    f.write(f"| Net               | **{tm['net']:+d}** |\n")
    f.write(f"| nt_tc before      | {tm['nt_tc_before']} |\n")
    f.write(f"| nt_tc after       | {tm['nt_tc_after']} |\n")
    f.write(f"| False tool-call rate (gold=no_tool) before | {tm['false_toolcall_rate_before']:.4f} |\n")
    f.write(f"| False tool-call rate (gold=no_tool) after  | {tm['false_toolcall_rate_after']:.4f} |\n")
    f.write(f"| Samples touched   | {tm['n_touched']} |\n")
    f.write(f"| Router fire rate  | {tm['router_fire_rate']:.4f} |\n")
    f.write(f"| Touched-correct damage rate | {tm['touched_correct_damage_rate']:.4f} |\n\n")

    f.write("## 6. Placebo Controls\n\n")
    f.write("| Placebo | Acc Before | Acc After | Fixed | Broke | Net | nt_tc before→after |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for pr in placebo_results:
        f.write(f"| {pr['type']} | {pr['baseline_acc']:.4f} | {pr['int_acc']:.4f} | "
                f"{pr['fixed']} | {pr['broke']} | **{pr['net']:+d}** | "
                f"{pr['nt_tc_before']}→{pr['nt_tc_after']} |\n")

    f.write("\n## 7. Interpretation\n\n")
    f.write(f"**Overall pilot verdict**: {interpret}\n\n")
    f.write(f"**Causal check (placebo)**: {causal}\n\n")

    if nt_tc_reduced > 0:
        f.write(f"The nt_tc (over-calling) error count was reduced from "
                f"{tm['nt_tc_before']} to {tm['nt_tc_after']} "
                f"({nt_tc_reduced} fewer false tool-calls) on the test split.\n\n")
    else:
        f.write(f"The nt_tc error count did not decrease "
                f"({tm['nt_tc_before']}→{tm['nt_tc_after']}).\n\n")

    f.write("### Diagnosis notes\n\n")
    f.write("- If router OOD: val_auc will be near 0.5 and fire rate will not track nt_tc errors.\n")
    f.write("- If direction weak: Real direction will show Net ≈ 0, similar to random/reverse.\n")
    f.write("- If insufficient channel structure: Fixed will be low even at high alpha.\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MetaTool-Binary SAKIKO pilot")
    parser.add_argument("--model_key", default="phi35")
    parser.add_argument("--obs_layer", type=int, default=OBS_LAYER)
    parser.add_argument("--force_acts", action="store_true",
                        help="Re-extract activations even if cached")
    args = parser.parse_args()

    log.info("MetaTool-Binary SAKIKO Pilot — model_key=%s, obs_layer=L%d",
             args.model_key, args.obs_layer)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    train_recs, val_recs, test_recs, _, split_stats = stage1_split_baseline()

    # ── Load model (needed for stages 2, 5, 6, 7) ───────────────────────
    mcfg, defaults = load_model_cfg(args.model_key)
    model_path = mcfg.get("local_path", mcfg["model_id"])
    tok = load_tokenizer(model_path, mcfg, defaults)
    model, device = load_model(model_path, mcfg, defaults)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    train_acts, val_acts, test_acts = stage2_extract_activations(
        model, tok, train_recs, val_recs, test_recs,
        args.obs_layer, device)

    # ── Stage 3 ───────────────────────────────────────────────────────────
    direction_info = stage3_difmean_direction(train_recs, train_acts)

    # ── Stage 4 ───────────────────────────────────────────────────────────
    router_info = stage4_train_router(
        train_recs, train_acts, val_recs, val_acts)

    # ── Stage 5: Val sweep ────────────────────────────────────────────────
    sweep_grid = sweep_val(
        model, tok, val_recs, val_acts, router_info,
        direction_info, device)

    # Lock best configuration from val sweep
    # Criterion: max net, then min broke, then max acc_after
    best_val = max(
        sweep_grid,
        key=lambda x: (x["net"], -x["broke"], x["val_acc_after"]))
    locked_cfg = {
        "inj_layer": best_val["inj_layer"],
        "alpha":     best_val["alpha"],
        "threshold": best_val["threshold"],
        "inj_mode":  INJ_MODE,
        "val_fixed": best_val["fixed"],
        "val_broke": best_val["broke"],
        "val_net":   best_val["net"],
        "val_acc_before": best_val["val_acc_before"],
        "val_acc_after":  best_val["val_acc_after"],
    }
    log.info("=== Stage 6: Locked config from val ===")
    log.info("  inj_layer=%d  alpha=%.1f  threshold=%.2f",
             locked_cfg["inj_layer"], locked_cfg["alpha"], locked_cfg["threshold"])
    log.info("  Val: fixed=%d  broke=%d  net=%+d  acc %.4f→%.4f",
             locked_cfg["val_fixed"], locked_cfg["val_broke"], locked_cfg["val_net"],
             locked_cfg["val_acc_before"], locked_cfg["val_acc_after"])

    # ── Stage 6: Locked test ──────────────────────────────────────────────
    detail_records = stage6_locked_test(
        model, tok, test_recs, test_acts, router_info,
        direction_info, locked_cfg, device)

    test_metrics = compute_test_metrics(test_recs, detail_records)
    log.info("=== Locked Test Results ===")
    log.info("  acc:   %.4f → %.4f  (Δ%+.4f)",
             test_metrics["baseline_acc"],
             test_metrics["int_acc"],
             test_metrics["delta_acc"])
    log.info("  fixed=%d  broke=%d  net=%+d",
             test_metrics["fixed"], test_metrics["broke"], test_metrics["net"])
    log.info("  nt_tc: %d → %d",
             test_metrics["nt_tc_before"], test_metrics["nt_tc_after"])

    # ── Stage 7: Placebo ──────────────────────────────────────────────────
    placebo_results = stage7_placebo_controls(
        model, tok, test_recs, test_acts,
        router_info, direction_info, locked_cfg, device)

    # ── Stage 8: Write report ─────────────────────────────────────────────
    stage8_write_report(
        split_stats, direction_info, router_info,
        sweep_grid, locked_cfg, test_metrics,
        placebo_results, detail_records)

    log.info("=== MetaTool-Binary SAKIKO pilot complete ===")


if __name__ == "__main__":
    main()
