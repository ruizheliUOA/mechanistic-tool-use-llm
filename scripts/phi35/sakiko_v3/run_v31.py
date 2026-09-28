#!/usr/bin/env python3
"""
SAKIKO V3.1 — Cascade Binary Router + CA→direct Correction
============================================================

V3 → V3.1 的核心改动:
  1. 级联二分类 Router: 3 个独立 LR (is_rfi_tc, is_ca_tc, is_ca_direct)
     替代 V3 的单个 3-class LR, 各 router 独立阈值, 单发触发 (max prob wins)
  2. 新增 CA→direct 纠正通道 (PCA 基底以正确 cannot_answer 样本为参考)
  3. 评估循环扩展为 3 通道级联, 逐样本记录触发通道

所有设计点均为 "初始工程设定, 不代表已证明最优", 后续可做消融。
不修改 run_v3.py — 保持 V3 可完整复现。

使用方法:
    cd /path/to/project
    source .venv/bin/activate
    python sakiko_v3/run_v31.py [--config CONFIG] [--output PATH]

可用 config:
    v31_cascade     — 3 通道级联 (RFI→TC + CA→TC + CA→direct)
    v31_ablation_2ch — 级联 router 但不加 CA→direct 通道 (对照用)
"""

import gc, json, logging, sys, time, argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════
# Constants & Paths
# ═══════════════════════════════════════════════════════════════════════════

ROOT       = Path(__file__).resolve().parents[1]
CACHE_DIR  = ROOT / "sakiko_v3" / "cache"
RESULT_DIR = ROOT / "sakiko_v3" / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(ROOT / ".cache/modelscope/LLM-Research/Phi-3___5-mini-instruct")
DEVICE     = "cuda:0"
SEED       = 42

LABEL_VOCAB = ["tool_call", "direct", "request_for_info", "cannot_answer"]
RFI2TC     = "request_for_info\u2192tool_call"
CA2TC      = "cannot_answer\u2192tool_call"
CA2DIRECT  = "cannot_answer\u2192direct"

# Target Direction Consistency Gate
TARGET_GATE = {
    "rfi_tc":     {"request_for_info", "cannot_answer"},
    "ca_tc":      {"cannot_answer"},
    "ca_direct":  {"cannot_answer"},
}

# ═══════════════════════════════════════════════════════════════════════════
# Configurations
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "v31_cascade": {
        "obs_layer": 18,
        "rfi_inj_layer": 14,
        "rfi_inj_mode": "mlp_all",
        "rfi_alpha": 5.0,
        "threshold_rfi": 0.60,
        "ca_tc_inj_layer": 16,
        "ca_tc_inj_mode": "mlp_prompt",
        "ca_tc_alpha": 2.0,
        "threshold_ca_tc": 0.60,
        "ca_direct_inj_layer": 16,
        "ca_direct_inj_mode": "mlp_prompt",
        "ca_direct_alpha": 2.0,
        "threshold_ca_direct": 0.60,
        "n_comp": 64,
        "n_pcs": 20,
    },
    # -- V3.1 优化: CA→direct α=8.0 (alpha sweep 最优) --
    "v31_cascade_opt": {
        "obs_layer": 18,
        "rfi_inj_layer": 14,
        "rfi_inj_mode": "mlp_all",
        "rfi_alpha": 5.0,
        "threshold_rfi": 0.60,
        "ca_tc_inj_layer": 16,
        "ca_tc_inj_mode": "mlp_prompt",
        "ca_tc_alpha": 2.0,
        "threshold_ca_tc": 0.60,
        "ca_direct_inj_layer": 16,
        "ca_direct_inj_mode": "mlp_prompt",
        "ca_direct_alpha": 8.0,
        "threshold_ca_direct": 0.60,
        "n_comp": 64,
        "n_pcs": 20,
    },
    "v31_ablation_2ch": {
        "obs_layer": 18,
        "rfi_inj_layer": 14,
        "rfi_inj_mode": "mlp_all",
        "rfi_alpha": 5.0,
        "threshold_rfi": 0.60,
        "ca_tc_inj_layer": 16,
        "ca_tc_inj_mode": "mlp_prompt",
        "ca_tc_alpha": 2.0,
        "threshold_ca_tc": 0.60,
        "ca_direct_inj_layer": None,
        "ca_direct_inj_mode": None,
        "ca_direct_alpha": 0,
        "threshold_ca_direct": 1.0,
        "n_comp": 64,
        "n_pcs": 20,
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sakiko_v31")


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    path = ROOT / "trace_db" / "w2c_phi35.jsonl"
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    log.info("Loaded %d samples from %s", len(rows), path)
    return rows


def build_meta(rows):
    meta = []
    for i, row in enumerate(rows):
        gold, pred = row["gold"], row["pred"]
        correct = (pred == gold)
        error_type = "correct" if correct else f"{gold}\u2192{pred}"
        meta.append({
            "idx": i, "gold": gold, "pred": pred,
            "correct": correct, "error_type": error_type,
        })
    return meta


def load_model():
    log.info("Loading model: %s", MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE)
    model.eval()
    log.info("Model loaded. Layers: %d, Hidden: %d",
             len(model.model.layers), model.config.hidden_size)
    return model, tok


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Activation Collection (复用 V3 缓存)
# ═══════════════════════════════════════════════════════════════════════════

def collect_activations(model, tok, rows, layer_idx, force=False):
    cache_path = CACHE_DIR / f"acts_L{layer_idx}.npy"
    if not force and cache_path.exists():
        acts = np.load(cache_path)
        log.info("Loaded cached L%d activations: %s", layer_idx, acts.shape)
        return acts

    log.info("Collecting L%d activations for %d samples...", layer_idx, len(rows))
    D = model.config.hidden_size
    n = len(rows)
    acts = np.zeros((n, D), dtype=np.float32)
    mlp = model.model.layers[layer_idx].mlp

    for i, row in enumerate(tqdm(rows, desc=f"Collect-L{layer_idx}")):
        inputs = tok(row["prompt"], return_tensors="pt",
                     truncation=True, max_length=8192)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        last_pos = inputs["input_ids"].shape[1] - 1

        captured = {}
        def hook_fn(module, inp, out, _c=captured, _p=last_pos):
            _c["act"] = out[0, _p, :].detach().cpu().float().numpy()
        h = mlp.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        h.remove()
        acts[i] = captured["act"]
        del inputs
        if (i + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()

    np.save(cache_path, acts)
    log.info("Saved L%d activations: %s -> %s", layer_idx, acts.shape, cache_path)
    return acts


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Train / Test Split
# ═══════════════════════════════════════════════════════════════════════════

def split_data(meta, seed=SEED):
    raw_labels = [m["error_type"] for m in meta]
    counts = Counter(raw_labels)
    strat_labels = [l if counts[l] >= 5 else "_rare" for l in raw_labels]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, test_idx = next(sss.split(range(len(meta)), strat_labels))
    log.info("Split: train=%d, test=%d", len(train_idx), len(test_idx))
    return list(train_idx.astype(int)), list(test_idx.astype(int))


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: PCA Basis (参数化 ref_gold — V3.1 关键改动)
# ═══════════════════════════════════════════════════════════════════════════

def train_pca_basis(acts, meta, train_idx, error_type, n_comp=64, ref_gold=None):
    """
    ref_gold: 参考样本的 gold 标签.
      - RFI→TC / CA→TC: ref_gold="tool_call"
      - CA→direct:     ref_gold="cannot_answer"
      若不指定, 从 error_type 自动推断 (gold 部分).
    """
    if ref_gold is None:
        ref_gold = error_type.split("\u2192")[0]

    err_idx = [i for i in train_idx if meta[i]["error_type"] == error_type]
    ref_idx = [i for i in train_idx
               if meta[i]["correct"] and meta[i]["gold"] == ref_gold]

    if len(err_idx) < 10 or len(ref_idx) < 5:
        log.warning("Insufficient data for %s: err=%d ref=%d",
                     error_type, len(err_idx), len(ref_idx))
        return None

    combined = np.concatenate([acts[err_idx], acts[ref_idx]])
    scaler = StandardScaler()
    combined_s = scaler.fit_transform(combined)
    n_comp = min(n_comp, combined_s.shape[0] - 1, combined_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(combined_s)

    err_pcs = pca.transform(scaler.transform(acts[err_idx]))
    ref_pcs = pca.transform(scaler.transform(acts[ref_idx]))

    basis = {
        "components": pca.components_.astype(np.float32),
        "mean_": scaler.mean_.astype(np.float32),
        "scale_": scaler.scale_.astype(np.float32),
        "err_mean_pc": err_pcs.mean(axis=0).astype(np.float32),
        "ref_mean_pc": ref_pcs.mean(axis=0).astype(np.float32),
        "n_err": len(err_idx), "n_ref": len(ref_idx),
    }
    log.info("PCA basis [%s]: err=%d, ref=%d (ref_gold=%s), K=%d, var=%.3f",
             error_type, len(err_idx), len(ref_idx), ref_gold, n_comp,
             pca.explained_variance_ratio_.sum())
    return basis


def compute_correction_vec(basis, n_pcs=20, alpha=3.0):
    K = min(n_pcs, basis["components"].shape[0])
    dirs = basis["components"][:K] / basis["scale_"][np.newaxis, :]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    unit_dirs = dirs / (norms + 1e-12)
    delta_pc = basis["ref_mean_pc"][:K] - basis["err_mean_pc"][:K]
    delta_orig = delta_pc * norms.flatten()

    correction = np.zeros(basis["components"].shape[1], dtype=np.float32)
    for i in range(K):
        correction += alpha * delta_orig[i] * unit_dirs[i]
    return correction


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Cascade Binary Router (V3.1 核心)
# ═══════════════════════════════════════════════════════════════════════════

def train_binary_router(acts, meta, train_idx, target_error_type, name=""):
    idx_pos = [i for i in train_idx if meta[i]["error_type"] == target_error_type]
    idx_neg = [i for i in train_idx if meta[i]["correct"]]

    if len(idx_pos) < 10:
        log.warning("Router [%s]: insufficient positive (%d)", name, len(idx_pos))
        return None

    all_idx = idx_neg + idx_pos
    X = acts[all_idx]
    y = np.array([0]*len(idx_neg) + [1]*len(idx_pos))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)

    n_min = min(len(idx_neg), len(idx_pos))
    cv_k = min(5, n_min)
    if cv_k >= 2:
        cv = cross_val_score(clf, X_s, y, cv=cv_k, scoring="accuracy")
    else:
        cv = np.array([0.0])

    clf.fit(X_s, y)
    log.info("Router [%s]: neg=%d, pos=%d, cv=%.4f\u00b1%.4f",
             name, len(idx_neg), len(idx_pos), cv.mean(), cv.std())
    return {
        "scaler": scaler, "clf": clf,
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        "n_pos": len(idx_pos), "n_neg": len(idx_neg),
        "target": target_error_type,
    }


def cascade_route(routers, acts_row, cfg):
    """
    级联决策: 各 router 独立打分, 取超阈值且 prob 最大者触发.
    单发触发 (最多一个通道).
    """
    candidates = []
    for ch_name, router in routers.items():
        if router is None:
            continue
        threshold_key = {
            "rfi_tc":     "threshold_rfi",
            "ca_tc":      "threshold_ca_tc",
            "ca_direct":  "threshold_ca_direct",
        }[ch_name]
        threshold = cfg[threshold_key]
        X_s = router["scaler"].transform(acts_row.reshape(1, -1))
        prob = router["clf"].predict_proba(X_s)[0, 1]
        if prob >= threshold:
            candidates.append((ch_name, prob))

    if not candidates:
        return None, 0.0

    priority = {"rfi_tc": 0, "ca_tc": 1, "ca_direct": 2}
    candidates.sort(key=lambda x: (-x[1], priority.get(x[0], 99)))
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Inference with Correction Hook
# ═══════════════════════════════════════════════════════════════════════════

def score_candidate_hooked(model, tok, prompt_text, candidate_text,
                           correction_vec, layer_idx, mode="mlp_all"):
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    cand_ids = tok.encode(candidate_text, add_special_tokens=False)
    if len(cand_ids) == 0:
        return -1e9

    input_ids = torch.tensor([prompt_ids + cand_ids], device=DEVICE)
    prompt_len = len(prompt_ids)
    corr_t = torch.from_numpy(correction_vec).to(torch.float32).to(DEVICE)

    def hook_fn(module, inp, output):
        if mode == "mlp_all":
            output[:, :, :] += corr_t.to(output.dtype)
        elif mode == "mlp_prompt":
            output[:, :prompt_len, :] += corr_t.to(output.dtype)
        elif mode == "mlp_last":
            output[0, prompt_len - 1, :] += corr_t.to(output.dtype)
        return output

    mlp = model.model.layers[layer_idx].mlp
    h = mlp.register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(input_ids)
    h.remove()

    logits = out.logits
    cand_len = len(cand_ids)
    logits_slice = logits[0, prompt_len - 1: prompt_len + cand_len - 1, :]
    log_probs = torch.log_softmax(logits_slice.float(), dim=-1)
    token_logps = [log_probs[i, cand_ids[i]].item() for i in range(cand_len)]
    del out, input_ids
    return sum(token_logps) / cand_len


def predict_with_correction(model, tok, row, correction_vec, layer_idx, mode):
    scores = {}
    for label in LABEL_VOCAB:
        cand = row["answers"].get(label, "")
        if not cand:
            scores[label] = -1e9
        else:
            scores[label] = score_candidate_hooked(
                model, tok, row["prompt"], cand, correction_vec, layer_idx, mode)
    pred = max(LABEL_VOCAB, key=lambda l: scores[l])
    return pred, scores


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Evaluation with Cascade Router
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, tok, rows, meta, test_idx, acts, routers, cfg,
             corrections):
    results = []
    route_counts = Counter()

    for idx in tqdm(test_idx, desc="Evaluating"):
        row = rows[idx]
        m = meta[idx]
        gold = m["gold"]
        clean_pred = m["pred"]

        ch_name, ch_prob = cascade_route(routers, acts[idx], cfg)

        int_pred = clean_pred
        route = "none"

        if ch_name is not None and ch_name in corrections:
            gate_blocked = clean_pred in TARGET_GATE.get(ch_name, set())
            if not gate_blocked:
                corr_info = corrections[ch_name]
                int_pred, _ = predict_with_correction(
                    model, tok, row, corr_info["vec"],
                    corr_info["layer"], corr_info["mode"])
                route = ch_name
                route_counts[ch_name] += 1
            else:
                route = ch_name + "_gated"

        results.append({
            "idx": int(idx), "gold": gold,
            "clean_pred": clean_pred, "int_pred": int_pred,
            "route": route, "route_prob": round(ch_prob, 4),
            "error_type": m["error_type"],
        })

        if len(results) % 100 == 0:
            gc.collect(); torch.cuda.empty_cache()

    log.info("Route counts: %s (total routed: %d / %d)",
             dict(route_counts), sum(route_counts.values()), len(test_idx))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Metrics & Report
# ═══════════════════════════════════════════════════════════════════════════

ALL_ERROR_CHANNELS = [RFI2TC, CA2TC, CA2DIRECT]

def compute_metrics(results, cfg):
    n = len(results)
    clean_preds = [r["clean_pred"] for r in results]
    int_preds = [r["int_pred"] for r in results]
    golds = [r["gold"] for r in results]

    clean_acc = 100 * sum(c == g for c, g in zip(clean_preds, golds)) / n
    int_acc = 100 * sum(i == g for i, g in zip(int_preds, golds)) / n
    clean_f1 = 100 * f1_score(golds, clean_preds, average="macro", zero_division=0)
    int_f1 = 100 * f1_score(golds, int_preds, average="macro", zero_division=0)

    breakdown = {}
    for etype in ALL_ERROR_CHANNELS:
        sub = [r for r in results if r["error_type"] == etype]
        if not sub:
            continue
        corrected = sum(1 for r in sub
                        if r["clean_pred"] != r["gold"] and r["int_pred"] == r["gold"])
        worsened = sum(1 for r in sub
                       if r["clean_pred"] == r["gold"] and r["int_pred"] != r["gold"])
        routed = sum(1 for r in sub if r["route"] != "none")
        breakdown[etype] = {
            "n": len(sub), "corrected": corrected, "worsened": worsened,
            "routed": routed,
            "correction_rate": round(100 * corrected / len(sub), 1) if sub else 0,
        }

    clean_sub = [r for r in results if r["error_type"] == "correct"]
    n_clean = len(clean_sub)
    damaged = sum(1 for r in clean_sub if r["int_pred"] != r["clean_pred"])

    total_fixed = sum(1 for r in results
                      if r["clean_pred"] != r["gold"] and r["int_pred"] == r["gold"])
    total_broke = sum(1 for r in results
                      if r["clean_pred"] == r["gold"] and r["int_pred"] != r["gold"])
    net = total_fixed - total_broke

    channel_stats = {}
    for ch in ["rfi_tc", "ca_tc", "ca_direct"]:
        ch_results = [r for r in results if r["route"] == ch]
        n_ch = len(ch_results)
        if n_ch == 0:
            continue
        ch_fixed = sum(1 for r in ch_results
                       if r["clean_pred"] != r["gold"] and r["int_pred"] == r["gold"])
        ch_broke = sum(1 for r in ch_results
                       if r["clean_pred"] == r["gold"] and r["int_pred"] != r["gold"])
        channel_stats[ch] = {
            "n_routed": n_ch,
            "fixed": ch_fixed, "broke": ch_broke,
            "net": ch_fixed - ch_broke,
            "precision": round(100 * ch_fixed / n_ch, 1) if n_ch else 0,
        }

    metrics = {
        "n_test": n,
        "baseline_accuracy": round(clean_acc, 2),
        "baseline_f1": round(clean_f1, 2),
        "intervened_accuracy": round(int_acc, 2),
        "intervened_f1": round(int_f1, 2),
        "delta_accuracy": round(int_acc - clean_acc, 2),
        "delta_f1": round(int_f1 - clean_f1, 2),
        "breakdown": breakdown,
        "clean_damage": {"n": n_clean, "damaged": damaged,
                         "rate": round(100 * damaged / n_clean, 1) if n_clean else 0},
        "total_corrected": total_fixed,
        "total_broken": total_broke,
        "net_benefit": net,
        "channel_stats": channel_stats,
        "config": {k: v for k, v in cfg.items() if not isinstance(v, (np.ndarray,))},
    }
    return metrics


def print_report(metrics, cfg, cfg_name):
    print("\n" + "=" * 70)
    print(f"  SAKIKO V3.1 \u2014 Evaluation Report [{cfg_name}]")
    print("=" * 70)

    print(f"\n  Config:")
    if cfg.get("rfi_inj_layer"):
        print(f"    RFI\u2192TC:     L{cfg['rfi_inj_layer']} {cfg['rfi_inj_mode']} \u03b1={cfg['rfi_alpha']} thr={cfg['threshold_rfi']}")
    if cfg.get("ca_tc_inj_layer"):
        print(f"    CA\u2192TC:      L{cfg['ca_tc_inj_layer']} {cfg['ca_tc_inj_mode']} \u03b1={cfg['ca_tc_alpha']} thr={cfg['threshold_ca_tc']}")
    if cfg.get("ca_direct_inj_layer"):
        print(f"    CA\u2192direct:  L{cfg['ca_direct_inj_layer']} {cfg['ca_direct_inj_mode']} \u03b1={cfg['ca_direct_alpha']} thr={cfg['threshold_ca_direct']}")
    print(f"    Observation: L{cfg['obs_layer']}")

    m = metrics
    print(f"\n  Test samples: {m['n_test']}")
    print(f"\n  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    print(f"  \u2502  Metric     \u2502 Baseline \u2502 V3.1     \u2502 Delta    \u2502")
    print(f"  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    print(f"  \u2502  Accuracy   \u2502 {m['baseline_accuracy']:>6.2f}%  \u2502 {m['intervened_accuracy']:>6.2f}%  \u2502 {m['delta_accuracy']:>+6.2f}%  \u2502")
    print(f"  \u2502  Macro-F1   \u2502 {m['baseline_f1']:>6.2f}%  \u2502 {m['intervened_f1']:>6.2f}%  \u2502 {m['delta_f1']:>+6.2f}%  \u2502")
    print(f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")

    for etype, info in m.get("breakdown", {}).items():
        print(f"\n  {etype}:")
        print(f"    n={info['n']}, routed={info['routed']}, corrected={info['corrected']} ({info['correction_rate']}%), worsened={info['worsened']}")

    print("\n  Per-channel stats:")
    for ch, cs in m.get("channel_stats", {}).items():
        print(f"    {ch}: routed={cs['n_routed']}, fixed={cs['fixed']}, broke={cs['broke']}, net={cs['net']}, precision={cs['precision']}%")

    cd = m["clean_damage"]
    print(f"\n  Clean damage: {cd['damaged']}/{cd['n']} ({cd['rate']}%)")
    print(f"  Total: fixed={m['total_corrected']}, broke={m['total_broken']}, net={m['net_benefit']}")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAKIKO V3.1 \u2014 Cascade Binary Router Pipeline")
    parser.add_argument("--config", default="v31_cascade",
                        choices=list(CONFIGS.keys()),
                        help="Configuration preset")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    t0 = time.time()
    cfg_name = args.config
    cfg = CONFIGS[cfg_name]

    log.info("=" * 60)
    log.info("SAKIKO V3.1 Pipeline \u2014 config: %s", cfg_name)
    log.info("=" * 60)

    # Step 1: Load data
    rows = load_data()
    meta = build_meta(rows)

    n_correct = sum(1 for m in meta if m["correct"])
    log.info("Baseline (Phase1 avg_logp): %d/%d = %.2f%%",
             n_correct, len(meta), 100 * n_correct / len(meta))
    etypes = Counter(m["error_type"] for m in meta if not m["correct"])
    for k, v in sorted(etypes.items(), key=lambda x: -x[1]):
        log.info("  Error: %s = %d", k, v)

    # Step 2: Load model + activations
    model, tok = load_model()
    acts_obs = collect_activations(model, tok, rows, cfg["obs_layer"])

    # Step 3: Split (与 V3 相同 seed)
    train_idx, test_idx = split_data(meta)
    test_counts = Counter(meta[i]["error_type"] for i in test_idx)
    for k in [RFI2TC, CA2TC, CA2DIRECT, "correct"]:
        log.info("  Test %s: %d", k, test_counts.get(k, 0))

    # Step 4: Train PCA bases
    log.info("Training PCA bases...")
    basis_rfi = train_pca_basis(acts_obs, meta, train_idx, RFI2TC,
                                 cfg["n_comp"], ref_gold="tool_call")
    basis_ca_tc = train_pca_basis(acts_obs, meta, train_idx, CA2TC,
                                   cfg["n_comp"], ref_gold="tool_call")
    basis_ca_direct = None
    if cfg.get("ca_direct_inj_layer") is not None:
        basis_ca_direct = train_pca_basis(acts_obs, meta, train_idx, CA2DIRECT,
                                           cfg["n_comp"], ref_gold="cannot_answer")

    # Step 5: Correction vectors
    log.info("Computing correction vectors...")
    corrections = {}

    if basis_rfi is not None and cfg.get("rfi_inj_layer") is not None:
        corr_rfi = compute_correction_vec(basis_rfi, cfg["n_pcs"], cfg["rfi_alpha"])
        corrections["rfi_tc"] = {
            "vec": corr_rfi, "layer": cfg["rfi_inj_layer"], "mode": cfg["rfi_inj_mode"],
        }
        log.info("  RFI\u2192TC: L%d %s \u03b1=%.1f norm=%.3f",
                 cfg["rfi_inj_layer"], cfg["rfi_inj_mode"], cfg["rfi_alpha"],
                 np.linalg.norm(corr_rfi))

    if basis_ca_tc is not None and cfg.get("ca_tc_inj_layer") is not None:
        corr_ca_tc = compute_correction_vec(basis_ca_tc, cfg["n_pcs"], cfg["ca_tc_alpha"])
        corrections["ca_tc"] = {
            "vec": corr_ca_tc, "layer": cfg["ca_tc_inj_layer"], "mode": cfg["ca_tc_inj_mode"],
        }
        log.info("  CA\u2192TC: L%d %s \u03b1=%.1f norm=%.3f",
                 cfg["ca_tc_inj_layer"], cfg["ca_tc_inj_mode"], cfg["ca_tc_alpha"],
                 np.linalg.norm(corr_ca_tc))

    if basis_ca_direct is not None and cfg.get("ca_direct_inj_layer") is not None:
        corr_ca_dir = compute_correction_vec(basis_ca_direct, cfg["n_pcs"], cfg["ca_direct_alpha"])
        corrections["ca_direct"] = {
            "vec": corr_ca_dir, "layer": cfg["ca_direct_inj_layer"], "mode": cfg["ca_direct_inj_mode"],
        }
        log.info("  CA\u2192direct: L%d %s \u03b1=%.1f norm=%.3f",
                 cfg["ca_direct_inj_layer"], cfg["ca_direct_inj_mode"],
                 cfg["ca_direct_alpha"], np.linalg.norm(corr_ca_dir))

    # Step 6: Train cascade binary routers
    log.info("Training cascade binary routers...")
    routers = {}
    routers["rfi_tc"] = train_binary_router(
        acts_obs, meta, train_idx, RFI2TC, name="rfi_tc")
    routers["ca_tc"] = train_binary_router(
        acts_obs, meta, train_idx, CA2TC, name="ca_tc")
    if cfg.get("ca_direct_inj_layer") is not None:
        routers["ca_direct"] = train_binary_router(
            acts_obs, meta, train_idx, CA2DIRECT, name="ca_direct")
    else:
        routers["ca_direct"] = None

    # Step 7: Evaluate
    log.info("Evaluating on test set...")
    results = evaluate(model, tok, rows, meta, test_idx, acts_obs,
                       routers, cfg, corrections)
    metrics = compute_metrics(results, cfg)

    # Router CV info
    metrics["router_cv"] = {}
    for ch_name, r in routers.items():
        if r is not None:
            metrics["router_cv"][ch_name] = {
                "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                "n_pos": r["n_pos"], "n_neg": r["n_neg"],
            }

    metrics["config_name"] = cfg_name
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    metrics["elapsed_min"] = round((time.time() - t0) / 60, 1)

    # Step 8: Output
    print_report(metrics, cfg, cfg_name)

    out_path = Path(args.output) if args.output else RESULT_DIR / f"v31_{cfg_name}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False,
                  default=lambda o: int(o) if hasattr(o, "item") else o)
    log.info("Results saved to %s", out_path)

    detail_path = RESULT_DIR / f"v31_{cfg_name}_details.jsonl"
    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False,
                               default=lambda o: int(o) if hasattr(o, "item") else o) + "\n")
    log.info("Details saved to %s", detail_path)

    log.info("Done. Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
