"""
phase7_lib.py — shared library for Phase 7 (development-only rho/specificity study).
====================================================================================
HARD SPLIT FIREWALL: this module refuses to operate on test rows. Every experiment index set
passes through `assert_no_test()`, and `experiment_rows()` rejects any row whose split=="test".
Phase 7 uses TRAIN (scale + directions + routers) and VALIDATION (all intervention measurement)
only. No test example is ever loaded into an experiment, scored, or used for selection.

Reuses the archived, numerically-exact scoring path (avg_logp over the 4 W2C candidate texts,
per-candidate forward, MLP-output hook at the injection layer) so that Phase-7 validation
numbers are comparable to the archived validation numbers.

Magnitude: the archived implementation adds  corr = alpha * median_train(||h_obs||) * unit,
so ||delta_h|| = alpha * median_train(||h_obs||)  (unit is L2-normalized).
Phase 7 re-parameterizes by the architecture-normalized ratio measured AT THE INJECTION LAYER:

    rho = ||delta_h|| / median_train(||h_inj||)      =>      alpha = rho * med_inj / med_obs
"""
from __future__ import annotations
import json, sys, logging
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "final" / "results" / "splits"
RAW_MCQ = ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"
OUT = ROOT / "final" / "results" / "actionability_gate_development"
CACHE_OUT = Path("/root/autodl-tmp/phase7_cache")   # OUTSIDE repo (never committed)
CACHE_OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
GOLD_CLASSES = ["tool_call", "request_for_info", "cannot_answer"]
MAXTOK = 8192

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phase7")

# ── model registry (both weights already local; NO download in Phase 7) ──────────
MODELS = {
    "qwen25_7b": {
        "path": str(ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"),
        "layers": 28, "hidden": 3584,
        "cache": ROOT / "data/processed/qwen25_7b_w2c/cache",
        "details": ROOT / "data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl",
        # archived argmax order for this model's evaluator
        "label_order": ["tool_call", "direct", "request_for_info", "cannot_answer"],
    },
    "mistral7b_v03": {
        "path": "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3",
        "layers": 32, "hidden": 4096,
        "cache": Path("/root/autodl-tmp/phase5_mistral_cache"),
        "details": Path("/root/autodl-tmp/phase5_mistral_cache/mistral7b_w2c_baseline_details.jsonl"),
        "label_order": ["direct", "tool_call", "request_for_info", "cannot_answer"],
    },
}

# ── FIREWALL ────────────────────────────────────────────────────────────────────
def _load_splits():
    tr = list(map(int, json.load(open(SPLIT_DIR / "train_idx.json"))))
    va = list(map(int, json.load(open(SPLIT_DIR / "val_idx.json"))))
    te = list(map(int, json.load(open(SPLIT_DIR / "test_idx.json"))))
    return tr, va, te


TRAIN_IDX, VAL_IDX, TEST_IDX = _load_splits()
TEST_SET = frozenset(TEST_IDX)
ALLOWED = frozenset(TRAIN_IDX) | frozenset(VAL_IDX)


def assert_no_test(idxs, where=""):
    """Hard firewall: raise if any index belongs to the test split."""
    bad = TEST_SET.intersection(int(i) for i in idxs)
    if bad:
        raise RuntimeError(
            f"SPLIT FIREWALL VIOLATION [{where}]: {len(bad)} test rows requested "
            f"(e.g. {sorted(bad)[:5]}). Phase 7 is train/validation-only.")
    unknown = set(int(i) for i in idxs) - ALLOWED
    if unknown:
        raise RuntimeError(f"FIREWALL [{where}]: {len(unknown)} rows outside train∪val.")
    return True


def experiment_rows(rows):
    """Reject any experiment row carrying split == 'test'."""
    for r in rows:
        if str(r.get("split", "")).lower() == "test":
            raise RuntimeError(f"SPLIT FIREWALL VIOLATION: row with split=='test': {r}")
    return rows


def splits():
    return list(TRAIN_IDX), list(VAL_IDX)   # test deliberately not returned


# ── data ────────────────────────────────────────────────────────────────────────
def load_ds_raw():
    """Raw jsonl -> union-normalized -> Dataset -> shuffle(42). Verified identical to the
    HF When2Call test/mcq seed-42 order (Phase-5 preflight)."""
    from datasets import Dataset
    rows = [json.loads(l) for l in open(RAW_MCQ)]
    keys = sorted({k for r in rows for k in r.keys()})
    norm = [{k: r.get(k, None) for k in keys} for r in rows]
    return Dataset.from_list(norm).shuffle(seed=SEED)


def load_meta(model_key):
    """Per-row (gold, pred, correct, etype) from the model's archived baseline details.
    NOTE: the details file is read for ROW ALIGNMENT only; test rows are masked immediately
    and can never enter a statistic (any access is firewalled by assert_no_test)."""
    M = MODELS[model_key]
    det = [json.loads(l) for l in open(M["details"])]
    meta = []
    for i, d in enumerate(det):
        if i in TEST_SET:
            meta.append({"idx": i, "split": "test", "masked": True})   # masked: unusable
            continue
        gold, pred = d["gold"], d["pred"]
        meta.append({"idx": i, "split": "train" if i in set(TRAIN_IDX) else "val",
                     "gold": gold, "pred": pred, "correct": pred == gold,
                     "etype": "correct" if pred == gold else f"{gold}__{pred}",
                     "masked": False})
    return meta


# ── prompt / model ──────────────────────────────────────────────────────────────
def parse_tools(tools):
    out = []
    for t in tools or []:
        if isinstance(t, str):
            try:
                out.append(json.loads(t))
            except json.JSONDecodeError:
                out.append({"raw": t})
        else:
            out.append(t)
    return out


def build_prompt_messages(question, tools):
    parts = ["You are a helpful assistant."]
    if tools:
        parts.append("You have access to the following tools:\n"
                     + json.dumps(tools, indent=2, ensure_ascii=False))
    else:
        parts.append("No tools are available.")
    parts.append("Given the user's question, choose the most appropriate response "
                 "from the provided options.")
    return [{"role": "system", "content": "\n\n".join(parts)},
            {"role": "user", "content": question}]


def make_prompt_text(tok, sample):
    msgs = build_prompt_messages(sample["question"], parse_tools(sample["tools"]))
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        s = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs)
        return s + "<|assistant|>\n"


def load_model(model_key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(M["path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        M["path"], torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    dt = next(model.parameters()).dtype
    if dt != torch.bfloat16:
        raise RuntimeError(f"expected bf16, got {dt}")
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("quantization config present; bf16 required")
    dev = next(model.parameters()).device
    assert model.config.num_hidden_layers == M["layers"], "layer count mismatch"
    assert model.config.hidden_size == M["hidden"], "hidden size mismatch"
    log.info("%s loaded | layers=%d hidden=%d dtype=%s VRAM=%.2fGB", model_key,
             model.config.num_hidden_layers, model.config.hidden_size, dt,
             torch.cuda.memory_reserved(0) / 1e9)
    return model, tok, dev


# ── scoring (archived-exact path) ───────────────────────────────────────────────
def score_candidate(model, tok, dev, prompt_ids, cand_text, corr_vec=None, inj_layer=None):
    import torch
    cids = tok.encode(cand_text, add_special_tokens=False)
    if not cids:
        return -1e9
    inp = torch.cat([prompt_ids, torch.tensor([cids], device=dev)], dim=1)
    plen = prompt_ids.shape[1]; clen = len(cids)
    h = None
    if corr_vec is not None and inj_layer is not None:
        ct = torch.from_numpy(corr_vec).to(torch.bfloat16).to(dev)

        def hook(m, _in, out):
            if isinstance(out, tuple):
                out[0][:, :, :] += ct; return out
            out[:, :, :] += ct; return out
        h = model.model.layers[inj_layer].mlp.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(inp).logits
    if h:
        h.remove()
    sl = logits[0, plen - 1: plen + clen - 1, :]
    lp = torch.log_softmax(sl.float(), dim=-1)
    val = sum(lp[i, cids[i]].item() for i in range(clen)) / clen
    del inp
    return val


def predict(model, tok, dev, sample, label_order, corr_vec=None, inj_layer=None):
    import torch
    pt = make_prompt_text(tok, sample)
    pids = torch.tensor([tok.encode(pt, add_special_tokens=False)], device=dev)
    sc = {}
    for lb in label_order:
        cand = sample["answers"].get(lb, "")
        sc[lb] = (score_candidate(model, tok, dev, pids, cand, corr_vec, inj_layer)
                  if cand else -1e9)
    del pids
    return max(label_order, key=lambda lb: sc[lb]), sc


# ── geometry (train-only) ───────────────────────────────────────────────────────
def diffmean(A, err, ref):
    raw = A[ref].mean(0) - A[err].mean(0)
    n = float(np.linalg.norm(raw))
    return (raw / (n + 1e-12)).astype(np.float32), n


def pca_dir(A, err, ref, k=1):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    idx = err + ref
    sc = StandardScaler().fit(A[idx]); X = sc.transform(A[idx])
    pca = PCA(n_components=min(k, X.shape[1], len(idx) - 1), random_state=42).fit(X)
    delta = (sc.transform(A[ref].mean(0, keepdims=True))[0]
             - sc.transform(A[err].mean(0, keepdims=True))[0])
    comp = pca.components_[0]
    if np.dot(comp, delta) < 0:
        comp = -comp
    raw = comp / sc.scale_
    n = float(np.linalg.norm(raw))
    return (raw / (n + 1e-12)).astype(np.float32), n


def direction(A, tr, meta, etype, ref_gold, method):
    assert_no_test(tr, "direction/train")
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    unit, nrm = (diffmean(A, err, ref) if method == "diffmean" else pca_dir(A, err, ref, 1))
    return unit, nrm, len(err), len(ref)


def fit_router(A, tr, va, meta, etype, from_pred):
    """LR router, matched-pred negatives (archived Phase-3/5 convention). Train-only fit."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    assert_no_test(tr, "router/train"); assert_no_test(va, "router/val")
    pos = [i for i in tr if meta[i]["etype"] == etype]
    neg = [i for i in tr if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    sc = StandardScaler().fit(A[neg + pos])
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42).fit(
        sc.transform(A[neg + pos]), np.array([0] * len(neg) + [1] * len(pos)))
    vpos = [i for i in va if meta[i]["etype"] == etype]
    vneg = [i for i in va if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    vauc = None
    if vpos and vneg:
        y = np.array([0] * len(vneg) + [1] * len(vpos))
        vauc = float(roc_auc_score(y, clf.predict_proba(sc.transform(A[vneg + vpos]))[:, 1]))
    return {"scaler": sc, "clf": clf, "val_auc": vauc, "n_pos": len(pos), "n_neg": len(neg)}


def rscore(router, A, idxs):
    assert_no_test(idxs, "rscore")
    return dict(zip(idxs, router["clf"].predict_proba(
        router["scaler"].transform(A[idxs]))[:, 1]))


# ── metrics (validation only) ───────────────────────────────────────────────────
def metrics(idxs, meta, base_pred, int_pred, channels):
    from sklearn.metrics import f1_score
    assert_no_test(idxs, "metrics")
    fixed = broke = 0
    for i in idxs:
        g = meta[i]["gold"]
        if base_pred[i] != g and int_pred[i] == g:
            fixed += 1
        elif base_pred[i] == g and int_pred[i] != g:
            broke += 1
    nt_before = {ch: sum(1 for i in idxs if meta[i]["etype"] == d["etype"])
                 for ch, d in channels.items()}
    nt_after = {ch: sum(1 for i in idxs
                        if meta[i]["gold"] == d["gold"] and int_pred[i] == d["from_pred"])
                for ch, d in channels.items()}
    acc = sum(1 for i in idxs if int_pred[i] == meta[i]["gold"]) / len(idxs)
    golds = [meta[i]["gold"] for i in idxs]; preds = [int_pred[i] for i in idxs]
    active = [c for c in GOLD_CLASSES if c in set(golds)]
    f1 = float(f1_score(golds, preds, labels=active, average="macro", zero_division=0))
    return {"acc": round(acc, 5), "macro_f1_3cls": round(f1, 5),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "nt_before": nt_before, "nt_after": nt_after}


def random_unit(d, seed):
    """Reproducible matched-norm random direction (unit vector; norm matched by caller)."""
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)
