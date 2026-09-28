"""
phase5_mistral_lib.py — Phase-5 target-native SAKIKO-CA on Mistral-7B-Instruct-v0.3 × W2C.
============================================================================================
Standalone, model-INDEPENDENT of the Qwen archive: this module never imports a Qwen model
path, Qwen layer constant, Qwen router/direction, or the Qwen channel set. It reuses only
*math conventions* (avg_logp scoring, DiffMean/PCA-1, LR router, Net=Fixed-Broke) —
re-implemented here so the second-architecture claim is clean.

Everything Mistral-specific (weights, tokenizer, chat template, activations, routers,
directions, layers, thresholds) is learned on Mistral itself.

Data ordering is reproduced from the local raw W2C jsonl via the archived shim recipe
(`load_ds_raw`), which was verified this session to be bit-identical to
`load_dataset("nvidia/When2Call","test",split="mcq").shuffle(seed=42)`.

Layer grid is defined by NORMALIZED DEPTH over Mistral's 32 layers (NO Qwen indices):
  obs  ≈ {0.40, 0.55, 0.70, 0.85}·L  -> [13, 18, 22, 27]
  inj  = obs + {0, -2, -4}  (≈ {0, -0.06L, -0.12L})
  wrong-layer control = L5 (≈0.15L, shallow)

Caches (activations, per-example baseline details) live OUTSIDE the git repo at
/root/autodl-tmp/phase5_mistral_cache/ so they can never be committed.
"""
from __future__ import annotations
import json, sys, logging
from collections import Counter
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]                      # repo root
MODEL_PATH = "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3"  # outside repo
MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"

RAW_MCQ = ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"  # gitignored raw
SPLIT_DIR = ROOT / "final" / "results" / "splits"               # fixed seed-42 split (Phi-stratified)
OUT = ROOT / "final" / "results" / "mistral7b_w2c_sakiko_ca"

CACHE_DIR = Path("/root/autodl-tmp/phase5_mistral_cache")       # OUTSIDE repo (never committed)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DETAILS = CACHE_DIR / "mistral7b_w2c_baseline_details.jsonl"

# ── model / label constants ───────────────────────────────────────────────────
N_LAYERS = 32
HIDDEN = 4096
SEED = 42

# argmax candidate order (fixed; ties essentially never occur with real-valued avg_logp)
LABEL_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
GOLD_CLASSES = ["tool_call", "request_for_info", "cannot_answer"]  # `direct` is never gold

# normalized-depth grids (rounded to valid indices; documented in PHASE5_PROTOCOL.md)
OBS_LAYERS = [13, 18, 22, 27]           # round({0.40,0.55,0.70,0.85}*32)
INJ_OFFSETS = [0, -2, -4]               # obs + {0, ~-0.06L, ~-0.12L}
WRONG_LAYER = 5                         # ~0.15L shallow control

ALPHAS = [0.5, 1.0, 2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
# R1 degeneracy gate. The archived absolute norm floor (5.0) was CALIBRATED ON QWEN, whose
# MLP-output norms are ~5-10x larger than Mistral's (Mistral median act-norm 1.6-4.5). An
# absolute floor does not transfer across architectures, so Phase-5 uses the MODEL-AGNOSTIC
# relative criterion the method intends: reject a layer only if its DiffMean norm is a
# negligible fraction (< DEGEN_RATIO_FLOOR) of the median activation norm at that layer, then
# select the obs layer by MAX norm-ratio (tie-break router AUC). This is a train-only geometry
# calibration decided before any test inspection.
DEGEN_RATIO_FLOOR = 0.05                # R1: reject norm_ratio < 0.05 (near-zero direction)
COS_THR = 0.6                           # R2 DiffMean/PCA-1 trigger
N_RANDOM = 20                           # placebo random directions

MAXTOK = 8192

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phase5")


# ── data ──────────────────────────────────────────────────────────────────────
def load_ds_raw():
    """Raw jsonl -> union-normalized rows -> Dataset -> shuffle(seed=42).
    Verified bit-identical to load_dataset('nvidia/When2Call','test','mcq').shuffle(42)."""
    from datasets import Dataset
    rows = [json.loads(l) for l in open(RAW_MCQ)]
    keys = sorted({k for r in rows for k in r.keys()})
    norm = [{k: r.get(k, None) for k in keys} for r in rows]
    return Dataset.from_list(norm).shuffle(seed=SEED)


def load_meta(base_details_path=None):
    """Per-sample (gold, pred, correct, etype) from the Mistral baseline details."""
    bd = Path(base_details_path) if base_details_path else BASE_DETAILS
    ds = load_ds_raw()
    det = [json.loads(l) for l in open(bd)]
    assert len(ds) == len(det), (len(ds), len(det))
    meta = []
    for i, (s, d) in enumerate(zip(ds, det)):
        assert s["uuid"] == d["uuid"], i
        gold, pred = d["gold"], d["pred"]
        correct = (pred == gold)
        etype = "correct" if correct else f"{gold}__{pred}"
        meta.append({"idx": i, "uuid": s["uuid"], "gold": gold, "pred": pred,
                     "correct": correct, "etype": etype})
    return ds, meta


def load_splits():
    return (list(map(int, json.load(open(SPLIT_DIR / "train_idx.json")))),
            list(map(int, json.load(open(SPLIT_DIR / "val_idx.json")))),
            list(map(int, json.load(open(SPLIT_DIR / "test_idx.json")))))


def gen_split(meta, seed):
    """Deterministic stratified 70/15/15 on the model's own error type (rare<5 pooled)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    n = len(meta)
    raw = [m["etype"] for m in meta]
    cnt = Counter(raw)
    strat = [e if cnt[e] >= 5 else "_rare" for e in raw]
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trv, te = next(s1.split(range(n), strat))
    strat_trv = [strat[i] for i in trv]
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    tr_rel, va_rel = next(s2.split(range(len(trv)), strat_trv))
    tr = sorted(int(trv[i]) for i in tr_rel)
    va = sorted(int(trv[i]) for i in va_rel)
    te = sorted(int(i) for i in te)
    return tr, va, te


# ── prompt / model ────────────────────────────────────────────────────────────
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
    # Mistral-v0.3 chat template merges system+user into a single [INST]..[/INST] block
    # (verified this session); apply_chat_template handles it natively.
    msgs = build_prompt_messages(sample["question"], parse_tools(sample["tools"]))
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(assert_bf16=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    # hard guard: no silent quantization / dtype fallback
    p_dtype = next(model.parameters()).dtype
    if assert_bf16 and p_dtype != torch.bfloat16:
        raise RuntimeError(f"Expected bf16 params, got {p_dtype} — refusing to proceed.")
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("Quantization config present — bf16 required, aborting.")
    dev = next(model.parameters()).device
    log.info("Mistral loaded | layers=%d hidden=%d dtype=%s | VRAM=%.2fGB",
             model.config.num_hidden_layers, model.config.hidden_size, p_dtype,
             torch.cuda.memory_reserved(0) / 1e9)
    return model, tok, dev


# ── scoring (avg_logp; optional MLP-output steering at inj_layer) ──────────────
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


def predict(model, tok, dev, sample, corr_vec=None, inj_layer=None):
    import torch
    pt = make_prompt_text(tok, sample)
    pids = torch.tensor([tok.encode(pt, add_special_tokens=False)], device=dev)
    sc = {}
    for lb in LABEL_ORDER:
        cand = sample["answers"].get(lb, "")
        sc[lb] = (score_candidate(model, tok, dev, pids, cand, corr_vec, inj_layer)
                  if cand else -1e9)
    del pids
    return max(LABEL_ORDER, key=lambda lb: sc[lb]), sc


# ── geometry: directions & router (pure numpy / sklearn; Mistral activations) ──
def diffmean(A, err, ref):
    raw = A[ref].mean(0) - A[err].mean(0)
    nrm = float(np.linalg.norm(raw))
    return (raw / (nrm + 1e-12)).astype(np.float32), nrm


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
    nrm = float(np.linalg.norm(raw))
    return (raw / (nrm + 1e-12)).astype(np.float32), nrm


def direction(A, tr, meta, etype, ref_gold, method="diffmean"):
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    unit, nrm = (diffmean(A, err, ref) if method == "diffmean"
                 else pca_dir(A, err, ref, k=1))
    mn = float(np.median(np.linalg.norm(A[tr], axis=1)))
    return unit.astype(np.float32), float(nrm), mn, len(err), len(ref)


def cos_dm_pc1(A, tr, meta, etype, ref_gold):
    dm, _ = None, None
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    if len(err) < 5 or len(ref) < 5:
        return None
    dm, _ = diffmean(A, err, ref)
    pc, _ = pca_dir(A, err, ref, k=1)
    return float(abs(np.dot(dm, pc)))


def fit_router(A, tr, va, meta, etype, from_pred, negatives="matched_pred"):
    """LR router. positives = channel train errors; negatives = matched-pred (default) or correct."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    pos = [i for i in tr if meta[i]["etype"] == etype]
    if negatives == "matched_pred":
        neg = [i for i in tr if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    else:
        neg = [i for i in tr if meta[i]["correct"]]
    if len(pos) < 10 or len(neg) < 10:
        return None
    sc = StandardScaler().fit(A[neg + pos])
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                             random_state=42).fit(
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


# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(idxs, meta, base_pred, int_pred, channels):
    """Fixed/Broke/Net + per-channel residual before/after + acc + macro-F1(3cls).
    `channels` maps ch_id -> {"gold":..,"from_pred":..,"etype":..}."""
    from sklearn.metrics import f1_score
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
    chan_fixed = {ch: sum(1 for i in idxs if meta[i]["etype"] == d["etype"]
                          and int_pred[i] == meta[i]["gold"])
                  for ch, d in channels.items()}
    acc = sum(1 for i in idxs if int_pred[i] == meta[i]["gold"]) / len(idxs)
    golds = [meta[i]["gold"] for i in idxs]; preds = [int_pred[i] for i in idxs]
    active = [c for c in GOLD_CLASSES if c in set(golds)]
    f1 = float(f1_score(golds, preds, labels=active, average="macro", zero_division=0))
    return {"acc": round(acc, 5), "macro_f1_3cls": round(f1, 5),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "nt_before": nt_before, "nt_after": nt_after, "chan_fixed": chan_fixed}


def transition_matrix(idxs, meta, pred):
    m = Counter((meta[i]["gold"], pred[i]) for i in idxs)
    labels = ["tool_call", "direct", "request_for_info", "cannot_answer"]
    return {g: {p: int(m.get((g, p), 0)) for p in labels} for g in GOLD_CLASSES}


def destination_table(idxs, meta, base_pred, int_pred, touched):
    tset = set(touched); moves = Counter()
    for i in idxs:
        if i in tset:
            moves[(meta[i]["gold"], base_pred[i], int_pred[i])] += 1
    return [{"gold": g, "from": bp, "to": ip, "n": n}
            for (g, bp, ip), n in sorted(moves.items(), key=lambda x: -x[1])]
