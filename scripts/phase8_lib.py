"""
phase8_lib.py — Phase 8 prospective validation library (untouched architecture).
================================================================================
Registers the Phase-8 target model and reuses the FROZEN Gate-v2 machinery verbatim:
  - avg_logp W2C readout (same 4 candidate texts, argmax) as Phi/Qwen/Mistral;
  - normalized-depth layer grid round({0.40,0.55,0.70,0.85}*L);
  - rho = ||dh|| / median_train(||h_MLP(inj)||)  (train-only scale);
  - rho grid {0.25,0.5,1,2,4,8}; DiffMean/PCA-1; LR matched-pred router;
  - Stage-4/5 thresholds and the split firewall.

No frozen scientific choice is changed here. The target model, its layer count, and its chat
template are the only new inputs; the layer grid is COMPUTED from the model's own depth.

Firewall: assert_no_test() guards every INTERVENTION index set. Baseline predictions on the
full 3652-row set (phase8_baseline.py) are consumed by discovery for support counts only; the
intervention/steering evaluation on test rows is blocked until the frozen-decisions file is
hashed.
"""
from __future__ import annotations
import json, sys, logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "final" / "results" / "splits"
RAW_MCQ = ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"
OUT = ROOT / "final" / "results" / "phase8_prospective_llama"
CACHE = Path("/root/autodl-tmp/phase8_cache")
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
GOLD_CLASSES = ["tool_call", "request_for_info", "cannot_answer"]
LABEL_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
MAXTOK = 8192

# FROZEN gate constants (verbatim from ACTIONABILITY_GATE_V2_SPEC.md)
RHO_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RHO_MAX = 8.0
N_RANDOM = 20
VAL_SEED_BLOCK = 2000
TEST_SEED_BLOCK = 1000
DEGEN_RATIO_FLOOR = 0.05
COS_THR = 0.6
SPLITHALF_FLOOR = 0.5
AUC_FLOOR = 0.75
PREC_FLOOR = 0.5
REF_FLOOR = 80
POWER_FLOOR = 30
Z_MIN = 2.0
FRAC_GE_MAX = 0.05
RESIDUAL_FRAC = 0.75
BROKE_RATIO = 0.5
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
INJ_OFFSET_FRACS = [0.0, -0.06, -0.12]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phase8")

MODEL_KEY = "llama31_8b"
MODELS = {
    "llama31_8b": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "path": "/root/autodl-tmp/models/Llama-3.1-8B-Instruct",
        "label_order": LABEL_ORDER,
    },
}
BASE_DETAILS = CACHE / f"{MODEL_KEY}_w2c_baseline_details.jsonl"


def model_config():
    return json.loads((Path(MODELS[MODEL_KEY]["path"]) / "config.json").read_text())


def n_layers():
    return int(model_config()["num_hidden_layers"])


def hidden_size():
    return int(model_config()["hidden_size"])


def obs_grid():
    L = n_layers()
    return sorted({int(round(f * L)) for f in (0.40, 0.55, 0.70, 0.85)})


def inj_candidates(obs):
    L = n_layers()
    return sorted({max(0, obs + int(round(fr * L))) for fr in INJ_OFFSET_FRACS})


def wrong_layer():
    return int(round(0.15 * n_layers()))


# ── firewall ────────────────────────────────────────────────────────────────────
def _load_splits():
    tr = list(map(int, json.load(open(SPLIT_DIR / "train_idx.json"))))
    va = list(map(int, json.load(open(SPLIT_DIR / "val_idx.json"))))
    te = list(map(int, json.load(open(SPLIT_DIR / "test_idx.json"))))
    return tr, va, te


TRAIN_IDX, VAL_IDX, TEST_IDX = _load_splits()
TEST_SET = frozenset(TEST_IDX)


def assert_no_test(idxs, where=""):
    bad = TEST_SET.intersection(int(i) for i in idxs)
    if bad:
        raise RuntimeError(f"SPLIT FIREWALL VIOLATION [{where}]: {len(bad)} test rows in an "
                           f"intervention index set (e.g. {sorted(bad)[:5]}).")
    return True


def splits_train_val():
    return list(TRAIN_IDX), list(VAL_IDX)


def splits_all():
    return list(TRAIN_IDX), list(VAL_IDX), list(TEST_IDX)


# ── data ────────────────────────────────────────────────────────────────────────
def load_ds_raw():
    from datasets import Dataset
    rows = [json.loads(l) for l in open(RAW_MCQ)]
    keys = sorted({k for r in rows for k in r.keys()})
    norm = [{k: r.get(k, None) for k in keys} for r in rows]
    return Dataset.from_list(norm).shuffle(seed=SEED)


def load_meta(details_path=None):
    bd = Path(details_path) if details_path else BASE_DETAILS
    ds = load_ds_raw()
    det = [json.loads(l) for l in open(bd)]
    assert len(ds) == len(det), (len(ds), len(det))
    split_of = {}
    for nm, idxs in [("train", TRAIN_IDX), ("val", VAL_IDX), ("test", TEST_IDX)]:
        for i in idxs:
            split_of[i] = nm
    meta = []
    for i, (s, d) in enumerate(zip(ds, det)):
        assert s["uuid"] == d["uuid"], i
        g, p = d["gold"], d["pred"]
        meta.append({"idx": i, "uuid": s["uuid"], "split": split_of[i], "gold": g, "pred": p,
                     "correct": p == g, "etype": "correct" if p == g else f"{g}__{p}"})
    return ds, meta


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
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = MODELS[MODEL_KEY]
    tok = AutoTokenizer.from_pretrained(M["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        M["path"], torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    dt = next(model.parameters()).dtype
    if dt != torch.bfloat16:
        raise RuntimeError(f"expected bf16, got {dt}")
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("quantization config present; bf16 required")
    dev = next(model.parameters()).device
    log.info("%s loaded | layers=%d hidden=%d dtype=%s VRAM=%.2fGB", MODEL_KEY,
             model.config.num_hidden_layers, model.config.hidden_size, dt,
             torch.cuda.memory_reserved(0) / 1e9)
    return model, tok, dev


# ── scoring (archived-exact avg_logp; optional MLP-output steering) ──────────────
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


def cos_dm_pc1(A, tr, meta, etype, ref_gold):
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    if len(err) < 5 or len(ref) < 5:
        return None
    dm, _ = diffmean(A, err, ref); pc, _ = pca_dir(A, err, ref, 1)
    return float(abs(np.dot(dm, pc)))


def splithalf_cos(A, tr, meta, etype, ref_gold, n=20):
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    if len(err) < 10 or len(ref) < 6:
        return None
    rng = np.random.RandomState(0); cos = []
    for _ in range(n):
        pe, pr = rng.permutation(err), rng.permutation(ref)
        d1, _ = diffmean(A, list(pe[:len(pe)//2]), list(pr[:len(pr)//2]))
        d2, _ = diffmean(A, list(pe[len(pe)//2:]), list(pr[len(pr)//2:]))
        cos.append(float(d1 @ d2))
    return float(np.mean(cos))


def fit_router(A, tr, va, meta, etype, from_pred):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    assert_no_test(tr, "router/train"); assert_no_test(va, "router/val")
    pos = [i for i in tr if meta[i]["etype"] == etype]
    neg = [i for i in tr if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    if len(pos) < 10 or len(neg) < 10:
        return None
    sc = StandardScaler().fit(A[neg + pos])
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=42).fit(
        sc.transform(A[neg + pos]), np.array([0] * len(neg) + [1] * len(pos)))
    vpos = [i for i in va if meta[i]["etype"] == etype]
    vneg = [i for i in va if meta[i]["pred"] == from_pred and meta[i]["etype"] != etype]
    vauc = None; pr = {}; sc_pos = sc_neg = None
    if vpos and vneg:
        y = np.array([0] * len(vneg) + [1] * len(vpos))
        prob = clf.predict_proba(sc.transform(A[vneg + vpos]))[:, 1]
        vauc = float(roc_auc_score(y, prob))
        sc_pos = clf.predict_proba(sc.transform(A[vpos]))[:, 1]
        sc_neg = clf.predict_proba(sc.transform(A[vneg]))[:, 1]
        for thr in THRESHOLDS:
            tp = int((sc_pos >= thr).sum()); fp = int((sc_neg >= thr).sum())
            pr[str(thr)] = {"precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                            "recall": round(tp / len(sc_pos), 3), "fires": tp + fp}
    return {"scaler": sc, "clf": clf, "val_auc": vauc, "n_pos": len(pos), "n_neg": len(neg),
            "pr": pr,
            "margin_gap": (float(np.median(sc_pos) - np.median(sc_neg))
                           if sc_pos is not None else None),
            "fp_on_correct": (int(sum(1 for i, s in zip(vneg, sc_neg)
                                      if s >= 0.5 and meta[i]["correct"]))
                              if sc_neg is not None else None)}


def rscore(router, A, idxs):
    assert_no_test(idxs, "rscore")
    return dict(zip(idxs, router["clf"].predict_proba(
        router["scaler"].transform(A[idxs]))[:, 1]))


# ── metrics ─────────────────────────────────────────────────────────────────────
def metrics(idxs, meta, base_pred, int_pred, channels):
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
    acc = sum(1 for i in idxs if int_pred[i] == meta[i]["gold"]) / len(idxs)
    golds = [meta[i]["gold"] for i in idxs]; preds = [int_pred[i] for i in idxs]
    active = [c for c in GOLD_CLASSES if c in set(golds)]
    f1 = float(f1_score(golds, preds, labels=active, average="macro", zero_division=0))
    return {"acc": round(acc, 5), "macro_f1_3cls": round(f1, 5),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "nt_before": nt_before, "nt_after": nt_after}


def random_unit(d, seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)
