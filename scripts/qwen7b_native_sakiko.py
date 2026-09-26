"""
qwen7b_native_sakiko.py — Qwen2.5-7B target-native SAKIKO pilot on W2C (seed=42)
================================================================================
Target-native: all activations, routers, directions, layers, thresholds are
learned directly on Qwen2.5-7B. NO cross-model mapping, NO geometry.

Channels (gold-focused; `direct` is a W2C distractor, never gold):
  rfi_tc    : gold request_for_info, baseline pred tool_call   (over-call)
  ca_tc     : gold cannot_answer,    baseline pred tool_call   (over-call)
  ca_direct : gold cannot_answer,    baseline pred direct

Intervention:  h' = h + alpha * median_norm * unit(DiffMean direction)
               (MLP output at injection layer, all positions).
Gate: a channel touches a sample only if baseline pred == the channel's "from"
class, AND its router fires. Cascade priority: rfi_tc -> ca_tc -> ca_direct.

Stages (run in order; activations cached so the model pass happens once):
  extract  : load model, cache MLP acts at obs layers for all 3652 (one pass)
  analyze  : routers + DiffMean directions + reports          (no model)
  run      : val sweep -> lock -> locked test -> placebo       (model)

Usage:
  /path/to/project/.venv/bin/python scripts/qwen7b_native_sakiko.py --stage extract
  /path/to/project/.venv/bin/python scripts/qwen7b_native_sakiko.py --stage analyze
  /path/to/project/.venv/bin/python scripts/qwen7b_native_sakiko.py --stage run
"""

from __future__ import annotations
import argparse, gc, json, sys, time, logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct")
SPLIT_DIR = ROOT / "sakiko_v3" / "results" / "splits"
CACHE_DIR = ROOT / "data" / "processed" / "qwen25_7b_w2c" / "cache"
OUT = ROOT / "final" / "results" / "7b_w2c_sakiko"
BASE_DETAILS = ROOT / "final" / "results" / "7b_w2c_baseline" / "qwen25_7b_baseline_details.jsonl"
CACHE_DIR.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)

LABEL_VOCAB = ["tool_call", "direct", "request_for_info", "cannot_answer"]
OBS_LAYERS = [12, 16, 20]          # Qwen2.5-7B has 28 layers (43% / 57% / 71%)
DTYPE = torch.bfloat16
SEED = 42

# channel: (gold, from_pred, ref_gold)
CHANNELS = {
    "rfi_tc":    {"gold": "request_for_info", "from_pred": "tool_call", "ref_gold": "request_for_info", "target": "request_for_info"},
    "ca_tc":     {"gold": "cannot_answer",    "from_pred": "tool_call", "ref_gold": "cannot_answer",    "target": "cannot_answer"},
    "ca_direct": {"gold": "cannot_answer",    "from_pred": "direct",    "ref_gold": "cannot_answer",    "target": "cannot_answer"},
}
PRIORITY = ["rfi_tc", "ca_tc", "ca_direct"]

ALPHAS = [0.5, 1.0, 2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("qwen7b_sakiko")


# ── shared data prep ──────────────────────────────────────────────────────────

def load_meta_and_dataset():
    ds = load_dataset("nvidia/When2Call", "test", split="mcq").shuffle(seed=SEED)
    det = [json.loads(l) for l in open(BASE_DETAILS)]
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


def load_splits():
    tr = json.load(open(SPLIT_DIR / "train_idx.json"))
    va = json.load(open(SPLIT_DIR / "val_idx.json"))
    te = json.load(open(SPLIT_DIR / "test_idx.json"))
    return list(map(int, tr)), list(map(int, va)), list(map(int, te))


def build_prompt_messages(question, tools):
    parts = ["You are a helpful assistant."]
    if tools:
        parts.append("You have access to the following tools:\n" +
                     json.dumps(tools, indent=2, ensure_ascii=False))
    else:
        parts.append("No tools are available.")
    parts.append("Given the user's question, choose the most appropriate response "
                 "from the provided options.")
    return [{"role": "system", "content": "\n\n".join(parts)},
            {"role": "user", "content": question}]


def parse_tools(tools):
    out = []
    for t in tools or []:
        if isinstance(t, str):
            try: out.append(json.loads(t))
            except json.JSONDecodeError: out.append({"raw": t})
        else: out.append(t)
    return out


def make_prompt_text(tok, sample):
    msgs = build_prompt_messages(sample["question"], parse_tools(sample["tools"]))
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        s = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs)
        return s + "<|assistant|>\n"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device
    log.info("Model loaded | layers=%d hidden=%d | VRAM=%.2fGB",
             model.config.num_hidden_layers, model.config.hidden_size,
             torch.cuda.memory_reserved(0) / 1e9)
    return model, tok, dev


# ── STAGE extract ─────────────────────────────────────────────────────────────

def stage_extract(args):
    cache_paths = {L: CACHE_DIR / f"acts_L{L}.npy" for L in OBS_LAYERS}
    if all(p.exists() for p in cache_paths.values()) and not args.force:
        log.info("All obs-layer caches exist; skipping extraction.")
        return
    ds, meta = load_meta_and_dataset()
    model, tok, dev = load_model()
    D = model.config.hidden_size
    n = len(ds)
    MAXTOK = 8192
    acts = {L: np.zeros((n, D), dtype=np.float32) for L in OBS_LAYERS}
    mlps = {L: model.model.layers[L].mlp for L in OBS_LAYERS}
    skipped = []
    t0 = time.time()
    for i in tqdm(range(n), desc="extract", ncols=90):
        prompt = make_prompt_text(tok, ds[i])
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) > MAXTOK:
            skipped.append(i); continue
        last = len(ids) - 1
        inp = torch.tensor([ids], device=dev)
        cap = {}
        handles = []
        for L in OBS_LAYERS:
            def mk(L):
                def hook(m, _in, out, _L=L):
                    o = out[0] if isinstance(out, tuple) else out
                    cap[_L] = o[0, last, :].detach().float().cpu().numpy()
                return hook
            handles.append(mlps[L].register_forward_hook(mk(L)))
        with torch.no_grad():
            model(inp)
        for h in handles: h.remove()
        for L in OBS_LAYERS:
            acts[L][i] = cap[L]
        del inp
        if (i + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    for L in OBS_LAYERS:
        np.save(cache_paths[L], acts[L])
    elapsed = time.time() - t0
    log.info("Extraction done in %.1f min; skipped=%d", elapsed/60, len(skipped))
    rep = OUT / "qwen25_7b_activation_extraction_report.md"
    with open(rep, "w") as f:
        f.write("# Qwen2.5-7B Activation Extraction Report\n\n")
        f.write(f"- Model: Qwen2.5-7B-Instruct (28 layers, hidden={D}), bf16\n")
        f.write(f"- Decision position: last prompt token (matches W2C evaluator)\n")
        f.write(f"- Obs layers: {OBS_LAYERS} (43% / 57% / 71% depth)\n")
        f.write(f"- Samples: {n}; skipped (>{MAXTOK} tok): {len(skipped)}\n")
        f.write(f"- Cache: data/processed/qwen25_7b_w2c/cache/acts_L*.npy, shape ({n}, {D}), float32\n")
        f.write(f"- Row index == seed-42-shuffled W2C order == split index == baseline-details row\n")
        f.write(f"- Extraction time: {elapsed/60:.1f} min\n")
    log.info("Wrote %s", rep)


# ── STAGE analyze (routers + directions; no model) ───────────────────────────

def _etype(m): return m["etype"]

def stage_analyze(args):
    ds, meta = load_meta_and_dataset()
    tr, va, te = load_splits()
    acts = {L: np.load(CACHE_DIR / f"acts_L{L}.npy") for L in OBS_LAYERS}

    # split channel/correct counts
    def counts(idxs):
        c = Counter(meta[i]["etype"] for i in idxs)
        return {k: c.get(k, 0) for k in ["correct", "rfi_tc", "ca_tc", "ca_direct"]}
    split_counts = {"train": counts(tr), "val": counts(va), "test": counts(te)}
    log.info("split channel counts: %s", split_counts)

    # per-split baseline details (Step 1)
    with open(OUT / "qwen25_7b_split_baseline_details.jsonl", "w") as f:
        det = [json.loads(l) for l in open(BASE_DETAILS)]
        split_of = {}
        for name, idxs in [("train", tr), ("val", va), ("test", te)]:
            for i in idxs: split_of[i] = name
        for i, (m, d) in enumerate(zip(meta, det)):
            f.write(json.dumps({
                "idx": i, "uuid": m["uuid"], "split": split_of.get(i, "?"),
                "gold": m["gold"], "pred": m["pred"], "correct": m["correct"],
                "etype": m["etype"], "avg_logp": d["avg_logp"],
            }, ensure_ascii=False) + "\n")

    # ── routers per channel per obs layer ──
    def fit_router(L, ch):
        A = acts[L]
        pos = [i for i in tr if meta[i]["etype"] == ch]
        neg = [i for i in tr if meta[i]["correct"]]
        if len(pos) < 10: return None
        X = A[neg + pos]; y = np.array([0]*len(neg) + [1]*len(pos))
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)
        cvk = min(5, len(pos))
        cv = float(cross_val_score(clf, Xs, y, cv=cvk, scoring="roc_auc").mean()) if cvk >= 2 else None
        clf.fit(Xs, y)
        vpos = [i for i in va if meta[i]["etype"] == ch]
        vneg = [i for i in va if meta[i]["correct"]]
        vauc = None
        if vpos and vneg:
            Xv = sc.transform(A[vneg + vpos]); yv = np.array([0]*len(vneg)+[1]*len(vpos))
            vauc = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
        return {"scaler": sc, "clf": clf, "cv_auc": cv, "val_auc": vauc,
                "n_pos": len(pos), "n_neg": len(neg)}

    router_metrics = {}; chosen_obs = {}; routers_to_save = {}
    for ch in CHANNELS:
        per_layer = {}
        for L in OBS_LAYERS:
            r = fit_router(L, ch)
            if r:
                per_layer[L] = {"cv_auc": r["cv_auc"], "val_auc": r["val_auc"],
                                "n_pos": r["n_pos"], "n_neg": r["n_neg"]}
        if not per_layer:
            log.warning("channel %s: no router (insufficient pos)", ch); continue
        bestL = max(per_layer, key=lambda L: (per_layer[L]["val_auc"] or 0))
        chosen_obs[ch] = bestL
        # threshold P/R on val at best layer
        r = fit_router(bestL, ch); routers_to_save[ch] = (bestL, r)
        A = acts[bestL]
        v_all = va
        vy = np.array([1 if meta[i]["etype"] == ch else 0 for i in v_all])
        vscore = r["clf"].predict_proba(r["scaler"].transform(A[v_all]))[:, 1]
        pr = {}
        for thr in THRESHOLDS:
            fire = vscore >= thr
            tp = int(((fire == 1) & (vy == 1)).sum()); fp = int(((fire == 1) & (vy == 0)).sum())
            fn = int(((fire == 0) & (vy == 1)).sum())
            prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
            pr[str(thr)] = {"precision": round(prec, 3), "recall": round(rec, 3),
                            "fires_on_val": int(fire.sum())}
        router_metrics[ch] = {"chosen_obs_layer": bestL, "per_layer": per_layer,
                              "threshold_pr": pr}
        log.info("router %s: obs=L%d val_auc=%.3f pos=%d neg=%d",
                 ch, bestL, per_layer[bestL]["val_auc"], r["n_pos"], r["n_neg"])

    # ── DiffMean directions per channel at chosen obs ──
    directions = {}; dir_report = {}
    for ch, cfg in CHANNELS.items():
        if ch not in chosen_obs: continue
        L = chosen_obs[ch]; A = acts[L]
        err = [i for i in tr if meta[i]["etype"] == ch]
        ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == cfg["ref_gold"]]
        if len(err) < 10 or len(ref) < 5:
            dir_report[ch] = {"error": f"insufficient err={len(err)} ref={len(ref)}"}
            log.warning("direction %s: insufficient err=%d ref=%d", ch, len(err), len(ref)); continue
        emean = A[err].mean(0); rmean = A[ref].mean(0)
        raw = rmean - emean; nrm = float(np.linalg.norm(raw)); unit = raw/(nrm+1e-12)
        mednorm = float(np.median(np.linalg.norm(A[tr], axis=1)))
        directions[ch] = {"obs_layer": L, "unit": unit.astype(np.float32),
                          "median_norm": mednorm, "n_err": len(err), "n_ref": len(ref)}
        dir_report[ch] = {"obs_layer": L, "n_err": len(err), "n_ref": len(ref),
                          "direction_norm": round(nrm, 4), "median_norm": round(mednorm, 4),
                          "ref_pool_note": ("OK" if len(ref) >= 80 else f"LIMITED ({len(ref)} correct {cfg['ref_gold']})")}
        log.info("direction %s: obs=L%d err=%d ref=%d |dir|=%.3f mednorm=%.3f",
                 ch, L, len(err), len(ref), nrm, mednorm)

    # cosine sims between channel directions (same dim only if same layer; else note)
    cos = {}
    chs = list(directions)
    for a in range(len(chs)):
        for b in range(a+1, len(chs)):
            ca, cb = chs[a], chs[b]
            if directions[ca]["obs_layer"] == directions[cb]["obs_layer"]:
                u, v = directions[ca]["unit"], directions[cb]["unit"]
                cos[f"{ca}|{cb}"] = round(float(u @ v), 4)
            else:
                cos[f"{ca}|{cb}"] = f"different obs layers (L{directions[ca]['obs_layer']} vs L{directions[cb]['obs_layer']})"

    # save directions + routers
    np.savez(OUT / "qwen25_7b_directions.npz",
             **{f"{ch}_unit": directions[ch]["unit"] for ch in directions},
             **{f"{ch}_meta": np.array([directions[ch]["obs_layer"],
                                        directions[ch]["median_norm"]], dtype=np.float32)
                for ch in directions})
    import pickle
    with open(CACHE_DIR / "routers.pkl", "wb") as f:
        pickle.dump({ch: {"obs_layer": L, "scaler": r["scaler"], "clf": r["clf"]}
                     for ch, (L, r) in routers_to_save.items()}, f)
    with open(CACHE_DIR / "directions.pkl", "wb") as f:
        pickle.dump(directions, f)
    with open(CACHE_DIR / "chosen_obs.json", "w") as f:
        json.dump(chosen_obs, f)

    json.dump({"split_counts": split_counts, "router_metrics": router_metrics},
              open(OUT / "qwen25_7b_router_metrics.json", "w"), indent=2)
    json.dump({"directions": dir_report, "cosine_between_channels": cos},
              open(OUT / "qwen25_7b_directions_meta.json", "w"), indent=2)

    # reports
    with open(OUT / "qwen25_7b_split_baseline_summary.md", "w") as f:
        f.write("# Qwen2.5-7B — per-split baseline (W2C, seed=42)\n\n")
        f.write("| split | n | correct | rfi_tc | ca_tc | ca_direct |\n|---|---|---|---|---|---|\n")
        for nm, idxs in [("train", tr), ("val", va), ("test", te)]:
            c = split_counts[nm]
            f.write(f"| {nm} | {len(idxs)} | {c['correct']} | {c['rfi_tc']} | {c['ca_tc']} | {c['ca_direct']} |\n")
        f.write("\nSplit indices reused from sakiko_v3/results/splits (stratified on Phi-3.5 "
                "error types; here a fixed held-out partition). No tuning on test.\n")
    with open(OUT / "qwen25_7b_router_report.md", "w") as f:
        f.write("# Qwen2.5-7B Router Report\n\n")
        for ch in router_metrics:
            rm = router_metrics[ch]; L = rm["chosen_obs_layer"]
            f.write(f"## {ch}  (chosen obs L{L})\n\n")
            f.write("| obs layer | val AUC | cv AUC | n_pos | n_neg |\n|---|---|---|---|---|\n")
            for LL, d in rm["per_layer"].items():
                f.write(f"| L{LL} | {d['val_auc']:.3f} | {d['cv_auc']:.3f} | {d['n_pos']} | {d['n_neg']} |\n")
            f.write("\nThreshold P/R on val:\n\n| thr | precision | recall | fires |\n|---|---|---|---|\n")
            for thr, d in rm["threshold_pr"].items():
                f.write(f"| {thr} | {d['precision']} | {d['recall']} | {d['fires_on_val']} |\n")
            f.write("\n")
    with open(OUT / "qwen25_7b_direction_report.md", "w") as f:
        f.write("# Qwen2.5-7B DiffMean Direction Report\n\n")
        f.write("| channel | obs | n_err | n_ref | dir_norm | median_norm | ref pool |\n|---|---|---|---|---|---|---|\n")
        for ch, d in dir_report.items():
            if "error" in d:
                f.write(f"| {ch} | — | — | — | — | — | {d['error']} |\n")
            else:
                f.write(f"| {ch} | L{d['obs_layer']} | {d['n_err']} | {d['n_ref']} | "
                        f"{d['direction_norm']} | {d['median_norm']} | {d['ref_pool_note']} |\n")
        f.write(f"\nCosine between channel directions: {json.dumps(cos)}\n")
    log.info("analyze done; reports written to %s", OUT)


# ── hooked scoring (run stage) ───────────────────────────────────────────────

def score_hooked(model, tok, dev, prompt_text, cand_text, corr_vec, inj_layer):
    pids = tok.encode(prompt_text, add_special_tokens=False)
    cids = tok.encode(cand_text, add_special_tokens=False)
    if not cids: return -1e9
    inp = torch.tensor([pids + cids], device=dev)
    plen = len(pids); clen = len(cids)
    ct = torch.from_numpy(corr_vec).to(DTYPE).to(dev) if corr_vec is not None else None
    h = None
    if ct is not None:
        def hook(m, _in, out):
            if isinstance(out, tuple):
                out0 = out[0]; out0[:, :, :] += ct; return (out0,) + out[1:]
            out[:, :, :] += ct; return out
        h = model.model.layers[inj_layer].mlp.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(inp).logits
    if h: h.remove()
    sl = logits[0, plen-1: plen+clen-1, :]
    lp = torch.log_softmax(sl.float(), dim=-1)
    out = sum(lp[i, cids[i]].item() for i in range(clen)) / clen
    del inp
    return out


def predict_hooked(model, tok, dev, sample, corr_vec, inj_layer):
    pt = make_prompt_text(tok, sample)
    sc = {lb: score_hooked(model, tok, dev, pt, sample["answers"].get(lb, ""), corr_vec, inj_layer)
          if sample["answers"].get(lb, "") else -1e9 for lb in LABEL_VOCAB}
    return max(LABEL_VOCAB, key=lambda lb: sc[lb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract", "analyze", "run"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage == "extract": stage_extract(args)
    elif args.stage == "analyze": stage_analyze(args)
    elif args.stage == "run":
        from qwen7b_native_sakiko_run import stage_run
        stage_run()


if __name__ == "__main__":
    main()
