"""
metatool_qwen7b_sakiko_ca.py — Phase 2: target-native SAKIKO-CA on MetaTool × Qwen2.5-7B.
==========================================================================================
Implements the locked PHASE2_PROTOCOL.md end-to-end, per channel and fully automatic:

  fresh baseline (R0, from metatool_qwen7b_baseline.py)
  -> transition discovery (Phase-1 pipeline, R0-gated)
  -> intervention-scope filter (train>=30 err, val>=5 err, ref>=30, R1 survivor)
  -> R1 obs-layer selection (norm>=5 gate, max norm/median ratio, tie-break router AUC)
  -> R2 direction method (DiffMean default; PCA-1 candidate only if cos(DM,PC1)<0.6)
  -> per-channel LR router (pos = channel train errors, neg = same-side train correct)
  -> val-only sweep: inj in {obs,obs-2,obs-4} x alpha {0.5,1,2,4,6} x thr {0.4..0.8}
  -> one-shot locked test arms  B (over-call) / C (under-call) / D (dual)
  -> controls: reverse, 20 matched-norm random draws, ungated  (test, locked config)
  -> multiseed (only via --multiseed after the pilot gate passes)

Gating NEVER sees gold labels: fire iff router_prob >= thr AND baseline_pred == channel side.
Usage:
  python scripts/metatool_qwen7b_sakiko_ca.py               # seed-42 pilot + controls
  python scripts/metatool_qwen7b_sakiko_ca.py --multiseed   # seeds 123/456/789/2024 (gate-passed)
"""
from __future__ import annotations
import argparse, json, random, sys, time, logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from discover_sakiko_channels import transition_discovery  # noqa: E402

DATA_DIR = ROOT / "data/processed/metatool_binary"
CACHE = DATA_DIR / "cache_qwen7b"
BASE_DIR = ROOT / "final/results/metatool_qwen7b_sakiko_ca"
MODEL_PATH = ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"
DTYPE = torch.bfloat16

OBS_LAYERS = [12, 16, 20, 24]
ALPHAS = [0.5, 1.0, 2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
R1_NORM_GATE = 5.0
R2_COS = 0.6
N_RANDOM = 20
SEEDS_MULTI = [123, 456, 789, 2024]

CANDIDATES = {"tool_call": "Yes", "no_tool": "No"}
LABEL_ORDER = ["tool_call", "no_tool"]

# channel definitions: side = baseline_pred value that makes a sample eligible
CHANNELS = {
    "nt_tc": {"gold": "no_tool", "pred": "tool_call", "side": "tool_call",
              "desc": "over-call: false Yes -> push toward No"},
    "tc_nt": {"gold": "tool_call", "pred": "no_tool", "side": "no_tool",
              "desc": "under-call: false No -> push toward Yes"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("p2")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_prompt_text(tp, tok):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tp}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gen_split(recs, seed):
    """Deterministic stratified 70/15/15 by gold label (protocol section 1-3)."""
    rng = random.Random(seed)
    split_of = {}
    for g in sorted({r["gold_response_mode"] for r in recs}):
        ids = sorted(r["sample_id"] for r in recs if r["gold_response_mode"] == g)
        rng.shuffle(ids)
        n = len(ids); ntr = round(n * .70); nva = round(n * .15)
        for i, sid in enumerate(ids):
            split_of[sid] = "train" if i < ntr else ("val" if i < ntr + nva else "test")
    return split_of


@torch.no_grad()
def predict_hooked(model, tok, rec, corr_vec, inj_layer, cand_ids_map):
    """Argmax Yes/No with corr_vec added to MLP output at inj_layer (mlp_all convention)."""
    pids = tok.encode(build_prompt_text(rec["thought_prompt"], tok), add_special_tokens=False)
    corr_t = torch.from_numpy(corr_vec).to(model.device)
    mlp = model.model.layers[inj_layer].mlp

    def _hook(module, inp, out):
        out += corr_t.to(out.dtype)
        return out

    scores = {}
    h = mlp.register_forward_hook(_hook)
    try:
        for lbl, cids in cand_ids_map.items():
            ids = torch.tensor([pids + cids], device=model.device)
            logits = model(ids).logits
            pl, cl = len(pids), len(cids)
            lp = torch.log_softmax(logits[0, pl - 1: pl + cl - 1, :].float(), dim=-1)
            scores[lbl] = sum(lp[i, cids[i]].item() for i in range(cl)) / cl
            del ids
    finally:
        h.remove()
    return max(LABEL_ORDER, key=lambda lb: scores[lb])


def fit_channel(ch, recs, acts, base_pred, split_of):
    """Scope filter + R1 + R2 + router for one channel. CPU only. Returns cfg or None."""
    C = CHANNELS[ch]
    idx = {s: [i for i, r in enumerate(recs) if split_of[r["sample_id"]] == s]
           for s in ("train", "val", "test")}
    def pools(split):
        err = [i for i in idx[split] if recs[i]["gold_response_mode"] == C["gold"]
               and base_pred[i] == C["pred"]]
        ref = [i for i in idx[split] if recs[i]["gold_response_mode"] == C["gold"]
               and base_pred[i] == C["gold"]]
        return err, ref
    tr_err, tr_ref = pools("train")
    va_err, _ = pools("val")
    te_err, _ = pools("test")
    scope = {"train_err": len(tr_err), "val_err": len(va_err), "test_err": len(te_err),
             "train_ref": len(tr_ref)}
    if len(tr_err) < 30 or len(va_err) < 5 or len(tr_ref) < 30:
        return None, {"channel": ch, "scope": scope, "eligible": False,
                      "reason": "scope filter: needs train_err>=30, val_err>=5, train_ref>=30"}

    # R1 across obs candidates (+ per-layer router val AUC as tie-break/report)
    per_layer = {}
    for L in OBS_LAYERS:
        A = acts[L]
        dm = A[tr_ref].mean(0) - A[tr_err].mean(0)
        nrm = float(np.linalg.norm(dm))
        med = float(np.median(np.linalg.norm(A[idx["train"]], axis=1)))
        # router: pos = channel train errors, neg = same-side train correct
        neg = [i for i in idx["train"]
               if base_pred[i] == C["side"] and recs[i]["gold_response_mode"] == base_pred[i]]
        X = np.vstack([A[neg], A[tr_err]]); y = np.r_[np.zeros(len(neg)), np.ones(len(tr_err))]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                                 random_state=42).fit(sc.transform(X), y)
        vpos, _ = pools("val")
        vneg = [i for i in idx["val"]
                if base_pred[i] == C["side"] and recs[i]["gold_response_mode"] == base_pred[i]]
        yv = np.r_[np.zeros(len(vneg)), np.ones(len(vpos))]
        pv = clf.predict_proba(sc.transform(A[vneg + vpos]))[:, 1]
        auc = float(roc_auc_score(yv, pv)) if len(set(yv)) > 1 else 0.0
        per_layer[L] = {"dm_norm": round(nrm, 3), "median_act_norm": round(med, 3),
                        "ratio": round(nrm / med, 4), "router_val_auc": round(auc, 4),
                        "n_neg_train": len(neg)}
    survivors = [L for L in OBS_LAYERS if per_layer[L]["dm_norm"] >= R1_NORM_GATE]
    if not survivors:
        return None, {"channel": ch, "scope": scope, "per_layer": per_layer, "eligible": False,
                      "reason": f"R1: all DiffMean norms < {R1_NORM_GATE} (degenerate)"}
    obs = max(survivors, key=lambda L: (per_layer[L]["ratio"], per_layer[L]["router_val_auc"]))

    A = acts[obs]
    dm = A[tr_ref].mean(0) - A[tr_err].mean(0)
    unit_dm = (dm / (np.linalg.norm(dm) + 1e-12)).astype(np.float32)
    med = float(np.median(np.linalg.norm(A[idx["train"]], axis=1)))
    # R2: PCA-1 of err pool
    E = A[tr_err] - A[tr_err].mean(0)
    _, _, Vt = np.linalg.svd(E, full_matrices=False)
    pc1 = Vt[0]
    cos = float(np.dot(unit_dm, pc1))
    if cos < 0:
        pc1, cos = -pc1, -cos
    dir_cands = {"diffmean": unit_dm}
    if cos < R2_COS:
        dir_cands["pca1"] = pc1.astype(np.float32)

    # final router at obs
    neg = [i for i in idx["train"]
           if base_pred[i] == C["side"] and recs[i]["gold_response_mode"] == base_pred[i]]
    X = np.vstack([A[neg], A[tr_err]]); y = np.r_[np.zeros(len(neg)), np.ones(len(tr_err))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                             random_state=42).fit(sc.transform(X), y)
    probs_all = clf.predict_proba(sc.transform(A)).astype(np.float32)[:, 1]

    cfg = {"channel": ch, "side": C["side"], "gold": C["gold"], "scope": scope,
           "per_layer": per_layer, "obs": obs, "median_norm": med,
           "dm_norm": per_layer[obs]["dm_norm"], "cos_dm_pc1": round(cos, 4),
           "router_val_auc": per_layer[obs]["router_val_auc"],
           "dir_cands": dir_cands, "probs_all": probs_all}
    return cfg, {"channel": ch, "scope": scope, "per_layer": per_layer, "eligible": True,
                 "obs": obs, "cos_dm_pc1": round(cos, 4),
                 "pca1_evaluated": "pca1" in dir_cands}


def arm_metrics(recs, idxs, base_pred, preds, ch_fires=None):
    """Full metric block for a set of final predictions on idxs."""
    gold = [recs[i]["gold_response_mode"] for i in idxs]
    base = [base_pred[i] for i in idxs]
    n = len(idxs)
    conf = {g: {p: 0 for p in LABEL_ORDER} for g in LABEL_ORDER}
    for g, p in zip(gold, preds):
        conf[g][p] += 1
    predc = Counter(preds)
    f1s = []
    for g in LABEL_ORDER:
        ng = sum(conf[g].values())
        rec_ = conf[g][g] / ng if ng else 0
        pre_ = conf[g][g] / predc[g] if predc[g] else 0
        f1s.append(2 * pre_ * rec_ / (pre_ + rec_) if pre_ + rec_ else 0)
    fixed = sum(1 for g, b, p in zip(gold, base, preds) if b != g and p == g)
    broke = sum(1 for g, b, p in zip(gold, base, preds) if b == g and p != g)
    m = {"n": n, "accuracy": round(sum(g == p for g, p in zip(gold, preds)) / n, 5),
         "macro_f1": round(sum(f1s) / 2, 5), "confusion_gold_by_pred": conf,
         "fixed": fixed, "broke": broke, "net": fixed - broke,
         "nt_tc": conf["no_tool"]["tool_call"], "tc_nt": conf["tool_call"]["no_tool"],
         "false_toolcall_rate": round(conf["no_tool"]["tool_call"] /
                                      max(sum(conf["no_tool"].values()), 1), 5),
         "false_notool_rate": round(conf["tool_call"]["no_tool"] /
                                    max(sum(conf["tool_call"].values()), 1), 5),
         "true_toolcall_damage": sum(1 for g, b, p in zip(gold, base, preds)
                                     if g == "tool_call" and b == g and p != g),
         "true_notool_damage": sum(1 for g, b, p in zip(gold, base, preds)
                                   if g == "no_tool" and b == g and p != g)}
    if ch_fires is not None:
        m["n_touched"] = int(sum(len(v) for v in ch_fires.values()))
        m["router_fire_rate"] = round(m["n_touched"] / n, 4)
        m["touched_correct_damage"] = sum(
            1 for i, g, b, p in zip(idxs, gold, base, preds)
            if b == g and p != g and any(i in v for v in ch_fires.values()))
        m["per_channel_fired"] = {c: len(v) for c, v in ch_fires.items()}
    return m


def run_seed(seed, recs, acts, base_pred, split_of, model, tok, cand_ids_map,
             out_dir, do_controls, locked_from=None):
    """Fit channels, val-tune (or reuse locked cfg), run locked arms. Returns record."""
    idx = {s: [i for i, r in enumerate(recs) if split_of[r["sample_id"]] == s]
           for s in ("train", "val", "test")}
    fitted, reports = {}, {}
    for ch in CHANNELS:
        cfg, rep = fit_channel(ch, recs, acts, base_pred, split_of)
        fitted[ch], reports[ch] = cfg, rep

    # ── val sweep per eligible channel ──
    locked = {}
    sweep_rows = []
    for ch, cfg in fitted.items():
        if cfg is None:
            continue
        side = cfg["side"]; probs = cfg["probs_all"]
        elig_val = [i for i in idx["val"] if base_pred[i] == side
                    and probs[i] >= min(THRESHOLDS)]
        log.info("[seed %d] %s obs=L%d auc=%.3f cos=%.2f | val precompute pool=%d",
                 seed, ch, cfg["obs"], cfg["router_val_auc"], cfg["cos_dm_pc1"],
                 len(elig_val))
        best = None
        for dname, unit in cfg["dir_cands"].items():
            for inj in [cfg["obs"], cfg["obs"] - 2, cfg["obs"] - 4]:
                for alpha in ALPHAS:
                    corr = (alpha * cfg["median_norm"] * unit).astype(np.float32)
                    cpred = {i: predict_hooked(model, tok, recs[i], corr, inj, cand_ids_map)
                             for i in elig_val}
                    for thr in THRESHOLDS:
                        preds = []
                        fires = [i for i in elig_val if probs[i] >= thr]
                        for i in idx["val"]:
                            preds.append(cpred[i] if i in fires else base_pred[i])
                        m = arm_metrics(recs, idx["val"], base_pred, preds,
                                        {ch: set(fires)})
                        row = {"channel": ch, "dir": dname, "inj": inj, "alpha": alpha,
                               "thr": thr, **{k: m[k] for k in
                               ("fixed", "broke", "net", "n_touched", "nt_tc", "tc_nt")}}
                        sweep_rows.append(row)
                        key = (m["net"], -m["broke"], -alpha)
                        if best is None or key > best[0]:
                            best = (key, row, corr, dname, inj, alpha, thr)
        _, brow, corr, dname, inj, alpha, thr = best
        locked[ch] = {"obs": cfg["obs"], "inj": inj, "alpha": alpha, "thr": thr,
                      "dir": dname, "corr": corr, "unit": cfg["dir_cands"][dname],
                      "median_norm": cfg["median_norm"], "probs": cfg["probs_all"],
                      "side": cfg["side"], "router_val_auc": cfg["router_val_auc"],
                      "dm_norm": cfg["dm_norm"], "cos_dm_pc1": cfg["cos_dm_pc1"],
                      "val_best": brow}
        log.info("[seed %d] LOCKED %s: %s inj=L%d a=%.1f thr=%.1f (val net %+d)",
                 seed, ch, dname, inj, alpha, thr, brow["net"])
    if locked_from:
        pass  # (reserved) multiseed refits per seed per protocol

    # ── partition check (val+test eligibility overlap) ──
    part = {}
    for sname in ("val", "test"):
        el = {ch: {i for i in idx[sname] if base_pred[i] == locked[ch]["side"]
                   and locked[ch]["probs"][i] >= locked[ch]["thr"]}
              for ch in locked}
        both = set.intersection(*el.values()) if len(el) == 2 else set()
        neither = [i for i in idx[sname] if all(i not in v for v in el.values())]
        part[sname] = {ch: len(v) for ch, v in el.items()}
        part[sname].update({"overlap": len(both), "neither": len(neither)})

    # ── locked test arms ──
    test_idx = idx["test"]
    arms = {"A_baseline": arm_metrics(recs, test_idx, base_pred,
                                      [base_pred[i] for i in test_idx])}
    cpred_test = {}
    for ch, L in locked.items():
        fires = [i for i in test_idx if base_pred[i] == L["side"] and L["probs"][i] >= L["thr"]]
        cpred_test[ch] = {i: predict_hooked(model, tok, recs[i], L["corr"], L["inj"],
                                            cand_ids_map) for i in fires}
    def stack(chs, order):
        preds, fired = [], {c: set(cpred_test[c]) for c in chs}
        for i in test_idx:
            p = base_pred[i]
            for c in order:
                if c in chs and i in cpred_test[c]:
                    p = cpred_test[c][i]
            preds.append(p)
        return preds, fired
    if "nt_tc" in locked:
        p, f = stack(["nt_tc"], ["nt_tc"])
        arms["B_overcall_only"] = arm_metrics(recs, test_idx, base_pred, p, f)
    else:
        arms["B_overcall_only"] = {"NOT_RUN": reports["nt_tc"]["reason"]}
    if "tc_nt" in locked:
        p, f = stack(["tc_nt"], ["tc_nt"])
        arms["C_undercall_only"] = arm_metrics(recs, test_idx, base_pred, p, f)
    else:
        arms["C_undercall_only"] = {"NOT_RUN": reports["tc_nt"]["reason"]}
    if len(locked) == 2:
        p1, f = stack(["nt_tc", "tc_nt"], ["nt_tc", "tc_nt"])
        p2, _ = stack(["nt_tc", "tc_nt"], ["tc_nt", "nt_tc"])
        arms["D_dual"] = arm_metrics(recs, test_idx, base_pred, p1, f)
        arms["D_dual"]["order_invariant"] = p1 == p2
        arms["D_dual"]["additivity"] = {
            "net_B": arms["B_overcall_only"]["net"], "net_C": arms["C_undercall_only"]["net"],
            "net_D": arms["D_dual"]["net"],
            "additive": arms["D_dual"]["net"] ==
                        arms["B_overcall_only"]["net"] + arms["C_undercall_only"]["net"]}
    elif locked:
        arms["D_dual"] = {"NOT_RUN": "only one eligible channel"}

    rec_out = {"seed": seed, "reports": reports, "partition_check": part,
               "locked": {ch: {k: v for k, v in L.items()
                               if k not in ("corr", "unit", "probs")}
                          for ch, L in locked.items()},
               "arms": arms}

    # ── controls (pilot only) ──
    if do_controls and locked:
        PL = out_dir / "placebos"; PL.mkdir(parents=True, exist_ok=True)
        rev, rnd, ung = {}, {}, {}
        for ch, L in locked.items():
            fires = list(cpred_test[ch])
            # reverse
            corr_r = (-L["corr"]).astype(np.float32)
            pr = {i: predict_hooked(model, tok, recs[i], corr_r, L["inj"], cand_ids_map)
                  for i in fires}
            preds = [pr.get(i, base_pred[i]) for i in test_idx]
            rev[ch] = arm_metrics(recs, test_idx, base_pred, preds, {ch: set(fires)})
            # random distribution
            nets = []
            mag = float(np.linalg.norm(L["corr"]))
            for d in range(N_RANDOM):
                g = np.random.RandomState(1000 + d).randn(L["corr"].shape[0]).astype(np.float32)
                g = g / np.linalg.norm(g) * mag
                pd_ = {i: predict_hooked(model, tok, recs[i], g, L["inj"], cand_ids_map)
                       for i in fires}
                preds = [pd_.get(i, base_pred[i]) for i in test_idx]
                nets.append(arm_metrics(recs, test_idx, base_pred, preds)["net"])
                log.info("[random %s] draw %d net=%+d", ch, d, nets[-1])
            real_net = (arms["B_overcall_only"] if ch == "nt_tc"
                        else arms["C_undercall_only"])["net"]
            rnd[ch] = {"real_net": real_net, "draws": nets,
                       "mean": round(float(np.mean(nets)), 2),
                       "std": round(float(np.std(nets)), 2), "max": int(max(nets)),
                       "n_ge_real": int(sum(n >= real_net for n in nets)),
                       "real_percentile": round(100 * float(np.mean([n < real_net
                                                                     for n in nets])), 1)}
            # ungated (all same-side test samples)
            allside = [i for i in test_idx if base_pred[i] == L["side"]]
            pu = {i: (cpred_test[ch][i] if i in cpred_test[ch] else
                      predict_hooked(model, tok, recs[i], L["corr"], L["inj"], cand_ids_map))
                  for i in allside}
            preds = [pu.get(i, base_pred[i]) for i in test_idx]
            ung[ch] = arm_metrics(recs, test_idx, base_pred, preds, {ch: set(allside)})
        json.dump(rev, open(PL / "reverse_controls.json", "w"), indent=2)
        json.dump(rnd, open(PL / "random_direction_distribution.json", "w"), indent=2)
        json.dump({"gated": {c: (arms["B_overcall_only"] if c == "nt_tc"
                                 else arms["C_undercall_only"]) for c in locked},
                   "ungated": ung,
                   "note": "random distribution doubles as the perturbation-magnitude control "
                           "(matched norm, random geometry, same router/layer/threshold)"},
                  open(PL / "gating_decomposition.json", "w"), indent=2)
        rec_out["controls"] = {"reverse": rev, "random": rnd, "ungated": ung}

    return rec_out, sweep_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed", action="store_true")
    args = ap.parse_args()

    recs = load_jsonl(DATA_DIR / "metatool_binary_all.jsonl")
    sid2i = {r["sample_id"]: i for i, r in enumerate(recs)}
    bl = load_jsonl(BASE_DIR / "baseline/metatool_qwen7b_baseline_details.jsonl")
    base_pred = [None] * len(recs)
    for d in bl:
        base_pred[sid2i[d["sample_id"]]] = d["pred"]
    assert all(p is not None for p in base_pred), "baseline must cover all rows"
    acts = {L: np.load(CACHE / f"acts_L{L}.npy") for L in OBS_LAYERS}

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cand_ids_map = {l: tok.encode(t, add_special_tokens=False) for l, t in CANDIDATES.items()}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="cuda:0", trust_remote_code=True).eval()

    if not args.multiseed:
        # ── seed-42 pilot on the ARCHIVED split ──
        split_of = {}
        for s in ("train", "val", "test"):
            for r in load_jsonl(DATA_DIR / f"metatool_binary_{s}.jsonl"):
                split_of[r["sample_id"]] = s
        # verify regenerated seed-42 split vs archived
        regen = gen_split(recs, 42)
        same = sum(regen[k] == split_of[k] for k in split_of)
        log.info("seed-42 regen vs archived split: %d/%d identical", same, len(split_of))

        # discovery on the fresh baseline (Phase-1 pipeline)
        CD = BASE_DIR / "channel_discovery"; CD.mkdir(parents=True, exist_ok=True)
        rows = [{"gold": r["gold_response_mode"], "pred": base_pred[i],
                 "split": split_of[r["sample_id"]]} for i, r in enumerate(recs)]
        disc, _ = transition_discovery(rows, "metatool_binary_qwen25_7b",
                                       "Qwen2.5-7B-Instruct", {"no_tool__tool_call"},
                                       note="fresh bf16 baseline, native Yes/No readout")
        json.dump(disc, open(CD / "metatool_qwen7b_channel_discovery.json", "w"), indent=2)

        out, sweep = run_seed(42, recs, acts, base_pred, split_of, model, tok,
                              cand_ids_map, BASE_DIR, do_controls=True)
        PD = BASE_DIR / "pilot"; PD.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(PD / "pilot_summary.json", "w"), indent=2)
        with open(PD / "pilot_details.jsonl", "w") as f:
            for row in sweep:
                f.write(json.dumps(row) + "\n")
        log.info("pilot done -> %s", PD)
    else:
        MS = BASE_DIR / "multiseed"; MS.mkdir(parents=True, exist_ok=True)
        rows_out = []
        rowf = MS / "multiseed_rows.jsonl"
        done = set()
        if rowf.exists():
            done = {json.loads(l)["seed"] for l in open(rowf)}
        fout = open(rowf, "a")
        for seed in SEEDS_MULTI:
            if seed in done:
                continue
            split_of = gen_split(recs, seed)
            out, _ = run_seed(seed, recs, acts, base_pred, split_of, model, tok,
                              cand_ids_map, BASE_DIR, do_controls=False)
            fout.write(json.dumps(out) + "\n"); fout.flush()
            rows_out.append(out)
        fout.close()
        log.info("multiseed done -> %s", rowf)


if __name__ == "__main__":
    main()
