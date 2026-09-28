"""
metatool_qwen7b_baseline.py — Phase 2 R0: fresh Qwen2.5-7B MetaTool-Binary baseline (bf16).
============================================================================================
Native readout ONLY (locked in PHASE2_PROTOCOL.md): MetaTool `thought_prompt` as the user message
(system = "You are a helpful assistant.", Qwen chat template, add_generation_prompt=True) scored
against the single-token candidates  tool_call -> "Yes",  no_tool -> "No"  by avg logprob.
Alternative candidate wordings are banned (archived candidate-sensitivity audit: they collapse).

Evaluates ALL 1040 rows (baseline is split-independent); R0 validity is assessed on the archived
seed-42 split (728/156/156). Includes the R0 internal-stability check (50-row deterministic
re-score) and margin statistics.

Outputs -> final/results/metatool_qwen7b_sakiko_ca/baseline/
Usage:   python scripts/metatool_qwen7b_baseline.py
"""
from __future__ import annotations
import json, sys, time, logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/metatool_binary"
OUT = ROOT / "final/results/metatool_qwen7b_sakiko_ca/baseline"
MODEL_PATH = ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"
DTYPE = torch.bfloat16
MAX_PROMPT_TOKENS = 8192

CANDIDATES = {"tool_call": "Yes", "no_tool": "No"}
LABEL_ORDER = ["tool_call", "no_tool"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("r0")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_prompt_text(thought_prompt: str, tok) -> str:
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": thought_prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_row(model, tok, prompt_ids, cand_ids_map, device):
    scored = {}
    for lbl, cids in cand_ids_map.items():
        ids = torch.cat([prompt_ids,
                         torch.tensor([cids], device=device)], dim=1)
        logits = model(ids).logits
        pl, cl = prompt_ids.shape[1], len(cids)
        lp = torch.log_softmax(logits[0, pl - 1: pl + cl - 1, :].float(), dim=-1)
        tk = [lp[i, cids[i]].item() for i in range(cl)]
        scored[lbl] = {"sum_logp": sum(tk), "avg_logp": sum(tk) / cl, "n_tokens": cl}
    return scored


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_jsonl(DATA_DIR / "metatool_binary_all.jsonl")
    split_of = {}
    for s in ("train", "val", "test"):
        for r in load_jsonl(DATA_DIR / f"metatool_binary_{s}.jsonl"):
            split_of[r["sample_id"]] = s
    log.info("rows=%d | split sizes=%s", len(recs),
             dict(Counter(split_of.values())))

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cand_ids_map = {lbl: tok.encode(txt, add_special_tokens=False)
                    for lbl, txt in CANDIDATES.items()}
    log.info("candidate tokenization: %s",
             {l: ids for l, ids in cand_ids_map.items()})
    assert all(len(v) == 1 for v in cand_ids_map.values()), "candidates must be single tokens"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="cuda:0",
        trust_remote_code=True).eval()
    device = model.device
    log.info("model loaded bf16 | dtype=%s", next(model.parameters()).dtype)

    details, skipped = [], 0
    t0 = time.time()
    for i, r in enumerate(recs):
        prompt_text = build_prompt_text(r["thought_prompt"], tok)
        pids = tok.encode(prompt_text, add_special_tokens=False)
        if len(pids) > MAX_PROMPT_TOKENS:
            skipped += 1
            continue
        prompt_ids = torch.tensor([pids], device=device)
        scored = score_row(model, tok, prompt_ids, cand_ids_map, device)
        del prompt_ids
        pred = max(LABEL_ORDER, key=lambda lb: scored[lb]["avg_logp"])
        details.append({
            "sample_id": r["sample_id"], "split": split_of[r["sample_id"]],
            "gold": r["gold_response_mode"], "pred": pred,
            "correct": pred == r["gold_response_mode"],
            "margin": round(scored["tool_call"]["avg_logp"] - scored["no_tool"]["avg_logp"], 5),
            "scored": {l: {k: round(v, 5) for k, v in d.items()} for l, d in scored.items()},
        })
        if (i + 1) % 200 == 0:
            log.info("%d/%d (%.1f min)", i + 1, len(recs), (time.time() - t0) / 60)

    with open(OUT / "metatool_qwen7b_baseline_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ── R0 stability check: deterministic 50-row re-score ──
    stable = 0
    for d in details[:50]:
        r = next(x for x in recs if x["sample_id"] == d["sample_id"])
        prompt_ids = torch.tensor([tok.encode(build_prompt_text(r["thought_prompt"], tok),
                                              add_special_tokens=False)], device=device)
        s2 = score_row(model, tok, prompt_ids, cand_ids_map, device)
        p2 = max(LABEL_ORDER, key=lambda lb: s2[lb]["avg_logp"])
        stable += int(p2 == d["pred"])
        del prompt_ids
    log.info("re-score stability: %d/50 identical", stable)

    # ── metrics ──
    def block(rows):
        n = len(rows)
        acc = sum(d["correct"] for d in rows) / n
        conf = {g: {p: 0 for p in LABEL_ORDER} for g in LABEL_ORDER}
        for d in rows:
            conf[d["gold"]][d["pred"]] += 1
        f1s = []
        predc = Counter(d["pred"] for d in rows)
        for g in LABEL_ORDER:
            ng = sum(conf[g].values())
            rec_ = conf[g][g] / ng if ng else 0
            pre_ = conf[g][g] / predc[g] if predc[g] else 0
            f1s.append(2 * pre_ * rec_ / (pre_ + rec_) if pre_ + rec_ else 0)
        return {"n": n, "accuracy": round(acc, 5), "macro_f1": round(sum(f1s) / 2, 5),
                "confusion_gold_by_pred": conf,
                "pred_distribution": dict(predc),
                "gold_distribution": dict(Counter(d["gold"] for d in rows)),
                "nt_to_tc_errors": conf["no_tool"]["tool_call"],
                "tc_to_nt_errors": conf["tool_call"]["no_tool"],
                "false_toolcall_rate": round(conf["no_tool"]["tool_call"] /
                                             max(sum(conf["no_tool"].values()), 1), 5),
                "false_notool_rate": round(conf["tool_call"]["no_tool"] /
                                           max(sum(conf["tool_call"].values()), 1), 5)}

    overall = block(details)
    per_split = {s: block([d for d in details if d["split"] == s])
                 for s in ("train", "val", "test")}
    margins = np.array([abs(d["margin"]) for d in details])
    valid_scores = all(np.isfinite([d["scored"][l]["avg_logp"] for l in LABEL_ORDER]).all()
                       for d in details)

    # ── R0 verdict (criteria locked in PHASE2_PROTOCOL.md §8) ──
    maj_gold = max(overall["gold_distribution"].values()) / overall["n"]
    maj_pred = max(overall["pred_distribution"].values()) / overall["n"]
    err_support = {
        "no_tool__tool_call": {s: per_split[s]["nt_to_tc_errors"] for s in per_split},
        "tool_call__no_tool": {s: per_split[s]["tc_to_nt_errors"] for s in per_split},
    }
    def supported(ch):
        return (err_support[ch]["train"] >= 50 and
                min(err_support[ch]["val"], err_support[ch]["test"]) >= 15)
    r0 = {
        "1_acc_gt_majority_by_005": overall["accuracy"] >= maj_gold + 0.05,
        "2_no_collapse": not (maj_pred > 0.60 and overall["accuracy"] < maj_gold),
        "3_scores_valid": bool(valid_scores),
        "4_error_transition_support": {c: supported(c) for c in err_support},
        "4_pass": any(supported(c) for c in err_support),
        "5_readout_stable": {"rescore_identical_of_50": stable,
                             "pass": stable == 50,
                             "nonzero_margin_share": round(float((margins > 0).mean()), 4)},
        "6_no_leakage": "prompt = thought_prompt only; gold never in prompt/router/gate",
    }
    r0_pass = (r0["1_acc_gt_majority_by_005"] and r0["2_no_collapse"] and
               r0["3_scores_valid"] and r0["4_pass"] and r0["5_readout_stable"]["pass"])

    summary = {
        "model": "Qwen2.5-7B-Instruct", "precision": "bf16",
        "model_path": ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct",
        "readout": "native thought_prompt + Yes/No single-token avg_logp",
        "candidate_tokenization": {l: ids for l, ids in cand_ids_map.items()},
        "n_total": len(recs), "n_evaluated": len(details), "skipped_long_or_oom": skipped,
        "majority_gold_share": round(maj_gold, 4),
        "majority_pred_share": round(maj_pred, 4),
        "overall": overall, "per_split": per_split,
        "error_transition_support": err_support,
        "margin_stats": {"mean_abs": round(float(margins.mean()), 4),
                         "p05_abs": round(float(np.percentile(margins, 5)), 4)},
        "R0_checks": r0, "R0_PASS": bool(r0_pass),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    json.dump(summary, open(OUT / "metatool_qwen7b_baseline_summary.json", "w"),
              indent=2, ensure_ascii=False)
    log.info("acc=%.4f  nt->tc=%d  tc->nt=%d  R0_PASS=%s  (%.1f min)",
             overall["accuracy"], overall["nt_to_tc_errors"],
             overall["tc_to_nt_errors"], r0_pass, summary["runtime_min"])


if __name__ == "__main__":
    main()
