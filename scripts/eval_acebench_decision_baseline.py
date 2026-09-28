"""
eval_acebench_decision_baseline.py — Qwen2.5-7B BASELINE ONLY on the ACEBench decision task.
=============================================================================================
Minimal new-benchmark pilot (Phase 1, section G): baseline + error decomposition. NO SAKIKO
intervention, NO activation extraction, NO tuning. Mirrors the archived W2C/MetaTool protocol:
score 4 neutral candidate responses by average token log-probability; pred = argmax.

Label-leakage control: the prompt contains ONLY the function schemas + the user question + the four
candidate response strings (every mode described symmetrically); no ACEBench category vocabulary.

Candidate-sensitivity control (MetaTool lesson): a second English phrasing set is scored on the
first 200 EN rows; agreement and accuracy are reported.

Outputs -> final/results/cross_dataset_channel_discovery/new_benchmark_pilot/
Usage:   python scripts/eval_acebench_decision_baseline.py
"""
from __future__ import annotations
import json, sys, time, logging
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/acebench_decision/acebench_decision_all.jsonl"
OUT = ROOT / "final/results/cross_dataset_channel_discovery/new_benchmark_pilot"
MODEL_PATH = ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"
DTYPE = torch.bfloat16
MAX_LEN = 4096

MODES = ["tool_call", "ask_user", "flag_param_error", "cannot_comply"]

CANDIDATES = {
    "en": {
        "tool_call": "I will call one of the provided functions now with the appropriate arguments taken from the request.",
        "ask_user": "I cannot call a function yet because a required piece of information is missing, so I will ask the user to provide it.",
        "flag_param_error": "I will point out that one of the provided argument values is invalid because it violates the function's allowed format or range.",
        "cannot_comply": "I will explain that none of the provided functions can accomplish this request, so I cannot complete it.",
    },
    "zh": {
        "tool_call": "我现在会根据请求中的信息，使用合适的参数调用所提供的某个函数。",
        "ask_user": "由于缺少必需的信息，我暂时无法调用函数，我会先向用户询问缺失的内容。",
        "flag_param_error": "我会指出请求中提供的某个参数值不符合函数要求的格式或取值范围，是无效的。",
        "cannot_comply": "我会说明所提供的函数都无法完成这个请求，因此我无法完成它。",
    },
}
# alternative EN phrasing for the candidate-sensitivity check (first 200 EN rows)
CANDIDATES_ALT_EN = {
    "tool_call": "Call a function with the given arguments.",
    "ask_user": "Ask the user for the missing required information first.",
    "flag_param_error": "Point out that a given argument value is invalid.",
    "cannot_comply": "State that the available functions cannot handle this request.",
}

SYSTEM = ("You are an assistant with access to the following functions:\n{funcs}\n"
          "Read the user's request and decide how you should respond.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("pilot")


def build_prompt(tok, r):
    funcs = json.dumps(r["functions"], ensure_ascii=False)
    msgs = [{"role": "system", "content": SYSTEM.format(funcs=funcs)},
            {"role": "user", "content": r["question"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_candidates(model, tok, prompt, cands):
    """avg logprob of each candidate continuation given the prompt (one forward per candidate)."""
    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN - 128).input_ids
    out = {}
    for mode, text in cands.items():
        c_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([p_ids, c_ids], dim=1).to(model.device)
        logits = model(ids).logits[0]
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        tgt = ids[0, 1:]
        n_c = c_ids.shape[1]
        cand_lp = lp[-n_c:, :].gather(1, tgt[-n_c:].unsqueeze(1)).squeeze(1)
        out[mode] = {"avg_logp": round(float(cand_lp.mean()), 5), "n_tokens": n_c}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    log.info("rows=%d | loading %s (bf16)", len(rows), MODEL_PATH.name)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE,
                                                 device_map="cuda").eval()

    detf = open(OUT / "acebench_pilot_baseline_details.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    for i, r in enumerate(rows):
        prompt = build_prompt(tok, r)
        scored = score_candidates(model, tok, prompt, CANDIDATES[r["lang"]])
        pred = max(scored, key=lambda m: scored[m]["avg_logp"])
        rec = {"sample_id": r["sample_id"], "lang": r["lang"], "split": r["split"],
               "source_file": r["source_file"], "gold": r["gold_mode"], "pred": pred,
               "correct": pred == r["gold_mode"], "scored": scored}
        detf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            detf.flush()
            log.info("%d/%d (%.1f min elapsed)", i + 1, len(rows), (time.time() - t0) / 60)
    detf.close()

    # candidate-sensitivity: alternative EN phrasing on first 200 EN rows
    en_rows = [r for r in rows if r["lang"] == "en"][:200]
    sens = []
    for r in en_rows:
        prompt = build_prompt(tok, r)
        scored = score_candidates(model, tok, prompt, CANDIDATES_ALT_EN)
        sens.append({"sample_id": r["sample_id"], "gold": r["gold_mode"],
                     "pred_alt": max(scored, key=lambda m: scored[m]["avg_logp"])})
    json.dump(sens, open(OUT / "acebench_pilot_candidate_sensitivity.json", "w"), indent=1)

    # summary
    det = [json.loads(l) for l in open(OUT / "acebench_pilot_baseline_details.jsonl", encoding="utf-8")]
    acc = sum(d["correct"] for d in det) / len(det)
    conf = {g: {p: 0 for p in MODES} for g in MODES}
    for d in det:
        conf[d["gold"]][d["pred"]] += 1
    per_lang = {}
    for lang in ("en", "zh"):
        sub = [d for d in det if d["lang"] == lang]
        per_lang[lang] = {"n": len(sub), "acc": round(sum(d["correct"] for d in sub) / len(sub), 4)}
    recalls = []
    for g in MODES:
        n_g = sum(conf[g].values())
        if n_g:
            recalls.append(conf[g][g] / n_g)
    pred_c = Counter(d["pred"] for d in det)
    precs = []
    for g in MODES:
        if pred_c[g]:
            precs.append(conf[g][g] / pred_c[g])
    macro_f1 = None
    f1s = []
    for g in MODES:
        n_g = sum(conf[g].values())
        rec_ = conf[g][g] / n_g if n_g else 0
        pre_ = conf[g][g] / pred_c[g] if pred_c[g] else 0
        f1s.append(2 * pre_ * rec_ / (pre_ + rec_) if (pre_ + rec_) else 0)
    macro_f1 = sum(f1s) / len(f1s)

    alt_map = {s["sample_id"]: s["pred_alt"] for s in sens}
    both = [(d["pred"], alt_map[d["sample_id"]], d["gold"]) for d in det if d["sample_id"] in alt_map]
    agree = sum(1 for a, b, _ in both if a == b) / len(both)
    acc_alt = sum(1 for _, b, g in both if b == g) / len(both)
    acc_main_sub = sum(1 for a, _, g in both if a == g) / len(both)

    summary = {
        "model": "Qwen2.5-7B-Instruct (bf16)", "n": len(det),
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_lang": per_lang, "confusion_gold_by_pred": conf,
        "pred_distribution": dict(pred_c),
        "per_class_recall": {g: round(conf[g][g] / sum(conf[g].values()), 4) for g in MODES},
        "candidate_sensitivity": {"n": len(both), "agreement_main_vs_alt": round(agree, 4),
                                  "acc_main_on_subset": round(acc_main_sub, 4),
                                  "acc_alt_on_subset": round(acc_alt, 4)},
        "runtime_min": round((time.time() - t0) / 60, 1),
        "protocol_note": "4-way candidate scoring by avg token logprob; language-matched candidates; "
                         "no gold-category vocabulary in prompts (leakage control)",
    }
    json.dump(summary, open(OUT / "acebench_pilot_baseline_summary.json", "w"),
              indent=2, ensure_ascii=False)
    log.info("acc=%.4f macro_f1=%.4f | sens agree=%.3f | %.1f min",
             acc, macro_f1, agree, summary["runtime_min"])


if __name__ == "__main__":
    main()
