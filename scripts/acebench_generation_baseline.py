"""
acebench_generation_baseline.py — Phase 4 generation-based ACEBench readout (locked code path).
================================================================================================
One code path for all three stages (PHASE4_GENERATION_PROTOCOL.md):

  --smoke    Step 7: 48 train-only rows (12/mode: 6 EN + 6 ZH, lowest sample_id), full checklist
  --full     Step 9: all 800 rows once, metrics + committable summaries
  --replay   R0.6: re-generate the fixed 50-row audit subset, compare labels to the full run

Prompts: the OFFICIAL ACEBench special-data system prompt per language, loaded VERBATIM from the
saved official files (.cache/acebench_raw/official_prompt_{en,zh}.py; sha256 in the lock manifest),
`{function}` <- json.dumps(row.functions), `{time}` <- row.time; user message = official
USER_PROMPT_{EN,ZH} with the question restored to its raw "user: ..." form. Qwen chat template,
add_generation_prompt=True. Greedy, max_new_tokens=256, batch 1, ascending sample_id, bf16.

Raw generations stay in gitignored .cache/acebench_phase4/ (they embed dataset text). Committable
summaries -> final/results/acebench_generation_readout/baseline/. Gold is joined only after parsing.
"""
from __future__ import annotations
import argparse, gc, json, logging, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from acebench_parse_outputs import parse, PARSER_VERSION, LABELS  # noqa: E402

DATA = ROOT / ".cache/acebench_phase4/acebench_decision_all_with_time.jsonl"
PROMPT_EN = ROOT / ".cache/acebench_raw/official_prompt_en.py"
PROMPT_ZH = ROOT / ".cache/acebench_raw/official_prompt_zh.py"
MODEL_PATH = ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"
CACHE_OUT = ROOT / ".cache/acebench_phase4"
RES = ROOT / "final/results/acebench_generation_readout"
BASE = RES / "baseline"
DTYPE = torch.bfloat16
MAX_PROMPT_TOK = 8192
MAX_NEW_TOK = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("p4.gen")


def load_official_prompts():
    ns_en, ns_zh = {}, {}
    exec(compile(PROMPT_EN.read_text(encoding="utf-8"), str(PROMPT_EN), "exec"), ns_en)
    exec(compile(PROMPT_ZH.read_text(encoding="utf-8"), str(PROMPT_ZH), "exec"), ns_zh)
    return {"en": (ns_en["SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN"], ns_en["USER_PROMPT_EN"]),
            "zh": (ns_zh["SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH"], ns_zh["USER_PROMPT_ZH"])}


def build_messages(row, prompts):
    sys_t, user_t = prompts[row["lang"]]
    system = (sys_t.replace("{function}", json.dumps(row["functions"], ensure_ascii=False))
                   .replace("{time}", row.get("time") or ""))
    q = row["question"]
    if not q.startswith("user:"):
        q = "user: " + q
    user = user_t.replace("{question}", q)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE,
                                                 device_map="cuda:0").eval()
    dtypes = {str(p.dtype) for p in model.parameters()}
    quant = getattr(model.config, "quantization_config", None) is not None
    vram = torch.cuda.memory_reserved(0) / 1e9
    log.info("model loaded | dtypes=%s quantized=%s vram=%.2fGB layers=%d",
             dtypes, quant, vram, model.config.num_hidden_layers)
    assert dtypes == {"torch.bfloat16"} and not quant, "bf16-only requirement violated"
    return model, tok


def smoke_rows(rows):
    out = []
    for mode in ["tool_call", "ask_user", "flag_param_error", "cannot_comply"]:
        for lang in ["en", "zh"]:
            pick = [r for r in rows if r["gold_mode"] == mode and r["lang"] == lang
                    and r["split"] == "train"]
            pick.sort(key=lambda r: r["sample_id"])
            out.extend(pick[:6])
    return out


def replay_rows(rows):
    """Fixed 50-row deterministic audit subset (locked before the full run):
    per mode, lowest-sample_id rows: tool_call 10 EN + 10 ZH; each special 5 EN + 5 ZH."""
    out = []
    quota = {"tool_call": 10, "ask_user": 5, "flag_param_error": 5, "cannot_comply": 5}
    for mode, q in quota.items():
        for lang in ["en", "zh"]:
            pick = sorted([r for r in rows if r["gold_mode"] == mode and r["lang"] == lang],
                          key=lambda r: r["sample_id"])[:q]
            out.extend(pick)
    return sorted(out, key=lambda r: r["sample_id"])


@torch.no_grad()
def generate(model, tok, messages):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    n_in = ids.input_ids.shape[1]
    if n_in > MAX_PROMPT_TOK:
        return None, n_in, 0, "skipped_too_long"
    ids = {k: v.to(model.device) for k, v in ids.items()}
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOK, do_sample=False, num_beams=1,
                         pad_token_id=tok.pad_token_id)
    gen = out[0, n_in:]
    n_gen = gen.shape[0]
    finished = bool((gen == tok.eos_token_id).any()) or n_gen < MAX_NEW_TOK
    text = tok.decode(gen, skip_special_tokens=True)
    return text, n_in, n_gen, ("eos" if finished else "length")


def run(rows, model, tok, prompts, tag):
    detf = open(CACHE_OUT / f"generation_details_{tag}.jsonl", "w", encoding="utf-8")
    recs, t0 = [], time.time()
    for k, r in enumerate(rows):
        text, n_in, n_gen, status = generate(model, tok, build_messages(r, prompts))
        p = parse(text if text is not None else "")
        rec = {"sample_id": r["sample_id"], "lang": r["lang"], "split": r["split"],
               "gold": r["gold_mode"], "pred": p["label"], "rule_path": p["rule_path"],
               "ambiguous": p["ambiguous"], "conflict": p["conflict"],
               "unparseable_reason": p["unparseable_reason"], "calls": p["calls"],
               "n_prompt_tok": n_in, "n_gen_tok": n_gen, "finish": status,
               "output": text}
        recs.append(rec)
        detf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (k + 1) % 25 == 0:
            detf.flush()
            el = time.time() - t0
            log.info("%d/%d (%.2fs/row, ETA %.0f min)", k + 1, len(rows), el / (k + 1),
                     el / (k + 1) * (len(rows) - k - 1) / 60)
            gc.collect(); torch.cuda.empty_cache()
    detf.close()
    log.info("%s done: %d rows in %.1f min", tag, len(recs), (time.time() - t0) / 60)
    return recs


def metrics(recs):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    n = len(recs)
    golds = [r["gold"] for r in recs]
    preds = [r["pred"] for r in recs]
    acc = sum(g == p for g, p in zip(golds, preds)) / n
    per_class_recall = {}
    for lab in LABELS:
        g_idx = [i for i, g in enumerate(golds) if g == lab]
        per_class_recall[lab] = (sum(preds[i] == lab for i in g_idx) / len(g_idx)) if g_idx else None
    bal_acc = float(np.mean([v for v in per_class_recall.values() if v is not None]))
    macro_f1 = float(f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0))
    pr, rc, f1, sup = precision_recall_fscore_support(golds, preds, labels=LABELS,
                                                      zero_division=0)
    pred_labels = LABELS + ["UNKNOWN"]
    conf = {g: {p: 0 for p in pred_labels} for g in LABELS}
    for g, p in zip(golds, preds):
        conf[g][p if p in pred_labels else "UNKNOWN"] += 1
    pdist = Counter(preds)
    lens = sorted(r["n_gen_tok"] for r in recs)
    def pct(q):
        return lens[min(len(lens) - 1, int(q * len(lens)))]
    out = {
        "n": n, "accuracy": round(acc, 5), "balanced_accuracy": round(bal_acc, 5),
        "macro_f1": round(macro_f1, 5), "majority_gold_share": 0.625,
        "per_class": {lab: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4),
                            "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                      for i, lab in enumerate(LABELS)},
        "per_class_recall": {k: (round(v, 4) if v is not None else None)
                             for k, v in per_class_recall.items()},
        "prediction_distribution": dict(pdist),
        "max_pred_share": round(max(pdist.values()) / n, 4),
        "unknown_rate": round(pdist.get("UNKNOWN", 0) / n, 4),
        "ambiguous_rate": round(sum(r["ambiguous"] for r in recs) / n, 4),
        "conflict_rate": round(sum(r["conflict"] for r in recs) / n, 4),
        "skipped": sum(r["finish"] == "skipped_too_long" for r in recs),
        "empty_rate": round(sum(not (r["output"] or "").strip() for r in recs) / n, 4),
        "eos_rate": round(sum(r["finish"] == "eos" for r in recs) / n, 4),
        "gen_len": {"mean": round(float(np.mean(lens)), 1), "p50": pct(0.5), "p90": pct(0.9),
                    "p99": pct(0.99), "max": lens[-1]},
        "confusion_gold_by_pred": conf,
        "parser_version": PARSER_VERSION,
    }
    for key, fn in [("per_lang", lambda r: r["lang"]), ("per_split", lambda r: r["split"])]:
        sub = {}
        for v in sorted({fn(r) for r in recs}):
            rs = [r for r in recs if fn(r) == v]
            sub[v] = {"n": len(rs),
                      "acc": round(sum(r["gold"] == r["pred"] for r in rs) / len(rs), 4),
                      "unknown_rate": round(sum(r["pred"] == "UNKNOWN" for r in rs) / len(rs), 4)}
        out[key] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "full", "replay"])
    a = ap.parse_args()
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    prompts = load_official_prompts()
    import transformers
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "dtype": "bfloat16", "gen_params": {"do_sample": False, "num_beams": 1,
                                               "max_new_tokens": MAX_NEW_TOK},
           "model_path": str(MODEL_PATH.relative_to(ROOT)), "parser_version": PARSER_VERSION}

    model, tok = load_model()
    if a.stage == "smoke":
        rs = smoke_rows(rows)
        recs = run(rs, model, tok, prompts, "smoke")
        m = metrics(recs)
        outs = [r["output"] for r in recs]
        checks = {
            "n": len(recs), "oom_or_skipped": m["skipped"],
            "parser_non_unknown_rate": round(1 - m["unknown_rate"], 4),
            "eos_rate": m["eos_rate"],
            "max_identical_output_share": round(max(Counter(outs).values()) / len(outs), 4),
            "gate": {"loads_bf16": True, "no_oom": m["skipped"] == 0,
                     "non_unknown_ge_0.70": (1 - m["unknown_rate"]) >= 0.70,
                     "eos_ge_0.90": m["eos_rate"] >= 0.90,
                     "no_boilerplate_collapse": max(Counter(outs).values()) / len(outs) <= 0.50},
        }
        checks["gate"]["PASS"] = all(v for k, v in checks["gate"].items())
        json.dump({"env": env, "metrics": m, "checks": checks},
                  open(RES / "smoke_test_report.json", "w"), indent=2, ensure_ascii=False)
        log.info("SMOKE GATE: %s | %s", "PASS" if checks["gate"]["PASS"] else "FAIL",
                 json.dumps(checks["gate"]))
    elif a.stage == "full":
        recs = run(rows, model, tok, prompts, "full")
        m = metrics(recs)
        json.dump({"env": env, **m}, open(BASE / "acebench_generation_baseline_summary.json", "w"),
                  indent=2, ensure_ascii=False)
        with open(BASE / "acebench_generation_confusion_matrix.csv", "w") as f:
            cols = LABELS + ["UNKNOWN"]
            f.write("gold\\pred," + ",".join(cols) + "\n")
            for g in LABELS:
                f.write(g + "," + ",".join(str(m["confusion_gold_by_pred"][g][c])
                                           for c in cols) + "\n")
        with open(BASE / "acebench_generation_per_label_metrics.csv", "w") as f:
            f.write("label,precision,recall,f1,support\n")
            for lab, d in m["per_class"].items():
                f.write(f"{lab},{d['precision']},{d['recall']},{d['f1']},{d['support']}\n")
        with open(BASE / "acebench_generation_prediction_distribution.csv", "w") as f:
            f.write("pred,count\n")
            for k, v in sorted(m["prediction_distribution"].items()):
                f.write(f"{k},{v}\n")
        log.info("FULL: acc=%.4f bal=%.4f macroF1=%.4f unknown=%.3f maxshare=%.3f",
                 m["accuracy"], m["balanced_accuracy"], m["macro_f1"], m["unknown_rate"],
                 m["max_pred_share"])
    else:  # replay
        full = {json.loads(l)["sample_id"]: json.loads(l)["pred"]
                for l in open(CACHE_OUT / "generation_details_full.jsonl", encoding="utf-8")}
        rs = replay_rows(rows)
        recs = run(rs, model, tok, prompts, "replay")
        same = sum(full[r["sample_id"]] == r["pred"] for r in recs)
        rep = {"n": len(recs), "identical_labels": same,
               "agreement": round(same / len(recs), 4),
               "diffs": [{"sample_id": r["sample_id"], "full": full[r["sample_id"]],
                          "replay": r["pred"]} for r in recs
                         if full[r["sample_id"]] != r["pred"]]}
        json.dump(rep, open(BASE / "replay_determinism_check.json", "w"), indent=2)
        log.info("REPLAY agreement: %.4f (%d/%d)", rep["agreement"], same, len(recs))


if __name__ == "__main__":
    main()
