"""
eval_w2c_7b_baseline.py — W2C MCQ avg_logp baseline for a ~7B model
====================================================================
Standalone replica of src/phase1/w2c_mcq_eval.py's scoring, taking a model
path/id DIRECTLY (no configs/models.yaml edit, no W2C file touched). Used to
baseline a recent ~7B open-weight model on When2Call and decompose its error
channels, before any SAKIKO intervention.

Scoring (identical protocol to the Phi-3.5 SAKIKO baseline):
  - When2Call test/mcq, shuffled with seed=42
  - 4 candidate response modes: direct / tool_call / request_for_info / cannot_answer
  - length-normalised log-prob (avg_logp) of each candidate; argmax = prediction
  - NO activation extraction, NO steering, NO router, NO intervention

Error decomposition channels (gold ∈ {tool_call, request_for_info, cannot_answer};
direct is a distractor, never gold):
  rfi_tc    : gold request_for_info, pred tool_call
  ca_tc     : gold cannot_answer,    pred tool_call
  ca_direct : gold cannot_answer,    pred direct
  other     : every remaining misclassification

Usage:
  /path/to/project/.venv/bin/python scripts/eval_w2c_7b_baseline.py \\
      --model_path /path/to/Qwen2.5-7B-Instruct \\
      --model_name qwen25_7b \\
      [--dtype bfloat16] [--max_samples 0] [--max_prompt_tokens 8192]
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
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_ORDER = ["direct", "tool_call", "request_for_info", "cannot_answer"]
GOLD_CLASSES = ["tool_call", "request_for_info", "cannot_answer"]  # direct never gold

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "final" / "results" / "7b_w2c_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("w2c_7b")


def build_prompt_messages(question: str, tools: list) -> list[dict]:
    system_parts = ["You are a helpful assistant."]
    if tools:
        system_parts.append("You have access to the following tools:\n"
                             + json.dumps(tools, indent=2, ensure_ascii=False))
    else:
        system_parts.append("No tools are available.")
    system_parts.append("Given the user's question, choose the most appropriate "
                         "response from the provided options.")
    return [{"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": question}]


def score_candidate(model, tokenizer, prompt_ids, candidate_text, device) -> float:
    cand_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    if len(cand_ids) == 0:
        return -1e9
    cand_tensor = torch.tensor([cand_ids], device=device)
    input_ids = torch.cat([prompt_ids, cand_tensor], dim=1)
    with torch.no_grad():
        logits = model(input_ids).logits
    prompt_len = prompt_ids.shape[1]
    cand_len = len(cand_ids)
    logits_slice = logits[0, prompt_len - 1: prompt_len + cand_len - 1, :]
    log_probs = torch.log_softmax(logits_slice.float(), dim=-1)
    token_logps = [log_probs[i, cand_ids[i]].item() for i in range(cand_len)]
    return sum(token_logps) / cand_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_prompt_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    log.info("Model: %s  (%s, dtype=%s)", args.model_name, args.model_path, args.dtype)
    log.info("Loading When2Call test/mcq …")
    ds = load_dataset("nvidia/When2Call", "test", split="mcq")
    ds = ds.shuffle(seed=args.seed)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Samples: %d", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda:0",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    vram = torch.cuda.memory_reserved(0) / 1e9
    log.info("Loaded in %.1fs | device=%s | layers=%d hidden=%d | VRAM reserved=%.2fGB",
             time.time() - t0, device, model.config.num_hidden_layers,
             model.config.hidden_size, vram)

    preds, golds, records = [], [], []
    skipped = 0
    t1 = time.time()
    for idx, sample in enumerate(tqdm(ds, desc="W2C-MCQ-7B", ncols=100)):
        uuid = sample.get("uuid", f"idx_{idx}")
        question = sample["question"]
        tools = sample["tools"]
        answers = sample["answers"]
        gold = sample["correct_answer"]

        parsed_tools = []
        if tools:
            for t in tools:
                if isinstance(t, str):
                    try:
                        parsed_tools.append(json.loads(t))
                    except json.JSONDecodeError:
                        parsed_tools.append({"raw": t})
                else:
                    parsed_tools.append(t)

        messages = build_prompt_messages(question, parsed_tools)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = ""
            for m in messages:
                prompt_text += f"<|{m['role']}|>\n{m['content']}\n"
            prompt_text += "<|assistant|>\n"

        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_token_ids) > args.max_prompt_tokens:
            skipped += 1
            continue

        scored = {}
        try:
            prompt_ids = torch.tensor([prompt_token_ids], device=device)
            for label in LABEL_ORDER:
                cand = answers.get(label, "")
                scored[label] = (score_candidate(model, tokenizer, prompt_ids, cand, device)
                                 if cand else -1e9)
            del prompt_ids
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("OOM on %s (len=%d) — skipping", uuid, len(prompt_token_ids))
                skipped += 1
                gc.collect(); torch.cuda.empty_cache()
                continue
            raise

        pred = max(LABEL_ORDER, key=lambda lb: scored[lb])
        preds.append(pred); golds.append(gold)
        records.append({
            "uuid": uuid, "gold": gold, "pred": pred,
            "correct": bool(pred == gold),
            "tools_empty": (len(tools) == 0),
            "avg_logp": {k: round(v, 4) for k, v in scored.items()},
        })
        if (idx + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()

    elapsed = time.time() - t1
    log.info("Eval done in %.1fs (%.2fs/sample); skipped=%d", elapsed,
             elapsed / max(len(records), 1), skipped)

    # ── Metrics ──
    acc = accuracy_score(golds, preds)
    gold_has = set(golds)
    active = [c for c in GOLD_CLASSES if c in gold_has]
    macro_f1_3 = f1_score(golds, preds, labels=active, average="macro", zero_division=0)
    macro_f1_4 = f1_score(golds, preds, labels=LABEL_ORDER, average="macro", zero_division=0)
    cm = confusion_matrix(golds, preds, labels=LABEL_ORDER)
    cm_dict = {g: {p: int(cm[i][j]) for j, p in enumerate(LABEL_ORDER)}
               for i, g in enumerate(LABEL_ORDER)}
    gold_dist = dict(Counter(golds))
    pred_dist = dict(Counter(preds))
    n_err = sum(1 for r in records if not r["correct"])

    # ── Error channels ──
    def cnt(gold_lbl, pred_lbl):
        return sum(1 for r in records if r["gold"] == gold_lbl and r["pred"] == pred_lbl)
    rfi_tc = cnt("request_for_info", "tool_call")
    ca_tc = cnt("cannot_answer", "tool_call")
    ca_direct = cnt("cannot_answer", "direct")
    three = rfi_tc + ca_tc + ca_direct
    other = n_err - three

    all_err_boundaries = Counter(f"{r['gold']}→{r['pred']}" for r in records
                                 if not r["correct"])

    TRAIN_FRAC = 0.70  # SAKIKO 70/15/15 split of the evaluated trace
    def stable(ch_count):
        est_train = ch_count * TRAIN_FRAC
        return est_train, ("stable" if est_train >= 80 else
                           "marginal" if est_train >= 40 else "unstable")

    channels = {}
    for name, c in [("rfi_tc", rfi_tc), ("ca_tc", ca_tc), ("ca_direct", ca_direct)]:
        est_train, verdict = stable(c)
        channels[name] = {
            "count": c,
            "pct_of_errors": round(100 * c / n_err, 1) if n_err else 0.0,
            "est_train_errors_@0.70": round(est_train, 1),
            "stability_for_direction": verdict,
        }
    three_cov = round(100 * three / n_err, 1) if n_err else 0.0

    # ── Print report ──
    sep = "=" * 66
    log.info(sep)
    log.info("W2C MCQ baseline | %s", args.model_name)
    log.info(sep)
    log.info("N=%d  skipped=%d  Accuracy=%.4f (%.1f%%)", len(records), skipped, acc, acc*100)
    log.info("Macro-F1 (3-cls)=%.4f  Macro-F1 (4-opt)=%.4f", macro_f1_3, macro_f1_4)
    log.info("Gold dist: %s", gold_dist)
    log.info("Pred dist: %s", pred_dist)
    log.info("Total errors: %d", n_err)
    log.info("Confusion (rows=gold, cols=pred) order=%s", LABEL_ORDER)
    for g in LABEL_ORDER:
        log.info("  %-18s %s", g, [cm_dict[g][p] for p in LABEL_ORDER])
    log.info("Channels: rfi_tc=%d (%.1f%%)  ca_tc=%d (%.1f%%)  ca_direct=%d (%.1f%%)",
             rfi_tc, channels["rfi_tc"]["pct_of_errors"],
             ca_tc, channels["ca_tc"]["pct_of_errors"],
             ca_direct, channels["ca_direct"]["pct_of_errors"])
    log.info("3-channel coverage of all errors: %.1f%%  | other=%d", three_cov, other)
    log.info(sep)
    print("\n" + classification_report(golds, preds, labels=LABEL_ORDER, zero_division=0))

    # ── Save details JSONL ──
    details_path = OUT_DIR / f"{args.model_name}_baseline_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Details → %s (%d)", details_path, len(records))

    # ── Save error decomposition JSON ──
    decomp = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "n_eval": len(records),
        "skipped": skipped,
        "accuracy": round(acc, 5),
        "macro_f1_3cls": round(macro_f1_3, 5),
        "macro_f1_4opt": round(macro_f1_4, 5),
        "gold_distribution": gold_dist,
        "pred_distribution": pred_dist,
        "confusion_matrix": cm_dict,
        "total_errors": n_err,
        "channels": channels,
        "three_channel_coverage_pct": three_cov,
        "other_errors": other,
        "all_error_boundaries": dict(sorted(all_err_boundaries.items(), key=lambda x: -x[1])),
        "train_frac_assumed": TRAIN_FRAC,
        "stability_threshold_train_errors": 80,
    }
    decomp_path = OUT_DIR / f"{args.model_name}_error_decomposition.json"
    decomp_path.write_text(json.dumps(decomp, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Error decomposition → %s", decomp_path)

    # stash summary inputs for the markdown writer
    (OUT_DIR / f"{args.model_name}_baseline_summary_inputs.json").write_text(
        json.dumps({**decomp, "elapsed_s": round(elapsed, 1),
                    "vram_reserved_gb": round(vram, 2)}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
