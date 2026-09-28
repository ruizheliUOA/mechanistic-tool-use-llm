"""
eval_metatool_binary_baseline.py  –  MetaTool-Binary logprob baseline
=======================================================================
Scores each MetaTool-Binary query against two binary candidates:

    "Yes"  →  predicted tool_call
    "No"   →  predicted no_tool

The evaluation uses MetaTool's own thought_prompt field as the user
message (wrapped in the model's chat template), then measures avg_logp
of each single-token candidate at the first assistant-turn position.

Usage:
    python scripts/eval_metatool_binary_baseline.py \\
        --model_key phi35 \\
        --input data/processed/metatool_binary/metatool_binary_all.jsonl

    # Dry-run (validates data without loading model):
    python scripts/eval_metatool_binary_baseline.py \\
        --model_key phi35 \\
        --input data/processed/metatool_binary/metatool_binary_all.jsonl \\
        --dry_run

CANDIDATE STRINGS (documented):
    "Yes"  →  gold_response_mode == tool_call
    "No"   →  gold_response_mode == no_tool

NOTE: Single-token candidates mean avg_logp == sum_logp. Absolute accuracy
is NOT directly comparable to W2C multi-word logprob scores. The primary
signal of interest is the direction and count of errors, especially
no_tool → tool_call (false tool-call rate).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import gc
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── Constants ────────────────────────────────────────────────────────────────

# Binary candidate strings and their SAKIKO label mappings.
# These mirror MetaTool's original "yes/no" evaluation protocol.
CANDIDATES = {
    "tool_call": "Yes",
    "no_tool":   "No",
}
LABEL_ORDER = ["tool_call", "no_tool"]

ROOT       = Path(__file__).resolve().parents[1]
CFG_MODELS = ROOT / "configs" / "models.yaml"

OUT_DIR    = ROOT / "final" / "results" / "dataset_extension"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_models_config() -> dict:
    with open(CFG_MODELS, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt_messages(thought_prompt: str) -> list[dict]:
    """
    Wrap MetaTool's thought_prompt as a user message in chat format.
    The thought_prompt already contains the full task description,
    few-shot examples, and query — ending with 'answer:'.
    The model's chat template adds <|assistant|> for the generation prefix,
    at which point we score 'Yes' or 'No'.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": thought_prompt},
    ]


def score_candidate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    candidate_text: str,
    device: torch.device,
) -> dict:
    """
    Compute avg logprob of candidate_text conditioned on prompt_ids.
    For single-token candidates (Yes/No), avg_logp == sum_logp.
    """
    cand_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    if not cand_ids:
        return {"sum_logp": -1e9, "avg_logp": -1e9, "n_tokens": 0}

    cand_tensor = torch.tensor([cand_ids], device=device)
    input_ids = torch.cat([prompt_ids, cand_tensor], dim=1)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    prompt_len = prompt_ids.shape[1]
    cand_len   = len(cand_ids)
    logits_slice = logits[0, prompt_len - 1: prompt_len + cand_len - 1, :]
    log_probs    = torch.log_softmax(logits_slice, dim=-1)

    token_logps = [log_probs[i, tid].item() for i, tid in enumerate(cand_ids)]
    sum_logp = sum(token_logps)
    avg_logp = sum_logp / cand_len
    return {"sum_logp": sum_logp, "avg_logp": avg_logp, "n_tokens": cand_len}


# ─── Dry-run ──────────────────────────────────────────────────────────────────

def dry_run(records: list[dict], tokenizer, max_prompt_tokens: int):
    log.info("=== DRY-RUN MODE (no model loaded) ===")
    log.info("Total samples : %d", len(records))
    label_dist = Counter(r["gold_response_mode"] for r in records)
    log.info("Gold distribution: %s", dict(label_dist))

    # Validate prompt construction
    n_too_long = 0
    for r in records:
        messages = build_prompt_messages(r["thought_prompt"])
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = ""
            for m in messages:
                prompt_text += f"<|{m['role']}|>\n{m['content']}\n"
            prompt_text += "<|assistant|>\n"
        toks = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(toks) > max_prompt_tokens:
            n_too_long += 1

    log.info("Samples exceeding max_prompt_tokens=%d: %d", max_prompt_tokens, n_too_long)
    log.info("Candidate strings:")
    for lbl, cand in CANDIDATES.items():
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        log.info("  %s → '%s'  (tokens: %s)", lbl, cand, cand_ids)
    log.info("=== DRY-RUN PASSED ===")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MetaTool-Binary logprob baseline")
    parser.add_argument("--model_key", required=True,
                        help="Key in configs/models.yaml, e.g. phi35")
    parser.add_argument("--input", required=True,
                        help="Path to metatool_binary_all.jsonl (or split)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate data and prompt construction without loading model")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = run all; >0 = limit for smoke test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_models_config()
    if args.model_key not in cfg["models"]:
        log.error("model_key '%s' not in models.yaml. Available: %s",
                  args.model_key, list(cfg["models"].keys()))
        sys.exit(1)

    mcfg       = cfg["models"][args.model_key]
    defaults   = cfg["defaults"]
    model_id   = mcfg["model_id"]
    short_name = mcfg["short_name"]
    model_path = mcfg.get("local_path", model_id)
    MAX_PROMPT_TOKENS = int(mcfg.get("max_prompt_tokens",
                                      defaults.get("max_prompt_tokens", 8192)))

    log.info("Model key  : %s  (%s)", args.model_key, model_id)
    log.info("Model path : %s", model_path)
    log.info("Input file : %s", args.input)

    records = load_jsonl(args.input)
    if args.max_samples > 0:
        records = records[:args.max_samples]
    log.info("Records loaded: %d", len(records))

    # ── Load tokenizer (always needed, even for dry-run) ─────────────────
    log.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=mcfg.get("trust_remote_code",
                                    defaults.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        dry_run(records, tokenizer, MAX_PROMPT_TOKENS)
        return

    # ── Load model ────────────────────────────────────────────────────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(defaults.get("torch_dtype", "float16"), torch.float16)
    use_4bit    = defaults.get("load_in_4bit", False)

    if args.device == "auto":
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if torch.cuda.is_available() else None
    else:
        device     = torch.device(args.device)
        device_map = None

    quant_kwargs: dict = {}
    if use_4bit and torch.cuda.is_available():
        log.info("4-bit NF4 quantization enabled")
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )

    attn_impl = mcfg.get("attn_implementation")
    if attn_impl:
        quant_kwargs["attn_implementation"] = attn_impl

    log.info("Loading model (dtype=%s, device_map=%s, 4bit=%s) …",
             torch_dtype, device_map, use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=mcfg.get("trust_remote_code",
                                    defaults.get("trust_remote_code", True)),
        **quant_kwargs,
    )
    model.eval()
    if device_map is None and not use_4bit:
        model = model.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    preds_list = []
    golds_list = []
    detail_records = []
    skipped_oom = 0

    t0 = time.time()
    for rec in tqdm(records, desc="MetaTool-Binary", ncols=100):
        sample_id   = rec["sample_id"]
        query       = rec["query"]
        gold_label  = rec["gold_response_mode"]
        thought_prompt = rec["thought_prompt"]

        messages = build_prompt_messages(thought_prompt)

        # Apply chat template
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = ""
            for m in messages:
                prompt_text += f"<|{m['role']}|>\n{m['content']}\n"
            prompt_text += "<|assistant|>\n"

        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_token_ids) > MAX_PROMPT_TOKENS:
            skipped_oom += 1
            continue

        scored: dict = {}
        prompt_ids = None
        try:
            prompt_ids = torch.tensor([prompt_token_ids], device=device)
            for lbl, cand_text in CANDIDATES.items():
                scored[lbl] = score_candidate(
                    model, tokenizer, prompt_ids, cand_text, device
                )
            del prompt_ids
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("OOM on %s (prompt_len=%d) – skipping",
                             sample_id, len(prompt_token_ids))
                skipped_oom += 1
                if prompt_ids is not None:
                    del prompt_ids
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise

        pred_label = max(LABEL_ORDER, key=lambda lb: scored[lb]["avg_logp"])

        preds_list.append(pred_label)
        golds_list.append(gold_label)

        detail_records.append({
            "sample_id":          sample_id,
            "query":              query,
            "gold_response_mode": gold_label,
            "pred_response_mode": pred_label,
            "correct":            (pred_label == gold_label),
            "scored":             scored,
            "candidate_strings":  CANDIDATES,
        })

    elapsed = time.time() - t0
    log.info("Evaluation done in %.1f s  (%.2f s/sample)",
             elapsed, elapsed / max(len(records), 1))

    # ── Write per-sample details ──────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "metatool_binary_baseline_details.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in detail_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Detail trace → %s (%d records)", detail_path, len(detail_records))

    # ── Compute metrics ────────────────────────────────────────────────────
    acc = accuracy_score(golds_list, preds_list)

    cm = confusion_matrix(golds_list, preds_list, labels=LABEL_ORDER)
    # rows = gold, cols = pred
    # LABEL_ORDER = ["tool_call", "no_tool"]
    # cm[0][0] = gold=tc, pred=tc  (correct)
    # cm[0][1] = gold=tc, pred=nt  (false no-tool)
    # cm[1][0] = gold=nt, pred=tc  (false tool-call)
    # cm[1][1] = gold=nt, pred=nt  (correct)

    cm_dict = {}
    for i, g in enumerate(LABEL_ORDER):
        cm_dict[g] = {p: int(cm[i][j]) for j, p in enumerate(LABEL_ORDER)}

    n_gold_tc        = sum(1 for g in golds_list if g == "tool_call")
    n_gold_nt        = sum(1 for g in golds_list if g == "no_tool")
    tc_to_nt_errors  = cm_dict["tool_call"]["no_tool"]   # false no-tool
    nt_to_tc_errors  = cm_dict["no_tool"]["tool_call"]   # false tool-call

    false_toolcall_rate = nt_to_tc_errors / n_gold_nt if n_gold_nt > 0 else 0.0
    false_notool_rate   = tc_to_nt_errors / n_gold_tc if n_gold_tc > 0 else 0.0

    error_pool = {}
    for r in detail_records:
        if not r["correct"]:
            k = f"{r['gold_response_mode']}_to_{r['pred_response_mode']}"
            error_pool[k] = error_pool.get(k, 0) + 1

    # ── Decision rule ─────────────────────────────────────────────────────
    if nt_to_tc_errors >= 100:
        verdict = "ENOUGH: MetaTool-Binary has sufficient over-calling errors for MetaTool-native binary SAKIKO."
    elif nt_to_tc_errors > 0:
        verdict = "WEAK: MetaTool-Binary is weak external validation only; prioritize ToolSandbox-Derived annotation later."
    else:
        verdict = "TOO EASY: MetaTool-Binary is only a sanity check; move to ToolSandbox-Derived."

    # ── Print report ──────────────────────────────────────────────────────
    sep = "=" * 65
    log.info(sep)
    log.info("MetaTool-Binary Baseline  |  model: %s", model_id)
    log.info(sep)
    log.info("N evaluated      : %d  (skipped: %d)", len(detail_records), skipped_oom)
    log.info("Accuracy         : %.4f  (%.1f%%)", acc, acc * 100)
    log.info("")
    log.info("Confusion matrix (rows=gold, cols=pred):")
    log.info("                   pred=tool_call  pred=no_tool")
    log.info("  gold=tool_call       %4d            %4d",
             cm_dict["tool_call"]["tool_call"], cm_dict["tool_call"]["no_tool"])
    log.info("  gold=no_tool         %4d            %4d",
             cm_dict["no_tool"]["tool_call"], cm_dict["no_tool"]["no_tool"])
    log.info("")
    log.info("tool_call → no_tool errors (false no-tool)   : %d", tc_to_nt_errors)
    log.info("no_tool   → tool_call errors (false tool-call): %d", nt_to_tc_errors)
    log.info("")
    log.info("False tool-call rate (gold=no_tool )  : %.4f  (%d / %d)",
             false_toolcall_rate, nt_to_tc_errors, n_gold_nt)
    log.info("False no-tool   rate (gold=tool_call) : %.4f  (%d / %d)",
             false_notool_rate, tc_to_nt_errors, n_gold_tc)
    log.info("")
    log.info("DECISION RULE VERDICT:")
    log.info("  %s", verdict)
    log.info(sep)

    print("\n" + classification_report(
        golds_list, preds_list, labels=LABEL_ORDER, zero_division=0
    ))

    # ── Write summary markdown ─────────────────────────────────────────────
    summary_md = OUT_DIR / "metatool_binary_baseline_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(f"# MetaTool-Binary Baseline Summary\n\n")
        f.write(f"**Model**: {model_id}  \n")
        f.write(f"**Model key**: {short_name}  \n")
        f.write(f"**Input**: {args.input}  \n")
        f.write(f"**Elapsed**: {elapsed:.1f} s  \n\n")
        f.write(f"## Candidate Strings\n\n")
        f.write(f"| SAKIKO label | Candidate text | Note |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| tool_call | `\"Yes\"` | single token |\n")
        f.write(f"| no_tool   | `\"No\"`  | single token |\n\n")
        f.write(f"Prompt: MetaTool `thought_prompt` wrapped in chat template (user message).  \n")
        f.write(f"Scoring: avg_logp of candidate continuation at first assistant-turn.  \n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| N evaluated | {len(detail_records)} |\n")
        f.write(f"| Skipped (OOM/long) | {skipped_oom} |\n")
        f.write(f"| Accuracy | {acc:.4f} ({acc*100:.1f}%) |\n")
        f.write(f"| tool_call→no_tool errors (false no-tool) | {tc_to_nt_errors} |\n")
        f.write(f"| no_tool→tool_call errors (false tool-call) | **{nt_to_tc_errors}** |\n")
        f.write(f"| False tool-call rate (gold=no_tool) | {false_toolcall_rate:.4f} |\n")
        f.write(f"| False no-tool rate (gold=tool_call) | {false_notool_rate:.4f} |\n\n")
        f.write(f"## Confusion Matrix\n\n")
        f.write(f"| | pred=tool_call | pred=no_tool |\n|---|---|---|\n")
        f.write(f"| gold=tool_call | {cm_dict['tool_call']['tool_call']} | {cm_dict['tool_call']['no_tool']} |\n")
        f.write(f"| gold=no_tool | {cm_dict['no_tool']['tool_call']} | {cm_dict['no_tool']['no_tool']} |\n\n")
        f.write(f"## Decision Rule Verdict\n\n")
        f.write(f"> {verdict}\n")

    log.info("Summary → %s", summary_md)

    # Also save JSON summary
    summary_json = OUT_DIR / "metatool_binary_baseline_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "model_id":               model_id,
            "short_name":             short_name,
            "input_file":             args.input,
            "n_evaluated":            len(detail_records),
            "skipped":                skipped_oom,
            "accuracy":               round(acc, 5),
            "confusion_matrix":       cm_dict,
            "tc_to_nt_errors":        tc_to_nt_errors,
            "nt_to_tc_errors":        nt_to_tc_errors,
            "false_toolcall_rate":    round(false_toolcall_rate, 5),
            "false_notool_rate":      round(false_notool_rate, 5),
            "n_gold_tool_call":       n_gold_tc,
            "n_gold_no_tool":         n_gold_nt,
            "error_pool":             error_pool,
            "candidate_strings":      CANDIDATES,
            "decision_verdict":       verdict,
            "elapsed_s":              round(elapsed, 1),
        }, f, indent=2, ensure_ascii=False)
    log.info("JSON summary → %s", summary_json)


if __name__ == "__main__":
    main()
