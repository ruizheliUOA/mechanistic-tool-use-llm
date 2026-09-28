"""
metatool_qwen7b_extract_acts.py — Phase 2: Qwen2.5-7B activations on MetaTool prompts.
=======================================================================================
Captures the MLP output at the LAST PROMPT TOKEN for observation-layer candidates
L12/L16/L20/L24 (locked in PHASE2_PROTOCOL.md §11), one forward pass per sample over
metatool_binary_all.jsonl (split-invariant; rows indexed by position = all.jsonl order).

Cache -> data/processed/metatool_binary/cache_qwen7b/acts_L{L}.npy   (gitignored, regenerable)
Usage:   python scripts/metatool_qwen7b_extract_acts.py
"""
from __future__ import annotations
import json, sys, time, logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/metatool_binary/metatool_binary_all.jsonl"
CACHE = ROOT / "data/processed/metatool_binary/cache_qwen7b"
MODEL_PATH = ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"
OBS_LAYERS = [12, 16, 20, 24]
DTYPE = torch.bfloat16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("acts")


def build_prompt_text(thought_prompt, tok):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": thought_prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if all((CACHE / f"acts_L{L}.npy").exists() for L in OBS_LAYERS):
        log.info("all caches exist — nothing to do")
        return
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="cuda:0", trust_remote_code=True).eval()
    D = model.config.hidden_size
    acts = {L: np.zeros((len(recs), D), dtype=np.float32) for L in OBS_LAYERS}

    t0 = time.time()
    for i, r in enumerate(recs):
        ids = tok.encode(build_prompt_text(r["thought_prompt"], tok),
                         add_special_tokens=False)
        last = len(ids) - 1
        input_ids = torch.tensor([ids], device=model.device)
        captured = {}
        handles = []
        for L in OBS_LAYERS:
            def mk(LL):
                def _h(module, inp, out, _c=captured, _p=last, _L=LL):
                    _c[_L] = out[0, _p, :].detach().cpu().float().numpy()
                return _h
            handles.append(model.model.layers[L].mlp.register_forward_hook(mk(L)))
        with torch.no_grad():
            model(input_ids)
        for h in handles:
            h.remove()
        for L in OBS_LAYERS:
            acts[L][i] = captured[L]
        del input_ids
        if (i + 1) % 200 == 0:
            log.info("%d/%d (%.1f min)", i + 1, len(recs), (time.time() - t0) / 60)

    for L in OBS_LAYERS:
        np.save(CACHE / f"acts_L{L}.npy", acts[L])
        log.info("saved acts_L%d.npy %s", L, acts[L].shape)
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
