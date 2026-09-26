"""
phase3_smoke_test.py — strict preflight smoke test (N samples) on the EXACT Phase-3 baseline path.
==================================================================================================
Reuses phase3_lib.load_ds_raw (archived shuffle-seed-42 loader) and qwen7b_native_sakiko.load_model
(bf16, device_map cuda:0) and the SAME avg_logp scoring loop as phase3_baseline_regen.py. Verifies:
model loads in bf16 (not 4-bit), no OOM, valid predictions/scores, schema matches the planned
baseline_details_regen.jsonl rows. Writes NOTHING to the real baseline file.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L
import qwen7b_native_sakiko as P

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EXPECTED_KEYS = {"uuid", "gold", "pred", "correct", "avg_logp"}
LABELS = ["direct", "tool_call", "request_for_info", "cannot_answer"]


def score_row(model, tok, dev, s):
    prompt = P.make_prompt_text(tok, s)
    pids = tok.encode(prompt, add_special_tokens=False)
    prompt_ids = torch.tensor([pids], device=dev)
    scored = {}
    for lb in LABELS:
        cand = s["answers"].get(lb, "")
        if not cand:
            scored[lb] = -1e9
            continue
        cids = tok.encode(cand, add_special_tokens=False)
        inp = torch.cat([prompt_ids, torch.tensor([cids], device=dev)], dim=1)
        with torch.no_grad():
            logits = model(inp).logits
        sl = logits[0, len(pids) - 1: len(pids) + len(cids) - 1, :]
        lp = torch.log_softmax(sl.float(), dim=-1)
        scored[lb] = sum(lp[j, cids[j]].item() for j in range(len(cids))) / len(cids)
    pred = max(scored, key=scored.get)
    return {"uuid": s["uuid"], "gold": s["correct_answer"], "pred": pred,
            "correct": bool(pred == s["correct_answer"]),
            "avg_logp": {k: round(v, 4) for k, v in scored.items()}}


def main():
    ds = L.load_ds_raw()
    print(f"[loader] rows={len(ds)}  first_uuid={ds[0]['uuid']}")
    model, tok, dev = P.load_model()

    # --- precision / dtype checks ---
    param_dtypes = {str(p.dtype) for p in model.parameters()}
    is_bf16 = param_dtypes == {"torch.bfloat16"}
    quantized = getattr(model.config, "quantization_config", None) is not None
    vram = torch.cuda.memory_reserved(0) / 1e9
    print(f"[dtype] param_dtypes={param_dtypes}  bf16_only={is_bf16}  "
          f"quantized={quantized}  vram_reserved={vram:.2f}GB")

    rows, oom = [], False
    try:
        for i in range(N):
            r = score_row(model, tok, dev, ds[i])
            rows.append(r)
            print(f"[row {i}] uuid={r['uuid']} gold={r['gold']} pred={r['pred']} "
                  f"logp={r['avg_logp']}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom = True
            print("[OOM] CUDA OOM during scoring")
        else:
            raise

    # --- schema / validity checks ---
    schema_ok = all(set(r.keys()) == EXPECTED_KEYS for r in rows)
    preds_valid = all(r["pred"] in LABELS for r in rows)
    scores_valid = all(
        all(isinstance(v, float) for v in r["avg_logp"].values())
        and any(v > -1e8 for v in r["avg_logp"].values())
        for r in rows)
    # margins non-degenerate: best strictly > second-best on every row
    margins_ok = True
    for r in rows:
        vals = sorted(r["avg_logp"].values(), reverse=True)
        if not (vals[0] > vals[1]):
            margins_ok = False

    checks = {
        "model_loads": True,
        "bf16_not_4bit": is_bf16 and not quantized,
        "vram_consistent_with_bf16": 12.0 <= vram <= 22.0,  # ~15.4GB expected; 4-bit would be ~5GB
        "no_oom": not oom,
        "n_rows": len(rows) == N,
        "schema_matches": schema_ok,
        "preds_valid_label": preds_valid,
        "scores_valid_float": scores_valid,
        "margins_non_degenerate": margins_ok,
    }
    print("\n=== SMOKE CHECKS ===")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(checks.values()) else "BLOCKED"
    print(f"\nSMOKE_VERDICT: {verdict}")
    print(f"EXPECTED_SCHEMA_KEYS: {sorted(EXPECTED_KEYS)}")
    sys.exit(0 if verdict == "PASS" else 3)


if __name__ == "__main__":
    main()
