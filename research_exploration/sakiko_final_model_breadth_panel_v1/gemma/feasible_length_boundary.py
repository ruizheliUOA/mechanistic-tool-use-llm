"""Measure (a) the full Gemma-rendered prompt-length distribution over all 3652 rows and
(b) the empirical maximum feasible sequence length for the frozen 4-mode gradient stack
under BF16 + eager attention + validated checkpointing on this GPU."""
import json, os, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_test_vram_v4 import (MODEL_DIR, RAW, MODES, L_INJ, set_determinism,
                                render_prompt_gemma, candidate_ids, gradient_candidate, mem)
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.init(); set_determinism()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
rows = [json.loads(l) for l in open(RAW)]
print(f"population rows: {len(rows)}")

# (a) full distribution: prompt tokens and max full sequence (prompt + longest candidate)
recs = []
for i, s in enumerate(rows):
    try:
        p = render_prompt_gemma(tok, s)
        pl = len(tok.encode(p, add_special_tokens=False))
        cl = max(len(tok.encode(s["answers"][m], add_special_tokens=False)) for m in MODES)
        recs.append({"idx": i, "prompt_tokens": pl, "max_seq": pl + cl})
    except Exception as e:
        recs.append({"idx": i, "error": f"{type(e).__name__}: {str(e)[:80]}"})
ok = [r for r in recs if "max_seq" in r]
ms = np.array([r["max_seq"] for r in ok])
dist = {"n": len(ok), "render_errors": len(recs)-len(ok),
        "min": int(ms.min()), "p50": int(np.percentile(ms,50)), "p90": int(np.percentile(ms,90)),
        "p95": int(np.percentile(ms,95)), "p99": int(np.percentile(ms,99)), "max": int(ms.max())}
print("max_seq distribution:", json.dumps(dist))

# (b) empirical feasibility boundary by ascending real samples
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
    torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
model.eval(); model.config.use_cache = False
for p_ in model.parameters(): p_.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train(True)   # validated bit-identical in v4; all dropout probabilities are zero
device = next(model.parameters()).device

order = sorted(ok, key=lambda r: r["max_seq"])
lo, hi = 0, len(order)-1
last_ok, first_fail = None, None
probe = []
while lo <= hi:
    mid = (lo+hi)//2
    r = order[mid]; s = rows[r["idx"]]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    try:
        prompt = render_prompt_gemma(tok, s)
        for m in MODES:
            pids, cids = candidate_ids(tok, prompt, s["answers"][m])
            gradient_candidate(model, device, pids, cids)
            torch.cuda.empty_cache()
        probe.append({"max_seq": r["max_seq"], "status": "OK", "peak_gb": mem()["peak_allocated_gb"]})
        last_ok = r["max_seq"]; lo = mid+1
    except torch.cuda.OutOfMemoryError:
        probe.append({"max_seq": r["max_seq"], "status": "OOM", "peak_gb": mem()["peak_allocated_gb"]})
        first_fail = r["max_seq"] if first_fail is None else min(first_fail, r["max_seq"])
        hi = mid-1; torch.cuda.empty_cache()
    print(f"  probe max_seq={r['max_seq']:5d} -> {probe[-1]['status']}")

boundary = first_fail if first_fail else None
excluded = int((ms >= boundary).sum()) if boundary else 0
out = {"distribution": dist, "probes": probe, "last_feasible_max_seq": last_ok,
       "first_infeasible_max_seq": first_fail,
       "rows_at_or_above_first_infeasible": excluded,
       "fraction_of_population": round(excluded/len(ok), 6) if boundary else 0.0,
       "attention_implementation": "eager (frozen - NOT changed)",
       "gradient_checkpointing": "enabled, validated bit-identical in VRAM_SMOKE_TEST_V4.json"}
print(json.dumps({k: out[k] for k in out if k != "probes"}, indent=2))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FEASIBLE_LENGTH_BOUNDARY.json"), "w"), indent=2)
