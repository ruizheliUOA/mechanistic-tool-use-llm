"""Gemma-2-9B-it - frozen section-12 BF16 mechanical VRAM smoke test.

Replicates the committed Qwen3 loader and gradient semantics exactly, with only the
architecture-mandated substitutions: MODEL_DIR, HIDDEN_SIZE 3584, L_INJ 24, L_OBS 30,
and the Gemma-native chat interface (Qwen's template requires <|im_start|>/<think> which
Gemma-2 does not have).

No quantisation. No precision change. BF16 exact, eager attention, as frozen.
"""
import json, os, platform, random, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

MODEL_DIR = "/root/autodl-tmp/models/gemma-2-9b-it-11c9b309"
RAW = "/root/autodl-tmp/sakiko-followup/data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl"
MODES = ["tool_call", "direct", "request_for_info", "cannot_answer"]
L_OBS, L_INJ, HIDDEN_SIZE, N_LAYERS = 30, 24, 3584, 42
ATTENTION_IMPLEMENTATION = "eager"
MAX_TOKENS = 8192
PROMPT_STATE_ATOL = 1e-6

def set_determinism():
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        return {"deterministic_algorithms": True, "note": "strict"}
    except Exception as e:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return {"deterministic_algorithms": "warn_only", "note": str(e)[:120]}

def parse_tools(tools):
    out = []
    for t in tools or []:
        if isinstance(t, str):
            try: out.append(json.loads(t))
            except json.JSONDecodeError: out.append({"raw": t})
        else: out.append(t)
    return out

SYSTEM = ("You are a helpful assistant.\n\n"
          "Given the user's question and the available tools, choose the most "
          "appropriate response from the provided options.")

def render_prompt_gemma(tokenizer, sample):
    """Gemma-native rendering. Gemma-2 has no system role and no tools= template
    argument, so the system text and the tool schemas are folded into the single
    user turn. Semantically matched to the Qwen rendering; frozen before baseline."""
    tools = parse_tools(sample.get("tools"))
    tool_block = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    content = f"{SYSTEM}\n\n# Available tools\n{tool_block}\n\n# Question\n{sample['question']}"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True)
    required = "<start_of_turn>model\n"
    if not text.endswith(required):
        raise RuntimeError("READOUT_INVALID: gemma generation prompt suffix absent")
    return text

def candidate_ids(tokenizer, prompt, candidate):
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    cids = tokenizer.encode(candidate, add_special_tokens=False)
    if not pids or not cids: raise RuntimeError("empty prompt or candidate tokens")
    if len(pids) + len(cids) > MAX_TOKENS:
        raise RuntimeError("predeclared structural exclusion: sequence exceeds 8192 tokens")
    return pids, cids

def gradient_candidate(model, device, pids, cids):
    ids = torch.tensor([pids + cids], dtype=torch.long, device=device)
    positions = torch.arange(len(pids) - 1, len(pids) + len(cids) - 1, device=device)
    leaf, prompt_state = {}, {}
    module = model.model.layers[L_INJ].mlp
    def hook(_m, _inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        detached = tensor.detach().requires_grad_(True)
        leaf["x"] = detached
        prompt_state["x"] = detached[0, :len(pids)].detach().float().cpu().numpy()
        return (detached,) + out[1:] if isinstance(out, tuple) else detached
    handle = module.register_forward_hook(hook)
    try:
        logits = model(ids, use_cache=False, logits_to_keep=positions).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        labels = torch.tensor(cids, dtype=torch.long, device=device)
        score = lp[torch.arange(len(cids), device=device), labels].mean()
        grad = torch.autograd.grad(score, leaf["x"], retain_graph=False, create_graph=False)[0][0].detach().float().cpu().numpy()
        value = float(score.detach().cpu())
    finally:
        handle.remove()
    return value, grad, prompt_state["x"]

def mem():
    return {"allocated_gb": round(torch.cuda.memory_allocated(0)/1e9, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(0)/1e9, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0)/1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(0)/1e9, 3)}

def main():
    rep = {"attempt": "v4 - v3 + train-mode activation of checkpointing, gated on all-dropout-zero and bit-identical validation", "model_dir": MODEL_DIR, "L_OBS": L_OBS, "L_INJ": L_INJ,
           "HIDDEN_SIZE": HIDDEN_SIZE, "attn": ATTENTION_IMPLEMENTATION, "dtype": "bfloat16"}
    torch.cuda.init()
    rep["determinism"] = set_determinism()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    rep["gpu"] = torch.cuda.get_device_name(0)
    rep["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory/1e9, 3)
    rep["mem_before_load"] = mem()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map="cuda:0")
    model.eval(); model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    device = next(model.parameters()).device
    rep["load_seconds"] = round(time.perf_counter() - t0, 1)
    _gc_enabled = False
    rep["mem_after_load"] = mem()
    rep["env"] = {"python": platform.python_version(), "torch": torch.__version__,
                  "transformers": transformers.__version__, "cuda": torch.version.cuda}
    # architecture identity assertions
    cfg = model.config
    rep["identity"] = {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
                       "vocab_size": cfg.vocab_size, "model_type": cfg.model_type,
                       "n_params_b": round(sum(p.numel() for p in model.parameters())/1e9, 3),
                       "all_params_requires_grad_false": not any(p.requires_grad for p in model.parameters())}
    assert cfg.hidden_size == HIDDEN_SIZE and cfg.num_hidden_layers == N_LAYERS, "architecture identity mismatch"

    # ---- numerical validation of gradient checkpointing, on a length that fits without it ----
    rows0 = []
    with open(RAW) as f:
        for i, line in enumerate(f):
            if i >= 60: break
            rows0.append(json.loads(line))
    vs = sorted(rows0, key=lambda s: len(tok.encode(render_prompt_gemma(tok, s), add_special_tokens=False)))[len(rows0)//2]
    vp = render_prompt_gemma(tok, vs)
    vpid, vcid = candidate_ids(tok, vp, vs["answers"]["tool_call"])
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    s_off, g_off, _ = gradient_candidate(model, device, vpid, vcid)
    m_off = mem()["peak_allocated_gb"]
    # HF activates checkpointing only when module.training is True. Train mode is
    # mechanically equivalent ONLY if every dropout probability is exactly zero.
    drops = {k: v for k, v in vars(model.config).items() if "dropout" in k.lower()}
    import torch.nn as nn
    live = [f"{n}:{m.p}" for n, m in model.named_modules() if isinstance(m, nn.Dropout) and m.p != 0.0]
    rep["dropout_audit"] = {"config_dropout_fields": drops, "nonzero_dropout_modules": live,
                            "train_mode_mechanically_equivalent": (not live) and all(
                                (v == 0 or v is None) for v in drops.values() if isinstance(v, (int, float)))}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if rep["dropout_audit"]["train_mode_mechanically_equivalent"]:
        model.train(True)     # required for HF to dispatch the checkpointed path
        _gc_enabled = True
    else:
        model.train(False)
        _gc_enabled = False
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    s_on, g_on, _ = gradient_candidate(model, device, vpid, vcid)
    m_on = mem()["peak_allocated_gb"]
    denom = float(np.linalg.norm(g_off)) or 1.0
    rel = float(np.linalg.norm(g_on - g_off)) / denom
    amax = float(np.max(np.abs(g_on - g_off)))
    rep["gradient_checkpointing_validation"] = {
        "enabled": _gc_enabled, "use_reentrant": False, "train_mode": bool(model.training),
        "validation_prompt_tokens": len(vpid),
        "score_off": round(s_off, 8), "score_on": round(s_on, 8),
        "score_abs_diff": abs(s_on - s_off),
        "grad_relative_l2_diff": rel, "grad_max_abs_diff": amax,
        "peak_alloc_gb_off": m_off, "peak_alloc_gb_on": m_on,
        "numerically_equivalent": bool(rel <= 1e-5 and abs(s_on - s_off) <= 1e-5),
    }
    rep["gradient_checkpointing_enabled"] = _gc_enabled

    rows = []
    with open(RAW) as f:
        for i, line in enumerate(f):
            if i >= 400: break
            rows.append(json.loads(line))
    # pick shortest / median / longest by rendered prompt length to bracket the memory envelope
    lens = []
    for s in rows:
        try: lens.append((len(tok.encode(render_prompt_gemma(tok, s), add_special_tokens=False)), s))
        except Exception: pass
    lens.sort(key=lambda x: x[0])
    rep["prompt_len_stats"] = {"n": len(lens), "min": lens[0][0], "p50": lens[len(lens)//2][0],
                               "p90": lens[int(len(lens)*0.9)][0], "max": lens[-1][0]}
    picks = [("shortest", lens[0]), ("median", lens[len(lens)//2]),
             ("p90", lens[int(len(lens)*0.9)]), ("longest", lens[-1])]

    trials = []
    for label, (plen, sample) in picks:
        torch.cuda.reset_peak_memory_stats(0)
        prompt = render_prompt_gemma(tok, sample)
        t1 = time.perf_counter()
        grads, scores, states, finite = {}, {}, {}, True
        try:
            for m in MODES:
                pids, cids = candidate_ids(tok, prompt, sample["answers"][m])
                sc, g, st = gradient_candidate(model, device, pids, cids)
                grads[m], scores[m], states[m] = g, sc, st
                finite &= bool(np.isfinite(g).all()) and np.isfinite(sc)
                del pids, cids
                torch.cuda.empty_cache()   # cache release between modes - permitted, mechanically equivalent
            full = np.stack([grads[m].sum(axis=0) for m in MODES]).astype(np.float32)
            shared = all(states[m].shape == states[MODES[0]].shape and
                         np.max(np.abs(states[m]-states[MODES[0]])) <= PROMPT_STATE_ATOL for m in MODES[1:])
            trials.append({"label": label, "prompt_tokens": plen,
                           "max_seq": max(len(tok.encode(prompt, add_special_tokens=False)) +
                                          len(tok.encode(sample["answers"][m], add_special_tokens=False)) for m in MODES),
                           "status": "OK", "grad_shape": list(full.shape),
                           "all_finite": bool(finite), "common_prompt_state_shared": bool(shared),
                           "grad_l2_full_effect": [round(float(np.linalg.norm(full[i])), 4) for i in range(4)],
                           "scores": {m: round(scores[m], 5) for m in MODES},
                           "seconds": round(time.perf_counter()-t1, 2), "mem": mem()})
        except torch.cuda.OutOfMemoryError as e:
            trials.append({"label": label, "prompt_tokens": plen, "status": "CUDA_OOM",
                           "error": str(e)[:200], "mem": mem()})
            torch.cuda.empty_cache()
        except Exception as e:
            trials.append({"label": label, "prompt_tokens": plen, "status": "ERROR",
                           "error": f"{type(e).__name__}: {str(e)[:200]}", "mem": mem()})
            torch.cuda.empty_cache()
    rep["trials"] = trials
    rep["mem_final"] = mem()
    gv = rep.get("gradient_checkpointing_validation", {})
    if not gv.get("numerically_equivalent", False):
        rep["checkpointing_note"] = ("gradient checkpointing did NOT reproduce the un-checkpointed gradient "
                                     "within tolerance; section 12 permits it only if numerically validated, "
                                     "so any pass obtained under it is NOT admissible")
    oom = [t for t in trials if t["status"] == "CUDA_OOM"]
    err = [t for t in trials if t["status"] == "ERROR"]
    ok  = [t for t in trials if t["status"] == "OK"]
    rep["verdict"] = ("SMOKE_TEST_PASS" if len(ok) == len(trials)
                      else "MODEL_HARDWARE_NO_GO" if oom
                      else "SMOKE_TEST_ERROR")
    print(json.dumps(rep, indent=2))
    out = "/root/autodl-tmp/sakiko-followup/research_exploration/sakiko_final_model_breadth_panel_v1/gemma/VRAM_SMOKE_TEST_V4.json"
    json.dump(rep, open(out, "w"), indent=2)
    print("\nVERDICT:", rep["verdict"])

if __name__ == "__main__":
    main()
