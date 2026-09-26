"""
metatool_binary_fp16_confirmation.py
=====================================
fp16 confirmation of the MetaTool-Binary SAKIKO over-calling pilot.

The 4-bit NF4 audit was positive (Net = 12.2 ± 5.1, 5/5 seeds) but every
MetaTool run so far used quantization. This script re-runs the SAME pipeline
in **fp16** (no 4-bit), with a SEPARATE fp16 activation cache and a fresh
fp16 baseline (activations and baseline predictions both change with
precision), so 4-bit and fp16 numbers are never mixed.

Pipeline (per seed): baseline (Yes/No re-scored in fp16) -> L18 activation
extraction -> nt_tc DiffMean direction -> LR router -> val sweep (layer x
alpha x threshold) -> locked test -> placebo (real/random/reverse).

Reuses the pure-logic helpers from metatool_binary_robustness_audit; only the
model loader and activation cache are fp16-specific.

Usage:
  cd /path/to/project
  /path/to/project/.venv/bin/python scripts/metatool_binary_fp16_confirmation.py \\
      --load_mode fp16          # fp16 (default); 4bit only for A/B re-check

IMPORTANT: MetaTool-Binary, NOT W2C-4way. Candidate strings: Yes / No.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Reuse audit + pilot logic ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_metatool_binary import stratified_split
from metatool_binary_robustness_audit import (
    load_all_records, load_yesno_baseline,
    difmean, train_router, router_scores, predict_corrected_c,
    eval_split, val_sweep_and_lock, run_locked_test, rescore_baseline,
    error_type,
    SEEDS, CANDIDATE_SETS, INJ_LAYERS, ALPHAS, THRESHOLDS, OBS_LAYER, INJ_MODE,
)
from metatool_binary_sakiko_pilot import build_prompt_text

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
CFG_MODELS = ROOT / "configs" / "models.yaml"
DATA_DIR  = ROOT / "data" / "processed" / "metatool_binary"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR   = ROOT / "final" / "results" / "dataset_extension"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

YESNO = CANDIDATE_SETS["yes_no"]
N_RANDOM = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mt_fp16")


# ── Model loading (fp16 explicit; 4bit only for optional A/B) ────────────────

def load_model_mode(model_key: str, load_mode: str):
    with open(CFG_MODELS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg     = cfg["models"][model_key]
    defaults = cfg["defaults"]
    model_path = mcfg.get("local_path", mcfg["model_id"])

    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=mcfg.get("trust_remote_code",
                                    defaults.get("trust_remote_code", True)))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=mcfg.get("trust_remote_code",
                                    defaults.get("trust_remote_code", True)),
    )
    attn = mcfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn

    if load_mode == "fp16":
        log.info("Loading model in fp16 (NO quantization) …")
    elif load_mode == "4bit":
        log.info("Loading model in 4-bit NF4 (A/B re-check only) …")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
    else:
        raise ValueError(f"Unknown load_mode: {load_mode}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    device = next(model.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("Model loaded in %.1fs  | mode=%s | device=%s | "
                 "VRAM alloc=%.2fGB reserved=%.2fGB / %.1fGB",
                 time.time() - t0, load_mode, device, alloc, reserved, total)
    return model, tok, device, model_path


# ── fp16 activation extraction (separate cache) ──────────────────────────────

def extract_acts_fp16(model, tok, thought, obs_layer, device, load_mode, force=False):
    tag = "fp16" if load_mode == "fp16" else "4bit"
    cache = CACHE_DIR / f"acts_all_L{obs_layer}_{tag}.npy"
    if cache.exists() and not force:
        acts = np.load(cache)
        log.info("Loaded cached %s acts_all: %s", tag, acts.shape)
        return acts

    n = len(thought)
    D = model.config.hidden_size
    acts = np.zeros((n, D), dtype=np.float32)
    mlp = model.model.layers[obs_layer].mlp
    for oid in tqdm(range(n), desc=f"Acts-{tag}-L{obs_layer}"):
        prompt_text = build_prompt_text(thought[oid], tok)
        ids = tok.encode(prompt_text, add_special_tokens=False)
        last = len(ids) - 1
        inp = torch.tensor([ids], device=device)
        cap = {}
        def _h(m, i, o, _c=cap, _p=last):
            _c["a"] = o[0, _p, :].detach().cpu().float().numpy()
        h = mlp.register_forward_hook(_h)
        with torch.no_grad():
            model(inp)
        h.remove()
        acts[oid] = cap["a"]
        del inp
        if (oid + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    np.save(cache, acts)
    log.info("Saved %s acts_all: %s -> %s", tag, acts.shape, cache)
    return acts


# ── Single-direction evaluation (for placebo real/random/reverse) ────────────

def eval_direction(model, tok, test_oids, golds, base_pred, thought,
                   unit_dir, median_norm, router, acts_all, locked, device):
    corr = (locked["alpha"] * median_norm * unit_dir).astype(np.float32)
    tsc = router_scores(router, test_oids, acts_all)
    tsc_by = {o: float(tsc[k]) for k, o in enumerate(test_oids)}
    touched = [o for o in test_oids
               if tsc_by[o] >= locked["threshold"] and base_pred[o] == "tool_call"]
    int_pred = dict(base_pred)
    for o in touched:
        int_pred[o] = predict_corrected_c(
            model, tok, thought[o], corr, locked["inj_layer"], device, YESNO)
    m = eval_split(test_oids, golds, base_pred, int_pred)
    m["n_touched"] = len(touched)
    return m


# ── One full seed run (fp16) ─────────────────────────────────────────────────

def run_seed(model, tok, records, golds, thought, fp16_bp, acts_all, seed, device,
             want_placebo=False, n_random=0):
    train, val, test = stratified_split(records, seed=seed)
    tro = [r["original_id"] for r in train]
    vao = [r["original_id"] for r in val]
    teo = [r["original_id"] for r in test]

    direction = difmean(tro, acts_all, golds, fp16_bp)
    router    = train_router(tro, vao, acts_all, golds, fp16_bp)
    if direction is None or router is None:
        return None

    locked, _grid = val_sweep_and_lock(
        model, tok, vao, golds, fp16_bp, thought, direction, router, acts_all, device)
    test_m, _ipred, _tsc = run_locked_test(
        model, tok, teo, golds, fp16_bp, thought, direction, router,
        acts_all, locked, device, candidates=YESNO)

    row = {
        "seed": seed,
        "router_val_auc": round(router["val_auc"], 5) if router["val_auc"] else None,
        "router_cv_auc":  round(router["cv_auc"], 5) if router["cv_auc"] else None,
        "direction_n_err": direction["n_err"], "direction_n_ref": direction["n_ref"],
        "selected_inj_layer": locked["inj_layer"],
        "selected_alpha": locked["alpha"],
        "selected_threshold": locked["threshold"],
        "val_net": locked["val_net"],
        "test_baseline_acc": test_m["baseline_acc"],
        "test_sakiko_acc":   test_m["sakiko_acc"],
        "test_delta_acc":    test_m["delta_acc"],
        "test_fixed": test_m["fixed"], "test_broke": test_m["broke"],
        "test_net": test_m["net"],
        "nt_tc_before": test_m["nt_tc_before"], "nt_tc_after": test_m["nt_tc_after"],
        "false_toolcall_rate_before": test_m["false_toolcall_rate_before"],
        "false_toolcall_rate_after":  test_m["false_toolcall_rate_after"],
        "n_touched": test_m["n_touched"],
        "router_fire_rate": test_m["router_fire_rate"],
    }

    placebo = None
    if want_placebo:
        median_norm = direction["median_norm"]
        unit = direction["unit"]
        D = acts_all.shape[1]
        real = eval_direction(model, tok, teo, golds, fp16_bp, thought,
                              unit, median_norm, router, acts_all, locked, device)
        rev = eval_direction(model, tok, teo, golds, fp16_bp, thought,
                             -unit, median_norm, router, acts_all, locked, device)
        rng = np.random.RandomState(1000)
        rdir = rng.randn(D).astype(np.float32); rdir /= (np.linalg.norm(rdir) + 1e-12)
        rand1 = eval_direction(model, tok, teo, golds, fp16_bp, thought,
                               rdir, median_norm, router, acts_all, locked, device)
        placebo = {"real": real, "reverse": rev, "random1": rand1}

        random_nets = []
        if n_random > 0:
            for i in range(n_random):
                rs = np.random.RandomState(2000 + i)
                rd = rs.randn(D).astype(np.float32); rd /= (np.linalg.norm(rd) + 1e-12)
                mr = eval_direction(model, tok, teo, golds, fp16_bp, thought,
                                    rd, median_norm, router, acts_all, locked, device)
                random_nets.append(mr["net"])
                gc.collect(); torch.cuda.empty_cache()
            arr = np.array(random_nets, dtype=float)
            n_ge = int((arr >= real["net"]).sum())
            placebo["random_distribution"] = {
                "n_random": n_random,
                "real_net": real["net"],
                "random_mean": round(float(arr.mean()), 4),
                "random_std":  round(float(arr.std()), 4),
                "random_max":  int(arr.max()),
                "random_min":  int(arr.min()),
                "random_median": float(np.median(arr)),
                "n_random_ge_real": n_ge,
                "real_rank_among_random": int((arr > real["net"]).sum()) + 1,
                "real_percentile": round(100.0 * (1 - n_ge / n_random), 1),
                "random_nets": random_nets,
            }

    return {"row": row, "locked": locked, "placebo": placebo,
            "direction": direction, "router": router,
            "test_oids": teo}


# ── 4-bit reference (from the audit, for the A/B comparison column) ───────────

def load_4bit_reference():
    """Per-seed 4-bit results from the audit, for side-by-side reporting."""
    p = OUT_DIR / "metatool_binary_multiseed_results.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {r["seed"]: r for r in data}


# ── Report writers ────────────────────────────────────────────────────────────

def write_reports(load_mode, seed42, multiseed_rows, ref4bit, mem_note):
    # JSON (seed=42 detail + placebo)
    s42 = seed42["row"]
    p   = seed42["placebo"]
    conf_json = {
        "dataset": "MetaTool-Binary",
        "model": "microsoft/Phi-3.5-mini-instruct",
        "load_mode": load_mode,
        "precision_note": mem_note,
        "candidate_strings": YESNO,
        "seed42_fp16": s42,
        "seed42_placebo": p,
        "seed42_4bit_reference": ref4bit.get(42, None),
    }
    (OUT_DIR / "metatool_binary_fp16_confirmation.json").write_text(
        json.dumps(conf_json, indent=2, ensure_ascii=False), encoding="utf-8")

    if multiseed_rows:
        nets = [r["test_net"] for r in multiseed_rows]
        deltas = [r["test_delta_acc"] for r in multiseed_rows]
        aucs = [r["router_val_auc"] for r in multiseed_rows if r["router_val_auc"]]
        agg = {
            "n_seeds": len(multiseed_rows),
            "net_mean": round(float(np.mean(nets)), 3),
            "net_std":  round(float(np.std(nets)), 3),
            "net_min":  int(np.min(nets)), "net_max": int(np.max(nets)),
            "n_positive_net": int(sum(1 for x in nets if x > 0)),
            "delta_acc_mean": round(float(np.mean(deltas)), 5),
            "router_val_auc_mean": round(float(np.mean(aucs)), 5) if aucs else None,
        }
        (OUT_DIR / "metatool_binary_fp16_multiseed_results.json").write_text(
            json.dumps({"load_mode": load_mode, "aggregate": agg,
                        "per_seed": multiseed_rows,
                        "four_bit_reference_per_seed": ref4bit},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        agg = None

    write_md(load_mode, seed42, multiseed_rows, agg, ref4bit, mem_note)
    return agg


def write_md(load_mode, seed42, multiseed_rows, agg, ref4bit, mem_note):
    s = seed42["row"]; p = seed42["placebo"]
    L = []; A = L.append
    A("# MetaTool-Binary SAKIKO — fp16 Confirmation\n")
    A("> **CAVEATS**")
    A("> - MetaTool-Binary, NOT W2C-4way. Binary `tool_call` / `no_tool`, readout `Yes`/`No`.")
    A("> - Over-calling channel (nt_tc) only. NOT numerically comparable to W2C.")
    A(f"> - This run: **{load_mode}** (no 4-bit). Activations + baseline re-derived in {load_mode}.\n")
    A(f"**Precision / memory:** {mem_note}\n")

    # 4-bit reference for seed 42
    r4 = ref4bit.get(42)
    def cell(v): return "—" if v is None else v
    if r4:
        r4_nttc = f"{r4['nt_tc_before']}→{r4['nt_tc_after']}"
        r4_ftc  = f"{r4['false_toolcall_rate_before']:.3f}→{r4['false_toolcall_rate_after']:.3f}"
        r4_cfg  = f"L{r4['selected_inj_layer']}/{r4['selected_alpha']}/{r4['selected_threshold']}"
    else:
        r4_nttc = r4_ftc = r4_cfg = None
    s_nttc = f"{s['nt_tc_before']}→{s['nt_tc_after']}"
    s_ftc  = f"{s['false_toolcall_rate_before']:.3f}→{s['false_toolcall_rate_after']:.3f}"
    s_cfg  = f"L{s['selected_inj_layer']}/{s['selected_alpha']}/{s['selected_threshold']}"

    A("## Step 3 — seed=42 fp16 (vs 4-bit reference)\n")
    A("| Metric | 4-bit NF4 | **fp16** |")
    A("|---|---|---|")
    A(f"| Baseline test acc | {cell(r4 and r4['test_baseline_acc'])} | **{s['test_baseline_acc']:.4f}** |")
    A(f"| SAKIKO test acc | {cell(r4 and r4['test_sakiko_acc'])} | **{s['test_sakiko_acc']:.4f}** |")
    A(f"| Δacc | {cell(r4 and r4['test_delta_acc'])} | {s['test_delta_acc']:+.4f} |")
    A(f"| Fixed | {cell(r4 and r4['test_fixed'])} | {s['test_fixed']} |")
    A(f"| Broke | {cell(r4 and r4['test_broke'])} | {s['test_broke']} |")
    A(f"| **Net** | {cell(r4 and r4['test_net'])} | **{s['test_net']:+d}** |")
    A(f"| nt_tc before→after | {cell(r4_nttc)} | {s_nttc} |")
    A(f"| FTC rate before→after | {cell(r4_ftc)} | {s_ftc} |")
    A(f"| Selected cfg (L/α/thr) | {cell(r4_cfg)} | {s_cfg} |")
    A(f"| Router val AUC | {cell(r4 and r4['router_val_auc'])} | {s['router_val_auc']} |")
    A("")

    if p:
        A("### seed=42 fp16 placebo\n")
        A("| Direction | Net | Fixed | Broke | nt_tc after |")
        A("|---|---|---|---|---|")
        for name in ["real", "reverse", "random1"]:
            m = p[name]
            A(f"| {name} | **{m['net']:+d}** | {m['fixed']} | {m['broke']} | {m['nt_tc_after']} |")
        if "random_distribution" in p:
            d = p["random_distribution"]
            A("")
            A(f"**Random distribution ({d['n_random']} dirs):** real net {d['real_net']} | "
              f"random mean {d['random_mean']} ± {d['random_std']} (max {d['random_max']}, "
              f"min {d['random_min']}) | real rank {d['real_rank_among_random']}/{d['n_random']} "
              f"({d['real_percentile']}%ile).")
        A("")

    if multiseed_rows:
        A("## Step 4 — fp16 multi-seed\n")
        A("| Seed | Base acc | SAKIKO acc | Δacc | Fixed | Broke | Net | nt_tc→ | Cfg | val AUC | 4-bit Net |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in multiseed_rows:
            r4s = ref4bit.get(r["seed"])
            A(f"| {r['seed']} | {r['test_baseline_acc']:.4f} | {r['test_sakiko_acc']:.4f} | "
              f"{r['test_delta_acc']:+.4f} | {r['test_fixed']} | {r['test_broke']} | "
              f"**{r['test_net']:+d}** | {r['nt_tc_before']}→{r['nt_tc_after']} | "
              f"L{r['selected_inj_layer']}/{r['selected_alpha']}/{r['selected_threshold']} | "
              f"{r['router_val_auc']} | {r4s['test_net'] if r4s else '—'} |")
        A("")
        A(f"**fp16 aggregate:** Net mean = {agg['net_mean']} ± {agg['net_std']} "
          f"(min {agg['net_min']}, max {agg['net_max']}); positive-net = "
          f"{agg['n_positive_net']}/{agg['n_seeds']}; mean Δacc = {agg['delta_acc_mean']:+.4f}; "
          f"mean router val AUC = {agg['router_val_auc_mean']}.\n")
        # 4-bit aggregate for comparison
        ref_nets = [ref4bit[k]["test_net"] for k in ref4bit] if ref4bit else []
        if ref_nets:
            A(f"**4-bit aggregate (audit):** Net mean = {round(float(np.mean(ref_nets)),3)} ± "
              f"{round(float(np.std(ref_nets)),3)} (positive-net {sum(1 for x in ref_nets if x>0)}/{len(ref_nets)}).\n")

    # Conclusions
    A("## Conclusions\n")
    survives = s["test_net"] > 0 and s["nt_tc_after"] < s["nt_tc_before"]
    A(f"1. **Does the positive result survive fp16?** "
      f"{'**Yes**' if survives else '**No**'} — seed=42 fp16 Net = {s['test_net']:+d}, "
      f"nt_tc {s['nt_tc_before']}→{s['nt_tc_after']}, acc {s['test_baseline_acc']:.4f}→{s['test_sakiko_acc']:.4f}.\n")

    if multiseed_rows:
        pos = agg["n_positive_net"]; n = agg["n_seeds"]
        A(f"2. **Sign positive across seeds (fp16)?** "
          f"{'**Yes, all '+str(n)+'/'+str(n)+'**' if pos==n else f'**{pos}/{n}**'} — "
          f"Net = {agg['net_mean']} ± {agg['net_std']} (min {agg['net_min']}).\n")
        all_down = all(r["nt_tc_after"] < r["nt_tc_before"] for r in multiseed_rows)
        A(f"3. **Over-calling reduction remains?** "
          f"{'**Yes** on every seed' if all_down else '**Mostly**'} — "
          f"nt_tc decreases in {sum(1 for r in multiseed_rows if r['nt_tc_after']<r['nt_tc_before'])}/{n} seeds.\n")
        # closeness
        ref_nets = [ref4bit[r['seed']]['test_net'] for r in multiseed_rows if ref4bit.get(r['seed'])]
        fp_nets  = [r['test_net'] for r in multiseed_rows if ref4bit.get(r['seed'])]
        if ref_nets:
            diff = round(float(np.mean(fp_nets) - np.mean(ref_nets)), 2)
            A(f"4. **fp16 close to 4-bit?** fp16 Net mean {agg['net_mean']} vs 4-bit "
              f"{round(float(np.mean(ref_nets)),3)} (Δ = {diff:+}). "
              f"{'Consistent — the effect is not a quantization artifact.' if agg['n_positive_net']==n else 'Partly consistent.'}\n")
    else:
        A("2-4. Multi-seed not run (seed=42 fp16 did not meet the go criterion). "
          "See diagnosis above.\n")

    A("5. **Recommended next step:**")
    if survives and multiseed_rows and agg["n_positive_net"] == agg["n_seeds"]:
        A("   - The MetaTool-Binary over-calling result is **precision-robust** (positive in both")
        A("     4-bit and fp16, all seeds). Report it as a **cross-dataset binary over-calling")
        A("     extension** around W2C — with the Part-2 router-vs-direction nuance stated.")
        A("   - Then proceed to **ToolSandbox-derived** multi-class annotation, where direction")
        A("     specificity can be tested beyond the binary Yes/No readout.")
    elif survives:
        A("   - seed=42 fp16 is positive; report as a **confirmed quantization-robust pilot** and")
        A("     proceed cautiously. Consider ToolSandbox next.")
    else:
        A("   - fp16 does **not** confirm the 4-bit result. **Downgrade MetaTool to a quantized")
        A("     pilot only** and do not claim external validation; investigate before ToolSandbox.")
    A("")
    (OUT_DIR / "metatool_binary_fp16_confirmation_summary.md").write_text(
        "\n".join(L), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="phi35")
    ap.add_argument("--load_mode", default="fp16", choices=["fp16", "4bit"])
    ap.add_argument("--obs_layer", type=int, default=OBS_LAYER)
    ap.add_argument("--force_acts", action="store_true")
    ap.add_argument("--n_random", type=int, default=N_RANDOM,
                    help="random directions for seed=42 distribution (Step 5)")
    args = ap.parse_args()

    t0 = time.time()
    records, golds, thought = load_all_records()
    log.info("Loaded %d MetaTool-Binary records", len(records))

    model, tok, device, model_path = load_model_mode(args.model_key, args.load_mode)
    mem_note = (f"Phi-3.5-mini loaded in {args.load_mode}; "
                f"VRAM reserved {torch.cuda.memory_reserved(0)/1e9:.2f}GB / "
                f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB on RTX 4090.")

    # fp16 activations (separate cache)
    acts_all = extract_acts_fp16(model, tok, thought, args.obs_layer, device,
                                 args.load_mode, force=args.force_acts)

    # fp16 baseline (Yes/No) re-scored from scratch — precision changes predictions
    log.info("Re-scoring %s baseline (Yes/No) on all %d samples ...",
             args.load_mode, len(records))
    all_oids = list(range(len(records)))
    fp_bp = rescore_baseline(model, tok, all_oids, thought, YESNO, device)
    base_acc_all = sum(fp_bp[o] == golds[o] for o in all_oids) / len(all_oids)
    nt_tc_all = sum(1 for o in all_oids if golds[o] == "no_tool" and fp_bp[o] == "tool_call")
    log.info("%s baseline (all 1040): acc=%.4f  nt_tc(over-call)=%d",
             args.load_mode, base_acc_all, nt_tc_all)

    ref4bit = load_4bit_reference()

    # ── Step 3: seed=42 first, with placebo (real/reverse/1 random) ──────────
    log.info("================  Step 3: seed=42 %s  ================", args.load_mode)
    seed42 = run_seed(model, tok, records, golds, thought, fp_bp, acts_all, 42, device,
                      want_placebo=True, n_random=args.n_random)
    s = seed42["row"]
    log.info("seed=42 %s: acc %.4f→%.4f  fixed=%d broke=%d net=%+d  nt_tc %d→%d  "
             "cfg=L%d a=%.1f thr=%.2f  valAUC=%s",
             args.load_mode, s["test_baseline_acc"], s["test_sakiko_acc"],
             s["test_fixed"], s["test_broke"], s["test_net"],
             s["nt_tc_before"], s["nt_tc_after"],
             s["selected_inj_layer"], s["selected_alpha"], s["selected_threshold"],
             s["router_val_auc"])
    pr = seed42["placebo"]
    log.info("seed=42 placebo: real net=%+d | reverse net=%+d | random1 net=%+d",
             pr["real"]["net"], pr["reverse"]["net"], pr["random1"]["net"])
    if "random_distribution" in pr:
        d = pr["random_distribution"]
        log.info("seed=42 random dist (%d): real=%d rank=%d/%d mean=%.2f max=%d",
                 d["n_random"], d["real_net"], d["real_rank_among_random"],
                 d["n_random"], d["random_mean"], d["random_max"])

    # ── Step 4: decide whether to run multi-seed ─────────────────────────────
    go = (s["test_net"] > 0) and (s["nt_tc_after"] < s["nt_tc_before"])
    multiseed_rows = []
    if go:
        log.info("================  Step 4: %s multi-seed  ================", args.load_mode)
        multiseed_rows.append(s)  # seed 42 already done
        for seed in [x for x in SEEDS if x != 42]:
            res = run_seed(model, tok, records, golds, thought, fp_bp, acts_all,
                           seed, device, want_placebo=False)
            if res is None:
                log.warning("seed %d: insufficient structure, skipped", seed)
                continue
            rr = res["row"]
            multiseed_rows.append(rr)
            log.info("seed=%d %s: acc %.4f→%.4f  fixed=%d broke=%d net=%+d  nt_tc %d→%d  "
                     "cfg=L%d a=%.1f thr=%.2f  valAUC=%s",
                     seed, args.load_mode, rr["test_baseline_acc"], rr["test_sakiko_acc"],
                     rr["test_fixed"], rr["test_broke"], rr["test_net"],
                     rr["nt_tc_before"], rr["nt_tc_after"],
                     rr["selected_inj_layer"], rr["selected_alpha"], rr["selected_threshold"],
                     rr["router_val_auc"])
    else:
        log.warning("seed=42 %s did NOT meet go-criterion (net>0 and nt_tc down). "
                    "Skipping multi-seed; writing diagnosis.", args.load_mode)

    agg = write_reports(args.load_mode, seed42, multiseed_rows, ref4bit, mem_note)

    log.info("================  DONE in %.1f min  ================", (time.time() - t0) / 60)
    if agg:
        log.info("fp16 multiseed: net mean=%.2f (%d/%d positive)",
                 agg["net_mean"], agg["n_positive_net"], agg["n_seeds"])


if __name__ == "__main__":
    main()
