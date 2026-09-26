#!/usr/bin/env python3
"""Qwen3-8B SAKIKO Stage 2A V2 — DEV-only destination-resolved behavioral calibration.

Applies the frozen dose rule  h' = h + q * s_c * d  at the frozen L_inj site to
Router-gated DEV rows, re-scores all four response modes with the frozen readout,
and resolves the complete destination flow.

All scientific definitions are inherited by IMPORTING the committed V1 runner
(scoring, prompt rendering, candidate tokenization) and the committed V2 runner
(verified memory/attention path). This script adds an intervention hook and
destination bookkeeping, nothing else.

The sealed evaluation partition is never opened.

Modes:
  --preflight          configuration verification only, no intervention
  --pilot              fixed engineering pilot
  --dose-calibration   Block 4: d_grad and DiffMean across the frozen q grid
  --controls           Block 6: control arms at each channel's selected dose
  --analyze            Blocks 5/7/8: gate, score comparator, cost, outputs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
V1D = ROOT / "final/results/qwen3_stage0_1"
V2D = ROOT / "final/results/qwen3_stage0_1_v2"
LCD = ROOT / "final/results/qwen3_stage1_5_layer_control"
S15 = ROOT / "final/results/qwen3_stage1_5"
DOSE = ROOT / "final/results/qwen3_stage2a_dose"
OUT = ROOT / "final/results/qwen3_stage2a_v2"
SELF = ROOT / "scripts/qwen3_stage2a_v2_dev_calibration.py"

HASHES = OUT / "QWEN3_STAGE2A_V2_HASHES.json"
PREFLIGHT = OUT / "QWEN3_STAGE2A_V2_PREFLIGHT.json"
PILOT = OUT / "QWEN3_STAGE2A_V2_PILOT.json"
RECORDS = OUT / "QWEN3_STAGE2A_V2_DEV_RECORDS.jsonl"
RETRY = OUT / "QWEN3_STAGE2A_V2_RETRY_LEDGER.json"
LOG = OUT / "QWEN3_STAGE2A_V2_LOG.jsonl"
SELECTED = OUT / "QWEN3_STAGE2A_V2_SELECTED_DOSE.json"

# ---- frozen Stage 2A V2 constants ----
Q_GRID = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260801
DEV_RANDOM_SEEDS = [20260811, 20260812, 20260813, 20260814,
                    20260815, 20260816, 20260817, 20260818]
WRONG_LAYER = 26                      # L_obs; the other frozen site
PILOT_PER_CHANNEL = 4
MIN_HEADROOM_BYTES = int(1.5 * 2 ** 30)
FORMAL_RUNTIME_CEILING_SECONDS = 3 * 60 * 60
MIN_FREE_DISK_BYTES = int(1.5 * 2 ** 30)
# Relative tolerance on the injected perturbation norm. float32 carries about
# 1.2e-7 relative precision, so an absolute tolerance would fail on the tensor
# representation alone rather than on any budget error.
PERTURBATION_RTOL = 1e-5

GATE = {"source_exits_min": 10, "target_hit_min": 0.50,
        "clean_collateral_max": 0.05}


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def write_json_atomic(p: Path, obj: Any) -> None:
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(t, p)


def log_event(event: str, **kw: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "event": event, **kw}, sort_keys=True, default=str) + "\n")


def append_retry(mode: str, code: int, reason: str | None) -> None:
    o = json.loads(RETRY.read_text()) if RETRY.exists() else {"schema_version": 1, "invocations": []}
    o["invocations"].append({"utc": utc_now(), "mode": mode, "exit_code": code, "reason": reason,
                             "script_sha256": sha256_file(SELF),
                             "script_commit": subprocess.check_output(
                                 ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    write_json_atomic(RETRY, o)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self):
        self.v1 = load_module(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1")
        self.v2 = load_module(ROOT / "scripts/qwen3_8b_stage0_1_v2.py", "q3v2")
        self.p2 = json.loads((V2D / "QWEN3_8B_STAGE0_1_PROTOCOL_V2.json").read_text())
        self.L_INJ = self.p2["layer_mapping"]["new_L_inj"]
        self.L_OBS = self.p2["layer_mapping"]["new_L_obs"]
        self.MODES = self.p2["readout"]["mode_order"]
        self.dose = json.loads((DOSE / "QWEN3_STAGE2A_DOSE_DECLARATION.json").read_text())
        self.rm = json.loads((V2D / "QWEN3_V2_ROUTER_METRICS.json").read_text())
        self.rows = [json.loads(l) for l in (V1D / "QWEN3_STAGE1_BASELINE_ROWS.jsonl").open()]
        self.split = {r["sample_id"]: r["v2_split"]
                      for r in csv.DictReader((V2D / "QWEN3_V2_SPLIT_INDEX.csv").open())}
        self.ledger = [r for r in csv.DictReader((V2D / "QWEN3_V2_SUPPORT_LEDGER.csv").open())
                       if r["support_eligible"] == "True"]
        self.channels = [c["channel"] for c in self.ledger]
        from safetensors import safe_open
        with safe_open(V1D / "QWEN3_STAGE1_ACTIVATIONS.safetensors", framework="numpy") as f:
            order = json.loads(f.metadata()["sample_order_json"])
            self.A26 = f.get_tensor(f"activations_L{self.L_OBS}")
        assert [o["sample_id"] for o in order] == [r["sample_id"] for r in self.rows]
        self.idx = {r["project_index"]: i for i, r in enumerate(self.rows)}
        self.dgrad, self.d26 = {}, {}
        with safe_open(S15 / "QWEN3_STAGE1_5_DGRAD.safetensors", framework="numpy") as f:
            for k in f.keys():
                self.dgrad[k.replace("d_grad__", "")] = f.get_tensor(k)
        for c in self.ledger:
            with safe_open(V2D / f"directions/{c['channel']}.safetensors", framework="numpy") as f:
                self.d26[c["channel"]] = f.get_tensor("direction")
        self.byid = {r["sample_id"]: r for r in self.rows}

    def meta(self, ch):
        c = next(x for x in self.ledger if x["channel"] == ch)
        return c["gold"], c["source"], self.rm["channels"][ch]["selected_tau"], self.dose["per_channel"][ch]["s_c"]

    def router_prob(self, ch, rows):
        from safetensors import safe_open
        with safe_open(V2D / f"routers/{ch}.safetensors", framework="numpy") as f:
            sm = f.get_tensor("scaler_mean").astype(np.float64)
            ss = f.get_tensor("scaler_scale").astype(np.float64)
            co = f.get_tensor("classifier_coef").astype(np.float64)
            ic = f.get_tensor("classifier_intercept").astype(np.float64)
        X = self.A26[[self.idx[r["project_index"]] for r in rows]].astype(np.float64)
        return 1.0 / (1.0 + np.exp(-(((X - sm) / ss) @ co + ic[0])))

    def dev_pred_eq_source(self, ch):
        _, src, _, _ = self.meta(ch)
        sel = [r for r in self.rows if self.split[r["sample_id"]] == "dev" and r["prediction"] == src]
        return sorted(sel, key=lambda r: r["sample_id"])

    def dev_routed(self, ch):
        _, _, tau, _ = self.meta(ch)
        pop = self.dev_pred_eq_source(ch)
        pr = self.router_prob(ch, pop)
        return [r for r, p in zip(pop, pr) if p >= tau], pop, pr

    def dev_channel_errors(self, ch):
        g, s, _, _ = self.meta(ch)
        sel = [r for r in self.rows if self.split[r["sample_id"]] == "dev"
               and r["gold"] == g and r["prediction"] == s]
        return sorted(sel, key=lambda r: r["sample_id"])


def random_unit(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------- intervention
def make_hook(model, layer: int, delta_vec):
    """Add a constant vector to the MLP output at every position of `layer`."""
    import torch
    module = model.model.layers[layer].mlp
    state = {"delta": delta_vec}

    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if state["delta"] is None:
            return out
        d = state["delta"].to(dtype=t.dtype, device=t.device)
        new = t + d
        return (new,) + out[1:] if isinstance(out, tuple) else new

    h = module.register_forward_hook(hook)
    return h, state


def score_with_intervention(ctx, model, tokenizer, device, sample, layer, delta):
    """Re-score all four modes under a constant additive intervention."""
    import torch
    v1 = ctx.v1
    prompt, _, _ = v1.render_prompt(tokenizer, sample)
    handle, state = make_hook(model, layer, delta)
    try:
        scores, lengths = {}, {}
        for m in ctx.MODES:
            pids, cids = v1.candidate_ids(tokenizer, prompt, sample["answers"][m])
            s, _ = v1.score_candidate(model, device, pids, cids, None)
            scores[m] = s
            lengths[m] = len(cids)
    finally:
        handle.remove()
    vals = np.asarray([scores[m] for m in ctx.MODES], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("non-finite intervened score")
    order = np.argsort(-vals, kind="stable")
    return scores, ctx.MODES[int(order[0])], ctx.MODES[int(order[1])], lengths


def run_arm(ctx, model, tokenizer, device, data, ch, arm, q, direction, layer, population, fh):
    """Score one arm at one dose over `population`; append per-sample records."""
    import torch
    g, src, tau, s_c = ctx.meta(ch)
    absn = q * s_c
    delta = None
    achieved_norm = 0.0
    if direction is not None and q != 0.0:
        # Build the perturbation in float64 and cast once, so the only error is
        # the unavoidable float32 representation of the injected tensor.
        d64 = np.ascontiguousarray(direction.astype(np.float64))
        d64 = d64 / np.linalg.norm(d64)
        delta = torch.from_numpy((d64 * absn).astype(np.float32)).to("cuda:0")
        achieved_norm = float(torch.linalg.vector_norm(delta.double()).item())
        # float32 has ~1.2e-7 relative precision; the budget check must be
        # relative, not absolute, or it fails on representation alone.
        rel = abs(achieved_norm - absn) / max(1.0, absn)
        if rel > PERTURBATION_RTOL:
            raise RuntimeError(
                f"perturbation norm mismatch: achieved {achieved_norm} vs target {absn} "
                f"(relative {rel:.3e} > {PERTURBATION_RTOL:.3e})")
    recs = []
    for n, r in enumerate(population):
        sample = data.sample(r["project_index"])
        sc, pred, runner, lengths = score_with_intervention(
            ctx, model, tokenizer, device, sample, layer, delta)
        rec = {"channel": ch, "arm": arm, "q": q, "abs_delta_norm": absn, "layer": layer,
               "sample_id": r["sample_id"], "project_index": r["project_index"],
               "gold": r["gold"], "pred_base": r["prediction"],
               "scores_base": r["scores"], "scores_int": sc,
               "pred_int": pred, "runner_up_int": runner,
               "execution_order": n}
        recs.append(rec)
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return recs


# ---------------------------------------------------------------- metrics
def destination_metrics(ctx, ch, recs, all_dev_effect):
    """recs: intervened records for routed rows. all_dev_effect maps sample_id->pred_int."""
    g, src, _, _ = ctx.meta(ch)
    errs = ctx.dev_channel_errors(ch)
    err_ids = {r["sample_id"] for r in errs}
    n_err = len(errs)
    src_exits = gold_arr = w2w = 0
    for r in errs:
        pi = all_dev_effect.get(r["sample_id"], r["prediction"])
        if pi != src:
            src_exits += 1
            if pi == g:
                gold_arr += 1
            else:
                w2w += 1
    target_hit = (gold_arr / src_exits) if src_exits > 0 else None
    tg_count = gold_arr - w2w
    tg_rate = tg_count / n_err if n_err else None
    dev = [r for r in ctx.rows if ctx.split[r["sample_id"]] == "dev"]
    fixed = broke = 0
    base_correct = [r for r in dev if r["prediction"] == r["gold"]]
    coll = 0
    for r in dev:
        pi = all_dev_effect.get(r["sample_id"], r["prediction"])
        was = r["prediction"] == r["gold"]
        now = pi == r["gold"]
        if not was and now:
            fixed += 1
        if was and not now:
            broke += 1
    for r in base_correct:
        pi = all_dev_effect.get(r["sample_id"], r["prediction"])
        if pi != r["gold"]:
            coll += 1
    return {"n_channel_error": n_err, "source_exits": src_exits, "gold_arrivals": gold_arr,
            "wrong_to_wrong": w2w, "target_hit": target_hit,
            "target_gain_count": tg_count, "target_gain_rate": tg_rate,
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "baseline_correct_n": len(base_correct), "collateral_count": coll,
            "clean_collateral_rate": coll / len(base_correct) if base_correct else None}


def score_deltas(ctx, ch, recs):
    g, src, _, _ = ctx.meta(ch)
    errs = {r["sample_id"] for r in ctx.dev_channel_errors(ch)}
    dm, dg, ds, pos = [], [], [], 0
    other = {}
    for r in recs:
        if r["sample_id"] not in errs:
            continue
        b, i = r["scores_base"], r["scores_int"]
        m0 = b[g] - b[src]; m1 = i[g] - i[src]
        dm.append(m1 - m0); dg.append(i[g] - b[g]); ds.append(i[src] - b[src])
        pos += int((m1 - m0) > 0)
        for mode in ctx.MODES:
            if mode not in (g, src):
                other.setdefault(mode, []).append(i[mode] - b[mode])
    if not dm:
        return {"n": 0}
    return {"n": len(dm),
            "delta_margin_median": float(np.median(dm)), "delta_margin_mean": float(np.mean(dm)),
            "delta_e_gold_median": float(np.median(dg)), "delta_e_source_median": float(np.median(ds)),
            "delta_other_modes_median": {k: float(np.median(v)) for k, v in other.items()},
            "positive_margin_change_fraction": pos / len(dm)}


def boot_ci(values, stat, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return [None, None]
    arr = np.asarray(values)
    out = np.empty(draws)
    for b in range(draws):
        out[b] = stat(arr[rng.integers(0, n, n)])
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


# ---------------------------------------------------------------- modes
def preflight() -> None:
    ctx = Ctx()
    checks = {}
    man = json.loads(HASHES.read_text())
    bad = [r for r, e in man["artifact_sha256"].items()
           if not (ROOT / r).is_file() or sha256_file(ROOT / r) != e]
    if bad:
        raise RuntimeError(f"frozen artifact mismatch: {bad}")
    checks["frozen_artifacts"] = len(man["artifact_sha256"])
    for name, exp in ctx.p2["model"]["file_sha256"].items():
        p = ctx.v1.MODEL_DIR / name
        if not p.exists() or sha256_file(p.resolve()) != exp:
            raise RuntimeError(f"model file mismatch {name}")
    checks["model_files"] = len(ctx.p2["model"]["file_sha256"])
    import torch, transformers, tokenizers
    for got, want in ((transformers.__version__, ctx.p2["model"]["transformers"]),
                      (tokenizers.__version__, ctx.p2["model"]["tokenizers"]),
                      (torch.__version__, ctx.p2["model"]["torch"])):
        if got != want:
            raise RuntimeError(f"version mismatch {got} != {want}")
    atol = ctx.p2["direction"]["norm_atol"]
    for ch in ctx.channels:
        for tag, v in (("d_grad", ctx.dgrad[ch]), ("d_L26", ctx.d26[ch])):
            nn = float(np.linalg.norm(v.astype(np.float64)))
            if not np.all(np.isfinite(v)) or abs(nn - 1) > atol:
                raise RuntimeError(f"direction not finite/unit: {tag} {ch}")
    checks["directions_unit_norm"] = True
    routed_counts = {}
    for ch in ctx.channels:
        _, _, tau, _ = ctx.meta(ch)
        routed, pop, pr = ctx.dev_routed(ch)
        committed = {gg["tau"]: gg for gg in ctx.rm["channels"][ch]["tau_grid"]}
        for t, gg in committed.items():
            if int((pr >= t).sum()) != gg["routed"]:
                raise RuntimeError(f"router routed-count mismatch {ch} tau={t}")
        routed_counts[ch] = {"pred_eq_source": len(pop), "routed_at_tau": len(routed), "tau": tau}
    checks["router_reproduces_committed_counts"] = True
    checks["dev_routed"] = routed_counts
    dosetab = {}
    for ch in ctx.channels:
        _, _, _, s_c = ctx.meta(ch)
        if s_c != ctx.dose["per_channel"][ch]["s_c"]:
            raise RuntimeError("s_c mismatch")
        dosetab[ch] = {str(q): q * s_c for q in Q_GRID}
    checks["dose_table"] = dosetab
    dev = [r for r in ctx.rows if ctx.split[r["sample_id"]] == "dev"]
    checks["dev_population"] = {"n": len(dev),
                                "order_sha256": ctx.v1.index_order_hash(sorted(r["sample_id"] for r in dev))}
    for nm in ("QWEN3_STAGE2A_V2_DEV_RECORDS.jsonl", "QWEN3_STAGE2A_V2_RESULTS.json"):
        if (OUT / nm).exists():
            raise RuntimeError(f"namespace not empty: {nm}")
    checks["namespace_empty"] = True
    checks["sealed_evaluation"] = {"opened": False, "rows_touched": 0}
    payload = {"schema_version": 1, "status": "PASS", "utc": utc_now(), "checks": checks,
               "intervention_run": False, "evaluation_accessed": False}
    write_json_atomic(PREFLIGHT, payload)
    append_retry("preflight", 0, None)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def pilot() -> None:
    import torch
    ctx = Ctx()
    if json.loads(PREFLIGHT.read_text()).get("status") != "PASS":
        raise RuntimeError("preflight missing")
    model, tokenizer, device, env = ctx.v1.load_model()
    env["chunk_checkpointed_attention"] = ctx.v2.install_chunk_checkpointed_attention(ctx.v1)
    data = ctx.v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    detail, cells = [], 0
    tmp = OUT / "pilot_records.jsonl"
    with tmp.open("w") as fh:
        for ch in ctx.channels:
            g, src, tau, s_c = ctx.meta(ch)
            errs = ctx.dev_channel_errors(ch)[:PILOT_PER_CHANNEL]
            # add the longest DEV routed sequence to exercise the attention backend
            routed, _, _ = ctx.dev_routed(ch)
            lens = []
            for r in routed:
                s = data.sample(r["project_index"])
                p, _, _ = ctx.v1.render_prompt(tokenizer, s)
                lens.append((len(tokenizer.encode(p, add_special_tokens=False)), r))
            longest = max(lens, key=lambda x: x[0])[1]
            pop = errs + [longest]
            arms = [("zero", None, ctx.L_INJ, 0.0), ("d_grad", ctx.dgrad[ch], ctx.L_INJ, 1.0),
                    ("d_L26_diffmean", ctx.d26[ch], ctx.L_INJ, 1.0),
                    ("reverse_d_grad", -ctx.dgrad[ch], ctx.L_INJ, 1.0),
                    ("wrong_layer_d_grad", ctx.dgrad[ch], WRONG_LAYER, 1.0),
                    ("ungated_d_grad", ctx.dgrad[ch], ctx.L_INJ, 1.0)]
            for i, sd in enumerate(DEV_RANDOM_SEEDS):
                arms.append((f"dev_random_{i}", random_unit(sd, 4096), ctx.L_INJ, 1.0))
            for arm, d, layer, q in arms:
                recs = run_arm(ctx, model, tokenizer, device, data, ch, arm, q, d, layer, pop, fh)
                cells += len(recs)
                if arm == "zero":
                    same = all(all(abs(r["scores_int"][m] - r["scores_base"][m]) == 0.0 for m in ctx.MODES)
                               for r in recs)
                    detail.append({"channel": ch, "check": "zero_control_exact_equality", "pass": bool(same)})
                    if not same:
                        raise RuntimeError(f"zero control not exactly equal to baseline for {ch}")
            # deterministic replay on one sample
            s = data.sample(pop[0]["project_index"])
            dvec = torch.from_numpy(np.ascontiguousarray(ctx.dgrad[ch].astype(np.float32))).cuda() * float(1.0 * s_c)
            a = score_with_intervention(ctx, model, tokenizer, device, s, ctx.L_INJ, dvec)
            b = score_with_intervention(ctx, model, tokenizer, device, s, ctx.L_INJ, dvec)
            det = all(a[0][m] == b[0][m] for m in ctx.MODES) and a[1] == b[1]
            detail.append({"channel": ch, "check": "deterministic_replay", "pass": bool(det)})
            if not det:
                raise RuntimeError(f"non-deterministic replay for {ch}")
    elapsed = time.perf_counter() - started
    per_cell = elapsed / max(1, cells)
    peak_res = int(torch.cuda.max_memory_reserved(0))
    total = int(torch.cuda.get_device_properties(0).total_memory)
    headroom = total - peak_res
    routed_tot = sum(len(ctx.dev_routed(ch)[0]) for ch in ctx.channels)
    op_tot = sum(len(ctx.dev_pred_eq_source(ch)) for ch in ctx.channels)
    block4 = per_cell * (routed_tot * (1 + 2 * (len(Q_GRID) - 1)))
    block6 = per_cell * (routed_tot * 10 + op_tot)
    proj = {"seconds_per_cell": per_cell, "routed_total": routed_tot, "pred_eq_source_total": op_tot,
            "projected_block4_seconds": block4, "projected_block6_seconds": block6,
            "projected_this_task_seconds": block4 + block6}
    for K in (39, 79):
        proj[f"projected_formal_K{K}_seconds"] = per_cell * (routed_tot * (K + 6) + op_tot)
    payload = {"schema_version": 1, "status": "PASS", "utc": utc_now(), "cells": cells,
               "elapsed_seconds": elapsed, "checks": detail, "projection": proj,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
               "peak_reserved_bytes": peak_res, "device_total_bytes": total,
               "headroom_bytes": headroom, "min_headroom_bytes": MIN_HEADROOM_BYTES,
               "dev_random_seeds": DEV_RANDOM_SEEDS,
               "wrong_layer": WRONG_LAYER, "environment": env, "evaluation_accessed": False}
    if headroom < MIN_HEADROOM_BYTES:
        payload["status"] = "QWEN3_STAGE2A_V2_ENGINEERING_BLOCKED"
    write_json_atomic(PILOT, payload)
    tmp.unlink(missing_ok=True)
    append_retry("pilot", 0, payload["status"] if payload["status"] != "PASS" else None)
    print(json.dumps({k: v for k, v in payload.items() if k != "environment"}, indent=2, sort_keys=True, default=str))


def dose_calibration() -> None:
    import torch
    ctx = Ctx()
    if json.loads(PILOT.read_text()).get("status") != "PASS":
        raise RuntimeError("pilot not PASS")
    model, tokenizer, device, env = ctx.v1.load_model()
    ctx.v2.install_chunk_checkpointed_attention(ctx.v1)
    data = ctx.v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with RECORDS.open("a") as fh:
        for ch in ctx.channels:
            routed, _, _ = ctx.dev_routed(ch)
            run_arm(ctx, model, tokenizer, device, data, ch, "zero", 0.0, None, ctx.L_INJ, routed, fh)
            log_event("ARM_DONE", channel=ch, arm="zero", q=0.0, n=len(routed))
            for q in Q_GRID[1:]:
                for arm, d in (("d_grad", ctx.dgrad[ch]), ("d_L26_diffmean", ctx.d26[ch])):
                    run_arm(ctx, model, tokenizer, device, data, ch, arm, q, d, ctx.L_INJ, routed, fh)
                    log_event("ARM_DONE", channel=ch, arm=arm, q=q, n=len(routed))
                    print(f"{ch} {arm} q={q} n={len(routed)}", flush=True)
    write_json_atomic(OUT / "QWEN3_STAGE2A_V2_DOSE_RUNTIME.json", {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))})
    append_retry("dose-calibration", 0, None)
    print("DOSE_CALIBRATION_COMPLETE")


def controls() -> None:
    import torch
    ctx = Ctx()
    sel = json.loads(SELECTED.read_text())
    model, tokenizer, device, env = ctx.v1.load_model()
    ctx.v2.install_chunk_checkpointed_attention(ctx.v1)
    data = ctx.v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with RECORDS.open("a") as fh:
        for ch, info in sel["selected"].items():
            if info["selected_q"] is None:
                continue
            q = info["selected_q"]
            routed, pop, _ = ctx.dev_routed(ch)
            arms = [("reverse_d_grad", -ctx.dgrad[ch], ctx.L_INJ, routed),
                    ("wrong_layer_d_grad", ctx.dgrad[ch], WRONG_LAYER, routed),
                    ("ungated_d_grad", ctx.dgrad[ch], ctx.L_INJ, pop)]
            for i, sd in enumerate(DEV_RANDOM_SEEDS):
                arms.append((f"dev_random_{i}", random_unit(sd, 4096), ctx.L_INJ, routed))
            for arm, d, layer, population in arms:
                run_arm(ctx, model, tokenizer, device, data, ch, arm, q, d, layer, population, fh)
                log_event("ARM_DONE", channel=ch, arm=arm, q=q, n=len(population))
                print(f"{ch} {arm} q={q} n={len(population)}", flush=True)
    write_json_atomic(OUT / "QWEN3_STAGE2A_V2_CONTROL_RUNTIME.json", {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))})
    append_retry("controls", 0, None)
    print("CONTROLS_COMPLETE")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("preflight", "pilot", "dose-calibration", "controls"):
        g.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    mode = ("preflight" if a.preflight else "pilot" if a.pilot
            else "dose-calibration" if a.dose_calibration else "controls")
    fn = {"preflight": preflight, "pilot": pilot,
          "dose-calibration": dose_calibration, "controls": controls}[mode]
    try:
        fn()
    except Exception as exc:
        try:
            append_retry(mode, 1, f"{type(exc).__name__}: {exc}")
            log_event("FAILURE", mode=mode, error=str(exc))
        finally:
            raise


if __name__ == "__main__":
    main()
