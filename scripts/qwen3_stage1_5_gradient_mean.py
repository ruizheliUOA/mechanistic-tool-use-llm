#!/usr/bin/env python3
"""Qwen3-8B SAKIKO Stage 1.5 — TRAIN-only injection-site contrast-axis direction.

Tests exactly one closed-form rank-1 estimator:

    d_grad,c = unit( mean_i unit( G_i,g - G_i,p ) )

over the frozen V2 TRAIN error population of each eligible channel, with
gradients taken at the frozen L_inj site. No optimization, weighting, PCA, LDA,
layer search, sign search, clustering, sample filtering or dose tuning.

Every frozen scientific definition is inherited by IMPORTING the committed V1
runner (scoring, gradient object, geometry metrics) and the committed V2 runner
(memory/checkpointing path), so this script adds an estimator and nothing else.

The sealed evaluation partition is never opened.

Modes:
  --preflight      read-only verification + fixed pilot + cost gate
  --run-train      Block 3: TRAIN gradients and d_grad (GPU)
  --dev-transfer   Block 4/5: DEV transfer test and advancement gate (CPU)
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
OUT = ROOT / "final/results/qwen3_stage1_5"
SELF = ROOT / "scripts/qwen3_stage1_5_gradient_mean.py"

PROTOCOL = OUT / "QWEN3_STAGE1_5_PROTOCOL.json"
HASHES = OUT / "QWEN3_STAGE1_5_HASHES.json"
PREFLIGHT = OUT / "QWEN3_STAGE1_5_PREFLIGHT.json"
RETRY = OUT / "QWEN3_STAGE1_5_RETRY_LEDGER.json"
LOG = OUT / "QWEN3_STAGE1_5_LOG.jsonl"

# ---- frozen Stage 1.5 constants, declared before any gradient is computed ----
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_DRAWS = 10000
MAX_PROJECTED_SECONDS = 2 * 60 * 60
MIN_HEADROOM_BYTES = int(1.5 * 2 ** 30)
PILOT_PER_CHANNEL = 2
EXPECTED_TRAIN_ERRORS = {
    "request_for_info__to__tool_call": 448,
    "cannot_answer__to__tool_call": 314,
    "cannot_answer__to__direct": 269,
}
RESULT_NAMES = {
    "QWEN3_STAGE1_5_TRAIN_GRADIENTS.safetensors",
    "QWEN3_STAGE1_5_DGRAD.safetensors",
    "QWEN3_STAGE1_5_TRAIN_GEOMETRY.json",
    "qwen3_stage1_5_results.json",
    "qwen3_stage1_5_dev_comparison.csv",
}


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json_atomic(p: Path, obj: Any) -> None:
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(t, p)


def log_event(event: str, **kw: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "event": event, **kw}, sort_keys=True) + "\n")


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


# ---------------------------------------------------------------- shared setup
def context():
    v1 = load_module(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1")
    v2 = load_module(ROOT / "scripts/qwen3_8b_stage0_1_v2.py", "q3v2")
    proto2 = json.loads((V2D / "QWEN3_8B_STAGE0_1_PROTOCOL_V2.json").read_text())
    rows = [json.loads(l) for l in (V1D / "QWEN3_STAGE1_BASELINE_ROWS.jsonl").open()]
    split = {r["sample_id"]: r["v2_split"] for r in csv.DictReader((V2D / "QWEN3_V2_SPLIT_INDEX.csv").open())}
    ledger = [r for r in csv.DictReader((V2D / "QWEN3_V2_SUPPORT_LEDGER.csv").open())
              if r["support_eligible"] == "True"]
    return v1, v2, proto2, rows, split, ledger


def train_errors(rows, split, gold, source):
    """Frozen V2 TRAIN error population, ordered by immutable UUID."""
    sel = [r for r in rows if split[r["sample_id"]] == "train"
           and r["gold"] == gold and r["prediction"] == source]
    return sorted(sel, key=lambda r: r["sample_id"])


def gradient_object(v1, model, tokenizer, device, sample):
    """The exact V2 sequence-level object: G_i,m at L_inj, full-effect positions."""
    prompt, _, _ = v1.render_prompt(tokenizer, sample)
    G = {}
    for m in v1.MODES:
        pids, cids = v1.candidate_ids(tokenizer, prompt, sample["answers"][m])
        _score, grad, _state = v1.gradient_candidate(model, device, pids, cids)
        G[m] = grad.sum(axis=0).astype(np.float32)
    return np.stack([G[m] for m in v1.MODES]).astype(np.float32)


# ---------------------------------------------------------------- preflight
def preflight() -> None:
    v1, v2, proto2, rows, split, ledger = context()
    checks = {}

    manifest = json.loads(HASHES.read_text())
    bad = [rel for rel, exp in manifest["artifact_sha256"].items()
           if not (ROOT / rel).is_file() or sha256_file(ROOT / rel) != exp]
    if bad:
        raise RuntimeError(f"Stage 1.5 frozen artifact hash mismatch: {bad}")
    checks["stage1_5_frozen_artifacts_verified"] = len(manifest["artifact_sha256"])

    inputs = json.loads((OUT / "QWEN3_STAGE1_5_INPUT_MANIFEST.json").read_text())
    for rel, exp in inputs["input_sha256"].items():
        got = sha256_file(ROOT / rel)
        if got != exp:
            raise RuntimeError(f"inherited input hash drift: {rel}")
    checks["inherited_inputs_verified"] = len(inputs["input_sha256"])

    for name, expected in proto2["model"]["file_sha256"].items():
        p = v1.MODEL_DIR / name
        if not p.exists() or sha256_file(p.resolve()) != expected:
            raise RuntimeError(f"model file mismatch {name}")
    checks["model_files_verified"] = len(proto2["model"]["file_sha256"])

    import torch, transformers, tokenizers
    for got, want, lbl in ((transformers.__version__, proto2["model"]["transformers"], "transformers"),
                           (tokenizers.__version__, proto2["model"]["tokenizers"], "tokenizers"),
                           (torch.__version__, proto2["model"]["torch"], "torch")):
        if got != want:
            raise RuntimeError(f"{lbl} version mismatch: {got} != {want}")
    checks["software_versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                   "tokenizers": tokenizers.__version__}

    counts = {}
    for c in ledger:
        ch = c["channel"]
        n = len(train_errors(rows, split, c["gold"], c["source"]))
        if n != EXPECTED_TRAIN_ERRORS[ch] or n != int(c["train_error_support"]):
            raise RuntimeError(f"TRAIN error population mismatch for {ch}: {n}")
        counts[ch] = n
    checks["train_error_counts"] = counts

    for name in RESULT_NAMES:
        if (OUT / name).exists():
            raise RuntimeError(f"Stage 1.5 result namespace not empty: {name}")
    checks["result_namespace_empty"] = True

    # ---- fixed pilot ----
    model, tokenizer, device, env = v1.load_model()
    env["gradient_checkpointing"] = v2.install_gradient_checkpointing(model, proto2["layer_mapping"]["new_L_inj"])
    env["chunk_checkpointed_attention"] = v2.install_chunk_checkpointed_attention(v1)
    data = v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    pilot = []
    for c in ledger:
        for r in train_errors(rows, split, c["gold"], c["source"])[:PILOT_PER_CHANNEL]:
            s = data.sample(r["project_index"])
            a = gradient_object(v1, model, tokenizer, device, s)
            b = gradient_object(v1, model, tokenizer, device, s)
            ok_shape = a.shape == (len(v1.MODES), proto2["model"]["hidden_size"])
            ok_fin = bool(np.all(np.isfinite(a)))
            ok_det = bool(np.array_equal(a, b))
            if not (ok_shape and ok_fin and ok_det):
                raise RuntimeError(f"pilot failure on {r['sample_id']}: shape={ok_shape} finite={ok_fin} det={ok_det}")
            pilot.append({"channel": c["channel"], "sample_id": r["sample_id"],
                          "shape": list(a.shape), "finite": ok_fin,
                          "deterministic_repeat_bitwise": ok_det})
    elapsed = time.perf_counter() - started
    per_sample = elapsed / (2 * len(pilot))          # each pilot sample computed twice
    total = sum(EXPECTED_TRAIN_ERRORS.values())
    projected = per_sample * total
    peak_alloc = int(torch.cuda.max_memory_allocated(0))
    peak_res = int(torch.cuda.max_memory_reserved(0))
    total_mem = int(torch.cuda.get_device_properties(0).total_memory)
    headroom = total_mem - peak_res
    bytes_per_sample = len(v1.MODES) * proto2["model"]["hidden_size"] * 4
    projected_disk = total * (bytes_per_sample + proto2["model"]["hidden_size"] * 4)

    payload = {
        "schema_version": 1, "status": "PASS", "utc": utc_now(),
        "checks": checks, "pilot": pilot,
        "pilot_samples": len(pilot), "pilot_elapsed_seconds": elapsed,
        "seconds_per_sample": per_sample,
        "projected_train_seconds": projected, "cost_ceiling_seconds": MAX_PROJECTED_SECONDS,
        "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_res,
        "device_total_bytes": total_mem, "headroom_bytes": headroom,
        "min_headroom_bytes": MIN_HEADROOM_BYTES,
        "projected_disk_bytes": projected_disk,
        "environment": env,
        "evaluation_accessed": False, "stage2_artifact_created": False,
    }
    if projected > MAX_PROJECTED_SECONDS or headroom < MIN_HEADROOM_BYTES:
        payload["status"] = "QWEN3_STAGE1_5_ENGINEERING_BLOCKED"
        write_json_atomic(PREFLIGHT, payload)
        append_retry("preflight", 0, payload["status"])
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        print("QWEN3_STAGE1_5_ENGINEERING_BLOCKED")
        return
    write_json_atomic(PREFLIGHT, payload)
    append_retry("preflight", 0, None)
    print(json.dumps({k: v for k, v in payload.items() if k != "pilot"}, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------- Block 3
def run_train() -> None:
    from safetensors.numpy import save_file
    import torch
    pf = json.loads(PREFLIGHT.read_text())
    if pf.get("status") != "PASS":
        raise RuntimeError("valid Stage 1.5 preflight missing")
    v1, v2, proto2, rows, split, ledger = context()
    tol = proto2["geometry"]["contrast_norm_tolerance"]
    dim = proto2["model"]["hidden_size"]

    model, tokenizer, device, env = v1.load_model()
    env["gradient_checkpointing"] = v2.install_gradient_checkpointing(model, proto2["layer_mapping"]["new_L_inj"])
    env["chunk_checkpointed_attention"] = v2.install_chunk_checkpointed_attention(v1)
    data = v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    tensors, meta_records, per_channel, dgrad = {}, [], {}, {}
    for c in ledger:
        ch, g, p = c["channel"], c["gold"], c["source"]
        gi, pi = v1.MODES.index(g), v1.MODES.index(p)
        pop = train_errors(rows, split, g, p)
        if len(pop) != EXPECTED_TRAIN_ERRORS[ch]:
            raise RuntimeError(f"population drift {ch}")
        W, invalid = [], 0
        for ordinal, r in enumerate(pop):
            G = gradient_object(v1, model, tokenizer, device, data.sample(r["project_index"]))
            if not np.all(np.isfinite(G)):
                raise RuntimeError(f"non-finite gradient {r['sample_id']}")
            z = G[gi].astype(np.float64) - G[pi].astype(np.float64)
            nz = float(np.linalg.norm(z))
            if nz <= tol:
                raise RuntimeError(f"contrast norm below frozen tolerance for {r['sample_id']}: {nz}")
            w = (z / nz)
            W.append(w)
            key = f"gradient__{ch}__{ordinal:04d}"
            tensors[key] = G
            tensors[f"w__{ch}__{ordinal:04d}"] = w.astype(np.float32)
            meta_records.append({"tensor_key": key, "w_key": f"w__{ch}__{ordinal:04d}",
                                 "channel": ch, "ordinal": ordinal,
                                 "sample_id": r["sample_id"], "project_index": r["project_index"],
                                 "contrast_norm": nz})
            if (ordinal + 1) % 50 == 0:
                print(f"{ch} {ordinal+1}/{len(pop)}", flush=True)
        Wm = np.stack(W)
        wbar = Wm.mean(0)
        wbar_norm = float(np.linalg.norm(wbar))
        d = (wbar / wbar_norm).astype(np.float32)
        dgrad[ch] = d
        per_channel[ch] = {
            "gold": g, "source": p, "n": len(pop), "invalid_gradients": invalid,
            "wbar_train_norm": wbar_norm,
            "chance_reference_one_over_sqrt_n": 1.0 / np.sqrt(len(pop)),
            "d_grad_norm": float(np.linalg.norm(d.astype(np.float64))),
            "d_grad_sha256": sha_arr(d),
            "population_sha256": v1.index_order_hash([r["sample_id"] for r in pop]),
            "sample_ids": [r["sample_id"] for r in pop],
        }
        log_event("CHANNEL_COMPLETE", channel=ch, n=len(pop), wbar_norm=wbar_norm)
        print(f"  {ch}: n={len(pop)} ||wbar||={wbar_norm:.6f} chance={1/np.sqrt(len(pop)):.6f}", flush=True)

    save_file(tensors, OUT / "QWEN3_STAGE1_5_TRAIN_GRADIENTS.safetensors", metadata={
        "schema_version": "1", "scope": "train_only_injection_site_gradients",
        "layer": str(proto2["layer_mapping"]["new_L_inj"]),
        "site": proto2["layer_mapping"]["site"],
        "aggregation": "full-effect: sum over prompt and candidate positions",
        "mode_order": json.dumps(v1.MODES), "dimension": str(dim),
        "contains_dataset_text": "false",
        "records_json": json.dumps(meta_records, separators=(",", ":"))})
    save_file({f"d_grad__{ch}": dgrad[ch] for ch in dgrad}, OUT / "QWEN3_STAGE1_5_DGRAD.safetensors",
              metadata={"schema_version": "1",
                        "estimator": "unit(mean_i unit(G_i,g - G_i,p)) over frozen V2 TRAIN errors",
                        "layer": str(proto2["layer_mapping"]["new_L_inj"]),
                        "sign_convention": "+d increases e_gold - e_source to first order",
                        "dimension": str(dim), "contains_dataset_text": "false",
                        "channels": json.dumps(sorted(dgrad))})

    chs = sorted(dgrad)
    M = np.stack([dgrad[c].astype(np.float64) for c in chs])
    write_json_atomic(OUT / "QWEN3_STAGE1_5_TRAIN_GEOMETRY.json", {
        "schema_version": 1, "per_channel": per_channel,
        "gram": {"channels": chs, "matrix": (M @ M.T).tolist()},
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "environment": env, "evaluation_accessed": False})
    append_retry("run-train", 0, None)
    print("TRAIN_GRADIENT_MEAN_COMPLETE")


# ---------------------------------------------------------------- Block 4 + 5
def dev_transfer() -> None:
    from safetensors import safe_open
    v1, v2, proto2, rows, split, ledger = context()
    dim = proto2["model"]["hidden_size"]
    denom_tol = proto2["geometry"]["denominator_tolerance"]

    with safe_open(OUT / "QWEN3_STAGE1_5_DGRAD.safetensors", framework="numpy") as f:
        dgrad = {k.replace("d_grad__", ""): f.get_tensor(k) for k in f.keys()}
    with safe_open(LCD / "same_layer_diffmean_vectors.safetensors", framework="numpy") as f:
        d21 = {k.replace("direction__", ""): f.get_tensor(k) for k in f.keys()}
    d26 = {}
    for c in ledger:
        with safe_open(V2D / "directions" / f"{c['channel']}.safetensors", framework="numpy") as f:
            d26[c["channel"]] = f.get_tensor("direction")
    with safe_open(V2D / "QWEN3_V2_RAW_GRADIENTS.safetensors", framework="numpy") as f:
        recs = json.loads(f.metadata()["records_json"])
        devg = {r["tensor_key"]: f.get_tensor(r["tensor_key"]) for r in recs}
    by_ch = {}
    for r in recs:
        if r["variant"] == "full_effect":
            by_ch.setdefault(r["channel"], []).append(r)
    for ch in by_ch:
        by_ch[ch].sort(key=lambda r: r["tensor_key"])

    ests = {"L26_diffmean_frozen": d26, "L21_diffmean_committed": d21, "L21_gradient_mean": dgrad}
    results, csv_rows = {}, []
    for c in ledger:
        ch, g, p = c["channel"], c["gold"], c["source"]
        gi, pi = v1.MODES.index(g), v1.MODES.index(p)
        mats = [devg[r["tensor_key"]] for r in by_ch[ch]]
        Wdev = []
        for m in mats:
            z = m[gi].astype(np.float64) - m[pi].astype(np.float64)
            Wdev.append(z / np.linalg.norm(z))
        Wdev = np.stack(Wdev)
        per = {}
        for name, dd in ests.items():
            d = dd[ch].astype(np.float64)
            vals = [v1.geometry_metrics(m, d, g, p, "full_effect") for m in mats]
            valid = [x for x in vals if x["status"] == "VALID"]
            s = np.array([x["signed_contrast"] for x in valid])
            ac = np.array([x["a_contrast"] for x in valid]); ad = np.array([x["a_dec"] for x in valid])
            ao = np.array([x["a_offaxis"] for x in valid])
            q = np.array([x["q_offaxis"] for x in valid if x.get("q_offaxis") is not None])
            n1 = float(np.median([x["a_contrast_rank1_null_p95"] for x in valid]))
            nr = float(np.median([x["a_dec_rank_matched_null_p95"] for x in valid]))
            per[name] = {"n_valid": len(valid),
                         "mean_signed_s": float(s.mean()), "median_signed_s": float(np.median(s)),
                         "negative_fraction": float(np.mean(s < 0)),
                         "median_a_contrast": float(np.median(ac)), "a_contrast_null_ratio": float(np.median(ac)/n1),
                         "median_a_dec": float(np.median(ad)), "a_dec_null_ratio": float(np.median(ad)/nr),
                         "Q_sum": float(ao.sum()/ad.sum()),
                         "median_q": float(np.median(q)) if len(q) else None,
                         "_s": s, "_ao": ao, "_ad": ad}
        wbar_dev = Wdev.mean(0); wbar_dev = wbar_dev / np.linalg.norm(wbar_dev)
        transfer_cos = float(dgrad[ch].astype(np.float64) @ wbar_dev)

        n = per["L21_gradient_mean"]["n_valid"]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bs = {k: np.empty(BOOTSTRAP_DRAWS) for k in
              ("transfer", "neg_grad", "Q_grad", "dQ26", "dQ21", "dneg26", "dneg21")}
        G = per["L21_gradient_mean"]
        for b in range(BOOTSTRAP_DRAWS):
            i = rng.integers(0, n, n)
            wb = Wdev[i].mean(0); wb = wb / np.linalg.norm(wb)
            bs["transfer"][b] = dgrad[ch].astype(np.float64) @ wb
            bs["neg_grad"][b] = np.mean(G["_s"][i] < 0)
            qg = G["_ao"][i].sum() / G["_ad"][i].sum()
            bs["Q_grad"][b] = qg
            for tag, key in (("L26_diffmean_frozen", "26"), ("L21_diffmean_committed", "21")):
                o = per[tag]
                bs[f"dQ{key}"][b] = qg - (o["_ao"][i].sum() / o["_ad"][i].sum())
                bs[f"dneg{key}"][b] = np.mean(G["_s"][i] < 0) - np.mean(o["_s"][i] < 0)
        ci = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()}

        cond = {
            "1_finite_unit_float32": bool(np.all(np.isfinite(dgrad[ch])) and dgrad[ch].dtype == np.float32
                                          and dgrad[ch].shape == (dim,)
                                          and abs(float(np.linalg.norm(dgrad[ch].astype(np.float64))) - 1) < 1e-5),
            "2_transfer_ci_lower_gt_0": ci["transfer"][0] > 0,
            "3_negative_fraction_lt_0.5": G["negative_fraction"] < 0.5,
            "4_negative_fraction_ci_upper_lt_0.5": ci["neg_grad"][1] < 0.5,
            "5_dQ_vs_L26_ci_upper_lt_0": ci["dQ26"][1] < 0,
            "6_dQ_vs_L21_ci_upper_lt_0": ci["dQ21"][1] < 0,
            "7_dneg_vs_L26_ci_upper_lt_0": ci["dneg26"][1] < 0,
            "8_dneg_vs_L21_ci_upper_lt_0": ci["dneg21"][1] < 0,
            "9_no_inconsistency": True,
        }
        decision = "GRADIENT_MEAN_ADVANCE" if all(cond.values()) else "GRADIENT_MEAN_DO_NOT_ADVANCE"
        results[ch] = {
            "gold": g, "source": p, "n_dev": n,
            "estimators": {k: {kk: vv for kk, vv in per[k].items() if not kk.startswith("_")} for k in per},
            "transfer_cosine": transfer_cos, "transfer_cosine_ci95": ci["transfer"],
            "negative_fraction_ci95": ci["neg_grad"], "Q_sum_ci95": ci["Q_grad"],
            "delta_Q_vs_L26_ci95": ci["dQ26"], "delta_Q_vs_L21_ci95": ci["dQ21"],
            "delta_negative_vs_L26_ci95": ci["dneg26"], "delta_negative_vs_L21_ci95": ci["dneg21"],
            "delta_Q_vs_L26": G["Q_sum"] - per["L26_diffmean_frozen"]["Q_sum"],
            "delta_Q_vs_L21": G["Q_sum"] - per["L21_diffmean_committed"]["Q_sum"],
            "delta_negative_vs_L26": G["negative_fraction"] - per["L26_diffmean_frozen"]["negative_fraction"],
            "delta_negative_vs_L21": G["negative_fraction"] - per["L21_diffmean_committed"]["negative_fraction"],
            "gate_conditions": cond, "decision": decision,
            "cos_dgrad_L26": float(dgrad[ch].astype(np.float64) @ d26[ch].astype(np.float64)),
            "cos_dgrad_L21": float(dgrad[ch].astype(np.float64) @ d21[ch].astype(np.float64)),
        }
        for name in ests:
            e = results[ch]["estimators"][name]
            csv_rows.append({"channel": ch, "estimator": name, "n_dev": n, **e})

    # drift check against committed V2 / layer-control reports
    lc = json.loads((LCD / "layer_control_consolidation.json").read_text())
    drift = []
    for ch in results:
        for name, tag in (("L26_diffmean_frozen", "L26_cross_layer"), ("L21_diffmean_committed", "L21_same_layer")):
            a = results[ch]["estimators"][name]["Q_sum"]
            b = lc["dev_geometry"][ch][tag]["Q_sum"]
            if abs(a - b) > 1e-9:
                drift.append({"channel": ch, "estimator": name, "recomputed": a, "committed": b})
    if drift:
        raise RuntimeError(f"unexplained numerical drift vs committed reports: {drift}")

    advanced = [ch for ch in results if results[ch]["decision"] == "GRADIENT_MEAN_ADVANCE"]
    outcome = ("QWEN3_STAGE1_5_COMPLETE_GRAD_ARM_ADVANCED" if advanced
               else "QWEN3_STAGE1_5_COMPLETE_NO_GRAD_ARM_ADVANCED")
    train_geo = json.loads((OUT / "QWEN3_STAGE1_5_TRAIN_GEOMETRY.json").read_text())
    write_json_atomic(OUT / "qwen3_stage1_5_results.json", {
        "schema_version": 1, "utc": utc_now(),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "paired": True},
        "train_geometry": {"per_channel": {k: {kk: vv for kk, vv in v.items() if kk != "sample_ids"}
                                           for k, v in train_geo["per_channel"].items()},
                           "gram": train_geo["gram"]},
        "dev_transfer": results, "advanced_channels": advanced,
        "committed_report_agreement_verified": True,
        "outcome": outcome, "evaluation_accessed": False, "stage2_artifact_created": False})
    fields = sorted({k for r in csv_rows for k in r})
    with (OUT / "qwen3_stage1_5_dev_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "estimator"] + [k for k in fields if k not in ("channel", "estimator")],
                           lineterminator="\n")
        w.writeheader(); w.writerows(csv_rows)
    append_retry("dev-transfer", 0, None)
    for ch in results:
        r = results[ch]
        print(f"{ch}: transfer={r['transfer_cosine']:+.4f} CI{r['transfer_cosine_ci95']} -> {r['decision']}")
    print(outcome)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--run-train", action="store_true")
    g.add_argument("--dev-transfer", action="store_true")
    a = ap.parse_args()
    mode = "preflight" if a.preflight else "run-train" if a.run_train else "dev-transfer"
    try:
        {"preflight": preflight, "run-train": run_train, "dev-transfer": dev_transfer}[mode]()
    except Exception as exc:
        try:
            append_retry(mode, 1, f"{type(exc).__name__}: {exc}")
            log_event("STAGE1_5_FAILURE", mode=mode, error=str(exc))
        finally:
            raise


if __name__ == "__main__":
    main()
