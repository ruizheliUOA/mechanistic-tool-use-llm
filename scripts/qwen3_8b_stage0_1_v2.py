#!/usr/bin/env python3
"""Qwen3-8B Stage 0/1 Protocol V2 runner.

V2 is one disclosed development-stage allocation amendment. It reallocates the
already authorized 3,104 non-test rows, adopts the audited long-sequence
attention backend, and recomputes the frozen support gate once.

Every scientific definition is inherited. Frozen numeric definitions are reused
by IMPORTING the committed V1 runner module rather than restating them, so the
direction estimator, Router specification, geometry definitions and eligibility
logic are literally the same code paths V1 used.

The sealed evaluation partition is never opened. Its identity is inherited by
reference from the frozen V1 manifest and its hash is never recomputed from
payload.

Modes:
  --preflight     configuration-only verification (no split, no support counts)
  --apply-split   Block 4: build the split ONCE and recompute the support gate
  --block5        Blocks 5-6: directions, Routers, dev geometry, handoff
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
V1DIR = ROOT / "final/results/qwen3_stage0_1"
OUT = ROOT / "final/results/qwen3_stage0_1_v2"
V1_RUNNER = ROOT / "scripts/qwen3_8b_stage0_1.py"
SELF = ROOT / "scripts/qwen3_8b_stage0_1_v2.py"

PROTOCOL_V2 = OUT / "QWEN3_8B_STAGE0_1_PROTOCOL_V2.json"
HASHES_V2 = OUT / "QWEN3_8B_STAGE0_1_HASHES_V2.json"
PREFLIGHT_V2 = OUT / "QWEN3_V2_PREFLIGHT.json"
SPLIT_INDEX = OUT / "QWEN3_V2_SPLIT_INDEX.csv"
SPLIT_MANIFEST = OUT / "QWEN3_V2_SPLIT_MANIFEST.json"
SUPPORT_LEDGER = OUT / "QWEN3_V2_SUPPORT_LEDGER.csv"
STATE = OUT / "QWEN3_V2_STATE.json"
RETRY = OUT / "QWEN3_V2_RETRY_LEDGER.json"
LOG = OUT / "QWEN3_V2_LOG.jsonl"

GEOMETRY_MAX_SECONDS = 2 * 60 * 60

# Block 4/5 scientific outputs that must not pre-exist before first computation.
V2_RESULT_NAMES = {
    "QWEN3_V2_SPLIT_INDEX.csv", "QWEN3_V2_SPLIT_MANIFEST.json",
    "QWEN3_V2_SUPPORT_LEDGER.csv", "QWEN3_V2_STATE.json",
    "QWEN3_V2_COMPLETE_CHANNEL_LEDGER.csv", "QWEN3_V2_DIRECTION_GRAM_MATRIX.csv",
    "QWEN3_V2_ROUTER_TABLE.md", "QWEN3_V2_GEOMETRY_TABLE.md",
    "QWEN3_V2_PREDICTION_TABLE.md", "QWEN3_V2_STAGE2_BRANCH_HANDOFF.md",
    "QWEN3_V2_MANIFEST.json", "QWEN3_V2_HASHES.sha256",
    "QWEN3_V2_RAW_GRADIENTS.safetensors", "QWEN3_V2_GEOMETRY_PILOT.json",
    "QWEN3_V2_RUNTIME.json",
}


# ---------------------------------------------------------------- utilities
def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json_atomic(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_text_atomic(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def write_csv_atomic(p: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, p)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def log_event(event: str, **fields: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "event": event, "schema_version": 1, **fields}, sort_keys=True) + "\n")


def append_retry(mode: str, exit_code: int, reason: str | None, invalidated: list[str]) -> None:
    obj = read_json(RETRY) if RETRY.exists() else {"schema_version": 1, "invocations": []}
    obj["invocations"].append({
        "utc": utc_now(), "mode": mode,
        "command": f"python scripts/qwen3_8b_stage0_1_v2.py --{mode}",
        "exit_code": exit_code, "reason": reason, "outputs_invalidated": invalidated,
        "script_commit": git_head(), "script_sha256": sha256_file(SELF),
    })
    write_json_atomic(RETRY, obj)


def install_gradient_checkpointing(model, l_inj: int) -> dict:
    """Engineering-only memory strategy for the frozen dev-geometry gradient.

    The frozen geometry detaches a leaf at layer `l_inj`'s MLP output, so only
    layers strictly after `l_inj` retain an autograd graph. Storing their eager
    attention probabilities costs 32*S^2*2 bytes per layer, which exceeds a
    24 GiB card for the longer DEV rows. Wrapping those layers in
    torch.utils.checkpoint stores layer-boundary hidden states instead and
    recomputes each layer's internals during backward.

    This is a compute/memory tradeoff, not a definitional change: the recomputed
    forward is the same deterministic function of the same inputs, the model is
    in eval mode with no dropout, deterministic algorithms are enabled, and RNG
    state is preserved by checkpoint. Layer `l_inj` itself is NOT checkpointed,
    because recomputing it would re-run the leaf-creating hook and sever the
    graph the gradient is taken through.
    """
    import torch
    import torch.utils.checkpoint as cp

    wrapped = []
    for i, layer in enumerate(model.model.layers):
        if i <= l_inj:
            continue
        original = layer.forward

        def fwd(*args, _original=original, **kwargs):
            if torch.is_grad_enabled() and any(
                    isinstance(a, torch.Tensor) and a.requires_grad for a in args):
                return cp.checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        layer.forward = fwd
        wrapped.append(i)
    return {"installed": True, "checkpointed_layers": wrapped,
            "excluded_leaf_layer": l_inj,
            "scientific_definitions_changed": False,
            "rationale": "recompute instead of store; identical deterministic function"}


def install_chunk_checkpointed_attention(v1) -> dict:
    """Engineering-only memory strategy for the geometry backward pass.

    Above the frozen guard threshold the frozen backend already evaluates eager
    attention in head chunks, but under autograd every chunk's attention
    probabilities are retained simultaneously, so chunking alone does not reduce
    backward memory. This wrapper performs the SAME chunked computation with the
    SAME chunk size and the SAME tensor shapes, wrapping each chunk in
    torch.utils.checkpoint so that only one chunk's probabilities are live at a
    time and the rest are recomputed.

    Operations, shapes and kernel selection are unchanged, so outputs are
    preserved exactly; only when they are computed changes. Sequences at or
    below the frozen threshold are delegated unchanged to the frozen guard.
    """
    import torch
    import torch.utils.checkpoint as cp
    from torch import nn
    import transformers.models.qwen3.modeling_qwen3 as qwen3

    frozen_guard = qwen3.eager_attention_forward
    threshold = v1.ATTENTION_GUARD_THRESHOLD_TOKENS
    chunk = v1.ATTENTION_GUARD_HEAD_CHUNK

    def wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.shape[-2] <= threshold:
            return frozen_guard(module, query, key, value, attention_mask, scaling,
                                dropout=dropout, **kwargs)
        key_states = qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen3.repeat_kv(value, module.num_key_value_groups)
        mask = None if attention_mask is None else attention_mask[:, :, :, : key_states.shape[-2]]

        def one_chunk(q_c, k_c, v_c, msk):
            w = torch.matmul(q_c, k_c.transpose(2, 3)) * scaling
            if msk is not None:
                w = w + msk
            w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
            w = nn.functional.dropout(w, p=dropout, training=module.training)
            return torch.matmul(w, v_c)

        pieces = []
        for start in range(0, query.shape[1], chunk):
            stop = min(start + chunk, query.shape[1])
            args = (query[:, start:stop], key_states[:, start:stop], value_states[:, start:stop], mask)
            if torch.is_grad_enabled() and query.requires_grad:
                pieces.append(cp.checkpoint(one_chunk, *args, use_reentrant=False))
            else:
                pieces.append(one_chunk(*args))
        return torch.cat(pieces, dim=1).transpose(1, 2).contiguous(), None

    qwen3.eager_attention_forward = wrapper
    return {"installed": True, "threshold_tokens": threshold, "head_chunk": chunk,
            "below_threshold": "delegated unchanged to the frozen V1 guard",
            "shapes_and_kernels_unchanged": True,
            "scientific_definitions_changed": False}


def load_v1_module():
    spec = importlib.util.spec_from_file_location("qwen3_v1_runner", V1_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_rows() -> list[dict]:
    return [json.loads(l) for l in (V1DIR / "QWEN3_STAGE1_BASELINE_ROWS.jsonl").open(encoding="utf-8")]


# ---------------------------------------------------------------- split rule
def compute_split(proto: dict, rows: list[dict]) -> list[dict]:
    """The exact committed V2 allocation rule. Depends only on immutable UUID,
    gold label, the fixed literal salt and the total target sizes."""
    rule = proto["v2_allocation_rule"]
    salt = rule["salt"]
    dev_target = rule["targets"]["dev"]
    train_target = rule["targets"]["train"]

    strata: dict[str, list[dict]] = {}
    for r in rows:
        h = sha256_bytes((salt + r["sample_id"]).encode("utf-8"))
        strata.setdefault(r["gold"], []).append({
            "project_index": r["project_index"], "sample_id": r["sample_id"],
            "gold": r["gold"], "split_hash": h,
        })
    for g in strata:
        strata[g].sort(key=lambda x: (x["split_hash"], x["sample_id"]))

    sizes = {g: len(v) for g, v in strata.items()}
    total = sum(sizes.values())
    exact = {g: sizes[g] * dev_target / total for g in sizes}
    base = {g: int(math.floor(exact[g])) for g in sizes}
    remaining = dev_target - sum(base.values())
    for g in sorted(sizes, key=lambda g: (-(exact[g] - base[g]), g))[:remaining]:
        base[g] += 1
    assert sum(base.values()) == dev_target

    out: list[dict] = []
    for g in sorted(strata):
        for rank, row in enumerate(strata[g]):
            out.append({**row, "gold_stratum_rank": rank,
                        "v2_split": "dev" if rank < base[g] else "train"})
    out.sort(key=lambda r: r["project_index"])
    assert sum(1 for r in out if r["v2_split"] == "dev") == dev_target
    assert sum(1 for r in out if r["v2_split"] == "train") == train_target
    return out


def build_support_ledger(proto: dict, rows: list[dict], assign: dict[str, str]) -> list[dict]:
    modes = proto["readout"]["mode_order"]
    gate = proto["channel_discovery"]["support_gate"]
    tr = [r for r in rows if assign[r["sample_id"]] == "train"]
    dv = [r for r in rows if assign[r["sample_id"]] == "dev"]
    ledger = []
    for gold in modes:
        tr_ref = sum(1 for r in tr if r["gold"] == gold and r["prediction"] == gold)
        dv_ref = sum(1 for r in dv if r["gold"] == gold and r["prediction"] == gold)
        for pred in modes:
            if pred == gold:
                continue
            tr_err = sum(1 for r in tr if r["gold"] == gold and r["prediction"] == pred)
            dv_err = sum(1 for r in dv if r["gold"] == gold and r["prediction"] == pred)
            fails = []
            if tr_err < gate["train_error_min"]:
                fails.append("TRAIN_ERROR_LT_%d" % gate["train_error_min"])
            if tr_ref < gate["train_correct_reference_min"]:
                fails.append("TRAIN_CORRECT_REFERENCE_LT_%d" % gate["train_correct_reference_min"])
            if dv_err < gate["dev_error_min"]:
                fails.append("DEV_ERROR_LT_%d" % gate["dev_error_min"])
            if dv_ref < gate["dev_correct_reference_min"]:
                fails.append("DEV_CORRECT_REFERENCE_LT_%d" % gate["dev_correct_reference_min"])
            ledger.append({
                "channel": f"{gold}__to__{pred}", "gold": gold, "source": pred,
                "train_error_support": tr_err, "train_correct_reference_support": tr_ref,
                "dev_error_support": dv_err, "dev_correct_reference_support": dv_ref,
                "train_error_pass": tr_err >= gate["train_error_min"],
                "train_correct_reference_pass": tr_ref >= gate["train_correct_reference_min"],
                "dev_error_pass": dv_err >= gate["dev_error_min"],
                "dev_correct_reference_pass": dv_ref >= gate["dev_correct_reference_min"],
                "support_eligible": not fails,
                "support_exclusion_reasons": ";".join(fails),
                "direction_valid": False, "router_eligible": False,
                "geometry_valid": False, "stage2_eligible": False,
                "stage2_exclusion_reasons": "NOT_YET_EVALUATED",
            })
    return ledger


# ---------------------------------------------------------------- preflight
def preflight() -> None:
    if git_status():
        raise RuntimeError("V2 preflight requires a clean committed worktree")
    proto = read_json(PROTOCOL_V2)
    manifest = read_json(HASHES_V2)

    mismatches = []
    for rel, expected in manifest["artifact_sha256"].items():
        p = ROOT / rel
        if not p.is_file():
            mismatches.append([rel, "MISSING", expected])
        elif sha256_file(p) != expected:
            mismatches.append([rel, sha256_file(p), expected])
    if mismatches:
        raise RuntimeError(f"V2 artifact hash mismatch: {mismatches}")

    v1 = load_v1_module()

    # model / tokenizer identity (no full model load)
    model_files = {}
    for name, expected in proto["model"]["file_sha256"].items():
        p = v1.MODEL_DIR / name
        if not p.exists() or not p.resolve().is_file():
            raise RuntimeError(f"model snapshot missing {name}")
        size = p.resolve().stat().st_size
        if size != proto["model"]["weight_file_sizes"].get(name, size):
            raise RuntimeError(f"model size mismatch {name}")
        got = sha256_file(p.resolve())
        if got != expected:
            raise RuntimeError(f"model sha256 mismatch {name}")
        model_files[name] = {"size": size, "sha256": got}

    import torch, transformers, tokenizers
    if transformers.__version__ != proto["model"]["transformers"]:
        raise RuntimeError("transformers version mismatch")
    if tokenizers.__version__ != proto["model"]["tokenizers"]:
        raise RuntimeError("tokenizers version mismatch")
    if torch.__version__ != proto["model"]["torch"]:
        raise RuntimeError("torch version mismatch")

    # inherited non-test population identity
    reuse = proto["artifact_reuse"]
    for key, path in (("baseline_sha256", V1DIR / "QWEN3_STAGE1_BASELINE_ROWS.jsonl"),
                      ("activations_sha256", V1DIR / "QWEN3_STAGE1_ACTIVATIONS.safetensors")):
        got = sha256_file(path)
        if got != reuse[key]:
            raise RuntimeError(f"inherited artifact hash mismatch: {path.name}")
    rows = baseline_rows()
    if len(rows) != proto["data"]["non_test_population"]:
        raise RuntimeError("inherited non-test population size mismatch")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate immutable UUID in inherited population")

    # attention backend
    audit = read_json(OUT / "QWEN3_V2_ATTENTION_BACKEND_AUDIT.json")
    if audit["disposition"] != "ATTENTION_BACKEND_ACCEPTED":
        raise RuntimeError("attention backend not accepted")
    guard = v1.install_long_sequence_attention_guard()
    backend_hash = sha256_file(OUT / "QWEN3_V2_ATTENTION_BACKEND_AUDIT.json")

    # artifact audit outcome
    inv = read_json(OUT / "QWEN3_V2_ARTIFACT_INVENTORY.json")
    if inv["outcome"] not in ("CPU_SUFFICIENT", "GPU_ACTIVATION_PASS_REQUIRED"):
        raise RuntimeError(f"artifact audit outcome blocks V2: {inv['outcome']}")

    # activation availability at both frozen sites
    from safetensors import safe_open
    with safe_open(V1DIR / "QWEN3_STAGE1_ACTIVATIONS.safetensors", framework="numpy") as f:
        keys = set(f.keys())
        act_meta = f.metadata()
    for site in (proto["layer_mapping"]["new_L_obs"], proto["layer_mapping"]["new_L_inj"]):
        if f"activations_L{site}" not in keys:
            raise RuntimeError(f"missing activation site L{site}")

    # empty V2 result namespace
    for name in sorted(V2_RESULT_NAMES):
        if (OUT / name).exists():
            raise RuntimeError(f"V2 result namespace not empty: {name}")
    for d in ("directions", "routers"):
        if (OUT / d).exists():
            raise RuntimeError(f"V2 result namespace directory not empty: {d}")

    # LFS availability for V2 numerical outputs
    subprocess.run(["git", "lfs", "version"], cwd=ROOT, check=True, capture_output=True)
    attr = subprocess.check_output(
        ["git", "check-attr", "filter", "--",
         "final/results/qwen3_stage0_1_v2/directions/PROBE.safetensors"], cwd=ROOT, text=True).strip()
    if not attr.endswith(": lfs"):
        raise RuntimeError("Git LFS attribute unavailable for V2 numerical outputs")

    payload = {
        "schema_version": 1, "status": "PASS", "utc": utc_now(),
        "protocol_version": proto["protocol_version"],
        "head": git_head(),
        "v2_hash_manifest_sha256": sha256_file(HASHES_V2),
        "v2_runner_sha256": sha256_file(SELF),
        "v1_runner_sha256": sha256_file(V1_RUNNER),
        "attention_backend_audit_sha256": backend_hash,
        "attention_backend_disposition": audit["disposition"],
        "attention_guard_runtime": guard,
        "artifact_audit_outcome": inv["outcome"],
        "model_identity": {"repository": proto["model"]["repository"],
                           "revision": proto["model"]["revision"],
                           "tokenizer_revision": proto["model"]["tokenizer_revision"],
                           "files": model_files},
        "inherited_non_test_population": {
            "rows": len(rows), "unique_sample_ids": len({r["sample_id"] for r in rows}),
            "baseline_sha256": reuse["baseline_sha256"],
            "activations_sha256": reuse["activations_sha256"],
            "activation_sites_present": sorted(keys),
            "activation_site_metadata": act_meta.get("site"),
        },
        "sealed_evaluation": {
            "identity_inherited_by_reference": True,
            "manifest_opened": False,
            "hash_recomputed_from_payload": False,
            "count": proto["data"]["v2_project_split"]["sealed_evaluation"],
            "source_manifest": proto["data"]["sealed_evaluation_identity"]["source_manifest"],
        },
        "v2_output_namespace_empty": True,
        "split_computed": False,
        "support_counts_computed": False,
        "test_data_loaded": False,
        "full_model_loaded": False,
        "geometry_computed": False,
        "stage2_authorized": False,
    }
    write_json_atomic(PREFLIGHT_V2, payload)
    append_retry("preflight", 0, None, [])
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------- Block 4
def apply_split() -> None:
    if git_status():
        raise RuntimeError("Block 4 requires a clean committed worktree")
    pf = read_json(PREFLIGHT_V2)
    if pf.get("status") != "PASS":
        raise RuntimeError("valid V2 preflight missing")
    for p in (SPLIT_INDEX, SPLIT_MANIFEST, SUPPORT_LEDGER, STATE):
        if p.exists():
            raise RuntimeError(f"atomic-output rule: {p.name} already promoted; no recomputation permitted")

    proto = read_json(PROTOCOL_V2)
    rows = baseline_rows()
    started = time.perf_counter()

    split = compute_split(proto, rows)
    assign = {r["sample_id"]: r["v2_split"] for r in split}

    inherited = {r["sample_id"] for r in rows}
    produced = {r["sample_id"] for r in split}
    train_ids = {r["sample_id"] for r in split if r["v2_split"] == "train"}
    dev_ids = {r["sample_id"] for r in split if r["v2_split"] == "dev"}
    if produced != inherited:
        raise RuntimeError("split population differs from inherited non-test population")
    if train_ids & dev_ids:
        raise RuntimeError("train/dev intersection non-empty")
    if train_ids | dev_ids != inherited:
        raise RuntimeError("union differs from inherited population")
    if len(split) != len(inherited):
        raise RuntimeError("row appears more than once")
    if len(train_ids) != proto["v2_allocation_rule"]["targets"]["train"]:
        raise RuntimeError("train size mismatch")
    if len(dev_ids) != proto["v2_allocation_rule"]["targets"]["dev"]:
        raise RuntimeError("dev size mismatch")

    ledger = build_support_ledger(proto, rows, assign)
    k = sum(1 for c in ledger if c["support_eligible"])

    gold_by_split = {s: dict(Counter(r["gold"] for r in split if r["v2_split"] == s)) for s in ("train", "dev")}
    manifest = {
        "schema_version": 1, "protocol_version": proto["protocol_version"], "utc": utc_now(),
        "rule": proto["v2_allocation_rule"],
        "counts": {"train": len(train_ids), "dev": len(dev_ids), "total_non_test": len(split)},
        "gold_distribution": gold_by_split,
        "union_equals_inherited_population": True,
        "intersection_empty": True,
        "every_row_exactly_once": True,
        "unknown_rows": 0,
        "sealed_evaluation": {
            "identity_inherited_unchanged_from_v1_frozen_manifest": True,
            "opened": False, "count": proto["data"]["v2_project_split"]["sealed_evaluation"],
        },
        "train_sample_id_order_sha256": sha256_bytes(
            ("\n".join(sorted(train_ids)) + "\n").encode()),
        "dev_sample_id_order_sha256": sha256_bytes(
            ("\n".join(sorted(dev_ids)) + "\n").encode()),
        "split_assignment_sha256": sha256_bytes(
            ("\n".join(f"{r['sample_id']}\t{r['v2_split']}" for r in sorted(split, key=lambda x: x["sample_id"])) + "\n").encode()),
        "K_support_eligible": k,
        "elapsed_seconds": time.perf_counter() - started,
    }

    fields = ["project_index", "sample_id", "gold", "split_hash", "gold_stratum_rank", "v2_split"]
    write_csv_atomic(SPLIT_INDEX, fields, split)
    write_json_atomic(SPLIT_MANIFEST, manifest)
    write_csv_atomic(SUPPORT_LEDGER, list(ledger[0].keys()), ledger)
    branch = stage2_branch(proto, k)
    write_json_atomic(STATE, {
        "schema_version": 1, "utc": utc_now(), "block4_promoted": True,
        "K_support_eligible": k, "inherited_branch": branch,
        "split_index_sha256": sha256_file(SPLIT_INDEX),
        "support_ledger_sha256": sha256_file(SUPPORT_LEDGER),
        "block5_completed": False,
    })
    log_event("V2_SPLIT_AND_SUPPORT_PROMOTED", K=k, branch=branch["branch"])
    append_retry("apply-split", 0, None, [])
    print(json.dumps({"counts": manifest["counts"], "gold_distribution": gold_by_split,
                      "K_support_eligible": k, "inherited_branch": branch}, indent=2, sort_keys=True))
    if k == 0:
        print("QWEN3_V2_NO_SUPPORT_ELIGIBLE_CHANNELS")


def stage2_branch(proto: dict, k: int) -> dict:
    b = proto["stage2_branches"]
    if k >= 5:
        return {"K": k, "branch": b["K_ge_5"]["branch"], "future_primary": b["K_ge_5"]["future_primary"]}
    if k == 4:
        return {"K": k, "branch": b["K_eq_4"]["branch"], "future_primary": b["K_eq_4"]["future_primary"]}
    if k >= 1:
        return {"K": k, "branch": b["K_1_to_3"]["branch"], "future_primary": b["K_1_to_3"]["future_primary"],
                "predictor_claim": b["K_1_to_3"]["predictor_claim"]}
    return {"K": k, "branch": b["K_0"]["branch"], "outcome": b["K_0"]["outcome"],
            "stage2_justified": b["K_0"]["stage2_justified"]}


# ---------------------------------------------------------------- Block 5-6
def block5() -> None:
    state = read_json(STATE)
    if not state.get("block4_promoted"):
        raise RuntimeError("Block 4 not promoted")
    k = state["K_support_eligible"]
    if k < 1:
        raise RuntimeError("K=0; Block 5 must not run")

    proto = read_json(PROTOCOL_V2)
    v1 = load_v1_module()
    v1.OUT = OUT                      # redirect frozen writers into the V2 namespace
    started = time.perf_counter()

    rows_all = baseline_rows()
    split = list(csv.DictReader(SPLIT_INDEX.open(encoding="utf-8")))
    assign = {r["sample_id"]: r["v2_split"] for r in split}
    rows = [{**r, "split": assign[r["sample_id"]]} for r in rows_all]

    from safetensors import safe_open
    from safetensors.numpy import load_file
    act_path = V1DIR / "QWEN3_STAGE1_ACTIVATIONS.safetensors"
    with safe_open(act_path, framework="numpy") as fh:
        order = json.loads(fh.metadata()["sample_order_json"])
    if [o["sample_id"] for o in order] != [r["sample_id"] for r in rows_all]:
        raise RuntimeError("activation order does not align with baseline rows")
    acts = load_file(act_path)
    activations = acts[f"activations_L{proto['layer_mapping']['new_L_obs']}"]

    ledger = [dict(c) for c in csv.DictReader(SUPPORT_LEDGER.open(encoding="utf-8"))]
    for c in ledger:
        for key in ("train_error_support", "train_correct_reference_support",
                    "dev_error_support", "dev_correct_reference_support"):
            c[key] = int(c[key])
        c["support_eligible"] = c["support_eligible"] == "True"
        c["direction_valid"] = False; c["router_eligible"] = False
        c["geometry_valid"] = False; c["stage2_eligible"] = False

    # 5.1 + 5.2 : frozen direction estimator and Router, reusing V1 code
    router_metrics, directions, gram = v1.fit_directions_routers(rows, activations, ledger)
    log_event("V2_DIRECTIONS_AND_ROUTERS_COMPLETE", channels=len(directions))

    # 5.3 geometry pilot and cost ceiling
    model, tokenizer, device, environment = v1.load_model()
    environment["gradient_checkpointing"] = install_gradient_checkpointing(
        model, proto["layer_mapping"]["new_L_inj"])
    environment["chunk_checkpointed_attention"] = install_chunk_checkpointed_attention(v1)
    data = v1.TrainDevData()
    dev_err = []
    for c in ledger:
        if c["support_eligible"] and c["direction_valid"]:
            dev_err += [r for r in rows if r["split"] == "dev"
                        and r["gold"] == c["gold"] and r["prediction"] == c["source"]]
    pilot_rows = sorted({r["sample_id"]: r for r in dev_err}.values(), key=lambda r: r["sample_id"])[:4]
    p0 = time.perf_counter()
    pilot_detail = []
    for r in pilot_rows:
        s = dict(data.sample(r["project_index"])); s["_baseline_prediction"] = r["prediction"]
        ch = next(c for c in ledger if c["gold"] == r["gold"] and c["source"] == r["prediction"])
        recs, mats = v1.compute_geometry_one(model, tokenizer, device, s, directions[ch["channel"]])
        full = mats["full_effect"]
        pilot_detail.append({
            "sample_id": r["sample_id"], "channel": ch["channel"],
            "gradient_shape": list(full.shape), "finite": bool(np.all(np.isfinite(full))),
            "variants": sorted(mats), "statuses": [x.get("status") for x in recs],
        })
        if full.shape != (4, proto["model"]["hidden_size"]) or not np.all(np.isfinite(full)):
            raise RuntimeError("geometry pilot invalid gradient")
    pilot_seconds = time.perf_counter() - p0
    per_sample = pilot_seconds / max(1, len(pilot_rows))

    # deterministic replay on the first pilot row
    r0 = pilot_rows[0]
    s0 = dict(data.sample(r0["project_index"])); s0["_baseline_prediction"] = r0["prediction"]
    ch0 = next(c for c in ledger if c["gold"] == r0["gold"] and c["source"] == r0["prediction"])
    a_rec, a_mat = v1.compute_geometry_one(model, tokenizer, device, s0, directions[ch0["channel"]])
    b_rec, b_mat = v1.compute_geometry_one(model, tokenizer, device, s0, directions[ch0["channel"]])
    replay = bool(np.array_equal(a_mat["full_effect"], b_mat["full_effect"]))
    import torch
    total_dev_err = len(dev_err)
    projected = per_sample * total_dev_err
    pilot = {
        "schema_version": 1, "selection": "ascending immutable UUID over V2 DEV channel-error rows",
        "pilot_rows": [r["sample_id"] for r in pilot_rows], "pilot_detail": pilot_detail,
        "pilot_seconds": pilot_seconds, "seconds_per_sample": per_sample,
        "dev_error_samples_total": total_dev_err,
        "projected_geometry_seconds": projected,
        "cost_ceiling_seconds": GEOMETRY_MAX_SECONDS,
        "deterministic_replay_bitwise": replay,
        "memory_released_between_samples": True,
        "peak_gpu_memory_bytes_after_pilot": int(torch.cuda.max_memory_allocated(0)),
    }
    write_json_atomic(OUT / "QWEN3_V2_GEOMETRY_PILOT.json", pilot)
    if not replay:
        raise RuntimeError("geometry pilot deterministic replay failed")
    if projected > GEOMETRY_MAX_SECONDS:
        write_json_atomic(OUT / "QWEN3_V2_RUNTIME.json", {
            "status": "QWEN3_V2_GEOMETRY_COST_BLOCKED", "pilot": pilot,
            "total_seconds": time.perf_counter() - started})
        append_retry("block5", 0, "QWEN3_V2_GEOMETRY_COST_BLOCKED", [])
        print("QWEN3_V2_GEOMETRY_COST_BLOCKED")
        return

    # 5.4 full dev geometry, reusing the frozen V1 implementation
    g0 = time.perf_counter()
    analysis, geo_records = v1.run_geometry(model, tokenizer, device, rows, ledger, directions)
    geometry_seconds = time.perf_counter() - g0
    log_event("V2_DEV_GEOMETRY_COMPLETE", records=len(geo_records))

    k_stage2 = sum(1 for c in ledger if c["stage2_eligible"])
    branch = stage2_branch(proto, k_stage2)
    emit_block6(proto, ledger, analysis, router_metrics, gram, branch, k, k_stage2)

    runtime = {
        "status": "QWEN3_V2_COMPLETE_STAGE2_NOT_AUTHORIZED" if k_stage2 >= 1 else "QWEN3_V2_NO_STAGE2_ELIGIBLE_CHANNELS",
        "utc": utc_now(), "environment": environment, "pilot": pilot,
        "geometry_seconds": geometry_seconds, "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "K_support_eligible": k, "K_stage2_eligible": k_stage2,
        "branch": branch, "evaluation_accessed": False, "stage2_authorized": False,
    }
    write_json_atomic(OUT / "QWEN3_V2_RUNTIME.json", runtime)
    st = read_json(STATE); st["block5_completed"] = True
    st["K_stage2_eligible"] = k_stage2; st["final_branch"] = branch
    write_json_atomic(STATE, st)
    append_retry("block5", 0, None, [])
    print(json.dumps(runtime, indent=2, sort_keys=True, default=str))
    print(runtime["status"])


def emit_block6(proto, ledger, analysis, router_metrics, gram, branch, k, k_stage2) -> None:
    write_csv_atomic(OUT / "QWEN3_V2_COMPLETE_CHANNEL_LEDGER.csv", list(ledger[0].keys()), ledger)

    chs = gram["channels"]
    grid = [["channel"] + chs]
    for i, a in enumerate(chs):
        grid.append([a] + [f"{gram['signed_cosine_matrix'][i][j]:.6f}" for j in range(len(chs))])
    write_text_atomic(OUT / "QWEN3_V2_DIRECTION_GRAM_MATRIX.csv",
                      "\n".join(",".join(r) for r in grid) + "\n")

    rt = ["# Qwen3 V2 Router table", "", "STAGE_2_NOT_AUTHORIZED", "",
          "| channel | train pos | train neg | dev ROC-AUC | dev PR-AUC | tau | precision | recall | specificity | routed | eligible |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for ch in sorted(router_metrics):
        m = router_metrics[ch]; sel = m["selected_metrics"] or {}
        rt.append(f"| {ch} | {m['train_positive_n']} | {m['train_all_correct_negative_n']} | "
                  f"{m['dev_comparable']['ROC_AUC']:.4f} | {m['dev_comparable']['PR_AUC']:.4f} | "
                  f"{m['selected_tau']} | {sel.get('precision', float('nan')):.4f} | {sel.get('recall', float('nan')):.4f} | "
                  f"{sel.get('specificity', float('nan')):.4f} | {sel.get('routed', 0)} | {m['eligible']} |")
    write_text_atomic(OUT / "QWEN3_V2_ROUTER_TABLE.md", "\n".join(rt) + "\n")

    def fmt(x):
        return "—" if x is None else (f"{x:.6f}" if isinstance(x, float) else str(x))

    gt = ["# Qwen3 V2 dev geometry table", "", "STAGE_2_NOT_AUTHORIZED",
          "", "Developmental train/dev result. The evaluation split was not accessed.", "",
          "| channel | n total | n valid | Q_sum full | Q_sum common | neg-sign full | neg-sign common | resultant full | null p |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in ledger:
        if not c["support_eligible"]:
            continue
        a = analysis.get(c["channel"], {}); full = a.get("full_effect", {}); com = a.get("common_prompt", {})
        conc = full.get("concentration", {})
        gt.append(f"| {c['channel']} | {full.get('n_total',0)} | {full.get('n_valid',0)} | "
                  f"{fmt(full.get('Q_sum_channel'))} | {fmt(com.get('Q_sum_channel'))} | "
                  f"{fmt(full.get('negative_sign_fraction'))} | {fmt(com.get('negative_sign_fraction'))} | "
                  f"{fmt(conc.get('resultant_length'))} | "
                  f"{fmt((conc.get('random_axis_null') or {}).get('empirical_one_sided_p'))} |")
    cons = v1_consistency(analysis)
    gt += ["", f"Full/common ordering consistency: `{json.dumps(cons, sort_keys=True)}`",
           "", "Geometry never adds or removes a channel."]
    write_text_atomic(OUT / "QWEN3_V2_GEOMETRY_TABLE.md", "\n".join(gt) + "\n")

    pt = ["# Qwen3 V2 prediction table", "", "STAGE_2_NOT_AUTHORIZED", "",
          "One row per support-eligible channel.", "",
          "| channel | train err | train ref | dev err | dev ref | ROC-AUC | PR-AUC | tau | precision | "
          "Q_sum full | Q_sum common | neg-sign full | neg-sign common | concentration | concentration null p | direction sha256 | eligible | exclusion |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for c in ledger:
        if not c["support_eligible"]:
            continue
        ch = c["channel"]; m = router_metrics.get(ch, {}); sel = (m.get("selected_metrics") or {})
        a = analysis.get(ch, {}); full = a.get("full_effect", {}); com = a.get("common_prompt", {})
        conc = full.get("concentration", {})
        pt.append(
            f"| {ch} | {c['train_error_support']} | {c['train_correct_reference_support']} | "
            f"{c['dev_error_support']} | {c['dev_correct_reference_support']} | "
            f"{fmt(m.get('dev_comparable',{}).get('ROC_AUC'))} | {fmt(m.get('dev_comparable',{}).get('PR_AUC'))} | "
            f"{m.get('selected_tau')} | {fmt(sel.get('precision'))} | "
            f"{fmt(full.get('Q_sum_channel'))} | {fmt(com.get('Q_sum_channel'))} | "
            f"{fmt(full.get('negative_sign_fraction'))} | {fmt(com.get('negative_sign_fraction'))} | "
            f"{fmt(conc.get('resultant_length'))} | {fmt((conc.get('random_axis_null') or {}).get('empirical_one_sided_p'))} | "
            f"`{m.get('direction_unit_sha256','—')}` | {c['stage2_eligible']} | {c['stage2_exclusion_reasons'] or '—'} |")
    write_text_atomic(OUT / "QWEN3_V2_PREDICTION_TABLE.md", "\n".join(pt) + "\n")

    hand = ["# Qwen3 V2 Stage 2 branch handoff", "", "## STAGE_2_NOT_AUTHORIZED", "",
            "This task ends before Stage 2 preregistration and before any evaluation/test access.",
            "No Stage 2 preregistration, runner, execution freeze, preflight or test prediction was created.", "",
            f"K (support-eligible) = **{k}**", f"K_stage2 (support + direction + Router + geometry) = **{k_stage2}**", "",
            f"Inherited branch: **{branch.get('branch')}**", "",
            f"Future primary: {branch.get('future_primary', 'no Stage 2 intervention justified')}", "",
            "## Claim boundary", "",
            "Future evidence from this line must be labelled:", "",
            f"> {proto['disclosure']['required_label']}", "",
            "not a pristine no-amendment confirmation.", "",
            "## Disclosure", "",
            f"- V1 allocation: {proto['disclosure']['v1_allocation']}",
            f"- V1 result: {proto['disclosure']['v1_result']}",
            f"- V1 failure cause: {proto['disclosure']['v1_failure_cause']}",
            f"- V2 changed: {proto['disclosure']['v2_changes']}",
            "- evaluation split sealed and unaccessed: True",
            "- support threshold lowered: False",
            "- channel manually included: False",
            "- alternative V2 split tried: False", "",
            "## Support-eligible channels ordered by dev Q_sum (low to high)", "",
            "| rank | channel | Q_sum full | negative-sign fraction |", "|---:|---|---:|---:|"]
    ordered = [c for c in ledger if c["support_eligible"]
               and analysis.get(c["channel"], {}).get("full_effect", {}).get("Q_sum_channel") is not None]
    ordered.sort(key=lambda c: analysis[c["channel"]]["full_effect"]["Q_sum_channel"])
    for n, c in enumerate(ordered, 1):
        g = analysis[c["channel"]]["full_effect"]
        hand.append(f"| {n} | {c['channel']} | {g['Q_sum_channel']:.6f} | {g['negative_sign_fraction']:.6f} |")
    hand += ["", "## Complete eligibility", "",
             "| channel | support | direction | Router | geometry | Stage2 | exclusion |", "|---|---|---|---|---|---|---|"]
    for c in ledger:
        hand.append(f"| {c['channel']} | {c['support_eligible']} | {c['direction_valid']} | {c['router_eligible']} | "
                    f"{c['geometry_valid']} | {c['stage2_eligible']} | "
                    f"{c['stage2_exclusion_reasons'] or c['support_exclusion_reasons'] or '—'} |")
    hand += ["", "## Direction Gram matrix", "", f"`{json.dumps(gram, sort_keys=True)}`", "",
             "Evaluation rows, IDs, labels, predictions, scores, support counts, aggregates and channel",
             "membership were not accessed at any point.", ""]
    write_text_atomic(OUT / "QWEN3_V2_STAGE2_BRANCH_HANDOFF.md", "\n".join(hand) + "\n")

    files = sorted(p for p in OUT.rglob("*") if p.is_file()
                   and p.name not in ("QWEN3_V2_HASHES.sha256", "QWEN3_V2_MANIFEST.json"))
    write_text_atomic(OUT / "QWEN3_V2_HASHES.sha256",
                      "".join(f"{sha256_file(p)}  {p.relative_to(ROOT)}\n" for p in files))
    write_json_atomic(OUT / "QWEN3_V2_MANIFEST.json", {
        "schema_version": 1, "utc": utc_now(), "protocol_version": proto["protocol_version"],
        "stage2_authorized": False, "K_support_eligible": k, "K_stage2_eligible": k_stage2,
        "branch": branch, "head_before_result_commit": git_head(),
        "v2_runner_sha256": sha256_file(SELF), "v1_runner_sha256": sha256_file(V1_RUNNER),
        "files": [str(p.relative_to(ROOT)) for p in files],
        "evaluation_accessed": False,
    })


def v1_consistency(analysis):
    v1 = load_v1_module()
    return v1.full_common_ordering_consistency(analysis)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--apply-split", action="store_true")
    g.add_argument("--block5", action="store_true")
    a = ap.parse_args()
    mode = "preflight" if a.preflight else "apply-split" if a.apply_split else "block5"
    try:
        if a.preflight:
            preflight()
        elif a.apply_split:
            apply_split()
        else:
            block5()
    except Exception as exc:
        try:
            append_retry(mode, 1, f"{type(exc).__name__}: {exc}", [])
            log_event("V2_FAILURE", mode=mode, error_type=type(exc).__name__, error=str(exc))
        finally:
            raise


if __name__ == "__main__":
    main()
