#!/usr/bin/env python3
"""Qwen3-8B Stage 0/1: sealed train/dev baseline, channels, routers, geometry.

This runner has no evaluation/test loader. It supports only:
  --build-block0-index  infrastructure-only selective train/dev byte index
  --preflight           configuration/provenance/template checks; no model load
  --run-stage1          frozen TRAIN/DEV developmental execution
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
OUT = ROOT / "final/results/qwen3_stage0_1"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_REPOSITORY = "Qwen/Qwen3-8B"
MODEL_DIR = ROOT / ".cache/qwen3_stage0_1" / MODEL_REVISION
META_SOURCE = Path("/tmp/qwen3_stage0_1_meta_b968826d9c46dd6066d109eabc6255188de91218")
DATASET_REPOSITORY = "nvidia/When2Call"
DATASET_REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
RAW_SOURCE = ROOT / "data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl"
BASELINE_SOURCE = ROOT / "data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl"
BASELINE_INDEX = ROOT / "final/results/qwen25_stage_b_preparation/qwen25_baseline_train_dev_byte_index.csv"
TRAIN_MANIFEST = ROOT / "final/results/splits/train_idx.json"
DEV_MANIFEST = ROOT / "final/results/splits/val_idx.json"
INDEX = OUT / "QWEN3_8B_TRAIN_DEV_BYTE_INDEX.csv"
INDEX_MANIFEST = OUT / "QWEN3_8B_TRAIN_DEV_BYTE_INDEX_MANIFEST.json"
PROTOCOL_JSON = OUT / "QWEN3_8B_STAGE0_1_PROTOCOL.json"
PROTOCOL_HASHES = OUT / "QWEN3_8B_STAGE0_1_HASHES.json"
PREFLIGHT = OUT / "QWEN3_8B_STAGE0_1_PREFLIGHT.json"
PROMPT_PILOT = OUT / "QWEN3_8B_TRAIN_PROMPT_PILOT.json"
PREFLIGHT_HASHES = OUT / "QWEN3_8B_STAGE0_1_PREFLIGHT_HASHES.json"
RETRY_LEDGER = OUT / "QWEN3_8B_STAGE0_1_RETRY_LEDGER.json"

EXPECTED_COUNTS = {"train": 2556, "dev": 548}
TOTAL_ROWS = 3652
SEALED_EVALUATION_COUNT = 548
SPLIT_SEED = 42
MODES = ["tool_call", "direct", "request_for_info", "cannot_answer"]
OLD_NUM_LAYERS = 28
OLD_L_OBS = 20
OLD_L_INJ = 16
NEW_NUM_LAYERS = 36
L_OBS = 26
L_INJ = 21
HIDDEN_SIZE = 4096
DTYPE_NAME = "bfloat16"
ATTENTION_IMPLEMENTATION = "eager"
MAX_TOKENS = 8192
SCORE_REPLAY_ATOL = 1e-5
SCORE_TIE_TOL = 1e-8
PROMPT_STATE_ATOL = 1e-6
CONTRAST_NORM_TOL = 1e-8
SVD_ABS_TOL = 1e-8
SVD_REL_TOL = 1e-6
OFFAXIS_TOL = 1e-7
DECISION_DENOM_TOL = 1e-12
DIRECTION_NORM_ATOL = 1e-5
SUPPORT_GATE = {
    "train_error_min": 60,
    "train_correct_reference_min": 60,
    "dev_error_min": 30,
    "dev_correct_reference_min": 30,
}
ROUTER_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
ROUTER_ROC_MIN = 0.75
ROUTER_PRECISION_MIN = 0.50
ROUTER_SEED = 42
NULL_SEED = 20260731
NULL_DRAWS = 10000
MAX_PROJECTED_SECONDS = 4 * 60 * 60

FILE_HASHES = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
}
FILE_SIZES = {
    "config.json": 728,
    "generation_config.json": 239,
    "merges.txt": 1671853,
    "model.safetensors.index.json": 32878,
    "tokenizer.json": 11422654,
    "tokenizer_config.json": 9732,
    "vocab.json": 2776833,
    "model-00001-of-00005.safetensors": 3996250744,
    "model-00002-of-00005.safetensors": 3993160032,
    "model-00003-of-00005.safetensors": 3959604768,
    "model-00004-of-00005.safetensors": 3187841392,
    "model-00005-of-00005.safetensors": 1244659840,
}
INPUT_HASHES = {
    "raw_source": "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c",
    "baseline_source": "46d3bdf5e598fec3683b0fe2b6f8d22d75086268f0573de9bb9c0520e3743944",
    "train_manifest": "7b843b43080e946595eb48ffbad800b5b7223cc693ccc2c9c49da47c2393b1fc",
    "dev_manifest": "2d2f078a47ae7702b5ccceb23a86d9d4cf223d7b7cc0de4eb8090e809afa3610",
    "baseline_index": "691e9c6ea72ecce27cf211a95fd44d892adc400de77d56c7d313227632388bab",
}
RESULT_NAMES = {
    "QWEN3_STAGE1_BASELINE_ROWS.jsonl",
    "QWEN3_STAGE1_BASELINE_SUMMARY.json",
    "QWEN3_STAGE1_BASELINE_SUMMARY.md",
    "QWEN3_STAGE1_COMPLETE_CHANNEL_LEDGER.csv",
    "QWEN3_STAGE1_ACTIVATIONS.safetensors",
    "QWEN3_STAGE1_DIRECTION_GRAM.json",
    "QWEN3_STAGE1_ROUTER_METRICS.json",
    "QWEN3_STAGE1_GEOMETRY_ROWS.csv",
    "QWEN3_STAGE1_GEOMETRY_ANALYSIS.json",
    "QWEN3_STAGE1_GEOMETRY_TABLE.md",
    "QWEN3_STAGE1_RAW_GRADIENTS.safetensors",
    "QWEN3_STAGE2_BRANCH_HANDOFF.md",
    "QWEN3_STAGE1_ENVIRONMENT.json",
    "QWEN3_STAGE1_RUNTIME.json",
    "QWEN3_STAGE1_LOG.jsonl",
    "QWEN3_STAGE1_RESULT_HASHES.json",
    "QWEN3_STAGE1_FAILURE.json",
}
BASELINE_ROW_SCHEMA = {
    "schema_version": "integer=1",
    "project_index": "integer; authorized train/dev index only",
    "sample_id": "immutable UUID string",
    "split": "enum[train,dev]",
    "gold": f"enum[{','.join(MODES)}]",
    "scores": f"object with exactly {MODES}; finite float mean candidate log-probabilities",
    "prediction": f"enum[{','.join(MODES)}]",
    "runner_up": f"enum[{','.join(MODES)}]",
    "candidate_lengths": f"object with exactly {MODES}; positive integers",
    "prompt_tokens": "positive integer",
    "margin_vector": "object with all 12 ordered pairwise score differences",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def log_event(event: str, **fields: Any) -> None:
    payload = {"schema_version": 1, "utc": utc_now(), "event": event, **fields}
    with (OUT / "QWEN3_STAGE1_LOG.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def append_retry(mode: str, command: str, exit_code: int, reason: str | None, invalidated: list[str]) -> None:
    if RETRY_LEDGER.exists():
        obj = read_json(RETRY_LEDGER)
    else:
        obj = {"schema_version": 1, "invocations": []}
    script = ROOT / "scripts/qwen3_8b_stage0_1.py"
    obj["invocations"].append({
        "utc": utc_now(),
        "mode": mode,
        "command": command,
        "exit_code": exit_code,
        "reason": reason,
        "outputs_invalidated": invalidated,
        "script_commit": git_head() if (ROOT / ".git").exists() else None,
        "script_sha256": sha256_file(script) if script.exists() else None,
    })
    write_json(RETRY_LEDGER, obj)


def index_order_hash(values: Iterable[Any]) -> str:
    return sha256_bytes(("\n".join(map(str, values)) + "\n").encode())


def validate_split_manifest(path: Path, split: str) -> list[int]:
    if sha256_file(path) != INPUT_HASHES[f"{split}_manifest"]:
        raise RuntimeError(f"{split} manifest hash mismatch")
    values = read_json(path)
    if not isinstance(values, list) or any(type(x) is not int for x in values):
        raise RuntimeError(f"invalid {split} manifest schema")
    if values != sorted(values) or len(values) != len(set(values)):
        raise RuntimeError(f"invalid {split} manifest order/uniqueness")
    if len(values) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"invalid {split} count")
    if any(x < 0 or x >= TOTAL_ROWS for x in values):
        raise RuntimeError(f"invalid {split} index range")
    return values


def structural_offsets(path: Path) -> tuple[list[tuple[int, int]], str]:
    offsets: list[tuple[int, int]] = []
    h = hashlib.sha256()
    offset = 0
    with path.open("rb") as f:
        for line_no, raw in enumerate(f):
            if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                raise RuntimeError(f"invalid source line ending {line_no}")
            h.update(raw)
            offsets.append((offset, len(raw) - 1))
            offset += len(raw)
    if len(offsets) != TOTAL_ROWS:
        raise RuntimeError(f"raw row count {len(offsets)} != {TOTAL_ROWS}")
    return offsets, h.hexdigest()


def load_baseline_index() -> dict[int, dict[str, str]]:
    if sha256_file(BASELINE_INDEX) != INPUT_HASHES["baseline_index"]:
        raise RuntimeError("baseline index hash mismatch")
    with BASELINE_INDEX.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["row_index"]): r for r in rows}


def direct_seek_json(path: Path, offset: int, length: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(offset)
        raw = f.read(length)
    if len(raw) != length or b"\n" in raw or b"\r" in raw:
        raise RuntimeError("direct-seek boundary violation")
    return json.loads(raw)


def build_block0_index() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path, key in ((RAW_SOURCE, "raw_source"), (BASELINE_SOURCE, "baseline_source")):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source missing/nonregular/symlink: {path}")
        if sha256_file(path) != INPUT_HASHES[key]:
            raise RuntimeError(f"source hash mismatch: {path}")
    train = validate_split_manifest(TRAIN_MANIFEST, "train")
    dev = validate_split_manifest(DEV_MANIFEST, "dev")
    if set(train) & set(dev):
        raise RuntimeError("train/dev overlap")
    if TOTAL_ROWS - len(train) - len(dev) != SEALED_EVALUATION_COUNT:
        raise RuntimeError("sealed evaluation structural count mismatch")
    offsets, raw_hash = structural_offsets(RAW_SOURCE)
    if raw_hash != INPUT_HASHES["raw_source"]:
        raise RuntimeError("raw source streaming hash mismatch")
    permutation = np.random.default_rng(SPLIT_SEED).permutation(TOTAL_ROWS)
    baseline_map = load_baseline_index()
    allowed = [(i, "train") for i in train] + [(i, "dev") for i in dev]
    output: list[dict[str, Any]] = []
    uuid_mismatches = 0
    for project_index, split in allowed:
        raw_row = int(permutation[project_index])
        offset, length = offsets[raw_row]
        sample = direct_seek_json(RAW_SOURCE, offset, length)
        b = baseline_map.get(project_index)
        if b is None or b["split_label"] != split:
            raise RuntimeError("baseline index correspondence failure")
        baseline = direct_seek_json(BASELINE_SOURCE, int(b["byte_offset"]), int(b["byte_length"]))
        if sample.get("uuid") != baseline.get("uuid"):
            uuid_mismatches += 1
        if sorted(sample.get("answers", {})) != sorted(MODES):
            raise RuntimeError("authorized row lacks exact four answer modes")
        output.append({
            "project_index": project_index,
            "raw_row_index": raw_row,
            "byte_offset": offset,
            "byte_length": length,
            "split": split,
            "sample_id": sample["uuid"],
            "sample_id_sha256": sha256_bytes(sample["uuid"].encode()),
        })
    if uuid_mismatches:
        raise RuntimeError(f"authorized UUID mismatches: {uuid_mismatches}")
    output.sort(key=lambda r: (0 if r["split"] == "train" else 1, r["project_index"]))
    with INDEX.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        w.writeheader(); w.writerows(output)
    manifest = {
        "schema_version": 1,
        "purpose": "QWEN3_TRAIN_DEV_SELECTIVE_ACCESS_ONLY",
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": "test",
        "dataset_split": "mcq",
        "raw_source": str(RAW_SOURCE.relative_to(ROOT)),
        "raw_source_sha256": raw_hash,
        "raw_source_rows_structurally_scanned": TOTAL_ROWS,
        "raw_source_payloads_parsed": len(output),
        "non_allowlisted_payloads_parsed": 0,
        "evaluation_manifest_opened": False,
        "evaluation_index_emitted": False,
        "sealed_evaluation_count_structural_only": SEALED_EVALUATION_COUNT,
        "split_seed": SPLIT_SEED,
        "shuffle_algorithm": "numpy.random.default_rng(42).permutation(3652), numpy 1.26.3",
        "permutation_sha256_int64_le": sha256_bytes(np.asarray(permutation, dtype="<i8").tobytes()),
        "counts": EXPECTED_COUNTS,
        "train_project_index_order_sha256": index_order_hash(train),
        "dev_project_index_order_sha256": index_order_hash(dev),
        "train_sample_id_order_sha256": index_order_hash([r["sample_id"] for r in output if r["split"] == "train"]),
        "dev_sample_id_order_sha256": index_order_hash([r["sample_id"] for r in output if r["split"] == "dev"]),
        "uuid_correspondence_checked_against_authorized_baseline_rows": len(output),
        "uuid_mismatch_count": uuid_mismatches,
        "index_path": str(INDEX.relative_to(ROOT)),
        "index_sha256": sha256_file(INDEX),
        "access_contract": "Consumers seek directly to emitted offsets and parse only emitted train/dev rows.",
    }
    write_json(INDEX_MANIFEST, manifest)
    if not RETRY_LEDGER.exists():
        write_json(RETRY_LEDGER, {"schema_version": 1, "invocations": []})
    print(json.dumps(manifest, indent=2, sort_keys=True))


class TrainDevData:
    def __init__(self) -> None:
        manifest = read_json(INDEX_MANIFEST)
        if manifest["index_sha256"] != sha256_file(INDEX):
            raise RuntimeError("selective index hash mismatch")
        with INDEX.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.records = {
            int(r["project_index"]): {
                **r,
                "project_index": int(r["project_index"]),
                "raw_row_index": int(r["raw_row_index"]),
                "byte_offset": int(r["byte_offset"]),
                "byte_length": int(r["byte_length"]),
            }
            for r in rows
        }
        self.train = sorted(i for i, r in self.records.items() if r["split"] == "train")
        self.dev = sorted(i for i, r in self.records.items() if r["split"] == "dev")
        if len(self.train) != 2556 or len(self.dev) != 548:
            raise RuntimeError("selective index count mismatch")
        self._cache: dict[int, dict[str, Any]] = {}

    def sample(self, index: int) -> dict[str, Any]:
        if index not in self.records:
            raise RuntimeError("firewall: requested non-authorized project index")
        if index not in self._cache:
            r = self.records[index]
            obj = direct_seek_json(RAW_SOURCE, r["byte_offset"], r["byte_length"])
            if obj.get("uuid") != r["sample_id"]:
                raise RuntimeError("sample identity mismatch")
            self._cache[index] = obj
        return self._cache[index]


def parse_tools(tools: Any) -> list[Any]:
    out = []
    for t in tools or []:
        if isinstance(t, str):
            try:
                out.append(json.loads(t))
            except json.JSONDecodeError:
                out.append({"raw": t})
        else:
            out.append(t)
    return out


def build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a helpful assistant.\n\n"
        "Given the user's question and the available tools, choose the most "
        "appropriate response from the provided options."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": sample["question"]}]


def render_prompt(tokenizer: Any, sample: dict[str, Any]) -> tuple[str, list[dict[str, str]], list[Any]]:
    messages = build_messages(sample)
    tools = parse_tools(sample.get("tools"))
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    required = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not text.endswith(required):
        raise RuntimeError("READOUT_INVALID_THINKING_MODE: deterministic empty-think suffix absent")
    between = text.rsplit("<think>\n", 1)[-1].split("\n</think>", 1)[0]
    if between != "":
        raise RuntimeError("READOUT_INVALID_THINKING_MODE: nonempty reasoning content")
    return text, messages, tools


def verify_block0_hashes() -> dict[str, Any]:
    manifest = read_json(PROTOCOL_HASHES)
    mismatches = []
    for rel, expected in manifest["artifact_sha256"].items():
        path = ROOT / rel
        if not path.is_file():
            mismatches.append([rel, "MISSING", expected])
        else:
            got = sha256_file(path)
            if got != expected:
                mismatches.append([rel, got, expected])
    if mismatches:
        raise RuntimeError(f"Block0 artifact hash mismatch: {mismatches}")
    return manifest


def verify_model_snapshot(hash_weights: bool = True) -> dict[str, Any]:
    observed = {}
    for name, expected in FILE_HASHES.items():
        path = MODEL_DIR / name
        if not path.exists() or not path.resolve().is_file():
            raise RuntimeError(f"model snapshot missing {name}")
        size = path.resolve().stat().st_size
        if size != FILE_SIZES[name]:
            raise RuntimeError(f"model snapshot size mismatch {name}: {size}")
        got = sha256_file(path.resolve()) if (hash_weights or not name.startswith("model-")) else None
        if got is not None and got != expected:
            raise RuntimeError(f"model snapshot SHA256 mismatch {name}")
        observed[name] = {"size": size, "sha256": got or expected, "resolved_path": str(path.resolve())}
    config = read_json(MODEL_DIR / "config.json")
    identity = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "architecture": config["architectures"],
        "model_type": config["model_type"],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "torch_dtype": config["torch_dtype"],
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "files": observed,
    }
    if identity["hidden_size"] != HIDDEN_SIZE or identity["num_hidden_layers"] != NEW_NUM_LAYERS:
        raise RuntimeError("model architecture identity mismatch")
    return identity


def preflight() -> None:
    if git_status():
        raise RuntimeError("preflight requires clean committed worktree")
    block0 = verify_block0_hashes()
    model_identity = verify_model_snapshot(hash_weights=True)
    import torch
    import transformers
    import tokenizers
    if transformers.__version__ != "4.51.0" or tokenizers.__version__ != "0.21.1":
        raise RuntimeError("frozen transformers/tokenizers version mismatch")
    if torch.__version__ != "2.1.2+cu121":
        raise RuntimeError("frozen torch version mismatch")
    if L_OBS != round(OLD_L_OBS / (OLD_NUM_LAYERS - 1) * (NEW_NUM_LAYERS - 1)):
        raise RuntimeError("L_obs mapping mismatch")
    if L_INJ != round(OLD_L_INJ / (OLD_NUM_LAYERS - 1) * (NEW_NUM_LAYERS - 1)):
        raise RuntimeError("L_inj mapping mismatch")
    for path, key in (
        (RAW_SOURCE, "raw_source"),
        (TRAIN_MANIFEST, "train_manifest"),
        (DEV_MANIFEST, "dev_manifest"),
    ):
        if not path.is_file() or path.is_symlink() or sha256_file(path) != INPUT_HASHES[key]:
            raise RuntimeError(f"preflight input identity failure: {path}")
    index_manifest = read_json(INDEX_MANIFEST)
    if (
        index_manifest.get("evaluation_manifest_opened") is not False
        or index_manifest.get("evaluation_index_emitted") is not False
        or index_manifest.get("non_allowlisted_payloads_parsed") != 0
    ):
        raise RuntimeError("selective-access firewall manifest failure")
    subprocess.run(["git", "lfs", "version"], cwd=ROOT, check=True, capture_output=True, text=True)
    lfs_probe = "final/results/qwen3_stage0_1/directions/PREFLIGHT_PROBE.safetensors"
    attr = subprocess.check_output(["git", "check-attr", "filter", "--", lfs_probe], cwd=ROOT, text=True).strip()
    if not attr.endswith(": lfs"):
        raise RuntimeError("Git LFS attribute unavailable for numerical outputs")
    for name in RESULT_NAMES:
        if (OUT / name).exists():
            raise RuntimeError(f"result namespace not empty: {name}")
    for name in ("directions", "routers"):
        if (OUT / name).exists():
            raise RuntimeError(f"result namespace directory not empty: {name}")
    data = TrainDevData()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    if not isinstance(tokenizer.chat_template, str) or sha256_bytes(tokenizer.chat_template.encode()) != "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8":
        raise RuntimeError("tokenizer runtime chat-template SHA256 mismatch")

    train_by_id = sorted((data.sample(i)["uuid"], i) for i in data.train)
    sample_id, index = train_by_id[0]
    sample = data.sample(index)
    prompt, messages, tools = render_prompt(tokenizer, sample)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidates = {}
    for mode in MODES:
        text = sample["answers"][mode]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"empty candidate {mode}")
        candidates[mode] = {
            "text": text,
            "token_ids": ids,
            "candidate_tokens": len(ids),
            "score_logit_positions_inclusive": [len(prompt_ids) - 1, len(prompt_ids) + len(ids) - 2],
            "label_positions_inclusive": [len(prompt_ids), len(prompt_ids) + len(ids) - 1],
            "prompt_excluded_from_average": True,
            "scoring_mask": [0] * len(prompt_ids) + [1] * len(ids),
        }
    pilot = {
        "schema_version": 1,
        "scope": "AUTHORIZED_TRAIN_PROMPT_ONLY_NO_MODEL_SCORE",
        "selection": "lexicographically smallest immutable TRAIN sample UUID",
        "project_index": index,
        "sample_id": sample_id,
        "system_user_messages": messages,
        "tools": tools,
        "tools_placement": "official Qwen3 chat template <tools> block before system/user turns",
        "template_call": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "tokenize": False,
            "tools_passed_separately": True,
        },
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": sha256_bytes(prompt.encode()),
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": len(prompt_ids),
        "assistant_candidate_boundary": "immediately after deterministic empty <think>...</think> block",
        "generated_reasoning_tokens_before_candidate": 0,
        "mode_order": MODES,
        "candidates": candidates,
    }
    write_json(PROMPT_PILOT, pilot)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "utc": utc_now(),
        "block0_commit": git_head(),
        "block0_hash_manifest_sha256": sha256_file(PROTOCOL_HASHES),
        "script_sha256": sha256_file(ROOT / "scripts/qwen3_8b_stage0_1.py"),
        "protocol_artifacts_verified": len(block0["artifact_sha256"]),
        "model_identity": model_identity,
        "tokenizer_config_sha256": FILE_HASHES["tokenizer_config.json"],
        "chat_template_sha256_utf8": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "thinking_mode": "DETERMINISTIC_EMPTY_THINK_BLOCK_ENABLE_THINKING_FALSE",
        "train_dev_firewall": {
            "train_count": len(data.train),
            "dev_count": len(data.dev),
            "evaluation_rows_ids_labels_predictions_scores_supports_aggregates_accessed": False,
            "evaluation_manifest_or_index_opened": False,
            "non_allowlisted_payloads_parsed": 0,
        },
        "layer_mapping": {"old": {"layers": 28, "L_obs": 20, "L_inj": 16}, "new": {"layers": 36, "L_obs": 26, "L_inj": 21}},
        "candidate_scoring": {
            "teacher_forced_forwards_per_sample": 4,
            "candidate_specific_multi_token": True,
            "mean_candidate_token_log_probability": True,
            "single_token_proxy": False,
            "free_generation": False,
            "mode_order": MODES,
        },
        "full_model_loaded": False,
        "qwen3_scores_computed": False,
        "qwen3_channels_counted": False,
        "stage2_code_or_outputs_created": False,
        "prompt_pilot_path": str(PROMPT_PILOT.relative_to(ROOT)),
        "prompt_pilot_sha256": sha256_file(PROMPT_PILOT),
    }
    write_json(PREFLIGHT, payload)
    append_retry("preflight", "python scripts/qwen3_8b_stage0_1.py --preflight", 0, None, [])
    write_json(PREFLIGHT_HASHES, {
        "schema_version": 1,
        "utc": utc_now(),
        "sha256": {
            str(PREFLIGHT.relative_to(ROOT)): sha256_file(PREFLIGHT),
            str(PROMPT_PILOT.relative_to(ROOT)): sha256_file(PROMPT_PILOT),
            str(RETRY_LEDGER.relative_to(ROOT)): sha256_file(RETRY_LEDGER),
        },
    })
    print(json.dumps(payload, indent=2, sort_keys=True))


ATTENTION_GUARD_THRESHOLD_TOKENS = 4096
ATTENTION_GUARD_HEAD_CHUNK = 8


def install_long_sequence_attention_guard() -> dict[str, Any]:
    """Engineering-only memory guard for the frozen eager attention path.

    The frozen attention implementation stays "eager". Sequences at or below
    the threshold are dispatched to the unmodified upstream function, so the
    overwhelming majority of rows follow a bit-for-bit unchanged path. Longer
    sequences evaluate the identical eager expression in head chunks: softmax
    is taken over the key axis independently for each (head, query) row, so
    partitioning the head axis changes no value while bounding the transient
    [1, heads, S, S] float32 buffer. Attention probabilities are returned as
    None because the runner never requests output_attentions and the decoder
    layer discards them; rebuilding them would defeat the guard.

    This changes no scientific definition: attention implementation, dtype,
    batch size, populations, candidate construction and scoring are untouched.
    """
    import torch
    from torch import nn
    import transformers.models.qwen3.modeling_qwen3 as qwen3

    upstream = qwen3.eager_attention_forward
    if getattr(upstream, "_sakiko_guard", False):
        return {"installed": False, "reason": "already installed"}

    def guarded(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.shape[-2] <= ATTENTION_GUARD_THRESHOLD_TOKENS:
            return upstream(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        key_states = qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen3.repeat_kv(value, module.num_key_value_groups)
        pieces = []
        for start in range(0, query.shape[1], ATTENTION_GUARD_HEAD_CHUNK):
            stop = min(start + ATTENTION_GUARD_HEAD_CHUNK, query.shape[1])
            weights = torch.matmul(query[:, start:stop], key_states[:, start:stop].transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
            weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = nn.functional.dropout(weights, p=dropout, training=module.training)
            pieces.append(torch.matmul(weights, value_states[:, start:stop]))
            del weights
        attn_output = torch.cat(pieces, dim=1).transpose(1, 2).contiguous()
        return attn_output, None

    guarded._sakiko_guard = True
    qwen3.eager_attention_forward = guarded
    return {
        "installed": True,
        "threshold_tokens": ATTENTION_GUARD_THRESHOLD_TOKENS,
        "head_chunk": ATTENTION_GUARD_HEAD_CHUNK,
        "below_threshold_path": "unmodified upstream eager_attention_forward",
        "scientific_definitions_changed": False,
    }


def set_determinism() -> None:
    random.seed(42); np.random.seed(42)
    import torch
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def load_model() -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from importlib.metadata import version as package_version
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_identity = verify_model_snapshot(hash_weights=False)
    attention_guard = install_long_sequence_attention_guard()
    set_determinism()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map="cuda:0",
    )
    model.eval(); model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "tokenizers": package_version("tokenizers"),
        "accelerate": package_version("accelerate"),
        "safetensors": package_version("safetensors"),
        "scikit_learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0),
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "model_load_seconds": time.perf_counter() - started,
        "model_identity": model_identity,
        "long_sequence_attention_guard": attention_guard,
    }
    if model.config.num_hidden_layers != NEW_NUM_LAYERS or model.config.hidden_size != HIDDEN_SIZE:
        raise RuntimeError("loaded model architecture mismatch")
    return model, tokenizer, device, environment


def candidate_ids(tokenizer: Any, prompt: str, candidate: str) -> tuple[list[int], list[int]]:
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    cids = tokenizer.encode(candidate, add_special_tokens=False)
    if not pids or not cids:
        raise RuntimeError("empty prompt or candidate tokens")
    if len(pids) + len(cids) > MAX_TOKENS:
        raise RuntimeError("predeclared structural exclusion: sequence exceeds 8192 tokens")
    return pids, cids


def score_candidate(
    model: Any,
    device: Any,
    pids: list[int],
    cids: list[int],
    capture_modules: dict[str, Any] | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    import torch
    ids = torch.tensor([pids + cids], dtype=torch.long, device=device)
    positions = torch.arange(len(pids) - 1, len(pids) + len(cids) - 1, device=device)
    captured: dict[str, np.ndarray] = {}
    handles = []
    for key, module in (capture_modules or {}).items():
        def hook(_m: Any, _inp: Any, out: Any, capture_key: str = key) -> None:
            tensor = out[0] if isinstance(out, tuple) else out
            captured[capture_key] = tensor[0, len(pids) - 1].detach().float().cpu().numpy()
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            logits = model(ids, use_cache=False, logits_to_keep=positions).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            labels = torch.tensor(cids, dtype=torch.long, device=device)
            score = float(lp[torch.arange(len(cids), device=device), labels].mean().item())
    finally:
        for handle in handles:
            handle.remove()
    return score, captured


def score_sample(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: dict[str, Any],
    capture_modules: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prompt, _, _ = render_prompt(tokenizer, sample)
    scores: dict[str, float] = {}; lengths: dict[str, int] = {}; activations: dict[str, np.ndarray] = {}
    for mode_i, mode in enumerate(MODES):
        pids, cids = candidate_ids(tokenizer, prompt, sample["answers"][mode])
        score, cap = score_candidate(model, device, pids, cids, capture_modules if mode_i == 0 else None)
        scores[mode] = score; lengths[mode] = len(cids)
        activations.update(cap)
    values = np.asarray([scores[m] for m in MODES], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("nonfinite four-candidate score")
    order = np.argsort(-values, kind="stable")
    if values[order[0]] - values[order[1]] <= SCORE_TIE_TOL:
        raise RuntimeError("unresolved score tie")
    pred = MODES[int(order[0])]; runner = MODES[int(order[1])]
    margins = {f"{a}_minus_{b}": float(scores[a] - scores[b]) for a in MODES for b in MODES if a != b}
    return {
        "scores": scores,
        "prediction": pred,
        "runner_up": runner,
        "candidate_lengths": lengths,
        "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "margin_vector": margins,
    }, activations


def fixed_pilot(model: Any, tokenizer: Any, device: Any, data: TrainDevData) -> dict[str, Any]:
    selected = [i for _, i in sorted((data.sample(i)["uuid"], i) for i in data.train)[:8]]
    repeats = []
    start = time.perf_counter()
    for rep in range(2):
        out = []
        for i in selected:
            result, _ = score_sample(model, tokenizer, device, data.sample(i), None)
            out.append({"project_index": i, "sample_id": data.sample(i)["uuid"], **result})
        repeats.append(out)
    elapsed = time.perf_counter() - start
    max_abs = 0.0
    for a, b in zip(repeats[0], repeats[1]):
        if a["prediction"] != b["prediction"] or a["runner_up"] != b["runner_up"]:
            raise RuntimeError("fixed pilot prediction/runner-up replay mismatch")
        max_abs = max(max_abs, max(abs(a["scores"][m] - b["scores"][m]) for m in MODES))
    if max_abs > SCORE_REPLAY_ATOL:
        raise RuntimeError(f"fixed pilot score replay deviation {max_abs}")
    seconds_per_sample = elapsed / 16.0
    prediction_hashes = [index_order_hash([f"{row['sample_id']}\t{row['prediction']}" for row in repeat]) for repeat in repeats]
    if prediction_hashes[0] != prediction_hashes[1]:
        raise RuntimeError("fixed pilot prediction-vector hash mismatch")
    baseline_projection = seconds_per_sample * (len(data.train) + len(data.dev))
    gradient_started = time.perf_counter()
    for i in selected[:2]:
        sample = data.sample(i)
        prompt, _, _ = render_prompt(tokenizer, sample)
        for mode in MODES:
            pids, cids = candidate_ids(tokenizer, prompt, sample["answers"][mode])
            score, gradient, _ = gradient_candidate(model, device, pids, cids)
            if not np.isfinite(score) or gradient.shape[1] != HIDDEN_SIZE or not np.all(np.isfinite(gradient)):
                raise RuntimeError("fixed gradient-timing pilot invalid")
    gradient_elapsed = time.perf_counter() - gradient_started
    gradient_seconds_per_sample = gradient_elapsed / 2.0
    geometry_upper_bound = gradient_seconds_per_sample * len(data.dev) * 1.2
    stage1_projection = baseline_projection + geometry_upper_bound
    return {
        "selection": "eight TRAIN rows by ascending immutable sample UUID",
        "project_indices": selected,
        "sample_ids": [data.sample(i)["uuid"] for i in selected],
        "repetitions": 2,
        "score_replay_atol": SCORE_REPLAY_ATOL,
        "max_absolute_score_deviation": max_abs,
        "predictions_identical": True,
        "runner_ups_identical": True,
        "prediction_vector_sha256_repetitions": prediction_hashes,
        "candidate_masks_valid": True,
        "elapsed_seconds": elapsed,
        "seconds_per_sample": seconds_per_sample,
        "projected_baseline_seconds": baseline_projection,
        "gradient_timing_train_rows": 2,
        "gradient_timing_elapsed_seconds": gradient_elapsed,
        "gradient_seconds_per_sample": gradient_seconds_per_sample,
        "projected_geometry_upper_bound_seconds": geometry_upper_bound,
        "projected_stage1_upper_bound_seconds": stage1_projection,
        "full_rows": repeats,
    }


def baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import f1_score
    gold = [r["gold"] for r in rows]; pred = [r["prediction"] for r in rows]
    confusion = {g: {p: 0 for p in MODES} for g in MODES}
    for g, p in zip(gold, pred): confusion[g][p] += 1
    return {
        "n": len(rows),
        "accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])),
        "macro_f1_four_mode": float(f1_score(gold, pred, labels=MODES, average="macro", zero_division=0)),
        "gold_support": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(pred)),
        "confusion_gold_rows_prediction_columns": confusion,
        "prediction_vector_sha256": index_order_hash([f"{r['sample_id']}\t{r['prediction']}" for r in rows]),
    }


def channel_name(gold: str, source: str) -> str:
    return f"{gold}__to__{source}"


def build_channel_ledger(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for gold in MODES:
        tr_ref = sum(r["gold"] == gold and r["prediction"] == gold for r in train_rows)
        dv_ref = sum(r["gold"] == gold and r["prediction"] == gold for r in dev_rows)
        for pred in MODES:
            if pred == gold: continue
            tr_err = sum(r["gold"] == gold and r["prediction"] == pred for r in train_rows)
            dv_err = sum(r["gold"] == gold and r["prediction"] == pred for r in dev_rows)
            failures = []
            if tr_err < SUPPORT_GATE["train_error_min"]: failures.append("TRAIN_ERROR_LT_60")
            if tr_ref < SUPPORT_GATE["train_correct_reference_min"]: failures.append("TRAIN_CORRECT_REFERENCE_LT_60")
            if dv_err < SUPPORT_GATE["dev_error_min"]: failures.append("DEV_ERROR_LT_30")
            if dv_ref < SUPPORT_GATE["dev_correct_reference_min"]: failures.append("DEV_CORRECT_REFERENCE_LT_30")
            output.append({
                "channel": channel_name(gold, pred), "gold": gold, "source": pred,
                "train_error_support": tr_err, "train_correct_reference_support": tr_ref,
                "dev_error_support": dv_err, "dev_correct_reference_support": dv_ref,
                "support_eligible": not failures,
                "support_exclusion_reasons": ";".join(failures),
                "router_eligible": False, "direction_valid": False,
                "geometry_valid": False, "stage2_eligible": False,
                "stage2_exclusion_reasons": "NOT_YET_EVALUATED",
            })
    return output


def save_baseline(
    rows: list[dict[str, Any]],
    activations_by_layer: dict[str, np.ndarray],
    pilot: dict[str, Any],
    environment: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from safetensors.numpy import save_file
    path = OUT / "QWEN3_STAGE1_BASELINE_ROWS.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    ledger = build_channel_ledger(train_rows, dev_rows)
    summary = {
        "schema_version": 1,
        "scope": "TRAIN_DEV_ONLY_DEVELOPMENTAL",
        "pilot": pilot,
        "train": baseline_metrics(train_rows),
        "baseline_row_schema": BASELINE_ROW_SCHEMA,
        "baseline_row_schema_sha256": sha256_bytes(stable_json(BASELINE_ROW_SCHEMA).encode()),
        "dev": baseline_metrics(dev_rows),
        "directed_error_transitions": ledger,
        "support_gate": SUPPORT_GATE,
        "evaluation_accessed": False,
    }
    write_json(OUT / "QWEN3_STAGE1_BASELINE_SUMMARY.json", summary)
    lines = ["# Qwen3-8B Stage 1 train/dev baseline", "", "Developmental only; evaluation was not accessed.", "",
             "| split | n | accuracy | macro-F1 |", "|---|---:|---:|---:|",
             f"| train | {len(train_rows)} | {summary['train']['accuracy']:.6f} | {summary['train']['macro_f1_four_mode']:.6f} |",
             f"| dev | {len(dev_rows)} | {summary['dev']['accuracy']:.6f} | {summary['dev']['macro_f1_four_mode']:.6f} |", "",
             "## Complete directed transitions", "", "| channel | train error | train ref | dev error | dev ref | support eligible | exclusion |",
             "|---|---:|---:|---:|---:|---|---|"]
    for c in ledger:
        lines.append(f"| {c['channel']} | {c['train_error_support']} | {c['train_correct_reference_support']} | {c['dev_error_support']} | {c['dev_correct_reference_support']} | {c['support_eligible']} | {c['support_exclusion_reasons'] or '—'} |")
    (OUT / "QWEN3_STAGE1_BASELINE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_file({key: value.astype(np.float32) for key, value in activations_by_layer.items()}, OUT / "QWEN3_STAGE1_ACTIVATIONS.safetensors", metadata={
        "schema_version": "1", "scope": "train_dev_only", "layers": json.dumps([L_OBS, L_INJ]),
        "site": "mlp_output_last_prompt_token",
        "sample_order_json": json.dumps([{"project_index": r["project_index"], "sample_id": r["sample_id"], "split": r["split"]} for r in rows], separators=(",", ":")),
        "contains_dataset_text": "false", "dtype": "float32",
        "tensor_shapes": json.dumps({key: list(value.shape) for key, value in activations_by_layer.items()}, sort_keys=True),
    })
    write_json(OUT / "QWEN3_STAGE1_ENVIRONMENT.json", environment)
    return summary, ledger


def fit_directions_routers(rows: list[dict[str, Any]], activations: np.ndarray, ledger: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from safetensors.numpy import save_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    row_at = {r["project_index"]: n for n, r in enumerate(rows)}
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    directions: dict[str, np.ndarray] = {}; routers = {}; metrics = {}
    direction_dir = OUT / "directions"; router_dir = OUT / "routers"
    direction_dir.mkdir(exist_ok=True); router_dir.mkdir(exist_ok=True)
    for c in ledger:
        if not c["support_eligible"]: continue
        ch, gold, source = c["channel"], c["gold"], c["source"]
        err = [r for r in train_rows if r["gold"] == gold and r["prediction"] == source]
        ref = [r for r in train_rows if r["gold"] == gold and r["prediction"] == gold]
        err_ids = [r["sample_id"] for r in err]; ref_ids = [r["sample_id"] for r in ref]
        err_a = activations[[row_at[r["project_index"]] for r in err]].astype(np.float64)
        ref_a = activations[[row_at[r["project_index"]] for r in ref]].astype(np.float64)
        raw = ref_a.mean(0) - err_a.mean(0); raw_norm = float(np.linalg.norm(raw))
        if not np.all(np.isfinite(raw)) or raw_norm <= CONTRAST_NORM_TOL:
            c["direction_valid"] = False; continue
        unit = (raw / raw_norm).astype(np.float32)
        if abs(float(np.linalg.norm(unit)) - 1.0) > DIRECTION_NORM_ATOL:
            raise RuntimeError(f"unit direction norm failure {ch}")
        directions[ch] = unit; c["direction_valid"] = True
        unit_hash = sha256_bytes(np.ascontiguousarray(unit).tobytes())
        metadata = {
            "schema_version": "1", "channel": ch, "gold": gold, "source": source,
            "layer": str(L_OBS), "hidden_dimension": str(HIDDEN_SIZE), "dtype": "float32",
            "raw_norm": repr(raw_norm), "unit_vector_sha256": unit_hash,
            "sign_convention": "mean(correct gold reference)-mean(gold-to-source error); positive intended goldward",
            "error_population_sha256": index_order_hash(err_ids), "reference_population_sha256": index_order_hash(ref_ids),
            "error_n": str(len(err)), "reference_n": str(len(ref)), "contains_dataset_text": "false",
        }
        dpath = direction_dir / f"{ch}.safetensors"
        save_file({"direction": unit}, dpath, metadata=metadata)
        pos = err
        neg = [r for r in train_rows if r["gold"] == r["prediction"]]
        train_design = neg + pos
        X = activations[[row_at[r["project_index"]] for r in train_design]].astype(np.float64)
        y = np.asarray([0] * len(neg) + [1] * len(pos), dtype=np.int64)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000, tol=1e-4, random_state=ROUTER_SEED, class_weight=None, fit_intercept=True)
        clf.fit(scaler.transform(X), y)
        if int(clf.n_iter_[0]) >= 2000: raise RuntimeError(f"Router did not converge {ch}")
        dv_pos = [r for r in dev_rows if r["gold"] == gold and r["prediction"] == source]
        dv_correct = [r for r in dev_rows if r["gold"] == r["prediction"]]
        comp = dv_correct + dv_pos
        comp_y = np.asarray([0] * len(dv_correct) + [1] * len(dv_pos), dtype=np.int64)
        comp_p = clf.predict_proba(scaler.transform(activations[[row_at[r["project_index"]] for r in comp]]))[:, 1]
        roc = float(roc_auc_score(comp_y, comp_p)); pr = float(average_precision_score(comp_y, comp_p))
        operational = [r for r in dev_rows if r["prediction"] == source]
        op_y = np.asarray([int(r["gold"] == gold) for r in operational], dtype=np.int64)
        op_p = clf.predict_proba(scaler.transform(activations[[row_at[r["project_index"]] for r in operational]]))[:, 1]
        grid = []
        tau = None
        for t in ROUTER_GRID:
            pred = op_p >= t; tp = int(np.sum(pred & (op_y == 1))); fp = int(np.sum(pred & (op_y == 0)))
            tn = int(np.sum((~pred) & (op_y == 0))); fn = int(np.sum((~pred) & (op_y == 1)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            grid.append({"tau": t, "precision": precision, "recall": recall, "specificity": specificity, "routed": int(np.sum(pred)), "tp": tp, "fp": fp, "tn": tn, "fn": fn})
            if tau is None and precision >= ROUTER_PRECISION_MIN: tau = t
        selected = next((x for x in grid if x["tau"] == tau), None)
        eligible = roc >= ROUTER_ROC_MIN and selected is not None and selected["precision"] >= ROUTER_PRECISION_MIN
        c["router_eligible"] = bool(eligible)
        routers[ch] = {"scaler": scaler, "clf": clf, "tau": tau}
        rpath = router_dir / f"{ch}.safetensors"
        save_file({
            "scaler_mean": scaler.mean_.astype(np.float32), "scaler_scale": scaler.scale_.astype(np.float32),
            "classifier_coef": clf.coef_[0].astype(np.float32), "classifier_intercept": clf.intercept_.astype(np.float32),
        }, rpath, metadata={"schema_version": "1", "channel": ch, "classifier": "StandardScaler+LogisticRegression", "C": "1.0", "solver": "liblinear", "seed": "42", "tau": str(tau), "contains_dataset_text": "false"})
        metrics[ch] = {
            "train_positive_n": len(pos), "train_all_correct_negative_n": len(neg),
            "dev_comparable": {"positive_n": len(dv_pos), "all_correct_negative_n": len(dv_correct), "ROC_AUC": roc, "PR_AUC": pr},
            "dev_operational_source_matched_n": len(operational), "tau_grid": grid, "selected_tau": tau,
            "selected_metrics": selected, "eligible": bool(eligible), "converged_iterations": int(clf.n_iter_[0]),
            "direction_path": str(dpath.relative_to(ROOT)), "direction_file_sha256": sha256_file(dpath),
            "direction_unit_sha256": unit_hash, "direction_raw_norm": raw_norm,
            "router_path": str(rpath.relative_to(ROOT)), "router_file_sha256": sha256_file(rpath),
        }
    chs = sorted(directions)
    matrix = np.stack([directions[ch].astype(np.float64) for ch in chs]) if chs else np.empty((0, HIDDEN_SIZE))
    gram = matrix @ matrix.T if chs else np.empty((0, 0))
    gram_obj = {
        "channels": chs, "signed_cosine_matrix": gram.tolist(),
        "flags_abs_cosine_gt_0_4": [
            {"a": chs[i], "b": chs[j], "cosine": float(gram[i, j]), "flag": "NON_ORTHOGONAL_MISMATCHED_CONTROL"}
            for i in range(len(chs)) for j in range(i + 1, len(chs)) if abs(float(gram[i, j])) > 0.4
        ],
    }
    write_json(OUT / "QWEN3_STAGE1_DIRECTION_GRAM.json", gram_obj)
    write_json(OUT / "QWEN3_STAGE1_ROUTER_METRICS.json", {"schema_version": 1, "specification": {
        "feature": f"L{L_OBS} MLP output final prompt token", "train_negatives": "all correct train rows",
        "classifier": "StandardScaler + LogisticRegression(C=1,L2,liblinear,max_iter=2000,tol=1e-4,seed=42,class_weight=None)",
        "tau_grid": ROUTER_GRID, "tau_rule": "smallest tau with source-matched dev precision >=0.50",
        "eligibility": "dev comparable ROC-AUC >=0.75 and selected tau precision >=0.50",
    }, "channels": metrics})
    return metrics, directions, gram_obj


def gradient_candidate(model: Any, device: Any, pids: list[int], cids: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
    import torch
    ids = torch.tensor([pids + cids], dtype=torch.long, device=device)
    positions = torch.arange(len(pids) - 1, len(pids) + len(cids) - 1, device=device)
    leaf: dict[str, Any] = {}; prompt_state: dict[str, np.ndarray] = {}
    module = model.model.layers[L_INJ].mlp
    def hook(_m: Any, _inp: Any, out: Any) -> Any:
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


def compute_geometry_one(model: Any, tokenizer: Any, device: Any, sample: dict[str, Any], direction: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    prompt, _, _ = render_prompt(tokenizer, sample)
    scores = {}; grads = {}; states = {}; prompt_len = None
    for mode in MODES:
        pids, cids = candidate_ids(tokenizer, prompt, sample["answers"][mode])
        if prompt_len is None: prompt_len = len(pids)
        score, grad, state = gradient_candidate(model, device, pids, cids)
        scores[mode] = score; grads[mode] = grad; states[mode] = state
    assert prompt_len is not None
    full = np.stack([grads[m].sum(axis=0) for m in MODES]).astype(np.float32)
    common_shared = all(states[m].shape == states[MODES[0]].shape and np.max(np.abs(states[m] - states[MODES[0]])) <= PROMPT_STATE_ATOL for m in MODES[1:])
    tensors = {"full_effect": full}
    if common_shared:
        tensors["common_prompt"] = np.stack([grads[m][:prompt_len].sum(axis=0) for m in MODES]).astype(np.float32)
    records = []
    gold = sample["correct_answer"]
    # source is the frozen baseline prediction and is attached by caller.
    for variant, matrix in tensors.items():
        records.append(geometry_metrics(matrix, direction, gold, sample["_baseline_prediction"], variant))
    if not common_shared:
        records.append({"variant": "common_prompt", "status": "COMMON_PROMPT_STATE_NOT_SHARED"})
    return records, tensors


def geometry_metrics(matrix: np.ndarray, direction: np.ndarray, gold: str, source: str, variant: str) -> dict[str, Any]:
    from scipy import stats
    g = MODES.index(gold); p = MODES.index(source)
    contrast = matrix[g].astype(np.float64) - matrix[p].astype(np.float64)
    contrast_norm = float(np.linalg.norm(contrast))
    centered = matrix.astype(np.float64) - matrix.astype(np.float64).mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    tol = max(SVD_ABS_TOL, (float(singular[0]) if len(singular) else 0.0) * SVD_REL_TOL)
    rank = int(np.sum(singular > tol))
    base = {"variant": variant, "contrast_norm": contrast_norm, "singular_values": singular.tolist(), "svd_rank_tolerance": tol,
            "observed_rank": rank, "per_mode_gradient_norms": {m: float(np.linalg.norm(matrix[i])) for i, m in enumerate(MODES)}}
    if contrast_norm <= CONTRAST_NORM_TOL:
        return {**base, "status": "CONTRAST_AXIS_DEGENERATE"}
    if rank == 0:
        return {**base, "status": "DECISION_SUBSPACE_DEGENERATE"}
    w = contrast / contrast_norm; d = direction.astype(np.float64)
    s = float(d @ w); a_contrast = s * s; basis = vh[:rank]
    a_dec = float(np.sum((basis @ d) ** 2)); a_off = a_dec - a_contrast
    clipped = False
    if a_off < 0 and a_off >= -OFFAXIS_TOL: a_off = 0.0; clipped = True
    if a_off < -OFFAXIS_TOL:
        return {**base, "status": "GEOMETRY_INCONSISTENCY_NEGATIVE_OFFAXIS", "signed_contrast": s, "a_contrast": a_contrast, "a_dec": a_dec, "a_offaxis": a_off}
    q = a_off / a_dec if a_dec > DECISION_DENOM_TOL else None
    rank1_p95 = float(stats.beta.ppf(0.95, 0.5, (HIDDEN_SIZE - 1) / 2))
    rank_p95 = float(stats.beta.ppf(0.95, rank / 2, (HIDDEN_SIZE - rank) / 2))
    return {**base, "status": "VALID", "signed_contrast": s, "negative_sign": s < 0, "contrast_axis": w.astype(np.float32),
            "a_contrast": a_contrast, "a_dec": a_dec, "a_offaxis": a_off, "q_offaxis": q, "offaxis_clipped": clipped,
            "a_contrast_rank1_null_p95": rank1_p95, "a_contrast_exceeds_null_p95": a_contrast > rank1_p95,
            "a_dec_rank_matched_null_p95": rank_p95, "a_dec_exceeds_rank_null_p95": a_dec > rank_p95}


def distribution(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(x): return {"n": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x)>1 else 0.0, "min": float(x.min()),
            "p25": float(np.quantile(x,.25)), "median": float(np.median(x)), "p75": float(np.quantile(x,.75)), "max": float(x.max())}


def resultant_null(n: int) -> np.ndarray:
    if n <= 0: return np.asarray([], dtype=np.float64)
    rng = np.random.default_rng(NULL_SEED + n); resultant = np.ones(NULL_DRAWS, dtype=np.float64)
    beta_shape = (HIDDEN_SIZE - 1) / 2
    for _ in range(1, n):
        cosine = 2 * rng.beta(beta_shape, beta_shape, size=NULL_DRAWS) - 1
        resultant = np.sqrt(np.maximum(0, resultant * resultant + 1 + 2 * resultant * cosine))
    return resultant / n


def summarize_geometry(records: list[dict[str, Any]], axes: dict[tuple[str, str], list[np.ndarray]]) -> dict[str, Any]:
    channels = sorted({r["channel"] for r in records}); out = {}
    for ch in channels:
        out[ch] = {}
        for variant in ("full_effect", "common_prompt"):
            group = [r for r in records if r["channel"] == ch and r["variant"] == variant]
            valid = [r for r in group if r["status"] == "VALID"]
            s = [r["signed_contrast"] for r in valid]; neg = float(np.mean([x < 0 for x in s])) if s else None
            denom = sum(r["a_dec"] for r in valid); qsum = sum(r["a_offaxis"] for r in valid) / denom if denom > DECISION_DENOM_TOL else None
            aa = axes.get((ch, variant), [])
            if aa:
                A = np.stack(aa).astype(np.float64); resultant = float(np.linalg.norm(A.mean(0))); null = resultant_null(len(A)); gram = A @ A.T; tri = gram[np.triu_indices(len(A),1)]
                hist_c, hist_e = np.histogram(tri,bins=np.linspace(-1,1,41))
                concentration = {"n_axes": len(A), "resultant_length": resultant, "heuristic_one_over_sqrt_n": 1/math.sqrt(len(A)),
                    "pairwise_cosine": {**distribution(tri), "histogram_edges": hist_e.tolist(), "histogram_counts": hist_c.tolist()},
                    "random_axis_null": {"trials": NULL_DRAWS, "seed": NULL_SEED+len(A), "mean": float(null.mean()), "p95": float(np.quantile(null,.95)), "p99": float(np.quantile(null,.99)), "empirical_one_sided_p": float((1+np.sum(null>=resultant))/(NULL_DRAWS+1))}}
            else: concentration = {"n_axes": 0, "status": "NO_VALID_AXES"}
            out[ch][variant] = {"n_total": len(group), "n_valid": len(valid), "status_counts": dict(Counter(r["status"] for r in group)),
                "signed_distribution": distribution(s), "negative_sign_fraction": neg, "Q_sum_channel": qsum,
                "per_sample_q": distribution([r["q_offaxis"] for r in valid if r.get("q_offaxis") is not None]),
                "rank_counts": dict(Counter(str(r["observed_rank"]) for r in valid)),
                "a_contrast_null_p95_exceedance_fraction": float(np.mean([r["a_contrast_exceeds_null_p95"] for r in valid])) if valid else None,
                "a_dec_rank_null_p95_exceedance_fraction": float(np.mean([r["a_dec_exceeds_rank_null_p95"] for r in valid])) if valid else None,
                "per_mode_gradient_norms": {m: distribution([r["per_mode_gradient_norms"][m] for r in valid]) for m in MODES},
                "singular_values_by_sample": [r["singular_values"] for r in valid], "concentration": concentration}
    return out

def full_common_ordering_consistency(analysis: dict[str, Any]) -> dict[str, Any]:
    comparable = sorted(ch for ch, value in analysis.items() if value.get("full_effect", {}).get("Q_sum_channel") is not None and value.get("common_prompt", {}).get("Q_sum_channel") is not None)
    full_order = sorted(comparable, key=lambda ch: analysis[ch]["full_effect"]["Q_sum_channel"])
    common_order = sorted(comparable, key=lambda ch: analysis[ch]["common_prompt"]["Q_sum_channel"])
    concordant = discordant = ties = 0
    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            left, right = comparable[i], comparable[j]
            product = (analysis[left]["full_effect"]["Q_sum_channel"] - analysis[right]["full_effect"]["Q_sum_channel"]) * (analysis[left]["common_prompt"]["Q_sum_channel"] - analysis[right]["common_prompt"]["Q_sum_channel"])
            if abs(product) <= 1e-15:
                ties += 1
            elif product > 0:
                concordant += 1
            else:
                discordant += 1
    tau = None
    p_value = None
    if len(comparable) >= 2:
        from scipy import stats
        result = stats.kendalltau([analysis[ch]["full_effect"]["Q_sum_channel"] for ch in comparable], [analysis[ch]["common_prompt"]["Q_sum_channel"] for ch in comparable])
        tau = float(result.statistic) if np.isfinite(result.statistic) else None
        p_value = float(result.pvalue) if np.isfinite(result.pvalue) else None
    return {
        "comparable_channels": comparable,
        "full_Q_sum_order_low_to_high": full_order,
        "common_Q_sum_order_low_to_high": common_order,
        "exact_order_equal": full_order == common_order,
        "pairwise": {"concordant": concordant, "discordant": discordant, "ties": ties},
        "kendall_tau_two_sided_descriptive": tau,

        "kendall_p_two_sided_descriptive": p_value,
    }


def run_geometry(model: Any, tokenizer: Any, device: Any, rows: list[dict[str, Any]], ledger: list[dict[str, Any]], directions: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from safetensors.numpy import save_file
    data = TrainDevData(); records = []; tensors = {}; tensor_meta = []; axes: dict[tuple[str,str], list[np.ndarray]] = {}
    for c in ledger:
        if not c["support_eligible"] or not c["direction_valid"]: continue
        ch, gold, source = c["channel"], c["gold"], c["source"]
        dev_errors = [r for r in rows if r["split"] == "dev" and r["gold"] == gold and r["prediction"] == source]
        for ordinal, base in enumerate(dev_errors):
            sample = dict(data.sample(base["project_index"]))
            sample["_baseline_prediction"] = base["prediction"]
            geos, mats = compute_geometry_one(model, tokenizer, device, sample, directions[ch])
            for geo in geos:
                axis = geo.pop("contrast_axis", None)
                rec = {"channel": ch, "gold": gold, "source": source, "project_index": base["project_index"], "sample_id": base["sample_id"], **geo}
                records.append(rec)
                if axis is not None: axes.setdefault((ch,geo["variant"]),[]).append(axis)
            for variant, matrix in mats.items():
                key = f"gradient__{ch}__{variant}__{ordinal:04d}"
                tensors[key] = matrix.astype(np.float32)
                tensor_meta.append({"tensor_key": key, "channel": ch, "variant": variant, "project_index": base["project_index"], "sample_id": base["sample_id"], "shape": [4,HIDDEN_SIZE], "dtype": "float32", "mode_order": MODES})
    raw = OUT / "QWEN3_STAGE1_RAW_GRADIENTS.safetensors"
    save_file(tensors, raw, metadata={"schema_version":"1","scope":"train_dev_dev_geometry_only","contains_dataset_text":"false","mode_order":json.dumps(MODES),"dimension":str(HIDDEN_SIZE),"records_json":json.dumps(tensor_meta,separators=(",",":"))})
    analysis = summarize_geometry(records, axes)
    for c in ledger:
        if not c["support_eligible"]: continue
        full = analysis.get(c["channel"],{}).get("full_effect",{})
        c["geometry_valid"] = bool(full and full.get("n_total",0) == c["dev_error_support"] and full.get("n_valid",0) == c["dev_error_support"] and c["dev_error_support"] >= 30)
        reasons=[]
        if not c["direction_valid"]: reasons.append("INVALID_DIRECTION")
        if not c["router_eligible"]: reasons.append("ROUTER_INELIGIBLE")
        if not c["geometry_valid"]: reasons.append("INVALID_DEV_GEOMETRY_POPULATION")
        c["stage2_eligible"] = c["support_eligible"] and not reasons
        c["stage2_exclusion_reasons"] = ";".join(reasons)
    consistency = full_common_ordering_consistency(analysis)
    write_json(OUT / "QWEN3_STAGE1_GEOMETRY_ANALYSIS.json", {"schema_version":1,"scope":"DEV_ONLY_DEVELOPMENTAL","definitions":{"Q_sum_channel":"sum a_offaxis / sum a_dec","primary":"Q_sum_channel","negative_sign_fraction":"fraction signed_contrast < 0","dev_first_order_destination":"NOT_COMPUTED_NO_STAGE2_DOSE_FROZEN"},"channels":analysis,"full_common_ordering_consistency":consistency,"evaluation_accessed":False})
    fields = ["channel","gold","source","project_index","sample_id","variant","status","contrast_norm","observed_rank","signed_contrast","negative_sign","a_contrast","a_dec","a_offaxis","q_offaxis","a_contrast_exceeds_null_p95","a_dec_exceeds_rank_null_p95","singular_values","per_mode_gradient_norms"]
    with (OUT/"QWEN3_STAGE1_GEOMETRY_ROWS.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore",lineterminator="\n"); w.writeheader()
        for r in records:
            rr=dict(r); rr["singular_values"]=json.dumps(rr.get("singular_values")); rr["per_mode_gradient_norms"]=json.dumps(rr.get("per_mode_gradient_norms"),sort_keys=True); w.writerow(rr)
    return analysis, records


def write_channel_ledger(ledger: list[dict[str, Any]]) -> None:
    fields=list(ledger[0])
    with (OUT/"QWEN3_STAGE1_COMPLETE_CHANNEL_LEDGER.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(ledger)


def stage2_branch(k: int) -> tuple[str,str]:
    if k >= 5: return "A", "QWEN3_STAGE1_ELIGIBLE_K_GE_5"
    if k == 4: return "B", "QWEN3_STAGE1_ELIGIBLE_K_EQ_4"
    if k >= 1: return "C", "QWEN3_STAGE1_ELIGIBLE_K_LE_3"
    return "D", "NO_ELIGIBLE_CHANNELS"


def write_geometry_and_handoff(ledger: list[dict[str, Any]], analysis: dict[str, Any], routers: dict[str, Any], gram: dict[str, Any]) -> tuple[int,str,str]:
    eligible=[c for c in ledger if c["stage2_eligible"]]; k=len(eligible); branch,outcome=stage2_branch(k)
    support_ordered=[c for c in ledger if c["support_eligible"] and analysis.get(c["channel"],{}).get("full_effect",{}).get("Q_sum_channel") is not None]
    support_ordered=sorted(support_ordered,key=lambda c: analysis[c["channel"]]["full_effect"]["Q_sum_channel"])
    negative_ordered=sorted(support_ordered,key=lambda c: analysis[c["channel"]]["full_effect"]["negative_sign_fraction"])
    consistency=full_common_ordering_consistency(analysis)
    lines=["# Qwen3-8B Stage 1 geometry table","","Developmental train/dev result. Evaluation was not accessed.","",
           "| channel | support | Router | Stage2 | n valid/full | Q_sum/full | negative/full | resultant/full | Q_sum/common | negative/common |","|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in ledger:
        if not c["support_eligible"]: continue
        a=analysis.get(c["channel"],{}); full=a.get("full_effect",{}); common=a.get("common_prompt",{})
        def fmt(x: Any)->str: return "—" if x is None else f"{x:.6f}"
        lines.append(f"| {c['channel']} | {c['support_eligible']} | {c['router_eligible']} | {c['stage2_eligible']} | {full.get('n_valid',0)} | {fmt(full.get('Q_sum_channel'))} | {fmt(full.get('negative_sign_fraction'))} | {fmt(full.get('concentration',{}).get('resultant_length'))} | {fmt(common.get('Q_sum_channel'))} | {fmt(common.get('negative_sign_fraction'))} |")
    lines += ["",f"Full/common ordering consistency: `{stable_json(consistency)}`","Geometry never adds or removes a channel."]
    (OUT/"QWEN3_STAGE1_GEOMETRY_TABLE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    hand=["# Qwen3 Stage 2 mechanical branch handoff","",f"K = **{k}** STAGE2_ELIGIBLE channels.","",f"Frozen statistical branch: **{branch}**.","",f"Mechanical Stage 1 outcome: `{outcome}`.","","## All support-eligible channels ordered by dev Q_sum","", "| rank | channel | Q_sum | negative-sign fraction |","|---:|---|---:|---:|"]
    for n,c in enumerate(support_ordered,1):
        g=analysis[c["channel"]]["full_effect"]; hand.append(f"| {n} | {c['channel']} | {g['Q_sum_channel']:.6f} | {g['negative_sign_fraction']:.6f} |")
    hand += ["","## Support-eligible channels ordered by negative-sign fraction (low to high)",""]
    for n,c in enumerate(negative_ordered,1):
        g=analysis[c["channel"]]["full_effect"]; hand.append(f"{n}. `{c['channel']}` — {g['negative_sign_fraction']:.6f}")
    hand += ["","## Full/common Q_sum ordering consistency","",f"`{stable_json(consistency)}`"]
    hand += ["","## Complete support/Router/Stage2 eligibility","","| channel | support | direction | Router | geometry | Stage2 | exclusion |","|---|---|---|---|---|---|---|"]
    for c in ledger: hand.append(f"| {c['channel']} | {c['support_eligible']} | {c['direction_valid']} | {c['router_eligible']} | {c['geometry_valid']} | {c['stage2_eligible']} | {c['stage2_exclusion_reasons'] or c['support_exclusion_reasons'] or '—'} |")
    hand += ["","## Frozen future statistical role",""]
    if branch in ("A","B"):
        hand += ["Future primary: exact one-sided Kendall tau between lower dev Q_sum and higher evaluation destination selectivity." + (" This is LOW_POWER and requires extremely strong ordering." if branch=="B" else ""),"Secondary: Spearman; negative-sign versus wrong-to-wrong; Router detectability versus selectivity."]
    elif branch=="C":
        hand += ["Predictor-validation claim is automatically withdrawn. Future primary is real direction versus prospectively generated matched-random directions for every eligible channel. Q_sum ordering remains secondary/descriptive."]
    else: hand += ["No Stage 2 intervention is justified."]
    hand += ["","## Direction Gram matrix","",f"`{json.dumps(gram,sort_keys=True)}`","","## Claim boundary","","This can support prospective held-out-model validation only, not new-data-held-out confirmation. Evaluation rows, IDs, labels, predictions, scores, support counts, aggregates, and channel membership were not accessed. No Stage 2 runner, preregistration, freeze, preflight, or execution was created."]
    (OUT/"QWEN3_STAGE2_BRANCH_HANDOFF.md").write_text("\n".join(hand)+"\n",encoding="utf-8")
    return k,branch,outcome


def result_hashes() -> None:
    files=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "QWEN3_STAGE1_RESULT_HASHES.json": files.append(p)
    write_json(OUT/"QWEN3_STAGE1_RESULT_HASHES.json", {"schema_version":1,"utc":utc_now(),"sha256":{str(p.relative_to(ROOT)):sha256_file(p) for p in files},"script_sha256":sha256_file(ROOT/"scripts/qwen3_8b_stage0_1.py"),"head_before_result_commit":git_head()})


def run_stage1() -> None:
    if git_status(): raise RuntimeError("Stage1 requires clean committed preflight state")
    pf=read_json(PREFLIGHT)
    if pf.get("status")!="PASS" or pf.get("full_model_loaded") is not False: raise RuntimeError("valid preflight missing")
    set_determinism(); data=TrainDevData()
    started=time.perf_counter(); model,tokenizer,device,environment=load_model()
    log_event("MODEL_LOADED", model_revision=MODEL_REVISION, device=str(device))
    pilot=fixed_pilot(model,tokenizer,device,data)
    log_event("FIXED_PILOT_COMPLETE", projected_stage1_upper_bound_seconds=pilot["projected_stage1_upper_bound_seconds"])
    if pilot["projected_stage1_upper_bound_seconds"] > MAX_PROJECTED_SECONDS:
        runtime={"status":"QWEN3_STAGE1_COST_BLOCKED","pilot":pilot,"runtime_seconds":time.perf_counter()-started,"peak_gpu_memory_bytes":__import__('torch').cuda.max_memory_allocated(0)}
        write_json(OUT/"QWEN3_STAGE1_RUNTIME.json",runtime); append_retry("run-stage1","python scripts/qwen3_8b_stage0_1.py --run-stage1",0,None,[]); result_hashes(); print("QWEN3_STAGE1_COST_BLOCKED"); return
    rows=[]; acts={f"activations_L{L_OBS}": [], f"activations_L{L_INJ}": []}
    modules={f"activations_L{L_OBS}": model.model.layers[L_OBS].mlp, f"activations_L{L_INJ}": model.model.layers[L_INJ].mlp}
    baseline_started=time.perf_counter()
    for split,indices in (("train",data.train),("dev",data.dev)):
        for n,i in enumerate(indices,1):
            sample=data.sample(i); scored,captured=score_sample(model,tokenizer,device,sample,modules)
            if set(captured) != set(modules): raise RuntimeError("missing frozen-layer activation")
            for key, act in captured.items():
                if act.shape!=(HIDDEN_SIZE,) or not np.all(np.isfinite(act)): raise RuntimeError(f"invalid activation {key}")
                acts[key].append(act.astype(np.float32))
            rows.append({"schema_version":1,"project_index":i,"sample_id":sample["uuid"],"split":split,"gold":sample["correct_answer"],**scored})
            if n%100==0: print(f"baseline {split} {n}/{len(indices)}",flush=True)
    baseline_seconds=time.perf_counter()-baseline_started
    activations_by_layer={key: np.stack(value) for key, value in acts.items()}
    activations=activations_by_layer[f"activations_L{L_OBS}"]
    summary,ledger=save_baseline(rows,activations_by_layer,pilot,environment)
    log_event("TRAIN_DEV_BASELINE_COMPLETE", train_rows=len(data.train), dev_rows=len(data.dev), baseline_seconds=baseline_seconds)
    if not any(c["support_eligible"] for c in ledger):
        write_channel_ledger(ledger); runtime={"status":"NO_SUPPORT_ELIGIBLE_CHANNELS","total_seconds":time.perf_counter()-started,"baseline_seconds":baseline_seconds,"peak_gpu_memory_bytes":__import__('torch').cuda.max_memory_allocated(0)}; write_json(OUT/"QWEN3_STAGE1_RUNTIME.json",runtime); append_retry("run-stage1","python scripts/qwen3_8b_stage0_1.py --run-stage1",0,None,[]); result_hashes(); print("NO_SUPPORT_ELIGIBLE_CHANNELS"); return
    router_metrics,directions,gram=fit_directions_routers(rows,activations,ledger)
    log_event("DIRECTIONS_AND_ROUTERS_COMPLETE", support_eligible=sum(c["support_eligible"] for c in ledger))
    geometry_started=time.perf_counter(); analysis,geo_records=run_geometry(model,tokenizer,device,rows,ledger,directions); geometry_seconds=time.perf_counter()-geometry_started
    write_channel_ledger(ledger); k,branch,outcome=write_geometry_and_handoff(ledger,analysis,router_metrics,gram)
    import torch
    log_event("DEV_GEOMETRY_COMPLETE", geometry_records=len(geo_records), stage2_eligible_k=k, branch=branch, outcome=outcome)
    runtime={"status":outcome,"utc":utc_now(),"model_load_seconds":environment["model_load_seconds"],"pilot_seconds":pilot["elapsed_seconds"],"baseline_seconds":baseline_seconds,"geometry_seconds":geometry_seconds,"total_seconds":time.perf_counter()-started,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(0),"train_samples":len(data.train),"dev_samples":len(data.dev),"geometry_records":len(geo_records),"K":k,"stage2_branch":branch,"gpu_retries":0,"evaluation_accessed":False}
    write_json(OUT/"QWEN3_STAGE1_RUNTIME.json",runtime)
    append_retry("run-stage1","python scripts/qwen3_8b_stage0_1.py --run-stage1",0,None,[]); result_hashes(); print(json.dumps(runtime,indent=2,sort_keys=True)); print(outcome)


def main() -> None:
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-block0-index",action="store_true"); g.add_argument("--preflight",action="store_true"); g.add_argument("--run-stage1",action="store_true")
    args=ap.parse_args()
    try:
        if args.build_block0_index: build_block0_index()
        elif args.preflight: preflight()
        else: run_stage1()
    except Exception as exc:
        mode="build-block0-index" if args.build_block0_index else "preflight" if args.preflight else "run-stage1"
        try:
            invalidated = []
            if args.run_stage1:
                invalidated = [str((OUT/name).relative_to(ROOT)) for name in sorted(RESULT_NAMES) if (OUT/name).exists()]
                invalidated += [str(path.relative_to(ROOT)) for dirname in ("directions", "routers") for path in sorted((OUT/dirname).rglob("*")) if path.is_file()]
            append_retry(mode,f"python scripts/qwen3_8b_stage0_1.py --{mode}",1,f"{type(exc).__name__}: {exc}",invalidated)
            if args.run_stage1:
                log_event("STAGE1_FAILURE", error_type=type(exc).__name__, error=str(exc))
                write_json(OUT/"QWEN3_STAGE1_FAILURE.json",{"schema_version":1,"utc":utc_now(),"mode":mode,"error_type":type(exc).__name__,"error":str(exc),"scientific_definitions_changed":False,"evaluation_accessed":False})
        finally:
            raise

if __name__=="__main__": main()
