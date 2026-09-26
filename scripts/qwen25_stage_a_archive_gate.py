#!/usr/bin/env python3
"""CPU-only archive and configuration gate for Qwen2.5 Stage A.

This script deliberately never opens the project test index and never
materializes a project-test dataset row.  JSONL sources that contain all
project splits are handled by an allow-list parser: only line numbers present
in the frozen train/dev index files are JSON-decoded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "final" / "results" / "qwen25_stage_a_jacobian_geometry"
MODEL_DIR = ROOT / ".cache" / "modelscope" / "Qwen" / "Qwen2___5-7B-Instruct"
MODEL_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION_REF = (
    Path("/root/.cache/huggingface/hub/")
    / "models--Qwen--Qwen2.5-7B-Instruct/refs/main"
)
DATASET_REPOSITORY = "nvidia/When2Call"
DATASET_REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
DATASET_CONFIG = "test"
DATASET_SPLIT = "mcq"
SHUFFLE_SEED = 42

TRAIN_INDEX = ROOT / "final" / "results" / "splits" / "train_idx.json"
DEV_INDEX = ROOT / "final" / "results" / "splits" / "val_idx.json"
LOCKED_CONFIG = (
    ROOT / "final" / "results" / "7b_w2c_sakiko"
    / "qwen25_7b_locked_config.json"
)
VAL_SWEEP = (
    ROOT / "final" / "results" / "7b_w2c_sakiko"
    / "qwen25_7b_val_sweep.json"
)
HISTORICAL_IMPLEMENTATION = ROOT / "scripts" / "qwen7b_native_sakiko.py"
HISTORICAL_RUNNER = ROOT / "scripts" / "qwen7b_native_sakiko_run.py"
BASELINE_IMPLEMENTATION = ROOT / "scripts" / "eval_w2c_7b_baseline.py"

DIRECTION_OID = (
    "6d765083b1bb9d9127b8c03946b4ed2c7a402b51213c748fe37fae024ed6b34c"
)
BASELINE_OID = (
    "e66a2faff706e44aff1370df79663aaf9257c3699f10b9b3d3d3731fb5903a79"
)
SPLIT_BASELINE_OID = (
    "d6d7c409e388d0debf6d7c60653e3149999429116e5d48fb14c777b507eca6b0"
)

CHANNELS = {
    "rfi_tc": {
        "gold": "request_for_info",
        "source": "tool_call",
        "obs_layer": 20,
        "inj_layer": 18,
        "rho": 2.0,
        "threshold": 0.7,
    },
    "ca_tc": {
        "gold": "cannot_answer",
        "source": "tool_call",
        "obs_layer": 20,
        "inj_layer": 16,
        "rho": 4.0,
        "threshold": 0.8,
    },
    "ca_direct": {
        "gold": "cannot_answer",
        "source": "direct",
        "obs_layer": 16,
        "inj_layer": 16,
        "rho": 6.0,
        "threshold": 0.4,
    },
}

MODES = ["tool_call", "direct", "request_for_info", "cannot_answer"]
NORM_TOLERANCE = 1e-5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def locate_lfs_object(oid: str) -> Path:
    path = ROOT / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not path.is_file():
        raise FileNotFoundError(f"missing LFS object {oid}: {path}")
    return path


def load_indices() -> tuple[list[int], list[int]]:
    train = list(map(int, json.loads(TRAIN_INDEX.read_text())))
    dev = list(map(int, json.loads(DEV_INDEX.read_text())))
    if len(train) != len(set(train)) or len(dev) != len(set(dev)):
        raise ValueError("duplicate split indices")
    if set(train) & set(dev):
        raise ValueError("train/dev indices overlap")
    return train, dev


def load_allowed_jsonl(
    path: Path, allowed: set[int]
) -> tuple[dict[int, dict[str, Any]], str]:
    """Decode only allow-listed lines; excluded lines remain opaque bytes."""
    selected: dict[int, dict[str, Any]] = {}
    slice_hash = hashlib.sha256()
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            if index not in allowed:
                continue
            slice_hash.update(index.to_bytes(8, "big"))
            slice_hash.update(line)
            selected[index] = json.loads(line)
    missing = allowed - set(selected)
    if missing:
        raise ValueError(f"allow-listed rows missing from {path}: {len(missing)}")
    return selected, slice_hash.hexdigest()


def csv_write(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_directions() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = locate_lfs_object(DIRECTION_OID)
    payload = path.read_bytes()
    file_hash = sha256_bytes(payload)
    if file_hash != DIRECTION_OID:
        raise ValueError(f"direction object hash mismatch: {file_hash}")
    archive = np.load(io.BytesIO(payload), allow_pickle=False)
    expected_names = {
        "rfi_tc_unit", "ca_tc_unit", "ca_direct_unit",
        "rfi_tc_meta", "ca_tc_meta", "ca_direct_meta",
    }
    if set(archive.files) != expected_names:
        raise ValueError(f"direction names mismatch: {archive.files}")

    vectors: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for channel in CHANNELS:
        vector = archive[f"{channel}_unit"]
        if vector.shape != (3584,):
            raise ValueError(f"{channel} shape {vector.shape}")
        if vector.dtype != np.float32:
            raise ValueError(f"{channel} dtype {vector.dtype}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{channel} contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise ValueError(f"{channel} norm {norm}")
        meta = archive[f"{channel}_meta"]
        vectors[channel] = vector.copy()
        details[channel] = {
            "shape": list(vector.shape),
            "dtype": str(vector.dtype),
            "finite": True,
            "norm": norm,
            "tensor_sha256": sha256_bytes(vector.tobytes(order="C")),
            "meta": {
                "obs_layer": int(meta[0]),
                "median_norm": float(meta[1]),
                "tensor_sha256": sha256_bytes(meta.tobytes(order="C")),
            },
        }
    cosines: dict[str, float] = {}
    names = list(CHANNELS)
    for ia, left in enumerate(names):
        for right in names[ia + 1:]:
            u = vectors[left].astype(np.float64)
            v = vectors[right].astype(np.float64)
            cosines[f"{left}|{right}"] = float(
                np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            )
    return {
        "lfs_oid_sha256": DIRECTION_OID,
        "materialized_path": str(path.relative_to(ROOT)),
        "size_bytes": len(payload),
        "verified_file_sha256": file_hash,
        "norm_tolerance": NORM_TOLERANCE,
        "vectors": details,
        "signed_cosines": cosines,
    }, vectors


def tokenizer_provenance() -> dict[str, Any]:
    config_path = MODEL_DIR / "tokenizer_config.json"
    tokenizer_json = MODEL_DIR / "tokenizer.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    template = config.get("chat_template")
    if not isinstance(template, str):
        raise ValueError("tokenizer chat_template is not a string")
    revision = MODEL_REVISION_REF.read_text().strip()
    return {
        "official_repository": MODEL_REPOSITORY,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "local_model_path": str(MODEL_DIR.relative_to(ROOT)),
        "config_sha256": sha256_file(MODEL_DIR / "config.json"),
        "tokenizer_config_sha256": sha256_bytes(config_bytes),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "chat_template_sha256_utf8": sha256_bytes(template.encode("utf-8")),
        "chat_template_utf8_bytes": len(template.encode("utf-8")),
        "tokenizer_class": config.get("tokenizer_class"),
        "bos_token": config.get("bos_token"),
        "eos_token": config.get("eos_token"),
        "pad_token": config.get("pad_token"),
        "model_max_length": config.get("model_max_length"),
        "source": str(MODEL_REVISION_REF),
    }


def build_configuration(
    train: list[int],
    dev: list[int],
    baseline_slice_hash: str,
    split_baseline_slice_hash: str,
    directions: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = tokenizer_provenance()
    locked = json.loads(LOCKED_CONFIG.read_text())["locked"]
    for channel, expected in CHANNELS.items():
        actual = locked[channel]
        for key, archive_key in (
            ("obs_layer", "obs_layer"),
            ("inj_layer", "inj_layer"),
            ("rho", "alpha"),
            ("threshold", "threshold"),
        ):
            if actual[archive_key] != expected[key]:
                raise ValueError(f"locked config mismatch {channel}.{key}")

    fields = [
        {
            "field": "a1 official model repository",
            "status": "RECOVERED",
            "value": MODEL_REPOSITORY,
            "source": "local HF cache namespace and historical model path",
        },
        {
            "field": "a2 exact immutable model revision",
            "status": "RECOVERED",
            "value": tokenizer["model_revision"],
            "source": str(MODEL_REVISION_REF),
        },
        {
            "field": "a3 exact tokenizer revision",
            "status": "RECOVERED",
            "value": tokenizer["tokenizer_revision"],
            "source": str(MODEL_REVISION_REF),
        },
        {
            "field": "a4 tokenizer config/chat template",
            "status": "RECOVERED",
            "value": (
                f"tokenizer_config={tokenizer['tokenizer_config_sha256']}; "
                f"chat_template={tokenizer['chat_template_sha256_utf8']}"
            ),
            "source": ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct/tokenizer_config.json",
        },
        {
            "field": "a5 exact train/dev split definitions",
            "status": "RECOVERED",
            "value": (
                f"shuffle_seed=42; train_n={len(train)}; dev_n={len(dev)}; "
                f"train_sha256={sha256_file(TRAIN_INDEX)}; "
                f"dev_sha256={sha256_file(DEV_INDEX)}"
            ),
            "source": "final/results/splits/{train_idx,val_idx}.json",
        },
        {
            "field": "a6 independent train/dev access path",
            "status": "RECOVERED",
            "value": (
                f"{DATASET_REPOSITORY}@{DATASET_REVISION}; "
                "offline cache -> seed-42 shuffle -> allow-listed train/dev select"
            ),
            "source": (
                "/root/.cache/huggingface/datasets/nvidia___when2_call/"
                f"test/0.0.0/{DATASET_REVISION}/dataset_info.json"
            ),
        },
        {
            "field": "a7 error population definition",
            "status": "RECOVERED",
            "value": (
                "CURRENT_ENVIRONMENT dev rows with un-intervened sequence-score "
                "argmax equal to channel source and gold equal to channel gold"
            ),
            "source": "frozen Stage A definition; historical source lines 51-62, 377-405",
        },
        {
            "field": "a8 archived sample IDs and ordering",
            "status": "PARTIAL",
            "value": (
                "all baseline train/dev IDs and seed-42 row order recovered; "
                "exact archived routed-dev IDs/order absent"
            ),
            "source": (
                f"LFS {SPLIT_BASELINE_OID}; safe train/dev slice "
                f"{split_baseline_slice_hash}"
            ),
        },
        {
            "field": "B exact scoring function",
            "status": "RECOVERED",
            "value": (
                "per-candidate teacher-forced mean token log-probability; "
                "prompt excluded; add_special_tokens=False; no padding/truncation; "
                "candidate-specific graph; logits[prompt_len-1:prompt_len+cand_len-1]"
            ),
            "source": "scripts/eval_w2c_7b_baseline.py:62-88,127-185; qwen7b_native_sakiko.py:377-405",
        },
        {
            "field": "C layers/tensor location",
            "status": "RECOVERED",
            "value": (
                "Qwen2DecoderLayer.model.layers[L_inj].mlp forward output, "
                "after MLP internals and before decoder-layer residual addition; "
                "rfi L18, ca_tc L16, ca_direct L16"
            ),
            "source": "scripts/qwen7b_native_sakiko.py:377-390; locked_config.json:2-29",
        },
        {
            "field": "D injection positions",
            "status": "RECOVERED",
            "value": (
                "all sequence positions in each concatenated prompt+candidate "
                "teacher-forced forward; active during the complete scoring forward"
            ),
            "source": "scripts/qwen7b_native_sakiko.py:377-396 (out[:, :, :] += ct)",
        },
        {
            "field": "E intervention scaling",
            "status": "RECOVERED",
            "value": (
                "binary-gated delta_h[i,p] = rho[channel] * median_norm[channel] "
                "* recovered_unit_direction[channel], constant at every position"
            ),
            "source": "scripts/qwen7b_native_sakiko_run.py:58-88,98-129; direction NPZ meta",
        },
        {
            "field": "F Router weights",
            "status": "MISSING_ARCHIVED_OBJECT",
            "value": (
                "recipe/features/obs layer/threshold/binary predicate recovered; "
                "original pickle weights and exact routed-dev order absent"
            ),
            "source": "scripts/qwen7b_native_sakiko.py:236-285,327-334; local artifact audit",
        },
        {
            "field": "G archived records",
            "status": "PARTIAL",
            "value": (
                "baseline train/dev predictions and four scores recovered; dev "
                "post-intervention four scores/destinations and dev mismatched "
                "per-sample controls absent"
            ),
            "source": (
                f"LFS {BASELINE_OID} safe slice {baseline_slice_hash}; "
                f"LFS {SPLIT_BASELINE_OID} safe slice {split_baseline_slice_hash}; "
                "historical runner writes per-sample intervention only for test"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "label": "CURRENT_ENVIRONMENT_GEOMETRY_ONLY",
        "gate_a_outcome": "GEOMETRY_ONLY_FEASIBLE",
        "reason": (
            "All mandatory geometry inputs are recovered. FULL_REPLAY_FEASIBLE "
            "fails because original Router weights, exact routed-dev order, and "
            "dev per-sample post-intervention records are absent."
        ),
        "model_tokenizer": tokenizer,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "upstream_config_named_test": DATASET_CONFIG,
            "upstream_split": DATASET_SPLIT,
            "project_boundary_note": (
                "The upstream dataset configuration is named 'test'; the project "
                "split is independently defined after seed-42 shuffle. Only "
                "allow-listed project train/dev rows are materialized."
            ),
            "shuffle_seed": SHUFFLE_SEED,
            "train_count": len(train),
            "dev_count": len(dev),
            "train_index_sha256": sha256_file(TRAIN_INDEX),
            "dev_index_sha256": sha256_file(DEV_INDEX),
            "sealed_extractor": (
                "load cached upstream dataset; shuffle(seed=42); access only "
                "indices in train_idx.json or val_idx.json"
            ),
        },
        "directions": directions,
        "channels": {
            channel: {
                **config,
                "median_norm": directions["vectors"][channel]["meta"]["median_norm"],
                "per_position_scaling_when_router_active": (
                    config["rho"]
                    * directions["vectors"][channel]["meta"]["median_norm"]
                ),
                "router_feature": (
                    "last-prompt-token MLP output at obs_layer, float32 cached"
                ),
                "router_recipe": (
                    "StandardScaler + LogisticRegression(C=1, solver=liblinear, "
                    "max_iter=2000, random_state=42); train positives=channel "
                    "errors, negatives=all correct train rows"
                ),
                "router_gate": (
                    f"baseline_pred == {config['source']} and "
                    f"predict_proba[:,1] >= {config['threshold']}"
                ),
            }
            for channel, config in CHANNELS.items()
        },
        "scoring": {
            "mode_order": MODES,
            "prompt": (
                "Qwen chat template applied to a system message containing tool "
                "JSON and a user question, add_generation_prompt=True"
            ),
            "candidate_source": "sample.answers[mode], distinct per sample and mode",
            "teacher_forced": True,
            "add_special_tokens": False,
            "bos_added": False,
            "eos_added": False,
            "padding": "none (batch size 1)",
            "truncation": "none; historical prompt skip threshold 8192 tokens",
            "masking": "no loss on prompt; no attention mask supplied at batch size 1",
            "length_normalization": "arithmetic mean over candidate token log-probabilities",
            "scoring_positions": "prompt_len-1 through prompt_len+candidate_len-2",
            "calibration": "none",
        },
        "fields": fields,
        "hard_boundary": {
            "project_test_index_opened": False,
            "project_test_rows_materialized": False,
            "project_test_labels_scores_or_aggregates_read": False,
        },
    }


def write_gate_a0(
    out: Path,
    dev: list[int],
    archived: dict[int, dict[str, Any]],
    directions: dict[str, Any],
) -> dict[str, Any]:
    four_score_rows = sum(
        1
        for index in dev
        if set(archived[index].get("avg_logp", {})) == set(MODES)
    )
    runner_fields = [
        "channel", "sample_id", "project_index", "baseline_source",
        "baseline_runner_up", "destination", "destination_matches_runner_up",
        "runner_up_is_gold", "source_runner_up_margin",
        "gold_best_other_wrong_margin", "reached_gold", "analysis_status",
    ]
    csv_write(out / "gate_a0_runner_up_samples.csv", runner_fields, [])
    runner_result = {
        "required_label": "POST_HOC_RUNNER_UP_CONSISTENT_DESTINATION_ANALYSIS",
        "status": "STRUCTURED_SKIP",
        "dev_rows_with_archived_four_mode_baseline_scores": four_score_rows,
        "dev_rows_with_archived_deployed_destinations": 0,
        "reason": (
            "Archived dev baseline four-mode scores exist, but no dev per-sample "
            "deployed intervention destinations/scores exist. The historical "
            "runner wrote per-sample intervention records only for the prohibited "
            "project test split. A moved-sample set and permutation null therefore "
            "cannot be formed without new GPU reproduction."
        ),
        "permutation_null": {
            "status": "NOT_RUN_NO_MOVED_SAMPLE_SET",
            "seed": 20260728,
            "permutations_requested": 10000,
        },
        "channels": {
            channel: {
                "routed_source_population": None,
                "moved_count": None,
                "gold_arrivals": None,
                "wrong_to_wrong_moves": None,
                "runner_up_destination_match_count": None,
                "runner_up_destination_match_rate": None,
                "status": "STRUCTURED_SKIP_NO_DEV_DESTINATIONS",
            }
            for channel in CHANNELS
        },
    }
    (out / "gate_a0_runner_up_analysis.json").write_text(
        json.dumps(runner_result, indent=2) + "\n"
    )
    (out / "GATE_A0_RUNNER_UP_ANALYSIS.md").write_text(
        "# Gate A0.1 — Runner-up-consistent destination flow\n\n"
        "**Required label:** "
        "`POST_HOC_RUNNER_UP_CONSISTENT_DESTINATION_ANALYSIS`\n\n"
        "**Status:** `STRUCTURED_SKIP`\n\n"
        f"The archive contains full four-mode baseline scores for all {four_score_rows} "
        "allowed dev rows, but it does not contain dev per-sample deployed "
        "intervention scores or destinations. Historical source inspection shows "
        "that the only per-sample intervention writer targets the sealed project "
        "test split; those records were not opened. Therefore routed/moved sets, "
        "destination-flow rates, and the 10,000-permutation null are undefined. "
        "The CSV contains the frozen schema and zero fabricated rows.\n\n"
        "This is an evidence-preserving structured skip, not a negative result.\n"
    )

    mismatch_fields = [
        "real_channel", "mismatched_direction", "signed_cosine",
        "absolute_cosine", "geometry_class", "dose_rho",
        "per_position_scale", "router_condition", "sample_population",
        "source_exits", "gold_arrivals", "wrong_to_wrong_transitions",
        "target_gain", "target_hit", "utility", "collateral_damage",
        "archived_specificity_verdict", "analysis_status",
    ]
    mismatch_rows: list[dict[str, Any]] = []
    for real_channel, config in CHANNELS.items():
        for mismatch in CHANNELS:
            if real_channel == mismatch:
                continue
            key = (
                f"{real_channel}|{mismatch}"
                if f"{real_channel}|{mismatch}" in directions["signed_cosines"]
                else f"{mismatch}|{real_channel}"
            )
            cosine = directions["signed_cosines"][key]
            mismatch_rows.append({
                "real_channel": real_channel,
                "mismatched_direction": mismatch,
                "signed_cosine": f"{cosine:.12g}",
                "absolute_cosine": f"{abs(cosine):.12g}",
                "geometry_class": (
                    "non-orthogonal mismatched control"
                    if abs(cosine) >= 0.1
                    else "near-orthogonal mismatched control"
                ),
                "dose_rho": config["rho"],
                "per_position_scale": (
                    config["rho"]
                    * directions["vectors"][real_channel]["meta"]["median_norm"]
                ),
                "router_condition": (
                    f"source={config['source']}; p>={config['threshold']}"
                ),
                "sample_population": "DEV_RECORD_ABSENT",
                "source_exits": "",
                "gold_arrivals": "",
                "wrong_to_wrong_transitions": "",
                "target_gain": "",
                "target_hit": "",
                "utility": "",
                "collateral_damage": "",
                "archived_specificity_verdict": "",
                "analysis_status": "STRUCTURED_SKIP_NO_DEV_MISMATCHED_RECORD",
            })
    csv_write(
        out / "gate_a0_mismatched_control_geometry.csv",
        mismatch_fields,
        mismatch_rows,
    )
    mismatch_result = {
        "status": "GEOMETRY_RECOMPUTED_BEHAVIOR_STRUCTURED_SKIP",
        "signed_cosine_gram": directions["signed_cosines"],
        "behavior_reason": (
            "No archived dev mismatched-direction per-sample behavior exists. "
            "Test-only control artifacts and aggregates were not opened."
        ),
        "rows": mismatch_rows,
    }
    (out / "gate_a0_mismatched_control_geometry.json").write_text(
        json.dumps(mismatch_result, indent=2) + "\n"
    )
    gram_lines = "\n".join(
        f"| {key.replace('|', ' ↔ ')} | {value:.9f} | {abs(value):.9f} | "
        f"{'non-orthogonal mismatched control' if abs(value) >= 0.1 else 'near-orthogonal mismatched control'} |"
        for key, value in directions["signed_cosines"].items()
    )
    (out / "GATE_A0_MISMATCHED_CONTROL_GEOMETRY.md").write_text(
        "# Gate A0.2 — Mismatched-control × direction geometry\n\n"
        "The signed Gram matrix was recomputed from the verified numerical LFS "
        "vectors without renormalizing or changing signs.\n\n"
        "| pair | signed cosine | absolute cosine | terminology |\n"
        "|---|---:|---:|---|\n"
        f"{gram_lines}\n\n"
        "**Behavior cross-reference status:** "
        "`STRUCTURED_SKIP_NO_DEV_MISMATCHED_RECORD`.\n\n"
        "No archived dev mismatched-direction per-sample behavior was found. "
        "The historical controls were implemented on the sealed project test "
        "split, so their artifacts and aggregates were deliberately not opened. "
        "The six ordered CSV rows expose the geometry and frozen historical dose/"
        "Router conditions, with all unavailable behavioral fields left empty. "
        "No transfer law or causal cosine claim is made.\n"
    )
    return {"runner_up": runner_result, "mismatched": mismatch_result}


def write_configuration_report(out: Path, config: dict[str, Any]) -> None:
    (out / "gate_a_configuration_recovery.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    table = "\n".join(
        f"| {row['field']} | {row['status']} | {row['value']} | {row['source']} |"
        for row in config["fields"]
    )
    channel_rows = "\n".join(
        f"| {channel} | L{data['obs_layer']} | L{data['inj_layer']} | "
        f"{data['rho']} | {data['median_norm']:.9f} | "
        f"{data['per_position_scaling_when_router_active']:.9f} | "
        f"{data['threshold']} |"
        for channel, data in config["channels"].items()
    )
    tok = config["model_tokenizer"]
    (out / "GATE_A_CONFIGURATION_RECOVERY.md").write_text(
        "# Gate A — Configuration recovery\n\n"
        "**Outcome:** `GEOMETRY_ONLY_FEASIBLE`\n\n"
        "**Downstream label:** `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`\n\n"
        "All inputs required for current-environment sequence-score Jacobian "
        "geometry are recovered. Full archived replay is not feasible because "
        "the original serialized Router weights, exact routed-dev ID order, and "
        "dev per-sample post-intervention records are absent.\n\n"
        "## Recovery table\n\n"
        "| field | status | recovered value | source |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        "## Model and tokenizer pin\n\n"
        f"- Official repository: `{tok['official_repository']}`\n"
        f"- Model revision: `{tok['model_revision']}`\n"
        f"- Tokenizer revision: `{tok['tokenizer_revision']}`\n"
        f"- Config SHA256: `{tok['config_sha256']}`\n"
        f"- Tokenizer config SHA256: `{tok['tokenizer_config_sha256']}`\n"
        f"- Tokenizer JSON SHA256: `{tok['tokenizer_json_sha256']}`\n"
        f"- Chat-template UTF-8 SHA256: `{tok['chat_template_sha256_utf8']}`\n\n"
        "## Frozen channel mechanics\n\n"
        "| channel | L_obs | L_inj | rho | median norm | active per-position scale | tau |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{channel_rows}\n\n"
        "The intervention is applied to the MLP module output at every position "
        "of each concatenated prompt+candidate teacher-forced forward. It is "
        "therefore before the decoder-layer residual addition and includes both "
        "prompt and candidate positions. Each answer mode has its own candidate "
        "sequence and its own forward graph.\n\n"
        "## Data firewall\n\n"
        "The independent access path uses the cached upstream dataset revision, "
        "the historical seed-42 shuffle, and only the explicit train/dev index "
        "allow lists. The project test index, project test rows, labels, scores, "
        "and aggregates were not opened.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    train, dev = load_indices()
    allowed = set(train) | set(dev)
    direction_info, _ = verify_directions()
    baseline, baseline_slice_hash = load_allowed_jsonl(
        locate_lfs_object(BASELINE_OID), allowed
    )
    split_baseline, split_baseline_slice_hash = load_allowed_jsonl(
        locate_lfs_object(SPLIT_BASELINE_OID), allowed
    )
    if any(
        baseline[index]["uuid"] != split_baseline[index]["uuid"]
        for index in allowed
    ):
        raise ValueError("baseline sources disagree on allowed UUIDs")
    if any(
        split_baseline[index]["split"]
        != ("train" if index in set(train) else "val")
        for index in allowed
    ):
        raise ValueError("archived split labels disagree with allow lists")

    gate_a0 = write_gate_a0(out, dev, split_baseline, direction_info)
    config = build_configuration(
        train,
        dev,
        baseline_slice_hash,
        split_baseline_slice_hash,
        direction_info,
    )
    write_configuration_report(out, config)
    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "gate_a0": {
            "runner_up_status": gate_a0["runner_up"]["status"],
            "mismatched_status": gate_a0["mismatched"]["status"],
        },
        "gate_a_outcome": config["gate_a_outcome"],
        "downstream_label": config["label"],
        "safe_archive_slices": {
            "baseline_lfs_oid": BASELINE_OID,
            "baseline_allowed_slice_sha256": baseline_slice_hash,
            "split_baseline_lfs_oid": SPLIT_BASELINE_OID,
            "split_baseline_allowed_slice_sha256": split_baseline_slice_hash,
            "allowed_rows": len(allowed),
            "train_rows": len(train),
            "dev_rows": len(dev),
        },
        "test_access": config["hard_boundary"],
    }
    (out / "gate_a_archive_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps({
        "output_dir": str(out),
        "gate_a_outcome": config["gate_a_outcome"],
        "direction_sha256": direction_info["verified_file_sha256"],
        "signed_cosines": direction_info["signed_cosines"],
        "train_rows": len(train),
        "dev_rows": len(dev),
        "test_rows_materialized": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
