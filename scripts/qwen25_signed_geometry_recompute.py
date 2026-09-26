#!/usr/bin/env python3
"""Dev-only signed Qwen2.5 Jacobian geometry recomputation.

Modes:
  prepare  CPU-only: freeze the Stage A sample IDs/order and input hashes.
  timing   GPU: fixed two-sample engineering timing check; no science outputs.
  full     GPU: recompute raw gradients, signed geometry and descriptive reports.

The program never opens the project test index. Baseline metadata are direct-seeked
through the committed train/dev byte allow-list; dataset samples are indexed only
after membership in the committed dev set is checked. Model/tokenizer loading is
local-only and Hugging Face/Transformers offline modes are forced.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from datasets import load_dataset
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen25_stage_a_geometry as A  # noqa: E402
import qwen25_stage_b_block01 as B  # noqa: E402
import qwen25_stage_b_router_refit as R  # noqa: E402

OUT = ROOT / "final/results/qwen25_branch_determination"
STAGE_A = ROOT / "final/results/qwen25_stage_a_jacobian_geometry"
STAGE_B_PREP = ROOT / "final/results/qwen25_stage_b_preparation"
RAW_A = STAGE_A / "jacobian_geometry.jsonl"
PILOT = STAGE_A / "gate_b_engineering_pilot.json"
POPULATION = OUT / "signed_geometry_population_manifest.json"
TIMING = OUT / "signed_geometry_timing.json"
RAW_GRADIENTS = OUT / "raw_gradients.safetensors"
SAMPLES_CSV = OUT / "qwen25_signed_geometry_samples.csv"
ANALYSIS_JSON = OUT / "signed_geometry_analysis.json"
RETRY_LEDGER = OUT / "signed_geometry_retry_ledger.json"
MANIFEST = OUT / "BRANCH_DETERMINATION_MANIFEST.json"

CHANNELS = ("rfi_tc", "ca_tc", "ca_direct")
GEOMETRIES = ("full_effect", "common_prompt")
MODES = tuple(A.MODES)
DIMENSION = A.DIMENSION
MODEL_REVISION = A.MODEL_REVISION
DATASET_REVISION = A.DATASET_REVISION
NEAR_ZERO = 1e-8
DENOMINATOR_EPSILON = 1e-8
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 10_000
CONCENTRATION_SEED = A.NULL_SEED
CONCENTRATION_TRIALS = 10_000
COST_LIMIT_SECONDS = 3600.0
COST_SAFETY_FACTOR = 1.20
EXPECTED_COUNTS = {"rfi_tc": 92, "ca_tc": 77, "ca_direct": 60}
EXPECTED_COMMON_VALID = {"rfi_tc": 60, "ca_tc": 46, "ca_direct": 32}
DIRECTION_TENSOR_SHA256 = {
    "rfi_tc": "678adedaeef8eb9975dfc72b78ae8e7636ab8aaa7372c92f1a4637238b66a0a8",
    "ca_tc": "4e99fe68d257c2223b7b226898a93a0b75407f60d5fa98bdd51eaf54304b4f93",
    "ca_direct": "0c0c1e58670aeaf644686c6c77f6d970f63ce928603a667c51f88c2f6c1455e4",
}
MODEL_IDENTITY_SHA256 = {
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def clean(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"non-finite output {result}")
    return result


def distribution(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "n": 0, "mean": None, "std": None, "min": None, "p25": None,
            "median": None, "p75": None, "max": None,
        }
    return {
        "n": int(len(x)),
        "mean": clean(np.mean(x)),
        "std": clean(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": clean(np.min(x)),
        "p25": clean(np.quantile(x, 0.25)),
        "median": clean(np.median(x)),
        "p75": clean(np.quantile(x, 0.75)),
        "max": clean(np.max(x)),
    }


def fractions(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "positive": clean(np.mean(x > NEAR_ZERO)),
        "negative": clean(np.mean(x < -NEAR_ZERO)),
        "near_zero": clean(np.mean(np.abs(x) <= NEAR_ZERO)),
        "near_zero_tolerance": NEAR_ZERO,
    }


def verify_model_identity() -> dict[str, Any]:
    stage_a_manifest = json.loads((STAGE_A / "STAGE_A_MANIFEST.json").read_text())
    frozen = stage_a_manifest["model"]
    if (
        frozen["official_repository"] != "Qwen/Qwen2.5-7B-Instruct"
        or frozen["model_revision"] != MODEL_REVISION
        or frozen["tokenizer_revision"] != MODEL_REVISION
    ):
        raise RuntimeError("Stage A model/revision identity changed")
    observed = {
        name: sha256_file(A.MODEL_DIR / name) for name in MODEL_IDENTITY_SHA256
    }
    if observed != MODEL_IDENTITY_SHA256:
        raise RuntimeError(f"local model/tokenizer identity mismatch: {observed}")
    tokenizer_config = json.loads((A.MODEL_DIR / "tokenizer_config.json").read_text())
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str):
        raise RuntimeError("local tokenizer chat template missing")
    chat_hash = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if chat_hash != CHAT_TEMPLATE_SHA256:
        raise RuntimeError(f"chat-template identity mismatch: {chat_hash}")
    ref = Path(
        "/root/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-7B-Instruct/refs/main"
    )
    if not ref.is_file() or ref.read_text().strip() != MODEL_REVISION:
        raise RuntimeError("local official model revision ref mismatch")
    return {
        "official_repository": frozen["official_repository"],
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "local_file_sha256": observed,
        "chat_template_sha256_utf8": chat_hash,
        "official_local_ref": ref.read_text().strip(),
    }


def load_stage_a_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in RAW_A.read_text().splitlines() if line]
    if len(rows) != 458:
        raise RuntimeError(f"Stage A geometry row count changed: {len(rows)}")
    return rows


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model_identity = verify_model_identity()
    rows = load_stage_a_rows()
    by_key = {(r["channel"], int(r["project_index"]), r["geometry_type"]): r for r in rows}
    full = [r for r in rows if r["geometry_type"] == "full_effect"]
    counts = Counter(r["channel"] for r in full)
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"Stage A population count mismatch: {counts}")
    entries = []
    for order, row in enumerate(full):
        key = (row["channel"], int(row["project_index"]), "common_prompt")
        common = by_key[key]
        entries.append({
            "order": order,
            "channel": row["channel"],
            "project_index": int(row["project_index"]),
            "sample_id": row["sample_id"],
            "gold": row["gold"],
            "source": row["source"],
            "router_active": bool(row["router_active"]),
            "actual_per_position_scaling": float(row["actual_per_position_scaling"]),
            "geometry": {
                "full_effect": {
                    "status": row["geometry_status"],
                    "a_contrast": row["a_contrast"],
                    "a_dec": row["a_dec"],
                    "a_offaxis": row["a_offaxis"],
                },
                "common_prompt": {
                    "status": common["geometry_status"],
                    "a_contrast": common["a_contrast"],
                    "a_dec": common["a_dec"],
                    "a_offaxis": common["a_offaxis"],
                },
            },
        })
    common_counts = Counter(
        e["channel"] for e in entries
        if e["geometry"]["common_prompt"]["status"] == "VALID"
    )
    if dict(common_counts) != EXPECTED_COMMON_VALID:
        raise RuntimeError(f"Stage A common-valid mismatch: {common_counts}")
    timing_orders = []
    for channel in ("rfi_tc", "ca_direct"):
        timing_orders.append(next(
            e["order"] for e in entries
            if e["channel"] == channel and e["router_active"]
        ))
    directions = A.load_directions()
    direction_hashes = {ch: sha256_array(directions[ch]) for ch in CHANNELS}
    if direction_hashes != DIRECTION_TENSOR_SHA256:
        raise RuntimeError(f"direction hash mismatch: {direction_hashes}")
    input_paths = [
        RAW_A,
        PILOT,
        STAGE_A / "gate_a_configuration_recovery.json",
        STAGE_A / "linearity.json",
        STAGE_B_PREP / "block1_geometry_verification.json",
        STAGE_B_PREP / "router_refit_metrics.json",
        STAGE_B_PREP / "qwen25_baseline_train_dev_byte_index.csv",
        STAGE_B_PREP / "qwen25_baseline_train_dev_byte_index.manifest.json",
        ROOT / "final/results/splits/val_idx.json",
    ]
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "scope": "CURRENT_ENVIRONMENT_DEV_ONLY_STAGE_A_POPULATION",
        "model_repository": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "model_identity": model_identity,
        "dataset_repository": "nvidia/When2Call",
        "dataset_revision": DATASET_REVISION,
        "mode_order": list(MODES),
        "dimension": DIMENSION,
        "direction_lfs_oid_sha256": A.DIRECTION_OID,
        "direction_tensor_sha256": direction_hashes,
        "stage_a_geometry_sha256": sha256_file(RAW_A),
        "sample_order_sha256": canonical_sha([
            [e["channel"], e["project_index"], e["sample_id"]] for e in entries
        ]),
        "channel_counts": dict(counts),
        "common_valid_counts": dict(common_counts),
        "router_active_counts": dict(Counter(
            e["channel"] for e in entries if e["router_active"]
        )),
        "timing_sample_orders": timing_orders,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths
        },
        "entries": entries,
        "test_firewall": {
            "project_test_manifest_or_index_opened": False,
            "project_test_row_id_label_prediction_score_or_aggregate_opened": False,
        },
    }
    write_json(POPULATION, payload)
    print(f"prepared {POPULATION} with {len(entries)} dev samples")


class DevData:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        rows, _train, dev = R.load_allowed_rows()
        self.rows = rows
        self.dev = frozenset(dev)
        requested = {int(entry["project_index"]) for entry in entries}
        if not requested <= self.dev:
            raise RuntimeError("population contains a non-dev project index")
        self.dataset = load_dataset(
            "nvidia/When2Call", "test", split="mcq"
        ).shuffle(seed=A.SEED)
        if len(self.dataset) != 3652:
            raise RuntimeError("dataset length mismatch")

    def sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        index = int(entry["project_index"])
        if index not in self.dev:
            raise PermissionError(f"non-dev sample access denied: {index}")
        sample = self.dataset[index]
        if sample["uuid"] != entry["sample_id"] or self.rows[index]["uuid"] != entry["sample_id"]:
            raise RuntimeError(f"dev identity mismatch at {index}")
        if sample["correct_answer"] != entry["gold"]:
            raise RuntimeError(f"dev gold mismatch at {index}")
        return sample


def load_population() -> dict[str, Any]:
    manifest = json.loads(POPULATION.read_text())
    if manifest["stage_a_geometry_sha256"] != sha256_file(RAW_A):
        raise RuntimeError("Stage A source changed after population freeze")
    entries = manifest["entries"]
    if [e["order"] for e in entries] != list(range(len(entries))):
        raise RuntimeError("non-contiguous frozen sample order")
    if manifest["sample_order_sha256"] != canonical_sha([
        [e["channel"], e["project_index"], e["sample_id"]] for e in entries
    ]):
        raise RuntimeError("population order hash mismatch")
    return manifest


def load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    population = load_population()
    pilot = json.loads(PILOT.read_text())
    if pilot["gate_b_outcome"] != "PASS":
        raise RuntimeError("Stage A engineering pilot did not pass")
    return population, pilot


def run_timing() -> None:
    population, _pilot = load_frozen()
    if verify_model_identity() != population["model_identity"]:
        raise RuntimeError("model identity changed after population freeze")
    entries = population["entries"]
    chosen = [entries[i] for i in population["timing_sample_orders"]]
    data = DevData(chosen)
    directions = A.load_directions()
    model, tokenizer, device, environment = A.load_model()
    jacobian_seconds, intervention_seconds = [], []
    started = time.perf_counter()
    try:
        for entry in chosen:
            sample = data.sample(entry)
            jac = A.jacobian_sample(
                model, tokenizer, device, sample,
                A.CHANNELS[entry["channel"]]["inj_layer"],
                capture_prompt_states=True,
            )
            jacobian_seconds.append(float(jac["runtime_seconds"]))
            scale = float(entry["actual_per_position_scaling"])
            if scale <= 0:
                raise RuntimeError("timing sample unexpectedly Router-inactive")
            intervention = A.score_sample(
                model, tokenizer, device, sample,
                inj_layer=A.CHANNELS[entry["channel"]]["inj_layer"],
                correction=(scale * directions[entry["channel"]]).astype(np.float32),
            )
            intervention_seconds.append(float(intervention["runtime_seconds"]))
            del jac, intervention
            gc.collect()
            __import__("torch").cuda.empty_cache()
        active = sum(e["router_active"] for e in entries)
        projected_raw = (
            np.mean(jacobian_seconds) * len(entries)
            + np.mean(intervention_seconds) * active
        )
        projected = float(projected_raw * COST_SAFETY_FACTOR)
        result = {
            "schema_version": 1,
            "completed_utc": utc_now(),
            "script_commit": git("rev-parse", "HEAD"),
            "command": "python scripts/qwen25_signed_geometry_recompute.py --mode timing",
            "fixed_sample_orders": population["timing_sample_orders"],
            "fixed_samples": [
                {
                    "order": e["order"], "channel": e["channel"],
                    "project_index": e["project_index"], "sample_id": e["sample_id"],
                } for e in chosen
            ],
            "jacobian_seconds": jacobian_seconds,
            "intervention_seconds": intervention_seconds,
            "projected_sample_count": len(entries),
            "projected_router_active_count": active,
            "safety_factor": COST_SAFETY_FACTOR,
            "projected_full_gpu_seconds": projected,
            "cost_limit_seconds": COST_LIMIT_SECONDS,
            "status": (
                "PASS" if projected <= COST_LIMIT_SECONDS
                else "SIGNED_GEOMETRY_COST_BLOCKED"
            ),
            "timing_wall_seconds_excluding_model_load": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(__import__("torch").cuda.max_memory_allocated(0)),
            "environment": environment,
            "test_access": False,
        }
        write_json(TIMING, result)
        print(json.dumps({
            "status": result["status"],
            "projected_full_gpu_seconds": projected,
            "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        }, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(42)
    finally:
        del model
        gc.collect()
        __import__("torch").cuda.empty_cache()


def geometry_for(
    gradients: dict[str, np.ndarray],
    direction: np.ndarray,
    entry: dict[str, Any],
    geometry: str,
    prompt_deviation: float,
    tolerances: dict[str, float],
    prompt_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    metrics, axis = A.geometry_metrics(
        gradients, direction, entry["gold"], entry["source"],
        tolerances["contrast_norm_tolerance"],
        tolerances["offaxis_numerical_tolerance"],
    )
    if geometry == "common_prompt" and prompt_deviation > prompt_tolerance:
        metrics.update({
            "status": "COMMON_PROMPT_STATE_NOT_SHARED",
            "a_contrast": None, "a_dec": None, "a_offaxis": None, "a_orth": None,
        })
        axis = None
    if axis is not None:
        contrast = (
            gradients[entry["gold"]].astype(np.float64)
            - gradients[entry["source"]].astype(np.float64)
        )
        raw_signed = clean(np.dot(direction.astype(np.float64), contrast))
        signed = clean(np.dot(direction.astype(np.float64), axis))
    else:
        raw_signed = signed = None
    metrics["signed_contrast"] = signed
    metrics["raw_signed_contrast"] = raw_signed
    return metrics, axis


def metric_pair(predicted: list[float], actual: list[float]) -> dict[str, Any]:
    return A.metric_pair(
        np.asarray(predicted, dtype=np.float64),
        np.asarray(actual, dtype=np.float64),
    )


def flow_table(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    table = {src: {dst: 0 for dst in MODES} for src in MODES}
    for row in rows:
        table[row["source"]][row[field]] += 1
    return table


def summarize_signed(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for channel in CHANNELS:
        out[channel] = {}
        for geometry in GEOMETRIES:
            group = [
                r for r in records
                if r["channel"] == channel and r["geometry_variant"] == geometry
                and r["geometry_status"] == "VALID"
            ]
            s = [r["signed_contrast"] for r in group]
            raw = [r["raw_signed_contrast"] for r in group]
            out[channel][geometry] = {
                "signed_unit_contrast": distribution(s),
                "signed_unit_fractions": fractions(s),
                "raw_dot_contrast": distribution(raw),
                "raw_dot_fractions": fractions(raw),
            }
    return out


def summarize_offaxis(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    null = B.null_summary(3)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["geometry_status"] == "VALID":
            grouped[(row["channel"], row["geometry_variant"])].append(row)
    metrics: dict[str, Any] = {}
    for channel in CHANNELS:
        metrics[channel] = {}
        for geometry in GEOMETRIES:
            group = grouped[(channel, geometry)]
            a_dec = np.asarray([r["a_dec"] for r in group], dtype=np.float64)
            a_off = np.asarray([r["a_offaxis"] for r in group], dtype=np.float64)
            qmask = a_dec > DENOMINATOR_EPSILON
            smask = a_dec > null["p95"]
            metrics[channel][geometry] = {
                "n_valid": len(group),
                "q_unconditional": distribution(a_off[qmask] / a_dec[qmask]),
                "q_signal_conditioned": distribution(a_off[smask] / a_dec[smask]),
                "n_above_rank_null_p95": int(np.sum(smask)),
                "Q_channel": clean(np.sum(a_off) / np.sum(a_dec)),
            }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap: dict[str, Any] = {}
    for geometry in GEOMETRIES:
        rfi = grouped[("rfi_tc", geometry)]
        bootstrap[geometry] = {
            "rfi_tc_vs_ca_tc": B.bootstrap_difference(
                rfi, [grouped[("ca_tc", geometry)]], rng
            ),
            "rfi_tc_vs_ca_direct": B.bootstrap_difference(
                rfi, [grouped[("ca_direct", geometry)]], rng
            ),
            "rfi_tc_vs_pooled_comparisons": B.bootstrap_difference(
                rfi,
                [grouped[("ca_tc", geometry)], grouped[("ca_direct", geometry)]],
                rng,
            ),
        }
    archived = json.loads(
        (STAGE_B_PREP / "block1_geometry_verification.json").read_text()
    )["bootstrap"]["results"]["full_effect"]["rfi_tc_vs_pooled_comparisons"][
        "Q_difference_percentile_95_CI"
    ]
    reproduced = bootstrap["full_effect"]["rfi_tc_vs_pooled_comparisons"][
        "Q_difference_percentile_95_CI"
    ]
    bootstrap["exact_reproduction_check"] = {
        "target": archived,
        "recomputed": reproduced,
        "allclose_rtol_1e-12_atol_1e-12": bool(np.allclose(
            archived, reproduced, rtol=1e-12, atol=1e-12
        )),
    }
    return {"rank_3_random_axis_null": null, "groups": metrics}, bootstrap


def summarize_margin(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for channel in CHANNELS:
        result[channel] = {}
        channel_rows = [r for r in samples if r["channel"] == channel]
        for name, rows in (
            ("all_channel_errors", channel_rows),
            ("router_active", [r for r in channel_rows if r["router_active"]]),
        ):
            linear = [r["linear_margin_change"] for r in rows]
            actual = [r["actual_margin_change"] for r in rows]
            result[channel][name] = {
                "n": len(rows),
                "linear_distribution": distribution(linear),
                "linear_fractions": fractions(linear),
                "association_with_actual_margin_change": metric_pair(linear, actual),
            }
    return result


def summarize_destinations(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for channel in CHANNELS:
        rows = [r for r in samples if r["channel"] == channel]
        moved = [r for r in rows if r["linear_predicted_destination"] != r["source"]]
        output[channel] = {
            "n": len(rows),
            "predicted_moved_count": len(moved),
            "predicted_gold_arrivals": sum(
                r["linear_predicted_destination"] == r["gold"] for r in rows
            ),
            "predicted_wrong_to_wrong_moves": sum(
                r["linear_predicted_destination"] not in {r["source"], r["gold"]}
                for r in rows
            ),
            "predicted_destination_correct_rate_among_predicted_moved": (
                clean(np.mean([
                    r["linear_predicted_destination"] == r["gold"] for r in moved
                ])) if moved else None
            ),
            "predicted_overall_gold_arrival_rate": clean(np.mean([
                r["linear_predicted_destination"] == r["gold"] for r in rows
            ])),
            "predicted_flow": flow_table(rows, "linear_predicted_destination"),
            "favored_mode_vs_runner_up": cross_table(
                rows, "first_order_favored_non_source", "baseline_non_source_runner_up"
            ),
            "favored_mode_vs_actual_destination": cross_table(
                rows, "first_order_favored_non_source", "actual_destination"
            ),
            "runner_up_vs_actual_destination": cross_table(
                rows, "baseline_non_source_runner_up", "actual_destination"
            ),
        }
    return output


def cross_table(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    table = {a: {b: 0 for b in MODES} for a in MODES}
    for row in rows:
        table[row[left]][row[right]] += 1
    return table


def summarize_concentration(axes: dict[tuple[str, str], list[np.ndarray]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for channel in CHANNELS:
        for geometry in GEOMETRIES:
            values = axes[(channel, geometry)]
            matrix = np.stack(values)
            n = len(matrix)
            resultant = clean(np.linalg.norm(matrix.mean(axis=0)))
            gram = matrix @ matrix.T
            pairwise = gram[np.triu_indices(n, k=1)]
            counts, edges = np.histogram(pairwise, bins=np.linspace(-1, 1, 41))
            null = A.resultant_null(n, trials=CONCENTRATION_TRIALS, seed=CONCENTRATION_SEED)
            groups[f"{channel}|{geometry}"] = {
                "n_axes": n,
                "resultant_length": resultant,
                "heuristic_one_over_sqrt_n": clean(1 / math.sqrt(n)),
                "pairwise_cosine": {
                    **distribution(pairwise),
                    "histogram_edges": edges.tolist(),
                    "histogram_counts": counts.tolist(),
                },
                "random_axis_null": {
                    "dimension": DIMENSION,
                    "trials": CONCENTRATION_TRIALS,
                    "seed": CONCENTRATION_SEED,
                    "mean": clean(np.mean(null)),
                    "p95": clean(np.quantile(null, 0.95)),
                    "p99": clean(np.quantile(null, 0.99)),
                    "empirical_one_sided_p": clean(
                        (1 + np.sum(null >= resultant)) / (len(null) + 1)
                    ),
                },
            }
    return {"method": "Stage A exact rotational Markov null", "groups": groups}


def consistency(records: list[dict[str, Any]], entries: list[dict[str, Any]]) -> dict[str, Any]:
    outside, absdev, reldev, status_mismatch = 0, [], [], []
    for row in records:
        entry = entries[int(row["order"])]
        archived = entry["geometry"][row["geometry_variant"]]
        if row["geometry_status"] != archived["status"]:
            status_mismatch.append({
                "order": row["order"], "geometry": row["geometry_variant"],
                "archived": archived["status"], "recomputed": row["geometry_status"],
            })
        if archived["status"] == "VALID":
            observed = float(row["signed_contrast"]) ** 2
            target = float(archived["a_contrast"])
            absdev.append(abs(observed - target))
            reldev.append(abs(observed - target) / max(abs(target), np.finfo(float).tiny))
            if not np.isclose(observed, target, rtol=1e-4, atol=1e-7):
                outside += 1
    result = {
        "population_count": len(entries),
        "sample_order_sha256": canonical_sha([
            [e["channel"], e["project_index"], e["sample_id"]] for e in entries
        ]),
        "validity_status_mismatch_count": len(status_mismatch),
        "validity_status_mismatches": status_mismatch,
        "s_squared_vs_stage_a_a_contrast": {
            "rtol": 1e-4, "atol": 1e-7,
            "n_compared": len(absdev),
            "maximum_absolute_deviation": max(absdev, default=0.0),
            "maximum_relative_deviation": max(reldev, default=0.0),
            "count_outside_tolerance": outside,
        },
        "status": (
            "PASS" if not outside and not status_mismatch
            else "SIGNED_RECOMPUTE_INCONSISTENT"
        ),
    }
    return result


def write_samples_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "order", "channel", "project_index", "sample_id", "geometry_variant",
        "geometry_status", "gold", "source", "router_active",
        "actual_per_position_scaling", "signed_contrast", "raw_signed_contrast",
        "a_contrast", "a_dec", "a_offaxis", "q", "linear_margin_change",
        "actual_margin_change", "baseline_non_source_runner_up",
        "first_order_favored_non_source", "linear_predicted_destination",
        "actual_destination", "gradient_tensor_key",
    ]
    with SAMPLES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})


def render_reports(analysis: dict[str, Any]) -> None:
    signed = analysis["signed_contrast"]
    consistency_result = analysis["stage_a_consistency"]
    lines = [
        "# Signed Qwen2.5 Jacobian geometry results", "",
        f"Status: `{consistency_result['status']}`. Scope: current-environment dev only.",
        "", "## Signed unit contrast", "",
        "| channel | variant | n | mean s | median s | positive | negative | near-zero |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ch in CHANNELS:
        for geo in GEOMETRIES:
            x = signed[ch][geo]
            d, f = x["signed_unit_contrast"], x["signed_unit_fractions"]
            lines.append(
                f"| {ch} | {geo} | {d['n']} | {d['mean']:.6g} | {d['median']:.6g} "
                f"| {f['positive']:.3f} | {f['negative']:.3f} | {f['near_zero']:.3f} |"
            )
    lines += [
        "", "## Stage A consistency", "",
        f"- maximum absolute deviation: "
        f"{consistency_result['s_squared_vs_stage_a_a_contrast']['maximum_absolute_deviation']:.6g}",
        f"- maximum relative deviation: "
        f"{consistency_result['s_squared_vs_stage_a_a_contrast']['maximum_relative_deviation']:.6g}",
        f"- outside tolerance: "
        f"{consistency_result['s_squared_vs_stage_a_a_contrast']['count_outside_tolerance']}",
        f"- validity-status mismatches: {consistency_result['validity_status_mismatch_count']}",
        "", "The deployed positive sign is `+rho * d_hat`: the historical hook adds the "
        "positive correction vector to the MLP output. Positive signed contrast therefore "
        "locally increases gold relative to the baseline source mode.",
        "", "Absolute first-order destination rates are descriptive, not confirmatory. "
        "Stage A operating-dose linearity was weak and channel-dependent: approximately "
        "0.288 (rfi_tc), 0.020 (ca_tc), and -0.335 (ca_direct).",
    ]
    (OUT / "SIGNED_GEOMETRY_RESULTS.md").write_text("\n".join(lines) + "\n")

    off = analysis["off_axis"]["groups"]
    conc = analysis["concentration"]["groups"]
    lines = [
        "# Off-axis and concentration analysis", "",
        "| channel | variant | Q_channel | median q | signal n | resultant | null p95 | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ch in CHANNELS:
        for geo in GEOMETRIES:
            o = off[ch][geo]
            c = conc[f"{ch}|{geo}"]
            lines.append(
                f"| {ch} | {geo} | {o['Q_channel']:.6f} | "
                f"{o['q_unconditional']['median']:.6f} | {o['n_above_rank_null_p95']} | "
                f"{c['resultant_length']:.6f} | {c['random_axis_null']['p95']:.6f} | "
                f"{c['random_axis_null']['empirical_one_sided_p']:.6g} |"
            )
    boot = analysis["bootstrap"]
    lines += [
        "", f"Bootstrap: {BOOTSTRAP_REPLICATES:,} stratified resamples, seed "
        f"`{BOOTSTRAP_SEED}`. The archived full-effect pooled Q CI reproduction "
        f"check is `{boot['exact_reproduction_check']['allclose_rtol_1e-12_atol_1e-12']}`.",
        "", "Pairwise-cosine histogram edges/counts and all three requested channel "
        "contrasts are stored in `signed_geometry_analysis.json`.",
    ]
    (OUT / "OFF_AXIS_AND_CONCENTRATION_ANALYSIS.md").write_text("\n".join(lines) + "\n")

    det = analysis["detectability_geometry"]
    lines = [
        "# Detectability–geometry dissociation", "",
        "| channel | dev ROC-AUC | dev PR-AUC | signed-positive | Q_channel | "
        "signal-conditioned off-axis | concentration | historical label |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in det:
        lines.append(
            f"| {row['channel']} | {row['current_router_ROC_AUC']:.6f} | "
            f"{row['current_router_PR_AUC']:.6f} | {row['signed_positive_fraction']:.3f} | "
            f"{row['Q_channel']:.6f} | {row['signal_conditioned_offaxis_share']:.6f} | "
            f"{row['concentration']:.6f} | {row['historical_behavior_label']} |"
        )
    lines += [
        "", "Across the three available Qwen2.5 channels, state detectability and "
        "direction-to-target geometric alignment exhibit different orderings.",
        "", "This is descriptive. It is not a formal anti-correlation, a universal "
        "Readable ≠ Correctable law, or a validated predictor claim.",
    ]
    (OUT / "DETECTABILITY_GEOMETRY_DISSOCIATION.md").write_text("\n".join(lines) + "\n")


def detectability_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    router = json.loads((STAGE_B_PREP / "router_refit_metrics.json").read_text())
    labels = {
        "rfi_tc": "historically unstable/weakest correction channel",
        "ca_tc": "historically stronger correction channel",
        "ca_direct": "historically stronger but destination-sensitive channel",
    }
    output = []
    for ch in CHANNELS:
        r = router["channels"][ch]["dev_comparable"]
        output.append({
            "channel": ch,
            "current_router_ROC_AUC": r["ROC_AUC"],
            "current_router_PR_AUC": r["PR_AUC"],
            "signed_positive_fraction": analysis["signed_contrast"][ch]["full_effect"][
                "signed_unit_fractions"
            ]["positive"],
            "Q_channel": analysis["off_axis"]["groups"][ch]["full_effect"]["Q_channel"],
            "signal_conditioned_offaxis_share": analysis["off_axis"]["groups"][ch][
                "full_effect"
            ]["q_signal_conditioned"]["mean"],
            "concentration": analysis["concentration"]["groups"][
                f"{ch}|full_effect"
            ]["resultant_length"],
            "historical_behavior_label": labels[ch],
        })
    return output


def write_decision() -> None:
    payload = {
        "schema_version": 1,
        "block1_outcome": "UNTOUCHED_POPULATION_PROVENANCE_UNRESOLVED",
        "block2_outcome": "TEST_SELECTION_CONTAMINATED",
        "block3_outcome": "SIGNED_GEOMETRY_RECOMPUTE_CONSISTENT",
        "decision_rule": "BRANCH_D",
        "next_action": "START_QWEN3_STAGE0_1",
        "execute_next_action_in_this_task": False,
        "rationale": (
            "No independent compatible population is established, and the historical "
            "Qwen evaluation result explicitly motivated same-population rfi configuration "
            "follow-ups. Dev-only geometry remains descriptive."
        ),
    }
    write_json(OUT / "branch_decision.json", payload)
    (OUT / "BRANCH_DECISION.md").write_text(
        "# Branch decision\n\n"
        "`START_QWEN3_STAGE0_1`\n\n"
        "Block 1 is `UNTOUCHED_POPULATION_PROVENANCE_UNRESOLVED`; Block 2 is "
        "`TEST_SELECTION_CONTAMINATED`; Block 3 is a consistent dev-only descriptive "
        "recomputation. Branch D therefore applies. This task does not start Qwen3.\n"
    )


def finalize_manifest(
    population: dict[str, Any], environment: dict[str, Any],
    runtime: dict[str, Any], consistency_result: dict[str, Any],
) -> None:
    outputs = [
        OUT / "UNTOUCHED_POPULATION_SEARCH.md",
        OUT / "dataset_config_split_inventory.csv",
        OUT / "PRIOR_LOOK_AUDIT.md",
        OUT / "prior_test_look_ledger.csv",
        OUT / "EXPOSURE_LEDGER.md",
        OUT / "SIGNED_GEOMETRY_RECOMPUTE_SPEC.md",
        POPULATION, TIMING, RAW_GRADIENTS, SAMPLES_CSV, ANALYSIS_JSON,
        OUT / "SIGNED_GEOMETRY_RESULTS.md",
        OUT / "OFF_AXIS_AND_CONCENTRATION_ANALYSIS.md",
        OUT / "DETECTABILITY_GEOMETRY_DISSOCIATION.md",
        OUT / "BRANCH_DECISION.md",
        OUT / "branch_decision.json",
        RETRY_LEDGER,
        ROOT / "scripts/qwen25_signed_geometry_recompute.py",
    ]
    manifest = {
        "schema_version": 1,
        "completed_utc": utc_now(),
        "starting_head": "8f4c98a589330de92d46514710b1c1eda88d5169",
        "run_script_commit": git("rev-parse", "HEAD"),
        "model_repository": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "direction_lfs_oid_sha256": A.DIRECTION_OID,
        "direction_tensor_sha256": population["direction_tensor_sha256"],
        "rng_seeds": {
            "model_and_dataset": A.SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "concentration": CONCENTRATION_SEED,
        },
        "input_sha256": population["input_sha256"],
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in outputs if path.is_file()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": environment["torch_version"],
            "transformers": environment["transformers_version"],
            "datasets": environment["datasets_version"],
            "safetensors": __import__("safetensors").__version__,
        },
        "runtime": runtime,
        "stage_a_consistency": consistency_result,
        "retry_ledger_path": str(RETRY_LEDGER.relative_to(ROOT)),
        "constraints": {
            "test_rows_ids_labels_predictions_scores_metrics_aggregates_accessed": False,
            "stage_b_prepared_or_run": False,
            "qwen3_started": False,
            "llama_phase10_3_work": False,
            "direction_layer_dose_router_retuned": False,
            "network_model_or_tokenizer_download": False,
        },
    }
    write_json(MANIFEST, manifest)


def run_full() -> None:
    population, pilot = load_frozen()
    if verify_model_identity() != population["model_identity"]:
        raise RuntimeError("model identity changed after population freeze")
    timing = json.loads(TIMING.read_text())
    if timing["status"] != "PASS" or timing["projected_full_gpu_seconds"] > COST_LIMIT_SECONDS:
        raise RuntimeError("timing gate did not pass")
    entries = population["entries"]
    data = DevData(entries)
    directions = A.load_directions()
    model, tokenizer, device, environment = A.load_model()
    tolerances = pilot["frozen_geometry_tolerances"]
    prompt_tolerance = pilot["well_posedness"]["frozen_prompt_state_absolute_tolerance"]
    tensors: dict[str, np.ndarray] = {}
    tensor_metadata = []
    records: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    axes: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    started = time.perf_counter()
    try:
        for ordinal, entry in enumerate(entries, start=1):
            channel = entry["channel"]
            direction = directions[channel]
            sample = data.sample(entry)
            jac = A.jacobian_sample(
                model, tokenizer, device, sample,
                A.CHANNELS[channel]["inj_layer"], capture_prompt_states=True,
            )
            base_scores = {mode: float(jac["scores"][mode]) for mode in MODES}
            baseline_prediction = A.top_prediction(base_scores)[0]
            if baseline_prediction != entry["source"]:
                raise RuntimeError(
                    f"baseline source mismatch order {entry['order']}: {baseline_prediction}"
                )
            scale = float(entry["actual_per_position_scaling"])
            if scale > 0:
                intervened = A.score_sample(
                    model, tokenizer, device, sample,
                    inj_layer=A.CHANNELS[channel]["inj_layer"],
                    correction=(scale * direction).astype(np.float32),
                )
                actual_scores = intervened["scores"]
                actual_destination = intervened["prediction"]
                intervention_seconds = intervened["runtime_seconds"]
            else:
                actual_scores = dict(base_scores)
                actual_destination = baseline_prediction
                intervention_seconds = 0.0
            gold, source = entry["gold"], entry["source"]
            actual_margin = (
                actual_scores[gold] - actual_scores[source]
                - (base_scores[gold] - base_scores[source])
            )
            full_delta = {
                mode: clean(scale * np.dot(
                    jac["g_full"][mode].astype(np.float64), direction.astype(np.float64)
                )) for mode in MODES
            }
            predicted_scores = {
                mode: base_scores[mode] + full_delta[mode] for mode in MODES
            }
            predicted_destination = A.top_prediction(predicted_scores)[0]
            non_source = [mode for mode in MODES if mode != source]
            favored = max(non_source, key=lambda mode: (full_delta[mode], -MODES.index(mode)))
            runner_up = max(
                non_source, key=lambda mode: (base_scores[mode], -MODES.index(mode))
            )
            linear_margin = full_delta[gold] - full_delta[source]
            common_by_geometry: dict[str, dict[str, Any]] = {}
            for geometry, gradients in (
                ("full_effect", jac["g_full"]),
                ("common_prompt", jac["g_prompt"]),
            ):
                metrics, axis = geometry_for(
                    gradients, direction, entry, geometry,
                    float(jac["max_prompt_hidden_deviation"]),
                    tolerances, prompt_tolerance,
                )
                key = None
                if metrics["status"] == "VALID":
                    key = f"gradient__{geometry}__{channel}__{entry['order']:04d}"
                    tensor = np.stack([gradients[mode] for mode in MODES]).astype(np.float32)
                    if tensor.shape != (4, DIMENSION) or not np.isfinite(tensor).all():
                        raise RuntimeError(f"invalid gradient tensor {key}")
                    tensors[key] = tensor
                    tensor_metadata.append({
                        "tensor_key": key,
                        "sample_id": entry["sample_id"],
                        "project_index": entry["project_index"],
                        "channel": channel,
                        "variant": geometry,
                        "mode_order": list(MODES),
                        "shape": [4, DIMENSION],
                        "dtype": "float32",
                    })
                    axes[(channel, geometry)].append(axis)
                q = (
                    metrics["a_offaxis"] / metrics["a_dec"]
                    if metrics["a_dec"] is not None
                    and metrics["a_dec"] > DENOMINATOR_EPSILON else None
                )
                row = {
                    "order": entry["order"],
                    "channel": channel,
                    "project_index": entry["project_index"],
                    "sample_id": entry["sample_id"],
                    "geometry_variant": geometry,
                    "geometry_status": metrics["status"],
                    "gold": gold,
                    "source": source,
                    "router_active": bool(entry["router_active"]),
                    "actual_per_position_scaling": scale,
                    "signed_contrast": metrics["signed_contrast"],
                    "raw_signed_contrast": metrics["raw_signed_contrast"],
                    "a_contrast": metrics["a_contrast"],
                    "a_dec": metrics["a_dec"],
                    "a_offaxis": metrics["a_offaxis"],
                    "q": q,
                    "linear_margin_change": linear_margin,
                    "actual_margin_change": actual_margin,
                    "baseline_non_source_runner_up": runner_up,
                    "first_order_favored_non_source": favored,
                    "linear_predicted_destination": predicted_destination,
                    "actual_destination": actual_destination,
                    "gradient_tensor_key": key,
                    "observed_rank": metrics["observed_rank"],
                    "contrast_norm": metrics["contrast_norm"],
                    "jacobian_runtime_seconds": jac["runtime_seconds"],
                    "intervention_runtime_seconds": intervention_seconds,
                    "max_prompt_hidden_deviation": jac["max_prompt_hidden_deviation"],
                }
                records.append(row)
                common_by_geometry[geometry] = row
            sample_rows.append(common_by_geometry["full_effect"])
            print(f"signed geometry {ordinal}/{len(entries)} {channel}", flush=True)
            del jac
            if "intervened" in locals():
                del intervened
            gc.collect()
            __import__("torch").cuda.empty_cache()

        consistency_result = consistency(records, entries)
        if consistency_result["status"] != "PASS":
            write_json(OUT / "SIGNED_RECOMPUTE_INCONSISTENCY.json", consistency_result)
            raise RuntimeError("SIGNED_RECOMPUTE_INCONSISTENT")
        save_file(
            tensors,
            str(RAW_GRADIENTS),
            metadata={
                "schema_version": "1",
                "mode_order": json.dumps(MODES),
                "dimension": str(DIMENSION),
                "records_json": json.dumps(
                    tensor_metadata, sort_keys=True, separators=(",", ":")
                ),
                "contains_dataset_text": "false",
                "scope": "current_environment_dev_only",
            },
        )
        write_samples_csv(records)
        signed = summarize_signed(records)
        offaxis, bootstrap = summarize_offaxis(records)
        concentration = summarize_concentration(axes)
        analysis = {
            "schema_version": 1,
            "completed_utc": utc_now(),
            "scope": "CURRENT_ENVIRONMENT_DEV_ONLY_DESCRIPTIVE",
            "sign_convention": {
                "deployed_sign": "+rho*d_hat added to MLP output",
                "semantic_status": "RESOLVED",
                "positive": "locally increases gold relative to source",
                "negative": "locally increases source relative to gold",
            },
            "stage_a_consistency": consistency_result,
            "signed_contrast": signed,
            "off_axis": offaxis,
            "bootstrap": bootstrap,
            "margin": summarize_margin(sample_rows),
            "destination": summarize_destinations(sample_rows),
            "concentration": concentration,
            "linearity_caveat": {
                "rfi_tc_stage_a_pearson_approx": 0.288,
                "ca_tc_stage_a_pearson_approx": 0.020,
                "ca_direct_stage_a_pearson_approx": -0.335,
                "interpretation": "absolute first-order destinations are descriptive only",
            },
            "raw_gradient_artifact": {
                "path": str(RAW_GRADIENTS.relative_to(ROOT)),
                "tensor_count": len(tensors),
                "record_count": len(tensor_metadata),
                "mode_order": list(MODES),
                "dimension": DIMENSION,
            },
            "test_access": False,
        }
        analysis["detectability_geometry"] = detectability_table(analysis)
        write_json(ANALYSIS_JSON, analysis)
        render_reports(analysis)
        write_decision()
        runtime = {
            "full_wall_seconds_excluding_model_load": time.perf_counter() - started,
            "model_load_seconds": environment["model_load_seconds"],
            "peak_gpu_memory_bytes": int(__import__("torch").cuda.max_memory_allocated(0)),
            "sample_count": len(entries),
            "valid_gradient_tensor_count": len(tensors),
        }
        finalize_manifest(population, environment, runtime, consistency_result)
        print(json.dumps({
            "status": "PASS",
            "runtime": runtime,
            "raw_gradient_sha256": sha256_file(RAW_GRADIENTS),
            "decision": "START_QWEN3_STAGE0_1",
        }, indent=2))
    finally:
        del model
        gc.collect()
        __import__("torch").cuda.empty_cache()


def append_retry(mode: str, command: str, exit_code: int, reason: str | None) -> None:
    ledger = json.loads(RETRY_LEDGER.read_text()) if RETRY_LEDGER.exists() else {
        "schema_version": 1, "invocations": []
    }
    ledger["invocations"].append({
        "utc": utc_now(),
        "mode": mode,
        "script_commit": git("rev-parse", "HEAD"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "command": command,
        "exit_code": exit_code,
        "reason": reason,
        "outputs_invalidated": (
            [] if exit_code == 0 else [
                str(path.relative_to(ROOT)) for path in
                [RAW_GRADIENTS, SAMPLES_CSV, ANALYSIS_JSON, MANIFEST]
            ]
        ),
    })
    write_json(RETRY_LEDGER, ledger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prepare", "timing", "full"))
    args = parser.parse_args()
    command = f"python scripts/qwen25_signed_geometry_recompute.py --mode {args.mode}"
    try:
        {"prepare": prepare, "timing": run_timing, "full": run_full}[args.mode]()
    except SystemExit as exc:
        code = int(exc.code or 0)
        append_retry(args.mode, command, code, "cost gate" if code == 42 else None)
        raise
    except Exception as exc:
        append_retry(args.mode, command, 1, f"{type(exc).__name__}: {exc}")
        raise
    else:
        append_retry(args.mode, command, 0, None)


if __name__ == "__main__":
    main()
