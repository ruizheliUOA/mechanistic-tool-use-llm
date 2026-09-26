#!/usr/bin/env python3
"""Fit frozen Qwen Stage B Routers on train and select tau on dev only.

No model is loaded. Baseline JSON records are read only by direct seek through
the committed train/dev allow-list byte index. Activations are memory-mapped
and indexed only at allowed train/dev rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/autodl-tmp/sakiko-followup")
OUT = ROOT / "final/results/qwen25_stage_b_preparation"
SOURCE = ROOT / "data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl"
INDEX = OUT / "qwen25_baseline_train_dev_byte_index.csv"
INDEX_MANIFEST = OUT / "qwen25_baseline_train_dev_byte_index.manifest.json"
SPEC = OUT / "qwen25_stage_b_router_refit_spec.json"
ROUTER_DIR = OUT / "routers"

CHANNELS = {
    "rfi_tc": {"gold": "request_for_info", "source": "tool_call", "L_obs": 20},
    "ca_tc": {"gold": "cannot_answer", "source": "tool_call", "L_obs": 20},
    "ca_direct": {"gold": "cannot_answer", "source": "direct", "L_obs": 16},
}
ARCHIVED_AUC = {
    "rfi_tc": 0.7545935845530988,
    "ca_tc": 0.9373507733831621,
    "ca_direct": 0.960428366200862,
}
TAU_GRID = (0.4, 0.5, 0.6, 0.7, 0.8)
AUC_FLOOR = 0.75
PRECISION_FLOOR = 0.50
SEED = 42
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_REPLICATES = 10_000
DIMENSION = 3584


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"nonfinite metric: {result}")
    return result


def load_allowed_rows() -> tuple[dict[int, dict[str, Any]], list[int], list[int]]:
    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    if manifest["index_sha256"] != sha256_file(INDEX):
        raise RuntimeError("byte index SHA256 mismatch")
    if manifest["test_manifest_opened"] or manifest["test_index_opened"]:
        raise RuntimeError("byte index firewall declaration failed")
    if manifest["non_allowlisted_rows_emitted"] != 0:
        raise RuntimeError("byte index contains non-allowlisted rows")

    offsets: list[tuple[int, int, int, str]] = []
    with INDEX.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "row_index",
            "byte_offset",
            "byte_length",
            "split_label",
        ]:
            raise RuntimeError("byte index header mismatch")
        for row in reader:
            split = row["split_label"]
            if split not in {"train", "dev"}:
                raise RuntimeError("forbidden split in byte index")
            offsets.append(
                (
                    int(row["row_index"]),
                    int(row["byte_offset"]),
                    int(row["byte_length"]),
                    split,
                )
            )

    rows: dict[int, dict[str, Any]] = {}
    split_rows: dict[str, list[int]] = {"train": [], "dev": []}
    with SOURCE.open("rb") as source:
        for row_index, byte_offset, byte_length, split in offsets:
            source.seek(byte_offset)
            payload = source.read(byte_length)
            if len(payload) != byte_length or b"\n" in payload or b"\r" in payload:
                raise RuntimeError(f"invalid direct-seek payload at allowed row {row_index}")
            record = json.loads(payload.decode("utf-8"))
            required = {"uuid", "gold", "pred", "correct"}
            if not required.issubset(record):
                raise RuntimeError(f"baseline schema mismatch at allowed row {row_index}")
            if row_index in rows:
                raise RuntimeError("duplicate allowed row")
            rows[row_index] = record
            split_rows[split].append(row_index)

    if len(split_rows["train"]) != 2556 or len(split_rows["dev"]) != 548:
        raise RuntimeError("train/dev direct-seek count mismatch")
    if set(split_rows["train"]) & set(split_rows["dev"]):
        raise RuntimeError("train/dev overlap")
    return rows, split_rows["train"], split_rows["dev"]


def binary_metrics(y: np.ndarray, score: np.ndarray, tau: float) -> dict[str, Any]:
    predicted = score >= tau
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    return {
        "precision": clean(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": clean(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": clean(tn / (tn + fp)) if tn + fp else 0.0,
        "routed_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def stratified_bootstrap(
    comparable_y: np.ndarray,
    comparable_score: np.ndarray,
    operational_y: np.ndarray,
    operational_score: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    comp_neg = np.flatnonzero(comparable_y == 0)
    comp_pos = np.flatnonzero(comparable_y == 1)
    op_neg = np.flatnonzero(operational_y == 0)
    op_pos = np.flatnonzero(operational_y == 1)
    values = {
        "ROC_AUC": np.empty(BOOTSTRAP_REPLICATES),
        "PR_AUC": np.empty(BOOTSTRAP_REPLICATES),
        "precision": np.empty(BOOTSTRAP_REPLICATES),
        "recall": np.empty(BOOTSTRAP_REPLICATES),
        "specificity": np.empty(BOOTSTRAP_REPLICATES),
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        ci = np.concatenate(
            [
                comp_neg[rng.integers(0, len(comp_neg), len(comp_neg))],
                comp_pos[rng.integers(0, len(comp_pos), len(comp_pos))],
            ]
        )
        oi = np.concatenate(
            [
                op_neg[rng.integers(0, len(op_neg), len(op_neg))],
                op_pos[rng.integers(0, len(op_pos), len(op_pos))],
            ]
        )
        values["ROC_AUC"][replicate] = roc_auc_score(
            comparable_y[ci], comparable_score[ci]
        )
        values["PR_AUC"][replicate] = average_precision_score(
            comparable_y[ci], comparable_score[ci]
        )
        metrics = binary_metrics(operational_y[oi], operational_score[oi], tau)
        values["precision"][replicate] = metrics["precision"]
        values["recall"][replicate] = metrics["recall"]
        values["specificity"][replicate] = metrics["specificity"]
    return {
        name: {
            "percentile_95_CI": [
                clean(np.quantile(array, 0.025)),
                clean(np.quantile(array, 0.975)),
            ]
        }
        for name, array in values.items()
    }


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["classifier"]["random_state"] != SEED:
        raise RuntimeError("spec seed mismatch")
    if spec["tau_selection"]["grid_ascending"] != list(TAU_GRID):
        raise RuntimeError("spec tau grid mismatch")
    if spec["minimum_admissible_quality"]["dev_ROC_AUC_floor"] != AUC_FLOOR:
        raise RuntimeError("spec AUC floor mismatch")

    rows, train, dev = load_allowed_rows()
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results: dict[str, Any] = {}

    for channel, channel_spec in CHANNELS.items():
        activation_path = (
            ROOT
            / f"data/processed/qwen25_7b_w2c/cache/"
            f"acts_L{channel_spec['L_obs']}.npy"
        )
        acts = np.load(activation_path, mmap_mode="r")
        if acts.shape != (3652, DIMENSION) or acts.dtype != np.float32:
            raise RuntimeError(f"activation cache mismatch for {channel}")

        train_positive = [
            index
            for index in train
            if rows[index]["gold"] == channel_spec["gold"]
            and rows[index]["pred"] == channel_spec["source"]
        ]
        train_negative = [index for index in train if bool(rows[index]["correct"])]
        train_order = train_negative + train_positive
        train_x = np.asarray(acts[train_order], dtype=np.float64)
        train_y = np.asarray(
            [0] * len(train_negative) + [1] * len(train_positive), dtype=np.int64
        )

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        classifier = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            tol=1e-4,
            random_state=SEED,
            class_weight=None,
            fit_intercept=True,
        )
        classifier.fit(train_scaled, train_y)
        if int(classifier.n_iter_[0]) >= 2000:
            raise RuntimeError(f"{channel} liblinear failed to converge")
        train_score = classifier.predict_proba(train_scaled)[:, 1]

        dev_positive = [
            index
            for index in dev
            if rows[index]["gold"] == channel_spec["gold"]
            and rows[index]["pred"] == channel_spec["source"]
        ]
        dev_correct = [index for index in dev if bool(rows[index]["correct"])]
        comparable_order = dev_correct + dev_positive
        comparable_y = np.asarray(
            [0] * len(dev_correct) + [1] * len(dev_positive), dtype=np.int64
        )
        comparable_x = np.asarray(acts[comparable_order], dtype=np.float64)
        comparable_score = classifier.predict_proba(
            scaler.transform(comparable_x)
        )[:, 1]
        auc = clean(roc_auc_score(comparable_y, comparable_score))
        pr_auc = clean(average_precision_score(comparable_y, comparable_score))

        operational_negative = [
            index
            for index in dev
            if rows[index]["pred"] == channel_spec["source"]
            and index not in set(dev_positive)
        ]
        operational_order = operational_negative + dev_positive
        operational_y = np.asarray(
            [0] * len(operational_negative) + [1] * len(dev_positive),
            dtype=np.int64,
        )
        operational_x = np.asarray(acts[operational_order], dtype=np.float64)
        operational_score = classifier.predict_proba(
            scaler.transform(operational_x)
        )[:, 1]
        grid_metrics = {
            str(tau): binary_metrics(operational_y, operational_score, tau)
            for tau in TAU_GRID
        }
        selected_tau = next(
            (
                tau
                for tau in TAU_GRID
                if grid_metrics[str(tau)]["precision"] >= PRECISION_FLOOR
            ),
            None,
        )
        selected_metrics = (
            grid_metrics[str(selected_tau)] if selected_tau is not None else None
        )
        bootstrap = (
            stratified_bootstrap(
                comparable_y,
                comparable_score,
                operational_y,
                operational_score,
                selected_tau,
                rng,
            )
            if selected_tau is not None
            else None
        )

        router_path = ROUTER_DIR / f"{channel}_router.npz"
        np.savez(
            router_path,
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            classifier_coef=np.asarray(classifier.coef_, dtype=np.float64),
            classifier_intercept=np.asarray(
                classifier.intercept_, dtype=np.float64
            ),
            classifier_classes=np.asarray(classifier.classes_, dtype=np.int64),
        )
        with np.load(router_path, allow_pickle=False) as check:
            if set(check.files) != {
                "scaler_mean",
                "scaler_scale",
                "classifier_coef",
                "classifier_intercept",
                "classifier_classes",
            }:
                raise RuntimeError(f"serialized Router fields mismatch: {channel}")

        auc_gap = abs(auc - ARCHIVED_AUC[channel])
        results[channel] = {
            "status": "CURRENT_ENVIRONMENT_TRAIN_ONLY_REFIT",
            "feature": {
                "L_obs": channel_spec["L_obs"],
                "site": "last-prompt-token MLP output",
                "activation_path": str(activation_path.relative_to(ROOT)),
                "activation_file_sha256": sha256_file(activation_path),
                "train_slice_sha256": sha256_arrays(
                    np.asarray(acts[train], dtype=np.float32)
                ),
                "dev_slice_sha256": sha256_arrays(
                    np.asarray(acts[dev], dtype=np.float32)
                ),
            },
            "train": {
                "positive_support": len(train_positive),
                "correct_negative_support": len(train_negative),
                "design_support": len(train_order),
                "row_order": "correct negatives then channel-error positives",
                "ROC_AUC_apparent": clean(roc_auc_score(train_y, train_score)),
                "PR_AUC_apparent": clean(
                    average_precision_score(train_y, train_score)
                ),
                "liblinear_iterations": int(classifier.n_iter_[0]),
                "converged": True,
            },
            "dev_comparable": {
                "positive_support": len(dev_positive),
                "correct_negative_support": len(dev_correct),
                "ROC_AUC": auc,
                "PR_AUC": pr_auc,
                "archived_ROC_AUC": ARCHIVED_AUC[channel],
                "absolute_ROC_AUC_gap": auc_gap,
            },
            "dev_operational_tau": {
                "positive_support": len(dev_positive),
                "same_source_other_negative_support": len(operational_negative),
                "grid": grid_metrics,
                "selected_tau": selected_tau,
                "selected_metrics": selected_metrics,
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed_stream": BOOTSTRAP_SEED,
                "stratification": "separate resampling of positive and negative labels",
                "metrics": bootstrap,
            },
            "serialization": {
                "path": str(router_path.relative_to(ROOT)),
                "model_file_sha256": sha256_file(router_path),
                "scaler_sha256": sha256_arrays(
                    np.asarray(scaler.mean_, dtype=np.float64),
                    np.asarray(scaler.scale_, dtype=np.float64),
                ),
                "classifier_sha256": sha256_arrays(
                    np.asarray(classifier.coef_, dtype=np.float64),
                    np.asarray(classifier.intercept_, dtype=np.float64),
                    np.asarray(classifier.classes_, dtype=np.int64),
                ),
            },
            "eligibility": {
                "AUC_floor_pass": auc >= AUC_FLOOR,
                "tau_precision_floor_pass": selected_tau is not None,
                "absolute_archived_gap_within_0_05": auc_gap <= 0.05,
            },
        }

    inadequate = [
        channel
        for channel, result in results.items()
        if not result["eligibility"]["AUC_floor_pass"]
        or not result["eligibility"]["tau_precision_floor_pass"]
    ]
    if inadequate:
        gate2 = "ROUTER_INADEQUATE"
        reason = f"required Router eligibility failed: {inadequate}"
    else:
        gate2 = "FRESH_ROUTER_PIPELINE_FEASIBLE"
        reason = (
            "All absolute floors pass. Original serialized Router weights/order are "
            "missing, so this is a current-environment prospective pipeline even when "
            "the comparable AUC gap is <=0.05."
        )

    payload = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_TRAIN_DEV_ONLY",
        "prefit_spec_commit": "8fd0b68",
        "gate2": gate2,
        "gate2_reason": reason,
        "channels": results,
        "firewall": {
            "baseline_access": "direct seek to committed train/dev byte offsets",
            "train_rows_parsed": 2556,
            "dev_rows_parsed": 548,
            "non_allowlisted_rows_parsed": 0,
            "test_manifest_opened": False,
            "test_index_opened": False,
            "test_row_id_content_label_score_or_aggregate_opened_by_this_script": False,
            "model_loaded": False,
            "GPU_used": False,
            "inference_run": False,
        },
    }
    metrics_path = OUT / "router_refit_metrics.json"
    write_json(metrics_path, payload)

    csv_path = OUT / "router_refit_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "channel",
            "train_positive",
            "train_correct_negative",
            "dev_positive",
            "dev_correct_negative",
            "dev_ROC_AUC",
            "dev_PR_AUC",
            "archived_ROC_AUC",
            "absolute_AUC_gap",
            "selected_tau",
            "precision",
            "recall",
            "specificity",
            "routed_count",
            "AUC_floor_pass",
            "tau_floor_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for channel, result in results.items():
            selected = result["dev_operational_tau"]["selected_metrics"] or {}
            writer.writerow(
                {
                    "channel": channel,
                    "train_positive": result["train"]["positive_support"],
                    "train_correct_negative": result["train"][
                        "correct_negative_support"
                    ],
                    "dev_positive": result["dev_comparable"]["positive_support"],
                    "dev_correct_negative": result["dev_comparable"][
                        "correct_negative_support"
                    ],
                    "dev_ROC_AUC": result["dev_comparable"]["ROC_AUC"],
                    "dev_PR_AUC": result["dev_comparable"]["PR_AUC"],
                    "archived_ROC_AUC": result["dev_comparable"][
                        "archived_ROC_AUC"
                    ],
                    "absolute_AUC_gap": result["dev_comparable"][
                        "absolute_ROC_AUC_gap"
                    ],
                    "selected_tau": result["dev_operational_tau"]["selected_tau"],
                    "precision": selected.get("precision"),
                    "recall": selected.get("recall"),
                    "specificity": selected.get("specificity"),
                    "routed_count": selected.get("routed_count"),
                    "AUC_floor_pass": result["eligibility"]["AUC_floor_pass"],
                    "tau_floor_pass": result["eligibility"][
                        "tau_precision_floor_pass"
                    ],
                }
            )

    table = []
    for channel, result in results.items():
        selected = result["dev_operational_tau"]["selected_metrics"]
        table.append(
            f"| {channel} | {result['train']['positive_support']} / "
            f"{result['train']['correct_negative_support']} | "
            f"{result['dev_comparable']['ROC_AUC']:.6f} | "
            f"{result['dev_comparable']['PR_AUC']:.6f} | "
            f"{result['dev_comparable']['absolute_ROC_AUC_gap']:.6f} | "
            f"{result['dev_operational_tau']['selected_tau']} | "
            f"{selected['precision']:.4f} | {selected['recall']:.4f} | "
            f"{selected['specificity']:.4f} | {selected['routed_count']} |"
            if selected is not None
            else (
                f"| {channel} | {result['train']['positive_support']} / "
                f"{result['train']['correct_negative_support']} | "
                f"{result['dev_comparable']['ROC_AUC']:.6f} | "
                f"{result['dev_comparable']['PR_AUC']:.6f} | "
                f"{result['dev_comparable']['absolute_ROC_AUC_gap']:.6f} | "
                "NONE | NA | NA | NA | NA |"
            )
        )
    report = f"""# Block 2 — Qwen train-only Router refit

**Gate 2: `{gate2}`**

{reason}

The estimator and metric populations were frozen and pushed at commit
`8fd0b68` before this fit. Baseline metadata was parsed only through direct
seeks to the committed train/dev byte index; activation arrays were indexed
only at those rows. No model or GPU was used.

| channel | train pos/neg | dev ROC-AUC | dev PR-AUC | |AUC gap| | tau | precision | recall | specificity | routed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

ROC-AUC/PR-AUC use channel errors versus correct dev rows for exact historical
comparability. Tau precision/recall/specificity use channel errors versus
other rows with the same baseline source prediction, matching the gate's
operational population. Ten-thousand stratified bootstrap intervals are in
`router_refit_metrics.json`.

The NPZ files store only numeric scaler/classifier arrays and are verified
with `allow_pickle=False`. Component and whole-file SHA256 values are included
in the JSON.
"""
    (OUT / "BLOCK2_ROUTER_REFIT_REPORT.md").write_text(report, encoding="utf-8")

    output_paths = [
        metrics_path,
        csv_path,
        OUT / "BLOCK2_ROUTER_REFIT_REPORT.md",
        *(ROUTER_DIR / f"{channel}_router.npz" for channel in CHANNELS),
    ]
    manifest = {
        "schema_version": 1,
        "stage": "BLOCK2_ROUTER_REFIT",
        "gate2": gate2,
        "prefit_spec_commit": "8fd0b68",
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths
        },
    }
    write_json(OUT / "BLOCK2_ROUTER_REFIT_MANIFEST.json", manifest)
    (OUT / "BLOCK2_ROUTER_REFIT_HASHES.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(ROOT)}" for path in output_paths
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate2": gate2,
                "channels": {
                    channel: {
                        "dev_ROC_AUC": result["dev_comparable"]["ROC_AUC"],
                        "dev_PR_AUC": result["dev_comparable"]["PR_AUC"],
                        "AUC_gap": result["dev_comparable"]["absolute_ROC_AUC_gap"],
                        "tau": result["dev_operational_tau"]["selected_tau"],
                        "selected_metrics": result["dev_operational_tau"][
                            "selected_metrics"
                        ],
                    }
                    for channel, result in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
