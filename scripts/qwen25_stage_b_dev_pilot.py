#!/usr/bin/env python3
"""Run the frozen train/dev-only Qwen Stage B engineering pilot.

This script uses the exact eight UUID-selected dev rows, the committed
train-only Router refits, and only locally cached model/tokenizer weights. It
does not read a project test manifest/index or execute Stage B.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import qwen25_stage_a_geometry as stage_a  # noqa: E402
import qwen25_stage_b_router_refit as refit  # noqa: E402


ROOT = Path("/root/autodl-tmp/sakiko-followup")
OUT = ROOT / "final/results/qwen25_stage_b_preparation"
SPEC_PATH = OUT / "qwen25_stage_b_dev_pilot_spec.json"
METRICS_PATH = OUT / "router_refit_metrics.json"
RAW_PATH = OUT / "dev_pilot_rows.jsonl"
SUMMARY_PATH = OUT / "dev_pilot_summary.json"
REPORT_PATH = OUT / "BLOCK2_FIXED_DEV_PILOT_REPORT.md"

MODES = ("tool_call", "direct", "request_for_info", "cannot_answer")
DIMENSION = 3584
PILOT_N = 8
WRONG_LAYER = 4
SCORE_TOLERANCE = 1e-7
EXPECTED_ARMS_PER_CHANNEL_SAMPLE = 29


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    import subprocess

    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def probability(channel: str, project_indices: list[int]) -> dict[int, float]:
    config = refit.CHANNELS[channel]
    router_path = refit.ROUTER_DIR / f"{channel}_router.npz"
    with np.load(router_path, allow_pickle=False) as router:
        mean = router["scaler_mean"]
        scale = router["scaler_scale"]
        coefficient = router["classifier_coef"].reshape(-1)
        intercept = float(router["classifier_intercept"][0])
    activation_path = (
        ROOT
        / "data/processed/qwen25_7b_w2c/cache"
        / f"acts_L{config['L_obs']}.npy"
    )
    activations = np.load(activation_path, mmap_mode="r")
    x = np.asarray(activations[project_indices], dtype=np.float64)
    logits = ((x - mean) / scale) @ coefficient + intercept
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1 / (1 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    scores[~positive] = exponent / (1 + exponent)
    return {
        index: float(value) for index, value in zip(project_indices, scores)
    }


def random_unit(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vector = rng.randn(DIMENSION).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"invalid random vector seed {seed}")
    return vector / norm


def score_vector(scores: dict[str, float]) -> np.ndarray:
    return np.asarray([scores[mode] for mode in MODES], dtype=np.float64)


def top_prediction(scores: dict[str, float]) -> str:
    return sorted(MODES, key=lambda mode: (-scores[mode], MODES.index(mode)))[0]


def margin(scores: dict[str, float], gold: str, source: str) -> float:
    return float(scores[gold] - scores[source])


def arm_record(
    *,
    channel: str,
    index: int,
    sample_id: str,
    gold: str,
    source: str,
    baseline_prediction: str,
    router_probability: float,
    router_active: bool,
    arm: str,
    arm_role: str,
    result: dict[str, Any],
    model_forward_executed: bool,
    gated: bool,
    injection_layer: int | None,
    intended_correction_norm: float,
    direction_source: str | None,
    seed: int | None = None,
) -> dict[str, Any]:
    scores = {mode: float(result["scores"][mode]) for mode in MODES}
    if not all(np.isfinite(list(scores.values()))):
        raise RuntimeError(f"nonfinite scores {channel}/{index}/{arm}")
    return {
        "schema_version": 1,
        "split": "dev",
        "selection": "first eight all-dev UUIDs ascending",
        "channel": channel,
        "project_index": index,
        "sample_id": sample_id,
        "gold": gold,
        "source": source,
        "baseline_prediction": baseline_prediction,
        "router_probability": router_probability,
        "router_tau": 0.4,
        "router_active": router_active,
        "arm": arm,
        "arm_role": arm_role,
        "gated": gated,
        "model_forward_executed": model_forward_executed,
        "injection_layer": injection_layer,
        "intended_correction_norm": intended_correction_norm,
        "direction_source": direction_source,
        "random_seed": seed,
        "scores": scores,
        "prediction": top_prediction(scores),
        "gold_minus_source_margin": margin(scores, gold, source),
        "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        "prompt_tokens": result.get("prompt_tokens"),
        "candidate_tokens": result.get("candidate_tokens"),
    }


def no_op(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "scores": dict(baseline["scores"]),
        "prediction": baseline["prediction"],
        "runtime_seconds": 0.0,
        "prompt_tokens": baseline["prompt_tokens"],
        "candidate_tokens": dict(baseline["candidate_tokens"]),
    }


def analytic_comparator(
    baseline: dict[str, Any],
    real: dict[str, Any],
    gold: str,
    source: str,
) -> tuple[dict[str, Any], float]:
    baseline_margin = margin(baseline["scores"], gold, source)
    real_margin = margin(real["scores"], gold, source)
    delta = real_margin - baseline_margin
    scores = dict(baseline["scores"])
    scores[gold] += delta / 2
    scores[source] -= delta / 2
    transformed_delta = margin(scores, gold, source) - baseline_margin
    if abs(transformed_delta - delta) > 1e-12:
        raise RuntimeError("score-level comparator identity failure")
    return (
        {
            "scores": scores,
            "prediction": top_prediction(scores),
            "runtime_seconds": 0.0,
            "prompt_tokens": baseline["prompt_tokens"],
            "candidate_tokens": dict(baseline["candidate_tokens"]),
        },
        delta,
    )


def run_model_arm(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: dict[str, Any],
    *,
    active: bool,
    gated: bool,
    baseline: dict[str, Any],
    layer: int,
    correction: np.ndarray,
) -> tuple[dict[str, Any], bool]:
    if gated and not active:
        return no_op(baseline), False
    result = stage_a.score_sample(
        model,
        tokenizer,
        device,
        sample,
        inj_layer=layer,
        correction=correction,
    )
    gc.collect()
    stage_a.torch.cuda.empty_cache()
    return result, True


def main() -> None:
    if RAW_PATH.exists() or SUMMARY_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("pilot output namespace is not empty")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics["gate2"] not in {
        "ROUTER_CONTINUITY_SUPPORTED",
        "FRESH_ROUTER_PIPELINE_FEASIBLE",
    }:
        raise RuntimeError("Gate 2 does not authorize dev pilot")
    if spec["wrong_layer"] != WRONG_LAYER:
        raise RuntimeError("wrong-layer spec mismatch")

    baseline_rows, _train, dev = refit.load_allowed_rows()
    selected = sorted(dev, key=lambda index: baseline_rows[index]["uuid"])[:PILOT_N]
    if len(selected) != PILOT_N:
        raise RuntimeError("fixed dev selection incomplete")
    selected_ids = [baseline_rows[index]["uuid"] for index in selected]
    if selected_ids != sorted(selected_ids) or len(set(selected_ids)) != PILOT_N:
        raise RuntimeError("fixed UUID ordering failure")

    # Upstream configuration is named "test"; the project dev boundary is the
    # byte-index allow-list above. Only selected project-dev rows are indexed.
    dataset = stage_a.load_dataset(
        "nvidia/When2Call", "test", split="mcq"
    ).shuffle(seed=42)
    samples: dict[int, dict[str, Any]] = {}
    for index in selected:
        sample = dataset[index]
        if sample["uuid"] != baseline_rows[index]["uuid"]:
            raise RuntimeError(f"dataset UUID mismatch at allowed dev row {index}")
        samples[index] = sample

    directions = stage_a.load_directions()
    direction_cosines = {
        f"{left}|{right}": float(directions[left] @ directions[right])
        for offset, left in enumerate(refit.CHANNELS)
        for right in list(refit.CHANNELS)[offset + 1 :]
    }
    router_probability = {
        channel: probability(channel, selected) for channel in refit.CHANNELS
    }

    model, tokenizer, device, environment = stage_a.load_model()
    pilot_started = time.perf_counter()
    baselines: dict[int, dict[str, Any]] = {}
    for index in selected:
        result = stage_a.score_sample(model, tokenizer, device, samples[index])
        expected = baseline_rows[index]["pred"]
        if result["prediction"] != expected:
            raise RuntimeError(
                f"current baseline mismatch at dev row {index}: "
                f"{result['prediction']} != {expected}"
            )
        baselines[index] = result
        gc.collect()
        stage_a.torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    nonidentity: dict[str, dict[str, Any]] = {}
    correction_norms: dict[str, float] = {}
    for channel, config in spec["channels"].items():
        gold = config["gold"]
        source = config["source"]
        recipient_norm = float(config["rho"] * config["median_norm"])
        correction_norms[channel] = recipient_norm
        real_unit = directions[channel]
        real_correction = (recipient_norm * real_unit).astype(np.float32)
        active_differences: list[float] = []
        ungated_differences: list[float] = []
        active_count = 0

        for index in selected:
            sample = samples[index]
            baseline = baselines[index]
            probability_value = router_probability[channel][index]
            active = (
                baseline["prediction"] == source
                and probability_value >= float(config["tau"])
            )
            active_count += int(active)
            common = {
                "channel": channel,
                "index": index,
                "sample_id": baseline_rows[index]["uuid"],
                "gold": gold,
                "source": source,
                "baseline_prediction": baseline["prediction"],
                "router_probability": probability_value,
                "router_active": active,
            }

            rows.append(
                arm_record(
                    **common,
                    arm="baseline",
                    arm_role="shared_unintervened_baseline",
                    result=baseline,
                    model_forward_executed=True,
                    gated=False,
                    injection_layer=None,
                    intended_correction_norm=0.0,
                    direction_source=None,
                )
            )

            real, real_executed = run_model_arm(
                model,
                tokenizer,
                device,
                sample,
                active=active,
                gated=True,
                baseline=baseline,
                layer=int(config["L_inj"]),
                correction=real_correction,
            )
            rows.append(
                arm_record(
                    **common,
                    arm="real_gated",
                    arm_role="planned_scientific_control_battery",
                    result=real,
                    model_forward_executed=real_executed,
                    gated=True,
                    injection_layer=int(config["L_inj"]),
                    intended_correction_norm=recipient_norm,
                    direction_source=channel,
                )
            )

            reverse, reverse_executed = run_model_arm(
                model,
                tokenizer,
                device,
                sample,
                active=active,
                gated=True,
                baseline=baseline,
                layer=int(config["L_inj"]),
                correction=-real_correction,
            )
            rows.append(
                arm_record(
                    **common,
                    arm="reverse_gated",
                    arm_role="planned_scientific_control_battery",
                    result=reverse,
                    model_forward_executed=reverse_executed,
                    gated=True,
                    injection_layer=int(config["L_inj"]),
                    intended_correction_norm=recipient_norm,
                    direction_source=f"negative_{channel}",
                )
            )
            if active:
                active_differences.append(
                    float(
                        np.max(
                            np.abs(
                                score_vector(real["scores"])
                                - score_vector(reverse["scores"])
                            )
                        )
                    )
                )

            for seed in config["random_seeds"]:
                correction = (recipient_norm * random_unit(int(seed))).astype(
                    np.float32
                )
                result, executed = run_model_arm(
                    model,
                    tokenizer,
                    device,
                    sample,
                    active=active,
                    gated=True,
                    baseline=baseline,
                    layer=int(config["L_inj"]),
                    correction=correction,
                )
                rows.append(
                    arm_record(
                        **common,
                        arm=f"matched_random_gated_seed_{seed}",
                        arm_role="planned_scientific_control_battery",
                        result=result,
                        model_forward_executed=executed,
                        gated=True,
                        injection_layer=int(config["L_inj"]),
                        intended_correction_norm=recipient_norm,
                        direction_source="matched_random",
                        seed=int(seed),
                    )
                )

            for donor in refit.CHANNELS:
                if donor == channel:
                    continue
                correction = (recipient_norm * directions[donor]).astype(np.float32)
                result, executed = run_model_arm(
                    model,
                    tokenizer,
                    device,
                    sample,
                    active=active,
                    gated=True,
                    baseline=baseline,
                    layer=int(config["L_inj"]),
                    correction=correction,
                )
                rows.append(
                    arm_record(
                        **common,
                        arm=f"mismatched_gated_from_{donor}",
                        arm_role="planned_scientific_control_battery",
                        result=result,
                        model_forward_executed=executed,
                        gated=True,
                        injection_layer=int(config["L_inj"]),
                        intended_correction_norm=recipient_norm,
                        direction_source=donor,
                    )
                )

            wrong, wrong_executed = run_model_arm(
                model,
                tokenizer,
                device,
                sample,
                active=active,
                gated=True,
                baseline=baseline,
                layer=WRONG_LAYER,
                correction=real_correction,
            )
            rows.append(
                arm_record(
                    **common,
                    arm=f"wrong_layer_L{WRONG_LAYER}_gated",
                    arm_role="planned_scientific_control_battery",
                    result=wrong,
                    model_forward_executed=wrong_executed,
                    gated=True,
                    injection_layer=WRONG_LAYER,
                    intended_correction_norm=recipient_norm,
                    direction_source=channel,
                )
            )

            ungated_real, _ = run_model_arm(
                model,
                tokenizer,
                device,
                sample,
                active=True,
                gated=False,
                baseline=baseline,
                layer=int(config["L_inj"]),
                correction=real_correction,
            )
            rows.append(
                arm_record(
                    **common,
                    arm="ungated_real",
                    arm_role="planned_scientific_control_battery",
                    result=ungated_real,
                    model_forward_executed=True,
                    gated=False,
                    injection_layer=int(config["L_inj"]),
                    intended_correction_norm=recipient_norm,
                    direction_source=channel,
                )
            )

            ungated_reverse, _ = run_model_arm(
                model,
                tokenizer,
                device,
                sample,
                active=True,
                gated=False,
                baseline=baseline,
                layer=int(config["L_inj"]),
                correction=-real_correction,
            )
            rows.append(
                arm_record(
                    **common,
                    arm="ungated_reverse_engineering_check",
                    arm_role="engineering_sign_check_only",
                    result=ungated_reverse,
                    model_forward_executed=True,
                    gated=False,
                    injection_layer=int(config["L_inj"]),
                    intended_correction_norm=recipient_norm,
                    direction_source=f"negative_{channel}",
                )
            )
            ungated_differences.append(
                float(
                    np.max(
                        np.abs(
                            score_vector(ungated_real["scores"])
                            - score_vector(ungated_reverse["scores"])
                        )
                    )
                )
            )

            comparator, comparator_delta = analytic_comparator(
                baseline, real, gold, source
            )
            comparator_row = arm_record(
                **common,
                arm="score_level_matched_counterfactual",
                arm_role="analytic_secondary_not_model_arm",
                result=comparator,
                model_forward_executed=False,
                gated=True,
                injection_layer=None,
                intended_correction_norm=0.0,
                direction_source="realized_real_margin_delta",
            )
            comparator_row["matched_margin_delta"] = comparator_delta
            rows.append(comparator_row)

        nonidentity[channel] = {
            "active_fixed_rows": active_count,
            "gated_real_reverse_status": (
                "PASS_ALL_ACTIVE_ROWS_NONIDENTICAL"
                if active_count
                and all(value > SCORE_TOLERANCE for value in active_differences)
                else (
                    "NOT_TESTABLE_NO_ACTIVE_FIXED_ROW"
                    if active_count == 0
                    else "FAIL_IDENTICAL_ACTIVE_ROW"
                )
            ),
            "gated_real_reverse_max_abs_differences": active_differences,
            "ungated_real_reverse_status": (
                "PASS_ALL_FIXED_ROWS_NONIDENTICAL"
                if all(value > SCORE_TOLERANCE for value in ungated_differences)
                else "FAIL_IDENTICAL_FIXED_ROW"
            ),
            "ungated_real_reverse_max_abs_differences": ungated_differences,
        }

    elapsed = time.perf_counter() - pilot_started
    expected_total_rows = (
        len(refit.CHANNELS) * PILOT_N * EXPECTED_ARMS_PER_CHANNEL_SAMPLE
    )
    if len(rows) != expected_total_rows:
        raise RuntimeError(f"arm row count {len(rows)} != {expected_total_rows}")
    arm_counts = Counter(row["arm"] for row in rows)
    per_pair_counts = Counter(
        (row["channel"], row["project_index"]) for row in rows
    )
    if set(per_pair_counts.values()) != {EXPECTED_ARMS_PER_CHANNEL_SAMPLE}:
        raise RuntimeError("per channel/sample arm count mismatch")
    if any(
        value["ungated_real_reverse_status"]
        != "PASS_ALL_FIXED_ROWS_NONIDENTICAL"
        for value in nonidentity.values()
    ):
        raise RuntimeError("ungated real/reverse engineering check failed")
    if any(
        value["gated_real_reverse_status"] == "FAIL_IDENTICAL_ACTIVE_ROW"
        for value in nonidentity.values()
    ):
        raise RuntimeError("gated real/reverse engineering check failed")

    model_rows = [
        row
        for row in rows
        if row["model_forward_executed"]
        and row["arm"] != "baseline"
        and row["arm"] != "ungated_reverse_engineering_check"
    ]
    intervention_mean = float(
        np.mean([row["runtime_seconds"] for row in model_rows])
    )
    baseline_mean = float(
        np.mean([value["runtime_seconds"] for value in baselines.values()])
    )
    dev_routed_counts = {
        channel: int(
            metrics["channels"][channel]["dev_operational_tau"][
                "selected_metrics"
            ]["routed_count"]
        )
        for channel in refit.CHANNELS
    }
    gated_arms_per_routed = 25
    ungated_scientific_calls = 548 * len(refit.CHANNELS)
    projected_gated_calls = gated_arms_per_routed * sum(dev_routed_counts.values())
    projected_baseline_calls = 548
    projected_seconds = (
        projected_baseline_calls * baseline_mean
        + (projected_gated_calls + ungated_scientific_calls) * intervention_mean
        + float(environment["model_load_seconds"])
    )
    conservative_seconds = 1.25 * projected_seconds
    cost_status = (
        "STAGE_B_COST_WITHIN_LIMIT"
        if conservative_seconds <= 3 * 3600
        else "STAGE_B_COST_BLOCKED"
    )

    checks = {
        "fixed_selection_same_for_all_channels": True,
        "selected_count": len(selected),
        "raw_rows_count": len(rows),
        "expected_raw_rows_count": expected_total_rows,
        "all_scores_finite": all(
            np.isfinite(list(row["scores"].values())).all() for row in rows
        ),
        "all_destinations_extracted": all(
            row["prediction"] in MODES for row in rows
        ),
        "all_pairs_have_expected_arm_count": set(per_pair_counts.values())
        == {EXPECTED_ARMS_PER_CHANNEL_SAMPLE},
        "random_arm_count_per_pair": all(
            sum(
                row["arm"].startswith("matched_random_gated_seed_")
                for row in rows
                if row["channel"] == channel and row["project_index"] == index
            )
            == 20
            for channel in refit.CHANNELS
            for index in selected
        ),
        "mismatched_arm_count_per_pair": all(
            sum(
                row["arm"].startswith("mismatched_gated_from_")
                for row in rows
                if row["channel"] == channel and row["project_index"] == index
            )
            == 2
            for channel in refit.CHANNELS
            for index in selected
        ),
        "recipient_correction_norms": correction_norms,
        "nonidentity": nonidentity,
        "score_level_comparator_identity": True,
    }
    pilot_status = (
        "PASS"
        if all(
            [
                checks["fixed_selection_same_for_all_channels"],
                checks["all_scores_finite"],
                checks["all_destinations_extracted"],
                checks["all_pairs_have_expected_arm_count"],
                checks["random_arm_count_per_pair"],
                checks["mismatched_arm_count_per_pair"],
                cost_status == "STAGE_B_COST_WITHIN_LIMIT",
            ]
        )
        else "FAIL"
    )

    with RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    summary = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_TRAIN_DEV_ENGINEERING_PILOT",
        "pilot_status": pilot_status,
        "git_head_at_execution": git("rev-parse", "HEAD"),
        "pilot_spec_sha256": sha256_file(SPEC_PATH),
        "selection": [
            {
                "project_index": index,
                "sample_id": baseline_rows[index]["uuid"],
            }
            for index in selected
        ],
        "arm_counts": dict(sorted(arm_counts.items())),
        "direction_cosines": direction_cosines,
        "checks": checks,
        "runtime": {
            "pilot_wall_seconds_excluding_model_load": elapsed,
            "model_load_seconds": environment["model_load_seconds"],
            "mean_baseline_four_mode_seconds": baseline_mean,
            "mean_executed_intervention_four_mode_seconds": intervention_mean,
            "peak_GPU_memory_bytes": int(
                stage_a.torch.cuda.max_memory_allocated(0)
            ),
        },
        "cost_projection": {
            "basis": (
                "dev operational routed counts as train/dev-only rate proxy; "
                "548-row future-size placeholder; no test support/count accessed"
            ),
            "dev_routed_count_proxy": dev_routed_counts,
            "gated_model_arms_per_routed_row": gated_arms_per_routed,
            "projected_baseline_calls": projected_baseline_calls,
            "projected_gated_calls": projected_gated_calls,
            "projected_ungated_scientific_calls": ungated_scientific_calls,
            "analytic_score_comparator_GPU_calls": 0,
            "engineering_only_reverse_calls_in_future_run": 0,
            "point_projected_GPU_seconds": projected_seconds,
            "point_projected_GPU_hours": projected_seconds / 3600,
            "conservative_multiplier": 1.25,
            "conservative_projected_GPU_seconds": conservative_seconds,
            "conservative_projected_GPU_hours": conservative_seconds / 3600,
            "limit_GPU_hours": 3,
            "status": cost_status,
        },
        "environment": {
            **environment,
            "python_version": platform.python_version(),
            "offline_environment": {
                "HF_DATASETS_OFFLINE": os.environ["HF_DATASETS_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            },
        },
        "firewall": {
            "project_split": "dev only",
            "train_rows_used_in_pilot": 0,
            "dev_rows_materialized": len(selected),
            "non_allowlisted_rows_materialized": 0,
            "test_manifest_opened": False,
            "test_index_opened": False,
            "test_rows_ids_labels_scores_or_aggregates_opened": False,
            "stage_b_executed": False,
            "qwen3_started": False,
        },
        "raw_rows_path": str(RAW_PATH.relative_to(ROOT)),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    active_table = "\n".join(
        f"| {channel} | {value['active_fixed_rows']} | "
        f"{value['gated_real_reverse_status']} | "
        f"{value['ungated_real_reverse_status']} |"
        for channel, value in nonidentity.items()
    )
    report = f"""# Block 2 — Fixed Qwen dev engineering pilot

**Pilot: `{pilot_status}`**
**Cost: `{cost_status}`**

The same eight all-dev rows were selected by ascending UUID for every
recipient channel. Selection did not use behavior, margin, geometry, archival
agreement or Router confidence.

All 29 recorded arms/checks per channel/sample were instantiated: baseline,
gated real/reverse, 20 recipient-scaled matched randoms, both recipient-scaled
mismatches, gated wrong-layer L4, ungated real, the engineering-only ungated
reverse sign check, and the analytic score-level comparator. Raw four-mode
scores and destinations are stored in `dev_pilot_rows.jsonl`.

| channel | active fixed rows | gated real/reverse | ungated sign check |
|---|---:|---|---|
{active_table}

Channels with no active fixed row retain
`NOT_TESTABLE_NO_ACTIVE_FIXED_ROW` for the gated comparison; the separately
frozen ungated sign check establishes nonidentity without changing samples.

## Cost projection

Using measured four-mode runtimes and dev routed counts only, the point
projection is {projected_seconds / 3600:.3f} GPU-hours. With the frozen 1.25×
engineering multiplier it is {conservative_seconds / 3600:.3f} GPU-hours,
against the 3-hour cap. The analytic score comparator costs no GPU calls and
the sign-only reverse check is excluded from a future scientific run.

This is a dev engineering result, not Stage B. It accessed no project test
manifest/index/row and did not discover test support.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "pilot_status": pilot_status,
                "active_fixed_rows": {
                    channel: value["active_fixed_rows"]
                    for channel, value in nonidentity.items()
                },
                "cost_status": cost_status,
                "conservative_GPU_hours": conservative_seconds / 3600,
                "raw_rows": len(rows),
                "peak_GPU_memory_bytes": summary["runtime"][
                    "peak_GPU_memory_bytes"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    stage_a.torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
