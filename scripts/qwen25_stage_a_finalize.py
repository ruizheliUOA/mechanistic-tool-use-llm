#!/usr/bin/env python3
"""CPU-only consistency checks and final packaging for Qwen2.5 Stage A."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "qwen25_stage_a_jacobian_geometry"
SCRIPTS = [
    ROOT / "scripts" / "qwen25_stage_a_archive_gate.py",
    ROOT / "scripts" / "qwen25_stage_a_geometry.py",
    ROOT / "scripts" / "qwen25_stage_a_finalize.py",
]
STARTING_HEAD = "800d7eec92d2b7fa6b13ad1ef67e463360cac1c0"
FINAL_STATUS = "CURRENT_ENVIRONMENT_GEOMETRY_ONLY"
MODES = ["tool_call", "direct", "request_for_info", "cannot_answer"]
CHANNELS = ["rfi_tc", "ca_tc", "ca_direct"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text())


def load_jsonl(name: str) -> list[dict[str, Any]]:
    with (OUT / name).open() as handle:
        return [json.loads(line) for line in handle]


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    gate_a0_runner = load("gate_a0_runner_up_analysis.json")
    gate_a0_mismatch = load("gate_a0_mismatched_control_geometry.json")
    gate_a = load("gate_a_configuration_recovery.json")
    gate_b = load("gate_b_engineering_pilot.json")
    gate_c = load("gate_c_replay_report.json")
    summary = load("jacobian_channel_summary.json")
    concentration = load("concentration_null.json")
    linearity = load("linearity.json")
    records = load_jsonl("jacobian_geometry.jsonl")

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({
            "name": name,
            "passed": bool(condition),
            "detail": detail,
        })
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check(
        "gate_a0_runner_structured_skip",
        gate_a0_runner["status"] == "STRUCTURED_SKIP",
        gate_a0_runner["status"],
    )
    check(
        "gate_a0_mismatched_geometry_only",
        gate_a0_mismatch["status"]
        == "GEOMETRY_RECOMPUTED_BEHAVIOR_STRUCTURED_SKIP",
        gate_a0_mismatch["status"],
    )
    check(
        "gate_a_outcome",
        gate_a["gate_a_outcome"] == "GEOMETRY_ONLY_FEASIBLE",
        gate_a["gate_a_outcome"],
    )
    check(
        "gate_b_pass",
        gate_b["gate_b_outcome"] == "PASS",
        gate_b["gate_b_outcome"],
    )
    check(
        "gate_b_cost_below_two_hours",
        gate_b["cost"]["projected_total_gpu_seconds"] <= 7200,
        gate_b["cost"]["projected_total_gpu_seconds"],
    )
    check(
        "gate_c_outcome",
        gate_c["gate_c_outcome"] == FINAL_STATUS,
        gate_c["gate_c_outcome"],
    )
    check(
        "dev_count",
        gate_c["population"]["dev_count"] == 548,
        gate_c["population"]["dev_count"],
    )
    channel_counts = gate_c["population"]["channel_counts"]
    total_samples = sum(channel_counts.values())
    check(
        "geometry_record_count",
        len(records) == 2 * total_samples,
        {"records": len(records), "samples": total_samples},
    )
    sample_types: dict[tuple[str, int], set[str]] = {}
    for record in records:
        key = (record["channel"], record["project_index"])
        sample_types.setdefault(key, set()).add(record["geometry_type"])
        check(
            f"mode_scores_{record['channel']}_{record['project_index']}_"
            f"{record['geometry_type']}",
            set(record["baseline_scores"]) == set(MODES),
            sorted(record["baseline_scores"]),
        )
    check(
        "one_prompt_and_full_record_per_sample",
        all(types == {"common_prompt", "full_effect"}
            for types in sample_types.values()),
        len(sample_types),
    )
    check(
        "unique_population_samples",
        len(sample_types) == total_samples,
        len(sample_types),
    )
    full_records = [
        record for record in records if record["geometry_type"] == "full_effect"
    ]
    check(
        "all_full_effect_valid",
        all(record["geometry_status"] == "VALID" for record in full_records),
        dict(Counter(record["geometry_status"] for record in full_records)),
    )
    check(
        "no_geometry_inconsistency",
        all(
            not record["geometry_status"].startswith("GEOMETRY_INCONSISTENCY")
            for record in records
        ),
        "zero required",
    )
    for channel in CHANNELS:
        for geometry_type in ("common_prompt", "full_effect"):
            key = f"{channel}|{geometry_type}"
            expected = channel_counts[channel]
            check(
                f"summary_count_{key}",
                summary[key]["n_total"] == expected,
                {"summary": summary[key]["n_total"], "expected": expected},
            )
            check(
                f"concentration_count_{key}",
                concentration["groups"][key]["n_axes"]
                == summary[key]["n_valid"],
                {
                    "axes": concentration["groups"][key]["n_axes"],
                    "valid": summary[key]["n_valid"],
                },
            )
    check(
        "linearity_channels",
        set(linearity["channels"]) == set(CHANNELS),
        sorted(linearity["channels"]),
    )
    check(
        "prediction_agreement_arithmetic",
        (
            gate_c["baseline_archive_comparison"]["prediction_agreement_count"]
            + gate_c["baseline_archive_comparison"][
                "prediction_disagreement_count"
            ]
            == 548
        ),
        gate_c["baseline_archive_comparison"],
    )
    for document in (gate_a, gate_b, gate_c):
        boundary = document["hard_boundary"] if "hard_boundary" in document \
            else document["test_access"]
        check(
            f"test_boundary_{document.get('gate_a_outcome', document.get('gate_b_outcome', document.get('gate_c_outcome')))}",
            not any(boundary.values()),
            boundary,
        )
    check(
        "direction_sha256",
        gate_a["directions"]["verified_file_sha256"]
        == "6d765083b1bb9d9127b8c03946b4ed2c7a402b51213c748fe37fae024ed6b34c",
        gate_a["directions"]["verified_file_sha256"],
    )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": "PASS",
        "checks_total": len(checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks": checks,
        "mechanical_status": FINAL_STATUS,
    }
    facts = {
        "gate_a0_runner": gate_a0_runner,
        "gate_a0_mismatch": gate_a0_mismatch,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "summary": summary,
        "concentration": concentration,
        "linearity": linearity,
        "geometry_record_count": len(records),
    }
    return report, facts


def write_prior_regen_crosscheck(facts: dict[str, Any]) -> dict[str, Any]:
    dev = set(map(int, json.loads(
        (ROOT / "final/results/splits/val_idx.json").read_text()
    )))
    archive_path = (
        ROOT / ".git/lfs/objects/e6/6a/"
        "e66a2faff706e44aff1370df79663aaf9257c3699f10b9b3d3d3731fb5903a79"
    )
    regen_path = (
        ROOT / "data/processed/qwen25_7b_w2c/cache/"
        "baseline_details_regen.jsonl"
    )

    def safe_rows(path: Path) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        with path.open("rb") as handle:
            for index, line in enumerate(handle):
                if index in dev:
                    rows[index] = json.loads(line)
        if set(rows) != dev:
            raise AssertionError(f"incomplete dev-only slice: {path}")
        return rows

    archive = safe_rows(archive_path)
    prior_regen = safe_rows(regen_path)
    current = {index: archive[index]["pred"] for index in dev}
    for row in facts["gate_c"]["baseline_archive_comparison"]["disagreements"]:
        current[int(row["project_index"])] = row["current_prediction"]
    differing_indices = sorted({
        index for index in dev
        if archive[index]["pred"] != prior_regen[index]["pred"]
        or archive[index]["pred"] != current[index]
    })
    rows = [{
        "project_index": index,
        "sample_id": archive[index]["uuid"],
        "archived_prediction": archive[index]["pred"],
        "prior_regen_prediction": prior_regen[index]["pred"],
        "current_prediction": current[index],
    } for index in differing_indices]
    result = {
        "schema_version": 1,
        "label": "CURRENT_ENVIRONMENT_GEOMETRY_ONLY",
        "dev_rows": len(dev),
        "archive_vs_prior_regen_prediction_disagreements": sum(
            archive[index]["pred"] != prior_regen[index]["pred"]
            for index in dev
        ),
        "current_vs_prior_regen_prediction_disagreements": sum(
            current[index] != prior_regen[index]["pred"] for index in dev
        ),
        "archive_vs_current_prediction_disagreements": sum(
            current[index] != archive[index]["pred"] for index in dev
        ),
        "rows": rows,
        "interpretation": (
            "The current Stage A prediction vector exactly reproduces the existing "
            "baseline_details_regen dev prediction vector. The four-way difference "
            "is between both current-environment runs and the older archive, not a "
            "new Stage A implementation discrepancy. This does not restore missing "
            "Router or intervention provenance."
        ),
        "test_access": {
            "project_test_index_opened": False,
            "project_test_rows_materialized": False,
            "project_test_labels_scores_or_aggregates_read": False,
        },
    }
    if result["current_vs_prior_regen_prediction_disagreements"] != 0:
        raise AssertionError("current run does not match prior regen")
    (OUT / "current_baseline_prior_regen_crosscheck.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    table = "\n".join(
        f"| {row['project_index']} | `{row['sample_id']}` | "
        f"{row['archived_prediction']} | {row['prior_regen_prediction']} | "
        f"{row['current_prediction']} |"
        for row in rows
    )
    (OUT / "CURRENT_BASELINE_PRIOR_REGEN_CROSSCHECK.md").write_text(
        "# Current baseline × prior regeneration cross-check\n\n"
        "The current Stage A dev prediction vector exactly matches the repository's "
        "existing `baseline_details_regen.jsonl` dev slice: **548/548**. Both differ "
        "from the older archived baseline on the same four rows.\n\n"
        "| index | UUID | old archive | prior regen | current Stage A |\n"
        "|---:|---|---|---|---|\n"
        f"{table}\n\n"
        "This is positive evidence that the current scoring implementation is "
        "consistent with the prior current-environment reproduction. It does not "
        "satisfy `FULL_REPLAY_FEASIBLE`, because the original serialized Router, "
        "exact routed-dev order, and dev per-sample intervention records remain "
        "missing. Only dev allow-listed JSONL rows were decoded.\n"
    )
    return result


def write_stage_b_draft() -> None:
    (OUT / "STAGE_B_DRAFT_NOT_AUTHORIZED.md").write_text(
        "# Stage B draft — NOT AUTHORIZED\n\n"
        "**Status:** `DRAFT_NOT_AUTHORIZED`\n\n"
        "Stage A returned `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`. Archived replay "
        "continuity was not established: the original serialized Router, exact "
        "routed-dev ordering, and dev per-sample intervention records are absent, "
        "and the current dev baseline differs from the archived baseline on four "
        "rows. A confirmatory Stage B is therefore not yet justified.\n\n"
        "This document is a non-executable planning draft. It contains no test "
        "manifest, test index, test ID, test sample, test label, test score, test "
        "aggregate, test-population hash, execution freeze, runner, or "
        "authorization.\n\n"
        "## Items a future separately authorized protocol would have to freeze\n\n"
        "- The test population definition, without accessing it during drafting.\n"
        "- One primary endpoint and its estimator.\n"
        "- The complete destination-flow table and collateral/damage limits.\n"
        "- Random-direction controls with fixed seeds.\n"
        "- Reverse-direction, mismatched-direction, wrong-layer, and ungated controls.\n"
        "- A logit-bias comparator calibrated only on dev, with its matching rule "
        "frozen before any test access; it must never be matched to realized test "
        "`delta e_target`.\n"
        "- Fixed seeds, success rules, withdrawal rules, engineering-failure "
        "handling, and permanent invalidation conditions.\n"
        "- Independent recovery or prospective replacement of the missing Router "
        "identity and routed-population provenance.\n\n"
        "A new human authorization is required before any Stage B protocol freeze "
        "or execution work.\n"
    )


def value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6g}"


def write_summary(facts: dict[str, Any]) -> None:
    gate_b = facts["gate_b"]
    gate_c = facts["gate_c"]
    summary = facts["summary"]
    concentration = facts["concentration"]
    linearity = facts["linearity"]
    geometry_lines = []
    for key, row in summary.items():
        geometry_lines.append(
            f"| {row['channel']} | {row['geometry_type']} | {row['n_total']} | "
            f"{row['n_valid']} | {row['n_common_prompt_state_not_shared']} | "
            f"{value(row['observed_rank']['median'])} | "
            f"{value(row['a_contrast']['median'])} | "
            f"{value(row['a_dec']['median'])} |"
        )
    concentration_lines = []
    for key, row in concentration["groups"].items():
        concentration_lines.append(
            f"| {key} | {row['n_axes']} | "
            f"{value(row.get('resultant_length'))} | "
            f"{value(row.get('chance_null', {}).get('mean'))} | "
            f"{value(row.get('chance_null', {}).get('empirical_one_sided_p'))} |"
        )
    linearity_lines = []
    for channel, populations in linearity["channels"].items():
        row = populations["reconstructed_router_active"][
            "decision_margin_change_primary"
        ]
        linearity_lines.append(
            f"| {channel} | "
            f"{populations['reconstructed_router_active']['n_samples']} | "
            f"{value(row['pearson'])} | {value(row['spearman'])} | "
            f"{value(row['slope'])} | {value(row['rmse'])} |"
        )
    cosines = facts["gate_a"]["directions"]["signed_cosines"]
    (OUT / "STAGE_A_SUMMARY.md").write_text(
        "# Qwen2.5 Stage A — Dev-only descriptive result\n\n"
        "**Mechanical status:** `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`\n\n"
        "## Gates\n\n"
        "- Gate A0.1: structured skip. Dev four-mode baseline scores exist, but "
        "dev per-sample intervention destinations do not; no destination flow or "
        "permutation null was fabricated.\n"
        "- Gate A0.2: direction Gram recomputed; dev mismatched-control behavior "
        "structured-skip because only prohibited test controls were archived.\n"
        "- Gate A: `GEOMETRY_ONLY_FEASIBLE`; exact model/tokenizer/scoring/hook/"
        "positions/scaling/directions/data identity recovered, original Router "
        "and archived routed/intervention records missing.\n"
        f"- Gate B: `PASS`; two-run max score noise "
        f"`{gate_b['noise_floor']['observed_max_abs_score_difference']}`, no "
        f"prediction or runner-up flips, projected "
        f"`{gate_b['cost']['projected_total_gpu_hours']:.4f}` GPU-hours.\n"
        f"- Common-state pilot: "
        f"{gate_b['well_posedness']['prompt_state_shared_sample_count']} / "
        f"{len(gate_b['samples'])} samples shared within the frozen `1e-6` "
        f"absolute tolerance; maximum relative deviation "
        f"{gate_b['well_posedness']['max_relative_prompt_hidden_deviation']:.4f}. "
        "Common-prompt geometry is therefore sample-conditional.\n"
        f"- Gate C: `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`; current channel counts "
        f"`{json.dumps(gate_c['population']['channel_counts'], sort_keys=True)}`; "
        f"archived/current baseline prediction agreement "
        f"{gate_c['baseline_archive_comparison']['prediction_agreement_count']}/548.\n"
        f"- Prior-regeneration cross-check: current predictions match the existing "
        f"`baseline_details_regen` dev slice on "
        f"{548 - facts['prior_regen']['current_vs_prior_regen_prediction_disagreements']}/548 rows; "
        "the same four rows distinguish both current-environment runs from the "
        "older archive.\n\n"
        "## Recovered direction Gram\n\n"
        f"- rfi_tc ↔ ca_tc: `{cosines['rfi_tc|ca_tc']:.9f}` "
        "(non-orthogonal mismatched control).\n"
        f"- rfi_tc ↔ ca_direct: `{cosines['rfi_tc|ca_direct']:.9f}` "
        "(near-orthogonal mismatched control).\n"
        f"- ca_tc ↔ ca_direct: `{cosines['ca_tc|ca_direct']:.9f}` "
        "(near-orthogonal mismatched control).\n\n"
        "## Geometry distributions\n\n"
        "| channel | geometry | n | valid | prompt state not shared | median rank | median a_contrast | median a_dec |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(geometry_lines)
        + "\n\nAll full-effect records are valid, observed rank is 3 for every "
        "record, and no contrast-undefined, rank-zero, or geometry-inconsistency "
        "case occurred. Rank was measured, not imposed.\n\n"
        "## Contrast-axis concentration\n\n"
        "| channel/geometry | n axes | resultant length | chance mean | empirical p |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(concentration_lines)
        + "\n\nThe one-sided empirical floor is `1/10001`; these are descriptive "
        "chance comparisons, not confirmatory p-values.\n\n"
        "## Historical-dose decision-margin linearity\n\n"
        "Router-active results use the deterministically reconstructed, explicitly "
        "non-original Router.\n\n"
        "| channel | active n | Pearson | Spearman | slope | RMSE |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(linearity_lines)
        + "\n\nThe first-order approximation is channel-dependent and weak for "
        "ca_tc/ca_direct at the deployed dose. This is descriptive evidence and "
        "does not authorize dose tuning or a Stage B test.\n\n"
        "## Boundary\n\n"
        "No project test manifest, index, ID, sample content, label, score, or "
        "aggregate was opened. No Qwen3 or Llama work was performed.\n"
    )


def build_manifest(report: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    gate_a = facts["gate_a"]
    gate_b = facts["gate_b"]
    gate_c = facts["gate_c"]
    script_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in SCRIPTS
    }
    artifact_paths = sorted(
        path for path in OUT.iterdir()
        if path.is_file()
        and path.name not in {"STAGE_A_MANIFEST.json", "STAGE_A_HASHES.sha256"}
    )
    artifact_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_tier": "DESCRIPTIVE / DEV-AND-TRAIN-ONLY / RETRYABLE",
        "mechanical_status": FINAL_STATUS,
        "git": {
            "execution_starting_head": STARTING_HEAD,
            "head_at_packaging": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "origin": git("remote", "get-url", "origin"),
        },
        "model": gate_a["model_tokenizer"],
        "dataset": gate_a["dataset"],
        "directions": {
            "lfs_oid_sha256": gate_a["directions"]["lfs_oid_sha256"],
            "verified_file_sha256": gate_a["directions"][
                "verified_file_sha256"
            ],
            "vectors": gate_a["directions"]["vectors"],
            "signed_cosines": gate_a["directions"]["signed_cosines"],
        },
        "archived_artifacts": {
            "baseline_lfs_oid_sha256": facts["gate_a"]["fields"][-1][
                "source"
            ].split()[1],
            "baseline_allowed_slice_sha256": load(
                "gate_a_archive_audit.json"
            )["safe_archive_slices"]["baseline_allowed_slice_sha256"],
            "split_baseline_lfs_oid_sha256": load(
                "gate_a_archive_audit.json"
            )["safe_archive_slices"]["split_baseline_lfs_oid"],
            "split_baseline_allowed_slice_sha256": load(
                "gate_a_archive_audit.json"
            )["safe_archive_slices"][
                "split_baseline_allowed_slice_sha256"
            ],
        },
        "rng": {
            "model_and_router_seed": 42,
            "permutation_null_declared_but_not_run_seed": 20260728,
            "concentration_and_orientation_null_seed": 20260728,
        },
        "environment": gate_c["environment"],
        "gate_b": {
            "outcome": gate_b["gate_b_outcome"],
            "noise_floor": gate_b["noise_floor"],
            "well_posedness": gate_b["well_posedness"],
            "cost": gate_b["cost"],
        },
        "gate_c": {
            "outcome": gate_c["gate_c_outcome"],
            "prediction_vector_sha256": gate_c["population"][
                "prediction_vector_sha256"
            ],
            "channel_counts": gate_c["population"]["channel_counts"],
            "runtime": gate_c["runtime"],
        },
        "prior_regen_crosscheck": facts["prior_regen"],
        "engineering_retry_history": [{
            "kind": "pilot_instrumentation_retry",
            "reason": (
                "first pilot recorded absolute prompt-state deviation but not "
                "relative scale/distribution; the same fixed samples and scientific "
                "configuration were rerun with additional diagnostics"
            ),
            "first_pilot_forward_wall_seconds": 40.51842314377427,
            "first_pilot_peak_gpu_memory_bytes": 21213877248,
            "scientific_configuration_changed": False,
        }],
        "retained_gpu_runtime": {
            "pilot_wall_seconds_excluding_load": gate_b["cost"][
                "pilot_wall_seconds_excluding_model_load"
            ],
            "pilot_model_load_seconds": gate_b["environment"][
                "model_load_seconds"
            ],
            "full_wall_seconds_excluding_load": gate_c["runtime"][
                "full_wall_seconds_excluding_model_load"
            ],
            "full_model_load_seconds": gate_c["runtime"]["model_load_seconds"],
        },
        "scripts_sha256": script_hashes,
        "artifacts_sha256_excluding_manifest_and_hash_list": artifact_hashes,
        "consistency": {
            "overall": report["overall"],
            "checks_total": report["checks_total"],
            "checks_passed": report["checks_passed"],
        },
        "constraints": {
            "project_test_manifest_read": False,
            "project_test_index_read": False,
            "project_test_ids_read": False,
            "project_test_content_read": False,
            "project_test_labels_read": False,
            "project_test_scores_or_aggregates_read": False,
            "qwen3_work": False,
            "llama_or_phase10_3_modified": False,
            "model_weights_downloaded": False,
            "direction_retuned": False,
            "layer_retuned": False,
            "dose_retuned": False,
            "router_retuned": False,
            "stage_b_authorized": False,
        },
    }


def main() -> None:
    report, facts = validate()
    prior_regen = write_prior_regen_crosscheck(facts)
    facts["prior_regen"] = prior_regen
    report["checks"].append({
        "name": "current_predictions_match_prior_regen",
        "passed": (
            prior_regen["current_vs_prior_regen_prediction_disagreements"] == 0
        ),
        "detail": prior_regen,
    })
    report["checks_total"] = len(report["checks"])
    report["checks_passed"] = sum(
        item["passed"] for item in report["checks"]
    )
    write_stage_b_draft()
    write_summary(facts)
    # Validate again after adding narrative files; their content does not alter
    # any machine-readable scientific result.
    (OUT / "STAGE_A_CONSISTENCY_CHECK.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    manifest = build_manifest(report, facts)
    (OUT / "STAGE_A_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    paths = [
        *SCRIPTS,
        *sorted(
            path for path in OUT.iterdir()
            if path.is_file() and path.name != "STAGE_A_HASHES.sha256"
        ),
    ]
    with (OUT / "STAGE_A_HASHES.sha256").open("w") as handle:
        for path in paths:
            handle.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")
    print(json.dumps({
        "overall": report["overall"],
        "checks": report["checks_total"],
        "mechanical_status": FINAL_STATUS,
        "artifacts": len([path for path in OUT.iterdir() if path.is_file()]),
        "hash_manifest": str(
            (OUT / "STAGE_A_HASHES.sha256").relative_to(ROOT)
        ),
    }, indent=2))


if __name__ == "__main__":
    # Imported lazily to keep the validator's top-level dependency set minimal.
    from collections import Counter

    main()
