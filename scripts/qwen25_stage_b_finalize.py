#!/usr/bin/env python3
"""Finalize descriptive Stage B preparation at Hard Stop 0.

This performs consistency and hash checks only. It does not import torch,
load a model, access data splits, or create Block 3/4 artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/autodl-tmp/sakiko-followup")
OUT = ROOT / "final/results/qwen25_stage_b_preparation"
START_HEAD = "0d9a0db1fe715562e422189968c3933435ef0179"
PREFIT_COMMITS = [
    "91fa4f0",
    "8fd0b68",
    "df6c248",
    "34f1c70",
]
CHANNELS = ("rfi_tc", "ca_tc", "ca_direct")
LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")

BLOCK0 = OUT / "block0_evaluation_admissibility_audit.json"
GEOMETRY = OUT / "block1_geometry_verification.json"
ROUTERS = OUT / "router_refit_metrics.json"
PILOT = OUT / "dev_pilot_summary.json"
RAW = OUT / "dev_pilot_rows.jsonl"
REPORT = OUT / "STAGE_B_PREPARATION_BLOCKED_REPORT.md"
CONSISTENCY = OUT / "STAGE_B_PREPARATION_CONSISTENCY_CHECK.json"
MANIFEST = OUT / "STAGE_B_PREPARATION_MANIFEST.json"
HASHES = OUT / "STAGE_B_PREPARATION_HASHES.sha256"

FORBIDDEN_OUTPUTS = [
    OUT / "STAGE_B_PREREGISTRATION.md",
    OUT / "STAGE_B_EXECUTION_FREEZE.md",
    OUT / "STAGE_B_MODEL_ENVIRONMENT_MANIFEST.json",
    OUT / "STAGE_B_RESULT_SCHEMA.json",
    OUT / "STAGE_B_VOID_SCHEMA.json",
    OUT / "STAGE_B_HUMAN_AUTHORIZATION_CHECKLIST.md",
    ROOT / "scripts/qwen25_stage_b_runner.py",
    ROOT / "scripts/qwen25_stage_b_preflight.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check_hash_list(path: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = ROOT / relative
        checks.append(
            {
                "name": f"hash:{relative}",
                "passed": bool(
                    LOWER_HEX_64.fullmatch(digest)
                    and target.is_file()
                    and sha256_file(target) == digest
                ),
            }
        )
    return checks


def add_check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def main() -> None:
    block0 = load(BLOCK0)
    geometry = load(GEOMETRY)
    routers = load(ROUTERS)
    pilot = load(PILOT)
    pilot_spec = load(OUT / "qwen25_stage_b_dev_pilot_spec.json")
    prefit_manifest = load(OUT / "BLOCK01_PREFIT_MANIFEST.json")
    router_manifest = load(OUT / "BLOCK2_ROUTER_REFIT_MANIFEST.json")

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "block0_historical_only",
        block0["classification"] == "HISTORICALLY_USED_EVALUATION_ONLY",
    )
    add_check(checks, "hard_stop_0", block0["hard_stop_0_triggered"] is True)
    add_check(
        checks,
        "audit_incident_disclosed",
        block0["audit_incident"]["occurred"] is True,
    )
    add_check(
        checks,
        "scope_exact_three",
        geometry["scope"]["included"] == list(CHANNELS),
    )
    add_check(
        checks,
        "excluded_channels",
        geometry["scope"]["excluded"]
        == {
            "ca_rfi": "OUT_OF_SCOPE_DIRECTION_NOT_AVAILABLE",
            "tc_rfi": "OUT_OF_SCOPE_DIRECTION_NOT_AVAILABLE",
        },
    )
    add_check(
        checks,
        "rank_verified_three",
        geometry["nulls"]["observed_valid_ranks"] == [3],
    )
    add_check(
        checks,
        "common_counts",
        geometry["full_common_consistency"]["observed_common_valid_counts"]
        == {"rfi_tc": 60, "ca_tc": 46, "ca_direct": 32},
    )
    add_check(
        checks,
        "full_common_Q_ordering",
        all(
            geometry["full_common_consistency"]["ordering"][variant][
                "rfi_lower_than_both"
            ]
            for variant in ("full_effect", "common_prompt")
        ),
    )
    pooled_ci = geometry["bootstrap"]["results"]["full_effect"][
        "rfi_tc_vs_pooled_comparisons"
    ]["Q_difference_percentile_95_CI"]
    add_check(checks, "pooled_Q_CI_below_zero", pooled_ci[1] < 0)
    add_check(
        checks,
        "signed_contrast_unavailable_not_inferred",
        geometry["signed_contrast"]["status"]
        == "UNAVAILABLE_FROM_STORED_STAGE_A_ARTIFACTS",
    )
    add_check(
        checks,
        "gradient_dominance_unavailable_not_inferred",
        geometry["gradient_norm_structure"][
            "centered_energy_dominance_status"
        ]
        == "UNAVAILABLE_FROM_STORED_STAGE_A_ARTIFACTS",
    )
    add_check(
        checks,
        "concentration_all_exceed_p99",
        all(
            group["observed_exceeds_null_p99"]
            for group in geometry["concentration"]["groups"].values()
        ),
    )
    add_check(
        checks,
        "gate1_weak",
        geometry["gate1"]["branch"] == "ORDERING_PREDICTION_WEAK",
    )
    add_check(
        checks,
        "gate2_fresh",
        routers["gate2"] == "FRESH_ROUTER_PIPELINE_FEASIBLE",
    )
    add_check(
        checks,
        "router_all_auc_floor",
        all(
            routers["channels"][channel]["eligibility"]["AUC_floor_pass"]
            for channel in CHANNELS
        ),
    )
    add_check(
        checks,
        "router_all_tau_floor",
        all(
            routers["channels"][channel]["eligibility"][
                "tau_precision_floor_pass"
            ]
            for channel in CHANNELS
        ),
    )
    add_check(
        checks,
        "router_all_gap_small",
        all(
            routers["channels"][channel]["dev_comparable"][
                "absolute_ROC_AUC_gap"
            ]
            <= 0.05
            for channel in CHANNELS
        ),
    )
    for channel in CHANNELS:
        router_path = ROOT / routers["channels"][channel]["serialization"]["path"]
        with np.load(router_path, allow_pickle=False) as archive:
            add_check(
                checks,
                f"router_npz_numeric_fields:{channel}",
                set(archive.files)
                == {
                    "scaler_mean",
                    "scaler_scale",
                    "classifier_coef",
                    "classifier_intercept",
                    "classifier_classes",
                }
                and all(archive[name].dtype.kind in "fiu" for name in archive.files),
            )
        add_check(
            checks,
            f"router_npz_hash:{channel}",
            sha256_file(router_path)
            == routers["channels"][channel]["serialization"]["model_file_sha256"],
        )

    add_check(checks, "pilot_pass", pilot["pilot_status"] == "PASS")
    add_check(
        checks,
        "cost_within_limit",
        pilot["cost_projection"]["status"] == "STAGE_B_COST_WITHIN_LIMIT"
        and pilot["cost_projection"]["conservative_projected_GPU_hours"] <= 3,
    )
    add_check(
        checks,
        "pilot_spec_hash",
        sha256_file(OUT / "qwen25_stage_b_dev_pilot_spec.json")
        == pilot["pilot_spec_sha256"],
    )
    raw_rows: list[dict[str, Any]] = []
    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            raw_rows.append(json.loads(line))
    add_check(checks, "pilot_raw_count", len(raw_rows) == 696)
    add_check(
        checks,
        "pilot_raw_dev_only",
        all(row["split"] == "dev" for row in raw_rows),
    )
    per_pair = Counter(
        (row["channel"], row["project_index"]) for row in raw_rows
    )
    add_check(
        checks,
        "pilot_29_arms_per_pair",
        len(per_pair) == 24 and set(per_pair.values()) == {29},
    )
    add_check(
        checks,
        "pilot_same_eight_rows",
        all(
            {
                row["project_index"]
                for row in raw_rows
                if row["channel"] == channel
            }
            == {
                entry["project_index"] for entry in pilot["selection"]
            }
            for channel in CHANNELS
        ),
    )
    add_check(
        checks,
        "pilot_all_finite",
        all(
            np.isfinite(list(row["scores"].values())).all() for row in raw_rows
        ),
    )
    add_check(
        checks,
        "pilot_random_20_each",
        all(
            sum(
                row["arm"].startswith("matched_random_gated_seed_")
                for row in raw_rows
                if row["channel"] == channel
                and row["project_index"] == entry["project_index"]
            )
            == 20
            for channel in CHANNELS
            for entry in pilot["selection"]
        ),
    )
    add_check(
        checks,
        "pilot_real_reverse_sign_checks",
        all(
            value["ungated_real_reverse_status"]
            == "PASS_ALL_FIXED_ROWS_NONIDENTICAL"
            and value["gated_real_reverse_status"]
            in {
                "PASS_ALL_ACTIVE_ROWS_NONIDENTICAL",
                "NOT_TESTABLE_NO_ACTIVE_FIXED_ROW",
            }
            for value in pilot["checks"]["nonidentity"].values()
        ),
    )
    add_check(
        checks,
        "pilot_frozen_seeds",
        all(
            len(pilot_spec["channels"][channel]["random_seeds"]) == 20
            and len(set(pilot_spec["channels"][channel]["random_seeds"])) == 20
            for channel in CHANNELS
        ),
    )
    add_check(
        checks,
        "forbidden_block3_block4_outputs_absent",
        all(not path.exists() for path in FORBIDDEN_OUTPUTS),
    )
    for commit in PREFIT_COMMITS:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT,
            check=False,
        )
        add_check(checks, f"required_commit_ancestor:{commit}", result.returncode == 0)
    checks.extend(
        check_hash_list(OUT / "BLOCK01_PREFIT_HASHES.sha256")
    )
    checks.extend(
        check_hash_list(OUT / "BLOCK2_ROUTER_REFIT_HASHES.sha256")
    )
    add_check(
        checks,
        "prefit_manifest_hash_format",
        all(
            LOWER_HEX_64.fullmatch(value)
            for value in prefit_manifest["files"].values()
        ),
    )
    add_check(
        checks,
        "router_manifest_hash_format",
        all(
            LOWER_HEX_64.fullmatch(value)
            for value in router_manifest["files"].values()
        ),
    )

    changed = set(
        filter(
            None,
            git("diff", "--name-only", f"{START_HEAD}..HEAD").splitlines(),
        )
    )
    permitted_prefixes = (
        "final/results/qwen25_stage_b_preparation/",
        "scripts/qwen25_stage_b_",
    )
    add_check(
        checks,
        "no_llama_or_phase10_3_modification",
        all(path.startswith(permitted_prefixes) for path in changed),
    )

    if not all(check["passed"] for check in checks):
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"consistency failure before final report: {failed}")

    null_contrast = geometry["nulls"]["a_contrast_single_axis"]
    null_dec = geometry["nulls"]["a_dec_verified_rank_specific"]
    geometry_rows = []
    for channel in CHANNELS:
        full_group = geometry["metrics"][channel]["full_effect"]
        common_group = geometry["metrics"][channel]["common_prompt"]
        full = full_group["offaxis"]
        common = common_group["offaxis"]
        geometry_rows.append(
            f"| {channel} | {full_group['n_valid']} | "
            f"{full['q_unconditional']['median']:.6f} | "
            f"{full['Q_ratio_of_sums']:.6f} | "
            f"{common_group['n_valid']} | "
            f"{common['q_unconditional']['median']:.6f} | "
            f"{common['Q_ratio_of_sums']:.6f} |"
        )
    layer_rows = []
    for channel in CHANNELS:
        value = geometry["layer_provenance"][channel]
        layer_rows.append(
            f"| {channel} | {value['L_obs']} | {value['L_inj']} | "
            f"{value['L_jacobian']} | {value['L_obs_equals_L_inj']} |"
        )
    router_rows = []
    for channel in CHANNELS:
        value = routers["channels"][channel]
        selected = value["dev_operational_tau"]["selected_metrics"]
        router_rows.append(
            f"| {channel} | {value['train']['positive_support']} / "
            f"{value['train']['correct_negative_support']} | "
            f"{value['dev_comparable']['ROC_AUC']:.6f} | "
            f"{value['dev_comparable']['PR_AUC']:.6f} | "
            f"{value['dev_comparable']['absolute_ROC_AUC_gap']:.6f} | "
            f"{value['dev_operational_tau']['selected_tau']} | "
            f"{selected['precision']:.4f} | {selected['recall']:.4f} | "
            f"{selected['specificity']:.4f} | {selected['routed_count']} |"
        )
    bootstrap_rows = []
    for geometry_type in ("full_effect", "common_prompt"):
        for comparison, result in geometry["bootstrap"]["results"][
            geometry_type
        ].items():
            point = result["observed"]["Q_difference_left_minus_comparison"]
            ci = result["Q_difference_percentile_95_CI"]
            bootstrap_rows.append(
                f"| {geometry_type} | {comparison} | {point:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] |"
            )

    report = f"""# Qwen2.5 Stage B preparation — blocked final report

**Final mechanical verdict: `STAGE_B_BLOCKED_NO_UNTOUCHED_EVALUATION`**

Starting HEAD: `{START_HEAD}`
Finalization input HEAD: `{git('rev-parse', 'HEAD')}`

## Block 0 — evaluation admissibility

Classification: `HISTORICALLY_USED_EVALUATION_ONLY`.

Repository source and Git metadata show that prior Qwen experiments evaluated
the fixed project evaluation population and used its outcomes/aggregates for
locked, follow-up, multiseed and control analyses. The split rule predates the
current geometry hypothesis and Stage A itself remained isolated, but the
population is not untouched. Hard Stop 0 therefore applies.

Audit disclosure: an initial read-only provenance command mistakenly opened a
historical Router metrics report and exposed aggregate test-split population
counts. No test row, ID, sample content, label/prediction vector or per-sample
score was opened. The report was not reopened. Consequently, zero
test-aggregate access cannot truthfully be confirmed.

## Scope

Included: `rfi_tc`, `ca_tc`, `ca_direct`. `ca_rfi` and `tc_rfi` are
`OUT_OF_SCOPE_DIRECTION_NOT_AVAILABLE`. This is not the complete Qwen channel
population and cannot validate a universal correctability predictor.

## Block 1 — geometry

| channel | L_obs | L_inj | Jacobian | equal |
|---|---:|---:|---:|---|
{chr(10).join(layer_rows)}

For layer mismatches, the vector was estimated in one layer's coordinates and
injected/differentiated at another; no cross-layer transport map was
estimated.

Corrected nulls (d=3584):

- `a_contrast` Beta(1/2,3583/2): mean {null_contrast['mean']:.9g}, median
  {null_contrast['median']:.9g}, p95 {null_contrast['p95']:.9g}, p99
  {null_contrast['p99']:.9g}.
- Every valid record verified rank 3. `a_dec` Beta(3/2,3581/2): mean
  {null_dec['mean']:.9g}, median {null_dec['median']:.9g}, p95
  {null_dec['p95']:.9g}, p99 {null_dec['p99']:.9g}.

| channel | full n | full median q | full Q | common n | common median q | common Q |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(geometry_rows)}

Ten-thousand stratified bootstraps (seed 20260729):

| geometry | comparison | Q difference (rfi−comparison) | percentile 95% CI |
|---|---|---:|---|
{chr(10).join(bootstrap_rows)}

All six concentration resultants exceed their authorized 10,000-trial
random-axis null p99. Full/common Q ordering is subset-stable. However, signed
contrast and centered-gradient dominance are not recoverable from stored Stage
A artifacts and were not inferred. Gate 1 is
`ORDERING_PREDICTION_WEAK`; three-channel ordering remains
secondary/exploratory.

Stage A Router-active counts 57/62/52 are
`RECONSTRUCTED_ROUTER_RULE`, not archived Router labels.

## Block 2 — train-only Router refit

Pre-fit specification:
`QWEN25_STAGE_B_ROUTER_REFIT_SPEC.md`; initial freeze `91fa4f0`, dev-population
clarification `8fd0b68`, both pushed before fitting.

| channel | train pos/neg | dev ROC-AUC | dev PR-AUC | |historical gap| | tau | precision | recall | specificity | routed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(router_rows)}

All absolute floors pass. Because the original serialized Router is missing,
Gate 2 is `FRESH_ROUTER_PIPELINE_FEASIBLE`, not an archived reconstruction.

The fixed dev pilot used the same first eight ascending UUIDs for every
channel, instantiated 20 randoms and all specified controls, stored 696 raw
rows, and passed all finite-score/destination/arm-count/scaling/comparator
checks. Active fixed rows were rfi=3, ca_tc=0, ca_direct=0; gated
real/reverse was nonidentical on all active rows, and the frozen ungated sign
check was nonidentical on all rows/channels.

Measured cost projection: {pilot['cost_projection']['point_projected_GPU_hours']:.3f}
GPU-hours; conservative 1.25× projection:
{pilot['cost_projection']['conservative_projected_GPU_hours']:.3f} GPU-hours,
below the 3-hour cap.

## Blocks 3–4

Not created because Hard Stop 0 fired:

- preregistration: `NOT_CREATED_HARD_STOP_0`;
- executable Stage B runner: `NOT_CREATED_HARD_STOP_0`;
- execution freeze/preflight/result/VOID schemas: `NOT_CREATED_HARD_STOP_0`.

No Stage B execution was performed.

## Constraint accounting

- Project test rows/IDs/content/labels/prediction vectors/per-sample scores:
  not accessed.
- Historical test aggregate: **accessed once accidentally during provenance
  search and disclosed above**; therefore the requested zero-test-access
  confirmation is not made.
- Router fitting: train only; tau/metrics: dev only.
- Pilot: eight fixed project-dev rows only.
- Model load/inference: local bf16 Qwen only for the authorized dev pilot;
  offline mode, no weight/tokenizer download.
- Qwen3: not started.
- Llama/Phase 10.3: not modified.
- Direction, scientific injection layer, dose and endpoint: not retuned.
- Stage B, Block 3 and Block 4: not executed/emitted.

## Mechanical conclusion

Useful descriptive geometry, Router and engineering feasibility evidence is
complete, internally consistent and pushed through the prerequisite commits.
It cannot become a confirmatory Stage B package without a genuinely untouched
evaluation population and new authorization.

`STAGE_B_BLOCKED_NO_UNTOUCHED_EVALUATION`
"""
    REPORT.write_text(report, encoding="utf-8")

    core_paths = sorted(
        [
            path
            for path in OUT.rglob("*")
            if path.is_file()
            and path not in {CONSISTENCY, MANIFEST, HASHES}
        ]
        + [
            ROOT / "scripts/qwen25_stage_b_block01.py",
            ROOT / "scripts/qwen25_stage_b_train_dev_index.py",
            ROOT / "scripts/qwen25_stage_b_router_refit.py",
            ROOT / "scripts/qwen25_stage_b_dev_pilot.py",
            ROOT / "scripts/qwen25_stage_b_finalize.py",
        ],
        key=lambda path: str(path.relative_to(ROOT)),
    )
    core_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in core_paths
    }
    consistency_payload = {
        "schema_version": 1,
        "overall": "PASS",
        "checks_total": len(checks),
        "checks_passed": sum(check["passed"] for check in checks),
        "checks": checks,
        "final_verdict": "STAGE_B_BLOCKED_NO_UNTOUCHED_EVALUATION",
        "hard_stop_0_respected": True,
        "block3_block4_absent": True,
        "core_artifact_hashes": core_hashes,
    }
    write_json(CONSISTENCY, consistency_payload)

    manifest_paths = core_paths + [CONSISTENCY]
    manifest_payload = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_STAGE_B_PREPARATION_COMPLETE",
        "final_verdict": "STAGE_B_BLOCKED_NO_UNTOUCHED_EVALUATION",
        "starting_head": START_HEAD,
        "finalization_input_head": git("rev-parse", "HEAD"),
        "prerequisite_commits": PREFIT_COMMITS,
        "block0": block0["classification"],
        "gate1": geometry["gate1"]["branch"],
        "gate2": routers["gate2"],
        "pilot": pilot["pilot_status"],
        "cost": pilot["cost_projection"]["status"],
        "block3": "NOT_CREATED_HARD_STOP_0",
        "block4": "NOT_CREATED_HARD_STOP_0",
        "constraint_accounting": {
            "test_rows_ids_content_labels_prediction_vectors_per_sample_scores_accessed": False,
            "historical_test_aggregate_access_incident": True,
            "zero_test_access_confirmation_possible": False,
            "stage_b_executed": False,
            "qwen3_started": False,
            "llama_or_phase10_3_modified": False,
            "direction_layer_dose_or_endpoint_retuned": False,
            "model_or_tokenizer_downloaded": False,
        },
        "artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in manifest_paths
        },
    }
    if not all(
        LOWER_HEX_64.fullmatch(value)
        for value in manifest_payload["artifacts"].values()
    ):
        raise RuntimeError("manifest SHA256 format failure")
    write_json(MANIFEST, manifest_payload)

    hash_paths = manifest_paths + [MANIFEST]
    HASHES.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(ROOT)}"
            for path in hash_paths
        )
        + "\n",
        encoding="utf-8",
    )
    final_hash_checks = check_hash_list(HASHES)
    if not all(check["passed"] for check in final_hash_checks):
        raise RuntimeError("final hash list verification failed")

    print(
        json.dumps(
            {
                "verdict": "STAGE_B_BLOCKED_NO_UNTOUCHED_EVALUATION",
                "checks": f"{len(checks)}/{len(checks)}",
                "artifacts_hashed": len(hash_paths),
                "report_sha256": sha256_file(REPORT),
                "consistency_sha256": sha256_file(CONSISTENCY),
                "manifest_sha256": sha256_file(MANIFEST),
                "hash_list_sha256": sha256_file(HASHES),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
