#!/usr/bin/env python3
"""
Phase 10.1 V2 — population-typed validator and frozen scientific runner.

The two modes are deliberately isolated:
  --gate-only     runs only run_validity_gate(), writes the two authorized gate
                  artifacts, and exits before scientific dependencies are imported.
  --run-analysis  reruns the same gate and proceeds only after every check passes.

No model, GPU, new prediction, new random direction, test row, whole-dataset cache,
or LFS object is used. Attempt 1 is immutable and is never imported or executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = ROOT / "final" / "results" / "phase10_1_directionality"
OUT = BASE_OUT / "v2"
RAW_PATH = ROOT / "final" / "results" / "phase10_mctl" / "phase10_mctl_raw.json"
HIST_GATE_PATH = ROOT / "final" / "results" / "phase10_mctl" / "phase10_validity_gate.json"
ROUTED_PATH = ROOT / "final" / "results" / "phase9_mechanism_and_ood" / "phase9_5_per_sample_routed.json"
V1_PREREG_PATH = BASE_OUT / "PHASE10_1_PREREGISTRATION.md"
V1_AUDIT_PATH = BASE_OUT / "PHASE10_1_INPUT_AUDIT.md"
V1_HASHES_PATH = BASE_OUT / "PHASE10_1_HASHES.json"
ATTEMPT1_PATH = ROOT / "scripts" / "phase10_1_directionality_analysis.py"
ATTEMPT1_GATE_PATH = BASE_OUT / "phase10_1_validity_gate.json"
ATTEMPT1_RESULT_PATH = BASE_OUT / "PHASE10_1_RESULTS.md"
V2_PREREG_PATH = BASE_OUT / "PHASE10_1_PREREGISTRATION_V2.md"
V2_AUDIT_PATH = BASE_OUT / "PHASE10_1_VALIDATOR_POPULATION_AUDIT.md"
V2_ERRATUM_PATH = BASE_OUT / "PHASE10_1_VALIDATOR_ERRATUM.md"
V2_HASHES_PATH = BASE_OUT / "PHASE10_1_HASHES_V2.json"
EXECUTION_FREEZE_PATH = BASE_OUT / "PHASE10_1_EXECUTION_FREEZE_V2.json"
GATE_JSON_PATH = BASE_OUT / "phase10_1_v2_gate_only.json"
GATE_REPORT_PATH = BASE_OUT / "PHASE10_1_V2_GATE_ONLY_REPORT.md"
VAL_IDX_PATH = ROOT / "final" / "results" / "splits" / "val_idx.json"

EXPECTED_HASHES = {
    V1_PREREG_PATH: "33649a3929bec9a3124b1495375c810c3c0eaffcb90fde3884b8945f63455340",
    V1_AUDIT_PATH: "423f8c41044f2e10aef2562b36a2d197b2e6799fc25c615d68f55ee20627a231",
    V1_HASHES_PATH: "a6a8cb27693e11cd9f0e35096cb9e9f17c3def7750bf7c38ff4bf3ddccdb8b5d",
    ATTEMPT1_PATH: "da77c606e25ac00810414066ab6028932b8809bab94398b879b3443ed1cbb580",
    ATTEMPT1_GATE_PATH: "7e7584348b28eeae82bdead3eca840a8d3ced85f2634a5b64c8e0c7808efe7c7",
    ATTEMPT1_RESULT_PATH: "2a901aa889cc73cf48121ea4b54bb8c729e5a2d1db231600135fdfa77112b045",
    V2_PREREG_PATH: "895e24853fd57ce517381aa233e40cb643eb8610c201e53a6f993e3a504e63f8",
    V2_AUDIT_PATH: "6b704046712c42e4ae8aa6e702075643ca4a20bdb66f15231d4a052578786794",
    V2_ERRATUM_PATH: "e9372a308954dcf5b6d88fb772a4a2fbbae1eebd819be8cfa4b8d0bfa53ee4dd",
    V2_HASHES_PATH: "c69cc68a3203f1c789f7c0953bde20c0545a80af680b5ae4e86013489fff6380",
    RAW_PATH: "c9324d46a417e70e2244ec154e9aff507ae9bcc126212b3e188f8a968a7d5f5d",
    HIST_GATE_PATH: "0b0a3dea8500a345f066c009414aeb25b3c98b09cc55ea03dfad3c3eb5508747",
    ROUTED_PATH: "525a6832fd24ac9aa2e256ac32666b0af98e2ffdd0aea8501c8bdc8ef92012fd",
    VAL_IDX_PATH: "2d2f078a47ae7702b5ccceb23a86d9d4cf223d7b7cc0de4eb8090e809afa3610",
}
EXPECTED_NONPRIMARY = {178, 364, 962, 1264, 2061, 3034}
EXPECTED_ROUTED_ID_HASH = "a6aa9f7803673f4b4e0b016df4fb4f1ed6a657e69ff61cb6543108d2e2cf2bb8"
EXPECTED_ROUTED_OWN_ID_HASH = "e358d242f2c1f251e64c157896074105270df2da7b566752f88eb2413e8b8ba1"
EXPECTED_ROUTED_ROWS_HASH = "769428d2e9b108c7bc72e638640b652c47b253e5a0d0487c9e2c19aa4fc01234"
MODES = ("tool_call", "cannot_answer", "direct", "request_for_info")
SOURCE = "tool_call"
GOLD = "cannot_answer"
WRONG = ("direct", "request_for_info")
RHOS = (0.0, 1.0, 2.0, 3.0, 4.0)
N_RANDOM = 20
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 10101
MATCH_TOL = 0.10
MATCH_MIN_SAMPLES = 48
MATCH_MIN_DIRECTIONS = 15
NEAR_ZERO = 1e-6
SD_FLOOR = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DuplicateKeyError(ValueError):
    pass


def no_duplicate_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json_strict(path: Path):
    return json.loads(path.read_text(), object_pairs_hook=no_duplicate_object)


def add_check(checks, name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def reconstruct_historical_events(raw, cell_key, routed_ids, own_ids):
    frozen = raw["frozen_baseline"]
    per_sample = raw["cells"][cell_key]["per_sample"]
    post = {idx: per_sample[idx]["pred"] for idx in routed_ids}

    stay = [int(idx) for idx in own_ids if post[idx] == SOURCE]
    to_gold = [int(idx) for idx in own_ids if post[idx] == GOLD]
    to_other = [int(idx) for idx in own_ids if post[idx] in WRONG]
    moved = [int(idx) for idx in own_ids if post[idx] != SOURCE]
    fixed = [
        int(idx)
        for idx in routed_ids
        if frozen[idx]["baseline_pred"] != frozen[idx]["gold"]
        and post[idx] == frozen[idx]["gold"]
    ]
    damaged = [
        int(idx)
        for idx in routed_ids
        if frozen[idx]["baseline_pred"] == frozen[idx]["gold"]
        and post[idx] != frozen[idx]["gold"]
    ]
    wrong_map = {str(int(idx)): post[idx] for idx in own_ids if post[idx] in WRONG}
    moved_n = len(moved)
    aggregate = {
        "fixed": len(fixed),
        "broke": len(damaged),
        "net": len(fixed) - len(damaged),
        "routed_own_before": len(own_ids),
        "routed_own_after": len(stay),
        "own_moved": moved_n,
        "own_to_gold": len(to_gold),
        "own_to_other_wrong": len(to_other),
        "target_gain": len(to_gold) - len(to_other),
        "target_hit_rate": round(len(to_gold) / moved_n, 4) if moved_n else None,
        "redistribution_rate": round(len(to_other) / moved_n, 4) if moved_n else None,
        "damage_on_correct": len(damaged),
    }
    return {
        "cell": cell_key,
        "id_to_post_prediction": {str(int(idx)): post[idx] for idx in routed_ids},
        "routed_own_stay_source_ids": stay,
        "routed_own_moved_ids": moved,
        "routed_own_to_gold_ids": to_gold,
        "routed_own_to_other_wrong": wrong_map,
        "fixed_ids": fixed,
        "damaged_baseline_correct_ids": damaged,
        "aggregate": aggregate,
    }


def run_validity_gate():
    checks = []
    failures = []
    inputs_read = [relative(path) for path in EXPECTED_HASHES]
    inputs_read.append(relative(EXECUTION_FREEZE_PATH))
    inputs_read = list(dict.fromkeys(inputs_read))

    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path) if path.exists() else "MISSING"
        ok = actual == expected
        add_check(checks, f"sha256:{relative(path)}", ok, f"expected={expected}; actual={actual}")
        if not ok:
            failures.append(f"hash mismatch: {relative(path)}")

    script_path = Path(__file__).resolve()
    try:
        freeze = load_json_strict(EXECUTION_FREEZE_PATH)
        freeze_ok = (
            freeze.get("repository", {}).get("root") == str(ROOT)
            and freeze.get("repository", {}).get("branch") == "exp/sakiko-followup-archive"
            and freeze.get("v2_script", {}).get("path") == relative(script_path)
            and freeze.get("v2_script", {}).get("sha256") == sha256(script_path)
            and set(freeze.get("allowed_modes", [])) == {"--gate-only", "--run-analysis"}
            and freeze.get("scientific_endpoints_computed") is False
            and freeze.get("full_validation_absolute_fields", {}).get("full_val_own_before", {}).get("status")
            == "UNAVAILABLE_NOT_GATEABLE"
            and freeze.get("full_validation_absolute_fields", {}).get("full_val_own_after", {}).get("status")
            == "UNAVAILABLE_NOT_GATEABLE"
        )
        add_check(checks, "execution_freeze_manifest", freeze_ok, "root, branch, script SHA, modes, and unavailable fields")
        if not freeze_ok:
            failures.append("execution_freeze_manifest")
    except Exception as exc:
        freeze = None
        add_check(checks, "execution_freeze_manifest", False, repr(exc))
        failures.append("execution_freeze_manifest")

    forbidden_input = any(
        "phase8_cache" in path or "/test" in path or "test_idx" in path
        for path in inputs_read
    )
    add_check(
        checks,
        "split_firewall_input_paths",
        not forbidden_input,
        json.dumps(inputs_read),
    )
    if forbidden_input:
        failures.append("split_firewall_input_paths")

    try:
        raw = load_json_strict(RAW_PATH)
        routed_prior = load_json_strict(ROUTED_PATH)
        val_idx = load_json_strict(VAL_IDX_PATH)
        hist_gate = load_json_strict(HIST_GATE_PATH)
        add_check(checks, "strict_json_no_duplicate_keys", True, "all JSON inputs parsed strictly")
    except Exception as exc:
        add_check(checks, "strict_json_no_duplicate_keys", False, repr(exc))
        failures.append(f"strict JSON parse failed: {exc}")
        return None, checks, failures, {
            "inputs_read": inputs_read,
            "historical_event_reconstruction": {},
        }

    frozen = raw.get("frozen_baseline", {})
    routed_ids = list(frozen.keys())
    routed_set = set(routed_ids)
    own_ids = [
        idx
        for idx in routed_ids
        if frozen[idx].get("is_own_channel_err") is True
        and frozen[idx].get("gold") == GOLD
        and frozen[idx].get("baseline_pred") == SOURCE
    ]
    own_set = set(own_ids)
    nonprimary = routed_set - own_set

    population_checks = [
        ("routed_population_n", len(routed_ids) == 74, f"n={len(routed_ids)}"),
        ("routed_own_before", len(own_ids) == 68, f"n={len(own_ids)}"),
        (
            "primary_strict_subset",
            own_set < routed_set and len(nonprimary) == 6,
            f"primary={len(own_set)} routed={len(routed_set)} extra={len(nonprimary)}",
        ),
        (
            "sensitivity_only_population_n",
            len(nonprimary) == 6,
            f"n={len(nonprimary)}",
        ),
        (
            "extra_six_ids",
            {int(idx) for idx in nonprimary} == EXPECTED_NONPRIMARY,
            f"actual={sorted(map(int, nonprimary))}",
        ),
        (
            "historical_route_predicate",
            all(
                frozen[idx].get("baseline_pred") == SOURCE
                and float(frozen[idx].get("router_score")) >= 0.4
                for idx in routed_ids
            ),
            'baseline_pred == "tool_call" AND router_score >= 0.4',
        ),
    ]
    for name, ok, detail in population_checks:
        add_check(checks, name, ok, detail)
        if not ok:
            failures.append(name)

    extra_records = [frozen[idx] for idx in nonprimary]
    extra_identity_ok = (
        sum(r["gold"] == "tool_call" and r["baseline_pred"] == "tool_call" for r in extra_records) == 5
        and sum(
            r["gold"] == "request_for_info" and r["baseline_pred"] == "tool_call"
            for r in extra_records
        ) == 1
    )
    add_check(checks, "extra_six_channel_identity", extra_identity_ok, "five TC→TC and one RFI→TC")
    if not extra_identity_ok:
        failures.append("extra_six_channel_identity")

    prior_by_id = {str(int(row["idx"])): row for row in routed_prior}
    provenance_fields = ("uuid", "gold", "baseline_pred", "is_own_channel_err", "router_score")
    prior_match = (
        len(routed_prior) == 74
        and list(prior_by_id) == routed_ids
        and all(
            all(prior_by_id[idx].get(field) == frozen[idx].get(field) for field in provenance_fields)
            for idx in routed_ids
        )
    )
    add_check(checks, "independent_routed_provenance_match", prior_match, "Phase 9.5 vs Phase 10 exact 74-row match")
    if not prior_match:
        failures.append("independent_routed_provenance_match")

    ordered_ids = [int(row["idx"]) for row in routed_prior]
    sorted_own_ids = sorted(int(row["idx"]) for row in routed_prior if row["is_own_channel_err"])
    provenance_rows = [
        {key: row[key] for key in ("idx", "uuid", "gold", "baseline_pred", "router_score", "is_own_channel_err")}
        for row in sorted(routed_prior, key=lambda item: int(item["idx"]))
    ]
    signatures = {
        "ordered_routed_ids_sha256": canonical_sha256(ordered_ids),
        "sorted_routed_own_ids_sha256": canonical_sha256(sorted_own_ids),
        "sorted_routed_provenance_rows_sha256": canonical_sha256(provenance_rows),
    }
    signature_ok = (
        signatures["ordered_routed_ids_sha256"] == EXPECTED_ROUTED_ID_HASH
        and signatures["sorted_routed_own_ids_sha256"] == EXPECTED_ROUTED_OWN_ID_HASH
        and signatures["sorted_routed_provenance_rows_sha256"] == EXPECTED_ROUTED_ROWS_HASH
    )
    add_check(checks, "routed_population_signatures", signature_ok, json.dumps(signatures, sort_keys=True))
    if not signature_ok:
        failures.append("routed_population_signatures")

    required_cells = {"zero"}
    for direction in ("deployed", "M1"):
        for kind in ("real", "reverse"):
            required_cells.update(f"{direction}|{kind}|{rho:.1f}" for rho in RHOS[1:])
    required_cells.update(f"random{k}|{rho:.1f}" for k in range(N_RANDOM) for rho in RHOS[1:])
    actual_cells = set(raw.get("cells", {}))
    cells_ok = actual_cells == required_cells and len(actual_cells) == 97
    add_check(
        checks,
        "required_97_cells",
        cells_ok,
        f"actual={len(actual_cells)} missing={sorted(required_cells-actual_cells)} extra={sorted(actual_cells-required_cells)}",
    )
    if not cells_ok:
        failures.append("required_97_cells")

    baseline_order = tuple(routed_ids)
    id_unique = len({int(idx) for idx in routed_ids}) == 74
    uuid_unique = len({frozen[idx]["uuid"] for idx in routed_ids}) == 74
    baseline_rows_ok = all(
        int(frozen[idx]["idx"]) == int(idx)
        and set(frozen[idx]["baseline_logp"]) == set(MODES)
        and all(math.isfinite(float(value)) for value in frozen[idx]["baseline_logp"].values())
        for idx in routed_ids
    )
    order_ok = record_count_ok = mode_ok = finite_ok = argmax_ok = True
    for cell in raw.get("cells", {}).values():
        per_sample = cell.get("per_sample", {})
        order_ok &= tuple(per_sample.keys()) == baseline_order
        record_count_ok &= len(per_sample) == 74
        for rec in per_sample.values():
            logp = rec.get("logp", {})
            mode_ok &= set(logp) == set(MODES) and rec.get("pred") in MODES
            finite_ok &= set(logp) == set(MODES) and all(math.isfinite(float(value)) for value in logp.values())
            if set(logp) == set(MODES):
                argmax_ok &= rec.get("pred") == max(MODES, key=lambda mode: float(logp[mode]))
    for name, ok, detail in [
        ("sample_id_uuid_uniqueness", id_unique and uuid_unique, f"idx={id_unique}; uuid={uuid_unique}"),
        ("frozen_baseline_rows", baseline_rows_ok, "idx association, four finite baseline scores"),
        ("cell_record_count", record_count_ok, "74 per cell"),
        ("aligned_sample_order", order_ok, "same ordered IDs in all cells"),
        ("four_response_modes", mode_ok, str(MODES)),
        ("finite_average_logprob", finite_ok, "all cell scores finite"),
        ("prediction_argmax_consistency", argmax_ok, "stored prediction equals four-score argmax"),
    ]:
        add_check(checks, name, ok, detail)
        if not ok:
            failures.append(name)

    zero = raw.get("cells", {}).get("zero", {}).get("per_sample", {})
    zero_pred_ok = all(zero.get(idx, {}).get("pred") == frozen[idx]["baseline_pred"] for idx in routed_ids)
    zero_score_ok = all(zero.get(idx, {}).get("logp") == frozen[idx]["baseline_logp"] for idx in routed_ids)
    add_check(checks, "zero_baseline_prediction_vector", zero_pred_ok, "74/74 exact")
    add_check(checks, "zero_baseline_score_vectors", zero_score_ok, "74 four-score vectors exact")
    if not zero_pred_ok:
        failures.append("zero_baseline_prediction_vector")
    if not zero_score_ok:
        failures.append("zero_baseline_score_vectors")

    val_set = set(map(int, val_idx))
    routed_in_val = set(map(int, routed_ids)) <= val_set
    add_check(checks, "all_routed_in_validation", routed_in_val, f"routed=74 validation_n={len(val_set)}")
    if not routed_in_val:
        failures.append("all_routed_in_validation")

    direct_ok = "direct" in MODES and "direct_answer" not in MODES
    add_check(checks, "label_compatibility", direct_ok, "direct_answer -> direct; no fifth stored class")
    if not direct_ok:
        failures.append("label_compatibility")

    historical = {}
    hist_rows = {(row.get("direction"), float(row.get("rho"))): row for row in hist_gate.get("rows", [])}
    historical_ok = hist_gate.get("passed") is True and len(hist_rows) == 6
    comparable = (
        "fixed",
        "broke",
        "net",
        "own_moved",
        "own_to_gold",
        "own_to_other_wrong",
        "target_gain",
        "target_hit_rate",
        "redistribution_rate",
        "damage_on_correct",
    )
    for direction in ("deployed", "M1"):
        for rho in (1.0, 2.0, 4.0):
            key = f"{direction}|real|{rho:.1f}"
            event = reconstruct_historical_events(raw, key, routed_ids, own_ids)
            aggregate = event["aggregate"]
            stored = raw["cells"][key]["agg"]
            row = hist_rows.get((direction, rho), {})
            partition_ok = (
                set(event["routed_own_stay_source_ids"])
                | set(event["routed_own_to_gold_ids"])
                | set(map(int, event["routed_own_to_other_wrong"]))
            ) == set(map(int, own_ids))
            partition_ok &= not (
                set(event["routed_own_stay_source_ids"])
                & set(event["routed_own_moved_ids"])
            )
            stored_ok = all(stored.get(field) == aggregate.get(field) for field in comparable)
            historical_row_ok = row.get("match") is True and all(
                stored.get(field) == value for field, value in row.get("got", {}).items()
            )
            cell_ok = partition_ok and stored_ok and historical_row_ok
            historical_ok &= cell_ok
            event["partition_passed"] = partition_ok
            event["stored_aggregate_comparison"] = {
                field: {
                    "reconstructed": aggregate.get(field),
                    "stored": stored.get(field),
                    "match": aggregate.get(field) == stored.get(field),
                }
                for field in comparable
            }
            event["historical_gate_row_match"] = historical_row_ok
            historical[key] = event
    add_check(checks, "historical_six_cell_event_reconstruction", historical_ok, "six exact ID-level partitions and aggregates")
    if not historical_ok:
        failures.append("historical_six_cell_event_reconstruction")

    deployed_after = historical.get("deployed|real|4.0", {}).get("aggregate", {}).get("routed_own_after")
    m1_after = historical.get("M1|real|4.0", {}).get("aggregate", {}).get("routed_own_after")
    explicit_counts_ok = deployed_after == 33 and m1_after == 54
    add_check(
        checks,
        "explicit_routed_own_after",
        explicit_counts_ok,
        f"deployed={deployed_after}; M1={m1_after}",
    )
    if not explicit_counts_ok:
        failures.append("explicit_routed_own_after")

    context = {
        "inputs_read": inputs_read,
        "population": {
            "routed_population": 74,
            "routed_own_before": 68,
            "sensitivity_only_population": 6,
            "extra_six_ids": sorted(EXPECTED_NONPRIMARY),
            **signatures,
        },
        "routed_counts": {
            "deployed_routed_own_after_rho4": deployed_after,
            "M1_routed_own_after_rho4": m1_after,
        },
        "full_validation_absolute_fields": {
            "full_val_own_before": {"status": "UNAVAILABLE_NOT_GATEABLE"},
            "full_val_own_after": {"status": "UNAVAILABLE_NOT_GATEABLE"},
        },
        "label_compatibility": {
            "canonical": "direct",
            "alias": "direct_answer",
            "mapping": "direct_answer -> direct",
            "creates_fifth_class": False,
        },
        "split_firewall": {
            "test_rows_read": False,
            "whole_dataset_cache_opened": False,
            "test_derived_ids_or_aggregates_used": False,
        },
        "historical_event_reconstruction": historical,
    }
    return raw, checks, failures, context


def write_gate_only(checks, failures, context, elapsed):
    status = "VOID" if failures else "READY_FOR_ANALYSIS_APPROVAL"
    result = {
        "status": status,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        **context,
        "gate_only": True,
        "scientific_endpoints_computed": False,
        "scientific_dependencies_loaded": False,
        "run_analysis_executed": False,
        "elapsed_cpu_wall_seconds": elapsed,
    }
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    GATE_JSON_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    report_lines = [
        "# Phase 10.1 V2 — Gate-Only Report",
        "",
        f"**Conclusion:** **{status}**",
        "**Scientific endpoints computed:** false",
        "**Run-analysis executed:** false",
        "",
        "## Checks",
        "",
    ]
    report_lines.extend(
        f"- {'PASS' if check['passed'] else 'FAIL'} — `{check['check']}`: {check['detail']}"
        for check in checks
    )
    if failures:
        report_lines.extend(["", "## Failures", ""] + [f"- {failure}" for failure in failures])
    report_lines.extend(
        [
            "",
            "## Population status",
            "",
            "- `routed_own_before = 68`",
            f"- `deployed routed_own_after = {context.get('routed_counts', {}).get('deployed_routed_own_after_rho4')}`",
            f"- `M1 routed_own_after = {context.get('routed_counts', {}).get('M1_routed_own_after_rho4')}`",
            "- `full_val_own_before = UNAVAILABLE_NOT_GATEABLE`",
            "- `full_val_own_after = UNAVAILABLE_NOT_GATEABLE`",
            "",
            "The gate stopped before every scientific endpoint function.",
        ]
    )
    GATE_REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    print(status)


def load_scientific_dependencies():
    global np, plt, LogisticRegression, log_loss, StandardScaler
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    import numpy as np_module
    from sklearn.linear_model import LogisticRegression as LogisticRegressionClass
    from sklearn.metrics import log_loss as log_loss_function
    from sklearn.preprocessing import StandardScaler as StandardScalerClass

    np = np_module
    plt = plt_module
    LogisticRegression = LogisticRegressionClass
    log_loss = log_loss_function
    StandardScaler = StandardScalerClass


def write_void(checks, failures, elapsed):
    OUT.mkdir(parents=True, exist_ok=True)
    gate = {
        "status": "VOID",
        "passed": False,
        "failures": failures,
        "checks": checks,
        "scientific_statistics_computed": False,
        "elapsed_cpu_wall_seconds": elapsed,
    }
    (OUT / "phase10_1_validity_gate.json").write_text(json.dumps(gate, indent=2))
    text = (
        "# Phase 10.1 Results — VOID\n\n"
        "The pre-analysis validity gate failed. No scientific statistic or mechanism verdict was "
        "computed.\n\n## Failures\n\n"
        + "\n".join(f"- {x}" for x in failures)
        + "\n"
    )
    (OUT / "PHASE10_1_RESULTS.md").write_text(text)


def st_value(logp, base):
    source_terms = [
        (float(logp[c]) - float(logp[SOURCE]))
        - (float(base[c]) - float(base[SOURCE]))
        for c in MODES
        if c != SOURCE
    ]
    target_terms = [
        (float(logp[GOLD]) - float(logp[w]))
        - (float(base[GOLD]) - float(base[w]))
        for w in WRONG
    ]
    return float(np.mean(source_terms)), float(np.mean(target_terms))


def build_curve(raw, cell_prefix, routed_ids):
    n = len(routed_ids)
    s = np.zeros((n, len(RHOS)), dtype=float)
    t = np.zeros((n, len(RHOS)), dtype=float)
    zero = raw["cells"]["zero"]["per_sample"]
    for j, rho in enumerate(RHOS[1:], start=1):
        cell = raw["cells"][f"{cell_prefix}|{rho:.1f}"]["per_sample"]
        for a, idx in enumerate(routed_ids):
            s[a, j], t[a, j] = st_value(cell[idx]["logp"], zero[idx]["logp"])
    return {"S": s, "T": t}


def auc(values):
    widths = np.diff(RHOS)
    return np.sum(0.5 * (values[:, :-1] + values[:, 1:]) * widths, axis=1) / 4.0


def percentile_ci(values, bootstrap_indices):
    draws = np.mean(values[bootstrap_indices], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def empirical_summary(real, random_values):
    rv = np.asarray(random_values, dtype=float)
    n_ge = int(np.sum(rv >= real))
    n_below = int(np.sum(rv < real))
    sd = float(np.std(rv, ddof=1))
    return {
        "real": float(real),
        "random_values": [float(x) for x in rv],
        "rank_desc_ties_against_real": 1 + n_ge,
        "n_random_ge_real": n_ge,
        "p_add_one": float((1 + n_ge) / 21),
        "p_min": float(1 / 21),
        "random_mean": float(np.mean(rv)),
        "random_sample_sd": sd,
        "random_min": float(np.min(rv)),
        "random_max": float(np.max(rv)),
        "real_minus_random_mean": float(real - np.mean(rv)),
        "strict_below_rank_fraction": float((1 + n_below) / 21),
        "exceeds_all_random": bool(real > np.max(rv)),
    }


def population_analysis(position_ids, curves, random_curves, boot_indices):
    dep_s_auc = auc(curves["deployed_real"]["S"])[position_ids]
    dep_t_auc = auc(curves["deployed_real"]["T"])[position_ids]
    rev_s_auc = auc(curves["deployed_reverse"]["S"])[position_ids]
    rev_t_auc = auc(curves["deployed_reverse"]["T"])[position_ids]
    random_s = [float(np.mean(auc(c["S"])[position_ids])) for c in random_curves]
    random_t = [float(np.mean(auc(c["T"])[position_ids])) for c in random_curves]
    s_summary = empirical_summary(float(np.mean(dep_s_auc)), random_s)
    t_summary = empirical_summary(float(np.mean(dep_t_auc)), random_t)
    s_summary["sample_bootstrap_ci_95"] = percentile_ci(dep_s_auc, boot_indices)
    t_summary["sample_bootstrap_ci_95"] = percentile_ci(dep_t_auc, boot_indices)
    reverse = {
        "mean_real_minus_reverse_S_AUC": float(np.mean(dep_s_auc - rev_s_auc)),
        "S_paired_bootstrap_ci_95": percentile_ci(dep_s_auc - rev_s_auc, boot_indices),
        "mean_real_minus_reverse_T_AUC": float(np.mean(dep_t_auc - rev_t_auc)),
        "T_paired_bootstrap_ci_95": percentile_ci(dep_t_auc - rev_t_auc, boot_indices),
        "reverse_mean_S_AUC": float(np.mean(rev_s_auc)),
        "reverse_mean_T_AUC": float(np.mean(rev_t_auc)),
    }
    if s_summary["random_sample_sd"] < SD_FLOOR or t_summary["random_sample_sd"] < SD_FLOOR:
        asym = {
            "status": "UNAVAILABLE",
            "reason": f"random SD below {SD_FLOOR}",
            "Z_S": None,
            "Z_T": None,
            "A_std": None,
            "R_S": s_summary["strict_below_rank_fraction"],
            "R_T": t_summary["strict_below_rank_fraction"],
        }
    else:
        z_s = (
            s_summary["real"] - s_summary["random_mean"]
        ) / s_summary["random_sample_sd"]
        z_t = (
            t_summary["real"] - t_summary["random_mean"]
        ) / t_summary["random_sample_sd"]
        asym = {
            "status": "AVAILABLE_EXPLORATORY",
            "Z_S": float(z_s),
            "Z_T": float(z_t),
            "A_std": float(z_s - z_t),
            "R_S": s_summary["strict_below_rank_fraction"],
            "R_T": t_summary["strict_below_rank_fraction"],
        }
    return {
        "n": len(position_ids),
        "S": s_summary,
        "T": t_summary,
        "real_vs_reverse": reverse,
        "standardized_asymmetry": asym,
    }


def transition_rows_and_metrics(raw, routed_ids, primary_ids, population_ids, direction):
    frozen = raw["frozen_baseline"]
    cell = raw["cells"][f"{direction}|real|4.0"]["per_sample"]
    matrix = Counter((frozen[i]["baseline_pred"], cell[i]["pred"]) for i in population_ids)
    rows = []
    for before in MODES:
        for after in MODES:
            rows.append(
                {
                    "direction": direction,
                    "population": "primary68" if len(population_ids) == 68 else "routed74",
                    "rho": 4,
                    "baseline_mode": before,
                    "intervention_mode": after,
                    "count": int(matrix[(before, after)]),
                }
            )
    primary_here = [i for i in primary_ids if i in set(population_ids)]
    source_exits = sum(cell[i]["pred"] != SOURCE for i in primary_here)
    target_arrivals = sum(cell[i]["pred"] == GOLD for i in primary_here)
    wrong_redistribution = sum(cell[i]["pred"] in WRONG for i in primary_here)
    nonprimary = [i for i in population_ids if i not in set(primary_ids)]
    collateral_changed = sum(cell[i]["pred"] != frozen[i]["baseline_pred"] for i in nonprimary)
    collateral_broke = sum(
        frozen[i]["baseline_pred"] == frozen[i]["gold"] and cell[i]["pred"] != frozen[i]["gold"]
        for i in nonprimary
    )
    collateral_fixed = sum(
        frozen[i]["baseline_pred"] != frozen[i]["gold"] and cell[i]["pred"] == frozen[i]["gold"]
        for i in nonprimary
    )
    collateral_wrong_to_wrong = sum(
        frozen[i]["baseline_pred"] != frozen[i]["gold"]
        and cell[i]["pred"] != frozen[i]["gold"]
        and cell[i]["pred"] != frozen[i]["baseline_pred"]
        for i in nonprimary
    )
    metrics = {
        "population": "primary68" if len(population_ids) == 68 else "routed74",
        "primary_channel_denominator": len(primary_here),
        "source_exits": int(source_exits),
        "target_arrivals": int(target_arrivals),
        "target_hit": float(target_arrivals / source_exits) if source_exits else None,
        "target_hit_exact_denominator": int(source_exits),
        "wrong_to_wrong_redistribution": int(wrong_redistribution),
        "Target_Gain": int(target_arrivals - wrong_redistribution),
        "collateral_population_n": len(nonprimary),
        "collateral_changed": int(collateral_changed),
        "collateral_fixed": int(collateral_fixed),
        "collateral_broke": int(collateral_broke),
        "collateral_wrong_to_wrong": int(collateral_wrong_to_wrong),
    }
    return rows, metrics


def matching_analysis(raw, routed_ids, primary_positions, curves, random_curves):
    real_s4 = curves["deployed_real"]["S"][primary_positions, 4]
    diagnostics = []
    selections = []
    for k, rc in enumerate(random_curves):
        rhos_selected = []
        distances = []
        positions_matched = []
        for local_pos, routed_pos in enumerate(primary_positions):
            candidates = []
            for j, rho in enumerate(RHOS):
                distance = abs(float(rc["S"][routed_pos, j]) - float(real_s4[local_pos]))
                candidates.append((distance, abs(float(rho) - 4.0), float(rho), j))
            distance, _, rho, j = min(candidates)
            if distance <= MATCH_TOL:
                positions_matched.append(int(routed_pos))
                rhos_selected.append(float(rho))
                distances.append(float(distance))
        n = len(positions_matched)
        mean_mismatch = float(np.mean(distances)) if distances else None
        qualified = (
            n >= MATCH_MIN_SAMPLES
            and mean_mismatch is not None
            and mean_mismatch < MATCH_TOL
        )
        diagnostics.append(
            {
                "random_direction": f"random{k}",
                "seed": 5000 + k,
                "matched_n": n,
                "coverage": float(n / 68),
                "mean_absolute_S_mismatch": mean_mismatch,
                "qualified": qualified,
            }
        )
        selections.append((positions_matched, rhos_selected))

    n_qualified = sum(d["qualified"] for d in diagnostics)
    feasible = n_qualified >= MATCH_MIN_DIRECTIONS
    result = {
        "status": "FEASIBLE" if feasible else "MATCHING INFEASIBLE",
        "tolerance": MATCH_TOL,
        "interpolation": False,
        "minimum_directions": MATCH_MIN_DIRECTIONS,
        "minimum_samples_per_direction": MATCH_MIN_SAMPLES,
        "qualified_directions": n_qualified,
        "diagnostics": diagnostics,
    }
    if not feasible:
        result["matched_outcomes_computed"] = False
        return result

    # Only after the S-only feasibility gate passes may T and destinations be read.
    summaries = []
    frozen = raw["frozen_baseline"]
    for k, (positions, selected_rhos) in enumerate(selections):
        if not diagnostics[k]["qualified"]:
            continue
        t_diffs = []
        real_gold = 0
        random_gold = 0
        real_dest = Counter()
        random_dest = Counter()
        for routed_pos, rho in zip(positions, selected_rhos):
            idx = routed_ids[routed_pos]
            j = int(rho)
            t_diffs.append(
                float(curves["deployed_real"]["T"][routed_pos, 4])
                - float(random_curves[k]["T"][routed_pos, j])
            )
            real_pred = raw["cells"]["deployed|real|4.0"]["per_sample"][idx]["pred"]
            if rho == 0:
                random_pred = raw["cells"]["zero"]["per_sample"][idx]["pred"]
            else:
                random_pred = raw["cells"][f"random{k}|{rho:.1f}"]["per_sample"][idx]["pred"]
            gold = frozen[idx]["gold"]
            real_gold += int(real_pred == gold)
            random_gold += int(random_pred == gold)
            real_dest[real_pred] += 1
            random_dest[random_pred] += 1
        n = len(positions)
        summaries.append(
            {
                "random_direction": f"random{k}",
                "matched_n": n,
                "mean_real_minus_random_T": float(np.mean(t_diffs)),
                "real_gold_arrivals": real_gold,
                "random_gold_arrivals": random_gold,
                "gold_arrival_rate_contrast": float((real_gold - random_gold) / n),
                "real_destination_counts": {m: int(real_dest[m]) for m in MODES},
                "random_destination_counts": {m: int(random_dest[m]) for m in MODES},
            }
        )
    result["matched_outcomes_computed"] = True
    result["direction_summaries"] = summaries
    return result


def heterogeneity_summary(s_auc, t_auc):
    ambiguous = (np.abs(s_auc) <= NEAR_ZERO) | (np.abs(t_auc) <= NEAR_ZERO)
    labels = np.full(len(s_auc), "ambiguous", dtype=object)
    labels[(s_auc > NEAR_ZERO) & (t_auc > NEAR_ZERO)] = "S+_T+"
    labels[(s_auc > NEAR_ZERO) & (t_auc < -NEAR_ZERO)] = "S+_T-"
    labels[(s_auc < -NEAR_ZERO) & (t_auc > NEAR_ZERO)] = "S-_T+"
    labels[(s_auc < -NEAR_ZERO) & (t_auc < -NEAR_ZERO)] = "S-_T-"
    counts = Counter(labels)
    nonambiguous = len(labels) - counts["ambiguous"]
    corr = float(np.corrcoef(s_auc, t_auc)[0, 1]) if np.std(s_auc) and np.std(t_auc) else None
    return {
        "near_zero_threshold": NEAR_ZERO,
        "ambiguous_n": int(counts["ambiguous"]),
        "nonambiguous_n": int(nonambiguous),
        "quadrant_counts": {q: int(counts[q]) for q in ("S+_T+", "S+_T-", "S-_T+", "S-_T-")},
        "quadrant_proportions_among_nonambiguous": {
            q: (float(counts[q] / nonambiguous) if nonambiguous else None)
            for q in ("S+_T+", "S+_T-", "S-_T+", "S-_T-")
        },
        "S_AUC": {
            "mean": float(np.mean(s_auc)),
            "median": float(np.median(s_auc)),
            "min": float(np.min(s_auc)),
            "max": float(np.max(s_auc)),
            "sample_sd": float(np.std(s_auc, ddof=1)),
        },
        "T_AUC": {
            "mean": float(np.mean(t_auc)),
            "median": float(np.median(t_auc)),
            "min": float(np.min(t_auc)),
            "max": float(np.max(t_auc)),
            "sample_sd": float(np.std(t_auc, ddof=1)),
        },
        "pearson_S_T": corr,
        "labels": labels.tolist(),
    }


def fixed_group_folds(groups, n_folds=5, seed=10101):
    unique = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = unique[rng.permutation(len(unique))]
    fold_of = {group: i % n_folds for i, group in enumerate(shuffled)}
    return np.asarray([fold_of[g] for g in groups], dtype=int)


def competitor_structure(raw, routed_ids, primary_positions):
    frozen = raw["frozen_baseline"]
    ids = [routed_ids[p] for p in primary_positions]
    x = np.asarray(
        [
            [
                float(frozen[i]["baseline_logp"][c])
                - float(frozen[i]["baseline_logp"][SOURCE])
                for c in (GOLD, "direct", "request_for_info")
            ]
            for i in ids
        ],
        dtype=float,
    )
    y = np.asarray(
        [raw["cells"]["deployed|real|4.0"]["per_sample"][i]["pred"] for i in ids],
        dtype=object,
    )
    groups = np.asarray([frozen[i]["uuid"] for i in ids], dtype=object)
    folds = fixed_group_folds(groups)
    classes = np.asarray(sorted(set(y)), dtype=object)

    def evaluate(labels):
        model_prob = np.zeros((len(labels), len(classes)), dtype=float)
        base_prob = np.zeros((len(labels), len(classes)), dtype=float)
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            if set(labels[train]) != set(classes):
                return None
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train])
            x_test = scaler.transform(x[test])
            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                fit_intercept=True,
                solver="lbfgs",
                max_iter=2000,
                multi_class="multinomial",
            )
            clf.fit(x_train, labels[train])
            pred = clf.predict_proba(x_test)
            for j, cls in enumerate(clf.classes_):
                model_prob[test, int(np.where(classes == cls)[0][0])] = pred[:, j]
            counts = Counter(labels[train])
            freq = np.asarray([counts[c] / int(np.sum(train)) for c in classes], dtype=float)
            base_prob[test, :] = freq
        model_loss = float(log_loss(labels, np.clip(model_prob, 1e-15, 1.0), labels=classes))
        baseline_loss = float(log_loss(labels, np.clip(base_prob, 1e-15, 1.0), labels=classes))
        return {
            "model_oos_log_loss": model_loss,
            "class_frequency_oos_log_loss": baseline_loss,
            "log_loss_improvement": baseline_loss - model_loss,
        }

    observed = evaluate(y)
    if observed is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "at least one training fold lacks an observed outcome class",
            "classes": classes.tolist(),
            "class_counts": dict(Counter(y)),
        }

    rng = np.random.Generator(np.random.PCG64(10101))
    null = []
    invalid = 0
    for _ in range(1000):
        permuted = y[rng.permutation(len(y))]
        score = evaluate(permuted)
        if score is None:
            invalid += 1
        else:
            null.append(score["log_loss_improvement"])
    result = {
        "status": "AVAILABLE",
        "classes": classes.tolist(),
        "class_counts": {str(k): int(v) for k, v in Counter(y).items()},
        "fold_sizes": [int(np.sum(folds == f)) for f in range(5)],
        **observed,
        "permutations_requested": 1000,
        "permutations_valid": len(null),
        "permutations_invalid": invalid,
    }
    if invalid:
        result.update(
            {
                "permutation_null_status": "UNAVAILABLE",
                "permutation_null_reason": "one or more permuted training folds lacked an outcome class",
            }
        )
    else:
        null_arr = np.asarray(null)
        result.update(
            {
                "permutation_null_status": "AVAILABLE",
                "permutation_improvement_mean": float(np.mean(null_arr)),
                "permutation_improvement_sample_sd": float(np.std(null_arr, ddof=1)),
                "permutation_improvement_min": float(np.min(null_arr)),
                "permutation_improvement_max": float(np.max(null_arr)),
                "permutation_n_ge_observed": int(np.sum(null_arr >= observed["log_loss_improvement"])),
                "permutation_add_one_p": float(
                    (1 + np.sum(null_arr >= observed["log_loss_improvement"])) / 1001
                ),
            }
        )
    return result


def m1_consistency(curves, primary_positions):
    result = {}
    for metric in ("S", "T"):
        deployed_mean = np.mean(curves["deployed_real"][metric][primary_positions], axis=0)
        m1_mean = np.mean(curves["M1_real"][metric][primary_positions], axis=0)
        deployed_auc = auc(curves["deployed_real"][metric])[primary_positions]
        m1_auc = auc(curves["M1_real"][metric])[primary_positions]
        sign_agreement = [
            bool(np.sign(deployed_mean[j]) == np.sign(m1_mean[j])) for j in range(1, 5)
        ]
        corr = (
            float(np.corrcoef(deployed_mean[1:], m1_mean[1:])[0, 1])
            if np.std(deployed_mean[1:]) and np.std(m1_mean[1:])
            else None
        )
        result[metric] = {
            "deployed_mean_trajectory": [float(x) for x in deployed_mean],
            "M1_mean_trajectory": [float(x) for x in m1_mean],
            "dose_sign_agreement": sign_agreement,
            "dose_sign_agreement_count": int(sum(sign_agreement)),
            "trajectory_pearson_r": corr,
            "per_sample_AUC_sign_agreement": float(
                np.mean(np.sign(deployed_auc) == np.sign(m1_auc))
            ),
            "deployed_mean_AUC": float(np.mean(deployed_auc)),
            "M1_mean_AUC": float(np.mean(m1_auc)),
        }
    return result


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(curves, random_curves, primary_positions, primary_result, heterogeneity):
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    random_s = np.asarray(primary_result["S"]["random_values"])
    real_s = primary_result["S"]["real"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.scatter(np.arange(20), random_s, color="#57606a", s=36, label="matched random")
    ax.axhline(real_s, color="#d1242f", linewidth=2, label="deployed real")
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels([str(i) for i in range(20)], fontsize=8)
    ax.set_xlabel("Frozen random direction k (seed 5000+k)")
    ax.set_ylabel("Mean normalized S-AUC (68 samples)")
    ax.set_title("Phase 10.1 direction-level source-displacement null")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig_phase10_1_s_null.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric in zip(axes, ("S", "T")):
        for key, label, color, ls in [
            ("deployed_real", "deployed real", "#d1242f", "-"),
            ("deployed_reverse", "deployed reverse", "#0969da", "--"),
            ("M1_real", "M1 real", "#8250df", "-."),
        ]:
            mean_curve = np.mean(curves[key][metric][primary_positions], axis=0)
            ax.plot(RHOS, mean_curve, marker="o", color=color, linestyle=ls, label=label)
        rand_direction_means = np.asarray(
            [np.mean(c[metric][primary_positions], axis=0) for c in random_curves]
        )
        rand_mean = np.mean(rand_direction_means, axis=0)
        rand_min = np.min(rand_direction_means, axis=0)
        rand_max = np.max(rand_direction_means, axis=0)
        ax.plot(RHOS, rand_mean, color="#57606a", marker="s", label="random mean")
        ax.fill_between(RHOS, rand_min, rand_max, color="#afb8c1", alpha=0.25, label="random min–max")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("rho")
        ax.set_ylabel(f"Mean {metric}(rho)")
        ax.set_title(f"{metric} evidence trajectory")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig_phase10_1_st_trajectories.png", dpi=180)
    plt.close(fig)

    s_auc = auc(curves["deployed_real"]["S"])[primary_positions]
    t_auc = auc(curves["deployed_real"]["T"])[primary_positions]
    colors = {
        "S+_T+": "#1a7f37",
        "S+_T-": "#d1242f",
        "S-_T+": "#0969da",
        "S-_T-": "#8250df",
        "ambiguous": "#8c959f",
    }
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    labels = np.asarray(heterogeneity["labels"])
    for label in colors:
        mask = labels == label
        if np.any(mask):
            ax.scatter(s_auc[mask], t_auc[mask], s=34, alpha=0.8, color=colors[label], label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Per-sample normalized S-AUC")
    ax.set_ylabel("Per-sample normalized T-AUC")
    ax.set_title("Phase 10.1 per-sample S/T heterogeneity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig_phase10_1_sample_heterogeneity.png", dpi=180)
    plt.close(fig)


def fmt(x, digits=6):
    if x is None:
        return "UNAVAILABLE"
    return f"{x:.{digits}f}"


def run_scientific_analysis():
    start = time.perf_counter()
    raw, checks, failures, _context = run_validity_gate()
    gate_elapsed = time.perf_counter() - start
    if failures:
        write_void(checks, failures, gate_elapsed)
        print("VOID:", "; ".join(failures))
        return 2

    load_scientific_dependencies()
    OUT.mkdir(parents=True, exist_ok=True)
    gate_result = {
        "status": "PASS",
        "passed": True,
        "failures": [],
        "checks": checks,
        "scientific_statistics_computed_before_gate": False,
        "gate_wall_seconds": gate_elapsed,
    }
    (OUT / "phase10_1_validity_gate.json").write_text(json.dumps(gate_result, indent=2))
    print(f"VALIDITY GATE PASS ({gate_elapsed:.3f}s); beginning frozen CPU analysis")

    frozen = raw["frozen_baseline"]
    routed_ids = list(frozen.keys())
    own_ids = [i for i in routed_ids if frozen[i]["is_own_channel_err"] is True]
    primary_positions = np.asarray([routed_ids.index(i) for i in own_ids], dtype=int)
    routed_positions = np.arange(len(routed_ids), dtype=int)

    curves = {
        "deployed_real": build_curve(raw, "deployed|real", routed_ids),
        "deployed_reverse": build_curve(raw, "deployed|reverse", routed_ids),
        "M1_real": build_curve(raw, "M1|real", routed_ids),
        "M1_reverse": build_curve(raw, "M1|reverse", routed_ids),
    }
    random_curves = [
        build_curve(raw, f"random{k}", routed_ids) for k in range(N_RANDOM)
    ]

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    boot_primary = rng.integers(0, len(primary_positions), size=(BOOTSTRAPS, len(primary_positions)))
    boot_routed = rng.integers(0, len(routed_positions), size=(BOOTSTRAPS, len(routed_positions)))
    primary = population_analysis(primary_positions, curves, random_curves, boot_primary)
    sensitivity = population_analysis(routed_positions, curves, random_curves, boot_routed)
    population_anomaly = (
        primary["S"]["exceeds_all_random"] != sensitivity["S"]["exceeds_all_random"]
    )

    m1 = m1_consistency(curves, primary_positions)
    matching = matching_analysis(raw, routed_ids, primary_positions, curves, random_curves)
    dep_s_auc_all = auc(curves["deployed_real"]["S"])
    dep_t_auc_all = auc(curves["deployed_real"]["T"])
    heterogeneity = heterogeneity_summary(
        dep_s_auc_all[primary_positions], dep_t_auc_all[primary_positions]
    )
    competitor = competitor_structure(raw, routed_ids, primary_positions)

    transition_rows = []
    behavior = {}
    for direction in ("deployed", "M1"):
        behavior[direction] = {}
        for pop_name, pop_ids in (("primary68", own_ids), ("routed74", routed_ids)):
            rows, metrics = transition_rows_and_metrics(
                raw, routed_ids, own_ids, pop_ids, direction
            )
            transition_rows.extend(rows)
            behavior[direction][pop_name] = metrics

    direction_rows = []
    for pop_name, positions, result in (
        ("primary68", primary_positions, primary),
        ("routed74", routed_positions, sensitivity),
    ):
        direction_rows.append(
            {
                "population": pop_name,
                "direction": "deployed_real",
                "role": "confirmatory_real",
                "seed": "",
                "mean_S_AUC": result["S"]["real"],
                "mean_T_AUC": result["T"]["real"],
                "S_ge_real": True,
                "T_ge_real": True,
            }
        )
        for k, curve in enumerate(random_curves):
            rs = float(np.mean(auc(curve["S"])[positions]))
            rt = float(np.mean(auc(curve["T"])[positions]))
            direction_rows.append(
                {
                    "population": pop_name,
                    "direction": f"random{k}",
                    "role": "matched_random_null",
                    "seed": 5000 + k,
                    "mean_S_AUC": rs,
                    "mean_T_AUC": rt,
                    "S_ge_real": rs >= result["S"]["real"],
                    "T_ge_real": rt >= result["T"]["real"],
                }
            )
        for key, role in (
            ("deployed_reverse", "mechanism_contrast"),
            ("M1_real", "exploratory_estimator"),
            ("M1_reverse", "exploratory_estimator_reverse"),
        ):
            direction_rows.append(
                {
                    "population": pop_name,
                    "direction": key,
                    "role": role,
                    "seed": "",
                    "mean_S_AUC": float(np.mean(auc(curves[key]["S"])[positions])),
                    "mean_T_AUC": float(np.mean(auc(curves[key]["T"])[positions])),
                    "S_ge_real": "",
                    "T_ge_real": "",
                }
            )

    per_sample_rows = []
    label_by_pos = {
        int(pos): label
        for pos, label in zip(primary_positions, heterogeneity["labels"])
    }
    for pos, idx in enumerate(routed_ids):
        row = {
            "idx": int(idx),
            "uuid": frozen[idx]["uuid"],
            "primary68": pos in set(primary_positions),
            "gold": frozen[idx]["gold"],
            "baseline_pred": frozen[idx]["baseline_pred"],
            "baseline_gold_adjacent": frozen[idx]["gold_adjacent"],
            "baseline_runner_up": frozen[idx]["runner_up"],
            "baseline_router_score": frozen[idx]["router_score"],
            "baseline_gold_minus_source": float(frozen[idx]["baseline_logp"][GOLD])
            - float(frozen[idx]["baseline_logp"][SOURCE]),
            "baseline_direct_minus_source": float(frozen[idx]["baseline_logp"]["direct"])
            - float(frozen[idx]["baseline_logp"][SOURCE]),
            "baseline_rfi_minus_source": float(
                frozen[idx]["baseline_logp"]["request_for_info"]
            )
            - float(frozen[idx]["baseline_logp"][SOURCE]),
            "deployed_S_AUC": dep_s_auc_all[pos],
            "deployed_T_AUC": dep_t_auc_all[pos],
            "deployed_reverse_S_AUC": auc(curves["deployed_reverse"]["S"])[pos],
            "deployed_reverse_T_AUC": auc(curves["deployed_reverse"]["T"])[pos],
            "M1_S_AUC": auc(curves["M1_real"]["S"])[pos],
            "M1_T_AUC": auc(curves["M1_real"]["T"])[pos],
            "deployed_pred_rho4": raw["cells"]["deployed|real|4.0"]["per_sample"][idx]["pred"],
            "deployed_reverse_pred_rho4": raw["cells"]["deployed|reverse|4.0"]["per_sample"][idx]["pred"],
            "M1_pred_rho4": raw["cells"]["M1|real|4.0"]["per_sample"][idx]["pred"],
            "heterogeneity_quadrant": label_by_pos.get(pos, "sensitivity_only"),
        }
        for j, rho in enumerate(RHOS):
            row[f"deployed_S_rho{int(rho)}"] = curves["deployed_real"]["S"][pos, j]
            row[f"deployed_T_rho{int(rho)}"] = curves["deployed_real"]["T"][pos, j]
        per_sample_rows.append(row)

    claim_supported = primary["S"]["exceeds_all_random"]
    if claim_supported:
        conclusion = (
            "The estimated direction exhibits a direction-specific source-displacement "
            "signature at the available empirical resolution."
        )
        interpretation_action = "SUPPORTED WITH EMPIRICAL-RESOLUTION LIMIT"
    else:
        conclusion = (
            "Under the frozen 20-direction null, source-displacement dynamics could not "
            "be distinguished from generic perturbation."
        )
        interpretation_action = "DIRECTION-SPECIFICITY INTERPRETATION WITHDRAWN"

    summary = {
        "protocol": "Phase 10.1 — Direction-Specificity Audit of Differential Source Displacement",
        "validity_gate": "PASS",
        "primary_population": primary,
        "sensitivity_population": sensitivity,
        "population_sensitive_anomaly": population_anomaly,
        "M1_exploratory_consistency": m1,
        "rho4_behavioral_flow": behavior,
        "matching": matching,
        "heterogeneity": {k: v for k, v in heterogeneity.items() if k != "labels"},
        "baseline_competitor_structure": competitor,
        "primary_direction_specificity": {
            "supported_at_available_empirical_resolution": claim_supported,
            "interpretation_action": interpretation_action,
            "conclusion": conclusion,
        },
        "historical_verdicts": {
            "Phase_10": "PARTIAL SUPPORT — UNCHANGED",
            "Phase_8_9_9_5": "UNCHANGED",
            "Gate_v2": "UNCHANGED",
        },
        "bootstrap": {
            "method": "sample-resampling percentile interval",
            "B": BOOTSTRAPS,
            "rng": "NumPy Generator(PCG64(10101))",
        },
    }

    write_csv(
        OUT / "phase10_1_direction_null.csv",
        direction_rows,
        [
            "population",
            "direction",
            "role",
            "seed",
            "mean_S_AUC",
            "mean_T_AUC",
            "S_ge_real",
            "T_ge_real",
        ],
    )
    write_csv(
        OUT / "phase10_1_transition_flows.csv",
        transition_rows,
        ["direction", "population", "rho", "baseline_mode", "intervention_mode", "count"],
    )
    write_csv(
        OUT / "phase10_1_per_sample.csv",
        per_sample_rows,
        list(per_sample_rows[0].keys()),
    )
    (OUT / "phase10_1_matching.json").write_text(json.dumps(matching, indent=2))
    make_plots(curves, random_curves, primary_positions, primary, heterogeneity)

    elapsed = time.perf_counter() - start
    summary["execution"] = {
        "wall_seconds": elapsed,
        "device": "CPU only",
        "model_loaded": False,
        "test_outcomes_accessed": False,
        "lfs_pull": False,
    }
    (OUT / "phase10_1_summary.json").write_text(json.dumps(summary, indent=2))

    b = behavior["deployed"]["primary68"]
    astd = primary["standardized_asymmetry"]
    h = summary["heterogeneity"]
    results_md = f"""# Phase 10.1 — Results

**Validity gate:** PASS
**Historical Phase 10 verdict:** PARTIAL SUPPORT — unchanged
**Primary conclusion:** {conclusion}

## Confirmatory direction-level test

- Deployed-real mean S-AUC: **{fmt(primary['S']['real'])}**
- Sample-bootstrap 95% CI: [{fmt(primary['S']['sample_bootstrap_ci_95'][0])}, {fmt(primary['S']['sample_bootstrap_ci_95'][1])}]
- Rank among real + 20 randoms: **{primary['S']['rank_desc_ties_against_real']}/21** (descending; ties against real)
- Random directions greater than or equal to real: **{primary['S']['n_random_ge_real']}/20**
- Exact add-one one-sided p: **{fmt(primary['S']['p_add_one'])}**
- Minimum attainable p: **{fmt(primary['S']['p_min'])}**
- Random mean ± sample SD: {fmt(primary['S']['random_mean'])} ± {fmt(primary['S']['random_sample_sd'])}
- Random range: [{fmt(primary['S']['random_min'])}, {fmt(primary['S']['random_max'])}]
- Real minus random mean: **{fmt(primary['S']['real_minus_random_mean'])}**

The full 20-direction null is in `phase10_1_direction_null.csv`. The direction, not each sample, is the empirical random replication.

## Target selectivity and standardized asymmetry

- Deployed-real mean T-AUC: **{fmt(primary['T']['real'])}**
- Sample-bootstrap 95% CI: [{fmt(primary['T']['sample_bootstrap_ci_95'][0])}, {fmt(primary['T']['sample_bootstrap_ci_95'][1])}]
- T rank: {primary['T']['rank_desc_ties_against_real']}/21
- T random mean ± sample SD: {fmt(primary['T']['random_mean'])} ± {fmt(primary['T']['random_sample_sd'])}
- T real minus random mean: {fmt(primary['T']['real_minus_random_mean'])}
- Z_S: {fmt(astd['Z_S'])}; Z_T: {fmt(astd['Z_T'])}; exploratory A_std: **{fmt(astd['A_std'])}**
- R_S: {fmt(astd['R_S'])}; R_T: {fmt(astd['R_T'])}

A_std is descriptive/exploratory and does not alter the confirmatory verdict.

## Real-versus-reverse

- Mean paired S-AUC difference: **{fmt(primary['real_vs_reverse']['mean_real_minus_reverse_S_AUC'])}**, 95% CI [{fmt(primary['real_vs_reverse']['S_paired_bootstrap_ci_95'][0])}, {fmt(primary['real_vs_reverse']['S_paired_bootstrap_ci_95'][1])}]
- Mean paired T-AUC difference: **{fmt(primary['real_vs_reverse']['mean_real_minus_reverse_T_AUC'])}**, 95% CI [{fmt(primary['real_vs_reverse']['T_paired_bootstrap_ci_95'][0])}, {fmt(primary['real_vs_reverse']['T_paired_bootstrap_ci_95'][1])}]

Reverse is a separate mechanism contrast and is not part of the 20-direction empirical null.

## Population sensitivity

- Primary 68 direction-specificity: {primary['S']['exceeds_all_random']}
- Routed 74 direction-specificity: {sensitivity['S']['exceeds_all_random']}
- Population-sensitive anomaly: **{population_anomaly}**
- Routed-74 mean S-AUC / rank / p: {fmt(sensitivity['S']['real'])} / {sensitivity['S']['rank_desc_ties_against_real']}/21 / {fmt(sensitivity['S']['p_add_one'])}
- Routed-74 mean T-AUC / rank: {fmt(sensitivity['T']['real'])} / {sensitivity['T']['rank_desc_ties_against_real']}/21

## Historical rho=4 behavioral flow

- Source exits: **{b['source_exits']}**
- Target arrivals: **{b['target_arrivals']}**
- Target-hit: **{fmt(b['target_hit'], 4)}** ({b['target_arrivals']}/{b['target_hit_exact_denominator']})
- Wrong-to-wrong redistribution: **{b['wrong_to_wrong_redistribution']}**
- Target Gain: **{b['Target_Gain']}**
- Routed-74 collateral: changed {behavior['deployed']['routed74']['collateral_changed']}, fixed {behavior['deployed']['routed74']['collateral_fixed']}, broke {behavior['deployed']['routed74']['collateral_broke']}, wrong-to-wrong {behavior['deployed']['routed74']['collateral_wrong_to_wrong']}

The complete transition matrices are in `phase10_1_transition_flows.csv`.

## Exploratory analyses

- Displacement matching: **{matching['status']}**; qualified directions {matching['qualified_directions']}/20.
- Per-sample ambiguity: {h['ambiguous_n']}/68; nonambiguous {h['nonambiguous_n']}/68.
- Quadrants among nonambiguous samples: {json.dumps(h['quadrant_counts'])}.
- Per-sample S/T Pearson correlation: {fmt(h['pearson_S_T'])}.
- Baseline competitor structure: {competitor['status']}; out-of-sample log-loss improvement: {fmt(competitor.get('log_loss_improvement'))}; permutation status: {competitor.get('permutation_null_status', 'UNAVAILABLE')}; add-one p: {fmt(competitor.get('permutation_add_one_p'))}.
- M1 is exploratory only. S dose-sign agreement: {m1['S']['dose_sign_agreement_count']}/4; T dose-sign agreement: {m1['T']['dose_sign_agreement_count']}/4.

No clustering or multimodal-geometry claim is made. Baseline competitor structure is only a predictor of which stored destination survives, not a discovered internal algorithm.

## Interpretation boundary

**{interpretation_action}.** {conclusion}

This result does not identify the model's true internal decision algorithm, prove that `direct` is an attractor, establish a need for nonlinear or low-rank intervention, prove a factorized policy, or establish a shared mechanism across SAKIKO positives.
"""
    (OUT / "PHASE10_1_RESULTS.md").write_text(results_md)

    handoff = f"""# Phase 10.1 — Execution Handoff

- Validity gate: PASS
- Frozen V2 preregistration SHA256: `{EXPECTED_HASHES[V2_PREREG_PATH]}`
- Frozen V1 input-audit SHA256: `{EXPECTED_HASHES[V1_AUDIT_PATH]}`
- Raw input SHA256: `{EXPECTED_HASHES[RAW_PATH]}`
- CPU wall time: {elapsed:.3f} seconds
- GPU execution: zero
- Model loading: zero
- New predictions: zero
- Test-outcome access: zero
- Git-LFS pull: zero
- New random directions: zero
- Historical verdict changes: zero
- Frozen protocol-file changes: zero
- Commit/push: zero

The analysis stopped after producing the preregistered CPU outputs and awaits human review.
"""
    (OUT / "PHASE10_1_EXECUTION_HANDOFF.md").write_text(handoff)
    print(f"DONE in {elapsed:.3f}s | {conclusion}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 10.1 V2 population-typed gate and frozen analysis runner"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--gate-only", action="store_true")
    modes.add_argument("--run-analysis", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gate_only:
        start = time.perf_counter()
        _raw, checks, failures, context = run_validity_gate()
        elapsed = time.perf_counter() - start
        write_gate_only(checks, failures, context, elapsed)
        return 2 if failures else 0
    if args.run_analysis:
        return run_scientific_analysis()
    raise AssertionError("argparse must select exactly one mode")


if __name__ == "__main__":
    sys.exit(main())
