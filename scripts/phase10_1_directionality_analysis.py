#!/usr/bin/env python3
"""
Phase 10.1 — Direction-Specificity Audit of Differential Source Displacement.

CPU-only retrospective analysis of the frozen Phase 10 MCTL raw JSON.
The validity gate runs and is written before any scientific statistic is computed.
No model, GPU, new prediction, new random direction, test outcome, or LFS object is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "phase10_1_directionality"
RAW_PATH = ROOT / "final" / "results" / "phase10_mctl" / "phase10_mctl_raw.json"
HIST_GATE_PATH = ROOT / "final" / "results" / "phase10_mctl" / "phase10_validity_gate.json"
PREREG_PATH = OUT / "PHASE10_1_PREREGISTRATION.md"
AUDIT_PATH = OUT / "PHASE10_1_INPUT_AUDIT.md"
HASHES_PATH = OUT / "PHASE10_1_HASHES.json"
VAL_IDX_PATH = ROOT / "final" / "results" / "splits" / "val_idx.json"
TEST_IDX_PATH = ROOT / "final" / "results" / "splits" / "test_idx.json"

EXPECTED = {
    PREREG_PATH: "33649a3929bec9a3124b1495375c810c3c0eaffcb90fde3884b8945f63455340",
    AUDIT_PATH: "423f8c41044f2e10aef2562b36a2d197b2e6799fc25c615d68f55ee20627a231",
    RAW_PATH: "c9324d46a417e70e2244ec154e9aff507ae9bcc126212b3e188f8a968a7d5f5d",
}
EXPECTED_NONPRIMARY = {178, 364, 962, 1264, 2061, 3034}
MODES = ("tool_call", "cannot_answer", "direct", "request_for_info")
SOURCE = "tool_call"
GOLD = "cannot_answer"
WRONG = ("direct", "request_for_info")
RHOS = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
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


def reconstructed_aggregate(raw, cell_key, primary_ids, routed_ids):
    frozen = raw["frozen_baseline"]
    ps = raw["cells"][cell_key]["per_sample"]
    fixed = sum(
        frozen[i]["baseline_pred"] != frozen[i]["gold"] and ps[i]["pred"] == frozen[i]["gold"]
        for i in routed_ids
    )
    broke = sum(
        frozen[i]["baseline_pred"] == frozen[i]["gold"] and ps[i]["pred"] != frozen[i]["gold"]
        for i in routed_ids
    )
    to_gold = sum(ps[i]["pred"] == frozen[i]["gold"] for i in primary_ids)
    to_other = sum(
        ps[i]["pred"] != frozen[i]["gold"] and ps[i]["pred"] != frozen[i]["baseline_pred"]
        for i in primary_ids
    )
    moved = sum(ps[i]["pred"] != frozen[i]["baseline_pred"] for i in primary_ids)
    damage = sum(
        frozen[i]["baseline_pred"] == frozen[i]["gold"] and ps[i]["pred"] != frozen[i]["gold"]
        for i in routed_ids
    )
    return {
        "fixed": fixed,
        "broke": broke,
        "net": fixed - broke,
        "own_before": len(primary_ids),
        "own_after": len(primary_ids) - moved,
        "own_moved": moved,
        "own_to_gold": to_gold,
        "own_to_other_wrong": to_other,
        "target_gain": to_gold - to_other,
        "target_hit_rate": round(to_gold / moved, 4) if moved else None,
        "redistribution_rate": round(to_other / moved, 4) if moved else None,
        "damage_on_correct": damage,
    }


def run_validity_gate():
    checks = []
    failures = []

    for path, expected in EXPECTED.items():
        actual = sha256(path) if path.exists() else "MISSING"
        ok = actual == expected
        add_check(checks, f"sha256:{path.name}", ok, f"expected={expected}; actual={actual}")
        if not ok:
            failures.append(f"hash mismatch: {path}")

    try:
        raw = load_json_strict(RAW_PATH)
        add_check(checks, "strict_json_no_duplicate_keys", True, "raw JSON parsed without duplicate keys")
    except Exception as exc:
        add_check(checks, "strict_json_no_duplicate_keys", False, repr(exc))
        failures.append(f"strict raw JSON parse failed: {exc}")
        return None, checks, failures

    frozen = raw.get("frozen_baseline", {})
    routed_ids = list(frozen.keys())
    routed_set = set(routed_ids)
    own_ids = [
        i
        for i in routed_ids
        if frozen[i].get("is_own_channel_err") is True
        and frozen[i].get("gold") == GOLD
        and frozen[i].get("baseline_pred") == SOURCE
    ]
    own_set = set(own_ids)
    nonprimary = routed_set - own_set

    conditions = [
        ("routed_population_n", len(routed_ids) == 74, f"n={len(routed_ids)}"),
        ("own_channel_population_n", len(own_ids) == 68, f"n={len(own_ids)}"),
        (
            "primary_strict_subset",
            own_set < routed_set and len(routed_set - own_set) == 6,
            f"primary={len(own_set)} routed={len(routed_set)} extra={len(routed_set-own_set)}",
        ),
        (
            "extra_six_ids",
            {int(i) for i in nonprimary} == EXPECTED_NONPRIMARY,
            f"actual={sorted(map(int, nonprimary))}",
        ),
    ]
    for name, ok, detail in conditions:
        add_check(checks, name, ok, detail)
        if not ok:
            failures.append(name)

    extra_records = [frozen[i] for i in nonprimary]
    extra_identity_ok = (
        sum(r["gold"] == "tool_call" and r["baseline_pred"] == "tool_call" for r in extra_records)
        == 5
        and sum(
            r["gold"] == "request_for_info" and r["baseline_pred"] == "tool_call"
            for r in extra_records
        )
        == 1
    )
    add_check(
        checks,
        "extra_six_channel_identity",
        extra_identity_ok,
        "expected five tool_call→tool_call and one request_for_info→tool_call",
    )
    if not extra_identity_ok:
        failures.append("extra_six_channel_identity")

    required_cells = {"zero"}
    for direction in ("deployed", "M1"):
        for kind in ("real", "reverse"):
            required_cells.update(f"{direction}|{kind}|{rho:.1f}" for rho in RHOS[1:])
    required_cells.update(
        f"random{k}|{rho:.1f}" for k in range(N_RANDOM) for rho in RHOS[1:]
    )
    actual_cells = set(raw.get("cells", {}))
    cell_ok = actual_cells == required_cells and len(actual_cells) == 97
    add_check(
        checks,
        "required_97_cells",
        cell_ok,
        f"actual={len(actual_cells)} missing={sorted(required_cells-actual_cells)} "
        f"extra={sorted(actual_cells-required_cells)}",
    )
    if not cell_ok:
        failures.append("required_97_cells")

    baseline_order = tuple(routed_ids)
    order_ok = True
    finite_ok = True
    mode_ok = True
    record_count_ok = True
    id_value_ok = len({int(i) for i in routed_ids}) == 74
    uuid_ok = len({frozen[i]["uuid"] for i in routed_ids}) == 74
    for cell in raw.get("cells", {}).values():
        ps = cell.get("per_sample", {})
        if tuple(ps.keys()) != baseline_order:
            order_ok = False
        if len(ps) != 74:
            record_count_ok = False
        for rec in ps.values():
            lp = rec.get("logp", {})
            if set(lp) != set(MODES):
                mode_ok = False
            if any(not math.isfinite(float(v)) for v in lp.values()):
                finite_ok = False
    for name, ok, detail in [
        ("sample_id_uniqueness", id_value_ok and uuid_ok, f"idx_unique={id_value_ok}; uuid_unique={uuid_ok}"),
        ("cell_record_count", record_count_ok, "expected 74 records in every cell"),
        ("aligned_sample_order", order_ok, "expected exact frozen_baseline key order in every cell"),
        ("four_response_modes", mode_ok, f"expected={MODES}"),
        ("finite_average_logprob", finite_ok, "all stored logp values finite"),
    ]:
        add_check(checks, name, ok, detail)
        if not ok:
            failures.append(name)

    zero_ok = all(
        raw["cells"]["zero"]["per_sample"][i]["pred"] == frozen[i]["baseline_pred"]
        for i in routed_ids
    )
    add_check(checks, "zero_reproduces_baseline", zero_ok, "74/74 required")
    if not zero_ok:
        failures.append("zero_reproduces_baseline")

    val_set = set(map(int, json.loads(VAL_IDX_PATH.read_text())))
    test_set = set(map(int, json.loads(TEST_IDX_PATH.read_text())))
    routed_int = set(map(int, routed_ids))
    val_ok = routed_int <= val_set
    test_ok = not (routed_int & test_set)
    add_check(checks, "all_routed_in_validation", val_ok, f"in_val={len(routed_int & val_set)}/74")
    add_check(checks, "zero_test_id_overlap", test_ok, f"intersection={sorted(routed_int & test_set)}")
    if not val_ok:
        failures.append("all_routed_in_validation")
    if not test_ok:
        failures.append("zero_test_id_overlap")

    random_ok = all(
        all(f"random{k}|{rho:.1f}" in raw["cells"] for rho in RHOS[1:])
        for k in range(N_RANDOM)
    )
    direction_ok = all(
        all(
            f"{d}|{kind}|{rho:.1f}" in raw["cells"]
            for kind in ("real", "reverse")
            for rho in RHOS[1:]
        )
        for d in ("deployed", "M1")
    )
    rho_ok = "zero" in raw["cells"] and random_ok and direction_ok
    add_check(checks, "twenty_random_directions_complete", random_ok, "20 identities × rho 1,2,3,4")
    add_check(checks, "real_reverse_deployed_M1_complete", direction_ok, "2 directions × 2 signs × 4 doses")
    add_check(checks, "rho_0_to_4_complete", rho_ok, "rho0 shared zero; rho1-4 per cell")
    if not random_ok:
        failures.append("twenty_random_directions_complete")
    if not direction_ok:
        failures.append("real_reverse_deployed_M1_complete")
    if not rho_ok:
        failures.append("rho_0_to_4_complete")

    aggregate_checks = {}
    aggregate_ok = True
    for direction in ("deployed", "M1"):
        key = f"{direction}|real|4.0"
        got = reconstructed_aggregate(raw, key, own_ids, routed_ids)
        stored = raw["cells"][key]["agg"]
        compare = {k: {"reconstructed": v, "stored": stored.get(k), "match": stored.get(k) == v}
                   for k, v in got.items()}
        ok = all(v["match"] for v in compare.values())
        aggregate_checks[direction] = compare
        aggregate_ok &= ok
    add_check(checks, "rho4_aggregate_reproduction", aggregate_ok, json.dumps(aggregate_checks))
    if not aggregate_ok:
        failures.append("rho4_aggregate_reproduction")

    try:
        hist_gate = load_json_strict(HIST_GATE_PATH)
        hist_rows_ok = hist_gate.get("passed") is True and len(hist_gate.get("rows", [])) == 6
        for row in hist_gate.get("rows", []):
            key = f"{row['direction']}|real|{float(row['rho']):.1f}"
            if key not in raw["cells"] or row.get("match") is not True:
                hist_rows_ok = False
                continue
            stored = raw["cells"][key]["agg"]
            for field, value in row.get("got", {}).items():
                if stored.get(field) != value:
                    hist_rows_ok = False
        add_check(checks, "historical_six_cell_gate_consistency", hist_rows_ok, "passed flag, six rows, raw aggregate equality")
        if not hist_rows_ok:
            failures.append("historical_six_cell_gate_consistency")
    except Exception as exc:
        add_check(checks, "historical_six_cell_gate_consistency", False, repr(exc))
        failures.append("historical_six_cell_gate_consistency")

    return raw, checks, failures


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


def main():
    start = time.perf_counter()
    raw, checks, failures = run_validity_gate()
    gate_elapsed = time.perf_counter() - start
    if failures:
        write_void(checks, failures, gate_elapsed)
        print("VOID:", "; ".join(failures))
        return 2

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
- Frozen preregistration SHA256: `{EXPECTED[PREREG_PATH]}`
- Frozen input-audit SHA256: `{EXPECTED[AUDIT_PATH]}`
- Raw input SHA256: `{EXPECTED[RAW_PATH]}`
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


if __name__ == "__main__":
    sys.exit(main())
