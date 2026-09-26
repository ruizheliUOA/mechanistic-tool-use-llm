#!/usr/bin/env python3
"""CPU-only Block 0/1 audit and pre-fit Router specification for Qwen Stage B.

This program opens only committed protocol/provenance metadata and the
explicitly authorized Stage A geometry artifacts. It does not import torch,
load a model, access a project test manifest/index/row, or run inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta


ROOT = Path("/root/autodl-tmp/sakiko-followup")
STAGE_A = ROOT / "final/results/qwen25_stage_a_jacobian_geometry"
OUT = ROOT / "final/results/qwen25_stage_b_preparation"
RAW = STAGE_A / "jacobian_geometry.jsonl"
CONCENTRATION = STAGE_A / "concentration_null.json"
CONFIG = STAGE_A / "gate_a_configuration_recovery.json"

DIMENSION = 3584
DENOMINATOR_EPSILON = 1e-8
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260729
CHANNELS = ("rfi_tc", "ca_tc", "ca_direct")
GEOMETRIES = ("full_effect", "common_prompt")
EXPECTED_COMMON_VALID = {"rfi_tc": 60, "ca_tc": 46, "ca_direct": 32}

LAYERS = {
    "rfi_tc": {
        "L_obs": 20,
        "L_inj": 18,
        "L_jacobian": 18,
        "direction_tensor_sha256": "678adedaeef8eb9975dfc72b78ae8e7636ab8aaa7372c92f1a4637238b66a0a8",
    },
    "ca_tc": {
        "L_obs": 20,
        "L_inj": 16,
        "L_jacobian": 16,
        "direction_tensor_sha256": "4e99fe68d257c2223b7b226898a93a0b75407f60d5fa98bdd51eaf54304b4f93",
    },
    "ca_direct": {
        "L_obs": 16,
        "L_inj": 16,
        "L_jacobian": 16,
        "direction_tensor_sha256": "0c0c1e58670aeaf644686c6c77f6d970f63ce928603a667c51f88c2f6c1455e4",
    },
}

ARCHIVED_AUC = {
    "rfi_tc": 0.7545935845530988,
    "ca_tc": 0.9373507733831621,
    "ca_direct": 0.960428366200862,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean_float(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"nonfinite output: {result}")
    return result


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "n": int(array.size),
        "mean": clean_float(np.mean(array)),
        "median": clean_float(np.median(array)),
        "p05": clean_float(np.quantile(array, 0.05)),
        "p25": clean_float(np.quantile(array, 0.25)),
        "p75": clean_float(np.quantile(array, 0.75)),
        "p95": clean_float(np.quantile(array, 0.95)),
        "min": clean_float(np.min(array)),
        "max": clean_float(np.max(array)),
    }


def null_summary(rank: int) -> dict[str, float]:
    alpha = rank / 2
    beta_parameter = (DIMENSION - rank) / 2
    return {
        "rank": rank,
        "alpha": alpha,
        "beta": beta_parameter,
        "mean": clean_float(rank / DIMENSION),
        "median": clean_float(beta.ppf(0.5, alpha, beta_parameter)),
        "p95": clean_float(beta.ppf(0.95, alpha, beta_parameter)),
        "p99": clean_float(beta.ppf(0.99, alpha, beta_parameter)),
    }


def metric_summary(rows: list[dict[str, Any]], rank_null: dict[str, float]) -> dict[str, Any]:
    a_contrast = np.asarray([row["a_contrast"] for row in rows], dtype=np.float64)
    a_dec = np.asarray([row["a_dec"] for row in rows], dtype=np.float64)
    a_offaxis = np.asarray([row["a_offaxis"] for row in rows], dtype=np.float64)
    rank = int(rows[0]["observed_rank"])
    if any(int(row["observed_rank"]) != rank for row in rows):
        raise RuntimeError("mixed ranks within channel/geometry")

    contrast_null = null_summary(1)
    q_mask = a_dec > DENOMINATOR_EPSILON
    signal_mask = a_dec > rank_null["p95"]
    q = a_offaxis[q_mask] / a_dec[q_mask]
    signal_q = a_offaxis[signal_mask] / a_dec[signal_mask]
    q_ratio = np.sum(a_offaxis) / np.sum(a_dec)

    def compare(observed: np.ndarray, null: dict[str, float]) -> dict[str, Any]:
        observed_median = clean_float(np.median(observed))
        cdf = clean_float(beta.cdf(observed_median, null["alpha"], null["beta"]))
        return {
            "observed": distribution(observed),
            "observed_median_over_null_median": clean_float(
                observed_median / null["median"]
            ),
            "observed_median_null_cdf": cdf,
            "observed_median_null_survival": clean_float(1.0 - cdf),
            "percent_exceeding_null_p95": clean_float(
                100 * np.mean(observed > null["p95"])
            ),
            "percent_exceeding_null_p99": clean_float(
                100 * np.mean(observed > null["p99"])
            ),
        }

    return {
        "n_valid": len(rows),
        "observed_rank_values": sorted({int(row["observed_rank"]) for row in rows}),
        "a_contrast": compare(a_contrast, contrast_null),
        "a_dec": compare(a_dec, rank_null),
        "offaxis": {
            "denominator_epsilon": DENOMINATOR_EPSILON,
            "n_excluded_by_denominator_epsilon": int(np.sum(~q_mask)),
            "n_above_rank_null_p95": int(np.sum(signal_mask)),
            "q_unconditional": distribution(q),
            "q_signal_conditioned": distribution(signal_q),
            "Q_ratio_of_sums": clean_float(q_ratio),
        },
    }


def q_and_Q(rows: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    a_dec = np.asarray([row["a_dec"] for row in rows], dtype=np.float64)
    a_offaxis = np.asarray([row["a_offaxis"] for row in rows], dtype=np.float64)
    mask = a_dec > DENOMINATOR_EPSILON
    return a_offaxis[mask] / a_dec[mask], clean_float(np.sum(a_offaxis) / np.sum(a_dec))


def bootstrap_difference(
    left: list[dict[str, Any]],
    comparisons: list[list[dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    left_dec = np.asarray([row["a_dec"] for row in left], dtype=np.float64)
    left_off = np.asarray([row["a_offaxis"] for row in left], dtype=np.float64)
    comparison_arrays = [
        (
            np.asarray([row["a_dec"] for row in rows], dtype=np.float64),
            np.asarray([row["a_offaxis"] for row in rows], dtype=np.float64),
        )
        for rows in comparisons
    ]

    left_q, left_Q = q_and_Q(left)
    comparison_rows = [row for group in comparisons for row in group]
    comparison_q, comparison_Q = q_and_Q(comparison_rows)
    observed = {
        "Q_difference_left_minus_comparison": clean_float(left_Q - comparison_Q),
        "median_q_difference_left_minus_comparison": clean_float(
            np.median(left_q) - np.median(comparison_q)
        ),
    }

    boot_Q = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    boot_median = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        li = rng.integers(0, len(left_dec), len(left_dec))
        ld = left_dec[li]
        lo = left_off[li]
        sampled_comparisons: list[tuple[np.ndarray, np.ndarray]] = []
        for dec, off in comparison_arrays:
            ci = rng.integers(0, len(dec), len(dec))
            sampled_comparisons.append((dec[ci], off[ci]))
        cd = np.concatenate([pair[0] for pair in sampled_comparisons])
        co = np.concatenate([pair[1] for pair in sampled_comparisons])
        boot_Q[replicate] = np.sum(lo) / np.sum(ld) - np.sum(co) / np.sum(cd)
        lq = lo[ld > DENOMINATOR_EPSILON] / ld[ld > DENOMINATOR_EPSILON]
        cq = co[cd > DENOMINATOR_EPSILON] / cd[cd > DENOMINATOR_EPSILON]
        boot_median[replicate] = np.median(lq) - np.median(cq)

    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed_stream": BOOTSTRAP_SEED,
        "sampling": "independent within-channel resampling; pooled comparison preserves strata",
        "observed": observed,
        "Q_difference_percentile_95_CI": [
            clean_float(np.quantile(boot_Q, 0.025)),
            clean_float(np.quantile(boot_Q, 0.975)),
        ],
        "median_q_difference_percentile_95_CI": [
            clean_float(np.quantile(boot_median, 0.025)),
            clean_float(np.quantile(boot_median, 0.975)),
        ],
    }


def gradient_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for mode, value in row["gradient_norms"].items():
            modes[mode].append(float(value))
    return {mode: distribution(values) for mode, values in sorted(modes.items())}


def block0() -> dict[str, Any]:
    return {
        "classification": "HISTORICALLY_USED_EVALUATION_ONLY",
        "hard_stop_0_triggered": True,
        "basis": [
            {
                "evidence": (
                    "scripts/qwen7b_native_sakiko_run.py:140-174 evaluates the locked "
                    "project test population and writes test details/summary."
                ),
                "kind": "source_and_git_history_metadata_only",
            },
            {
                "evidence": (
                    "scripts/qwen7b_native_sakiko_run.py:176-216 runs real, reverse, "
                    "random, and ungated controls on that evaluation population."
                ),
                "kind": "source_and_git_history_metadata_only",
            },
            {
                "evidence": (
                    "scripts/qwen7b_followup_locked_tests.py:1-17,67-151 declares and "
                    "executes locked-test confirmations."
                ),
                "kind": "source_and_git_history_metadata_only",
            },
            {
                "evidence": (
                    "scripts/qwen7b_method_diagnostics.py:170-228 and "
                    "scripts/qwen7b_native_sakiko_multiseed.py:180-260 perform further "
                    "locked evaluation analyses."
                ),
                "kind": "source_and_git_history_metadata_only",
            },
            {
                "evidence": (
                    "split manifests first entered Git at e1b4b1a2; Qwen scripts/results "
                    "were archived at de3d86cc. The split predates the present geometry "
                    "hypothesis, but its outcomes and aggregates were subsequently used."
                ),
                "kind": "git_history_metadata_only",
            },
            {
                "evidence": (
                    "Stage A manifest states it did not open project test material. "
                    "That isolation does not restore an evaluation population already "
                    "used by earlier Qwen work."
                ),
                "kind": "authorized_stage_a_manifest",
            },
        ],
        "audit_incident": {
            "occurred": True,
            "description": (
                "During this task's initial provenance search, a read-only command "
                "mistakenly opened the historical qwen25_7b_router_metrics.json report "
                "and exposed historical test-split aggregate population counts. No row, "
                "ID, sample content, label vector, prediction vector, or per-sample score "
                "was accessed. The command was stopped from being repeated, and all "
                "subsequent provenance work is source/filename/Git-metadata only."
            ),
            "effect": (
                "The requested zero-test-aggregate confirmation cannot truthfully be "
                "made. This is independently disqualifying and is recorded rather than "
                "concealed."
            ),
        },
        "consequence": (
            "Complete useful descriptive Blocks 1-2 only. Do not write a Stage B "
            "confirmatory preregistration, executable runner, execution freeze, or any "
            "formal test-population discovery artifact."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["channel"] not in CHANNELS:
                raise RuntimeError("unexpected channel")
            rows.append(row)
    concentration = json.loads(CONCENTRATION.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for channel in CHANNELS:
        for geometry in GEOMETRIES:
            grouped[(channel, geometry)] = [
                row
                for row in rows
                if row["channel"] == channel
                and row["geometry_type"] == geometry
                and row["geometry_status"] == "VALID"
            ]

    rank_values = sorted(
        {
            int(row["observed_rank"])
            for row in rows
            if row.get("a_dec") is not None and row["geometry_status"] == "VALID"
        }
    )
    if rank_values != [3]:
        raise RuntimeError(f"expected observed rank 3 only, found {rank_values}")
    rank_null = null_summary(3)

    metrics: dict[str, Any] = {}
    for channel in CHANNELS:
        metrics[channel] = {}
        for geometry in GEOMETRIES:
            metrics[channel][geometry] = metric_summary(
                grouped[(channel, geometry)], rank_null
            )

    actual_common = {
        channel: len(grouped[(channel, "common_prompt")]) for channel in CHANNELS
    }
    if actual_common != EXPECTED_COMMON_VALID:
        raise RuntimeError(f"common valid count mismatch: {actual_common}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstraps: dict[str, Any] = {}
    for geometry in GEOMETRIES:
        rfi = grouped[("rfi_tc", geometry)]
        bootstraps[geometry] = {
            "rfi_tc_vs_ca_tc": bootstrap_difference(
                rfi, [grouped[("ca_tc", geometry)]], rng
            ),
            "rfi_tc_vs_ca_direct": bootstrap_difference(
                rfi, [grouped[("ca_direct", geometry)]], rng
            ),
            "rfi_tc_vs_pooled_comparisons": bootstrap_difference(
                rfi,
                [
                    grouped[("ca_tc", geometry)],
                    grouped[("ca_direct", geometry)],
                ],
                rng,
            ),
        }

    orderings: dict[str, Any] = {}
    for geometry in GEOMETRIES:
        q_values = {
            channel: metrics[channel][geometry]["offaxis"]["Q_ratio_of_sums"]
            for channel in CHANNELS
        }
        orderings[geometry] = {
            "Q_values": q_values,
            "rfi_lower_than_both": (
                q_values["rfi_tc"] < q_values["ca_tc"]
                and q_values["rfi_tc"] < q_values["ca_direct"]
            ),
        }
    subset_stable = all(
        ordering["rfi_lower_than_both"] for ordering in orderings.values()
    )

    concentration_groups: dict[str, Any] = {}
    for channel in CHANNELS:
        for geometry in GEOMETRIES:
            key = f"{channel}|{geometry}"
            source = concentration["groups"][key]
            if source["n_axes"] != len(grouped[(channel, geometry)]):
                raise RuntimeError(f"concentration n mismatch for {key}")
            concentration_groups[key] = {
                **source,
                "observed_exceeds_null_p99": (
                    source["resultant_length"] > source["chance_null"]["p99"]
                ),
                "verification": (
                    "copied from authorized Stage A 10,000-trial rotational null after "
                    "n_axes cross-check; raw axes are not stored in Stage A JSONL"
                ),
            }

    router_counts = {
        channel: sum(
            bool(row["router_active"])
            for row in grouped[(channel, "full_effect")]
        )
        for channel in CHANNELS
    }
    if router_counts != {"rfi_tc": 57, "ca_tc": 62, "ca_direct": 52}:
        raise RuntimeError(f"Router-active count mismatch: {router_counts}")

    gradient: dict[str, Any] = {}
    for channel in CHANNELS:
        gradient[channel] = {
            geometry: gradient_summaries(grouped[(channel, geometry)])
            for geometry in GEOMETRIES
        }

    layer_provenance = {}
    for channel, layer in LAYERS.items():
        equal = layer["L_obs"] == layer["L_inj"]
        layer_provenance[channel] = {
            **layer,
            "L_obs_equals_L_inj": equal,
            "residual_stream_site": (
                "model.model.layers[L].mlp forward output; after MLP internals "
                "and before decoder-layer residual addition; all sequence positions"
            ),
            "source_references": [
                "final/results/qwen25_stage_a_jacobian_geometry/"
                "gate_a_configuration_recovery.json",
                "scripts/qwen7b_native_sakiko.py:377-396",
                "scripts/qwen25_stage_a_geometry.py:1470-1722",
            ],
            "mismatch_interpretation": (
                None
                if equal
                else (
                    "Direction estimated in one layer's feature coordinates and injected/"
                    "differentiated at another. Low absolute Jacobian alignment may partly "
                    "reflect this design; no cross-layer transport map was estimated."
                )
            ),
        }

    pooled_ci = bootstraps["full_effect"]["rfi_tc_vs_pooled_comparisons"][
        "Q_difference_percentile_95_CI"
    ]
    gate_conditions = {
        "full_Q_ordering": orderings["full_effect"]["rfi_lower_than_both"],
        "common_Q_ordering": orderings["common_prompt"]["rfi_lower_than_both"],
        "pooled_full_Q_bootstrap_CI_below_zero": pooled_ci[1] < 0,
        "rfi_signed_contrast_predominantly_gold_directed": False,
        "geometry_and_layer_provenance_compatible": True,
    }
    gate1 = {
        "branch": "ORDERING_PREDICTION_WEAK",
        "conditions": gate_conditions,
        "reason": (
            "Stage A stored squared a_contrast but not signed dot(d_hat,w_i), raw w_i, "
            "or per-sample actual margin change. Gold-directed sign cannot be recovered "
            "without rerunning Stage A, so mandatory condition 4 is unestablished. The "
            "three-channel ordering remains secondary/exploratory; rfi_tc real-direction "
            "specificity would be the primary branch if a future untouched population "
            "were independently authorized."
        ),
        "recorded_before_router_refit": True,
    }

    block1 = {
        "schema_version": 1,
        "scope": {
            "included": list(CHANNELS),
            "excluded": {
                "ca_rfi": "OUT_OF_SCOPE_DIRECTION_NOT_AVAILABLE",
                "tc_rfi": "OUT_OF_SCOPE_DIRECTION_NOT_AVAILABLE",
            },
            "claim_boundary": (
                "Prospective within-model three-channel ordering construct only; not the "
                "complete Qwen channel population and not a validated universal predictor."
            ),
        },
        "dimension": DIMENSION,
        "layer_provenance": layer_provenance,
        "nulls": {
            "a_contrast_single_axis": null_summary(1),
            "a_dec_verified_rank_specific": rank_null,
            "observed_valid_ranks": rank_values,
        },
        "signed_contrast": {
            "status": "UNAVAILABLE_FROM_STORED_STAGE_A_ARTIFACTS",
            "reason": (
                "Only squared a_contrast is stored; signed projection, raw contrast axes, "
                "and per-sample actual margin changes are absent. No sign is inferred."
            ),
            "association_with_actual_margin_change": None,
        },
        "metrics": metrics,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "results": bootstraps,
        },
        "concentration": {
            "source_sha256": sha256_file(CONCENTRATION),
            "stage_a_seed": concentration["seed"],
            "stage_a_null_trials": concentration["null_trials"],
            "method": concentration["null_method"],
            "groups": concentration_groups,
        },
        "full_common_consistency": {
            "expected_common_valid_counts": EXPECTED_COMMON_VALID,
            "observed_common_valid_counts": actual_common,
            "count_check": "PASS",
            "ordering": orderings,
            "subset_stable": subset_stable,
            "status": (
                "GEOMETRY_ORDERING_SUBSET_STABLE"
                if subset_stable
                else "GEOMETRY_ORDERING_NOT_SUBSET_STABLE"
            ),
        },
        "gradient_norm_structure": {
            "per_sample_source": (
                "jacobian_geometry.jsonl gradient_norms field contains every valid "
                "sample/mode norm"
            ),
            "distributions": gradient,
            "centered_energy_dominance_status": (
                "UNAVAILABLE_FROM_STORED_STAGE_A_ARTIFACTS"
            ),
            "reason": (
                "Mode norms do not determine norm(G_i,m - mean_m G_i,m). Raw gradient "
                "vectors and centered energies were not stored; balance/dominance and "
                "dominant-mode identity cannot be reconstructed without a rerun."
            ),
            "causal_claim": "NONE",
        },
        "router_active_definition": {
            "label": "RECONSTRUCTED_ROUTER_RULE",
            "stage_a_record_value": "RECONSTRUCTED_NOT_ARCHIVED",
            "counts": router_counts,
            "mechanism": (
                "Stage A deterministically refit StandardScaler + LogisticRegression "
                "(C=1, liblinear, max_iter=2000, random_state=42) using train positives="
                "current channel errors and negatives=all current correct train rows, "
                "then applied historical channel thresholds and source-prediction gate."
            ),
            "not_archived_router": True,
        },
        "gate1": gate1,
        "limitations": [
            "Signed contrast cannot be recovered from squared projections.",
            "Centered-gradient dominance cannot be recovered from mode norms alone.",
            "Concentration axes are absent from JSONL; authorized Stage A simulation is "
            "verified by hash/count and reported rather than spuriously recomputed.",
        ],
        "input_hashes": {
            str(RAW.relative_to(ROOT)): sha256_file(RAW),
            str(CONCENTRATION.relative_to(ROOT)): sha256_file(CONCENTRATION),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
        },
        "configuration_crosscheck": {
            "direction_lfs_oid": config["directions"]["lfs_oid_sha256"],
            "model_revision": config["model_tokenizer"]["model_revision"],
            "tokenizer_revision": config["model_tokenizer"]["tokenizer_revision"],
        },
    }

    block0_payload = block0()
    write_json(OUT / "block0_evaluation_admissibility_audit.json", block0_payload)
    write_json(OUT / "block1_geometry_verification.json", block1)

    csv_path = OUT / "block1_channel_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "channel",
            "geometry_type",
            "n_valid",
            "a_contrast_median",
            "a_contrast_median_over_null_median",
            "a_dec_median",
            "a_dec_median_over_null_median",
            "n_above_a_dec_null_p95",
            "q_mean",
            "q_median",
            "q_p25",
            "q_p75",
            "Q_ratio_of_sums",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for channel in CHANNELS:
            for geometry in GEOMETRIES:
                value = metrics[channel][geometry]
                q = value["offaxis"]["q_unconditional"]
                writer.writerow(
                    {
                        "channel": channel,
                        "geometry_type": geometry,
                        "n_valid": value["n_valid"],
                        "a_contrast_median": value["a_contrast"]["observed"]["median"],
                        "a_contrast_median_over_null_median": value["a_contrast"][
                            "observed_median_over_null_median"
                        ],
                        "a_dec_median": value["a_dec"]["observed"]["median"],
                        "a_dec_median_over_null_median": value["a_dec"][
                            "observed_median_over_null_median"
                        ],
                        "n_above_a_dec_null_p95": value["offaxis"][
                            "n_above_rank_null_p95"
                        ],
                        "q_mean": q["mean"],
                        "q_median": q["median"],
                        "q_p25": q["p25"],
                        "q_p75": q["p75"],
                        "Q_ratio_of_sums": value["offaxis"]["Q_ratio_of_sums"],
                    }
                )

    block0_md = """# Block 0 — Qwen evaluation-population admissibility audit

**Classification: `HISTORICALLY_USED_EVALUATION_ONLY`**

The fixed project evaluation split predates the present geometry hypothesis,
but it is not untouched. Repository source and Git metadata establish that
earlier Qwen work computed locked-evaluation predictions/outcomes, ran real,
reverse, random, ungated and follow-up controls, and archived results. Stage A
itself respected its test firewall; that cannot restore an already used
population.

Evidence is recorded mechanically in
`block0_evaluation_admissibility_audit.json`. No historical test artifact was
required to reach the classification.

## Audit incident

During the initial provenance search, one read-only command mistakenly opened
the historical `qwen25_7b_router_metrics.json` report and exposed aggregate
test-split population counts. It did **not** expose rows, IDs, sample content,
label/prediction vectors, or per-sample scores. The report was not reopened.
Consequently, this task cannot honestly claim zero test-aggregate access.

## Mechanical consequence

Hard Stop 0 applies. Blocks 1–2 may be retained as descriptive train/dev work,
but this task must not create a confirmatory preregistration, test runner,
execution freeze, or test-population discovery output.
"""
    (OUT / "BLOCK0_EVALUATION_ADMISSIBILITY_AUDIT.md").write_text(
        block0_md, encoding="utf-8"
    )

    contrast_null = block1["nulls"]["a_contrast_single_axis"]
    dec_null = block1["nulls"]["a_dec_verified_rank_specific"]
    table_lines = []
    for channel in CHANNELS:
        for geometry in GEOMETRIES:
            value = metrics[channel][geometry]
            q = value["offaxis"]["q_unconditional"]
            table_lines.append(
                f"| {channel} | {geometry} | {value['n_valid']} | "
                f"{q['median']:.6f} | {value['offaxis']['Q_ratio_of_sums']:.6f} | "
                f"{value['offaxis']['n_above_rank_null_p95']} |"
            )
    layer_lines = []
    for channel in CHANNELS:
        value = layer_provenance[channel]
        layer_lines.append(
            f"| {channel} | {value['L_obs']} | {value['L_inj']} | "
            f"{value['L_jacobian']} | {str(value['L_obs_equals_L_inj']).lower()} |"
        )
    geometry_md = f"""# Block 1 — Corrected Qwen Stage A geometry verification

**Gate 1: `ORDERING_PREDICTION_WEAK`**

The stable off-axis ordering is numerically preserved in both stored geometry
variants, but the mandatory signed gold-direction condition is not auditable:
Stage A stored squared contrast only. The result is therefore descriptive and
the three-channel ordering cannot become a confirmatory primary.

## Scope and layer provenance

| channel | L_obs | L_inj | Jacobian layer | equal |
|---|---:|---:|---:|---|
{chr(10).join(layer_lines)}

The site is `model.model.layers[L].mlp` forward output, after MLP internals and
before decoder-layer residual addition, at all sequence positions. For
`rfi_tc` and `ca_tc`, the direction was estimated in a different layer's
feature coordinates from the injection/Jacobian layer. No cross-layer
transport map was estimated; low absolute alignment can partly reflect that
choice. `ca_rfi` and `tc_rfi` are
`OUT_OF_SCOPE_DIRECTION_NOT_AVAILABLE`.

## Corrected orientation nulls

- `a_contrast`: Beta(1/2, (3584-1)/2), mean
  {contrast_null['mean']:.9g}, median {contrast_null['median']:.9g},
  p95 {contrast_null['p95']:.9g}, p99 {contrast_null['p99']:.9g}.
- Verified valid-record rank: 3 only. `a_dec`:
  Beta(3/2, (3584-3)/2), mean {dec_null['mean']:.9g}, median
  {dec_null['median']:.9g}, p95 {dec_null['p95']:.9g}, p99
  {dec_null['p99']:.9g}.

Observed/null median ratios, null CDFs and p95/p99 exceedance percentages are
in the JSON and CSV artifacts; observed medians are not compared only with a
null mean.

## Stable off-axis summaries

Denominator epsilon is `1e-8`, copied from Stage A's frozen off-axis numerical
tolerance (`scripts/qwen25_stage_a_geometry.py:847`), not selected from these
channel outcomes.

| channel | geometry | n valid | median q | Q=sum(off)/sum(dec) | n above null p95 |
|---|---|---:|---:|---:|---:|
{chr(10).join(table_lines)}

Ten-thousand channel-stratified bootstrap results (seed {BOOTSTRAP_SEED}) for
`rfi_tc` versus each comparison and their pooled stratum are in
`block1_geometry_verification.json`.

## Signed contrast

`UNAVAILABLE_FROM_STORED_STAGE_A_ARTIFACTS`. Neither signed
`dot(d_hat,w_i)`, raw axes, nor per-sample actual margin changes were retained.
No sign or sign/margin association is inferred from a squared projection.

## Concentration and gradient structure

The authorized Stage A concentration artifact (SHA256
`{sha256_file(CONCENTRATION)}`) contains 10,000-trial rotational nulls,
resultant length, full pairwise-cosine distributions/histograms, and
one-sided empirical p-values for all six channel/geometry groups. Counts were
cross-checked against valid raw records and all observed resultants exceed
their null p99.

Every per-sample, per-mode gradient norm remains reported in the Stage A
JSONL, and Block 1 reports their distributions. Centered-energy dominance
cannot be reconstructed from norms alone because raw gradient vectors were
not retained. The dominant mode and balanced-versus-dominated label are
therefore unavailable, with no causal inference made.

## Router-active definition

The counts 57/62/52 are `RECONSTRUCTED_ROUTER_RULE`, not archived Router
labels. Stage A refit its documented train-only logistic recipe and applied
the historical source predicate/threshold. The original `routers.pkl` was
absent.

## Consistency

Common-prompt valid counts exactly match 60/46/32. Both full-effect and
common-prompt `Q` preserve `rfi_tc < ca_tc` and `rfi_tc < ca_direct`.
Nevertheless, the unobservable signed criterion forces
`ORDERING_PREDICTION_WEAK`.
"""
    (OUT / "BLOCK1_GEOMETRY_VERIFICATION.md").write_text(
        geometry_md, encoding="utf-8"
    )

    router_spec = {
        "schema_version": 1,
        "status": "PREFIT_SPECIFICATION_FROZEN_PENDING_COMMIT",
        "scientific_definition_lock": "must be committed and pushed before fit",
        "channels": {
            channel: {
                "gold": config["channels"][channel]["gold"],
                "source": config["channels"][channel]["source"],
                "L_obs": LAYERS[channel]["L_obs"],
            }
            for channel in CHANNELS
        },
        "train_source": {
            "baseline": (
                "data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl "
                "via direct seeks from qwen25_baseline_train_dev_byte_index.csv"
            ),
            "row_index_manifest": "final/results/splits/train_idx.json",
            "count": 2556,
            "activation_arrays": {
                "L16": "data/processed/qwen25_7b_w2c/cache/acts_L16.npy",
                "L20": "data/processed/qwen25_7b_w2c/cache/acts_L20.npy",
            },
        },
        "dev_source": {
            "baseline": "same byte-indexed source; emitted dev offsets only",
            "row_index_manifest": "final/results/splits/val_idx.json",
            "count": 548,
            "metric_populations": {
                "archived_comparable_auc_and_pr_auc": (
                    "positives=current channel errors; negatives=current correctly "
                    "predicted dev rows"
                ),
                "operational_tau_precision_recall_specificity": (
                    "positives=current channel errors; negatives=other dev rows whose "
                    "current baseline prediction equals the channel source"
                ),
                "reason": (
                    "The first exactly matches the historical Qwen AUC design. The "
                    "second matches the population on which the binary source-prediction "
                    "gate can actually fire and the prospective Phase 8 implementation."
                ),
            },
        },
        "feature": {
            "site": "last-prompt-token MLP output at channel L_obs",
            "dtype_for_fit": "float64 after loading cached float32 features",
            "standardization": (
                "sklearn StandardScaler fit on the channel's full train design "
                "population only; applied unchanged to dev"
            ),
        },
        "labels": {
            "positive": (
                "current-environment baseline prediction equals channel source and "
                "gold equals channel gold"
            ),
            "negative": "all current-environment correctly predicted train rows",
            "other_rows": "excluded from that channel's Router fit",
        },
        "classifier": {
            "family": "sklearn.linear_model.LogisticRegression",
            "objective": "binary logistic loss with L2 penalty",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 2000,
            "tolerance": 0.0001,
            "random_state": 42,
            "class_weight": None,
            "fit_intercept": True,
            "convergence_rule": "solver terminates before max_iter; otherwise engineering failure",
        },
        "tau_selection": {
            "grid_ascending": [0.4, 0.5, 0.6, 0.7, 0.8],
            "criterion": (
                "smallest ascending grid threshold with dev precision >= 0.50"
            ),
            "dev_population": (
                "channel errors versus other dev rows with baseline prediction equal "
                "to the channel source"
            ),
            "undefined_precision": "not eligible",
            "provenance": [
                "scripts/phase8_gate_eval.py:175-177",
                "scripts/phase8_lib.py:304-336",
            ],
        },
        "minimum_admissible_quality": {
            "dev_ROC_AUC_floor": 0.75,
            "must_have_tau_with_dev_precision_at_least": 0.5,
            "provenance": [
                "final/results/actionability_gate_development/"
                "ACTIONABILITY_GATE_V2_SPEC.md:51-55",
                "final/results/actionability_gate_development/"
                "PROSPECTIVE_LLAMA_PROTOCOL.md:33",
            ],
            "note": (
                "The absolute floor and tau rule are inherited. This refit retains the "
                "Stage A all-correct-negative recipe rather than silently changing to "
                "the matched-pred negative population used by Gate v2."
            ),
        },
        "gate2": {
            "continuity": (
                "all Routers pass floor and each absolute dev ROC-AUC gap from archived "
                "comparable value is <= 0.05"
            ),
            "fresh_pipeline": (
                "all pass floor but any comparable gap >0.05 or exact archived Router "
                "cannot be reproduced"
            ),
            "inadequate": "any required Router dev ROC-AUC <0.75 or no eligible tau",
            "archived_comparable_dev_ROC_AUC": ARCHIVED_AUC,
        },
        "uncertainty": {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20260730,
            "method": "stratified dev resampling within positive/negative labels",
            "metrics": ["ROC_AUC", "PR_AUC", "precision", "recall", "specificity"],
        },
        "serialization": {
            "format": "NumPy NPZ with allow_pickle=False-compatible numeric arrays",
            "required": [
                "scaler_mean",
                "scaler_scale",
                "classifier_coef",
                "classifier_intercept",
                "classifier_classes",
            ],
            "hashes": "SHA256 for complete NPZ plus component array byte hashes",
        },
        "test_firewall": {
            "sequential_baseline_JSONL_parse": "FORBIDDEN",
            "allowed_access": "direct seek to emitted train/dev offsets only",
            "test_manifest_index_rows_labels_scores_aggregates": "FORBIDDEN",
        },
    }
    write_json(OUT / "qwen25_stage_b_router_refit_spec.json", router_spec)
    router_md = f"""# Qwen2.5 Stage B Router refit specification

**State: pre-fit; commit and push required before any fitting.**

This freezes the current-environment, train-only Router refit. It does not
reconstruct the missing archived pickle and does not authorize test access.

## Populations and features

- Channels: `rfi_tc`, `ca_tc`, `ca_direct`.
- Train: 2,556 committed project-train indices.
- Dev: 548 committed project-validation indices.
- Baseline metadata: direct seeks through the payload-blind train/dev byte
  index. Sequential JSONL parsing is forbidden.
- Feature: cached final-prompt-token MLP output; L20 for `rfi_tc`/`ca_tc`,
  L16 for `ca_direct`.
- Positives: current baseline source→gold channel errors.
- Negatives: all current correctly predicted train rows.
- Other rows are excluded from that channel fit.

## Frozen estimator

Fit `StandardScaler` on each channel's train design population, then
`LogisticRegression(C=1.0, penalty=L2, solver=liblinear, max_iter=2000,
tol=1e-4, random_state=42, class_weight=None, fit_intercept=True)`. Failure to
converge is an engineering failure; scientific parameters may not be changed.

## Frozen dev threshold and eligibility

Evaluate grid `[0.4, 0.5, 0.6, 0.7, 0.8]` in ascending order. Select the
smallest threshold with dev precision ≥0.50, exactly following
`scripts/phase8_gate_eval.py:175-177`. Tau precision/recall/specificity use
channel errors versus other dev rows with the same baseline source
prediction—the population on which the Router gate can actually fire
(`scripts/phase8_lib.py:304-336`).

For exact archived comparison, ROC-AUC and PR-AUC separately use channel
errors versus correctly predicted dev rows, matching the historical Qwen
design. Both populations and both metric sets are reported; they are never
silently conflated. Every channel must have comparable dev ROC-AUC
≥0.75 and an eligible operational threshold. The AUC floor is inherited from
`ACTIONABILITY_GATE_V2_SPEC.md:51-55`; it is a floor only.

The refit deliberately retains Stage A's all-correct-negative recipe. It does
not silently substitute the matched-pred negative population used in the
cross-architecture Gate v2 protocol.

Gate 2 is continuity only if all floors pass and every absolute gap from the
archived comparable dev AUC is ≤0.05; otherwise a floor-passing result is
`FRESH_ROUTER_PIPELINE_FEASIBLE`. Any floor/tau failure is
`ROUTER_INADEQUATE`.

Ten-thousand stratified dev bootstrap replicates use seed 20260730.

## Firewall

Only byte-indexed train/dev rows and train/dev activation slices may be
accessed. No test manifest, index, row, ID, label, score, aggregate, support
count, or prediction may be opened or inferred for fitting.
"""
    (OUT / "QWEN25_STAGE_B_ROUTER_REFIT_SPEC.md").write_text(
        router_md, encoding="utf-8"
    )

    prerequisite_manifest = {
        "schema_version": 1,
        "stage": "BLOCK0_BLOCK1_AND_PREFIT_SPEC",
        "constraints": {
            "GPU_used": False,
            "model_loaded": False,
            "inference_run": False,
            "stage_b_executed": False,
            "qwen3_work": False,
            "llama_or_phase10_3_modified": False,
            "direction_layer_or_dose_retuned": False,
            "test_row_id_content_label_score_access": False,
            "historical_test_aggregate_access_incident": True,
        },
        "files": {},
    }
    prerequisite_paths = [
        OUT / "BLOCK0_EVALUATION_ADMISSIBILITY_AUDIT.md",
        OUT / "block0_evaluation_admissibility_audit.json",
        OUT / "BLOCK1_GEOMETRY_VERIFICATION.md",
        OUT / "block1_geometry_verification.json",
        OUT / "block1_channel_metrics.csv",
        OUT / "QWEN25_STAGE_B_ROUTER_REFIT_SPEC.md",
        OUT / "qwen25_stage_b_router_refit_spec.json",
        OUT / "qwen25_baseline_train_dev_byte_index.csv",
        OUT / "qwen25_baseline_train_dev_byte_index.manifest.json",
    ]
    for path in prerequisite_paths:
        prerequisite_manifest["files"][str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(OUT / "BLOCK01_PREFIT_MANIFEST.json", prerequisite_manifest)

    hash_lines = [
        f"{sha256_file(path)}  {path.relative_to(ROOT)}" for path in prerequisite_paths
    ]
    (OUT / "BLOCK01_PREFIT_HASHES.sha256").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "block0": block0_payload["classification"],
                "gate1": gate1["branch"],
                "common_counts": actual_common,
                "orderings": orderings,
                "pooled_full_Q_bootstrap_CI": pooled_ci,
                "outputs": [str(path.relative_to(ROOT)) for path in prerequisite_paths],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
