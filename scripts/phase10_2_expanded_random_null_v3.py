#!/usr/bin/env python3
"""
Phase 10.2 V3: zero-first blocking same-run expanded random-null runner.

Gate-only is standard-library-only. It does not import NumPy, torch, transformers,
open model/data/cache files, generate vectors, or compute scientific endpoints.

Formal analysis is implemented behind --run-analysis and requires a new explicit
human approval. In that mode, zero is recomputed first and its 74 predictions are
an immediate blocking hard gate. Only after that gate passes are the
real/reverse/archived/fresh arms scored in the same uninterrupted process through
the same model, singleton batches, and scoring path.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "final" / "results" / "phase10_2_expanded_random_null"
V3_GATE_DIR = OUT_BASE / "v3_gate"
V3_ANALYSIS_DIR = OUT_BASE / "v3_analysis"
V3_STATE_DIR = Path("/root/autodl-tmp/phase10_2_v3_state")
V3_STATE_PATH = V3_STATE_DIR / "phase10_2_v3_same_run_raw.json"

PREEXEC_ERRATUM_V3_PATH = OUT_BASE / "PHASE10_2_PREEXECUTION_ERRATUM_V3.md"
PREREG_V3_PATH = OUT_BASE / "PHASE10_2_PREREGISTRATION_V3.md"
HASHES_V3_PATH = OUT_BASE / "PHASE10_2_HASHES_V3.json"
FREEZE_V3_PATH = OUT_BASE / "PHASE10_2_EXECUTION_FREEZE_V3.json"
AUDIT_V3_PATH = OUT_BASE / "PHASE10_2_SAME_RUN_IMPLEMENTATION_AUDIT_V3.md"
SELF_PATH = ROOT / "scripts" / "phase10_2_expanded_random_null_v3.py"

V2_ERRATUM_PATH = OUT_BASE / "PHASE10_2_PROTOCOL_ERRATUM.md"
V2_PREREG_PATH = OUT_BASE / "PHASE10_2_PREREGISTRATION_V2.md"
V2_HASHES_PATH = OUT_BASE / "PHASE10_2_HASHES_V2.json"
V2_FREEZE_PATH = OUT_BASE / "PHASE10_2_EXECUTION_FREEZE_V2.json"
V2_AUDIT_PATH = OUT_BASE / "PHASE10_2_SAME_RUN_AUDIT.md"
V2_RUNNER_PATH = ROOT / "scripts" / "phase10_2_expanded_random_null_v2.py"

V1_PREREG_PATH = OUT_BASE / "PHASE10_2_PREREGISTRATION.md"
V1_HASHES_PATH = OUT_BASE / "PHASE10_2_HASHES.json"
V1_FREEZE_PATH = OUT_BASE / "PHASE10_2_EXECUTION_FREEZE.json"
V1_RUNNER_PATH = ROOT / "scripts" / "phase10_2_expanded_random_null.py"
V1_GATE_JSON = OUT_BASE / "gate" / "phase10_2_gate_only.json"

P8_LIB_PATH = ROOT / "scripts" / "phase8_lib.py"
P10_PROBE_PATH = ROOT / "scripts" / "phase10_mctl_probe.py"
P10_RAW_PATH = ROOT / "final" / "results" / "phase10_mctl" / "phase10_mctl_raw.json"
P95_ROUTED_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase9_mechanism_and_ood"
    / "phase9_5_per_sample_routed.json"
)
VAL_PATH = ROOT / "final" / "results" / "splits" / "val_idx.json"
TRAIN_PATH = ROOT / "final" / "results" / "splits" / "train_idx.json"
P8_ACTIVATION_MANIFEST_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase8_prospective_llama"
    / "llama_activation_extraction.json"
)
P8_R0_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase8_prospective_llama"
    / "llama_r0_summary.json"
)

# The source is a full corpus. The loader decodes only the exact 74 raw offsets
# mapped from frozen validation IDs; it never opens a test manifest.
SOURCE_JSONL = (
    ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"
)
MODEL_PATH = Path("/root/autodl-tmp/models/Llama-3.1-8B-Instruct")

# Explicitly allowlisted historical geometry inputs. They may construct and
# fingerprint the deployed direction, but may never supply behavioural scores.
ACT_L22_PATH = Path("/root/autodl-tmp/phase8_cache/acts_L22.npy")
BASE_DETAILS_PATH = Path(
    "/root/autodl-tmp/phase8_cache/llama31_8b_w2c_baseline_details.jsonl"
)

EXPECTED_ROOT = "/root/autodl-tmp/sakiko-followup"
EXPECTED_BRANCH = "exp/sakiko-followup-archive"
MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_DTYPE = "bfloat16"; ATTENTION_IMPLEMENTATION = "sdpa"
N_LAYERS = 32
HIDDEN = 4096
DATASET_N = 3652
DATASET_SHUFFLE_SEED = 42

CHANNEL = "ca_tc"
SOURCE = "tool_call"
GOLD = "cannot_answer"
WRONG = ("direct", "request_for_info")
MODES = ("direct", "tool_call", "request_for_info", "cannot_answer")
OBS_LAYER = 22
INJ_LAYER = 18
ROUTER_THRESHOLD = 0.4
ROUTED_N = 74
PRIMARY_N = 68
MED_INJ = 4.6157
RHOS = (0.0, 1.0, 2.0, 3.0, 4.0)

OLD_SEED_START = 5000
OLD_K = 20
FRESH_SEED_START = 6000
FRESH_K = 79
BOOTSTRAP_SEED = 10101
BOOTSTRAPS = 10000
NEAR_ZERO = 1e-6

BATCH_SIZE = 1
SCORE_ABS_TOL = 5e-5
AUC_ABS_TOL = 1e-4
DEPLOYED_RAW_NORM = 1.430203914642334
DEPLOYED_RAW_NORM_TOL = 5e-7
UNIT_NORM_TOL = 1e-6
CORRECTION_NORM_TOL = 1e-5

RAW_RECORD_FIELDS = {
    "id",
    "uuid",
    "source",
    "gold",
    "arm",
    "random_seed",
    "direction_sha256",
    "rho",
    "response_mode_scores",
    "pred",
    "batch_index",
    "execution_order",
}

EXPECTED_ROUTED_ID_SHA = "a6aa9f7803673f4b4e0b016df4fb4f1ed6a657e69ff61cb6543108d2e2cf2bb8"
EXPECTED_PRIMARY_ID_SHA = "e358d242f2c1f251e64c157896074105270df2da7b566752f88eb2413e8b8ba1"
EXPECTED_ROUTED_UUID_SHA = "82631f98f61f52a29ee0035765258bec2630fd7948f7d8c8175d889b617a4a13"

EXPECTED_INPUT_HASHES = {
    "scripts/phase8_lib.py": "e8f314bea324a04c148a9dc20605fb690f0766f1e05eb9d569bffdbe11336d4c",
    "scripts/phase10_mctl_probe.py": "8195eeede6f2740cdf3808e14aa2e67deabd2198f3f944e6a5317bee3cb870e0",
    "final/results/phase10_mctl/phase10_mctl_raw.json": "c9324d46a417e70e2244ec154e9aff507ae9bcc126212b3e188f8a968a7d5f5d",
    "final/results/phase9_mechanism_and_ood/phase9_5_per_sample_routed.json": "525a6832fd24ac9aa2e256ac32666b0af98e2ffdd0aea8501c8bdc8ef92012fd",
    "final/results/splits/val_idx.json": "2d2f078a47ae7702b5ccceb23a86d9d4cf223d7b7cc0de4eb8090e809afa3610",
    "final/results/splits/train_idx.json": "7b843b43080e946595eb48ffbad800b5b7223cc693ccc2c9c49da47c2393b1fc",
    "final/results/phase8_prospective_llama/llama_activation_extraction.json": "99129953e3caf6fb94e203c14fa7f787c91700a9c3d7c74badaf519a0a6f6769",
    "final/results/phase8_prospective_llama/llama_r0_summary.json": "27f70675c8c30f5ca7d250b3df66cfdb5f270ed7a1271ab216d60c99bb193d20",
    "final/results/phase10_2_expanded_random_null/PHASE10_2_PREREGISTRATION.md": "3574f38c4fc914a467e49720a8f2e19e1e871cc8aed01c93a268031bbdd2dd4b",
    "final/results/phase10_2_expanded_random_null/PHASE10_2_HASHES.json": "316cab43378e0b4032622bcf293f07fab224ed3e2fbd8732c125e1ef6d0d1f28",
    "final/results/phase10_2_expanded_random_null/PHASE10_2_EXECUTION_FREEZE.json": "71b46ac09b012358a3a7b7e8098017c9911409a52067f747c954700ea5d5792b",
    "scripts/phase10_2_expanded_random_null.py": "4de85308c75c606465ce86b7fe4c739d6bc75e75bef829835b50f83bcf765809",
    "final/results/phase10_2_expanded_random_null/gate/phase10_2_gate_only.json": "985dc725ca109335b313ba6af34e14a2e502fac02ceee0f7ec097c0f0805df0c",
}

LOCAL_MODEL_METADATA_HASHES = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "generation_config.json": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6d60e9ac5ababfb940ff23bbcd424",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}


class DuplicateKeyError(ValueError):
    pass


class StructuralGateFailure(RuntimeError):
    pass


def strict_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json(path):
    return json.loads(path.read_text(), object_pairs_hook=strict_pairs)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return bytes_sha256(payload)


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def ast_assignments(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                for item, item_value in zip(target.elts, value):
                    if isinstance(item, ast.Name):
                        values[item.id] = item_value
    return values


def arm_order():
    return (
        ["zero", "deployed_real", "reverse"]
        + [f"archived_random{k}" for k in range(OLD_K)]
        + [f"fresh_random{k}" for k in range(FRESH_K)]
    )


def add_check(checks, failures, name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    if not passed:
        failures.append(name)


def gate_checks():
    checks = []
    failures = []

    def check(name, passed, detail):
        add_check(checks, failures, name, passed, detail)

    check("repository_root", str(ROOT) == EXPECTED_ROOT, ROOT)
    branch = git("branch", "--show-current")
    check("branch", branch == EXPECTED_BRANCH, branch)
    status = git("status", "--short", "--untracked-files=all")
    check("clean_worktree", status == "", status or "clean")
    divergence = git(
        "rev-list",
        "--left-right",
        "--count",
        "HEAD...origin/exp/sakiko-followup-archive",
    )
    check("upstream_divergence", divergence.replace("\t", " ") == "0 0", divergence)
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/exp/sakiko-followup-archive")
    check("local_remote_head_match", head == remote, f"{head} / {remote}")

    required = [
        PREEXEC_ERRATUM_V3_PATH,
        PREREG_V3_PATH,
        HASHES_V3_PATH,
        FREEZE_V3_PATH,
        AUDIT_V3_PATH,
        SELF_PATH,
    ]
    check(
        "v3_files_exist",
        all(path.is_file() for path in required),
        [str(path.relative_to(ROOT)) for path in required if not path.is_file()],
    )

    for rel, expected in EXPECTED_INPUT_HASHES.items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else "MISSING"
        check(f"sha256:{rel}", actual == expected, f"{expected} / {actual}")

    hashes_doc = strict_json(HASHES_V3_PATH)
    freeze = strict_json(FREEZE_V3_PATH)
    for rel, expected in hashes_doc["immutable_files"].items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else "MISSING"
        check(f"immutable_sha256:{rel}", actual == expected, f"{expected} / {actual}")

    v3_paths = {
        "preexecution_erratum_v3": PREEXEC_ERRATUM_V3_PATH,
        "preregistration_v3": PREREG_V3_PATH,
        "runner_v3": SELF_PATH,
        "implementation_audit_v3": AUDIT_V3_PATH,
    }
    for label, path in v3_paths.items():
        expected = hashes_doc["v3_files"][label]["sha256"]
        check(
            f"phase10_2_v3_hash:{label}",
            sha256(path) == expected,
            f"{expected} / {sha256(path)}",
        )
    for label, path in {**v3_paths, "hashes_v3": HASHES_V3_PATH}.items():
        expected = freeze["files"][label]["sha256"]
        check(
            f"execution_freeze_v3:{label}",
            sha256(path) == expected,
            f"{expected} / {sha256(path)}",
        )

    check("freeze_root", freeze["repository"]["root"] == EXPECTED_ROOT, freeze["repository"])
    check(
        "freeze_branch",
        freeze["repository"]["branch"] == EXPECTED_BRANCH,
        freeze["repository"],
    )
    auth = freeze["authorization"]
    check(
        "freeze_mode",
        auth["gate_only"] is True
        and auth["run_analysis"] is False
        and auth["model_load"] is False
        and auth["gpu"] is False
        and auth["random_generation"] is False
        and auth["scientific_endpoint_computation"] is False,
        auth,
    )
    same_run = freeze["same_run"]
    check(
        "same_run_freeze",
        same_run["required"] is True
        and same_run["single_model_load"] is True
        and same_run["batch_size"] == BATCH_SIZE
        and same_run["rho_grid"] == list(RHOS)
        and same_run["arm_order"] == arm_order()
        and same_run["arm_count"] == 102
        and same_run["cell_count"] == 510,
        same_run,
    )
    zero_rule = freeze["zero_blocking_gate"]
    check(
        "zero_blocking_freeze",
        zero_rule["recomputed_first"] is True
        and zero_rule["prediction_equality_n"] == ROUTED_N
        and zero_rule["before_remaining_arms"] is True
        and zero_rule["before_any_endpoint"] is True
        and zero_rule["mismatch_action"] == "IMMEDIATE_VOID_NO_RERUN",
        zero_rule,
    )
    storage = freeze["raw_trajectory_storage"]
    check(
        "raw_storage_freeze",
        storage["all_arms_all_rhos_all_ids"] is True
        and set(storage["per_record_fields"]) == RAW_RECORD_FIELDS,
        storage,
    )
    model_manifest = freeze["model_manifest"]
    check(
        "model_revision_manifest",
        model_manifest["repo"] == MODEL_REPO
        and model_manifest["revision"] == MODEL_REVISION
        and model_manifest["dtype"] == MODEL_DTYPE
        and model_manifest["attention_implementation"] == ATTENTION_IMPLEMENTATION
        and model_manifest["local_metadata_sha256"] == LOCAL_MODEL_METADATA_HASHES,
        {
            "repo": model_manifest["repo"],
            "revision": model_manifest["revision"],
            "dtype": model_manifest["dtype"],
            "attention": model_manifest["attention_implementation"],
        },
    )
    primary_freeze = freeze["primary"]
    check(
        "primary_hierarchy_freeze",
        "exactly 79" in primary_freeze["sole_endpoint"]
        and primary_freeze["archived_k20_in_primary"] is False
        and len(primary_freeze["success_requires_all"]) == 3
        and freeze["endpoint_hierarchy"]["sole_confirmatory"] == "T_AUC_REAL_VS_FRESH79"
        and freeze["endpoint_hierarchy"]["stored_full_trajectory_adds_endpoints"] is False,
        {
            "primary": primary_freeze,
            "hierarchy": freeze["endpoint_hierarchy"],
        },
    )

    raw = strict_json(P10_RAW_PATH)
    p95 = strict_json(P95_ROUTED_PATH)
    val_ids = set(map(int, strict_json(VAL_PATH)))
    frozen = raw["frozen_baseline"]
    routed_ids = list(frozen)
    routed_ints = list(map(int, routed_ids))
    routed_uuids = [frozen[idx]["uuid"] for idx in routed_ids]
    primary_ids = sorted(
        int(idx) for idx in routed_ids if frozen[idx]["is_own_channel_err"] is True
    )
    population_ok = (
        len(routed_ids) == ROUTED_N
        and len(set(routed_ids)) == ROUTED_N
        and len(primary_ids) == PRIMARY_N
        and set(routed_ints).issubset(val_ids)
        and canonical_hash(routed_ints) == EXPECTED_ROUTED_ID_SHA
        and canonical_hash(primary_ids) == EXPECTED_PRIMARY_ID_SHA
        and canonical_hash(routed_uuids) == EXPECTED_ROUTED_UUID_SHA
    )
    check(
        "frozen_population_order",
        population_ok,
        f"routed={len(routed_ids)} primary={len(primary_ids)}",
    )
    p95_map = {str(row["idx"]): row for row in p95}
    check(
        "independent_routed_provenance",
        set(p95_map) == set(routed_ids)
        and all(p95_map[idx]["uuid"] == frozen[idx]["uuid"] for idx in routed_ids),
        f"rows={len(p95_map)}",
    )

    config = raw["config"]
    config_ok = (
        config["channel"] == CHANNEL
        and config["obs"] == OBS_LAYER
        and config["inj"] == INJ_LAYER
        and config["thr"] == ROUTER_THRESHOLD
        and config["rhos"] == [1.0, 2.0, 3.0, 4.0]
        and config["n_routed"] == ROUTED_N
        and config["med_inj"] == MED_INJ
        and config["seed_block"] == OLD_SEED_START
        and config["n_random"] == OLD_K
        and abs(
            config["directions"]["deployed"]["raw_norm"] - DEPLOYED_RAW_NORM
        )
        <= DEPLOYED_RAW_NORM_TOL
    )
    check("historical_config", config_ok, config)

    expected_reference_cells = (
        {"zero"}
        | {f"deployed|real|{rho:.1f}" for rho in RHOS[1:]}
        | {f"deployed|reverse|{rho:.1f}" for rho in RHOS[1:]}
        | {
            f"random{k}|{rho:.1f}"
            for k in range(OLD_K)
            for rho in RHOS[1:]
        }
    )
    check(
        "historical_drift_reference_complete",
        expected_reference_cells.issubset(set(raw["cells"]))
        and all(
            len(raw["cells"][cell]["per_sample"]) == ROUTED_N
            for cell in expected_reference_cells
        ),
        f"required={len(expected_reference_cells)}",
    )

    old_seeds = set(range(OLD_SEED_START, OLD_SEED_START + OLD_K))
    fresh_seeds = set(range(FRESH_SEED_START, FRESH_SEED_START + FRESH_K))
    check(
        "seed_disjointness",
        len(old_seeds) == OLD_K
        and len(fresh_seeds) == FRESH_K
        and not old_seeds.intersection(fresh_seeds),
        sorted(old_seeds.intersection(fresh_seeds)),
    )
    batches = [[idx] for idx in routed_ids]
    check(
        "fixed_singleton_batch_partition",
        len(batches) == ROUTED_N
        and all(len(batch) == BATCH_SIZE for batch in batches),
        canonical_hash(batches),
    )

    tree = ast.parse(SELF_PATH.read_text())
    function_nodes = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required_functions = {
        "build_remaining_vectors",
        "run_zero_arm",
        "zero_prediction_gate",
        "run_remaining_arms_same_process",
        "validate_full_trajectory_storage",
        "audit_archived_continuity",
        "compute_within_run_endpoints",
    }
    check(
        "same_run_code_paths_present",
        required_functions.issubset(function_nodes),
        sorted(required_functions - set(function_nodes)),
    )

    def call_name(node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    run_node = function_nodes.get("run_analysis")
    calls = sorted(
        (
            (node.lineno, call_name(node))
            for node in ast.walk(run_node)
            if isinstance(node, ast.Call)
        ),
        key=lambda item: item[0],
    ) if run_node else []
    call_lines = {}
    for line, name in calls:
        call_lines.setdefault(name, line)
    required_order = [
        "load_model_once",
        "run_zero_arm",
        "zero_prediction_gate",
        "build_remaining_vectors",
        "run_remaining_arms_same_process",
        "validate_full_trajectory_storage",
        "compute_within_run_endpoints",
        "audit_archived_continuity",
        "write_results",
    ]
    order_ok = all(name in call_lines for name in required_order) and all(
        call_lines[left] < call_lines[right]
        for left, right in zip(required_order, required_order[1:])
    )
    check("zero_first_pre_endpoint_call_order", order_ok, call_lines)

    zero_node = function_nodes.get("zero_prediction_gate")
    zero_calls = {
        call_name(node)
        for node in ast.walk(zero_node)
        if isinstance(node, ast.Call)
    } if zero_node else set()
    check(
        "zero_gate_contains_no_endpoint_call",
        "compute_within_run_endpoints" not in zero_calls
        and "summarize_population" not in zero_calls
        and "empirical" not in zero_calls
        and "percentile_ci" not in zero_calls,
        sorted(zero_calls),
    )
    source_text = SELF_PATH.read_text()
    builder_text = ast.get_source_segment(
        source_text, function_nodes["build_remaining_vectors"]
    )
    zero_text = ast.get_source_segment(source_text, function_nodes["run_zero_arm"])
    remaining_text = ast.get_source_segment(
        source_text, function_nodes["run_remaining_arms_same_process"]
    )
    endpoint_text = ast.get_source_segment(
        source_text, function_nodes["compute_within_run_endpoints"]
    )
    check(
        "random_vector_builder_identity",
        builder_text.count("random_unit(seed)") == 2
        and "old_seeds" in builder_text
        and "fresh_seeds" in builder_text,
        "archived and fresh seed blocks call the same random_unit(seed)",
    )
    check(
        "common_arm_scoring_path",
        "run_arm_cells(" in zero_text
        and "run_arm_cells(" in remaining_text,
        "zero and every remaining arm use run_arm_cells",
    )
    check(
        "confirmatory_state_only",
        "raw" not in [arg.arg for arg in function_nodes["compute_within_run_endpoints"].args.args]
        and "reference_cell(" not in endpoint_text
        and "reference_curve(" not in endpoint_text,
        "confirmatory endpoint consumes only same-run state/context",
    )
    check(
        "verdict_and_withdrawal_language",
        "Phase 10.2 is VOID. No Phase 10.2 scientific result or claim is permitted."
        in source_text
        and "same-run recomputed deployed" in source_text
        and "did not independently confirm" in source_text,
        "frozen success, non-success, and VOID wording present",
    )

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    check(
        "no_legacy_scientific_import",
        "phase8_lib" not in imported_modules
        and "phase10_2_expanded_random_null" not in imported_modules,
        imported_modules,
    )

    v1_output_absent = (
        not (OUT_BASE / "analysis").exists()
        and not Path(
            "/root/autodl-tmp/phase10_2_state/phase10_2_fresh_random_state.json"
        ).exists()
    )
    v2_output_absent = (
        not Path("/root/autodl-tmp/phase10_2_v2_state").exists()
        and not (OUT_BASE / "v2_analysis").exists()
    )
    v3_output_absent = not V3_STATE_PATH.exists() and not V3_ANALYSIS_DIR.exists()
    check("v1_superseded_scientifically_unrun", v1_output_absent, v1_output_absent)
    check("v2_superseded_scientifically_unrun", v2_output_absent, v2_output_absent)
    check("no_v3_state_or_analysis_output", v3_output_absent, v3_output_absent)

    amendment_text = PREEXEC_ERRATUM_V3_PATH.read_text()
    check(
        "amendment_rules_separated",
        "Pre-execution design corrections" in amendment_text
        and "Post-execution amendments" in amendment_text
        and "do not consume a post-execution amendment budget" in amendment_text,
        "pre-execution corrections separated from post-execution firewall",
    )

    historical_probe = P10_PROBE_PATH.read_text()
    audit_text = AUDIT_V3_PATH.read_text()
    check(
        "historical_singleton_batching_continuity",
        "for i in idxs:" in historical_probe
        and "L.predict(model, tok, dev, ds[i], corr, inj)" in historical_probe
        and "BATCHING_CONTINUITY_CONFIRMED" in audit_text,
        "Phase 10 model scoring was sequential per-sample; Phase 10.1 was CPU-only",
    )
    check(
        "raw_record_schema_declared",
        set(freeze["raw_trajectory_storage"]["per_record_fields"])
        == RAW_RECORD_FIELDS,
        sorted(RAW_RECORD_FIELDS),
    )
    top_level_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    check(
        "test_cache_firewall_static",
        not any(name.startswith("TEST") for name in top_level_names)
        and auth["test_access"] is False
        and auth["cache_access"] is False
        and SOURCE_JSONL.name == "w2c_test_mcq.jsonl",
        "no test-manifest constant; exact validation-only selective loader frozen",
    )

    return raw, p95_map, checks, failures, {
        "routed_ids": routed_ids,
        "primary_ids": list(map(str, primary_ids)),
        "batches": batches,
        "batch_sha256": canonical_hash(batches),
        "old_seeds": sorted(old_seeds),
        "fresh_seeds": sorted(fresh_seeds),
        "frozen": frozen,
    }

def write_gate_only(checks, failures, context, elapsed):
    V3_GATE_DIR.mkdir(parents=True, exist_ok=True)
    passed = not failures
    status = "READY_FOR_PHASE10_2_GPU_APPROVAL" if passed else "BLOCKED"
    payload = {
        "status": status,
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "context": {
            "routed_n": len(context["routed_ids"]),
            "primary_n": len(context["primary_ids"]),
            "batch_size": BATCH_SIZE,
            "batch_partition_sha256": context["batch_sha256"],
            "arm_count": len(arm_order()),
            "arm_order": arm_order(),
            "rho_grid": list(RHOS),
            "archived_seeds": context["old_seeds"],
            "fresh_seeds": context["fresh_seeds"],
            "raw_record_fields": sorted(RAW_RECORD_FIELDS),
        },
        "gate_only": True,
        "run_analysis_executed": False,
        "zero_actual_model_reproduction_run": False,
        "runner_zero_recomputation_is_first_scored_arm": True,
        "runner_zero_prediction_equality_is_blocking": True,
        "runner_endpoints_unreachable_before_zero_gate": True,
        "model_loaded": False,
        "gpu_used": False,
        "random_vectors_generated": False,
        "new_predictions_generated": False,
        "phase10_2_scientific_endpoints_computed": False,
        "formal_gpu_analysis_run": False,
        "elapsed_cpu_wall_seconds": elapsed,
    }
    (V3_GATE_DIR / "phase10_2_v3_gate_only.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Phase 10.2 V3 — Gate-Only Report",
        "",
        f"**Status:** **{status}**",
        f"**Checks passed:** {sum(c['passed'] for c in checks)}/{len(checks)}",
        "**Zero actual model reproduction has run:** false",
        "**Model loaded:** false",
        "**GPU used:** false",
        "**Random vectors generated:** false",
        "**Scientific endpoints computed:** false",
        "**Formal GPU analysis run:** false",
        "",
        "The formal runner recomputes zero first and exact equality of all 74",
        "predictions is a blocking hard gate before any remaining arm or endpoint.",
        "Formal GPU analysis still requires separate explicit human approval.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if item['passed'] else 'FAIL'} — `{item['check']}`: {item['detail']}"
        for item in checks
    )
    (V3_GATE_DIR / "PHASE10_2_V3_GATE_ONLY_REPORT.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(status)


# Scientific dependencies are loaded only after --run-analysis passes the gate.
np = None
torch = None
transformers = None
AutoModelForCausalLM = None
AutoTokenizer = None


def load_scientific_dependencies():
    global np, torch, transformers, AutoModelForCausalLM, AutoTokenizer
    import numpy as np_module
    import torch as torch_module
    import transformers as transformers_module
    from transformers import AutoModelForCausalLM as ModelClass
    from transformers import AutoTokenizer as TokenizerClass

    np = np_module
    torch = torch_module
    transformers = transformers_module
    AutoModelForCausalLM = ModelClass
    AutoTokenizer = TokenizerClass


def set_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def parse_tools(tools):
    out = []
    for tool in tools or []:
        if isinstance(tool, str):
            try:
                out.append(json.loads(tool))
            except json.JSONDecodeError:
                out.append({"raw": tool})
        else:
            out.append(tool)
    return out


def build_prompt_messages(question, tools):
    parts = ["You are a helpful assistant."]
    if tools:
        parts.append(
            "You have access to the following tools:\n"
            + json.dumps(tools, indent=2, ensure_ascii=False)
        )
    else:
        parts.append("No tools are available.")
    parts.append(
        "Given the user's question, choose the most appropriate response "
        "from the provided options."
    )
    return [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": question},
    ]


def load_selected_jsonl_lines(path, wanted_indices):
    wanted = set(map(int, wanted_indices))
    selected = {}
    with path.open() as handle:
        for line_index, line in enumerate(handle):
            if line_index in wanted:
                selected[line_index] = json.loads(line)
    if set(selected) != wanted:
        raise StructuralGateFailure(
            f"selective loader incomplete for {path}: {len(selected)}/{len(wanted)}"
        )
    return selected


def load_selected_validation_rows(routed_ids, frozen):
    permutation = np.random.default_rng(DATASET_SHUFFLE_SEED).permutation(DATASET_N)
    raw_to_routed = {int(permutation[int(idx)]): str(idx) for idx in routed_ids}
    raw_rows = load_selected_jsonl_lines(SOURCE_JSONL, raw_to_routed)
    selected = {}
    for raw_idx, row in raw_rows.items():
        routed_idx = raw_to_routed[raw_idx]
        if row.get("uuid") != frozen[routed_idx]["uuid"]:
            raise StructuralGateFailure(
                f"validation UUID mismatch routed={routed_idx} raw={raw_idx}"
            )
        selected[routed_idx] = row
    if list(selected) != [idx for idx in routed_ids]:
        selected = {idx: selected[idx] for idx in routed_ids}
    return selected


def build_deployed_direction(p95_map, routed_ids):
    train_ids = list(map(int, strict_json(TRAIN_PATH)))
    train_details = load_selected_jsonl_lines(BASE_DETAILS_PATH, train_ids)
    activations = np.load(ACT_L22_PATH, mmap_mode="r")
    errors = [
        idx
        for idx in train_ids
        if train_details[idx]["gold"] == GOLD
        and train_details[idx]["pred"] == SOURCE
    ]
    references = [
        idx
        for idx in train_ids
        if train_details[idx]["gold"] == GOLD
        and train_details[idx]["pred"] == GOLD
    ]
    if len(errors) != 383 or len(references) != 146:
        raise StructuralGateFailure(
            f"deployed direction membership mismatch: {len(errors)}/{len(references)}"
        )
    raw = (
        np.asarray(activations[references], dtype=np.float32).mean(axis=0)
        - np.asarray(activations[errors], dtype=np.float32).mean(axis=0)
    )
    raw_norm = float(np.linalg.norm(raw))
    if abs(raw_norm - DEPLOYED_RAW_NORM) > DEPLOYED_RAW_NORM_TOL:
        raise StructuralGateFailure(
            f"deployed raw norm mismatch: {raw_norm} != {DEPLOYED_RAW_NORM}"
        )
    direction = (raw / (raw_norm + 1e-12)).astype(np.float32)
    projection_mismatches = []
    for idx in routed_ids:
        observed = round(float(np.asarray(activations[int(idx)]) @ direction), 4)
        expected = float(p95_map[idx]["proj22_deployed"])
        if observed != expected:
            projection_mismatches.append((idx, expected, observed))
    if projection_mismatches:
        raise StructuralGateFailure(
            f"deployed projection fingerprint mismatch: {projection_mismatches[:5]}"
        )
    return direction, {
        "algorithm": "unit(mean(correct cannot_answer) - mean(ca_tc errors))",
        "n_errors": len(errors),
        "n_references": len(references),
        "raw_norm": raw_norm,
        "direction_sha256": bytes_sha256(direction.tobytes(order="C")),
        "acts_L22_sha256": sha256(ACT_L22_PATH),
        "baseline_details_sha256": sha256(BASE_DETAILS_PATH),
        "projection_fingerprint_rows": len(routed_ids),
        "projection_precision": 4,
    }


def random_unit(seed):
    rng = np.random.RandomState(seed)
    vector = rng.randn(HIDDEN).astype(np.float32)
    unit = vector / (np.linalg.norm(vector) + 1e-12)
    return unit.astype(np.float32)


def vector_manifest_entry(vector, seed=None):
    norm = float(np.linalg.norm(vector))
    return {
        "seed": seed,
        "norm": norm,
        "sha256": bytes_sha256(vector.tobytes(order="C")),
    }


def build_remaining_vectors(deployed, old_seeds, fresh_seeds):
    vectors = {
        "deployed_real": deployed.astype(np.float32, copy=True),
        "reverse": (-deployed).astype(np.float32, copy=True),
    }
    for k, seed in enumerate(old_seeds):
        vectors[f"archived_random{k}"] = random_unit(seed)
    for k, seed in enumerate(fresh_seeds):
        vectors[f"fresh_random{k}"] = random_unit(seed)
    if list(vectors) != arm_order()[1:]:
        raise StructuralGateFailure("remaining-arm order mismatch while constructing vectors")
    manifest = {}
    for arm, vector in vectors.items():
        seed = (
            old_seeds[int(arm.removeprefix("archived_random"))]
            if arm.startswith("archived_random")
            else fresh_seeds[int(arm.removeprefix("fresh_random"))]
            if arm.startswith("fresh_random")
            else None
        )
        entry = vector_manifest_entry(vector, seed)
        if abs(entry["norm"] - 1.0) > UNIT_NORM_TOL:
            raise StructuralGateFailure(f"unit norm mismatch {arm}: {entry['norm']}")
        manifest[arm] = entry
    return vectors, manifest

def load_model_once():
    for name, expected in LOCAL_MODEL_METADATA_HASHES.items():
        path = MODEL_PATH / name
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            raise StructuralGateFailure(f"local model metadata mismatch {name}: {actual}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=ATTENTION_IMPLEMENTATION,
        local_files_only=True,
    )
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        raise StructuralGateFailure("model dtype mismatch")
    if getattr(model.config, "quantization_config", None) is not None:
        raise StructuralGateFailure("quantized model is forbidden")
    if model.config.num_hidden_layers != N_LAYERS:
        raise StructuralGateFailure("model layer-count mismatch")
    if model.config.hidden_size != HIDDEN:
        raise StructuralGateFailure("model hidden-size mismatch")
    return model, tokenizer, next(model.parameters()).device


def prompt_text(tokenizer, sample):
    messages = build_prompt_messages(
        sample["question"], parse_tools(sample["tools"])
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def score_candidate(model, tokenizer, device, prompt_ids, candidate, correction):
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        return -1e9
    inputs = torch.cat(
        [prompt_ids, torch.tensor([candidate_ids], device=device)], dim=1
    )
    prompt_length = prompt_ids.shape[1]
    candidate_length = len(candidate_ids)
    correction_tensor = torch.from_numpy(correction).to(torch.bfloat16).to(device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output[0][:, :, :] += correction_tensor
            return output
        output[:, :, :] += correction_tensor
        return output

    handle = model.model.layers[INJ_LAYER].mlp.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            logits = model(inputs).logits
    finally:
        handle.remove()
    selected = logits[
        0,
        prompt_length - 1 : prompt_length + candidate_length - 1,
        :,
    ]
    log_probs = torch.log_softmax(selected.float(), dim=-1)
    value = (
        sum(log_probs[i, candidate_ids[i]].item() for i in range(candidate_length))
        / candidate_length
    )
    del inputs
    return float(value)


def score_sample(model, tokenizer, device, sample, correction):
    text = prompt_text(tokenizer, sample)
    prompt_ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=False)], device=device
    )
    scores = {}
    for mode in MODES:
        candidate = sample["answers"].get(mode, "")
        scores[mode] = (
            score_candidate(
                model, tokenizer, device, prompt_ids, candidate, correction
            )
            if candidate
            else -1e9
        )
    del prompt_ids
    return {
        "pred": max(MODES, key=lambda mode: scores[mode]),
        "logp": {mode: round(float(scores[mode]), 5) for mode in MODES},
    }


def score_batch(model, tokenizer, device, selected_rows, batch, correction):
    if len(batch) != BATCH_SIZE:
        raise StructuralGateFailure("batch-size deviation")
    return {
        idx: score_sample(
            model, tokenizer, device, selected_rows[idx], correction
        )
        for idx in batch
    }


def environment_manifest(model, tokenizer, context):
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()[0]
    except Exception as exc:
        driver = f"UNAVAILABLE:{type(exc).__name__}"
    return {
        "gpu_model": torch.cuda.get_device_name(0),
        "nvidia_driver": driver,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH),
        "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": getattr(
            model.config, "_attn_implementation", "UNDECLARED"
        ),
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "autocast_enabled": torch.is_autocast_enabled(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "scoring": {
            "candidate_order": list(MODES),
            "average_candidate_token_log_probability": True,
            "log_softmax_dtype": "float32",
            "generation": False,
            "injection_layer": INJ_LAYER,
            "injection_site": "MLP output, all positions",
            "rho_grid": list(RHOS),
            "med_inj": MED_INJ,
        },
        "batch_size": BATCH_SIZE,
        "batches": context["batches"],
        "batch_partition_sha256": context["batch_sha256"],
        "arm_order": arm_order(),
        "archived_seeds": context["old_seeds"],
        "fresh_seeds": context["fresh_seeds"],
        "vectors": {},
        "deployed_direction": None,
        "tokenizer_class": tokenizer.__class__.__name__,
    }


def persist_state(state):
    temp = V3_STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False))
    os.replace(temp, V3_STATE_PATH)


def run_arm_cells(
    arm,
    vector,
    vector_info,
    model,
    tokenizer,
    device,
    selected_rows,
    context,
    state,
):
    for rho in RHOS:
        key = f"{arm}|{rho:.1f}"
        correction = (rho * MED_INJ * vector).astype(np.float32)
        expected_norm = 0.0 if arm == "zero" or rho == 0 else rho * MED_INJ
        correction_norm = float(np.linalg.norm(correction))
        if abs(correction_norm - expected_norm) > CORRECTION_NORM_TOL:
            raise StructuralGateFailure(
                f"correction norm mismatch {key}: {correction_norm}"
            )
        per_sample = {}
        cell_execution_order = len(state["cell_order"])
        for batch_index, batch in enumerate(context["batches"]):
            batch_result = score_batch(
                model,
                tokenizer,
                device,
                selected_rows,
                batch,
                correction,
            )
            if list(batch_result) != batch:
                raise StructuralGateFailure(f"batch membership mismatch {key}")
            for idx in batch:
                scored = batch_result[idx]
                frozen = context["frozen"][idx]
                scores = scored["logp"]
                record = {
                    "id": int(idx),
                    "uuid": frozen["uuid"],
                    "source": SOURCE,
                    "gold": frozen["gold"],
                    "arm": arm,
                    "random_seed": vector_info["seed"],
                    "direction_sha256": vector_info["sha256"],
                    "rho": rho,
                    "response_mode_scores": scores,
                    "logp": scores,
                    "pred": scored["pred"],
                    "batch_index": batch_index,
                    "execution_order": state["next_execution_order"],
                }
                state["next_execution_order"] += 1
                per_sample[idx] = record
        if list(per_sample) != context["routed_ids"]:
            raise StructuralGateFailure(f"sample order mismatch {key}")
        state["cells"][key] = {
            "arm": arm,
            "random_seed": vector_info["seed"],
            "direction_sha256": vector_info["sha256"],
            "rho": rho,
            "correction_norm": correction_norm,
            "batch_partition_sha256": context["batch_sha256"],
            "cell_execution_order": cell_execution_order,
            "per_sample": per_sample,
        }
        state["cell_order"].append(key)
        persist_state(state)


def run_zero_arm(model, tokenizer, device, selected_rows, context, state):
    zero = np.zeros(HIDDEN, dtype=np.float32)
    zero_info = vector_manifest_entry(zero, None)
    if abs(zero_info["norm"]) > UNIT_NORM_TOL:
        raise StructuralGateFailure("zero-vector norm mismatch")
    state["environment_manifest"]["vectors"] = {"zero": zero_info}
    run_arm_cells(
        "zero",
        zero,
        zero_info,
        model,
        tokenizer,
        device,
        selected_rows,
        context,
        state,
    )


def zero_prediction_gate(state, context):
    mismatches = []
    frozen = context["frozen"]
    for rho in RHOS:
        cell = state["cells"][f"zero|{rho:.1f}"]["per_sample"]
        for idx in context["routed_ids"]:
            expected = frozen[idx]["baseline_pred"]
            observed = cell[idx]["pred"]
            if observed != expected:
                mismatches.append(
                    {
                        "id": int(idx),
                        "uuid": frozen[idx]["uuid"],
                        "rho": rho,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    state["zero_prediction_gate"] = {
        "status": "PASS" if not mismatches else "FAIL",
        "unique_prediction_count": ROUTED_N,
        "zero_cells_checked": len(RHOS),
        "mismatches": mismatches,
        "scientific_endpoints_permitted": not mismatches,
    }
    persist_state(state)
    return mismatches


def run_remaining_arms_same_process(
    model, tokenizer, device, selected_rows, vectors, vector_manifest, context, state
):
    if list(vectors) != arm_order()[1:]:
        raise StructuralGateFailure("remaining arm execution order mismatch")
    for arm in arm_order()[1:]:
        run_arm_cells(
            arm,
            vectors[arm],
            vector_manifest[arm],
            model,
            tokenizer,
            device,
            selected_rows,
            context,
            state,
        )


def validate_full_trajectory_storage(state, context):
    expected_order = [
        f"{arm}|{rho:.1f}" for arm in arm_order() for rho in RHOS
    ]
    if state["cell_order"] != expected_order:
        raise StructuralGateFailure(
            f"same-run cell order mismatch: {len(state['cell_order'])}/{len(expected_order)}"
        )
    if list(state["cells"]) != expected_order:
        raise StructuralGateFailure("same-run state key order mismatch")
    observed_execution_orders = []
    for cell_index, key in enumerate(expected_order):
        cell = state["cells"][key]
        if cell["cell_execution_order"] != cell_index:
            raise StructuralGateFailure(f"cell execution order mismatch {key}")
        if cell["batch_partition_sha256"] != context["batch_sha256"]:
            raise StructuralGateFailure(f"batch partition mismatch {key}")
        if list(cell["per_sample"]) != context["routed_ids"]:
            raise StructuralGateFailure(f"ID order mismatch {key}")
        for batch_index, idx in enumerate(context["routed_ids"]):
            record = cell["per_sample"][idx]
            if not RAW_RECORD_FIELDS.issubset(record):
                missing = sorted(RAW_RECORD_FIELDS - set(record))
                raise StructuralGateFailure(f"raw schema missing {key} {idx}: {missing}")
            if record["id"] != int(idx):
                raise StructuralGateFailure(f"raw ID mismatch {key} {idx}")
            if record["uuid"] != context["frozen"][idx]["uuid"]:
                raise StructuralGateFailure(f"raw UUID mismatch {key} {idx}")
            if record["source"] != SOURCE or record["gold"] != context["frozen"][idx]["gold"]:
                raise StructuralGateFailure(f"raw source/gold mismatch {key} {idx}")
            if record["arm"] != cell["arm"] or record["rho"] != cell["rho"]:
                raise StructuralGateFailure(f"raw arm/rho mismatch {key} {idx}")
            if record["random_seed"] != cell["random_seed"]:
                raise StructuralGateFailure(f"raw seed mismatch {key} {idx}")
            if record["direction_sha256"] != cell["direction_sha256"]:
                raise StructuralGateFailure(f"raw direction hash mismatch {key} {idx}")
            if record["batch_index"] != batch_index:
                raise StructuralGateFailure(f"raw batch index mismatch {key} {idx}")
            if set(record["response_mode_scores"]) != set(MODES):
                raise StructuralGateFailure(f"mode mismatch {key} {idx}")
            if record["response_mode_scores"] != record["logp"]:
                raise StructuralGateFailure(f"score alias mismatch {key} {idx}")
            if not all(
                math.isfinite(float(value))
                for value in record["response_mode_scores"].values()
            ):
                raise StructuralGateFailure(f"nonfinite score {key} {idx}")
            observed_execution_orders.append(record["execution_order"])
    expected_records = len(arm_order()) * len(RHOS) * ROUTED_N
    if observed_execution_orders != list(range(expected_records)):
        raise StructuralGateFailure("per-record execution order is not contiguous")
    state["full_trajectory_storage_validated"] = {
        "status": "PASS",
        "arms": len(arm_order()),
        "rhos": len(RHOS),
        "ids": ROUTED_N,
        "records": expected_records,
        "required_fields": sorted(RAW_RECORD_FIELDS),
    }
    persist_state(state)

def st_value(logp, baseline):
    source_terms = [
        (float(logp[mode]) - float(logp[SOURCE]))
        - (float(baseline[mode]) - float(baseline[SOURCE]))
        for mode in MODES
        if mode != SOURCE
    ]
    target_terms = [
        (float(logp[GOLD]) - float(logp[wrong]))
        - (float(baseline[GOLD]) - float(baseline[wrong]))
        for wrong in WRONG
    ]
    return float(np.mean(source_terms)), float(np.mean(target_terms))


def auc(values):
    widths = np.diff(np.asarray(RHOS))
    return np.sum(
        0.5 * (values[:, :-1] + values[:, 1:]) * widths, axis=1
    ) / 4.0


def curve_from_same_run(state, arm, routed_ids):
    baseline = state["cells"]["zero|0.0"]["per_sample"]
    s = np.zeros((len(routed_ids), len(RHOS)), dtype=float)
    t = np.zeros_like(s)
    for rho_index, rho in enumerate(RHOS):
        cell = state["cells"][f"{arm}|{rho:.1f}"]["per_sample"]
        for position, idx in enumerate(routed_ids):
            s[position, rho_index], t[position, rho_index] = st_value(
                cell[idx]["logp"], baseline[idx]["logp"]
            )
    return {"S": s, "T": t}


def reference_cell(raw, arm, rho):
    if rho == 0.0 or arm == "zero":
        return raw["cells"]["zero"]["per_sample"]
    if arm == "deployed_real":
        return raw["cells"][f"deployed|real|{rho:.1f}"]["per_sample"]
    if arm == "reverse":
        return raw["cells"][f"deployed|reverse|{rho:.1f}"]["per_sample"]
    if arm.startswith("archived_random"):
        k = int(arm.removeprefix("archived_random"))
        return raw["cells"][f"random{k}|{rho:.1f}"]["per_sample"]
    raise KeyError(arm)


def reference_curve(raw, arm, routed_ids):
    baseline = raw["cells"]["zero"]["per_sample"]
    s = np.zeros((len(routed_ids), len(RHOS)), dtype=float)
    t = np.zeros_like(s)
    for rho_index, rho in enumerate(RHOS):
        cell = reference_cell(raw, arm, rho)
        for position, idx in enumerate(routed_ids):
            s[position, rho_index], t[position, rho_index] = st_value(
                cell[idx]["logp"], baseline[idx]["logp"]
            )
    return {"S": s, "T": t}


def audit_archived_continuity(state, raw, context):
    if state.get("zero_prediction_gate", {}).get("status") != "PASS":
        raise StructuralGateFailure("archived continuity audit reached before zero PASS")
    routed_ids = context["routed_ids"]
    arms = ["zero", "deployed_real", "reverse"] + [
        f"archived_random{k}" for k in range(OLD_K)
    ]
    prediction_mismatches = []
    max_score_delta = 0.0
    max_score_location = None
    per_arm = {}
    for arm in arms:
        arm_pred_mismatch = 0
        arm_score_max = 0.0
        for rho in RHOS:
            current = state["cells"][f"{arm}|{rho:.1f}"]["per_sample"]
            reference = reference_cell(raw, arm, rho)
            for idx in routed_ids:
                if current[idx]["pred"] != reference[idx]["pred"]:
                    arm_pred_mismatch += 1
                    prediction_mismatches.append(
                        {
                            "arm": arm,
                            "rho": rho,
                            "id": idx,
                            "old": reference[idx]["pred"],
                            "new": current[idx]["pred"],
                        }
                    )
                for mode in MODES:
                    delta = float(current[idx]["logp"][mode]) - float(
                        reference[idx]["logp"][mode]
                    )
                    if abs(delta) > arm_score_max:
                        arm_score_max = abs(delta)
                    if abs(delta) > max_score_delta:
                        max_score_delta = abs(delta)
                        max_score_location = {
                            "arm": arm,
                            "rho": rho,
                            "id": idx,
                            "mode": mode,
                            "signed_delta": delta,
                        }
        current_curve = curve_from_same_run(state, arm, routed_ids)
        old_curve = reference_curve(raw, arm, routed_ids)
        s_delta = auc(current_curve["S"]) - auc(old_curve["S"])
        t_delta = auc(current_curve["T"]) - auc(old_curve["T"])
        per_arm[arm] = {
            "prediction_mismatches": arm_pred_mismatch,
            "max_abs_score_delta": arm_score_max,
            "max_abs_sample_S_AUC_delta": float(np.max(np.abs(s_delta))),
            "max_abs_sample_T_AUC_delta": float(np.max(np.abs(t_delta))),
            "mean_S_AUC_signed_delta": float(np.mean(s_delta)),
            "mean_T_AUC_signed_delta": float(np.mean(t_delta)),
        }
    max_auc_delta = max(
        max(
            details["max_abs_sample_S_AUC_delta"],
            details["max_abs_sample_T_AUC_delta"],
        )
        for details in per_arm.values()
    )
    continuity = (
        not prediction_mismatches
        and max_score_delta <= SCORE_ABS_TOL
        and max_auc_delta <= AUC_ABS_TOL
    )
    return {
        "branch": (
            "A_ENVIRONMENT_CONTINUITY_CONFIRMED"
            if continuity
            else "B_NUMERICAL_DRIFT_DETECTED_WITHIN_RUN_VALID"
        ),
        "scientific_endpoints_permitted": True,
        "zero_prediction_gate": "PASS",
        "prediction_mismatch_count": len(prediction_mismatches),
        "prediction_mismatches": prediction_mismatches,
        "max_abs_score_delta": max_score_delta,
        "max_score_delta_location": max_score_location,
        "max_abs_sample_auc_delta": max_auc_delta,
        "tolerances": {
            "score_abs": SCORE_ABS_TOL,
            "auc_abs": AUC_ABS_TOL,
        },
        "per_arm": per_arm,
        "continuity_claim_permitted": continuity,
    }

def percentile_ci(values, bootstrap_indices):
    draws = np.mean(values[bootstrap_indices], axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def empirical(real, fresh_values):
    fresh = np.asarray(fresh_values, dtype=float)
    exceedance = int(np.sum(fresh >= real))
    return {
        "real": float(real),
        "fresh_values": [float(value) for value in fresh],
        "fresh_k": FRESH_K,
        "fresh_exceedance_count": exceedance,
        "rank_desc_ties_against_real": 1 + exceedance,
        "p_add_one": float((1 + exceedance) / 80),
        "fresh_mean": float(np.mean(fresh)),
        "fresh_sample_sd": float(np.std(fresh, ddof=1)),
        "fresh_min": float(np.min(fresh)),
        "fresh_max": float(np.max(fresh)),
        "real_minus_fresh_mean": float(real - np.mean(fresh)),
    }


def summarize_population(positions, real, reverse, fresh_curves, bootstrap):
    real_s = auc(real["S"])[positions]
    real_t = auc(real["T"])[positions]
    reverse_s = auc(reverse["S"])[positions]
    reverse_t = auc(reverse["T"])[positions]
    fresh_s = [
        float(np.mean(auc(curve["S"])[positions])) for curve in fresh_curves
    ]
    fresh_t = [
        float(np.mean(auc(curve["T"])[positions])) for curve in fresh_curves
    ]
    s = empirical(float(np.mean(real_s)), fresh_s)
    t = empirical(float(np.mean(real_t)), fresh_t)
    s["sample_bootstrap_ci_95"] = percentile_ci(real_s, bootstrap)
    t["sample_bootstrap_ci_95"] = percentile_ci(real_t, bootstrap)
    return {
        "n": len(positions),
        "S": s,
        "T": t,
        "real_minus_reverse": {
            "S": float(np.mean(real_s - reverse_s)),
            "S_bootstrap_ci_95": percentile_ci(real_s - reverse_s, bootstrap),
            "T": float(np.mean(real_t - reverse_t)),
            "T_bootstrap_ci_95": percentile_ci(real_t - reverse_t, bootstrap),
        },
    }


def mean_trajectory(curve, positions):
    return {
        "S": [float(value) for value in np.mean(curve["S"][positions], axis=0)],
        "T": [float(value) for value in np.mean(curve["T"][positions], axis=0)],
    }


def compute_within_run_endpoints(state, context):
    routed_ids = context["routed_ids"]
    primary_set = set(context["primary_ids"])
    primary_positions = np.asarray(
        [
            position
            for position, idx in enumerate(routed_ids)
            if idx in primary_set
        ],
        dtype=int,
    )
    routed_positions = np.arange(len(routed_ids), dtype=int)
    real = curve_from_same_run(state, "deployed_real", routed_ids)
    reverse = curve_from_same_run(state, "reverse", routed_ids)
    archived = [
        curve_from_same_run(state, f"archived_random{k}", routed_ids)
        for k in range(OLD_K)
    ]
    fresh = [
        curve_from_same_run(state, f"fresh_random{k}", routed_ids)
        for k in range(FRESH_K)
    ]

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    primary_bootstrap = rng.integers(
        0, len(primary_positions), size=(BOOTSTRAPS, len(primary_positions))
    )
    routed_bootstrap = rng.integers(
        0, len(routed_positions), size=(BOOTSTRAPS, len(routed_positions))
    )
    primary = summarize_population(
        primary_positions, real, reverse, fresh, primary_bootstrap
    )
    sensitivity = summarize_population(
        routed_positions, real, reverse, fresh, routed_bootstrap
    )
    t_primary = primary["T"]
    conditions = {
        "recomputed_real_T_AUC_gt_zero": t_primary["real"] > 0,
        "recomputed_real_T_bootstrap_lower_gt_zero": (
            t_primary["sample_bootstrap_ci_95"][0] > 0
        ),
        "fresh_exceedance_le_3_of_79": (
            t_primary["fresh_exceedance_count"] <= 3
        ),
    }
    success = all(conditions.values())

    real_s_auc = auc(real["S"])[primary_positions]
    real_t_auc = auc(real["T"])[primary_positions]
    quadrants = []
    for s_value, t_value in zip(real_s_auc, real_t_auc):
        if abs(s_value) <= NEAR_ZERO or abs(t_value) <= NEAR_ZERO:
            quadrants.append("ambiguous")
        elif s_value > 0 and t_value > 0:
            quadrants.append("S+_T+")
        elif s_value > 0:
            quadrants.append("S+_T-")
        elif t_value > 0:
            quadrants.append("S-_T+")
        else:
            quadrants.append("S-_T-")

    archived_control = {
        f"archived_random{k}": {
            "seed": context["old_seeds"][k],
            "primary68_mean_S_AUC": float(
                np.mean(auc(curve["S"])[primary_positions])
            ),
            "primary68_mean_T_AUC": float(
                np.mean(auc(curve["T"])[primary_positions])
            ),
        }
        for k, curve in enumerate(archived)
    }
    return {
        "status": "SUCCESS" if success else "NON_SUCCESS",
        "primary_success_conditions": conditions,
        "primary_population": primary,
        "sensitivity_population": sensitivity,
        "secondary": {
            "primary_S": primary["S"],
            "T_effect_size_gap": primary["T"]["real_minus_fresh_mean"],
            "real_minus_reverse": primary["real_minus_reverse"],
            "population_sensitivity_68_vs_74": sensitivity,
            "per_dose_trajectories": {
                "primary68": {
                    "real": mean_trajectory(real, primary_positions),
                    "reverse": mean_trajectory(reverse, primary_positions),
                    "fresh79": {
                        f"fresh_random{k}": mean_trajectory(
                            curve, primary_positions
                        )
                        for k, curve in enumerate(fresh)
                    },
                },
                "routed74": {
                    "real": mean_trajectory(real, routed_positions),
                    "reverse": mean_trajectory(reverse, routed_positions),
                    "fresh79": {
                        f"fresh_random{k}": mean_trajectory(
                            curve, routed_positions
                        )
                        for k, curve in enumerate(fresh)
                    },
                },
            },
            "rho4_S_PHASE10_1_MOTIVATED_NEW_DATA_SECONDARY": {
                "label": "PHASE10_1_MOTIVATED_NEW_DATA_SECONDARY",
                "primary68_real": float(
                    np.mean(real["S"][primary_positions, 4])
                ),
                "routed74_real": float(
                    np.mean(real["S"][routed_positions, 4])
                ),
            },
        },
        "exploratory": {
            "sample_quadrants_primary68": dict(Counter(quadrants)),
            "S_T_pearson_primary68": float(
                np.corrcoef(real_s_auc, real_t_auc)[0, 1]
            ),
            "S_sign_by_dose": [
                int(np.sign(value))
                for value in np.mean(real["S"][primary_positions], axis=0)
            ],
        },
        "archived20_environment_drift_controls_only": archived_control,
        "archived20_in_primary_p_value": False,
        "stored_phase10_1_scientific_values_in_primary": False,
    }


def write_structural_void(drift, state):
    V3_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": "C_STRUCTURAL_GATE_FAILURE_VOID",
        "drift": drift,
        "scientific_endpoints_computed": False,
        "withdrawal_language": (
            "Phase 10.2 is VOID. No Phase 10.2 scientific result or claim is "
            "permitted. Any provisional Phase 10.2 output must be withdrawn."
        ),
    }
    (V3_ANALYSIS_DIR / "phase10_2_v3_STRUCTURAL_VOID.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    state["structural_verdict"] = drift["branch"]
    state["scientific_endpoints_computed"] = False
    V3_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False))


def write_results(summary, drift, state, elapsed):
    V3_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (V3_ANALYSIS_DIR / "phase10_2_v3_drift_report.json").write_text(
        json.dumps(drift, indent=2, ensure_ascii=False) + "\n"
    )
    (V3_ANALYSIS_DIR / "phase10_2_v3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    null_rows = []
    for k, seed in enumerate(range(FRESH_SEED_START, FRESH_SEED_START + FRESH_K)):
        null_rows.append(
            {
                "identity": f"fresh_random{k}",
                "seed": seed,
                "primary68_mean_T_AUC": summary["primary_population"]["T"][
                    "fresh_values"
                ][k],
                "T_ge_recomputed_real": (
                    summary["primary_population"]["T"]["fresh_values"][k]
                    >= summary["primary_population"]["T"]["real"]
                ),
            }
        )
    with (
        V3_ANALYSIS_DIR / "phase10_2_v3_fresh_direction_null.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(null_rows[0]))
        writer.writeheader()
        writer.writerows(null_rows)

    if drift["branch"] == "A_ENVIRONMENT_CONTINUITY_CONFIRMED":
        drift_wording = (
            "Environment continuity with the frozen Phase 10.1 drift controls "
            "was confirmed within the preregistered tolerances."
        )
    else:
        drift_wording = (
            "Numerical drift was detected. The same-run comparison remains "
            "valid, but numerical continuity with Phase 10.1 is not claimed."
        )
    result_wording = (
        "On the frozen primary population, the same-run recomputed deployed "
        "direction's positive target selectivity exceeded the independent "
        "79-direction matched-norm null at the preregistered threshold."
        if summary["status"] == "SUCCESS"
        else "Phase 10.2 did not independently confirm positive target "
        "selectivity under all preregistered criteria."
    )
    primary = summary["primary_population"]["T"]
    report = f"""# Phase 10.2 V3 — Results

**Primary status:** **{summary['status']}**
**Drift branch:** **{drift['branch']}**

{drift_wording}

{result_wording}

## Sole primary

- Same-run recomputed real T-AUC: `{primary['real']}`
- Bootstrap 95% CI: `{primary['sample_bootstrap_ci_95']}`
- Fresh-random exceedance: `{primary['fresh_exceedance_count']}/79`
- Add-one p: `{primary['p_add_one']}`

The archived 20 directions were recomputed as environment-drift controls only.
They were excluded from the primary p-value. No stored Phase 10.1 scientific
value entered the primary comparison.
"""
    (V3_ANALYSIS_DIR / "PHASE10_2_V3_RESULTS.md").write_text(report)
    state["drift_branch"] = drift["branch"]
    state["scientific_endpoints_computed"] = True
    state["elapsed_wall_seconds"] = elapsed
    V3_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False))


def run_analysis():
    start = time.perf_counter()
    raw, p95_map, _checks, failures, context = gate_checks()
    if failures:
        raise StructuralGateFailure("Phase 10.2 V3 gate failed: " + ", ".join(failures))
    if V3_STATE_PATH.exists() or V3_ANALYSIS_DIR.exists():
        raise StructuralGateFailure("same-run outputs already exist; resume/rerun forbidden")

    load_scientific_dependencies()
    set_determinism()
    selected_rows = load_selected_validation_rows(
        context["routed_ids"], raw["frozen_baseline"]
    )

    model, tokenizer, device = load_model_once()
    manifest = environment_manifest(model, tokenizer, context)
    structural_checks = {
        "model_repo_exact": manifest["model_repo"] == MODEL_REPO,
        "model_revision_exact": manifest["model_revision"] == MODEL_REVISION,
        "dtype_exact": (
            manifest["dtype"] == "torch.bfloat16"
            and manifest["attention_implementation"] == ATTENTION_IMPLEMENTATION
        ),
        "routed_ids_order_exact": len(context["routed_ids"]) == ROUTED_N,
        "primary_ids_exact": len(context["primary_ids"]) == PRIMARY_N,
        "batch_partition_exact": (
            manifest["batch_partition_sha256"] == context["batch_sha256"]
        ),
        "rho_grid_exact": manifest["scoring"]["rho_grid"] == list(RHOS),
        "seed_sets_disjoint": not set(context["old_seeds"]).intersection(
            context["fresh_seeds"]
        ),
        "single_scoring_process": True,
        "stored_scientific_values_used_for_scoring": False,
        "test_manifest_accessed": False,
        "prohibited_cache_accessed": False,
    }
    if not all(structural_checks.values()):
        raise StructuralGateFailure(
            f"pre-zero structural failure: {structural_checks}"
        )

    V3_STATE_DIR.mkdir(parents=True, exist_ok=False)
    state = {
        "protocol": "Phase 10.2 V3 zero-first same-run raw state",
        "environment_manifest": manifest,
        "structural_checks": structural_checks,
        "cells": {},
        "cell_order": [],
        "next_execution_order": 0,
        "zero_prediction_gate": {"status": "NOT_RUN"},
        "full_trajectory_storage_validated": {"status": "NOT_RUN"},
        "scientific_endpoints_computed": False,
        "formal_run_resumable": False,
    }
    V3_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False))

    # Blocking first scored arm. No real/random vector exists yet, and no
    # scientific endpoint helper has been called.
    run_zero_arm(model, tokenizer, device, selected_rows, context, state)
    zero_mismatches = zero_prediction_gate(state, context)
    if zero_mismatches:
        drift = {
            "branch": "C_STRUCTURAL_GATE_FAILURE_VOID",
            "structural_failures": ["zero_baseline_prediction_mismatch"],
            "zero_prediction_mismatches": zero_mismatches,
            "scientific_endpoints_permitted": False,
        }
        write_structural_void(drift, state)
        print("C_STRUCTURAL_GATE_FAILURE_VOID")
        return 2

    # Only a passing exact 74-prediction zero gate permits creation or scoring
    # of deployed, reverse, archived-random, or fresh-random directions.
    deployed, direction_info = build_deployed_direction(
        p95_map, context["routed_ids"]
    )
    vectors, vector_manifest = build_remaining_vectors(
        deployed, context["old_seeds"], context["fresh_seeds"]
    )
    state["environment_manifest"]["vectors"].update(vector_manifest)
    state["environment_manifest"]["deployed_direction"] = direction_info
    state["structural_checks"]["direction_fingerprint_exact"] = (
        direction_info["projection_fingerprint_rows"] == ROUTED_N
    )
    if not state["structural_checks"]["direction_fingerprint_exact"]:
        raise StructuralGateFailure("deployed direction fingerprint mismatch")
    persist_state(state)

    run_remaining_arms_same_process(
        model,
        tokenizer,
        device,
        selected_rows,
        vectors,
        vector_manifest,
        context,
        state,
    )
    validate_full_trajectory_storage(state, context)

    # Sole primary and all secondary/exploratory endpoints are computed only
    # from the completed same-run state, after zero PASS.
    summary = compute_within_run_endpoints(state, context)

    # Historical Phase 10.1 values are first used here, after the within-run
    # endpoint has been computed, solely for archived-20 numerical continuity.
    drift = audit_archived_continuity(state, raw, context)
    summary["drift_branch"] = drift["branch"]
    write_results(summary, drift, state, time.perf_counter() - start)
    print(summary["status"])
    return 0

def write_exception_structural_void(error):
    V3_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": "C_STRUCTURAL_GATE_FAILURE_VOID",
        "reason": str(error),
        "scientific_endpoints_computed": False,
        "withdrawal_language": (
            "Phase 10.2 is VOID. No Phase 10.2 scientific result or claim is "
            "permitted. Any provisional Phase 10.2 output must be withdrawn."
        ),
    }
    (V3_ANALYSIS_DIR / "phase10_2_v3_STRUCTURAL_VOID.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate-only", action="store_true")
    mode.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args()
    if args.gate_only:
        start = time.perf_counter()
        _raw, _p95, checks, failures, context = gate_checks()
        write_gate_only(checks, failures, context, time.perf_counter() - start)
        return 0 if not failures else 2
    try:
        return run_analysis()
    except StructuralGateFailure as error:
        write_exception_structural_void(error)
        print("C_STRUCTURAL_GATE_FAILURE_VOID")
        return 2


if __name__ == "__main__":
    sys.exit(main())
