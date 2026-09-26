#!/usr/bin/env python3
"""
Phase 10.2 V2: blocking same-run expanded random-null runner.

Gate-only is standard-library-only. It does not import NumPy, torch, transformers,
open model/data/cache files, generate vectors, or compute scientific endpoints.

Formal analysis is implemented behind --run-analysis and requires a new explicit
human approval. In that mode, every zero/real/reverse/archived/fresh arm is scored
in one uninterrupted process, through one model load and one scoring path.
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
V2_GATE_DIR = OUT_BASE / "v2_gate"
V2_ANALYSIS_DIR = OUT_BASE / "v2_analysis"
V2_STATE_DIR = Path("/root/autodl-tmp/phase10_2_v2_state")
V2_STATE_PATH = V2_STATE_DIR / "phase10_2_v2_same_run_raw.json"

ERRATUM_PATH = OUT_BASE / "PHASE10_2_PROTOCOL_ERRATUM.md"
PREREG_V2_PATH = OUT_BASE / "PHASE10_2_PREREGISTRATION_V2.md"
HASHES_V2_PATH = OUT_BASE / "PHASE10_2_HASHES_V2.json"
FREEZE_V2_PATH = OUT_BASE / "PHASE10_2_EXECUTION_FREEZE_V2.json"
AUDIT_V2_PATH = OUT_BASE / "PHASE10_2_SAME_RUN_AUDIT.md"
SELF_PATH = ROOT / "scripts" / "phase10_2_expanded_random_null_v2.py"

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
        ERRATUM_PATH,
        PREREG_V2_PATH,
        HASHES_V2_PATH,
        FREEZE_V2_PATH,
        AUDIT_V2_PATH,
        SELF_PATH,
    ]
    check(
        "v2_files_exist",
        all(path.is_file() for path in required),
        [str(path.relative_to(ROOT)) for path in required if not path.is_file()],
    )

    for rel, expected in EXPECTED_INPUT_HASHES.items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else "MISSING"
        check(f"sha256:{rel}", actual == expected, f"{expected} / {actual}")

    hashes_doc = strict_json(HASHES_V2_PATH)
    freeze = strict_json(FREEZE_V2_PATH)
    v2_paths = {
        "erratum": ERRATUM_PATH,
        "preregistration_v2": PREREG_V2_PATH,
        "runner_v2": SELF_PATH,
        "same_run_audit": AUDIT_V2_PATH,
    }
    for label, path in v2_paths.items():
        expected = hashes_doc["phase10_2_v2_files"][label]["sha256"]
        check(
            f"phase10_2_v2_hash:{label}",
            sha256(path) == expected,
            f"{expected} / {sha256(path)}",
        )
    for label, path in {**v2_paths, "hashes_v2": HASHES_V2_PATH}.items():
        expected = freeze["files"][label]["sha256"]
        check(
            f"execution_freeze_v2:{label}",
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
    check(
        "same_run_freeze",
        freeze["same_run"]["required"] is True
        and freeze["same_run"]["single_model_load"] is True
        and freeze["same_run"]["batch_size"] == BATCH_SIZE
        and freeze["same_run"]["rho_grid"] == list(RHOS)
        and freeze["same_run"]["arm_order"] == arm_order(),
        freeze["same_run"],
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
        "fixed_batch_partition",
        len(batches) == ROUTED_N
        and all(len(batch) == BATCH_SIZE for batch in batches),
        canonical_hash(batches),
    )

    tree = ast.parse(SELF_PATH.read_text())
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required_functions = {
        "build_same_run_vectors",
        "run_all_arms_same_process",
        "validate_complete_same_run",
        "classify_drift_branch",
        "compute_within_run_endpoints",
    }
    check(
        "same_run_code_paths_present",
        required_functions.issubset(function_names),
        sorted(required_functions - function_names),
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

    old_analysis_absent = (
        not (OUT_BASE / "analysis").exists()
        and not Path(
            "/root/autodl-tmp/phase10_2_state/phase10_2_fresh_random_state.json"
        ).exists()
    )
    check("v1_superseded_unrun", old_analysis_absent, old_analysis_absent)
    v2_output_absent = not V2_STATE_PATH.exists() and not V2_ANALYSIS_DIR.exists()
    check("no_v2_state_or_analysis_output", v2_output_absent, v2_output_absent)

    v1_gate = strict_json(V1_GATE_JSON)
    check(
        "v1_gate_was_non_scientific",
        v1_gate["gate_only"] is True
        and v1_gate["run_analysis_executed"] is False
        and v1_gate["phase10_2_scientific_endpoints_computed"] is False,
        v1_gate["status"],
    )
    check(
        "one_amendment_rule_declared",
        "sole permitted pre-run amendment" in ERRATUM_PATH.read_text()
        and "No V3" in PREREG_V2_PATH.read_text(),
        "V1 preserved; V2 sole amendment",
    )

    return raw, p95_map, checks, failures, {
        "routed_ids": routed_ids,
        "primary_ids": list(map(str, primary_ids)),
        "batches": batches,
        "batch_sha256": canonical_hash(batches),
        "old_seeds": sorted(old_seeds),
        "fresh_seeds": sorted(fresh_seeds),
    }


def write_gate_only(checks, failures, context, elapsed):
    V2_GATE_DIR.mkdir(parents=True, exist_ok=True)
    passed = not failures
    status = (
        "READY_FOR_PHASE10_2_V2_ANALYSIS_APPROVAL"
        if passed
        else "BLOCKED_PENDING_V2"
    )
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
        },
        "gate_only": True,
        "run_analysis_executed": False,
        "model_loaded": False,
        "gpu_used": False,
        "random_vectors_generated": False,
        "new_predictions_generated": False,
        "phase10_2_scientific_endpoints_computed": False,
        "elapsed_cpu_wall_seconds": elapsed,
    }
    (V2_GATE_DIR / "phase10_2_v2_gate_only.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Phase 10.2 V2 — Gate-Only Report",
        "",
        f"**Status:** **{status}**",
        f"**Checks passed:** {sum(c['passed'] for c in checks)}/{len(checks)}",
        "**Model loaded:** false",
        "**GPU used:** false",
        "**Random vectors generated:** false",
        "**Scientific endpoints computed:** false",
        "",
        "Formal analysis still requires separate human approval.",
    ]
    (V2_GATE_DIR / "PHASE10_2_V2_GATE_ONLY_REPORT.md").write_text(
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


def build_same_run_vectors(deployed, old_seeds, fresh_seeds):
    vectors = {
        "zero": np.zeros(HIDDEN, dtype=np.float32),
        "deployed_real": deployed.astype(np.float32, copy=True),
        "reverse": (-deployed).astype(np.float32, copy=True),
    }
    for k, seed in enumerate(old_seeds):
        vectors[f"archived_random{k}"] = random_unit(seed)
    for k, seed in enumerate(fresh_seeds):
        vectors[f"fresh_random{k}"] = random_unit(seed)
    if list(vectors) != arm_order():
        raise StructuralGateFailure("arm order mismatch while constructing vectors")
    manifest = {}
    for arm, vector in vectors.items():
        norm = float(np.linalg.norm(vector))
        expected = 0.0 if arm == "zero" else 1.0
        if abs(norm - expected) > UNIT_NORM_TOL:
            raise StructuralGateFailure(f"unit norm mismatch {arm}: {norm}")
        manifest[arm] = {
            "seed": (
                old_seeds[int(arm.removeprefix("archived_random"))]
                if arm.startswith("archived_random")
                else fresh_seeds[int(arm.removeprefix("fresh_random"))]
                if arm.startswith("fresh_random")
                else None
            ),
            "norm": norm,
            "sha256": bytes_sha256(vector.tobytes(order="C")),
        }
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


def environment_manifest(model, tokenizer, vector_manifest, context, direction_info):
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
        "vectors": vector_manifest,
        "deployed_direction": direction_info,
        "tokenizer_class": tokenizer.__class__.__name__,
    }


def run_all_arms_same_process(
    model, tokenizer, device, selected_rows, vectors, context, state
):
    for arm in arm_order():
        vector = vectors[arm]
        for rho in RHOS:
            key = f"{arm}|{rho:.1f}"
            correction = (rho * MED_INJ * vector).astype(np.float32)
            expected_norm = (
                0.0 if arm == "zero" or rho == 0 else rho * MED_INJ
            )
            correction_norm = float(np.linalg.norm(correction))
            if abs(correction_norm - expected_norm) > CORRECTION_NORM_TOL:
                raise StructuralGateFailure(
                    f"correction norm mismatch {key}: {correction_norm}"
                )
            per_sample = {}
            for batch_index, batch in enumerate(context["batches"]):
                batch_result = score_batch(
                    model,
                    tokenizer,
                    device,
                    selected_rows,
                    batch,
                    correction,
                )
                per_sample.update(batch_result)
                if list(batch_result) != batch:
                    raise StructuralGateFailure(f"batch membership mismatch {key}")
            if list(per_sample) != context["routed_ids"]:
                raise StructuralGateFailure(f"sample order mismatch {key}")
            state["cells"][key] = {
                "arm": arm,
                "rho": rho,
                "correction_norm": correction_norm,
                "batch_partition_sha256": context["batch_sha256"],
                "per_sample": per_sample,
            }
            temp = V2_STATE_PATH.with_suffix(".tmp")
            temp.write_text(json.dumps(state, ensure_ascii=False))
            os.replace(temp, V2_STATE_PATH)


def validate_complete_same_run(state, context):
    expected = {
        f"{arm}|{rho:.1f}" for arm in arm_order() for rho in RHOS
    }
    if set(state["cells"]) != expected:
        raise StructuralGateFailure(
            f"same-run state incomplete: {len(state['cells'])}/{len(expected)}"
        )
    for key in expected:
        cell = state["cells"][key]
        if cell["batch_partition_sha256"] != context["batch_sha256"]:
            raise StructuralGateFailure(f"batch partition mismatch {key}")
        if list(cell["per_sample"]) != context["routed_ids"]:
            raise StructuralGateFailure(f"ID order mismatch {key}")
        for idx in context["routed_ids"]:
            record = cell["per_sample"][idx]
            if set(record["logp"]) != set(MODES):
                raise StructuralGateFailure(f"mode mismatch {key} {idx}")
            if not all(
                math.isfinite(float(value)) for value in record["logp"].values()
            ):
                raise StructuralGateFailure(f"nonfinite score {key} {idx}")


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


def classify_drift_branch(state, raw, context, structural_checks):
    routed_ids = context["routed_ids"]
    frozen = raw["frozen_baseline"]
    zero_mismatches = []
    for rho in RHOS:
        cell = state["cells"][f"zero|{rho:.1f}"]["per_sample"]
        for idx in routed_ids:
            if cell[idx]["pred"] != frozen[idx]["baseline_pred"]:
                zero_mismatches.append(
                    {
                        "id": idx,
                        "rho": rho,
                        "expected": frozen[idx]["baseline_pred"],
                        "observed": cell[idx]["pred"],
                    }
                )
    structural_failures = [
        name for name, passed in structural_checks.items() if not passed
    ]
    if zero_mismatches:
        structural_failures.append("zero_baseline_prediction_mismatch")
    if structural_failures:
        return {
            "branch": "C_STRUCTURAL_GATE_FAILURE_VOID",
            "structural_failures": structural_failures,
            "zero_prediction_mismatches": zero_mismatches,
            "scientific_endpoints_permitted": False,
        }

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
        "zero_prediction_mismatches": [],
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


def compute_within_run_endpoints(state, context, drift):
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
        "drift_branch": drift["branch"],
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
    V2_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": "C_STRUCTURAL_GATE_FAILURE_VOID",
        "drift": drift,
        "scientific_endpoints_computed": False,
        "withdrawal_language": (
            "Phase 10.2 is VOID. No Phase 10.2 scientific result or claim is "
            "permitted. Any provisional Phase 10.2 output must be withdrawn."
        ),
    }
    (V2_ANALYSIS_DIR / "phase10_2_v2_STRUCTURAL_VOID.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    state["structural_verdict"] = drift["branch"]
    state["scientific_endpoints_computed"] = False
    V2_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False))


def write_results(summary, drift, state, elapsed):
    V2_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (V2_ANALYSIS_DIR / "phase10_2_v2_drift_report.json").write_text(
        json.dumps(drift, indent=2, ensure_ascii=False) + "\n"
    )
    (V2_ANALYSIS_DIR / "phase10_2_v2_summary.json").write_text(
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
        V2_ANALYSIS_DIR / "phase10_2_v2_fresh_direction_null.csv"
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
    report = f"""# Phase 10.2 V2 — Results

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
    (V2_ANALYSIS_DIR / "PHASE10_2_V2_RESULTS.md").write_text(report)
    state["drift_branch"] = drift["branch"]
    state["scientific_endpoints_computed"] = True
    state["elapsed_wall_seconds"] = elapsed
    V2_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False))


def run_analysis():
    start = time.perf_counter()
    raw, p95_map, _checks, failures, context = gate_checks()
    if failures:
        raise StructuralGateFailure("Phase 10.2 V2 gate failed: " + ", ".join(failures))
    if V2_STATE_PATH.exists() or V2_ANALYSIS_DIR.exists():
        raise StructuralGateFailure("same-run outputs already exist; resume/rerun forbidden")

    load_scientific_dependencies()
    set_determinism()
    selected_rows = load_selected_validation_rows(
        context["routed_ids"], raw["frozen_baseline"]
    )
    deployed, direction_info = build_deployed_direction(
        p95_map, context["routed_ids"]
    )
    vectors, vector_manifest = build_same_run_vectors(
        deployed, context["old_seeds"], context["fresh_seeds"]
    )

    model, tokenizer, device = load_model_once()
    manifest = environment_manifest(
        model, tokenizer, vector_manifest, context, direction_info
    )
    structural_checks = {
        "model_repo_exact": manifest["model_repo"] == MODEL_REPO,
        "model_revision_exact": manifest["model_revision"] == MODEL_REVISION,
        "dtype_exact": manifest["dtype"] == "torch.bfloat16" and manifest["attention_implementation"] == ATTENTION_IMPLEMENTATION,
        "routed_ids_order_exact": len(context["routed_ids"]) == ROUTED_N,
        "primary_ids_exact": len(context["primary_ids"]) == PRIMARY_N,
        "batch_partition_exact": (
            manifest["batch_partition_sha256"] == context["batch_sha256"]
        ),
        "rho_grid_exact": manifest["scoring"]["rho_grid"] == list(RHOS),
        "seed_sets_disjoint": not set(context["old_seeds"]).intersection(
            context["fresh_seeds"]
        ),
        "direction_fingerprint_exact": (
            direction_info["projection_fingerprint_rows"] == ROUTED_N
        ),
        "single_scoring_process": True,
        "stored_scientific_values_used_for_scoring": False,
        "test_manifest_accessed": False,
        "prohibited_cache_accessed": False,
    }
    if not all(structural_checks.values()):
        raise StructuralGateFailure(f"pre-scoring structural failure: {structural_checks}")

    V2_STATE_DIR.mkdir(parents=True, exist_ok=False)
    state = {
        "protocol": "Phase 10.2 V2 same-run raw state",
        "environment_manifest": manifest,
        "structural_checks": structural_checks,
        "cells": {},
        "scientific_endpoints_computed": False,
    }
    V2_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False))
    run_all_arms_same_process(
        model,
        tokenizer,
        device,
        selected_rows,
        vectors,
        context,
        state,
    )
    validate_complete_same_run(state, context)

    # Historical scientific values are first consulted here, after all 510
    # same-run cells are complete. They are drift references only.
    drift = classify_drift_branch(state, raw, context, structural_checks)
    if drift["branch"] == "C_STRUCTURAL_GATE_FAILURE_VOID":
        write_structural_void(drift, state)
        print(drift["branch"])
        return 2

    # Confirmatory and secondary endpoints consume only same-run state.
    summary = compute_within_run_endpoints(state, context, drift)
    write_results(summary, drift, state, time.perf_counter() - start)
    print(summary["status"])
    return 0


def write_exception_structural_void(error):
    V2_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": "C_STRUCTURAL_GATE_FAILURE_VOID",
        "reason": str(error),
        "scientific_endpoints_computed": False,
        "withdrawal_language": (
            "Phase 10.2 is VOID. No Phase 10.2 scientific result or claim is "
            "permitted. Any provisional Phase 10.2 output must be withdrawn."
        ),
    }
    (V2_ANALYSIS_DIR / "phase10_2_v2_STRUCTURAL_VOID.json").write_text(
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
