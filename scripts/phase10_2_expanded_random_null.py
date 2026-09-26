#!/usr/bin/env python3
"""
Phase 10.2 expanded fresh-random confirmation runner.

Gate-only is intentionally standard-library-only. It does not import scientific
dependencies, open model/data/cache/test files, generate random vectors, or compute
scientific endpoints.

Formal analysis is implemented but requires an explicit --run-analysis invocation
after a separate human approval.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "final" / "results" / "phase10_2_expanded_random_null"
GATE_DIR = OUT_BASE / "gate"
ANALYSIS_DIR = OUT_BASE / "analysis"
STATE_DIR = Path("/root/autodl-tmp/phase10_2_state")
STATE_PATH = STATE_DIR / "phase10_2_fresh_random_state.json"

AUDIT_PATH = OUT_BASE / "PHASE10_2_INPUT_AUDIT.md"
PREREG_PATH = OUT_BASE / "PHASE10_2_PREREGISTRATION.md"
HASHES_PATH = OUT_BASE / "PHASE10_2_HASHES.json"
FREEZE_PATH = OUT_BASE / "PHASE10_2_EXECUTION_FREEZE.json"
SELF_PATH = ROOT / "scripts" / "phase10_2_expanded_random_null.py"

P8_LIB_PATH = ROOT / "scripts" / "phase8_lib.py"
P10_PROBE_PATH = ROOT / "scripts" / "phase10_mctl_probe.py"
RAW_PATH = ROOT / "final" / "results" / "phase10_mctl" / "phase10_mctl_raw.json"
P95_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase9_mechanism_and_ood"
    / "phase9_5_per_sample_routed.json"
)
VAL_PATH = ROOT / "final" / "results" / "splits" / "val_idx.json"
P101_SUMMARY_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase10_1_directionality"
    / "v2"
    / "phase10_1_summary.json"
)
P101_RESULT_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase10_1_directionality"
    / "v2"
    / "PHASE10_1_RESULTS.md"
)
P101_INTERP_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase10_1_directionality"
    / "v2"
    / "PHASE10_1_POSTRESULT_INTERPRETATION.md"
)
P101_PREREG_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase10_1_directionality"
    / "PHASE10_1_PREREGISTRATION_V2.md"
)
P101_FREEZE_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase10_1_directionality"
    / "PHASE10_1_EXECUTION_FREEZE_V2.json"
)
P101_GATE_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase10_1_directionality"
    / "v2"
    / "phase10_1_validity_gate.json"
)

PHASE8_R0_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase8_prospective_llama"
    / "llama_r0_summary.json"
)
PHASE8_ACT_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase8_prospective_llama"
    / "llama_activation_extraction.json"
)
PHASE8_DECISIONS_PATH = (
    ROOT
    / "final"
    / "results"
    / "phase8_prospective_llama"
    / "LLAMA_FROZEN_GATE_DECISIONS.json"
)

SOURCE_JSONL = (
    ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"
)
MODEL_PATH = Path("/root/autodl-tmp/models/Llama-3.1-8B-Instruct")

EXPECTED_ROOT = "/root/autodl-tmp/sakiko-followup"
EXPECTED_BRANCH = "exp/sakiko-followup-archive"
MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_DTYPE = "bfloat16"
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

EXPECTED_ROUTED_ID_SHA = "a6aa9f7803673f4b4e0b016df4fb4f1ed6a657e69ff61cb6543108d2e2cf2bb8"
EXPECTED_PRIMARY_ID_SHA = "e358d242f2c1f251e64c157896074105270df2da7b566752f88eb2413e8b8ba1"
EXPECTED_ROUTED_UUID_SHA = "82631f98f61f52a29ee0035765258bec2630fd7948f7d8c8175d889b617a4a13"

EXPECTED_INPUT_HASHES = {
    "scripts/phase8_lib.py": "e8f314bea324a04c148a9dc20605fb690f0766f1e05eb9d569bffdbe11336d4c",
    "scripts/phase10_mctl_probe.py": "8195eeede6f2740cdf3808e14aa2e67deabd2198f3f944e6a5317bee3cb870e0",
    "final/results/phase10_mctl/phase10_mctl_raw.json": "c9324d46a417e70e2244ec154e9aff507ae9bcc126212b3e188f8a968a7d5f5d",
    "final/results/phase9_mechanism_and_ood/phase9_5_per_sample_routed.json": "525a6832fd24ac9aa2e256ac32666b0af98e2ffdd0aea8501c8bdc8ef92012fd",
    "final/results/splits/val_idx.json": "2d2f078a47ae7702b5ccceb23a86d9d4cf223d7b7cc0de4eb8090e809afa3610",
    "final/results/phase10_1_directionality/PHASE10_1_PREREGISTRATION_V2.md": "895e24853fd57ce517381aa233e40cb643eb8610c201e53a6f993e3a504e63f8",
    "final/results/phase10_1_directionality/PHASE10_1_EXECUTION_FREEZE_V2.json": "c0ed97548718f120b609a93e5562bc074bfda4179378cbcad7d58abe0c3600b9",
    "final/results/phase10_1_directionality/v2/phase10_1_validity_gate.json": "e6aadbddebc8089a745d441dc3a1f0be8fe209fb8830021ef11e35f572590fb6",
    "final/results/phase10_1_directionality/v2/phase10_1_summary.json": "fbebbc7fc250974b67462822af4a581dc5624553b58968c3d0f66270c4f96d3c",
    "final/results/phase10_1_directionality/v2/PHASE10_1_RESULTS.md": "6e1b20eca92811cd69818f371bfca76bee179c169ba817eef270b06bf4baf6ca",
    "final/results/phase10_1_directionality/v2/PHASE10_1_POSTRESULT_INTERPRETATION.md": "82561f13977ee3ce2e23d30b1379fa418cc5e0afd2df8c472be77f547c4c4216",
    "final/results/phase8_prospective_llama/llama_r0_summary.json": "27f70675c8c30f5ca7d250b3df66cfdb5f270ed7a1271ab216d60c99bb193d20",
    "final/results/phase8_prospective_llama/llama_activation_extraction.json": "99129953e3caf6fb94e203c14fa7f787c91700a9c3d7c74badaf519a0a6f6769",
    "final/results/phase8_prospective_llama/LLAMA_FROZEN_GATE_DECISIONS.json": "45a3f8a78139ab1cb9a53f82e0fd5bdeaac933c9ff5e5cf7f75ed04af42a5119",
}


LOCAL_MODEL_METADATA_HASHES = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "generation_config.json": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6d60e9ac5ababfb940ff23bbcd424",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
}


class DuplicateKeyError(ValueError):
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
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def add_check(checks, name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def ast_assignments(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
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


def gate_checks(allow_partial_state=False):
    checks = []
    failures = []

    def check(name, passed, detail):
        add_check(checks, name, passed, detail)
        if not passed:
            failures.append(name)

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
    remote_head = git("rev-parse", "origin/exp/sakiko-followup-archive")
    check("local_remote_head_match", head == remote_head, f"{head} / {remote_head}")

    expected_paths = [AUDIT_PATH, PREREG_PATH, HASHES_PATH, SELF_PATH, FREEZE_PATH]
    check(
        "freeze_files_exist",
        all(p.is_file() for p in expected_paths),
        [str(p.relative_to(ROOT)) for p in expected_paths if not p.is_file()],
    )

    for rel, expected in EXPECTED_INPUT_HASHES.items():
        path = ROOT / rel
        actual = sha256(path) if path.is_file() else "MISSING"
        check(f"sha256:{rel}", actual == expected, f"expected={expected}; actual={actual}")

    hashes_doc = strict_json(HASHES_PATH)
    freeze = strict_json(FREEZE_PATH)
    for label, path in {
        "audit": AUDIT_PATH,
        "preregistration": PREREG_PATH,
        "runner": SELF_PATH,
    }.items():
        expected = hashes_doc["phase10_2_files"][label]["sha256"]
        actual = sha256(path)
        check(f"phase10_2_hash:{label}", actual == expected, f"{expected} / {actual}")

    freeze_files = freeze["files"]
    for label, path in {
        "audit": AUDIT_PATH,
        "preregistration": PREREG_PATH,
        "hashes": HASHES_PATH,
        "runner": SELF_PATH,
    }.items():
        expected = freeze_files[label]["sha256"]
        actual = sha256(path)
        check(f"execution_freeze:{label}", actual == expected, f"{expected} / {actual}")

    check("freeze_root", freeze["repository"]["root"] == EXPECTED_ROOT, freeze["repository"])
    check(
        "freeze_branch",
        freeze["repository"]["branch"] == EXPECTED_BRANCH,
        freeze["repository"],
    )
    check(
        "freeze_mode",
        freeze["authorization"]["gate_only"] is True
        and freeze["authorization"]["run_analysis"] is False,
        freeze["authorization"],
    )

    p8_source = P8_LIB_PATH.read_text()
    p10 = ast_assignments(P10_PROBE_PATH)
    check(
        "historical_model_declaration",
        ("\"repo\": \"" + MODEL_REPO + "\"") in p8_source
        and ("\"revision\": \"" + MODEL_REVISION + "\"") in p8_source,
        {"repo": MODEL_REPO, "revision": MODEL_REVISION},
    )
    check(
        "historical_phase10_constants",
        p10.get("CH") == CHANNEL
        and p10.get("GOLD") == GOLD
        and p10.get("FROM_PRED") == SOURCE
        and p10.get("OBS") == OBS_LAYER
        and p10.get("INJ") == INJ_LAYER
        and p10.get("THR") == ROUTER_THRESHOLD
        and p10.get("SEED_BLOCK") == OLD_SEED_START
        and p10.get("N_RANDOM") == OLD_K,
        {
            k: p10.get(k)
            for k in (
                "CH",
                "GOLD",
                "FROM_PRED",
                "OBS",
                "INJ",
                "THR",
                "SEED_BLOCK",
                "N_RANDOM",
            )
        },
    )

    r0 = strict_json(PHASE8_R0_PATH)
    act = strict_json(PHASE8_ACT_PATH)
    decisions = strict_json(PHASE8_DECISIONS_PATH)
    model_ok = (
        r0["model_repo"] == MODEL_REPO
        and r0["model_revision"] == MODEL_REVISION
        and r0["dtype"] == MODEL_DTYPE
        and act["model"] == MODEL_REPO
        and act["revision"] == MODEL_REVISION
        and act["n_layers_total"] == N_LAYERS
        and act["hidden"] == HIDDEN
        and decisions["model"] == MODEL_REPO
        and decisions["revision"] == MODEL_REVISION
    )
    check("independent_model_provenance", model_ok, MODEL_REVISION)

    raw = strict_json(RAW_PATH)
    p95 = strict_json(P95_PATH)
    val_ids = set(map(int, strict_json(VAL_PATH)))
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
    )
    check("phase10_config", config_ok, config)

    frozen = raw["frozen_baseline"]
    routed_ids = list(frozen)
    routed_ints = list(map(int, routed_ids))
    primary_ids = sorted(
        int(i) for i in routed_ids if frozen[i]["is_own_channel_err"] is True
    )
    routed_uuids = [frozen[i]["uuid"] for i in routed_ids]
    pop_ok = (
        len(routed_ids) == ROUTED_N
        and len(set(routed_ints)) == ROUTED_N
        and len(set(routed_uuids)) == ROUTED_N
        and len(primary_ids) == PRIMARY_N
        and set(routed_ints).issubset(val_ids)
        and all(frozen[i]["baseline_pred"] == SOURCE for i in routed_ids)
        and all(float(frozen[i]["router_score"]) >= ROUTER_THRESHOLD for i in routed_ids)
    )
    check("frozen_populations", pop_ok, f"routed={len(routed_ids)} primary={len(primary_ids)}")
    check(
        "ordered_routed_id_signature",
        canonical_hash(routed_ints) == EXPECTED_ROUTED_ID_SHA,
        canonical_hash(routed_ints),
    )
    check(
        "sorted_primary_id_signature",
        canonical_hash(primary_ids) == EXPECTED_PRIMARY_ID_SHA,
        canonical_hash(primary_ids),
    )
    check(
        "ordered_routed_uuid_signature",
        canonical_hash(routed_uuids) == EXPECTED_ROUTED_UUID_SHA,
        canonical_hash(routed_uuids),
    )

    p95_map = {str(row["idx"]): row for row in p95}
    p95_ok = set(p95_map) == set(routed_ids) and all(
        p95_map[i]["uuid"] == frozen[i]["uuid"]
        and p95_map[i]["gold"] == frozen[i]["gold"]
        and p95_map[i]["baseline_pred"] == frozen[i]["baseline_pred"]
        and p95_map[i]["router_score"] == frozen[i]["router_score"]
        and p95_map[i]["is_own_channel_err"] == frozen[i]["is_own_channel_err"]
        for i in routed_ids
    )
    check("independent_routed_provenance", p95_ok, f"rows={len(p95_map)}")

    expected_old = {
        f"random{k}|{rho:.1f}" for k in range(OLD_K) for rho in RHOS[1:]
    }
    actual_old = {key for key in raw["cells"] if key.startswith("random")}
    old_ok = actual_old == expected_old and all(
        len(raw["cells"][key]["per_sample"]) == ROUTED_N for key in expected_old
    )
    check(
        "archived_k20_identity_and_completeness",
        old_ok,
        f"expected={len(expected_old)} actual={len(actual_old)}",
    )

    old_seeds = set(range(OLD_SEED_START, OLD_SEED_START + OLD_K))
    fresh_seeds = list(range(FRESH_SEED_START, FRESH_SEED_START + FRESH_K))
    seed_ok = (
        len(fresh_seeds) == FRESH_K
        and len(set(fresh_seeds)) == FRESH_K
        and not old_seeds.intersection(fresh_seeds)
        and fresh_seeds[0] == 6000
        and fresh_seeds[-1] == 6078
    )
    check(
        "fresh_seed_schedule_without_generation",
        seed_ok,
        {
            "old": [min(old_seeds), max(old_seeds)],
            "fresh": [fresh_seeds[0], fresh_seeds[-1]],
            "intersection": sorted(old_seeds.intersection(fresh_seeds)),
        },
    )

    p101 = strict_json(P101_SUMMARY_PATH)
    discovery_ok = (
        p101["validity_gate"] == "PASS"
        and p101["primary_population"]["n"] == PRIMARY_N
        and p101["sensitivity_population"]["n"] == ROUTED_N
        and p101["primary_population"]["T"]["rank_desc_ties_against_real"] == 1
        and p101["primary_population"]["T"]["n_random_ge_real"] == 0
        and len(p101["primary_population"]["T"]["random_values"]) == OLD_K
    )
    check("phase10_1_discovery_record", discovery_ok, "K20 archival only")

    forbidden_existing = []
    if STATE_PATH.exists() and not allow_partial_state:
        forbidden_existing.append(str(STATE_PATH))
    if ANALYSIS_DIR.exists():
        forbidden_existing.extend(str(p) for p in ANALYSIS_DIR.rglob("*") if p.is_file())
    check(
        "no_phase10_2_state_or_analysis_output",
        not forbidden_existing,
        forbidden_existing,
    )

    self_tree = ast.parse(SELF_PATH.read_text())
    imported_modules = []
    for node in ast.walk(self_tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    check(
        "test_cache_firewall_static",
        "phase8_lib" not in imported_modules,
        "no legacy import; gate paths contain no test/cache input",
    )

    return raw, checks, failures, {
        "routed_ids": routed_ids,
        "primary_ids": list(map(str, primary_ids)),
        "fresh_seeds": fresh_seeds,
    }


def write_gate_only(checks, failures, context, elapsed):
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    passed = not failures
    status = "READY_FOR_PHASE10_2_ANALYSIS_APPROVAL" if passed else "VOID"
    payload = {
        "status": status,
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "context": {
            "routed_n": len(context["routed_ids"]),
            "primary_n": len(context["primary_ids"]),
            "fresh_k": len(context["fresh_seeds"]),
            "fresh_seed_min": min(context["fresh_seeds"]),
            "fresh_seed_max": max(context["fresh_seeds"]),
        },
        "gate_only": True,
        "run_analysis_executed": False,
        "model_loaded": False,
        "gpu_used": False,
        "random_vectors_generated": False,
        "new_predictions_generated": False,
        "test_manifest_opened": False,
        "test_outcomes_accessed": False,
        "cache_opened": False,
        "phase10_2_scientific_endpoints_computed": False,
        "elapsed_cpu_wall_seconds": elapsed,
    }
    gate_json = GATE_DIR / "phase10_2_gate_only.json"
    gate_report = GATE_DIR / "PHASE10_2_GATE_ONLY_REPORT.md"
    gate_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Phase 10.2 — Gate-Only Report",
        "",
        f"**Status:** **{status}**",
        f"**Checks passed:** {sum(c['passed'] for c in checks)}/{len(checks)}",
        "**Model loaded:** false",
        "**GPU used:** false",
        "**Random vectors generated:** false",
        "**New predictions generated:** false",
        "**Test/cache accessed:** false",
        "**Phase 10.2 scientific endpoints computed:** false",
        "**Run-analysis executed:** false",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if c['passed'] else 'FAIL'} — `{c['check']}`: {c['detail']}"
        for c in checks
    )
    if failures:
        lines.extend(["", "## Failures", ""] + [f"- {x}" for x in failures])
    lines.extend(
        [
            "",
            "## Frozen fresh null",
            "",
            "- Fresh directions: exactly 79",
            "- Seeds: 6000 through 6078",
            "- Archived discovery directions: seeds 5000 through 5019",
            "- Seed overlap: none",
            "- Primary p-value denominator: 80",
            "- Old K=20 excluded from primary p-value",
            "",
            "Formal GPU analysis still requires separate human approval.",
        ]
    )
    gate_report.write_text("\n".join(lines) + "\n")
    print(status)


# Scientific dependencies are declared only after the gate boundary.
np = None
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


def load_scientific_dependencies():
    global np, torch, AutoModelForCausalLM, AutoTokenizer
    import numpy as np_module
    import torch as torch_module
    from transformers import AutoModelForCausalLM as ModelClass
    from transformers import AutoTokenizer as TokenizerClass

    np = np_module
    torch = torch_module
    AutoModelForCausalLM = ModelClass
    AutoTokenizer = TokenizerClass


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


def load_selected_validation_rows(routed_ids, frozen):
    # Reproduce datasets.Dataset.shuffle(seed=42) without opening a split manifest.
    permutation = np.random.default_rng(DATASET_SHUFFLE_SEED).permutation(DATASET_N)
    raw_to_routed = {int(permutation[int(i)]): str(i) for i in routed_ids}
    selected = {}
    with SOURCE_JSONL.open() as f:
        for raw_idx, line in enumerate(f):
            routed_idx = raw_to_routed.get(raw_idx)
            if routed_idx is None:
                continue
            row = json.loads(line)
            if row.get("uuid") != frozen[routed_idx]["uuid"]:
                raise RuntimeError(
                    f"validation UUID mismatch routed={routed_idx} raw={raw_idx}"
                )
            selected[routed_idx] = row
    if set(selected) != set(routed_ids):
        raise RuntimeError(
            f"selective validation loader incomplete: {len(selected)}/{len(routed_ids)}"
        )
    return selected


def load_model():
    for name, expected in LOCAL_MODEL_METADATA_HASHES.items():
        path = MODEL_PATH / name
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            raise RuntimeError(
                f"local model metadata mismatch {name}: {actual} != {expected}"
            )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("model dtype mismatch")
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("quantized model is forbidden")
    if model.config.num_hidden_layers != N_LAYERS:
        raise RuntimeError("model layer-count mismatch")
    if model.config.hidden_size != HIDDEN:
        raise RuntimeError("model hidden-size mismatch")
    return model, tok, next(model.parameters()).device


def prompt_text(tok, sample):
    messages = build_prompt_messages(sample["question"], parse_tools(sample["tools"]))
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def score_candidate(model, tok, dev, prompt_ids, candidate, correction):
    cids = tok.encode(candidate, add_special_tokens=False)
    if not cids:
        return -1e9
    inp = torch.cat([prompt_ids, torch.tensor([cids], device=dev)], dim=1)
    plen = prompt_ids.shape[1]
    clen = len(cids)
    correction_tensor = torch.from_numpy(correction).to(torch.bfloat16).to(dev)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output[0][:, :, :] += correction_tensor
            return output
        output[:, :, :] += correction_tensor
        return output

    handle = model.model.layers[INJ_LAYER].mlp.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(inp).logits
    finally:
        handle.remove()
    selected = logits[0, plen - 1 : plen + clen - 1, :]
    log_probs = torch.log_softmax(selected.float(), dim=-1)
    value = sum(log_probs[i, cids[i]].item() for i in range(clen)) / clen
    del inp
    return float(value)


def score_sample(model, tok, dev, sample, correction):
    text = prompt_text(tok, sample)
    prompt_ids = torch.tensor(
        [tok.encode(text, add_special_tokens=False)], device=dev
    )
    scores = {}
    for label in MODES:
        candidate = sample["answers"].get(label, "")
        scores[label] = (
            score_candidate(model, tok, dev, prompt_ids, candidate, correction)
            if candidate
            else -1e9
        )
    del prompt_ids
    return {
        "pred": max(MODES, key=lambda label: scores[label]),
        "logp": {label: round(float(scores[label]), 5) for label in MODES},
    }


def random_unit(seed):
    rng = np.random.RandomState(seed)
    vector = rng.randn(HIDDEN).astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-12)


def st_value(logp, baseline):
    source_terms = [
        (float(logp[c]) - float(logp[SOURCE]))
        - (float(baseline[c]) - float(baseline[SOURCE]))
        for c in MODES
        if c != SOURCE
    ]
    target_terms = [
        (float(logp[GOLD]) - float(logp[w]))
        - (float(baseline[GOLD]) - float(baseline[w]))
        for w in WRONG
    ]
    return float(np.mean(source_terms)), float(np.mean(target_terms))


def auc(values):
    widths = np.diff(np.asarray(RHOS))
    return np.sum(
        0.5 * (values[:, :-1] + values[:, 1:]) * widths, axis=1
    ) / 4.0


def build_historical_curve(raw, prefix, routed_ids):
    s = np.zeros((len(routed_ids), len(RHOS)), dtype=float)
    t = np.zeros_like(s)
    zero = raw["cells"]["zero"]["per_sample"]
    for j, rho in enumerate(RHOS[1:], start=1):
        cell = raw["cells"][f"{prefix}|{rho:.1f}"]["per_sample"]
        for pos, idx in enumerate(routed_ids):
            s[pos, j], t[pos, j] = st_value(
                cell[idx]["logp"], zero[idx]["logp"]
            )
    return {"S": s, "T": t}


def build_fresh_curve(state, seed, routed_ids, zero):
    s = np.zeros((len(routed_ids), len(RHOS)), dtype=float)
    t = np.zeros_like(s)
    for j, rho in enumerate(RHOS[1:], start=1):
        cell = state["cells"][f"fresh_seed{seed}|{rho:.1f}"]
        for pos, idx in enumerate(routed_ids):
            s[pos, j], t[pos, j] = st_value(
                cell["per_sample"][idx]["logp"], zero[idx]["logp"]
            )
    return {"S": s, "T": t}


def mean_trajectory(curve, positions):
    return {
        "S": [float(x) for x in np.mean(curve["S"][positions], axis=0)],
        "T": [float(x) for x in np.mean(curve["T"][positions], axis=0)],
    }


def historical_class_trajectories(raw, prefix, routed_ids, positions):
    zero = raw["cells"]["zero"]["per_sample"]
    absolute = {mode: [0.0] for mode in MODES}
    relative_to_source = {mode: [0.0] for mode in MODES}
    for rho in RHOS[1:]:
        cell = raw["cells"][f"{prefix}|{rho:.1f}"]["per_sample"]
        for mode in MODES:
            abs_values = [
                float(cell[routed_ids[pos]]["logp"][mode])
                - float(zero[routed_ids[pos]]["logp"][mode])
                for pos in positions
            ]
            relative_values = [
                (float(cell[routed_ids[pos]]["logp"][mode])
                 - float(cell[routed_ids[pos]]["logp"][SOURCE]))
                - (float(zero[routed_ids[pos]]["logp"][mode])
                   - float(zero[routed_ids[pos]]["logp"][SOURCE]))
                for pos in positions
            ]
            absolute[mode].append(float(np.mean(abs_values)))
            relative_to_source[mode].append(float(np.mean(relative_values)))
    return {
        "absolute_logp_change": absolute,
        "relative_to_source_change": relative_to_source,
    }


def percentile_ci(values, bootstrap_indices):
    draws = np.mean(values[bootstrap_indices], axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def empirical(real, fresh_values):
    fresh = np.asarray(fresh_values, dtype=float)
    exceed = int(np.sum(fresh >= real))
    return {
        "real": float(real),
        "fresh_values": [float(x) for x in fresh],
        "fresh_k": FRESH_K,
        "fresh_exceedance_count": exceed,
        "rank_desc_ties_against_real": 1 + exceed,
        "p_add_one": float((1 + exceed) / 80),
        "fresh_mean": float(np.mean(fresh)),
        "fresh_sample_sd": float(np.std(fresh, ddof=1)),
        "fresh_min": float(np.min(fresh)),
        "fresh_max": float(np.max(fresh)),
        "real_minus_fresh_mean": float(real - np.mean(fresh)),
    }


def summarize_population(positions, real, reverse, fresh_curves, boot):
    real_s = auc(real["S"])[positions]
    real_t = auc(real["T"])[positions]
    reverse_s = auc(reverse["S"])[positions]
    reverse_t = auc(reverse["T"])[positions]
    fresh_s = [float(np.mean(auc(c["S"])[positions])) for c in fresh_curves]
    fresh_t = [float(np.mean(auc(c["T"])[positions])) for c in fresh_curves]
    s = empirical(float(np.mean(real_s)), fresh_s)
    t = empirical(float(np.mean(real_t)), fresh_t)
    s["sample_bootstrap_ci_95"] = percentile_ci(real_s, boot)
    t["sample_bootstrap_ci_95"] = percentile_ci(real_t, boot)
    return {
        "n": len(positions),
        "S": s,
        "T": t,
        "real_minus_reverse": {
            "S": float(np.mean(real_s - reverse_s)),
            "S_bootstrap_ci_95": percentile_ci(real_s - reverse_s, boot),
            "T": float(np.mean(real_t - reverse_t)),
            "T_bootstrap_ci_95": percentile_ci(real_t - reverse_t, boot),
        },
    }


def validate_complete_state(state, routed_ids, fresh_seeds):
    expected = {
        f"fresh_seed{seed}|{rho:.1f}"
        for seed in fresh_seeds
        for rho in RHOS[1:]
    }
    if set(state.get("cells", {})) != expected:
        raise RuntimeError(
            f"fresh state incomplete: {len(state.get('cells', {}))}/{len(expected)}"
        )
    for key in expected:
        sample = state["cells"][key]["per_sample"]
        if list(sample) != list(routed_ids):
            raise RuntimeError(f"ID order mismatch in {key}")
        for idx in routed_ids:
            record = sample[idx]
            if set(record["logp"]) != set(MODES):
                raise RuntimeError(f"mode mismatch {key} {idx}")
            if not all(math.isfinite(float(v)) for v in record["logp"].values()):
                raise RuntimeError(f"nonfinite score {key} {idx}")


def run_analysis():
    start = time.perf_counter()
    raw, checks, failures, context = gate_checks(allow_partial_state=True)
    if failures:
        raise RuntimeError("Phase 10.2 gate failed: " + ", ".join(failures))

    # Scientific imports and source/model access occur only beyond this point.
    load_scientific_dependencies()
    routed_ids = context["routed_ids"]
    primary_set = set(context["primary_ids"])
    primary_positions = np.asarray(
        [i for i, idx in enumerate(routed_ids) if idx in primary_set], dtype=int
    )
    routed_positions = np.arange(len(routed_ids), dtype=int)
    selected_rows = load_selected_validation_rows(routed_ids, raw["frozen_baseline"])

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        state = strict_json(STATE_PATH)
    else:
        state = {
            "protocol": "Phase 10.2 fresh random raw state",
            "fresh_seed_start": FRESH_SEED_START,
            "fresh_k": FRESH_K,
            "routed_ids": routed_ids,
            "cells": {},
            "scientific_endpoints_computed": False,
        }

    if state.get("scientific_endpoints_computed") is True:
        raise RuntimeError("Phase 10.2 endpoints already computed; rerun forbidden")
    if state.get("fresh_seed_start") != FRESH_SEED_START or state.get("fresh_k") != FRESH_K:
        raise RuntimeError("fresh state seed configuration mismatch")
    if state.get("routed_ids") != routed_ids:
        raise RuntimeError("fresh state population mismatch")

    model, tok, dev = load_model()
    for seed in context["fresh_seeds"]:
        unit = random_unit(seed)
        for rho in RHOS[1:]:
            key = f"fresh_seed{seed}|{rho:.1f}"
            if key in state["cells"]:
                continue
            correction = (rho * MED_INJ * unit).astype(np.float32)
            per_sample = {}
            for idx in routed_ids:
                per_sample[idx] = score_sample(
                    model, tok, dev, selected_rows[idx], correction
                )
            state["cells"][key] = {
                "seed": seed,
                "rho": rho,
                "per_sample": per_sample,
            }
            temp = STATE_PATH.with_suffix(".tmp")
            temp.write_text(json.dumps(state, ensure_ascii=False))
            os.replace(temp, STATE_PATH)

    validate_complete_state(state, routed_ids, context["fresh_seeds"])

    real = build_historical_curve(raw, "deployed|real", routed_ids)
    reverse = build_historical_curve(raw, "deployed|reverse", routed_ids)
    zero = raw["cells"]["zero"]["per_sample"]
    fresh_curves = [
        build_fresh_curve(state, seed, routed_ids, zero)
        for seed in context["fresh_seeds"]
    ]

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    boot_primary = rng.integers(
        0, len(primary_positions), size=(BOOTSTRAPS, len(primary_positions))
    )
    boot_routed = rng.integers(
        0, len(routed_positions), size=(BOOTSTRAPS, len(routed_positions))
    )
    primary = summarize_population(
        primary_positions, real, reverse, fresh_curves, boot_primary
    )
    sensitivity = summarize_population(
        routed_positions, real, reverse, fresh_curves, boot_routed
    )

    t_primary = primary["T"]
    success_conditions = {
        "real_T_AUC_gt_zero": t_primary["real"] > 0,
        "real_T_bootstrap_lower_gt_zero": t_primary["sample_bootstrap_ci_95"][0] > 0,
        "fresh_exceedance_le_3_of_79": t_primary["fresh_exceedance_count"] <= 3,
    }
    success = all(success_conditions.values())

    deployed_s = real["S"][primary_positions]
    deployed_t = real["T"][primary_positions]
    real_s_auc = auc(real["S"])[primary_positions]
    real_t_auc = auc(real["T"])[primary_positions]
    labels = []
    for s_value, t_value in zip(real_s_auc, real_t_auc):
        if abs(s_value) <= NEAR_ZERO or abs(t_value) <= NEAR_ZERO:
            labels.append("ambiguous")
        elif s_value > 0 and t_value > 0:
            labels.append("S+_T+")
        elif s_value > 0 and t_value < 0:
            labels.append("S+_T-")
        elif s_value < 0 and t_value > 0:
            labels.append("S-_T+")
        else:
            labels.append("S-_T-")

    trajectory_secondary = {
        "primary68": {
            "real": mean_trajectory(real, primary_positions),
            "reverse": mean_trajectory(reverse, primary_positions),
            "fresh79": {
                f"fresh_random{k}": {
                    "seed": context["fresh_seeds"][k],
                    **mean_trajectory(curve, primary_positions),
                }
                for k, curve in enumerate(fresh_curves)
            },
        },
        "routed74": {
            "real": mean_trajectory(real, routed_positions),
            "reverse": mean_trajectory(reverse, routed_positions),
            "fresh79": {
                f"fresh_random{k}": {
                    "seed": context["fresh_seeds"][k],
                    **mean_trajectory(curve, routed_positions),
                }
                for k, curve in enumerate(fresh_curves)
            },
        },
    }
    rho4_s_secondary = {
        "label": "PHASE10_1_MOTIVATED_NEW_DATA_SECONDARY",
        "primary68": empirical(
            float(np.mean(real["S"][primary_positions, 4])),
            [float(np.mean(c["S"][primary_positions, 4])) for c in fresh_curves],
        ),
        "routed74": empirical(
            float(np.mean(real["S"][routed_positions, 4])),
            [float(np.mean(c["S"][routed_positions, 4])) for c in fresh_curves],
        ),
    }
    class_trajectories = historical_class_trajectories(
        raw, "deployed|real", routed_ids, primary_positions
    )
    mid_dose_decomposition = {}
    for rho_index in (2, 3):
        source_change = class_trajectories["absolute_logp_change"][SOURCE][rho_index]
        non_source_change = float(np.mean([
            class_trajectories["absolute_logp_change"][mode][rho_index]
            for mode in MODES if mode != SOURCE
        ]))
        mid_dose_decomposition[f"rho{rho_index}"] = {
            "source_absolute_logp_change": source_change,
            "mean_non_source_absolute_logp_change": non_source_change,
            "S_equals_non_source_minus_source": non_source_change - source_change,
        }


    old = strict_json(P101_SUMMARY_PATH)
    summary = {
        "status": "SUCCESS" if success else "NON_SUCCESS",
        "primary_success_conditions": success_conditions,
        "primary_population": primary,
        "sensitivity_population": sensitivity,
        "secondary": {
            "direction_level_statistics_primary68": {
                "S": primary["S"],
                "T_effect_size_gap": primary["T"]["real_minus_fresh_mean"],
                "real_minus_reverse": primary["real_minus_reverse"],
            },
            "population_sensitivity_68_vs_74": sensitivity,
            "per_dose_direction_trajectories": trajectory_secondary,
            "rho4_S_PHASE10_1_MOTIVATED_NEW_DATA_SECONDARY": rho4_s_secondary,
        },
        "exploratory": {
            "sample_quadrants_primary68": dict(Counter(labels)),
            "S_T_pearson_primary68": float(np.corrcoef(real_s_auc, real_t_auc)[0, 1]),
            "S_sign_by_dose": [int(np.sign(x)) for x in np.mean(deployed_s, axis=0)],
            "S_sign_reversal_present": len(set(
                int(np.sign(x)) for x in np.mean(deployed_s, axis=0)[1:]
            )) > 1,
            "per_class_trajectories_primary68": class_trajectories,
            "mid_dose_source_vs_competitor_decomposition": mid_dose_decomposition,
        },
        "archival_hypothesis_generating_K20": {
            "primary_S": old["primary_population"]["S"],
            "primary_T": old["primary_population"]["T"],
            "pooled_with_fresh_null": False,
        },
        "execution": {
            "fresh_seed_start": FRESH_SEED_START,
            "fresh_k": FRESH_K,
            "old_k_in_primary_p": 0,
            "wall_seconds": time.perf_counter() - start,
        },
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "phase10_2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    null_rows = []
    for k, seed in enumerate(context["fresh_seeds"]):
        null_rows.append(
            {
                "identity": f"fresh_random{k}",
                "seed": seed,
                "primary68_mean_T_AUC": primary["T"]["fresh_values"][k],
                "primary68_mean_S_AUC": primary["S"]["fresh_values"][k],
                "routed74_mean_T_AUC": sensitivity["T"]["fresh_values"][k],
                "routed74_mean_S_AUC": sensitivity["S"]["fresh_values"][k],
                "T_ge_real_primary68": (
                    primary["T"]["fresh_values"][k] >= primary["T"]["real"]
                ),
            }
        )
    with (ANALYSIS_DIR / "phase10_2_fresh_direction_null.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(null_rows[0]))
        writer.writeheader()
        writer.writerows(null_rows)

    result_wording = (
        "On the frozen primary population, the deployed direction's positive "
        "target selectivity exceeded an independent 79-direction matched-norm "
        "null at the preregistered empirical threshold."
        if success
        else "Phase 10.2 did not independently confirm positive target "
        "selectivity under all preregistered criteria."
    )
    report = f"""# Phase 10.2 — Results

**Primary status:** **{summary['status']}**

{result_wording}

## Primary T-AUC confirmation

- Real T-AUC: `{t_primary['real']}`
- Bootstrap 95% CI: `{t_primary['sample_bootstrap_ci_95']}`
- Fresh-random exceedance: `{t_primary['fresh_exceedance_count']}/79`
- Rank: `{t_primary['rank_desc_ties_against_real']}/80`
- Add-one p: `{t_primary['p_add_one']}`
- Real minus fresh mean: `{t_primary['real_minus_fresh_mean']}`

All three preregistered success conditions: `{success_conditions}`.

The archived K=20 directions were excluded from the primary p-value and were not
pooled with the fresh K=79 null.
"""
    (ANALYSIS_DIR / "PHASE10_2_RESULTS.md").write_text(report)

    state["scientific_endpoints_computed"] = True
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False))
    os.replace(temp, STATE_PATH)
    print(summary["status"])
    return 0


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate-only", action="store_true")
    mode.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args()

    if args.gate_only:
        start = time.perf_counter()
        _raw, checks, failures, context = gate_checks()
        write_gate_only(checks, failures, context, time.perf_counter() - start)
        return 0 if not failures else 2

    return run_analysis()


if __name__ == "__main__":
    sys.exit(main())
