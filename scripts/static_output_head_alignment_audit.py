#!/usr/bin/env python3
"""Fail-closed static output-head alignment audit for stored SAKIKO directions.

This script intentionally does not import transformers, torch, safetensors, or
any project experiment module.  It opens no dataset, split index, activation
cache, model weight, or tokenizer.  The historical implementations establish
that the four response modes are sample-dependent candidate-answer sequences,
not four fixed one-token verbalizers.  In addition, no deployed direction is
locally available as a numerical vector.  Both facts preclude the requested
static unembedding-row decomposition.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "static_output_head_alignment"
CSV_PATH = OUT / "static_output_head_alignment.csv"
REPORT_PATH = OUT / "STATIC_OUTPUT_HEAD_ALIGNMENT_AUDIT.md"
MANIFEST_PATH = OUT / "static_output_head_alignment_manifest.json"
SCRIPT_PATH = Path(__file__).resolve()

RNG_SEED = 20260728
NULL_SAMPLE_COUNT_CONFIGURED = 1_000_000

CSV_COLUMNS = [
    "model",
    "channel",
    "estimator",
    "direction_layer",
    "injection_layer",
    "a_contrast",
    "a_offaxis",
    "a_dec",
    "a_orth",
    "decision_rank",
    "hidden_dimension",
    "null_mean",
    "null_p95",
    "null_p99",
    "stored_status",
    "is_injection_layer",
]

ALLOWED_STATUSES = {"stored", "recipe-only", "LFS-pointer", "missing"}
HARD_STOP = "HARD_STOP_SEQUENCE_LEVEL_SAMPLE_DEPENDENT_READOUT"
NOT_COMPUTED = "NOT_COMPUTED_HARD_STOP"
LAYER_UNAVAILABLE = "LAYER_PROFILE_UNAVAILABLE_RECONSTRUCTION_REQUIRED"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.rstrip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def provenance(path: str) -> dict[str, Any]:
    full = ROOT / path
    if not full.is_file():
        raise FileNotFoundError(full)
    log = git("log", "-1", "--format=%H|%cI|%s", "--", path)
    commit, committed_at, subject = log.split("|", 2)
    return {
        "path": path,
        "size_bytes": full.stat().st_size,
        "sha256": sha256(full),
        "provenance_commit": commit,
        "provenance_commit_time": committed_at,
        "provenance_subject": subject,
    }


def external_file(path: str) -> dict[str, Any]:
    full = Path(path)
    if not full.is_file():
        return {"path": path, "status": "missing"}
    return {
        "path": path,
        "status": "present",
        "size_bytes": full.stat().st_size,
        "sha256": sha256(full),
    }


def lfs_pointer(path: str) -> dict[str, Any]:
    full = ROOT / path
    raw = full.read_bytes()
    text = raw.decode("ascii", errors="strict")
    match = re.fullmatch(
        r"version https://git-lfs\.github\.com/spec/v1\n"
        r"oid sha256:([0-9a-f]{64})\n"
        r"size ([0-9]+)\n?",
        text,
    )
    if not match:
        return {
            "is_pointer": False,
            "worktree_size_bytes": len(raw),
            "worktree_sha256": hashlib.sha256(raw).hexdigest(),
        }
    oid, declared_size = match.groups()
    git_dir = Path(git("rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = ROOT / git_dir
    object_path = git_dir / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    object_present = object_path.is_file()
    object_valid = (
        object_present
        and object_path.stat().st_size == int(declared_size)
        and sha256(object_path) == oid
    )
    return {
        "is_pointer": True,
        "worktree_size_bytes": len(raw),
        "worktree_sha256": hashlib.sha256(raw).hexdigest(),
        "lfs_oid_sha256": oid,
        "lfs_declared_size_bytes": int(declared_size),
        "local_lfs_object_path": str(object_path),
        "local_lfs_object_present": object_present,
        "local_lfs_object_valid": object_valid,
    }


EVIDENCE_PATHS = [
    "final/results/clean/p0_locked_config.json",
    "final/scripts/p0_run_locked_eval.py",
    "final/scripts/run_v31.py",
    "final/results/7b_w2c_sakiko/qwen25_7b_directions.npz",
    "final/results/7b_w2c_sakiko/qwen25_7b_directions_meta.json",
    "final/results/7b_w2c_sakiko/qwen25_7b_locked_config.json",
    "scripts/qwen7b_native_sakiko.py",
    "final/results/mistral7b_w2c_sakiko_ca/pilot/locked_configs.json",
    "scripts/phase5_mistral_lib.py",
    "final/results/phase8_prospective_llama/LLAMA_FROZEN_GATE_DECISIONS.json",
    "final/results/phase8_prospective_llama/llama_rho_curves.json",
    "scripts/phase8_lib.py",
]


MODELS: dict[str, dict[str, Any]] = {
    "Phi-3.5-mini-instruct": {
        "hidden_dimension": 3072,
        "tokenizer": {
            "loader_target": (
                ".cache/modelscope/LLM-Research/Phi-3___5-mini-instruct"
            ),
            "upstream_identity": "microsoft/Phi-3.5-mini-instruct",
            "revision": "UNPINNED_IN_HISTORICAL_IMPLEMENTATION",
            "local_tokenizer_config": "NOT_PRESENT_AT_HISTORICAL_LOADER_TARGET",
            "implementation": "final/scripts/run_v31.py:49,159-160",
        },
        "readout": {
            "label_order": [
                "tool_call",
                "direct",
                "request_for_info",
                "cannot_answer",
            ],
            "verbalizer_expression": 'row["answers"].get(label, "")',
            "tokenization": (
                "tok.encode(candidate_text, add_special_tokens=False)"
            ),
            "scoring_position": (
                "all candidate positions; logits[prompt_len-1:"
                "prompt_len+cand_len-1]"
            ),
            "score": "mean token log-probability over candidate sequence",
            "implementation": "final/scripts/run_v31.py:357-402",
        },
        "config_declaration": {
            "weight_tying": "NOT_RECOVERED_NO_LOCAL_CONFIG",
            "norm_type": "NOT_RECOVERED_NO_LOCAL_CONFIG",
            "audit_branch": "NOT_ENTERED",
        },
    },
    "Qwen2.5-7B-Instruct": {
        "hidden_dimension": 3584,
        "tokenizer": {
            "loader_target": (
                ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct"
            ),
            "upstream_identity": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "UNPINNED_IN_HISTORICAL_IMPLEMENTATION",
            "local_tokenizer_config": (
                ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct/"
                "tokenizer_config.json"
            ),
            "implementation": "scripts/qwen7b_native_sakiko.py:43-44,137-139",
        },
        "readout": {
            "label_order": [
                "tool_call",
                "direct",
                "request_for_info",
                "cannot_answer",
            ],
            "verbalizer_expression": 'sample["answers"].get(label, "")',
            "tokenization": (
                "tok.encode(cand_text, add_special_tokens=False)"
            ),
            "scoring_position": (
                "all candidate positions; logits[plen-1:plen+clen-1]"
            ),
            "score": "mean token log-probability over candidate sequence",
            "implementation": "scripts/qwen7b_native_sakiko.py:377-405",
        },
        "config_declaration": {
            "weight_tying": "untied (tie_word_embeddings=false)",
            "norm_type": "RMSNorm (rms_norm_eps=1e-6)",
            "audit_branch": "NOT_ENTERED",
        },
    },
    "Mistral-7B-Instruct-v0.3": {
        "hidden_dimension": 4096,
        "tokenizer": {
            "loader_target": (
                "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3"
            ),
            "upstream_identity": "mistralai/Mistral-7B-Instruct-v0.3",
            "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
            "local_tokenizer_config": (
                "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3/"
                "tokenizer_config.json"
            ),
            "implementation": "scripts/phase5_mistral_lib.py:33-35,165-170",
        },
        "readout": {
            "label_order": [
                "direct",
                "tool_call",
                "request_for_info",
                "cannot_answer",
            ],
            "verbalizer_expression": 'sample["answers"].get(label, "")',
            "tokenization": (
                "tok.encode(cand_text, add_special_tokens=False)"
            ),
            "scoring_position": (
                "all candidate positions; logits[plen-1:plen+clen-1]"
            ),
            "score": "mean token log-probability over candidate sequence",
            "implementation": "scripts/phase5_mistral_lib.py:187-225",
        },
        "config_declaration": {
            "weight_tying": "untied (tie_word_embeddings=false)",
            "norm_type": "RMSNorm (rms_norm_eps=1e-5)",
            "audit_branch": "NOT_ENTERED",
        },
    },
    "Llama-3.1-8B-Instruct": {
        "hidden_dimension": 4096,
        "tokenizer": {
            "loader_target": (
                "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
            ),
            "upstream_identity": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
            "local_tokenizer_config": (
                "/root/autodl-tmp/models/Llama-3.1-8B-Instruct/"
                "tokenizer_config.json"
            ),
            "implementation": "scripts/phase8_lib.py:62-69,187-193",
        },
        "readout": {
            "label_order": [
                "direct",
                "tool_call",
                "request_for_info",
                "cannot_answer",
            ],
            "verbalizer_expression": 'sample["answers"].get(label, "")',
            "tokenization": (
                "tok.encode(cand_text, add_special_tokens=False)"
            ),
            "scoring_position": (
                "all candidate positions; logits[plen-1:plen+clen-1]"
            ),
            "score": "mean token log-probability over candidate sequence",
            "implementation": "scripts/phase8_lib.py:209-247",
        },
        "config_declaration": {
            "weight_tying": "untied (tie_word_embeddings=false)",
            "norm_type": "RMSNorm (rms_norm_eps=1e-5)",
            "audit_branch": "NOT_ENTERED",
        },
    },
}


CHANNELS: list[dict[str, Any]] = [
    {
        "model": "Phi-3.5-mini-instruct",
        "channel": "rfi_tc",
        "gold": "request_for_info",
        "source": "tool_call",
        "estimator": "PCA-top20 composite correction",
        "observation_layer": 18,
        "injection_layer": 14,
        "vector_dimension": 3072,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/clean/p0_locked_config.json",
            "final/scripts/run_v31.py",
        ],
        "sign_convention": "ref_mean_pc - err_mean_pc",
        "normalization_convention": (
            "inverse-scaled PCA axes individually L2-normalized; signed top-20 "
            "terms alpha-weighted and summed; no final unit normalization"
        ),
    },
    {
        "model": "Phi-3.5-mini-instruct",
        "channel": "ca_tc",
        "gold": "cannot_answer",
        "source": "tool_call",
        "estimator": "PCA-top20 composite correction",
        "observation_layer": 18,
        "injection_layer": 16,
        "vector_dimension": 3072,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/clean/p0_locked_config.json",
            "final/scripts/run_v31.py",
        ],
        "sign_convention": "ref_mean_pc - err_mean_pc",
        "normalization_convention": (
            "inverse-scaled PCA axes individually L2-normalized; signed top-20 "
            "terms alpha-weighted and summed; no final unit normalization"
        ),
    },
    {
        "model": "Phi-3.5-mini-instruct",
        "channel": "ca_direct",
        "gold": "cannot_answer",
        "source": "direct",
        "estimator": "PCA-top20 composite correction",
        "observation_layer": 18,
        "injection_layer": 16,
        "vector_dimension": 3072,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/clean/p0_locked_config.json",
            "final/scripts/run_v31.py",
        ],
        "sign_convention": "ref_mean_pc - err_mean_pc",
        "normalization_convention": (
            "inverse-scaled PCA axes individually L2-normalized; signed top-20 "
            "terms alpha-weighted and summed; no final unit normalization"
        ),
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "channel": "rfi_tc",
        "gold": "request_for_info",
        "source": "tool_call",
        "estimator": "diffmean",
        "observation_layer": 20,
        "injection_layer": 18,
        "vector_dimension": 3584,
        "stored_status": "LFS-pointer",
        "exact_paths": [
            "final/results/7b_w2c_sakiko/qwen25_7b_directions.npz"
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "channel": "ca_tc",
        "gold": "cannot_answer",
        "source": "tool_call",
        "estimator": "diffmean",
        "observation_layer": 20,
        "injection_layer": 16,
        "vector_dimension": 3584,
        "stored_status": "LFS-pointer",
        "exact_paths": [
            "final/results/7b_w2c_sakiko/qwen25_7b_directions.npz"
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "channel": "ca_direct",
        "gold": "cannot_answer",
        "source": "direct",
        "estimator": "diffmean",
        "observation_layer": 16,
        "injection_layer": 16,
        "vector_dimension": 3584,
        "stored_status": "LFS-pointer",
        "exact_paths": [
            "final/results/7b_w2c_sakiko/qwen25_7b_directions.npz"
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Mistral-7B-Instruct-v0.3",
        "channel": "rfi_tc",
        "gold": "request_for_info",
        "source": "tool_call",
        "estimator": "diffmean",
        "observation_layer": 22,
        "injection_layer": 20,
        "vector_dimension": 4096,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/mistral7b_w2c_sakiko_ca/pilot/locked_configs.json",
            "scripts/phase5_mistral_lib.py",
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Mistral-7B-Instruct-v0.3",
        "channel": "ca_tc",
        "gold": "cannot_answer",
        "source": "tool_call",
        "estimator": "diffmean",
        "observation_layer": 22,
        "injection_layer": 18,
        "vector_dimension": 4096,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/mistral7b_w2c_sakiko_ca/pilot/locked_configs.json",
            "scripts/phase5_mistral_lib.py",
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Mistral-7B-Instruct-v0.3",
        "channel": "ca_direct",
        "gold": "cannot_answer",
        "source": "direct",
        "estimator": "pca1",
        "observation_layer": 22,
        "injection_layer": 18,
        "vector_dimension": 4096,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/mistral7b_w2c_sakiko_ca/pilot/locked_configs.json",
            "scripts/phase5_mistral_lib.py",
        ],
        "sign_convention": (
            "PC1 sign aligned to standardized reference-minus-error mean"
        ),
        "normalization_convention": (
            "inverse StandardScaler scale, then final L2 unit normalization"
        ),
    },
    {
        "model": "Llama-3.1-8B-Instruct",
        "channel": "rfi_tc",
        "gold": "request_for_info",
        "source": "tool_call",
        "estimator": "diffmean",
        "observation_layer": 22,
        "injection_layer": 22,
        "vector_dimension": 4096,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/phase8_prospective_llama/llama_rho_curves.json",
            "scripts/phase8_lib.py",
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Llama-3.1-8B-Instruct",
        "channel": "ca_tc",
        "gold": "cannot_answer",
        "source": "tool_call",
        "estimator": "diffmean",
        "observation_layer": 22,
        "injection_layer": 18,
        "vector_dimension": 4096,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/phase8_prospective_llama/llama_rho_curves.json",
            "scripts/phase8_lib.py",
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
    {
        "model": "Llama-3.1-8B-Instruct",
        "channel": "ca_direct",
        "gold": "cannot_answer",
        "source": "direct",
        "estimator": "diffmean",
        "observation_layer": 22,
        "injection_layer": 18,
        "vector_dimension": 4096,
        "stored_status": "recipe-only",
        "exact_paths": [
            "final/results/phase8_prospective_llama/llama_rho_curves.json",
            "scripts/phase8_lib.py",
        ],
        "sign_convention": "correct gold-reference mean - channel-error mean",
        "normalization_convention": "final L2 unit normalization",
    },
]


LEGACY_REJECTED_CANDIDATES = [
    "old/sakiko_v2/artifacts/fisher_ca2tc_direction.npy",
    "old/sakiko_v2/artifacts/fisher_lda_direction.npy",
    "old/sakiko_v2/artifacts/fisher_mc_direction.npy",
]


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(cell(x) for x in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def build_report(
    *,
    head: str,
    generated_utc: str,
    evidence: dict[str, dict[str, Any]],
    qwen_pointer: dict[str, Any],
    legacy_candidates: list[dict[str, Any]],
) -> str:
    ledger_rows = []
    for row in CHANNELS:
        files = "<br>".join(f"`{p}`" for p in row["exact_paths"])
        hashes = "<br>".join(
            f"`{evidence[p]['sha256']}`" for p in row["exact_paths"]
        )
        commits = "<br>".join(
            f"`{evidence[p]['provenance_commit']}`"
            for p in row["exact_paths"]
        )
        ledger_rows.append(
            [
                row["model"],
                row["channel"],
                row["estimator"],
                row["observation_layer"],
                row["injection_layer"],
                row["vector_dimension"],
                row["stored_status"],
                files,
                hashes,
                commits,
                row["sign_convention"],
                row["normalization_convention"],
            ]
        )

    mode_rows = []
    branch_rows = []
    for model, info in MODELS.items():
        tok = info["tokenizer"]
        readout = info["readout"]
        mode_rows.append(
            [
                model,
                "<br>".join(readout["label_order"]),
                f"`{readout['verbalizer_expression']}`",
                "sample-dependent; no fixed token IDs",
                "not defined",
                readout["scoring_position"],
                readout["score"],
                HARD_STOP,
            ]
        )
        branch = info["config_declaration"]
        branch_rows.append(
            [
                model,
                tok["upstream_identity"],
                tok["revision"],
                tok["loader_target"],
                branch["weight_tying"],
                branch["norm_type"],
                branch["audit_branch"],
            ]
        )

    legacy_rows = []
    for item in legacy_candidates:
        ptr = item["pointer"]
        legacy_rows.append(
            [
                item["path"],
                ptr.get("worktree_sha256"),
                ptr.get("lfs_oid_sha256"),
                ptr.get("lfs_declared_size_bytes"),
                ptr.get("local_lfs_object_valid"),
                (
                    "rejected: legacy Fisher/LDA artifact does not equal the "
                    "deployed Phase-P0 PCA-top20 composite recipe"
                ),
            ]
        )

    report = f"""# Static Output-Head Alignment Audit of Stored SAKIKO Directions

**Tier:** POST-HOC DESCRIPTIVE / LIGHT
**Generated UTC:** `{generated_utc}`
**Source Git HEAD:** `{head}`
**Mechanical outcome:** `NO_ADMISSIBLE_DECOMPOSITION`

## Executive result

No historical channel can proceed to static output-head decomposition.

1. None of the 12 deployed channel directions is locally available as an
   actual numerical vector. Phi, Mistral, and Llama preserve reconstruction
   recipes only. Qwen preserves one Git-LFS pointer containing the three
   channel vectors, but the LFS object is not locally present and was not
   pulled.
2. Independently, all four historical model implementations score the
   sample-specific full candidate answer for each mode with mean sequence
   log-probability. The repository does not define four fixed mode
   verbalizers or four fixed mode token IDs. Replacing those answers with
   label-name tokens would invent a different readout. The task-mandated
   per-model hard stop therefore applies before any output-head parameter is
   read.

The CSV consequently contains 12 unavailable-channel rows; all decomposition and null fields are blank.
No safetensor, model weight, activation, dataset, split, or sealed-index file
was opened by the reproducible script. No tokenizer was instantiated; only
`tokenizer_config.json` and `config.json` metadata were hashed.

## Stored-direction ledger

{md_table(
    [
        "model",
        "channel",
        "estimator",
        "obs",
        "inj",
        "dim",
        "status",
        "exact path",
        "file SHA256",
        "provenance commit",
        "sign",
        "normalization",
    ],
    ledger_rows,
)}

Qwen pointer details:

- Worktree pointer SHA256: `{qwen_pointer['worktree_sha256']}`
- Expected LFS object SHA256: `{qwen_pointer['lfs_oid_sha256']}`
- Expected numerical object size: `{qwen_pointer['lfs_declared_size_bytes']}` bytes
- Valid local LFS object present: `{qwen_pointer['local_lfs_object_valid']}`
- No `git lfs pull` or equivalent network/materialization action was run.

The Qwen path is classified `LFS-pointer`, not `stored`: hashing the pointer
text is not evidence that the expected NPZ bytes are locally readable.

### Rejected legacy Phi candidates

{md_table(
    [
        "path",
        "pointer SHA256",
        "expected LFS object SHA256",
        "expected bytes",
        "valid local object",
        "adjudication",
    ],
    legacy_rows,
)}

These files are also LFS pointers, but more importantly their estimators do not
match the deployed Phase-P0 PCA-top20 composite correction. They are not used
as substitutes.

## Exact historical mode readout

{md_table(
    [
        "model",
        "mode keys / argmax order",
        "historical candidate expression",
        "exact token IDs",
        "four unique first tokens?",
        "scoring positions",
        "historical score",
        "adjudication",
    ],
    mode_rows,
)}

The four requested semantic modes are keys selecting answer text, not literal
verbalizers. Their token IDs vary with the sample answer, and recovering those
IDs would require opening the prohibited dataset. More fundamentally, the
historical score is sequence-level even if a particular answer happened to be
one token. No one-row unembedding proxy exactly represents this scoring rule.

## Tokenizer identity and declared architecture branches

{md_table(
    [
        "model",
        "tokenizer/model identity",
        "historical revision",
        "historical loader target",
        "config-declared tying",
        "config-declared norm",
        "static audit branch",
    ],
    branch_rows,
)}

The declarations for Qwen, Mistral, and Llama come from local `config.json`
metadata. They are reported for inventory only. They were not verified against
output-head tensors because the task requires stopping at the readout failure.
Phi's historical tokenizer target is absent and its revision was not pinned by
the scoring implementation.

## Static proxy, decision subspace, and decomposition

- Output-head/norm parameter names and shard hashes: `{NOT_COMPUTED}`
- Tied/untied tensor branch actually traversed: `{NOT_COMPUTED}`
- RMSNorm/LayerNorm proxy actually traversed: `{NOT_COMPUTED}`
- Singular values: `{NOT_COMPUTED}`
- Numerical-rank tolerance: `{NOT_COMPUTED}`
- Decision-subspace rank: `{NOT_COMPUTED}`
- Per-channel `a_contrast`, `a_offaxis`, `a_dec`, `a_orth`: `{NOT_COMPUTED}`

No claim is made that the config-declared branch equals a tensor-verified
branch.

Had an eligible model existed, the audit would have stated:

> This is a static parameter-space proxy. The true activation-conditioned
> Jacobian of RMSNorm or LayerNorm depends on the hidden activation and is not
> computed here.

That proxy was not constructed in this run.

## Random-orientation null

- Fixed seed reserved: `{RNG_SEED}`
- Configured Monte Carlo sample count: `{NULL_SAMPLE_COUNT_CONFIGURED}`
- Samples generated: `0`
- Theoretical mean/p50/p95/p99: `{NOT_COMPUTED}`
- Empirical mean/p95/p99/max: `{NOT_COMPUTED}`

The null depends on an observed decision rank. Since no decision subspace was
built, reporting Beta quantiles would manufacture a rank assumption.

## Layer profiles

Every channel: `{LAYER_UNAVAILABLE}`.

No one-layer vector was copied across depth. No recipe-only layer direction was
reconstructed.

## Consistency and scope checks

- Ledger rows: `12` (3 channels × 4 models).
- Locally numerical deployed directions: `0`.
- CSV channel rows: `12`; computed decomposition rows: `0`.
- All models stop at the historical sequence-level readout.
- Existing evidence remained read-only.
- CPU only; no GPU API was invoked.
- No full-model load, forward pass, backward pass, inference, prediction,
  scientific endpoint, dataset, sealed split index, activation cache, direction
  reconstruction, LFS pull, Qwen3 work, or Phase 10.3 modification.

## Interpretation boundary

This audit cannot characterize strong or weak alignment because no admissible
decomposition exists. It does not confirm F2, prove runner-up capture, make a
causal-mechanism claim, explain behavioral correction/failure, or remove the
need for any later experiment.
"""
    return report


def main() -> None:
    dirty = git("status", "--short").splitlines()
    allowed_dirty_paths = {
        str(SCRIPT_PATH.relative_to(ROOT)),
        str(CSV_PATH.relative_to(ROOT)),
        str(REPORT_PATH.relative_to(ROOT)),
        str(MANIFEST_PATH.relative_to(ROOT)),
    }
    allowed_output_dir = str(OUT.relative_to(ROOT)) + "/"
    unrelated_dirty = [
        line
        for line in dirty
        if line[3:] not in allowed_dirty_paths
        and line[3:] != allowed_output_dir
    ]
    if unrelated_dirty:
        raise RuntimeError(
            "Refusing to run with unrelated worktree changes: "
            + "; ".join(unrelated_dirty)
        )

    head = git("rev-parse", "HEAD")
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence = {path: provenance(path) for path in EVIDENCE_PATHS}

    qwen_path = (
        "final/results/7b_w2c_sakiko/qwen25_7b_directions.npz"
    )
    qwen_pointer = lfs_pointer(qwen_path)
    if not qwen_pointer.get("is_pointer"):
        raise RuntimeError("Expected Qwen direction artifact to be an LFS pointer")
    if qwen_pointer.get("local_lfs_object_valid"):
        raise RuntimeError(
            "A numerical Qwen direction object is now local; update the audit "
            "rather than silently retaining the pointer-only adjudication"
        )

    legacy_candidates = []
    for path in LEGACY_REJECTED_CANDIDATES:
        legacy_candidates.append(
            {
                "path": path,
                "evidence": provenance(path),
                "pointer": lfs_pointer(path),
            }
        )

    for row in CHANNELS:
        if row["stored_status"] not in ALLOWED_STATUSES:
            raise AssertionError(row)
        if row["stored_status"] == "stored":
            raise RuntimeError(
                "A locally numerical deployed direction requires a new "
                "eligibility review before any decomposition"
            )
        if row["vector_dimension"] != MODELS[row["model"]]["hidden_dimension"]:
            raise AssertionError(row)
        for path in row["exact_paths"]:
            if path not in evidence:
                raise AssertionError(f"unhashed evidence path: {path}")

    if len(CHANNELS) != 12:
        raise AssertionError("expected 12 channel ledger rows")
    if any(
        info["readout"]["score"]
        != "mean token log-probability over candidate sequence"
        for info in MODELS.values()
    ):
        raise AssertionError("all four historical readouts must be sequence-level")

    OUT.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in CHANNELS:
            writer.writerow({
                "model": row["model"],
                "channel": row["channel"],
                "estimator": row["estimator"],
                "direction_layer": row["observation_layer"],
                "injection_layer": row["injection_layer"],
                "a_contrast": "",
                "a_offaxis": "",
                "a_dec": "",
                "a_orth": "",
                "decision_rank": "",
                "hidden_dimension": row["vector_dimension"],
                "null_mean": "",
                "null_p95": "",
                "null_p99": "",
                "stored_status": row["stored_status"],
                "is_injection_layer": (
                    row["observation_layer"] == row["injection_layer"]
                ),
            })

    report = build_report(
        head=head,
        generated_utc=generated_utc,
        evidence=evidence,
        qwen_pointer=qwen_pointer,
        legacy_candidates=legacy_candidates,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    tokenizer_files = {
        "Qwen2.5-7B-Instruct": external_file(
            str(
                ROOT
                / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct/"
                "tokenizer_config.json"
            )
        ),
        "Mistral-7B-Instruct-v0.3": external_file(
            "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3/"
            "tokenizer_config.json"
        ),
        "Llama-3.1-8B-Instruct": external_file(
            "/root/autodl-tmp/models/Llama-3.1-8B-Instruct/"
            "tokenizer_config.json"
        ),
        "Phi-3.5-mini-instruct": {
            "path": str(
                ROOT
                / ".cache/modelscope/LLM-Research/"
                "Phi-3___5-mini-instruct/tokenizer_config.json"
            ),
            "status": "missing",
        },
    }

    manifest = {
        "schema_version": 1,
        "title": "Static output-head alignment audit of stored SAKIKO directions",
        "tier": "POST-HOC DESCRIPTIVE / LIGHT",
        "generated_utc": generated_utc,
        "git": {
            "head": head,
            "branch": git("branch", "--show-current"),
            "worktree_clean_at_task_start": True,
            "unrelated_worktree_changes_before_generation": [],
        },
        "mechanical_outcome": "NO_ADMISSIBLE_DECOMPOSITION",
        "eligibility": {
            "ledger_rows": len(CHANNELS),
            "locally_numerical_deployed_directions": 0,
            "csv_channel_rows": len(CHANNELS),
            "csv_decomposition_rows": 0,
            "all_models_readout_status": HARD_STOP,
        },
        "channels": CHANNELS,
        "models": MODELS,
        "direction_files": {
            qwen_path: {
                **evidence[qwen_path],
                **qwen_pointer,
            },
            "legacy_rejected_candidates": legacy_candidates,
        },
        "tokenizer_identity": tokenizer_files,
        "model_config_identity": {
            "Qwen2.5-7B-Instruct": external_file(str(
                ROOT / ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct/config.json"
            )),
            "Mistral-7B-Instruct-v0.3": external_file(
                "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3/config.json"
            ),
            "Llama-3.1-8B-Instruct": external_file(
                "/root/autodl-tmp/models/Llama-3.1-8B-Instruct/config.json"
            ),
            "Phi-3.5-mini-instruct": {
                "status": "missing_at_historical_loader_target"
            },
        },
        "mode_token_ids": {
            model: "NOT_DEFINED_SAMPLE_DEPENDENT_CANDIDATE_SEQUENCE"
            for model in MODELS
        },
        "weight_files": [],
        "weight_files_status": "NOT_READ_MODE_READOUT_HARD_STOP",
        "static_proxy": {
            "status": NOT_COMPUTED,
            "decision_rank": None,
            "singular_values": None,
            "rank_tolerance": None,
        },
        "random_orientation_null": {
            "status": NOT_COMPUTED,
            "seed": RNG_SEED,
            "sample_count_configured": NULL_SAMPLE_COUNT_CONFIGURED,
            "sample_count_generated": 0,
        },
        "software_versions": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "safetensors": package_version("safetensors"),
        },
        "source_evidence": evidence,
        "artifacts": {
            str(SCRIPT_PATH.relative_to(ROOT)): {
                "sha256": sha256(SCRIPT_PATH)
            },
            str(CSV_PATH.relative_to(ROOT)): {"sha256": sha256(CSV_PATH)},
            str(REPORT_PATH.relative_to(ROOT)): {
                "sha256": sha256(REPORT_PATH)
            },
        },
        "consistency_checks": {
            "ledger_is_three_channels_by_four_models": len(CHANNELS) == 12,
            "all_statuses_allowed": all(
                row["stored_status"] in ALLOWED_STATUSES for row in CHANNELS
            ),
            "no_stored_numerical_direction": not any(
                row["stored_status"] == "stored" for row in CHANNELS
            ),
            "qwen_worktree_file_is_lfs_pointer": qwen_pointer["is_pointer"],
            "qwen_numerical_lfs_object_absent": not qwen_pointer[
                "local_lfs_object_valid"
            ],
            "all_readouts_are_sample_dependent_sequences": True,
            "csv_header_exact": True,
            "csv_channel_rows_equal_ledger_rows": True,
            "csv_decomposition_fields_all_blank": True,
            "weight_files_read_zero": True,
            "null_samples_generated_zero": True,
        },
        "constraint_attestations": {
            "cpu_only": True,
            "gpu_used": False,
            "full_model_loaded": False,
            "forward_pass": False,
            "backward_pass": False,
            "inference_or_prediction": False,
            "scientific_endpoint": False,
            "dataset_source_accessed": False,
            "sealed_index_accessed": False,
            "activation_cache_accessed": False,
            "direction_reconstructed": False,
            "lfs_object_pulled": False,
            "qwen3_work": False,
            "phase10_3_modified": False,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "outcome": manifest["mechanical_outcome"],
            "ledger_rows": len(CHANNELS),
            "csv_channel_rows": len(CHANNELS),
            "decomposition_rows": 0,
            "report": str(REPORT_PATH.relative_to(ROOT)),
            "csv": str(CSV_PATH.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
