#!/usr/bin/env python3
"""Resolve Phase 10.2 tokenizer provenance without loading model weights."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/sakiko-followup")
OUT_DIR = ROOT / "final" / "results" / "phase10_2_expanded_random_null"

REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
CACHE_DIR = Path(
    "/root/autodl-tmp/hf_phase10_2_tokenizer_exact_revision_v4_source"
)
OLD_LOCAL_TOKENIZER_CONFIG = Path(
    "/root/autodl-tmp/models/Llama-3.1-8B-Instruct/tokenizer_config.json"
)
OLD_LOCAL_VALID_SHA256 = (
    "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
)
V3_MALFORMED_VALUE = (
    "177c7b61e616fecb84c17ce0591acb92c6d60e9ac5ababfb940ff23bbcd424"
)
STARTING_HEAD = "67f1d9e8a1017b89c6b4c7dacf1d204add58ee58"
REAUDIT_COMMIT = STARTING_HEAD

SCRIPT_PATH = ROOT / "scripts" / "phase10_2_tokenizer_upstream_resolution.py"
REPORT_MD = OUT_DIR / "PHASE10_2_TOKENIZER_UPSTREAM_RESOLUTION.md"
REPORT_JSON = OUT_DIR / "phase10_2_tokenizer_upstream_resolution.json"
HASH_CSV = OUT_DIR / "PHASE10_2_TOKENIZER_UPSTREAM_HASHES.csv"
DOWNLOAD_LOG = OUT_DIR / "PHASE10_2_TOKENIZER_UPSTREAM_DOWNLOAD.log"

ALLOW_PATTERNS = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "config.json",
    "*.model",
    "*.vocab",
    "merges.txt",
    "vocab.json",
    "sentencepiece*",
    "spiece.model",
)
IGNORE_PATTERNS = (
    "model-*.safetensors",
    "*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model*",
    "*.bin",
    "*.gguf",
    "*.pt",
    "*.pth",
    "original/consolidated*",
    "**/consolidated*",
    "*.npy",
    "*.npz",
    "*.jsonl",
    "data/**",
    "dataset/**",
)
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
SMOKE_TEST_TEXT = (
    "A public tokenizer provenance check: ask for clarification before acting."
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CacheNotFresh(RuntimeError):
    pass


class MaterializationInvalid(RuntimeError):
    pass


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if not SHA256_RE.fullmatch(value):
        raise MaterializationInvalid(f"invalid SHA256 generated for {path}")
    return value


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def sanitize_error(value: str) -> str:
    value = re.sub(r"hf_[A-Za-z0-9]{10,}", "<REDACTED_HF_TOKEN>", value)
    value = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)\S+",
        r"\1<REDACTED>",
        value,
    )
    value = re.sub(r"(?i)(token=)[^&\s]+", r"\1<REDACTED>", value)
    value = re.sub(r"https?://\S+\?\S+", "<URL_WITH_QUERY_REDACTED>", value)
    return value


def matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def inspect_cache_freshness() -> str:
    if not CACHE_DIR.exists():
        return "CACHE_FRESH"
    if not CACHE_DIR.is_dir():
        raise CacheNotFresh(f"cache path exists but is not a directory: {CACHE_DIR}")
    if any(CACHE_DIR.iterdir()):
        raise CacheNotFresh(f"dedicated cache is nonempty: {CACHE_DIR}")
    return "CACHE_FRESH_EMPTY_DIRECTORY"


def cache_support_inventory() -> dict[str, list[dict[str, Any]]]:
    inventory: dict[str, list[dict[str, Any]]] = {
        "refs": [],
        "locks": [],
        "download_metadata": [],
    }
    for path in sorted(CACHE_DIR.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(CACHE_DIR).as_posix()
        category = None
        if "/refs/" in f"/{relative}/" or relative.endswith("/refs"):
            category = "refs"
        elif "/.locks/" in f"/{relative}/" or relative.endswith(".lock"):
            category = "locks"
        elif (
            relative.endswith(".metadata")
            or "/.cache/" in f"/{relative}/"
            or relative.endswith(".json")
            and "snapshots/" not in relative
        ):
            category = "download_metadata"
        if category is None:
            continue
        stat_result = path.stat() if path.exists() else path.lstat()
        item: dict[str, Any] = {
            "path": str(path.absolute()),
            "relative_path": relative,
            "file_type": "symlink" if path.is_symlink() else "regular",
            "size_bytes": stat_result.st_size,
            "inode": stat_result.st_ino,
            "mtime_utc": utc_from_timestamp(stat_result.st_mtime),
        }
        if category == "refs" and path.is_file() and stat_result.st_size < 4096:
            item["value"] = path.read_text(errors="replace").strip()
        inventory[category].append(item)
    return inventory


def inspect_snapshot(snapshot_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_real = CACHE_DIR.resolve(strict=True)
    snapshot_absolute = snapshot_path.absolute()
    snapshot_real = snapshot_path.resolve(strict=True)

    incomplete = sorted(
        str(path.absolute())
        for path in CACHE_DIR.rglob("*")
        if path.name.endswith(".incomplete")
    )
    broken = sorted(
        str(path.absolute())
        for path in CACHE_DIR.rglob("*")
        if path.is_symlink() and not path.exists()
    )

    rows: list[dict[str, Any]] = []
    unexpected_weights: list[str] = []
    outside_cache: list[str] = []
    old_local_references: list[str] = []
    zero_byte_snapshot_files: list[str] = []

    for path in sorted(snapshot_path.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        if path.is_symlink() and not path.exists():
            continue

        relative = path.relative_to(snapshot_path).as_posix()
        real = path.resolve(strict=True)
        link_stat = path.lstat()
        target_stat = real.stat()
        digest = sha256_file(real)
        symlink_target = os.readlink(path) if path.is_symlink() else None
        is_inside = real.is_relative_to(cache_real)
        resolves_to_old = real.is_relative_to(
            OLD_LOCAL_TOKENIZER_CONFIG.parent.resolve(strict=True)
        )

        forbidden = matches_any(relative, IGNORE_PATTERNS)
        if forbidden:
            unexpected_weights.append(relative)
        if not is_inside:
            outside_cache.append(relative)
        if resolves_to_old:
            old_local_references.append(relative)
        if target_stat.st_size == 0:
            zero_byte_snapshot_files.append(relative)

        blob_path = str(real) if "blobs" in real.parts else None
        rows.append(
            {
                "relative_path": relative,
                "absolute_snapshot_path": str(path.absolute()),
                "realpath": str(real),
                "blob_path": blob_path,
                "symlink_target": symlink_target,
                "size_bytes": target_stat.st_size,
                "snapshot_inode": link_stat.st_ino,
                "inode": target_stat.st_ino,
                "snapshot_mtime_utc": utc_from_timestamp(link_stat.st_mtime),
                "mtime_utc": utc_from_timestamp(target_stat.st_mtime),
                "file_type": "symlink" if path.is_symlink() else "regular",
                "sha256": digest,
                "resolves_inside_dedicated_cache": is_inside,
                "resolves_to_old_local_model": resolves_to_old,
            }
        )

    present = {row["relative_path"] for row in rows}
    missing_required = [name for name in REQUIRED_FILES if name not in present]
    missing_blob_relation = [
        row["relative_path"] for row in rows if row["blob_path"] is None
    ]
    invalid_sha = [
        row["relative_path"]
        for row in rows
        if not SHA256_RE.fullmatch(str(row["sha256"]))
    ]

    checks = {
        "snapshot_path": str(snapshot_absolute),
        "snapshot_realpath": str(snapshot_real),
        "snapshot_directory_name": snapshot_path.name,
        "snapshot_file_count": len(rows),
        "incomplete_paths": incomplete,
        "broken_symlinks": broken,
        "zero_byte_snapshot_files": zero_byte_snapshot_files,
        "missing_required_files": missing_required,
        "unexpected_weight_files": unexpected_weights,
        "files_resolving_outside_cache": outside_cache,
        "files_resolving_to_old_local_model": old_local_references,
        "files_without_blob_relation": missing_blob_relation,
        "invalid_sha256_rows": invalid_sha,
        "support_inventory": cache_support_inventory(),
    }
    return rows, checks


def offline_tokenizer_validation(snapshot_path: Path) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )

    def smoke_once() -> dict[str, Any]:
        encoded = tokenizer(
            SMOKE_TEST_TEXT,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids = list(encoded["input_ids"])
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return {"token_ids": token_ids, "decoded": decoded}

    first = smoke_once()
    second = smoke_once()
    deterministic = first == second
    if not deterministic:
        raise MaterializationInvalid("tokenizer smoke test was not deterministic")

    tokenizer_json = snapshot_path / "tokenizer.json"
    return {
        "loaded_from": str(snapshot_path.absolute()),
        "local_files_only": True,
        "trust_remote_code": False,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "len_tokenizer": int(len(tokenizer)),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "model_max_length": tokenizer.model_max_length,
        "special_tokens_map": json_safe(tokenizer.special_tokens_map),
        "chat_template_sha256": sha256_text(tokenizer.chat_template),
        "tokenizer_config_json_sha256": sha256_file(
            snapshot_path / "tokenizer_config.json"
        ),
        "tokenizer_json_sha256": (
            sha256_file(tokenizer_json) if tokenizer_json.is_file() else None
        ),
        "smoke_test_text": SMOKE_TEST_TEXT,
        "smoke_run_1": first,
        "smoke_run_2": second,
        "deterministic": deterministic,
    }


def write_hash_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "relative_path",
        "absolute_snapshot_path",
        "realpath",
        "blob_path",
        "symlink_target",
        "size_bytes",
        "snapshot_inode",
        "inode",
        "snapshot_mtime_utc",
        "mtime_utc",
        "file_type",
        "sha256",
        "resolves_inside_dedicated_cache",
        "resolves_to_old_local_model",
    ]
    with HASH_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_resolution_json(result: dict[str, Any]) -> None:
    REPORT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_resolution_md(result: dict[str, Any]) -> None:
    checks = result.get("snapshot_checks", {})
    validation = result.get("offline_tokenizer_validation")
    comparison = result.get("digest_comparison", {})
    lines = [
        "# Phase 10.2 Tokenizer Upstream Resolution",
        "",
        f"**Final mechanical verdict:** `{result['final_verdict']}`",
        "",
        "## Frozen identity and scope",
        "",
        f"- Starting HEAD / regenerated re-audit commit: `{REAUDIT_COMMIT}`",
        f"- Official repository: `{REPO_ID}`",
        f"- Requested exact revision: `{REVISION}`",
        f"- Resolved revision: `{result.get('resolved_revision')}`",
        f"- Dedicated cache: `{CACHE_DIR}`",
        f"- Cache freshness: `{result.get('cache_freshness')}`",
        f"- Returned snapshot: `{result.get('snapshot_path')}`",
        f"- Snapshot realpath: `{result.get('snapshot_realpath')}`",
        "",
        "V3 remains permanently VOID. This materialization does not repair or",
        "revive V3, and no V4 file was created.",
        "",
        "## Digest adjudication",
        "",
        f"- Malformed V3 value: `{V3_MALFORMED_VALUE}`",
        f"- Old local valid SHA256: `{comparison.get('old_local_direct_sha256')}`",
        f"- Fresh official SHA256: `{comparison.get('fresh_tokenizer_config_sha256')}`",
        f"- Fresh equals old: `{comparison.get('fresh_equals_old_local_valid')}`",
        "",
        "## Snapshot and blob provenance",
        "",
        f"- Snapshot file count: `{checks.get('snapshot_file_count')}`",
        f"- Incomplete paths: `{checks.get('incomplete_paths')}`",
        f"- Broken symlinks: `{checks.get('broken_symlinks')}`",
        f"- Zero-byte snapshot files: `{checks.get('zero_byte_snapshot_files')}`",
        f"- Missing required files: `{checks.get('missing_required_files')}`",
        f"- Unexpected weights: `{checks.get('unexpected_weight_files')}`",
        f"- Files outside dedicated cache: `{checks.get('files_resolving_outside_cache')}`",
        f"- Files referring to old model: `{checks.get('files_resolving_to_old_local_model')}`",
        f"- Files without blob relation: `{checks.get('files_without_blob_relation')}`",
        "",
        "Refs, locks, API download metadata, snapshot paths, symlink targets,",
        "blob paths, sizes, inodes, mtimes, file types, and directly computed",
        "SHA256 values are preserved in the JSON and CSV evidence.",
        "",
        "## Tokenizer-only offline validation",
        "",
    ]
    if validation:
        lines.extend(
            [
                f"- Loaded from fresh snapshot: `{validation.get('loaded_from')}`",
                f"- Class: `{validation.get('tokenizer_class')}`",
                f"- Vocabulary size: `{validation.get('vocab_size')}`",
                f"- BOS/EOS/PAD IDs: `{validation.get('bos_token_id')}` / "
                f"`{validation.get('eos_token_id')}` / "
                f"`{validation.get('pad_token_id')}`",
                f"- Padding side: `{validation.get('padding_side')}`",
                f"- Model max length: `{validation.get('model_max_length')}`",
                f"- Chat-template SHA256: `{validation.get('chat_template_sha256')}`",
                f"- Deterministic two-run smoke test: `{validation.get('deterministic')}`",
            ]
        )
    else:
        lines.append("- Offline validation did not complete.")
    lines.extend(
        [
            "",
            "## Execution exclusions",
            "",
            "- zero GPU;",
            "- zero full-model loading;",
            "- zero inference and predictions;",
            "- zero scientific endpoints;",
            "- zero project dataset, routed-ID, validation, or test access;",
            "- old model/tokenizer unchanged;",
            "- V3 VOID unchanged;",
            "- no V4 created;",
            "- Qwen3 not started.",
            "",
        ]
    )
    if result.get("error"):
        lines.extend(["## Blocking error", "", f"`{result['error']}`", ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def classify_upstream_exception(exc: BaseException) -> tuple[str, str]:
    text = sanitize_error(f"{type(exc).__name__}: {exc}")
    lowered = text.lower()
    if any(token in lowered for token in ("401", "403", "gated", "auth")):
        return "UPSTREAM_CHECK_BLOCKED_AUTH", text
    if any(
        token in lowered
        for token in (
            "connection",
            "network",
            "timeout",
            "timed out",
            "name resolution",
            "dns",
            "offline",
        )
    ):
        return "UPSTREAM_CHECK_BLOCKED_NETWORK", text
    return "MATERIALIZATION_INVALID", text


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_utc = utc_now()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with DOWNLOAD_LOG.open("w", encoding="utf-8") as log_handle:
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, log_handle)

        result: dict[str, Any] = {
            "schema_version": 1,
            "phase": "Phase 10.2 tokenizer upstream provenance",
            "start_utc": start_utc,
            "end_utc": None,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "starting_head": STARTING_HEAD,
            "regenerated_reaudit_commit": REAUDIT_COMMIT,
            "script_path": str(SCRIPT_PATH),
            "official_repo_id": REPO_ID,
            "requested_revision": REVISION,
            "resolved_revision": None,
            "cache_dir": str(CACHE_DIR),
            "cache_freshness": None,
            "allow_patterns": list(ALLOW_PATTERNS),
            "ignore_patterns": list(IGNORE_PATTERNS),
            "snapshot_path": None,
            "snapshot_realpath": None,
            "snapshot_files": [],
            "snapshot_checks": {},
            "download_metadata": {},
            "offline_tokenizer_validation": None,
            "digest_comparison": {
                "v3_malformed_value": V3_MALFORMED_VALUE,
                "v3_malformed_hex_length": len(V3_MALFORMED_VALUE),
                "old_local_expected_sha256": OLD_LOCAL_VALID_SHA256,
                "old_local_direct_sha256": None,
                "fresh_tokenizer_config_sha256": None,
                "fresh_equals_old_local_valid": None,
                "malformed_transcription_relationship": (
                    OLD_LOCAL_VALID_SHA256[:34]
                    + OLD_LOCAL_VALID_SHA256[36:]
                    == V3_MALFORMED_VALUE
                ),
            },
            "v3_remains_permanently_void": True,
            "v4_created": False,
            "execution_exclusions": {
                "gpu_used": False,
                "full_model_loaded": False,
                "inference_run": False,
                "predictions_generated": False,
                "scientific_endpoints_computed": False,
                "project_data_accessed": False,
                "test_accessed": False,
                "old_model_or_tokenizer_modified": False,
                "qwen3_started": False,
            },
            "error": None,
            "final_verdict": "PROVENANCE_RESOLUTION_BLOCKED",
        }

        rows: list[dict[str, Any]] = []
        try:
            print(f"UTC_START={start_utc}")
            print(f"HOSTNAME={result['hostname']}")
            print(f"PYTHON_VERSION={result['python_version']}")
            print(f"REPO_ID={REPO_ID}")
            print(f"REQUESTED_REVISION={REVISION}")
            print(f"CACHE_DIR={CACHE_DIR}")

            freshness = inspect_cache_freshness()
            result["cache_freshness"] = freshness
            print(f"CACHE_FRESHNESS={freshness}")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            old_direct = sha256_file(OLD_LOCAL_TOKENIZER_CONFIG)
            result["digest_comparison"]["old_local_direct_sha256"] = old_direct
            print(f"OLD_LOCAL_DIRECT_SHA256={old_direct}")
            if old_direct != OLD_LOCAL_VALID_SHA256:
                raise MaterializationInvalid(
                    "old local tokenizer_config.json changed before resolution"
                )

            import huggingface_hub
            import transformers
            from huggingface_hub import HfApi, snapshot_download
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            result["huggingface_hub_version"] = huggingface_hub.__version__
            result["transformers_version"] = transformers.__version__
            print(f"HUGGINGFACE_HUB_VERSION={huggingface_hub.__version__}")
            print(f"TRANSFORMERS_VERSION={transformers.__version__}")

            try:
                info = HfApi().model_info(
                    REPO_ID,
                    revision=REVISION,
                    files_metadata=True,
                )
                resolved = str(info.sha)
                result["resolved_revision"] = resolved
                remote_files = []
                for sibling in info.siblings or []:
                    name = sibling.rfilename
                    if matches_any(name, ALLOW_PATTERNS) and not matches_any(
                        name, IGNORE_PATTERNS
                    ):
                        remote_files.append(
                            {
                                "relative_path": name,
                                "size": getattr(sibling, "size", None),
                                "blob_id": getattr(sibling, "blob_id", None),
                                "lfs": json_safe(getattr(sibling, "lfs", None)),
                            }
                        )
                result["download_metadata"] = {
                    "api_repo_id": info.id,
                    "api_resolved_sha": resolved,
                    "selected_remote_files": remote_files,
                    "selected_remote_file_count": len(remote_files),
                }
                print(f"API_RESOLVED_REVISION={resolved}")
                print(f"SELECTED_REMOTE_FILE_COUNT={len(remote_files)}")
                if resolved != REVISION:
                    raise MaterializationInvalid(
                        f"API resolved unexpected revision: {resolved}"
                    )

                returned = snapshot_download(
                    repo_id=REPO_ID,
                    revision=REVISION,
                    cache_dir=str(CACHE_DIR),
                    allow_patterns=list(ALLOW_PATTERNS),
                    ignore_patterns=list(IGNORE_PATTERNS),
                )
            except MaterializationInvalid:
                raise
            except BaseException as exc:
                verdict, error = classify_upstream_exception(exc)
                result["final_verdict"] = verdict
                result["error"] = error
                raise

            snapshot = Path(returned)
            result["snapshot_path"] = str(snapshot.absolute())
            result["snapshot_realpath"] = str(snapshot.resolve(strict=True))
            print(f"RETURNED_SNAPSHOT_PATH={result['snapshot_path']}")
            print(f"SNAPSHOT_REALPATH={result['snapshot_realpath']}")

            if snapshot.name != REVISION:
                raise MaterializationInvalid(
                    f"snapshot directory is not exact revision: {snapshot.name}"
                )
            if not snapshot.resolve(strict=True).is_relative_to(
                CACHE_DIR.resolve(strict=True)
            ):
                raise MaterializationInvalid(
                    "snapshot resolves outside the dedicated cache"
                )

            rows, checks = inspect_snapshot(snapshot)
            result["snapshot_files"] = rows
            result["snapshot_checks"] = checks

            blocking_lists = (
                "incomplete_paths",
                "broken_symlinks",
                "zero_byte_snapshot_files",
                "missing_required_files",
                "unexpected_weight_files",
                "files_resolving_outside_cache",
                "files_resolving_to_old_local_model",
                "files_without_blob_relation",
                "invalid_sha256_rows",
            )
            failures = {
                key: checks[key] for key in blocking_lists if checks.get(key)
            }
            if failures:
                raise MaterializationInvalid(
                    "snapshot/blob provenance checks failed: "
                    + json.dumps(failures, sort_keys=True)
                )
            if not rows:
                raise MaterializationInvalid("snapshot contains no downloaded files")

            fresh = sha256_file(snapshot / "tokenizer_config.json")
            result["digest_comparison"][
                "fresh_tokenizer_config_sha256"
            ] = fresh
            result["digest_comparison"][
                "fresh_equals_old_local_valid"
            ] = fresh == OLD_LOCAL_VALID_SHA256
            print(f"FRESH_TOKENIZER_CONFIG_SHA256={fresh}")

            if fresh != OLD_LOCAL_VALID_SHA256:
                result["final_verdict"] = "BLOCKED_UPSTREAM_MISMATCH"
            else:
                validation = offline_tokenizer_validation(snapshot)
                result["offline_tokenizer_validation"] = validation
                print(
                    "TOKENIZER_OFFLINE_VALIDATION="
                    + ("PASS" if validation["deterministic"] else "FAIL")
                )
                result["final_verdict"] = "READY_FOR_V4_PROTOCOL"

        except CacheNotFresh as exc:
            result["error"] = sanitize_error(f"{type(exc).__name__}: {exc}")
            result["final_verdict"] = "PROVENANCE_RESOLUTION_BLOCKED"
        except MaterializationInvalid as exc:
            result["error"] = sanitize_error(f"{type(exc).__name__}: {exc}")
            result["final_verdict"] = "MATERIALIZATION_INVALID"
        except BaseException as exc:
            if result["error"] is None:
                result["error"] = sanitize_error(f"{type(exc).__name__}: {exc}")
            if result["final_verdict"] == "PROVENANCE_RESOLUTION_BLOCKED":
                result["final_verdict"] = "MATERIALIZATION_INVALID"
        finally:
            result["end_utc"] = utc_now()
            write_hash_csv(rows)
            write_resolution_json(result)
            write_resolution_md(result)
            print(f"UTC_END={result['end_utc']}")
            print(f"FINAL_VERDICT={result['final_verdict']}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
