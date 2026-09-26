#!/usr/bin/env python3
"""Build the sealed four-column Phase 8 Llama baseline split index.

Infrastructure only. This program never imports model/data-science libraries,
never parses source payloads during indexing, and never computes scientific
quantities.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


ROOT = Path("/root/autodl-tmp/sakiko-followup")
EXPECTED_BRANCH = "exp/sakiko-followup-archive"

SOURCE_PATH = Path(
    "/root/autodl-tmp/phase8_cache/"
    "llama31_8b_w2c_baseline_details.jsonl"
)
SCHEMA_PATH = (
    ROOT / "final/results/phase10_3_split_index/SPLIT_INDEX_SCHEMA.md"
)
INDEX_PATH = (
    ROOT
    / "final/results/phase10_3_split_index/"
    "llama31_8b_w2c_baseline_details.split_index.csv"
)
SIDECAR_PATH = (
    ROOT
    / "final/results/phase10_3_split_index/"
    "llama31_8b_w2c_baseline_details.split_index.manifest.json"
)
LOG_PATH = (
    ROOT
    / "final/results/phase10_3_split_index/"
    "PHASE8_SPLIT_INDEX_BUILD.log"
)
BUILDER_PATH = ROOT / "scripts/build_phase8_baseline_split_index.py"
EXECUTION_MARKER_PATH = (
    ROOT
    / "final/results/phase10_3_split_index/"
    ".phase8_split_index.build_started"
)

SCHEMA_SHA256 = (
    "d6c7833b32c9135d571814de06b3e75bc9fa89d4911106eb9688e22e516a4ca3"
)
HEADER = ("row_index", "byte_offset", "byte_length", "split_label")
SPLIT_LABELS = ("train", "val", "test")
SPLIT_SOURCE_MODE = "EXTERNAL_ROW_INDEX_MANIFEST"
EXPECTED_TOTAL_ROWS = 3652
EXPECTED_COUNTS = {"train": 2556, "val": 548, "test": 548}
MANIFESTS = {
    "train": (
        ROOT / "final/results/splits/train_idx.json",
        "7b843b43080e946595eb48ffbad800b5b7223cc693ccc2c9c49da47c2393b1fc",
    ),
    "val": (
        ROOT / "final/results/splits/val_idx.json",
        "2d2f078a47ae7702b5ccceb23a86d9d4cf223d7b7cc0de4eb8090e809afa3610",
    ),
    "test": (
        ROOT / "final/results/splits/test_idx.json",
        "aa7dbd0750025fa81c2b5d14ef5be74b5a2ba0d8ec3b35fc476d2cf8a29ea1af",
    ),
}
TRAIN_VERIFY_K = 20
TRAIN_VERIFY_RULE = "floor(j * (n_train - 1) / 19), j=0,...,19"


class BuildFailure(Exception):
    """A sanitized structural failure with no source-derived text."""

    def __init__(
        self,
        category: str,
        *,
        row_index: int | None = None,
        byte_offset: int | None = None,
        reason: str,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.row_index = row_index
        self.byte_offset = byte_offset
        self.reason = reason

    def safe_line(self) -> str:
        row = "NA" if self.row_index is None else str(self.row_index)
        offset = "NA" if self.byte_offset is None else str(self.byte_offset)
        return (
            f"FAIL category={self.category} row_index={row} "
            f"byte_offset={offset} reason={self.reason}"
        )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(
    category: str,
    reason: str,
    *,
    row_index: int | None = None,
    byte_offset: int | None = None,
) -> None:
    raise BuildFailure(
        category,
        row_index=row_index,
        byte_offset=byte_offset,
        reason=reason,
    )


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_repository() -> tuple[str, str, str]:
    try:
        repository_root = git_value("rev-parse", "--show-toplevel")
        branch = git_value("branch", "--show-current")
        head = git_value("rev-parse", "HEAD")
        status_text = git_value("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError):
        fail("repository", "git_metadata_unavailable")
    if repository_root != str(ROOT):
        fail("repository", "root_mismatch")
    if branch != EXPECTED_BRANCH:
        fail("repository", "branch_mismatch")
    if status_text:
        fail("repository", "worktree_not_clean")
    return repository_root, branch, head


def verify_schema() -> str:
    if SCHEMA_PATH != (
        ROOT / "final/results/phase10_3_split_index/SPLIT_INDEX_SCHEMA.md"
    ):
        fail("schema", "path_mismatch")
    if not SCHEMA_PATH.is_file() or SCHEMA_PATH.is_symlink():
        fail("schema", "missing_or_nonregular")
    observed = sha256_file(SCHEMA_PATH)
    if observed != SCHEMA_SHA256:
        fail("schema", "sha256_mismatch")
    return observed


def load_and_verify_manifests() -> tuple[
    dict[int, str], dict[str, list[int]], list[dict[str, object]]
]:
    split_map: dict[int, str] = {}
    split_rows: dict[str, list[int]] = {}
    manifest_records: list[dict[str, object]] = []

    for label in SPLIT_LABELS:
        path, expected_sha = MANIFESTS[label]
        if not path.is_file() or path.is_symlink():
            fail("manifest", "missing_or_nonregular")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            fail("manifest", "sha256_mismatch")
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            fail("manifest", "invalid_json")
        if not isinstance(values, list):
            fail("manifest", "not_list")
        if any(type(value) is not int for value in values):
            fail("manifest", "non_integer_entry")
        if values != sorted(values):
            fail("manifest", "not_sorted")
        if len(values) != len(set(values)):
            fail("manifest", "duplicate_entry")
        if len(values) != EXPECTED_COUNTS[label]:
            fail("manifest", "count_mismatch")
        for row_index in values:
            if row_index < 0 or row_index >= EXPECTED_TOTAL_ROWS:
                fail("manifest", "out_of_range", row_index=row_index)
            if row_index in split_map:
                fail("manifest", "overlap", row_index=row_index)
            split_map[row_index] = label
        split_rows[label] = values
        manifest_records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": observed_sha,
                "split_label": label,
                "row_count": len(values),
            }
        )

    if set(split_map) != set(range(EXPECTED_TOTAL_ROWS)):
        fail("manifest", "incomplete_union")
    return split_map, split_rows, manifest_records


def verify_output_namespace() -> None:
    for path in (INDEX_PATH, SIDECAR_PATH, LOG_PATH, EXECUTION_MARKER_PATH):
        if path.exists() or path.is_symlink():
            fail("output_namespace", "prior_output_or_state")
    output_dir = INDEX_PATH.parent
    if not output_dir.is_dir():
        fail("output_namespace", "directory_missing")
    for path in output_dir.iterdir():
        if path.name.startswith(".phase8_split_index.tmp."):
            fail("output_namespace", "prior_temporary_state")


def source_stat_record(source_stat: os.stat_result) -> dict[str, object]:
    return {
        "inode": source_stat.st_ino,
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "mtime_utc": dt.datetime.fromtimestamp(
            source_stat.st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "mode": stat.filemode(source_stat.st_mode),
    }


def validate_source_before_open() -> tuple[Path, os.stat_result]:
    if str(SOURCE_PATH) != (
        "/root/autodl-tmp/phase8_cache/"
        "llama31_8b_w2c_baseline_details.jsonl"
    ):
        fail("source", "path_mismatch")
    if SOURCE_PATH.is_symlink():
        fail("source", "symlink_forbidden")
    try:
        source_stat = SOURCE_PATH.stat()
        source_realpath = SOURCE_PATH.resolve(strict=True)
    except OSError:
        fail("source", "missing_or_unstatable")
    if not stat.S_ISREG(source_stat.st_mode):
        fail("source", "not_regular")
    if source_realpath != SOURCE_PATH:
        fail("source", "realpath_mismatch")
    return source_realpath, source_stat


def create_execution_marker(start_utc: str) -> None:
    try:
        descriptor = os.open(
            EXECUTION_MARKER_PATH,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"execution_count=1\nstart_utc={start_utc}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        fail("execution", "prior_execution_state")
    except OSError:
        fail("execution", "marker_creation_failed")


def select_train_verification_rows(train_rows: list[int]) -> list[int]:
    if len(train_rows) < TRAIN_VERIFY_K:
        fail("train_seek", "fewer_than_20_train_rows")
    selected = [
        train_rows[math.floor(j * (len(train_rows) - 1) / 19)]
        for j in range(TRAIN_VERIFY_K)
    ]
    if len(selected) != TRAIN_VERIFY_K or len(set(selected)) != TRAIN_VERIFY_K:
        fail("train_seek", "selection_not_unique")
    return selected


def build_index_single_pass(
    index_temp: Path,
    split_map: dict[int, str],
    selected_train_rows: list[int],
    expected_source_size: int,
) -> tuple[str, int, dict[str, int], dict[int, tuple[int, int]]]:
    source_digest = hashlib.sha256()
    counts = {label: 0 for label in SPLIT_LABELS}
    selected_set = set(selected_train_rows)
    train_offsets: dict[int, tuple[int, int]] = {}
    row_count = 0
    byte_offset = 0

    try:
        source_handle = SOURCE_PATH.open("rb")
    except OSError:
        fail("source", "read_only_binary_open_failed")

    try:
        with source_handle, index_temp.open(
            "x", encoding="utf-8", newline=""
        ) as index_handle:
            writer = csv.writer(index_handle, lineterminator="\n")
            writer.writerow(HEADER)
            for row_index, line_buffer in enumerate(source_handle):
                if line_buffer.endswith(b"\r\n"):
                    fail(
                        "line_ending",
                        "crlf_forbidden",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                if not line_buffer.endswith(b"\n"):
                    fail(
                        "line_ending",
                        "missing_trailing_lf",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                byte_length = len(line_buffer) - 1
                if byte_length <= 0:
                    fail(
                        "line_structure",
                        "empty_record",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                if row_index not in split_map:
                    fail(
                        "manifest",
                        "row_without_membership",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                split_label = split_map[row_index]
                source_digest.update(line_buffer)
                writer.writerow(
                    (row_index, byte_offset, byte_length, split_label)
                )
                counts[split_label] += 1
                if row_index in selected_set:
                    if split_label != "train":
                        fail(
                            "train_seek",
                            "selected_row_not_train",
                            row_index=row_index,
                            byte_offset=byte_offset,
                        )
                    train_offsets[row_index] = (byte_offset, byte_length)
                byte_offset += len(line_buffer)
                row_count += 1
                line_buffer = None
            index_handle.flush()
            os.fsync(index_handle.fileno())
    except FileExistsError:
        fail("output_namespace", "temporary_index_exists")
    except OSError:
        fail("index_build", "io_failure")

    if byte_offset != expected_source_size:
        fail("source", "streamed_size_mismatch", byte_offset=byte_offset)
    if row_count != EXPECTED_TOTAL_ROWS:
        fail("row_count", "independent_count_mismatch", row_index=row_count)
    if counts != EXPECTED_COUNTS:
        fail("manifest", "per_split_count_mismatch")
    if set(train_offsets) != selected_set:
        fail("train_seek", "selected_offsets_incomplete")
    return source_digest.hexdigest(), row_count, counts, train_offsets


def verify_index_structure(
    index_temp: Path,
    split_map: dict[int, str],
    source_file_size: int,
) -> dict[str, object]:
    expected_row = 0
    previous_offset: int | None = None
    previous_length: int | None = None
    final_extent = 0
    adjacency_passed = 0
    counts = {label: 0 for label in SPLIT_LABELS}

    try:
        with index_temp.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader))
            except StopIteration:
                fail("index_structure", "missing_header")
            if header != HEADER:
                fail("index_structure", "header_mismatch")
            for fields in reader:
                if len(fields) != 4:
                    fail(
                        "index_structure",
                        "column_count_mismatch",
                        row_index=expected_row,
                    )
                try:
                    row_index = int(fields[0])
                    byte_offset = int(fields[1])
                    byte_length = int(fields[2])
                except ValueError:
                    fail(
                        "index_structure",
                        "non_integer_coordinate",
                        row_index=expected_row,
                    )
                split_label = fields[3]
                if row_index != expected_row:
                    fail(
                        "index_structure",
                        "row_index_discontinuity",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                if byte_offset < 0 or byte_length <= 0:
                    fail(
                        "index_structure",
                        "invalid_coordinate",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                if split_label not in SPLIT_LABELS:
                    fail(
                        "index_structure",
                        "unexpected_split_label",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                if split_map.get(row_index) != split_label:
                    fail(
                        "index_structure",
                        "manifest_label_mismatch",
                        row_index=row_index,
                        byte_offset=byte_offset,
                    )
                if previous_offset is not None and previous_length is not None:
                    if previous_offset + previous_length + 1 != byte_offset:
                        fail(
                            "index_structure",
                            "adjacency_equation_failed",
                            row_index=row_index,
                            byte_offset=byte_offset,
                        )
                    if byte_offset <= previous_offset:
                        fail(
                            "index_structure",
                            "offset_not_increasing",
                            row_index=row_index,
                            byte_offset=byte_offset,
                        )
                    adjacency_passed += 1
                counts[split_label] += 1
                final_extent = byte_offset + byte_length + 1
                previous_offset = byte_offset
                previous_length = byte_length
                expected_row += 1
    except OSError:
        fail("index_structure", "verification_io_failure")

    if expected_row != EXPECTED_TOTAL_ROWS:
        fail("index_structure", "row_count_mismatch", row_index=expected_row)
    if final_extent != source_file_size:
        fail(
            "index_structure",
            "final_file_size_equation_failed",
            row_index=expected_row - 1,
            byte_offset=final_extent,
        )
    if counts != EXPECTED_COUNTS:
        fail("index_structure", "aggregate_count_mismatch")
    if adjacency_passed != EXPECTED_TOTAL_ROWS - 1:
        fail("index_structure", "adjacency_count_mismatch")

    return {
        "status": "PASS",
        "csv_column_count": 4,
        "header_exact": True,
        "row_index_contiguous": True,
        "offsets_strictly_increasing": True,
        "adjacency_equations_passed": adjacency_passed,
        "adjacency_equations_expected": EXPECTED_TOTAL_ROWS - 1,
        "final_extent_bytes": final_extent,
        "source_file_size_bytes": source_file_size,
        "final_file_size_equation": True,
        "manifest_exact_coverage": True,
        "independent_row_count_match": True,
    }


def verify_selected_train_rows(
    selected_train_rows: list[int],
    train_offsets: dict[int, tuple[int, int]],
) -> dict[str, object]:
    passed = 0
    try:
        source_handle = SOURCE_PATH.open("rb")
    except OSError:
        fail("train_seek", "read_only_binary_open_failed")

    with source_handle:
        for row_index in selected_train_rows:
            byte_offset, byte_length = train_offsets[row_index]
            try:
                source_handle.seek(byte_offset)
                payload_buffer = source_handle.read(byte_length)
            except OSError:
                fail(
                    "train_seek",
                    "seek_or_read_failed",
                    row_index=row_index,
                    byte_offset=byte_offset,
                )
            if len(payload_buffer) != byte_length:
                fail(
                    "train_seek",
                    "short_read",
                    row_index=row_index,
                    byte_offset=byte_offset,
                )
            try:
                parsed_object = json.loads(payload_buffer.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                fail(
                    "train_seek",
                    "invalid_json_object",
                    row_index=row_index,
                    byte_offset=byte_offset,
                )
            if not isinstance(parsed_object, dict):
                fail(
                    "train_seek",
                    "top_level_not_object",
                    row_index=row_index,
                    byte_offset=byte_offset,
                )
            passed += 1
            parsed_object = None
            payload_buffer = None

    if passed != TRAIN_VERIFY_K:
        fail("train_seek", "verification_count_mismatch")
    return {
        "status": "PASS",
        "selection_rule": TRAIN_VERIFY_RULE,
        "attempted": TRAIN_VERIFY_K,
        "passed": passed,
        "validation_seek_reads": 0,
        "test_seek_reads": 0,
        "payload_or_id_retained": False,
    }


def atomic_write_text(path: Path, text: str, temp_path: Path) -> None:
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        fail("output_namespace", "temporary_metadata_exists")
    except OSError:
        fail("output_write", "metadata_io_failure")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the sealed Phase 8 baseline split index."
    )
    parser.add_argument("--build", action="store_true", help="execute once")
    arguments = parser.parse_args()
    if not arguments.build:
        parser.error("exactly one mode is required: --build")

    started_utc = utc_now()
    started_monotonic = time.monotonic()
    log_lines = [
        f"START utc={started_utc}",
        f"MODE {SPLIT_SOURCE_MODE}",
    ]

    process_id = os.getpid()
    index_temp = INDEX_PATH.parent / (
        f".phase8_split_index.tmp.{process_id}.index"
    )
    sidecar_temp = INDEX_PATH.parent / (
        f".phase8_split_index.tmp.{process_id}.sidecar"
    )
    log_temp = INDEX_PATH.parent / (
        f".phase8_split_index.tmp.{process_id}.log"
    )

    try:
        schema_sha = verify_schema()
        log_lines.append("CHECK schema_sha256 PASS")
        split_map, split_rows, manifest_records = load_and_verify_manifests()
        log_lines.append("CHECK external_manifest_coverage PASS")
        repository_root, branch, git_head = verify_repository()
        log_lines.append("CHECK repository_state PASS")
        verify_output_namespace()
        log_lines.append("CHECK output_namespace_empty PASS")
        builder_sha = sha256_file(BUILDER_PATH)

        source_realpath, source_before_stat = validate_source_before_open()
        source_before = source_stat_record(source_before_stat)
        log_lines.append("CHECK source_preopen_identity PASS")
        create_execution_marker(started_utc)
        log_lines.append("CHECK execution_count_1 PASS")

        selected_train_rows = select_train_verification_rows(
            split_rows["train"]
        )
        (
            source_sha,
            row_count,
            per_split_counts,
            train_offsets,
        ) = build_index_single_pass(
            index_temp,
            split_map,
            selected_train_rows,
            source_before_stat.st_size,
        )
        log_lines.append("CHECK single_sequential_index_pass PASS")

        source_after_pass_stat = SOURCE_PATH.stat()
        if (
            source_after_pass_stat.st_ino != source_before_stat.st_ino
            or source_after_pass_stat.st_size != source_before_stat.st_size
            or source_after_pass_stat.st_mtime_ns
            != source_before_stat.st_mtime_ns
            or source_after_pass_stat.st_mode != source_before_stat.st_mode
        ):
            fail("source", "identity_changed_after_stream")

        structural_result = verify_index_structure(
            index_temp, split_map, source_before_stat.st_size
        )
        log_lines.append("CHECK structural_verification PASS")
        train_seek_result = verify_selected_train_rows(
            selected_train_rows, train_offsets
        )
        log_lines.append("CHECK train_seek_verification_20_of_20 PASS")

        source_after_stat = SOURCE_PATH.stat()
        source_after = source_stat_record(source_after_stat)
        if (
            source_after_stat.st_ino != source_before_stat.st_ino
            or source_after_stat.st_size != source_before_stat.st_size
            or source_after_stat.st_mtime_ns != source_before_stat.st_mtime_ns
            or source_after_stat.st_mode != source_before_stat.st_mode
        ):
            fail("source", "identity_changed_after_seek")
        log_lines.append("CHECK source_postrun_identity PASS")

        index_sha = sha256_file(index_temp)
        completed_utc = utc_now()
        runtime_seconds = round(time.monotonic() - started_monotonic, 6)
        sidecar = {
            "source_path": str(SOURCE_PATH),
            "source_realpath": str(source_realpath),
            "source_file_size_bytes": source_before_stat.st_size,
            "source_sha256": source_sha,
            "source_inode": source_before_stat.st_ino,
            "source_mode": source_before["mode"],
            "source_mtime_before": source_before["mtime_utc"],
            "source_mtime_after": source_after["mtime_utc"],
            "total_row_count": row_count,
            "per_split_row_counts": per_split_counts,
            "split_label_vocabulary": list(SPLIT_LABELS),
            "selected_split_source_mode": SPLIT_SOURCE_MODE,
            "source_manifests": manifest_records,
            "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
            "schema_sha256": schema_sha,
            "builder_path": str(BUILDER_PATH.relative_to(ROOT)),
            "builder_sha256": builder_sha,
            "index_path": str(INDEX_PATH.relative_to(ROOT)),
            "index_sha256": index_sha,
            "utc_start_time": started_utc,
            "utc_completion_time": completed_utc,
            "runtime_seconds": runtime_seconds,
            "git_head": git_head,
            "repository_root": repository_root,
            "branch": branch,
            "execution_count": 1,
            "structural_verification": structural_result,
            "train_seek_verification": train_seek_result,
        }
        sidecar_text = (
            json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        )
        log_lines.extend(
            [
                "CHECK payload_persistence_zero PASS",
                "CHECK validation_test_seek_reads_zero PASS",
                f"END utc={completed_utc} status=PASS",
            ]
        )
        log_text = "\n".join(log_lines) + "\n"
        atomic_write_text(SIDECAR_PATH, sidecar_text, sidecar_temp)
        atomic_write_text(LOG_PATH, log_text, log_temp)

        os.replace(index_temp, INDEX_PATH)
        os.replace(sidecar_temp, SIDECAR_PATH)
        os.replace(log_temp, LOG_PATH)
        EXECUTION_MARKER_PATH.unlink()
        return 0
    except BuildFailure as error:
        try:
            with log_temp.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(error.safe_line() + "\n")
        except OSError:
            pass
        sys.stderr.write(error.safe_line() + "\n")
        return 1
    except Exception:
        safe_line = (
            "FAIL category=unexpected row_index=NA byte_offset=NA "
            "reason=unexpected_exception"
        )
        try:
            with log_temp.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(safe_line + "\n")
        except OSError:
            pass
        sys.stderr.write(safe_line + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
