#!/usr/bin/env python3
"""Build a payload-blind byte-offset index for Qwen train/dev baseline rows.

The source JSONL contains all project rows. This utility never parses source
payloads while indexing. It reads only line boundaries, consults only the
committed train and validation row-index manifests, and emits offsets only for
those two allowed populations. Rows outside the allow-list are neither parsed
nor represented in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/sakiko-followup")
SOURCE = ROOT / "data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl"
TRAIN_MANIFEST = ROOT / "final/results/splits/train_idx.json"
DEV_MANIFEST = ROOT / "final/results/splits/val_idx.json"
OUT_DIR = ROOT / "final/results/qwen25_stage_b_preparation"
INDEX = OUT_DIR / "qwen25_baseline_train_dev_byte_index.csv"
MANIFEST = OUT_DIR / "qwen25_baseline_train_dev_byte_index.manifest.json"

EXPECTED_SOURCE_ROWS = 3652
EXPECTED = {
    "train": {
        "count": 2556,
        "sha256": "7b843b43080e946595eb48ffbad800b5b7223cc693ccc2c9c49da47c2393b1fc",
    },
    "dev": {
        "count": 548,
        "sha256": "2d2f078a47ae7702b5ccceb23a86d9d4cf223d7b7cc0de4eb8090e809afa3610",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_allowlist(path: Path, split: str) -> list[int]:
    observed = sha256_file(path)
    if observed != EXPECTED[split]["sha256"]:
        raise RuntimeError(f"{split} manifest SHA256 mismatch")
    values = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(values, list)
        or any(type(value) is not int for value in values)
        or values != sorted(values)
        or len(values) != len(set(values))
        or len(values) != EXPECTED[split]["count"]
    ):
        raise RuntimeError(f"{split} manifest structure mismatch")
    if any(value < 0 or value >= EXPECTED_SOURCE_ROWS for value in values):
        raise RuntimeError(f"{split} manifest out-of-range index")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SOURCE.is_file() or SOURCE.is_symlink():
        raise RuntimeError("baseline source missing, nonregular, or symlinked")
    train = load_allowlist(TRAIN_MANIFEST, "train")
    dev = load_allowlist(DEV_MANIFEST, "dev")
    if set(train) & set(dev):
        raise RuntimeError("train/dev overlap")
    allowed = {row: "train" for row in train}
    allowed.update({row: "dev" for row in dev})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (INDEX.exists() or MANIFEST.exists()):
        raise RuntimeError("output exists; pass --overwrite only for deterministic rebuild")

    source_digest = hashlib.sha256()
    output_rows: list[tuple[int, int, int, str]] = []
    byte_offset = 0
    row_count = 0
    with SOURCE.open("rb") as source:
        for row_index, raw_line in enumerate(source):
            # Structural scan only: never decode or parse non-allowlisted payloads.
            if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
                raise RuntimeError(f"invalid line ending at row {row_index}")
            source_digest.update(raw_line)
            byte_length = len(raw_line) - 1
            if byte_length <= 0:
                raise RuntimeError(f"empty row at {row_index}")
            split = allowed.get(row_index)
            if split is not None:
                output_rows.append((row_index, byte_offset, byte_length, split))
            byte_offset += len(raw_line)
            row_count += 1
            raw_line = b""

    if row_count != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"source row count {row_count} != {EXPECTED_SOURCE_ROWS}")
    counts = {
        split: sum(1 for row in output_rows if row[3] == split)
        for split in ("train", "dev")
    }
    if counts != {key: value["count"] for key, value in EXPECTED.items()}:
        raise RuntimeError("indexed allow-list count mismatch")

    with INDEX.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("row_index", "byte_offset", "byte_length", "split_label"))
        writer.writerows(output_rows)

    manifest = {
        "schema_version": 1,
        "purpose": "PAYLOAD_BLIND_TRAIN_DEV_ONLY_BYTE_OFFSET_INDEX",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_digest.hexdigest(),
        "source_size_bytes": SOURCE.stat().st_size,
        "source_rows_structurally_scanned": row_count,
        "source_payloads_parsed_during_indexing": 0,
        "emitted_splits": counts,
        "emitted_rows": len(output_rows),
        "non_allowlisted_rows_emitted": 0,
        "non_allowlisted_payloads_decoded_or_parsed": 0,
        "test_manifest_opened": False,
        "test_index_opened": False,
        "train_manifest": {
            "path": str(TRAIN_MANIFEST.relative_to(ROOT)),
            "sha256": EXPECTED["train"]["sha256"],
        },
        "dev_manifest": {
            "path": str(DEV_MANIFEST.relative_to(ROOT)),
            "sha256": EXPECTED["dev"]["sha256"],
        },
        "index_path": str(INDEX.relative_to(ROOT)),
        "index_sha256": sha256_file(INDEX),
        "builder_path": "scripts/qwen25_stage_b_train_dev_index.py",
        "access_contract": (
            "Consumers must seek directly to emitted offsets and parse only emitted "
            "train/dev rows; sequential JSONL parsing is forbidden."
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
