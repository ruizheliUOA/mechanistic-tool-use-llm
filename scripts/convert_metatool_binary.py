"""
convert_metatool_binary.py  –  Task1.json → SAKIKO-compatible JSONL
=====================================================================
Converts MetaTool Task1.json (binary tool-need classification) into
SAKIKO-compatible JSONL records, then produces stratified 70/15/15 splits.

Usage:
  python scripts/convert_metatool_binary.py \\
      --task1_json  /path/to/Task1.json \\
      --out_dir     data/processed/metatool_binary

Label mapping (Task1 label → SAKIKO gold_response_mode):
  positive  →  tool_call
  negative  →  no_tool
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def load_task1(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    return data


def convert_item(item: dict, idx: int) -> dict:
    label = item["label"]          # "positive" or "negative"
    query = item["query"]
    thought_prompt = item.get("thought_prompt", "")
    tool_name = item.get("tool", None)   # string or None

    gold_tool_needed = (label == "positive")
    gold_response_mode = "tool_call" if gold_tool_needed else "no_tool"

    # Minimal tool descriptor (only for positive items that have a tool name)
    tools = []
    if tool_name:
        tools = [{"name": str(tool_name)}]

    return {
        "sample_id":          f"metatool_binary_{idx:04d}",
        "dataset":            "metatool_binary",
        "query":              query,
        "tools":              tools,
        "gold_tool_needed":   gold_tool_needed,
        "gold_response_mode": gold_response_mode,
        "label_granularity":  "binary",
        "label_source":       "metatool_task1",
        "manual_subtype":     None,
        "original_id":        idx,
        "original_label":     label,
        "thought_prompt":     thought_prompt,
    }


def stratified_split(records: list[dict], train_frac=0.70, val_frac=0.15, seed=42
                     ) -> tuple[list, list, list]:
    """Deterministic stratified 70/15/15 split by gold_response_mode."""
    rng = random.Random(seed)

    by_label: dict[str, list] = {}
    for r in records:
        lbl = r["gold_response_mode"]
        by_label.setdefault(lbl, []).append(r)

    train, val, test = [], [], []
    for lbl, items in by_label.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_frac)
        n_val   = round(n * val_frac)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    # Final shuffle of each split (keeps classes interleaved)
    for split in [train, val, test]:
        rng.shuffle(split)

    return train, val, test


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1_json", required=True, help="Path to Task1.json")
    parser.add_argument("--out_dir", default="data/processed/metatool_binary",
                        help="Output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_task1(args.task1_json)
    print(f"Loaded {len(raw):,} items from {args.task1_json}")
    print(f"Label distribution: {Counter(r['label'] for r in raw)}")

    records = [convert_item(item, idx) for idx, item in enumerate(raw)]

    label_dist = Counter(r["gold_response_mode"] for r in records)
    print(f"gold_response_mode distribution: {dict(label_dist)}")

    out_dir = Path(args.out_dir)

    # All
    write_jsonl(records, out_dir / "metatool_binary_all.jsonl")

    # Splits
    train, val, test = stratified_split(records, seed=args.seed)
    write_jsonl(train, out_dir / "metatool_binary_train.jsonl")
    write_jsonl(val,   out_dir / "metatool_binary_val.jsonl")
    write_jsonl(test,  out_dir / "metatool_binary_test.jsonl")

    # Quick validation
    print("\n=== Split summary ===")
    for name, split in [("all", records), ("train", train), ("val", val), ("test", test)]:
        dist = Counter(r["gold_response_mode"] for r in split)
        print(f"  {name:6s}: n={len(split):4d}  {dict(dist)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
