"""
convert_acebench_decision.py — build the ACEBench single-turn DECISION-MODE dataset (CPU-only).
================================================================================================
Converts the six single-turn ACEBench files (EN + ZH mirrors) into per-example decision-mode rows
with a deterministic stratified split. The gold decision mode is NATIVE — it is the ACEBench file
category, no relabeling:

  normal_single_turn_single_function / normal_single_turn_parallel_function / normal_similar_api
      -> tool_call         (the correct behaviour is to call the provided function(s))
  special_incomplete   -> ask_user          (a required parameter is missing -> ask the user)
  special_error_param  -> flag_param_error  (a parameter violates the schema constraint -> point it out)
  special_irrelevant   -> cannot_comply     (no provided function can do this -> say so)

Provenance: ACEBench (MIT license), github.com/ACEBench/ACEBench, paper arXiv:2501.12851.
Raw files are read from a local download directory (NOT committed); this script writes only the
converted decision rows to data/processed/acebench_decision/ (jsonl -> git-lfs by .gitattributes).

Label-leakage control: rows carry gold_mode for scoring, but the evaluation prompt built by
eval_acebench_decision_baseline.py contains ONLY the question + function schemas + neutral candidates.

Usage:
  python scripts/convert_acebench_decision.py --raw-dir <dir with data_en/ zh/ files> [--seed 42]
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "acebench_decision"

FILE2MODE = {
    "data_normal_single_turn_single_function": "tool_call",
    "data_normal_single_turn_parallel_function": "tool_call",
    "data_normal_similar_api": "tool_call",
    "data_special_incomplete": "ask_user",
    "data_special_error_param": "flag_param_error",
    "data_special_irrelevant": "cannot_comply",
}
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_lang(raw_dir: Path, lang: str):
    rows = []
    sub = raw_dir if lang == "en" else raw_dir / "zh"
    for stem, mode in FILE2MODE.items():
        f = sub / f"{stem}.json"
        if not f.exists():
            raise FileNotFoundError(f)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            q = r["question"]
            if isinstance(q, str) and q.startswith("user: "):
                q = q[len("user: "):]
            rows.append({
                "source_id": r["id"], "lang": lang, "source_file": f"{stem}.json",
                "gold_mode": mode, "question": q.strip(), "functions": r["function"],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_lang(Path(args.raw_dir), "en") + load_lang(Path(args.raw_dir), "zh")
    # deterministic stratified split by (gold_mode, lang)
    rng = random.Random(args.seed)
    by_stratum = {}
    for r in rows:
        by_stratum.setdefault((r["gold_mode"], r["lang"]), []).append(r)
    for stratum, rs in sorted(by_stratum.items()):
        rs.sort(key=lambda r: r["source_id"])
        rng.shuffle(rs)
        n = len(rs)
        n_tr = round(n * SPLIT_FRACS["train"]); n_va = round(n * SPLIT_FRACS["val"])
        for i, r in enumerate(rs):
            r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    rows.sort(key=lambda r: (r["lang"], r["source_id"]))
    for i, r in enumerate(rows):
        r["sample_id"] = f"acebench_decision_{i:04d}"

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "acebench_decision_all.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("train", "val", "test"):
        with open(OUT / f"acebench_decision_{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == s:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter((r["gold_mode"], r["lang"], r["split"]) for r in rows)
    summary = {
        "n_total": len(rows), "seed": args.seed, "split_fracs": SPLIT_FRACS,
        "gold_mode_distribution": dict(Counter(r["gold_mode"] for r in rows)),
        "per_mode_lang_split": {f"{m}|{l}|{s}": c for (m, l, s), c in sorted(dist.items())},
        "provenance": {"source": "github.com/ACEBench/ACEBench data_all/data_{en,zh}",
                       "license": "MIT", "paper": "arXiv:2501.12851",
                       "native_label_rule": FILE2MODE},
    }
    json.dump(summary, open(OUT / "acebench_decision_split_info.json", "w"), indent=2)
    print(json.dumps(summary["gold_mode_distribution"], indent=1))
    print("splits:", dict(Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
