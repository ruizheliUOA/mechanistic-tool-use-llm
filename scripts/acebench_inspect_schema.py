"""
acebench_inspect_schema.py — Phase 4 Step 1/2: verify raw ACEBench files and regenerate the
=============================================================================================
decision dataset WITHOUT touching the archived (LFS-pointer) artifacts.

- Verifies the 12 raw single-turn files (6 EN + 6 ZH mirrors): presence, line counts, schema,
  sha256 (recorded in the manifest).
- Regenerates the 800-row decision dataset with the EXACT archived conversion + split logic
  (imports scripts/convert_acebench_decision.py helpers; same seed 42), extended by the raw
  `time` field (needed by the official ACEBench prompt template; the archived converter
  dropped it — rows are otherwise identical and the split logic never sees `time`).
- Verifies the regenerated split against the archived acebench_decision_split_info.json
  (all 24 per-(mode,lang,split) cells must match exactly) and the archived first-200
  sample-order gold check (candidate-sensitivity file: sample 0000..0199 all gold tool_call).
- Writes the regenerated dataset to gitignored .cache/acebench_phase4/ and a small manifest to
  final/results/acebench_generation_readout/.

CPU-only. No model, no intervention. Usage: python scripts/acebench_inspect_schema.py
"""
from __future__ import annotations
import hashlib, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import convert_acebench_decision as C  # archived converter (FILE2MODE, SPLIT_FRACS, load_lang)

RAW = ROOT / ".cache" / "acebench_raw"
OUTD = ROOT / ".cache" / "acebench_phase4"          # gitignored dataset location
RES = ROOT / "final" / "results" / "acebench_generation_readout"
ARCH_SPLIT = ROOT / "data" / "processed" / "acebench_decision" / "acebench_decision_split_info.json"
ARCH_SENS = (ROOT / "final" / "results" / "cross_dataset_channel_discovery" /
             "new_benchmark_pilot" / "acebench_pilot_candidate_sensitivity.json")
SEED = 42
EXPECT_LINES = {  # per language
    "data_normal_single_turn_single_function": 100,
    "data_normal_single_turn_parallel_function": 100,
    "data_normal_similar_api": 50,
    "data_special_incomplete": 50,
    "data_special_error_param": 50,
    "data_special_irrelevant": 50,
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    # ---- 1. raw-file verification ----
    manifest = {"raw_dir": ".cache/acebench_raw (gitignored, local only)",
                "source": "github.com/ACEBench/ACEBench data_all/data_{en,zh} (MIT, arXiv:2501.12851)",
                "files": {}}
    schema_issues = []
    time_by_key = {}
    for lang in ("en", "zh"):
        sub = RAW if lang == "en" else RAW / "zh"
        for stem, n_expect in EXPECT_LINES.items():
            f = sub / f"{stem}.json"
            assert f.exists(), f"missing raw file {f}"
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            if len(rows) != n_expect:
                schema_issues.append(f"{lang}/{stem}: {len(rows)} rows != expected {n_expect}")
            for r in rows:
                if sorted(r.keys()) != ["function", "id", "question", "time"]:
                    schema_issues.append(f"{lang}/{stem}/{r.get('id')}: keys {sorted(r.keys())}")
                if not isinstance(r["function"], list) or not r["function"]:
                    schema_issues.append(f"{lang}/{stem}/{r['id']}: empty function list")
                time_by_key[(lang, r["id"])] = r.get("time", "")
            manifest["files"][f"{lang}/{stem}.json"] = {"n": len(rows), "sha256": sha256(f)}

    # ---- 2. regenerate decision rows with the ARCHIVED logic (identical code path) ----
    rows = C.load_lang(RAW, "en") + C.load_lang(RAW, "zh")
    rng = random.Random(SEED)
    by_stratum = {}
    for r in rows:
        by_stratum.setdefault((r["gold_mode"], r["lang"]), []).append(r)
    for stratum, rs in sorted(by_stratum.items()):
        rs.sort(key=lambda r: r["source_id"])
        rng.shuffle(rs)
        n = len(rs)
        n_tr = round(n * C.SPLIT_FRACS["train"]); n_va = round(n * C.SPLIT_FRACS["val"])
        for i, r in enumerate(rs):
            r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    rows.sort(key=lambda r: (r["lang"], r["source_id"]))
    for i, r in enumerate(rows):
        r["sample_id"] = f"acebench_decision_{i:04d}"
        r["time"] = time_by_key[(r["lang"], r["source_id"])]   # phase-4 extension (official prompt slot)

    # ---- 3. verify against archived split info + order check ----
    arch = json.loads(ARCH_SPLIT.read_text())
    dist = Counter((r["gold_mode"], r["lang"], r["split"]) for r in rows)
    regen_cells = {f"{m}|{l}|{s}": c for (m, l, s), c in sorted(dist.items())}
    cells_ok = regen_cells == arch["per_mode_lang_split"]
    gold_dist_ok = dict(Counter(r["gold_mode"] for r in rows)) == arch["gold_mode_distribution"]
    sens = json.loads(ARCH_SENS.read_text())
    id2gold = {r["sample_id"]: r["gold_mode"] for r in rows}
    order_ok = all(id2gold.get(s["sample_id"]) == s["gold"] for s in sens)
    print(f"cells_ok={cells_ok} gold_dist_ok={gold_dist_ok} order_ok(archived first-200)={order_ok} "
          f"schema_issues={len(schema_issues)}")
    if not (cells_ok and gold_dist_ok and order_ok and not schema_issues):
        for s in schema_issues[:10]:
            print("ISSUE:", s)
        print("VERIFICATION FAILED — do not proceed"); sys.exit(2)

    # ---- 4. write regenerated dataset (gitignored) + manifest (committable) ----
    out = OUTD / "acebench_decision_all_with_time.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest.update({
        "n_total": len(rows), "seed": SEED,
        "regen_dataset": ".cache/acebench_phase4/acebench_decision_all_with_time.jsonl (gitignored)",
        "regen_dataset_sha256": sha256(out),
        "split_verification": {"per_cell_match_vs_archived": cells_ok,
                               "gold_dist_match": gold_dist_ok,
                               "first200_order_match_vs_archived_pilot": order_ok},
        "schema_issues": schema_issues,
        "row_schema": ["sample_id", "source_id", "lang", "source_file", "gold_mode",
                       "question", "functions", "time", "split"],
        "difference_vs_archived_rows": "adds the raw 'time' field (official prompt slot); "
                                       "all other fields and the split are byte-logic identical",
    })
    json.dump(manifest, open(RES / "dataset_manifest.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote {out} ({len(rows)} rows) and {RES/'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
