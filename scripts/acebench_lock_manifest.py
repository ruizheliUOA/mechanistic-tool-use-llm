"""
acebench_lock_manifest.py — Phase 4 Step 8: freeze the readout, write READOUT_LOCK_MANIFEST.json.
==================================================================================================
Run ONLY after parser validation (Step 6) and the smoke gate (Step 7) have passed, and BEFORE the
full baseline. Hashes every readout-defining artifact; any later change to a hashed artifact is a
new protocol version requiring a fresh run.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "final/results/acebench_generation_readout"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    smoke = json.loads((RES / "smoke_test_report.json").read_text())
    assert smoke["checks"]["gate"]["PASS"], "smoke gate did not pass — cannot lock"
    files = {
        "protocol": "final/results/acebench_generation_readout/PHASE4_GENERATION_PROTOCOL.md",
        "parser": "scripts/acebench_parse_outputs.py",
        "generation_script": "scripts/acebench_generation_baseline.py",
        "dataset": ".cache/acebench_phase4/acebench_decision_all_with_time.jsonl",
        "official_prompt_en": ".cache/acebench_raw/official_prompt_en.py",
        "official_prompt_zh": ".cache/acebench_raw/official_prompt_zh.py",
        "model_config": ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct/config.json",
        "model_index": ".cache/modelscope/Qwen/Qwen2___5-7B-Instruct/model.safetensors.index.json",
    }
    manifest = {
        "locked_on": str(date.today()),
        "protocol_version": "1.0",
        "parser_version": "1.0",
        "model": "Qwen2.5-7B-Instruct bf16 (local shards; index total 15.231 GB)",
        "generation_params": {"do_sample": False, "num_beams": 1, "max_new_tokens": 256,
                              "batch_size": 1, "order": "ascending sample_id",
                              "max_prompt_tokens": 8192, "stop": "eos only"},
        "unknown_policy": "UNKNOWN is a first-class outcome; counted as error for accuracy; "
                          "excluded from channel-discovery transitions; never forced to a label",
        "split": "archived stratified 70/15/15 seed 42 (verified cell-exact vs archive)",
        "smoke_gate": smoke["checks"]["gate"],
        "env": smoke["env"],
        "sha256": {k: sha(ROOT / v) for k, v in files.items()},
        "files": files,
        "note": "No readout change is permitted after inspecting full-run results; any change is a "
                "new protocol version requiring a fresh run.",
    }
    json.dump(manifest, open(RES / "READOUT_LOCK_MANIFEST.json", "w"), indent=2)
    print(json.dumps({k: v[:16] for k, v in manifest["sha256"].items()}, indent=1))
    print("LOCKED")


if __name__ == "__main__":
    main()
