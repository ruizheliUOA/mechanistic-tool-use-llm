"""
phase7_gate_v2_freeze.py — freeze Actionability Gate v2 (hash manifest).
========================================================================
Computes SHA256 of every artifact that defines the frozen prospective specification:
the gate spec, the prospective protocol, and the exact code that implements rho
normalization, random-direction construction, validation-specificity measurement,
layer-grid generation, channel discovery, and the decision logic.

After this manifest is written, the specification is immutable for the prospective phase.
CPU-only; reads no data; touches no test artifact.
"""
from __future__ import annotations
import hashlib, json, sys, subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

ROOT = P.ROOT

# artifacts whose content defines the frozen gate
ARTIFACTS = {
    "gate_specification": "final/results/actionability_gate_development/ACTIONABILITY_GATE_V2_SPEC.md",
    "prospective_protocol_template": "final/results/actionability_gate_development/PROSPECTIVE_LLAMA_PROTOCOL.md",
    "phase7_protocol": "final/results/actionability_gate_development/PHASE7_PROTOCOL.md",
    "perturbation_normalization_spec": "final/results/actionability_gate_development/PERTURBATION_NORMALIZATION_SPEC.md",
    "mechanism_decision": "final/results/actionability_gate_development/PHASE7_MECHANISM_DECISION.md",
    # code
    "rho_normalization_and_scoring_code": "scripts/phase7_lib.py",
    "rho_calibration_code": "scripts/phase7_rho_calibration.py",
    "validation_specificity_code": "scripts/phase7_validation_specificity.py",
    "decision_code": "scripts/phase7_gate_v2_evaluation.py",
    "figures_code": "scripts/phase7_make_figures.py",
    "channel_discovery_code": "scripts/discover_sakiko_channels.py",
    "layer_grid_and_geometry_code": "scripts/phase5_mistral_lib.py",
    # development evidence the gate was tuned on
    "dev_rho_results": "final/results/actionability_gate_development/rho_specificity_results.json",
    "dev_gate_decisions": "final/results/actionability_gate_development/gate_v2_development_decisions.json",
    "rho_calibration_values": "final/results/actionability_gate_development/rho_calibration.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries, missing = {}, []
    for key, rel in ARTIFACTS.items():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel); continue
        entries[key] = {"path": rel, "sha256": sha256(p), "bytes": p.stat().st_size}
    head = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "gate_version": "Actionability Gate v2",
        "status": "FROZEN for prospective validation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_branch": branch, "repo_head_at_freeze": head,
        "development_evidence": {
            "models": ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"],
            "channels": ["qwen/ca_rfi (positive control)", "qwen/tc_rfi (negative control)",
                         "mistral/ca_tc (discriminator)", "mistral/rfi_tc (negative control)",
                         "mistral/ca_direct (negative control)"],
            "splits_used": ["train", "validation"],
            "test_used": False,
            "note": "No Phase-7 configuration was evaluated on a test split.",
        },
        "immutability_rule": (
            "After this manifest, no threshold, grid, normalization, seed block, or criterion "
            "may change before or during prospective validation. A failed prospective "
            "validation is published as a falsification; no Gate v3 may be fitted on the same "
            "prospective model."),
        "artifacts": entries,
        "missing_artifacts": missing,
    }
    outp = P.OUT / "GATE_V2_LOCK_MANIFEST.json"
    outp.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v["sha256"][:16] + "…" for k, v in entries.items()}, indent=2))
    if missing:
        print("MISSING (manifest incomplete):", missing)
    print("wrote", outp)


if __name__ == "__main__":
    main()
