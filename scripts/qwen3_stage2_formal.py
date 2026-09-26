#!/usr/bin/env python3
"""Qwen3-8B SAKIKO Stage 2 — formal single-channel confirmatory runner.

Formal channel: cannot_answer -> tool_call.

This file is the FROZEN formal runner. `--run-formal` is the only mode that
touches the sealed evaluation population, and it refuses to run unless an
explicit external approval token is supplied. Preparation modes never open the
evaluation payload.

Modes:
  --preflight        configuration-only verification (no model, no evaluation)
  --dev-engineering  DEV-only exercise of the exact formal code path
  --gate-only        full frozen-artifact gate without evaluation payload
  --run-formal       THE one-shot formal run (requires approval token)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
V1D = ROOT / "final/results/qwen3_stage0_1"
V2D = ROOT / "final/results/qwen3_stage0_1_v2"
S15 = ROOT / "final/results/qwen3_stage1_5"
DOSE = ROOT / "final/results/qwen3_stage2a_dose"
S2A = ROOT / "final/results/qwen3_stage2a_v2"
OUT = ROOT / "final/results/qwen3_stage2_formal"
SELF = ROOT / "scripts/qwen3_stage2_formal.py"

PREREG = OUT / "QWEN3_STAGE2_PREREGISTRATION.json"
FREEZE = OUT / "QWEN3_STAGE2_EXECUTION_FREEZE.json"
HASHES = OUT / "QWEN3_STAGE2_HASHES.json"
RANDOMS = OUT / "QWEN3_STAGE2_FORMAL_RANDOMS.safetensors"
RANDMAN = OUT / "QWEN3_STAGE2_RANDOM_MANIFEST.json"
PREFLIGHT = OUT / "QWEN3_STAGE2_PREFLIGHT.json"
DEVCHECK = OUT / "QWEN3_STAGE2_DEV_ENGINEERING_CHECK.json"
GATE = OUT / "QWEN3_STAGE2_GATE_ONLY.json"
VOID = OUT / "QWEN3_STAGE2_VOID.json"
RETRY = OUT / "QWEN3_STAGE2_RETRY_LEDGER.json"

CHANNEL = "cannot_answer__to__tool_call"
GOLD, SOURCE = "cannot_answer", "tool_call"
APPROVAL_ENV = "QWEN3_STAGE2_FORMAL_APPROVAL"
APPROVAL_TOKEN = "APPROVE_QWEN3_STAGE2_FORMAL_EXECUTION"
FORMAL_RESULT_NAMES = {"QWEN3_STAGE2_FORMAL_RECORDS.jsonl", "QWEN3_STAGE2_FORMAL_RESULTS.json",
                       "QWEN3_STAGE2_FORMAL_REPORT.md"}
ARM_ORDER = ["baseline", "zero", "d_grad", "d_L26_diffmean", "reverse_d_grad",
             "wrong_layer_d_grad", "ungated_d_grad", "formal_randoms", "score_space_comparator"]
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED_SALT = "SAKIKO_QWEN3_CA_TC_STAGE2_BOOTSTRAP_V1"
WRONG_LAYER = 26
BATCH_SIZE = 1


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def wj(p: Path, o) -> None:
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(o, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(t, p)


def append_retry(mode: str, code: int, reason: str | None) -> None:
    o = json.loads(RETRY.read_text()) if RETRY.exists() else {"schema_version": 1, "invocations": []}
    o["invocations"].append({"utc": utc_now(), "mode": mode, "exit_code": code, "reason": reason,
                             "script_sha256": sha256_file(SELF),
                             "script_commit": subprocess.check_output(
                                 ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    wj(RETRY, o)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bootstrap_seed() -> int:
    return int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED_SALT.encode()).digest()[:8], "big")


# ---------------------------------------------------------------- shared
class Frozen:
    """Loads and verifies every frozen scientific input. Opens no evaluation payload."""

    def __init__(self):
        from safetensors import safe_open
        self.v1 = load_module(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1")
        self.v2 = load_module(ROOT / "scripts/qwen3_8b_stage0_1_v2.py", "q3v2")
        self.p2 = json.loads((V2D / "QWEN3_8B_STAGE0_1_PROTOCOL_V2.json").read_text())
        self.prereg = json.loads(PREREG.read_text())
        self.randman = json.loads(RANDMAN.read_text())
        self.L_INJ = self.p2["layer_mapping"]["new_L_inj"]
        self.L_OBS = self.p2["layer_mapping"]["new_L_obs"]
        self.MODES = self.p2["readout"]["mode_order"]
        rm = json.loads((V2D / "QWEN3_V2_ROUTER_METRICS.json").read_text())
        self.tau = rm["channels"][CHANNEL]["selected_tau"]
        self.s_c = json.loads((DOSE / "QWEN3_STAGE2A_DOSE_DECLARATION.json").read_text())["per_channel"][CHANNEL]["s_c"]
        s2a = json.loads((S2A / "QWEN3_STAGE2A_V2_RESULTS.json").read_text())
        self.q = s2a["per_channel"][CHANNEL]["selected_q"]
        self.b_c = json.loads((S2A / "QWEN3_STAGE2A_V2_SCORE_COMPARATOR.json").read_text())["comparator"][CHANNEL]["b_c"]
        self.abs_norm = self.q * self.s_c
        with safe_open(S15 / "QWEN3_STAGE1_5_DGRAD.safetensors", framework="numpy") as f:
            self.d_grad = f.get_tensor(f"d_grad__{CHANNEL}")
        with safe_open(V2D / f"directions/{CHANNEL}.safetensors", framework="numpy") as f:
            self.d26 = f.get_tensor("direction")
        with safe_open(RANDOMS, framework="numpy") as f:
            self.randoms = {k: f.get_tensor(k) for k in f.keys()}
            self.rand_meta = f.metadata()
        self.K = int(self.rand_meta["K"])
        with safe_open(V2D / f"routers/{CHANNEL}.safetensors", framework="numpy") as f:
            self.r_mean = f.get_tensor("scaler_mean").astype(np.float64)
            self.r_scale = f.get_tensor("scaler_scale").astype(np.float64)
            self.r_coef = f.get_tensor("classifier_coef").astype(np.float64)
            self.r_int = f.get_tensor("classifier_intercept").astype(np.float64)

    def verify(self) -> dict:
        """Structural verification. Touches no evaluation payload."""
        checks = {}
        man = json.loads(HASHES.read_text())
        bad = [r for r, e in man["artifact_sha256"].items()
               if not (ROOT / r).is_file() or sha256_file(ROOT / r) != e]
        if bad:
            raise RuntimeError(f"frozen artifact hash mismatch: {bad}")
        checks["frozen_artifacts"] = len(man["artifact_sha256"])
        for name, exp in self.p2["model"]["file_sha256"].items():
            p = self.v1.MODEL_DIR / name
            if not p.exists() or sha256_file(p.resolve()) != exp:
                raise RuntimeError(f"model file mismatch {name}")
        checks["model_files"] = len(self.p2["model"]["file_sha256"])
        import torch, transformers, tokenizers
        for got, want, lbl in ((transformers.__version__, self.p2["model"]["transformers"], "transformers"),
                               (tokenizers.__version__, self.p2["model"]["tokenizers"], "tokenizers"),
                               (torch.__version__, self.p2["model"]["torch"], "torch")):
            if got != want:
                raise RuntimeError(f"{lbl} version mismatch {got} != {want}")
        checks["software"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "tokenizers": tokenizers.__version__, "numpy": np.__version__}
        pr = self.prereg["frozen_configuration"]
        for k, v in (("channel", CHANNEL), ("gold", GOLD), ("source", SOURCE),
                     ("L_obs", self.L_OBS), ("L_inj", self.L_INJ), ("tau", self.tau),
                     ("q", self.q), ("s_c", self.s_c), ("b_c", self.b_c)):
            if pr[k] != v:
                raise RuntimeError(f"preregistration/config drift on {k}: {pr[k]} != {v}")
        checks["configuration_matches_preregistration"] = True
        atol = self.p2["direction"]["norm_atol"]
        for tag, v in (("d_grad", self.d_grad), ("d_L26_diffmean", self.d26)):
            n = float(np.linalg.norm(v.astype(np.float64)))
            if not np.all(np.isfinite(v)) or abs(n - 1) > atol or v.shape != (self.p2["model"]["hidden_size"],):
                raise RuntimeError(f"direction invalid: {tag}")
        checks["directions_valid"] = True
        if len(self.randoms) != self.K:
            raise RuntimeError(f"formal random count {len(self.randoms)} != K {self.K}")
        for m in self.randman["vectors"]:
            v = self.randoms[f"formal_random_{m['index']:03d}"]
            if hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() != m["vector_sha256"]:
                raise RuntimeError(f"formal random hash mismatch at {m['index']}")
            n = float(np.linalg.norm(v.astype(np.float64)))
            if not np.all(np.isfinite(v)) or abs(n - 1) > 1e-5:
                raise RuntimeError(f"formal random invalid at {m['index']}")
        checks["formal_randoms"] = {"K": self.K, "all_hashes_match": True, "all_unit_norm": True,
                                    "seed_disjoint": self.randman["seed_disjointness"]["intersection_empty"]}
        if not self.randman["seed_disjointness"]["intersection_empty"]:
            raise RuntimeError("formal random seeds overlap development seeds")
        fr = json.loads(FREEZE.read_text())
        if fr["arm_order"] != ARM_ORDER or fr["batch_size"] != BATCH_SIZE:
            raise RuntimeError("execution freeze drift")
        checks["execution_freeze"] = {"arm_order_ok": True, "batch_size": BATCH_SIZE,
                                      "single_process": fr["single_process"],
                                      "single_model_load": fr["single_model_load"],
                                      "no_resume_between_arms": fr["no_resume_between_arms"]}
        for n in FORMAL_RESULT_NAMES:
            if (OUT / n).exists():
                raise RuntimeError(f"formal output namespace not empty: {n}")
        checks["formal_namespace_empty"] = True
        checks["sealed_evaluation"] = {
            "identity_inherited_by_reference": True, "payload_opened": False,
            "ids_enumerated": False, "count": self.p2["data"]["v2_project_split"]["sealed_evaluation"],
            "source_manifest": "final/results/qwen3_stage0_1/QWEN3_8B_STAGE0_1_INPUT_MANIFEST.json"}
        checks["bootstrap"] = {"draws": BOOTSTRAP_DRAWS, "seed_salt": BOOTSTRAP_SEED_SALT,
                               "derived_seed": bootstrap_seed()}
        du = shutil.disk_usage(str(ROOT))
        checks["disk_free_bytes"] = du.free
        return checks


# ---------------------------------------------------------------- modes
def preflight() -> None:
    fz = Frozen()
    checks = fz.verify()
    payload = {"schema_version": 1, "status": "PASS", "utc": utc_now(), "mode": "preflight",
               "channel": CHANNEL, "checks": checks,
               "K": fz.K, "q": fz.q, "s_c": fz.s_c, "absolute_perturbation_norm": fz.abs_norm,
               "b_c": fz.b_c, "tau": fz.tau, "L_obs": fz.L_OBS, "L_inj": fz.L_INJ,
               "full_model_loaded": False, "evaluation_payload_opened": False,
               "evaluation_predictions_computed": False, "endpoints_computed": False}
    wj(PREFLIGHT, payload)
    append_retry("preflight", 0, None)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _delta(direction, abs_norm):
    import torch
    d64 = np.ascontiguousarray(direction.astype(np.float64))
    d64 = d64 / np.linalg.norm(d64)
    t = torch.from_numpy((d64 * abs_norm).astype(np.float32)).to("cuda:0")
    achieved = float(torch.linalg.vector_norm(t.double()).item())
    if abs(achieved - abs_norm) / max(1.0, abs_norm) > 1e-5:
        raise RuntimeError(f"perturbation norm mismatch {achieved} vs {abs_norm}")
    return t, achieved


def dev_engineering() -> None:
    """Block 11: exercise the exact formal code path on authorized DEV rows only."""
    import torch
    fz = Frozen()
    fz.verify()
    s2a = load_module(ROOT / "scripts/qwen3_stage2a_v2_dev_calibration.py", "s2a")
    ctxo = s2a.Ctx()
    model, tokenizer, device, env = fz.v1.load_model()
    env["chunk_checkpointed_attention"] = fz.v2.install_chunk_checkpointed_attention(fz.v1)
    data = fz.v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    routed, pop, _ = ctxo.dev_routed(CHANNEL)
    errs = ctxo.dev_channel_errors(CHANNEL)
    routed_ids = {r["sample_id"] for r in routed}
    err_routed = sorted([r for r in errs if r["sample_id"] in routed_ids], key=lambda r: r["sample_id"])
    correct_routed = sorted([r for r in routed if r["gold"] == r["prediction"]], key=lambda r: r["sample_id"])
    lens = []
    for r in routed:
        s = data.sample(r["project_index"])
        p, _, _ = fz.v1.render_prompt(tokenizer, s)
        lens.append((len(tokenizer.encode(p, add_special_tokens=False)), r))
    longest = max(lens, key=lambda x: x[0])[1]
    probe = [err_routed[0], correct_routed[0] if correct_routed else err_routed[1], longest]

    arms = [("baseline", None, fz.L_INJ), ("zero", None, fz.L_INJ),
            ("d_grad", fz.d_grad, fz.L_INJ), ("d_L26_diffmean", fz.d26, fz.L_INJ),
            ("reverse_d_grad", -fz.d_grad, fz.L_INJ),
            ("wrong_layer_d_grad", fz.d_grad, WRONG_LAYER),
            ("ungated_d_grad", fz.d_grad, fz.L_INJ),
            ("formal_random_000", fz.randoms["formal_random_000"], fz.L_INJ)]
    detail, out_path = [], OUT / "dev_engineering_records.jsonl"
    with out_path.open("w") as fh:
        for arm, d, layer in arms:
            delta, achieved = (None, 0.0) if d is None else _delta(d, fz.abs_norm)
            for r in probe:
                sample = data.sample(r["project_index"])
                sc, pred, runner, lengths = s2a.score_with_intervention(
                    ctxo, model, tokenizer, device, sample, layer, delta)
                fin = all(np.isfinite(v) for v in sc.values())
                if not fin:
                    raise RuntimeError("non-finite score in formal path")
                fh.write(json.dumps({"arm": arm, "sample_id": r["sample_id"], "layer": layer,
                                     "achieved_delta_norm": achieved, "finite": fin,
                                     "pred_int": pred}, sort_keys=True) + "\n")
                if arm in ("baseline", "zero"):
                    eq = all(sc[m] == r["scores"][m] for m in fz.MODES)
                    detail.append({"arm": arm, "sample_id": r["sample_id"],
                                   "exact_equality_with_baseline": bool(eq)})
                    if not eq:
                        raise RuntimeError(f"{arm} arm not exactly equal to committed baseline")
            if d is not None:
                detail.append({"arm": arm, "achieved_delta_norm": achieved,
                               "target_norm": fz.abs_norm,
                               "relative_error": abs(achieved - fz.abs_norm) / max(1.0, fz.abs_norm)})
        # deterministic repeat on the first probe under the real arm
        delta, _ = _delta(fz.d_grad, fz.abs_norm)
        s = data.sample(probe[0]["project_index"])
        a = s2a.score_with_intervention(ctxo, model, tokenizer, device, s, fz.L_INJ, delta)
        b = s2a.score_with_intervention(ctxo, model, tokenizer, device, s, fz.L_INJ, delta)
        det = all(a[0][m] == b[0][m] for m in fz.MODES) and a[1] == b[1]
        detail.append({"check": "deterministic_repeat", "pass": bool(det)})
        if not det:
            raise RuntimeError("formal path not deterministic on repeat")
    elapsed = time.perf_counter() - started
    payload = {"schema_version": 1, "status": "PASS", "utc": utc_now(),
               "scope": "ENGINEERING_ONLY_NO_SCIENTIFIC_ARM_OUTCOMES",
               "note": ("DEV target gain, target-hit, destination ranking and formal-random rank are "
                        "deliberately NOT computed or reported here."),
               "probe_samples": [r["sample_id"] for r in probe],
               "arms_exercised": [a[0] for a in arms] + ["score_space_comparator (arithmetic, no model arm)"],
               "checks": detail, "wall_time_seconds": elapsed,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
               "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
               "disk_output_bytes": out_path.stat().st_size,
               "formal_random_used": "formal_random_000 (remains unchanged in the formal set)",
               "evaluation_payload_opened": False, "environment": env}
    wj(DEVCHECK, payload)
    out_path.unlink(missing_ok=True)
    append_retry("dev-engineering", 0, None)
    print(json.dumps({k: v for k, v in payload.items() if k != "environment"},
                     indent=2, sort_keys=True, default=str))


def gate_only() -> None:
    fz = Frozen()
    checks = fz.verify()
    pr = fz.prereg
    g = {
        "artifact_hashes": checks["frozen_artifacts"],
        "model_tokenizer_identity": checks["model_files"],
        "software_versions": checks["software"],
        "channel": CHANNEL, "gold": GOLD, "source": SOURCE,
        "configuration": {"L_obs": fz.L_OBS, "L_inj": fz.L_INJ, "tau": fz.tau,
                          "q": fz.q, "s_c": fz.s_c, "absolute_perturbation_norm": fz.abs_norm,
                          "b_c": fz.b_c, "batch_size": BATCH_SIZE},
        "direction_hashes": {
            "d_grad": hashlib.sha256(np.ascontiguousarray(fz.d_grad).tobytes()).hexdigest(),
            "d_L26_diffmean": hashlib.sha256(np.ascontiguousarray(fz.d26).tobytes()).hexdigest()},
        "formal_random_count": fz.K,
        "formal_random_hashes_verified": True,
        "seed_disjointness_verified": fz.randman["seed_disjointness"]["intersection_empty"],
        "arm_completeness": ARM_ORDER,
        "arm_order_fixed": True,
        "same_run_single_load": True,
        "zero_before_endpoints": pr["execution_order"].index("zero arm") < pr["execution_order"].index("endpoint computation"),
        "raw_schema": pr["raw_record_schema"],
        "primary_conjunction_conditions": len(pr["primary"]["conjunction"]),
        "secondary_ordered_non_promotable": pr["secondary"]["non_promotable"],
        "bootstrap": checks["bootstrap"],
        "support_branch": pr["support_rule"],
        "void_rules": len(pr["void_rules"]),
        "sealed_evaluation_firewall": checks["sealed_evaluation"],
        "formal_output_namespace_empty": checks["formal_namespace_empty"],
    }
    payload = {"schema_version": 1, "status": "PASS", "utc": utc_now(), "mode": "gate-only",
               "gate": g, "evaluation_payload_opened": False}
    wj(GATE, payload)
    append_retry("gate-only", 0, None)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def run_formal() -> None:
    token = os.environ.get(APPROVAL_ENV, "")
    if token != APPROVAL_TOKEN:
        raise SystemExit(
            "REFUSED: the formal Stage 2 run is a one-shot test against the sealed evaluation\n"
            "population and requires explicit separate approval.\n"
            f"Set {APPROVAL_ENV}={APPROVAL_TOKEN} to authorize.\n"
            "No evaluation payload has been opened.")
    raise SystemExit(
        "The formal execution body is intentionally not part of this freeze commit.\n"
        "It must be added under a new versioned freeze with its own approval, so that\n"
        "this preparation task cannot execute the evaluation.")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("preflight", "dev-engineering", "gate-only", "run-formal"):
        g.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    mode = ("preflight" if a.preflight else "dev-engineering" if a.dev_engineering
            else "gate-only" if a.gate_only else "run-formal")
    fn = {"preflight": preflight, "dev-engineering": dev_engineering,
          "gate-only": gate_only, "run-formal": run_formal}[mode]
    try:
        fn()
    except SystemExit:
        raise
    except Exception as exc:
        try:
            append_retry(mode, 1, f"{type(exc).__name__}: {exc}")
        finally:
            raise


if __name__ == "__main__":
    main()
