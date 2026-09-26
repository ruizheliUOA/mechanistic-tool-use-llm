#!/usr/bin/env python3
"""Qwen3-8B SAKIKO Stage 2 — formal runner V2 (execution body implemented).

Implements the execution body for the scientific design frozen at HEAD
42261b30255c167975357f225f49c427474c1fd4. It changes no scientific definition:
every frozen value is read from committed artifacts at runtime.

Evaluation access is confined to a single class, `EvaluationAccess`, constructed
only inside `run_formal`, only after the approval lock passes and the access
marker has been atomically promoted. Importing this module performs zero dataset
access and zero model loading.

Modes:
  --preflight              configuration-only (no model, no evaluation)
  --dev-engineering-check  authorized DEV rows only, engineering facts only
  --run-formal             THE one-shot formal run; requires external approval
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
import uuid
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
SELF = ROOT / "scripts/qwen3_stage2_formal_v2.py"

PREREG = OUT / "QWEN3_STAGE2_PREREGISTRATION.json"
FREEZE = OUT / "QWEN3_STAGE2_EXECUTION_FREEZE.json"
IMPL_FREEZE = OUT / "QWEN3_STAGE2_IMPLEMENTATION_FREEZE_V2.json"
HASHES_V2 = OUT / "QWEN3_STAGE2_HASHES_V2.json"
RANDOMS = OUT / "QWEN3_STAGE2_FORMAL_RANDOMS.safetensors"
RANDMAN = OUT / "QWEN3_STAGE2_RANDOM_MANIFEST.json"
PREFLIGHT = OUT / "QWEN3_STAGE2_PREFLIGHT_V2.json"
DEVCHECK = OUT / "QWEN3_STAGE2_DEV_ENGINEERING_CHECK_V2.json"
GATE = OUT / "QWEN3_STAGE2_GATE_ONLY_V2.json"
ACCESS_MARKER = OUT / "FORMAL_ACCESS_STARTED.json"
VOID = OUT / "QWEN3_STAGE2_VOID.json"
RECORDS = OUT / "QWEN3_STAGE2_FORMAL_RECORDS.jsonl"
RESULTS = OUT / "QWEN3_STAGE2_FORMAL_RESULTS.json"
REPORT = OUT / "QWEN3_STAGE2_FORMAL_REPORT.md"
RETRY = OUT / "QWEN3_STAGE2_RETRY_LEDGER_V2.json"

CHANNEL = "cannot_answer__to__tool_call"
GOLD, SOURCE = "cannot_answer", "tool_call"
APPROVAL_ENV = "QWEN3_STAGE2_FORMAL_APPROVAL"
APPROVAL_TOKEN = "APPROVE_QWEN3_STAGE2_FORMAL_EXECUTION"
FORMAL_OUTPUTS = [RECORDS, RESULTS, REPORT, ACCESS_MARKER, VOID]
PERTURBATION_RTOL = 1e-5
TOTAL_ROWS = 3652
SPLIT_SEED = 42


# ---------------------------------------------------------------- utilities
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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def append_retry(mode: str, code: int, reason: str | None) -> None:
    o = json.loads(RETRY.read_text()) if RETRY.exists() else {"schema_version": 1, "invocations": []}
    o["invocations"].append({"utc": utc_now(), "mode": mode, "exit_code": code, "reason": reason,
                             "script_sha256": sha256_file(SELF), "script_commit": git("rev-parse", "HEAD")})
    wj(RETRY, o)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- frozen inputs
class Frozen:
    """Loads and verifies every frozen scientific input. Opens no evaluation payload."""

    def __init__(self):
        from safetensors import safe_open
        self.v1 = load_module(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1")
        self.v2 = load_module(ROOT / "scripts/qwen3_8b_stage0_1_v2.py", "q3v2")
        self.s2a = load_module(ROOT / "scripts/qwen3_stage2a_v2_dev_calibration.py", "q3s2a")
        self.p2 = json.loads((V2D / "QWEN3_8B_STAGE0_1_PROTOCOL_V2.json").read_text())
        self.prereg = json.loads(PREREG.read_text())
        self.freeze = json.loads(FREEZE.read_text())
        self.randman = json.loads(RANDMAN.read_text())
        self.L_INJ = self.p2["layer_mapping"]["new_L_inj"]
        self.L_OBS = self.p2["layer_mapping"]["new_L_obs"]
        self.MODES = self.p2["readout"]["mode_order"]
        self.DIM = self.p2["model"]["hidden_size"]
        rm = json.loads((V2D / "QWEN3_V2_ROUTER_METRICS.json").read_text())
        self.tau = rm["channels"][CHANNEL]["selected_tau"]
        self.s_c = json.loads((DOSE / "QWEN3_STAGE2A_DOSE_DECLARATION.json").read_text())["per_channel"][CHANNEL]["s_c"]
        s2a_res = json.loads((S2A / "QWEN3_STAGE2A_V2_RESULTS.json").read_text())
        self.q = s2a_res["per_channel"][CHANNEL]["selected_q"]
        self.b_c = json.loads((S2A / "QWEN3_STAGE2A_V2_SCORE_COMPARATOR.json").read_text())["comparator"][CHANNEL]["b_c"]
        self.abs_norm = self.q * self.s_c
        self.arm_order = self.freeze["arm_order"]
        self.batch_size = self.freeze["batch_size"]
        self.wrong_layer = self.L_OBS
        self.boot_draws = self.prereg["bootstrap"]["draws"]
        self.boot_seed = int.from_bytes(
            hashlib.sha256(self.prereg["bootstrap"]["seed_salt"].encode()).digest()[:8], "big")
        with safe_open(S15 / "QWEN3_STAGE1_5_DGRAD.safetensors", framework="numpy") as f:
            self.d_grad = f.get_tensor(f"d_grad__{CHANNEL}")
        with safe_open(V2D / f"directions/{CHANNEL}.safetensors", framework="numpy") as f:
            self.d26 = f.get_tensor("direction")
        with safe_open(RANDOMS, framework="numpy") as f:
            self.randoms = {k: f.get_tensor(k) for k in f.keys()}
        self.K = int(self.randman["K"])
        with safe_open(V2D / f"routers/{CHANNEL}.safetensors", framework="numpy") as f:
            self.r_mean = f.get_tensor("scaler_mean").astype(np.float64)
            self.r_scale = f.get_tensor("scaler_scale").astype(np.float64)
            self.r_coef = f.get_tensor("classifier_coef").astype(np.float64)
            self.r_int = f.get_tensor("classifier_intercept").astype(np.float64)

    def verify(self) -> dict:
        checks: dict[str, Any] = {}
        man = json.loads(HASHES_V2.read_text())
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
                raise RuntimeError(f"configuration drift on {k}: {pr[k]} != {v}")
        checks["configuration_matches_preregistration"] = True
        atol = self.p2["direction"]["norm_atol"]
        for tag, v in (("d_grad", self.d_grad), ("d_L26_diffmean", self.d26)):
            n = float(np.linalg.norm(v.astype(np.float64)))
            if not np.all(np.isfinite(v)) or abs(n - 1) > atol or v.shape != (self.DIM,):
                raise RuntimeError(f"direction invalid: {tag}")
        rev = -self.d_grad
        if abs(float(np.linalg.norm(rev.astype(np.float64))) - 1) > atol:
            raise RuntimeError("reverse construction invalid")
        checks["directions_valid"] = True
        checks["reverse_construction_verified"] = True
        if len(self.randoms) != self.K:
            raise RuntimeError(f"random count {len(self.randoms)} != K {self.K}")
        order_ok = True
        for i, m in enumerate(self.randman["vectors"]):
            if m["index"] != i:
                order_ok = False
            v = self.randoms[f"formal_random_{m['index']:03d}"]
            if hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() != m["vector_sha256"]:
                raise RuntimeError(f"random hash mismatch at {m['index']}")
            if not np.all(np.isfinite(v)) or abs(float(np.linalg.norm(v.astype(np.float64))) - 1) > 1e-5:
                raise RuntimeError(f"random invalid at {m['index']}")
        if not order_ok or not self.randman["seed_disjointness"]["intersection_empty"]:
            raise RuntimeError("random seed order or disjointness failure")
        checks["formal_randoms"] = {"K": self.K, "hashes_match": True, "seed_order_ok": True,
                                    "seed_disjoint": True}
        if self.freeze["arm_order"] != self.arm_order or self.batch_size != 1:
            raise RuntimeError("execution freeze drift")
        checks["execution_freeze"] = {"arm_order": self.arm_order, "batch_size": self.batch_size,
                                      "single_process": self.freeze["single_process"],
                                      "single_model_load": self.freeze["single_model_load"],
                                      "no_resume_between_arms": self.freeze["no_resume_between_arms"]}
        for p in FORMAL_OUTPUTS:
            if p.exists():
                raise RuntimeError(f"formal output namespace not empty: {p.name}")
        checks["formal_namespace_empty"] = True
        checks["sealed_evaluation"] = {
            "identity_inherited_by_reference": True, "payload_opened": False, "ids_enumerated": False,
            "count": self.p2["data"]["v2_project_split"]["sealed_evaluation"],
            "source_manifest": "final/results/qwen3_stage0_1/QWEN3_8B_STAGE0_1_INPUT_MANIFEST.json"}
        checks["bootstrap"] = {"draws": self.boot_draws,
                               "seed_salt": self.prereg["bootstrap"]["seed_salt"],
                               "derived_seed": self.boot_seed}
        checks["raw_record_schema"] = self.prereg["raw_record_schema"]
        checks["primary_conjunction_conditions"] = len(self.prereg["primary"]["conjunction"])
        checks["void_rules"] = len(self.prereg["void_rules"])
        checks["disk_free_bytes"] = shutil.disk_usage(str(ROOT)).free
        return checks

    def router_prob(self, A26_rows: np.ndarray) -> np.ndarray:
        z = (A26_rows.astype(np.float64) - self.r_mean) / self.r_scale
        return 1.0 / (1.0 + np.exp(-(z @ self.r_coef + self.r_int[0])))


# ---------------------------------------------------------------- intervention
def build_delta(direction: np.ndarray, abs_norm: float):
    import torch
    d64 = np.ascontiguousarray(direction.astype(np.float64))
    d64 = d64 / np.linalg.norm(d64)
    t = torch.from_numpy((d64 * abs_norm).astype(np.float32)).to("cuda:0")
    achieved = float(torch.linalg.vector_norm(t.double()).item())
    rel = abs(achieved - abs_norm) / max(1.0, abs_norm)
    if rel > PERTURBATION_RTOL:
        raise RuntimeError(f"perturbation norm mismatch {achieved} vs {abs_norm} (rel {rel:.3e})")
    return t, achieved


def arm_plan(fz: Frozen):
    """The frozen arm battery, expanded in the frozen order. Read, never restated."""
    plan = []
    for arm in fz.arm_order:
        if arm == "baseline":
            plan.append(("baseline", None, fz.L_INJ, "routed_and_all", None))
        elif arm == "zero":
            plan.append(("zero", None, fz.L_INJ, "routed", None))
        elif arm == "d_grad":
            plan.append(("d_grad", fz.d_grad, fz.L_INJ, "routed", "d_grad"))
        elif arm == "d_L26_diffmean":
            plan.append(("d_L26_diffmean", fz.d26, fz.L_INJ, "routed", "d_L26_diffmean"))
        elif arm == "reverse_d_grad":
            plan.append(("reverse_d_grad", -fz.d_grad, fz.L_INJ, "routed", "reverse_d_grad"))
        elif arm == "wrong_layer_d_grad":
            plan.append(("wrong_layer_d_grad", fz.d_grad, fz.wrong_layer, "routed", "d_grad"))
        elif arm == "ungated_d_grad":
            plan.append(("ungated_d_grad", fz.d_grad, fz.L_INJ, "pred_eq_source", "d_grad"))
        elif arm == "formal_randoms":
            for i in range(fz.K):
                plan.append((f"formal_random_{i:03d}", fz.randoms[f"formal_random_{i:03d}"],
                             fz.L_INJ, "routed", f"formal_random_{i:03d}"))
        elif arm == "score_space_comparator":
            plan.append(("score_space_comparator", None, None, "routed", None))
        else:
            raise RuntimeError(f"unknown frozen arm {arm}")
    return plan


# ---------------------------------------------------------------- endpoints
def endpoints(fz: Frozen, base_by_id: dict, eff: dict, err_ids: list, correct_ids: list):
    se = ga = w2 = 0
    per = []
    for sid in err_ids:
        pi = eff.get(sid, base_by_id[sid]["prediction"])
        ex = pi != SOURCE
        gd = pi == GOLD
        se += int(ex); ga += int(ex and gd); w2 += int(ex and not gd)
        per.append((int(ex), int(ex and gd), int(ex and not gd)))
    n_err = len(err_ids)
    th = (ga / se) if se > 0 else None
    tg = ga - w2
    tgr = (tg / n_err) if n_err else None
    fixed = broke = coll = 0
    pc = []
    for sid, b in base_by_id.items():
        pi = eff.get(sid, b["prediction"])
        was, now = b["prediction"] == b["gold"], pi == b["gold"]
        fixed += int((not was) and now); broke += int(was and not now)
    for sid in correct_ids:
        pi = eff.get(sid, base_by_id[sid]["prediction"])
        bad = int(pi != base_by_id[sid]["gold"]); coll += bad; pc.append(bad)
    rng = np.random.default_rng(fz.boot_seed)
    pe = np.array(per) if per else np.zeros((0, 3))
    pca = np.array(pc) if pc else np.zeros(0)
    bt = {"tgr": np.empty(fz.boot_draws), "th": np.empty(fz.boot_draws), "cc": np.empty(fz.boot_draws)}
    for b in range(fz.boot_draws):
        i = rng.integers(0, len(pe), len(pe)) if len(pe) else np.zeros(0, dtype=int)
        s = pe[i] if len(pe) else pe
        bse = s[:, 0].sum() if len(s) else 0
        bga = s[:, 1].sum() if len(s) else 0
        bw2 = s[:, 2].sum() if len(s) else 0
        bt["tgr"][b] = (bga - bw2) / len(s) if len(s) else np.nan
        bt["th"][b] = (bga / bse) if bse > 0 else np.nan
        j = rng.integers(0, len(pca), len(pca)) if len(pca) else np.zeros(0, dtype=int)
        bt["cc"][b] = pca[j].mean() if len(pca) else np.nan

    def ci(a):
        a = a[~np.isnan(a)]
        return [float(np.quantile(a, .025)), float(np.quantile(a, .975))] if len(a) else [None, None]
    return {"n_channel_error": n_err, "source_exits": se, "gold_arrivals": ga, "wrong_to_wrong": w2,
            "target_hit": th, "target_gain_count": tg, "target_gain_rate": tgr,
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "baseline_correct_n": len(correct_ids), "collateral_count": coll,
            "clean_collateral_rate": (coll / len(correct_ids)) if correct_ids else None,
            "target_gain_rate_ci95": ci(bt["tgr"]), "target_hit_ci95": ci(bt["th"]),
            "clean_collateral_rate_ci95": ci(bt["cc"])}


def score_changes(fz: Frozen, base_by_id: dict, recs: list, err_ids: set):
    dm, dg, ds = [], [], []
    other: dict[str, list] = {}
    pos = 0
    for r in recs:
        if r["sample_id"] not in err_ids or r["scores_int"] is None:
            continue
        b, i = r["scores_base"], r["scores_int"]
        d = (i[GOLD] - i[SOURCE]) - (b[GOLD] - b[SOURCE])
        dm.append(d); dg.append(i[GOLD] - b[GOLD]); ds.append(i[SOURCE] - b[SOURCE])
        pos += int(d > 0)
        for m in fz.MODES:
            if m not in (GOLD, SOURCE):
                other.setdefault(m, []).append(i[m] - b[m])
    if not dm:
        return {"n": 0}
    return {"n": len(dm), "delta_margin_median": float(np.median(dm)),
            "delta_e_gold_median": float(np.median(dg)),
            "delta_e_source_median": float(np.median(ds)),
            "delta_other_modes_median": {k: float(np.median(v)) for k, v in other.items()},
            "positive_margin_change_fraction": pos / len(dm)}


# ---------------------------------------------------------------- evaluation access
class EvaluationAccess:
    """THE ONLY evaluation-reading component.

    Constructed exclusively inside run_formal, after the approval lock passes and
    after FORMAL_ACCESS_STARTED.json has been promoted. Never referenced by
    preflight, the DEV engineering check or gate-only.
    """

    def __init__(self, fz: Frozen):
        if not ACCESS_MARKER.exists():
            raise RuntimeError("refusing evaluation access before the access marker is promoted")
        v1 = fz.v1
        train = json.loads((ROOT / "final/results/splits/train_idx.json").read_text())
        dev = json.loads((ROOT / "final/results/splits/val_idx.json").read_text())
        authorized = set(train) | set(dev)
        self.indices = sorted(set(range(TOTAL_ROWS)) - authorized)
        if len(self.indices) != fz.p2["data"]["v2_project_split"]["sealed_evaluation"]:
            raise RuntimeError("evaluation population size mismatch")
        offsets, raw_hash = v1.structural_offsets(v1.RAW_SOURCE)
        if raw_hash != v1.INPUT_HASHES["raw_source"]:
            raise RuntimeError("raw source hash mismatch")
        self.perm = np.random.default_rng(SPLIT_SEED).permutation(TOTAL_ROWS)
        self.offsets = offsets
        self.v1 = v1

    def rows(self):
        for pi in self.indices:
            raw = int(self.perm[pi])
            off, ln = self.offsets[raw]
            obj = self.v1.direct_seek_json(self.v1.RAW_SOURCE, off, ln)
            yield pi, obj


# ---------------------------------------------------------------- modes
def preflight() -> None:
    fz = Frozen()
    checks = fz.verify()
    payload = {"schema_version": 1, "status": "PASS", "utc": utc_now(), "mode": "preflight",
               "runner": "v2", "runner_sha256": sha256_file(SELF), "git_head": git("rev-parse", "HEAD"),
               "channel": CHANNEL, "checks": checks, "K": fz.K, "q": fz.q, "s_c": fz.s_c,
               "absolute_perturbation_norm": fz.abs_norm, "b_c": fz.b_c, "tau": fz.tau,
               "L_obs": fz.L_OBS, "L_inj": fz.L_INJ, "wrong_layer": fz.wrong_layer,
               "expanded_arm_count": len(arm_plan(fz)),
               "full_model_loaded": False, "evaluation_payload_opened": False,
               "evaluation_ids_enumerated": False, "predictions_generated": False,
               "endpoints_computed": False}
    wj(PREFLIGHT, payload)
    append_retry("preflight", 0, None)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def dev_engineering_check() -> None:
    import torch
    fz = Frozen()
    fz.verify()
    ctxo = fz.s2a.Ctx()
    model, tokenizer, device, env = fz.v1.load_model()
    env["chunk_checkpointed_attention"] = fz.v2.install_chunk_checkpointed_attention(fz.v1)
    data = fz.v1.TrainDevData()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    routed, pop, _ = ctxo.dev_routed(CHANNEL)
    errs = ctxo.dev_channel_errors(CHANNEL)
    rid = {r["sample_id"] for r in routed}
    err_routed = sorted([r for r in errs if r["sample_id"] in rid], key=lambda r: r["sample_id"])
    correct_routed = sorted([r for r in routed if r["gold"] == r["prediction"]], key=lambda r: r["sample_id"])
    lens = []
    for r in routed:
        s = data.sample(r["project_index"])
        p, _, _ = fz.v1.render_prompt(tokenizer, s)
        lens.append((len(tokenizer.encode(p, add_special_tokens=False)), r))
    probe = [max(lens, key=lambda x: x[0])[1], err_routed[0],
             correct_routed[0] if correct_routed else err_routed[1]]

    checks, tmp = [], OUT / "dev_engineering_v2_records.jsonl"
    exercised = ["baseline", "zero", "d_grad", "formal_random_000", "d_L26_diffmean",
                 "reverse_d_grad", "wrong_layer_d_grad", "ungated_d_grad", "score_space_comparator"]
    with tmp.open("w") as fh:
        for arm in exercised:
            if arm == "score_space_comparator":
                for r in probe:
                    e = dict(r["scores"])
                    e[GOLD] += fz.b_c / 2.0
                    e[SOURCE] -= fz.b_c / 2.0
                    pred = sorted(((v, k) for k, v in e.items()), reverse=True)[0][1]
                    fh.write(json.dumps({"arm": arm, "sample_id": r["sample_id"], "pred_int": pred},
                                        sort_keys=True) + "\n")
                checks.append({"arm": arm, "model_arm_loaded": False, "uses_frozen_b_c": fz.b_c})
                continue
            d = {"baseline": None, "zero": None, "d_grad": fz.d_grad,
                 "formal_random_000": fz.randoms["formal_random_000"],
                 "d_L26_diffmean": fz.d26, "reverse_d_grad": -fz.d_grad,
                 "wrong_layer_d_grad": fz.d_grad, "ungated_d_grad": fz.d_grad}[arm]
            layer = fz.wrong_layer if arm == "wrong_layer_d_grad" else fz.L_INJ
            delta, achieved = (None, 0.0) if d is None else build_delta(d, fz.abs_norm)
            for r in probe:
                sample = data.sample(r["project_index"])
                sc, pred, runner, _ = fz.s2a.score_with_intervention(
                    ctxo, model, tokenizer, device, sample, layer, delta)
                if not all(np.isfinite(v) for v in sc.values()):
                    raise RuntimeError("non-finite score in formal path")
                fh.write(json.dumps({"arm": arm, "sample_id": r["sample_id"], "layer": layer,
                                     "achieved_delta_norm": achieved, "pred_int": pred,
                                     "scores_shape": len(sc)}, sort_keys=True) + "\n")
                if arm in ("baseline", "zero"):
                    eq = all(sc[m] == r["scores"][m] for m in fz.MODES)
                    checks.append({"arm": arm, "sample_id": r["sample_id"], "exact_zero_equality": bool(eq)})
                    if not eq:
                        raise RuntimeError(f"{arm} not exactly equal to committed baseline")
            if d is not None:
                checks.append({"arm": arm, "target_norm": fz.abs_norm, "achieved_delta_norm": achieved,
                               "relative_error": abs(achieved - fz.abs_norm) / max(1.0, fz.abs_norm),
                               "within_rtol": True, "tensor_shape": list(d.shape)})
        delta, _ = build_delta(fz.d_grad, fz.abs_norm)
        s = data.sample(probe[1]["project_index"])
        a = fz.s2a.score_with_intervention(ctxo, model, tokenizer, device, s, fz.L_INJ, delta)
        b = fz.s2a.score_with_intervention(ctxo, model, tokenizer, device, s, fz.L_INJ, delta)
        det = all(a[0][m] == b[0][m] for m in fz.MODES) and a[1] == b[1]
        checks.append({"check": "deterministic_repeat", "pass": bool(det)})
        if not det:
            raise RuntimeError("formal path not deterministic")
        # endpoint-input validation on synthetic effect maps (no DEV outcome reported)
        base_by_id = {r["sample_id"]: r for r in [probe[1], probe[2]]}
        _ = endpoints(fz, base_by_id, {}, [probe[1]["sample_id"]], [probe[2]["sample_id"]])
        checks.append({"check": "endpoint_input_validation", "pass": True,
                       "note": "structural only; no DEV outcome computed or reported"})
    elapsed = time.perf_counter() - started
    n_cells = len(probe) * (len(exercised) - 1) + 2
    per_cell = elapsed / max(1, n_cells)
    sealed = fz.p2["data"]["v2_project_split"]["sealed_evaluation"]
    s2a_pilot = json.loads((S2A / "QWEN3_STAGE2A_V2_PILOT.json").read_text())
    pj = s2a_pilot["projection"]
    scale = sealed / 1104
    routed_eval = len(routed) * scale
    op_eval = len(pop) * scale
    cells = sealed + routed_eval * (5 + fz.K) + op_eval
    proj_s = cells * pj["seconds_per_cell"]
    peak_res = int(torch.cuda.max_memory_reserved(0))
    total_mem = int(torch.cuda.get_device_properties(0).total_memory)
    headroom = total_mem - peak_res
    disk = shutil.disk_usage(str(ROOT)).free
    ok = (headroom >= 1.5 * 2 ** 30 and proj_s * 1.15 <= 3 * 3600 and disk >= 1.5 * 2 ** 30)
    payload = {"schema_version": 1, "status": "PASS" if ok else "QWEN3_STAGE2_RUNNER_ENGINEERING_BLOCKED",
               "utc": utc_now(), "runner": "v2", "runner_sha256": sha256_file(SELF),
               "scope": "ENGINEERING_ONLY_NO_SCIENTIFIC_ARM_OUTCOMES",
               "note": ("DEV Target Gain, target-hit, destination ranking, random rank and arm superiority "
                        "are deliberately NOT computed or reported."),
               "probe_samples": [r["sample_id"] for r in probe],
               "arms_exercised": exercised, "checks": checks,
               "wall_time_seconds": elapsed, "measured_seconds_per_cell": per_cell,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
               "peak_reserved_bytes": peak_res, "memory_headroom_bytes": headroom,
               "output_bytes": tmp.stat().st_size,
               "projected_formal_seconds": proj_s,
               "projected_formal_hours_with_contingency": proj_s * 1.15 / 3600,
               "projected_disk_free_bytes": disk,
               "evaluation_payload_opened": False, "environment": env}
    wj(DEVCHECK, payload)
    tmp.unlink(missing_ok=True)
    append_retry("dev-engineering-check", 0, None if ok else payload["status"])
    print(json.dumps({k: v for k, v in payload.items() if k != "environment"},
                     indent=2, sort_keys=True, default=str))


def gate_only() -> None:
    fz = Frozen()
    checks = fz.verify()
    pr = fz.prereg
    plan = arm_plan(fz)
    g = {
        "1_all_hashes": checks["frozen_artifacts"],
        "2_selected_channel": {"channel": CHANNEL, "gold": GOLD, "source": SOURCE},
        "3_d_grad_identity": hashlib.sha256(np.ascontiguousarray(fz.d_grad).tobytes()).hexdigest(),
        "4_router_identity": {"tau": fz.tau,
                              "coef_sha256": hashlib.sha256(np.ascontiguousarray(fz.r_coef).tobytes()).hexdigest()},
        "5_q_and_s_c": {"q": fz.q, "s_c": fz.s_c, "absolute_norm": fz.abs_norm},
        "6_K": fz.K,
        "7_random_identities_and_seed_order": checks["formal_randoms"],
        "8_arm_completeness_and_order": {"frozen_order": fz.arm_order, "expanded_cells": len(plan),
                                         "expected": 7 + fz.K + 1},
        "9_batch_size": fz.batch_size,
        "10_single_process_single_load": {"single_process": fz.freeze["single_process"],
                                          "single_model_load": fz.freeze["single_model_load"],
                                          "no_resume": fz.freeze["no_resume_between_arms"]},
        "11_zero_before_endpoints": pr["execution_order"].index("exact zero-equality gate")
                                    < pr["execution_order"].index("endpoint computation"),
        "12_formal_access_marker_logic": {"marker": ACCESS_MARKER.name,
                                          "written_before_first_evaluation_read": True,
                                          "evaluation_access_class_guarded_by_marker": True},
        "13_raw_record_schema": pr["raw_record_schema"],
        "14_atomic_output_logic": {"records_written_to_temp": True,
                                   "promoted_only_after_all_arms": True},
        "15_endpoint_hierarchy": {"primary": pr["primary"]["statistic"],
                                  "conjunction_conditions": len(pr["primary"]["conjunction"]),
                                  "secondary_count": len(pr["secondary"]["ordered"]),
                                  "secondary_non_promotable": pr["secondary"]["non_promotable"]},
        "16_bootstrap": checks["bootstrap"],
        "17_support_branch": pr["support_rule"],
        "18_ten_condition_conjunction": len(pr["primary"]["conjunction"]) == 10,
        "19_frozen_interpretation_branches": sorted(pr["interpretation_rules"].keys()),
        "20_void_schema_and_no_retry": {"rules": len(pr["void_rules"]),
                                        "no_automatic_retry": True,
                                        "schema": "QWEN3_STAGE2_VOID_SCHEMA.json"},
        "21_formal_authorization_lock": {"cli_flag": "--run-formal", "env_var": APPROVAL_ENV,
                                         "requires_head_match": True, "requires_divergence_0_0": True,
                                         "requires_clean_worktree": True,
                                         "requires_empty_namespace": True,
                                         "requires_hash_match": True},
        "22_sealed_evaluation_firewall": checks["sealed_evaluation"],
        "23_formal_output_namespace_empty": checks["formal_namespace_empty"],
    }
    bad = [k for k, v in g.items() if v is False]
    payload = {"schema_version": 1, "status": "PASS" if not bad else "QWEN3_STAGE2_RUNNER_GATE_BLOCKED",
               "utc": utc_now(), "runner": "v2", "runner_sha256": sha256_file(SELF),
               "mode": "gate-only", "gate": g, "failed_items": bad,
               "evaluation_payload_opened": False}
    wj(GATE, payload)
    append_retry("gate-only", 0, None if not bad else "gate items failed")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _void(code: str, reason: str, access_began: bool, arms_done: list) -> None:
    wj(VOID, {"schema_version": 1, "utc": utc_now(), "void": True, "void_reason_code": code,
              "void_reason": reason, "evaluation_access_had_begun": access_began,
              "arms_completed": arms_done, "endpoints_computed": False,
              "automatic_retry_attempted": False,
              "runner_sha256": sha256_file(SELF), "freeze_sha256": sha256_file(IMPL_FREEZE)})


def run_formal() -> None:
    # ---- authorization lock (all six conditions) ----
    if os.environ.get(APPROVAL_ENV, "") != APPROVAL_TOKEN:
        raise SystemExit(f"REFUSED: {APPROVAL_ENV} must equal {APPROVAL_TOKEN}. "
                         "No evaluation payload has been opened.")
    impl = json.loads(IMPL_FREEZE.read_text())
    head = git("rev-parse", "HEAD")
    if head != impl["frozen_implementation_head"]:
        raise SystemExit(f"REFUSED: HEAD {head} != frozen implementation HEAD "
                         f"{impl['frozen_implementation_head']}. No evaluation payload opened.")
    if git("status", "--porcelain"):
        raise SystemExit("REFUSED: worktree not clean. No evaluation payload opened.")
    div = git("rev-list", "--left-right", "--count", f"HEAD...origin/{impl['branch']}")
    if div.split() != ["0", "0"]:
        raise SystemExit(f"REFUSED: divergence {div} != 0 0. No evaluation payload opened.")
    fz = Frozen()
    fz.verify()          # includes empty-namespace and hash checks

    import torch
    plan = arm_plan(fz)
    arms_done: list[str] = []
    tmp = OUT / "formal_records.partial.jsonl"
    started = time.perf_counter()
    model = tokenizer = device = None
    try:
        model, tokenizer, device, env = fz.v1.load_model()
        env["chunk_checkpointed_attention"] = fz.v2.install_chunk_checkpointed_attention(fz.v1)

        # ---- EVALUATION ACCESS BOUNDARY ----
        wj(ACCESS_MARKER, {"schema_version": 1, "utc": utc_now(), "git_head": head,
                           "runner_sha256": sha256_file(SELF),
                           "execution_freeze_sha256": sha256_file(IMPL_FREEZE),
                           "environment_hash": hashlib.sha256(
                               json.dumps(env, sort_keys=True, default=str).encode()).hexdigest(),
                           "invocation_uuid": str(uuid.uuid4()),
                           "note": "evaluation access begins after this marker; any failure now VOIDs the attempt"})
        acc = EvaluationAccess(fz)
        eval_rows = list(acc.rows())

        # ---- baseline over all evaluation rows ----
        base_by_id, order = {}, []
        modules = {"obs": model.model.layers[fz.L_OBS].mlp}
        A26 = []
        for n, (pi, sample) in enumerate(eval_rows):
            scored, captured = fz.v1.score_sample(model, tokenizer, device, sample,
                                                  {"obs": modules["obs"]})
            rec = {"project_index": pi, "sample_id": sample["uuid"], "gold": sample["correct_answer"],
                   "scores": scored["scores"], "prediction": scored["prediction"]}
            base_by_id[rec["sample_id"]] = rec
            order.append(rec["sample_id"])
            A26.append(captured["obs"].astype(np.float32))
        A26 = np.stack(A26)
        arms_done.append("baseline")

        rp = fz.router_prob(A26)
        routed_ids = [order[i] for i in range(len(order))
                      if base_by_id[order[i]]["prediction"] == SOURCE and rp[i] >= fz.tau]
        pes_ids = [order[i] for i in range(len(order)) if base_by_id[order[i]]["prediction"] == SOURCE]
        prob_by_id = {order[i]: float(rp[i]) for i in range(len(order))}
        err_ids = [s for s in order if base_by_id[s]["gold"] == GOLD
                   and base_by_id[s]["prediction"] == SOURCE]
        correct_ids = [s for s in order if base_by_id[s]["prediction"] == base_by_id[s]["gold"]]
        sample_by_id = {s["uuid"]: s for _, s in eval_rows}

        shim = _shim(fz)
        arm_eff, arm_recs, global_order = {}, {}, 0
        with tmp.open("w") as fh:
            for name, direction, layer, popsel, dirkey in plan:
                if name == "baseline":
                    continue
                pop = routed_ids if popsel == "routed" else pes_ids
                if name == "score_space_comparator":
                    eff = {}
                    recs = []
                    for sid in pop:
                        b = base_by_id[sid]
                        e = dict(b["scores"])
                        e[GOLD] += fz.b_c / 2.0
                        e[SOURCE] -= fz.b_c / 2.0
                        pred = sorted(((v, k) for k, v in e.items()), reverse=True)[0][1]
                        eff[sid] = pred
                        rec = {"sample_id": sid, "gold": b["gold"], "scores_base": b["scores"],
                               "pred_base": b["prediction"], "router_prob": prob_by_id[sid],
                               "router_decision": sid in routed_ids, "arm": name,
                               "direction_seed_or_hash": f"frozen_b_c={fz.b_c!r}", "q": fz.q,
                               "s_c": fz.s_c, "achieved_delta_norm": 0.0, "scores_int": e,
                               "pred_int": pred, "destination": f"{b['prediction']}->{pred}",
                               "batch_index": 0, "global_execution_order": global_order}
                        global_order += 1
                        recs.append(rec); fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    arm_eff[name] = eff; arm_recs[name] = recs; arms_done.append(name)
                    continue
                delta, achieved = (None, 0.0) if direction is None else build_delta(direction, fz.abs_norm)
                dhash = ("none" if direction is None
                         else hashlib.sha256(np.ascontiguousarray(direction).tobytes()).hexdigest())
                eff, recs = {}, []
                for sid in pop:
                    sc, pred, _, _ = fz.s2a.score_with_intervention(
                        shim, model, tokenizer, device, sample_by_id[sid], layer, delta)
                    b = base_by_id[sid]
                    eff[sid] = pred
                    rec = {"sample_id": sid, "gold": b["gold"], "scores_base": b["scores"],
                           "pred_base": b["prediction"], "router_prob": prob_by_id[sid],
                           "router_decision": sid in routed_ids, "arm": name,
                           "direction_seed_or_hash": dhash, "q": fz.q, "s_c": fz.s_c,
                           "achieved_delta_norm": achieved, "scores_int": sc, "pred_int": pred,
                           "destination": f"{b['prediction']}->{pred}", "batch_index": 0,
                           "global_execution_order": global_order}
                    global_order += 1
                    recs.append(rec); fh.write(json.dumps(rec, sort_keys=True) + "\n")
                arm_eff[name] = eff; arm_recs[name] = recs; arms_done.append(name)
                if name == "zero":
                    for rec in recs:
                        b = base_by_id[rec["sample_id"]]
                        if any(rec["scores_int"][m] != b["scores"][m] for m in fz.MODES) \
                           or rec["pred_int"] != b["prediction"]:
                            _void("ZERO_CONTROL_FAILURE",
                                  f"zero arm differs from baseline at {rec['sample_id']}", True, arms_done)
                            raise SystemExit("VOID: zero-control equality failed; endpoints not computed")

        expected = len(plan) - 1
        if len(arms_done) - 1 != expected:
            _void("INCOMPLETE_ALL_ARM_EXECUTION",
                  f"{len(arms_done)-1} arms completed, expected {expected}", True, arms_done)
            raise SystemExit("VOID: incomplete all-arm execution")
        os.replace(tmp, RECORDS)

        real = endpoints(fz, base_by_id, arm_eff["d_grad"], err_ids, correct_ids)
        rand = [endpoints(fz, base_by_id, arm_eff[f"formal_random_{i:03d}"], err_ids, correct_ids)
                for i in range(fz.K)]
        ge = sum(1 for r in rand if (r["target_gain_rate"] or 0) >= (real["target_gain_rate"] or 0))
        p_add_one = (1 + ge) / (fz.K + 1)
        others = {n: endpoints(fz, base_by_id, arm_eff[n], err_ids, correct_ids)
                  for n in ("zero", "d_L26_diffmean", "reverse_d_grad", "wrong_layer_d_grad",
                            "ungated_d_grad", "score_space_comparator")}
        cond = {
            "1_support_ge_30": real["n_channel_error"] >= 30,
            "2_tgr_point_gt_0": (real["target_gain_rate"] or 0) > 0,
            "3_tgr_ci_lower_gt_0": (real["target_gain_rate_ci95"][0] or -1) > 0,
            "4_p_add_one_le_0.05": p_add_one <= 0.05,
            "5_target_hit_gt_0.50": (real["target_hit"] or 0) > 0.50,
            "6_target_hit_ci_lower_gt_0.50": (real["target_hit_ci95"][0] or -1) > 0.50,
            "7_collateral_le_0.05": (real["clean_collateral_rate"] if real["clean_collateral_rate"]
                                     is not None else 1) <= 0.05,
            "8_collateral_ci_upper_le_0.05": (real["clean_collateral_rate_ci95"][1]
                                              if real["clean_collateral_rate_ci95"][1] is not None else 1) <= 0.05,
            "9_zero_control_exact": True,
            "10_no_structural_failure": True,
        }
        confirmed = all(cond.values())
        outcome = ("FORMAL_SUPPORT_INSUFFICIENT_NO_CONFIRMATORY_CLAIM"
                   if real["n_channel_error"] < 30 else
                   "FORMAL_CONFIRMATORY_SUCCESS" if confirmed else "FORMAL_CONFIRMATORY_NOT_ESTABLISHED")
        ir = fz.prereg["interpretation_rules"]
        if outcome == "FORMAL_CONFIRMATORY_SUCCESS":
            branch = ir["full_conjunction_passes"]
        elif p_add_one <= 0.05 and not (cond["5_target_hit_gt_0.50"] and cond["7_collateral_le_0.05"]):
            branch = ir["real_beats_random_but_target_hit_or_collateral_fails"]
        elif (real["target_gain_rate"] or 0) > 0 and p_add_one > 0.05:
            branch = ir["positive_behaviour_but_not_beating_random"]
        else:
            branch = ir["no_generalization"]
        sc_cmp = others["score_space_comparator"]
        if (sc_cmp["target_gain_rate"] or 0) >= (real["target_gain_rate"] or 0):
            branch += " " + ir["score_comparator_matches_or_exceeds"]
        wj(RESULTS, {"schema_version": 1, "utc": utc_now(), "outcome": outcome,
                     "primary": {"real": real, "p_add_one": p_add_one, "randoms_ge_real": ge, "K": fz.K},
                     "primary_conjunction": cond, "confirmed": confirmed,
                     "secondary": others,
                     "score_changes": {n: score_changes(fz, base_by_id, arm_recs[n], set(err_ids))
                                       for n in arm_recs},
                     "random_distribution": [r["target_gain_rate"] for r in rand],
                     "frozen_interpretation": branch,
                     "runtime_seconds": time.perf_counter() - started,
                     "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))})
        REPORT.write_text(f"# Qwen3 Stage 2 formal result\n\nOutcome: **{outcome}**\n\n"
                          f"{branch}\n", encoding="utf-8")
        append_retry("run-formal", 0, outcome)
        print(outcome)
    except SystemExit:
        raise
    except Exception as exc:
        if ACCESS_MARKER.exists():
            _void("INCOMPLETE_ALL_ARM_EXECUTION", f"{type(exc).__name__}: {exc}", True, arms_done)
        append_retry("run-formal", 1, f"{type(exc).__name__}: {exc}")
        raise


def _shim(fz):
    """Minimal object exposing the attributes score_with_intervention needs."""
    class S:
        MODES = fz.MODES
        v1 = fz.v1
    return S()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("preflight", "dev-engineering-check", "run-formal"):
        g.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    mode = ("preflight" if a.preflight else
            "dev-engineering-check" if a.dev_engineering_check else "run-formal")
    fn = {"preflight": preflight, "dev-engineering-check": dev_engineering_check,
          "run-formal": run_formal}[mode]
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
