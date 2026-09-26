#!/usr/bin/env python3
"""Stage E — frozen DEV intervention.  No SEALED access.

Reuses the committed readout verbatim (`render_prompt`, `candidate_ids`, `score_candidate`)
and adds only the frozen intervention hook and destination bookkeeping, exactly as
`scripts/qwen3_stage2a_v2_dev_calibration.py` does for Qwen3-8B.

    h' = h + q * s_c * d   at every sequence position of layer L, matching the frozen
                           full-effect aggregation

Blocks:
  --dose-calibration  zero arm + {d_grad, L26 DiffMean} across the frozen q grid, on the
                      Router-gated DEV population
  --controls          at each channel's selected dose: reverse, wrong-layer, ungated,
                      the eight frozen DEV random directions, and the same-layer L21
                      DiffMean diagnostic

Only the frozen DEV random budget (8 development seeds) is used.  The K=59 formal random
null is not generated here and its seeds are disjoint from these by construction.
"""

import argparse
import json
import time

import numpy as np

import stage_d_e_common as C

RECORDS = C.HERE / "DEV_INTERVENTION_RECORDS.jsonl"


def make_hook(model, layer, delta):
    module = model.model.layers[layer].mlp

    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if delta is None:
            return out
        d = delta.to(dtype=t.dtype, device=t.device)
        new = t + d
        return (new,) + out[1:] if isinstance(out, tuple) else new

    return module.register_forward_hook(hook)


def score_with_intervention(ctx, sample, layer, delta):
    v1 = ctx.v1
    prompt, _, _ = v1.render_prompt(ctx.tok, sample)
    handle = make_hook(ctx.model, layer, delta)
    try:
        scores, lengths = {}, {}
        for m in v1.MODES:
            pids, cids = v1.candidate_ids(ctx.tok, prompt, sample["answers"][m])
            s, _ = v1.score_candidate(ctx.model, ctx.device, pids, cids, None)
            scores[m] = s
            lengths[m] = len(cids)
    finally:
        handle.remove()
    vals = np.asarray([scores[m] for m in v1.MODES], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("non-finite intervened score")
    order = np.argsort(-vals, kind="stable")
    return scores, v1.MODES[int(order[0])], v1.MODES[int(order[1])]


def run_arm(ctx, dose, ch, arm, q, direction, layer, population, fh):
    import torch
    s_c = dose["per_channel"][ch]["s_c"]
    absn = q * s_c
    delta, achieved = None, 0.0
    if direction is not None and q != 0.0:
        d64 = np.ascontiguousarray(direction.astype(np.float64))
        d64 = d64 / np.linalg.norm(d64)
        delta = torch.from_numpy((d64 * absn).astype(np.float32)).to("cuda:0")
        achieved = float(torch.linalg.vector_norm(delta.double()).item())
        rel = abs(achieved - absn) / max(1.0, absn)
        if rel > C.PERTURBATION_RTOL:
            raise RuntimeError("perturbation norm mismatch: %r vs %r (rel %.3e)"
                               % (achieved, absn, rel))
    n = 0
    for i, r in enumerate(population):
        sc, pred, runner = score_with_intervention(
            ctx, ctx.sample_of[r["project_index"]], layer, delta)
        fh.write(json.dumps({
            "channel": ch, "arm": arm, "q": q, "abs_delta_norm": absn,
            "achieved_delta_norm": achieved, "layer": layer,
            "sample_id": r["sample_id"], "project_index": r["project_index"],
            "gold": r["gold"], "pred_base": r["prediction"],
            "scores_base": r["scores"], "scores_int": sc,
            "pred_int": pred, "runner_up_int": runner, "execution_order": i,
        }, sort_keys=True) + "\n")
        n += 1
    fh.flush()
    return n


def directions(ctx):
    from safetensors import safe_open
    dg = {}
    dm = {"L26_diffmean_frozen": {}, "L21_diffmean_committed": {}}
    with safe_open(C.HERE / "PRIMARY_DIRECTIONS.safetensors", framework="numpy") as f:
        for k in f.keys():
            dg[k[len("d_grad__"):]] = f.get_tensor(k)
    with safe_open(C.HERE / "DIFFMEAN_DIRECTIONS.safetensors", framework="numpy") as f:
        for k in f.keys():
            for name in dm:
                pref = name + "__"
                if k.startswith(pref):
                    dm[name][k[len(pref):]] = f.get_tensor(k)
    for ch, _, _ in C.CHANNELS:
        for v in [dg[ch]] + [dm[n][ch] for n in dm]:
            nn = float(np.linalg.norm(v.astype(np.float64)))
            if not np.all(np.isfinite(v)) or abs(nn - 1.0) > 1e-5 or v.shape != (C.HIDDEN,):
                raise RuntimeError("direction not finite/unit/shaped for %s" % ch)
    return dg, dm


def qualified_channels():
    g = json.loads((C.HERE / "_GEOMETRY_RESULTS.json").read_text())
    return [ch for ch in g["qualified"]], g


def preflight(ctx, dose):
    """Verification only; no intervention."""
    checks = {"frozen_populations": {}, "router_reproduces_stage_c": True}
    import csv as _csv
    audit = {r["channel"].replace("->", "__to__"): r
             for r in _csv.DictReader(open(C.PROSP / "ROUTER_AUDIT.csv", encoding="utf-8"))}
    for ch, g, s in C.CHANNELS:
        routed, pop, pr = ctx.routed(ch)
        a = audit[ch]
        if len(routed) != int(a["tau_fired"]) or len(pop) != int(a["tau_population"]):
            raise RuntimeError("Router gate does not reproduce Stage C counts for %s" % ch)
        tp = sum(1 for r in routed if r["gold"] == g)
        if tp != int(a["tau_true_positives"]):
            raise RuntimeError("Router true-positive drift %s" % ch)
        checks["frozen_populations"][ch] = {
            "dev_pred_eq_source": len(pop), "dev_routed": len(routed),
            "dev_routed_channel_errors": tp,
            "dev_channel_errors_total": len(ctx.errors(ch, "dev")),
            "dev_routed_and_baseline_correct": sum(1 for r in routed
                                                   if r["prediction"] == r["gold"]),
            "dev_baseline_correct_total": sum(1 for r in ctx.split_rows("dev")
                                              if r["prediction"] == r["gold"]),
            "s_c": dose["per_channel"][ch]["s_c"], "tau": ctx.tau[ch]}
    return checks


def dose_calibration():
    import torch
    ctx = C.Ctx(need_model=True)
    dose = json.loads((C.HERE / "DOSE_DECLARATION.json").read_text())
    qual, _ = qualified_channels()
    dg, dm = directions(ctx)
    checks = preflight(ctx, dose)
    C.wj(C.HERE / "_STAGE_E_PREFLIGHT.json",
         {"status": "PASS", "checks": checks, "qualified_channels": qual,
          "intervention_run": False, "sealed_evaluation_accessed": False})

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    with open(RECORDS, "a", encoding="utf-8", newline="\n") as fh:
        for ch in qual:
            routed, _, _ = ctx.routed(ch)
            C.assert_sealed_clean(ctx, routed)
            n = run_arm(ctx, dose, ch, "zero", 0.0, None, C.L_INJ, routed, fh)
            print("%s zero q=0 n=%d  %.0fs" % (ch, n, time.perf_counter() - t0), flush=True)
            for q in C.Q_GRID[1:]:
                for arm, d in (("d_grad", dg[ch]),
                               ("d_L26_diffmean", dm["L26_diffmean_frozen"][ch])):
                    n = run_arm(ctx, dose, ch, arm, q, d, C.L_INJ, routed, fh)
                    print("%s %s q=%s n=%d  %.0fs peak %.2fGiB"
                          % (ch, arm, q, n, time.perf_counter() - t0,
                             torch.cuda.max_memory_allocated(0) / 2 ** 30), flush=True)
    C.wj(C.HERE / "_STAGE_E_DOSE_RUNTIME.json", {
        "elapsed_seconds": time.perf_counter() - t0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2 ** 30})
    print("DOSE_CALIBRATION_COMPLETE")


def controls():
    import torch
    ctx = C.Ctx(need_model=True)
    dose = json.loads((C.HERE / "DOSE_DECLARATION.json").read_text())
    sel = json.loads((C.HERE / "_SELECTED_DOSE.json").read_text())
    dg, dm = directions(ctx)
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    with open(RECORDS, "a", encoding="utf-8", newline="\n") as fh:
        for ch, info in sel["selected"].items():
            q = info["selected_q"]
            if q is None:
                print("%s: no admissible dose; controls skipped by the frozen rule" % ch)
                continue
            routed, pop, _ = ctx.routed(ch)
            C.assert_sealed_clean(ctx, pop)
            arms = [("reverse_d_grad", -dg[ch], C.L_INJ, routed),
                    ("wrong_layer_d_grad", dg[ch], C.WRONG_LAYER, routed),
                    ("ungated_d_grad", dg[ch], C.L_INJ, pop),
                    ("d_L21_same_layer_diffmean",
                     dm["L21_diffmean_committed"][ch], C.L_INJ, routed)]
            for i, sd in enumerate(C.DEV_RANDOM_SEEDS):
                arms.append(("dev_random_%d" % i, C.random_unit(sd), C.L_INJ, routed))
            for arm, d, layer, population in arms:
                n = run_arm(ctx, dose, ch, arm, q, d, layer, population, fh)
                print("%s %s q=%s n=%d  %.0fs" % (ch, arm, q, n, time.perf_counter() - t0),
                      flush=True)
    C.wj(C.HERE / "_STAGE_E_CONTROL_RUNTIME.json", {
        "elapsed_seconds": time.perf_counter() - t0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2 ** 30})
    print("CONTROLS_COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dose-calibration", action="store_true")
    g.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    (dose_calibration if a.dose_calibration else controls)()
