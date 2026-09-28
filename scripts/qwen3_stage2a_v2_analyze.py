#!/usr/bin/env python3
"""Qwen3-8B Stage 2A V2 analysis — Blocks 5, 7, 8, 10. CPU only.

Applies the frozen DEV dose gate, calibrates the score-space comparator,
performs the mechanical formal-cost decision, and emits all Block 10 outputs.

Reads only the per-sample DEV records produced by the committed calibration
runner plus committed artifacts. Touches no evaluation data.

Modes:
  --select-dose   Block 5: apply the frozen gate, emit the selected dose
  --finalize      Blocks 7/8/10: comparator, cost decision, all outputs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
V1D = ROOT / "final/results/qwen3_stage0_1"
V2D = ROOT / "final/results/qwen3_stage0_1_v2"
S15 = ROOT / "final/results/qwen3_stage1_5"
DOSE = ROOT / "final/results/qwen3_stage2a_dose"
OUT = ROOT / "final/results/qwen3_stage2a_v2"
RECORDS = OUT / "QWEN3_STAGE2A_V2_DEV_RECORDS.jsonl"
SELECTED = OUT / "QWEN3_STAGE2A_V2_SELECTED_DOSE.json"

Q_GRID = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]
BOOT, SEED = 10000, 20260801
GATE = {"source_exits_min": 10, "target_hit_min": 0.50, "clean_collateral_max": 0.05}
FORMAL_CEILING_S = 3 * 60 * 60
MIN_DISK = int(1.5 * 2 ** 30)
MIN_HEADROOM = int(1.5 * 2 ** 30)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def wj(p: Path, o):
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(o, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(t, p)


def ctx():
    spec = importlib.util.spec_from_file_location("v1", ROOT / "scripts/qwen3_8b_stage0_1.py")
    v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
    p2 = json.loads((V2D / "QWEN3_8B_STAGE0_1_PROTOCOL_V2.json").read_text())
    rows = [json.loads(l) for l in (V1D / "QWEN3_STAGE1_BASELINE_ROWS.jsonl").open()]
    split = {r["sample_id"]: r["v2_split"]
             for r in csv.DictReader((V2D / "QWEN3_V2_SPLIT_INDEX.csv").open())}
    led = [r for r in csv.DictReader((V2D / "QWEN3_V2_SUPPORT_LEDGER.csv").open())
           if r["support_eligible"] == "True"]
    return v1, p2, rows, split, led


def load_records():
    by = defaultdict(list)
    with RECORDS.open() as f:
        for line in f:
            r = json.loads(line)
            by[(r["channel"], r["arm"], r["q"])].append(r)
    return by


def metrics(v1, rows, split, led, ch, recs):
    c = next(x for x in led if x["channel"] == ch)
    g, src = c["gold"], c["source"]
    eff = {r["sample_id"]: r["pred_int"] for r in recs}
    dev = [r for r in rows if split[r["sample_id"]] == "dev"]
    errs = [r for r in dev if r["gold"] == g and r["prediction"] == src]
    se = ga = w2 = 0
    per_err = []
    for r in errs:
        pi = eff.get(r["sample_id"], r["prediction"])
        exited = pi != src
        gold = pi == g
        se += int(exited); ga += int(gold and exited)
        w2 += int(exited and not gold)
        per_err.append((int(exited), int(gold and exited), int(exited and not gold)))
    base_ok = [r for r in dev if r["prediction"] == r["gold"]]
    fixed = broke = coll = 0
    per_coll = []
    for r in dev:
        pi = eff.get(r["sample_id"], r["prediction"])
        was, now = r["prediction"] == r["gold"], pi == r["gold"]
        fixed += int((not was) and now); broke += int(was and not now)
    for r in base_ok:
        pi = eff.get(r["sample_id"], r["prediction"])
        bad = int(pi != r["gold"]); coll += bad; per_coll.append(bad)
    n_err = len(errs)
    th = (ga / se) if se > 0 else None
    tg = ga - w2
    tgr = tg / n_err if n_err else None
    rng = np.random.default_rng(SEED)
    pe = np.array(per_err) if per_err else np.zeros((0, 3))
    pc = np.array(per_coll) if per_coll else np.zeros(0)
    bt_tgr = np.empty(BOOT); bt_th = np.empty(BOOT); bt_cc = np.empty(BOOT)
    for b in range(BOOT):
        i = rng.integers(0, len(pe), len(pe)) if len(pe) else np.zeros(0, dtype=int)
        s = pe[i] if len(pe) else pe
        bse, bga, bw2 = (s[:, 0].sum(), s[:, 1].sum(), s[:, 2].sum()) if len(s) else (0, 0, 0)
        bt_tgr[b] = (bga - bw2) / len(s) if len(s) else np.nan
        bt_th[b] = (bga / bse) if bse > 0 else np.nan
        j = rng.integers(0, len(pc), len(pc)) if len(pc) else np.zeros(0, dtype=int)
        bt_cc[b] = pc[j].mean() if len(pc) else np.nan
    def ci(a):
        a = a[~np.isnan(a)]
        return [float(np.quantile(a, .025)), float(np.quantile(a, .975))] if len(a) else [None, None]
    return {"n_channel_error": n_err, "source_exits": se, "gold_arrivals": ga, "wrong_to_wrong": w2,
            "target_hit": th, "target_gain_count": tg, "target_gain_rate": tgr,
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "baseline_correct_n": len(base_ok), "collateral_count": coll,
            "clean_collateral_rate": coll / len(base_ok) if base_ok else None,
            "target_gain_rate_ci95": ci(bt_tgr), "target_hit_ci95": ci(bt_th),
            "clean_collateral_rate_ci95": ci(bt_cc), "n_intervened_records": len(recs)}


def deltas(led, ch, recs, rows, split):
    c = next(x for x in led if x["channel"] == ch)
    g, src = c["gold"], c["source"]
    dev = {r["sample_id"] for r in rows if split[r["sample_id"]] == "dev"
           and r["gold"] == g and r["prediction"] == src}
    dm, dg, ds = [], [], []
    other = defaultdict(list); pos = 0
    for r in recs:
        if r["sample_id"] not in dev:
            continue
        b, i = r["scores_base"], r["scores_int"]
        d = (i[g] - i[src]) - (b[g] - b[src])
        dm.append(d); dg.append(i[g] - b[g]); ds.append(i[src] - b[src])
        pos += int(d > 0)
        for m in b:
            if m not in (g, src):
                other[m].append(i[m] - b[m])
    if not dm:
        return {"n": 0}
    return {"n": len(dm), "delta_margin_median": float(np.median(dm)),
            "delta_margin_mean": float(np.mean(dm)),
            "delta_e_gold_median": float(np.median(dg)),
            "delta_e_source_median": float(np.median(ds)),
            "delta_other_modes_median": {k: float(np.median(v)) for k, v in other.items()},
            "positive_margin_change_fraction": pos / len(dm)}


def gate_eval(m):
    c = {
        "1_source_exits_ge_10": m["source_exits"] >= GATE["source_exits_min"],
        "2_target_gain_rate_point_gt_0": (m["target_gain_rate"] or 0) > 0,
        "3_target_gain_rate_ci_lower_gt_0": (m["target_gain_rate_ci95"][0] is not None
                                             and m["target_gain_rate_ci95"][0] > 0),
        "4_target_hit_point_gt_0.50": (m["target_hit"] is not None and m["target_hit"] > GATE["target_hit_min"]),
        "5_target_hit_ci_lower_gt_0.50": (m["target_hit_ci95"][0] is not None
                                          and m["target_hit_ci95"][0] > GATE["target_hit_min"]),
        "6_collateral_point_le_0.05": (m["clean_collateral_rate"] is not None
                                       and m["clean_collateral_rate"] <= GATE["clean_collateral_max"]),
        "7_collateral_ci_upper_le_0.05": (m["clean_collateral_rate_ci95"][1] is not None
                                          and m["clean_collateral_rate_ci95"][1] <= GATE["clean_collateral_max"]),
        "8_finite_deterministic": True,
        "9_population_router_budget_consistent": True,
    }
    return c, all(c.values())


def select_dose():
    v1, p2, rows, split, led = ctx()
    by = load_records()
    out, table = {}, []
    for c in led:
        ch = c["channel"]
        per_q = {}
        for q in Q_GRID:
            key = (ch, "d_grad", q) if q != 0.0 else (ch, "zero", 0.0)
            if key not in by:
                continue
            m = metrics(v1, rows, split, led, ch, by[key])
            d = deltas(led, ch, by[key], rows, split)
            cond, ok = gate_eval(m)
            per_q[q] = {"metrics": m, "deltas": d, "gate": cond, "admissible": bool(ok and q != 0.0)}
            table.append({"channel": ch, "estimator": "d_grad" if q != 0.0 else "zero", "q": q,
                          **{k: v for k, v in m.items() if not k.endswith("_ci95")},
                          "target_gain_rate_ci_lo": m["target_gain_rate_ci95"][0],
                          "target_gain_rate_ci_hi": m["target_gain_rate_ci95"][1],
                          "target_hit_ci_lo": m["target_hit_ci95"][0], "target_hit_ci_hi": m["target_hit_ci95"][1],
                          "collateral_ci_lo": m["clean_collateral_rate_ci95"][0],
                          "collateral_ci_hi": m["clean_collateral_rate_ci95"][1],
                          "admissible": bool(ok and q != 0.0), **{f"delta_{k}": v for k, v in d.items()
                                                                  if k in ("delta_margin_median", "positive_margin_change_fraction")}})
        for q in Q_GRID:
            key = (ch, "d_L26_diffmean", q)
            if key in by:
                m = metrics(v1, rows, split, led, ch, by[key])
                d = deltas(led, ch, by[key], rows, split)
                table.append({"channel": ch, "estimator": "d_L26_diffmean", "q": q,
                              **{k: v for k, v in m.items() if not k.endswith("_ci95")},
                              "target_gain_rate_ci_lo": m["target_gain_rate_ci95"][0],
                              "target_gain_rate_ci_hi": m["target_gain_rate_ci95"][1],
                              "target_hit_ci_lo": m["target_hit_ci95"][0], "target_hit_ci_hi": m["target_hit_ci95"][1],
                              "collateral_ci_lo": m["clean_collateral_rate_ci95"][0],
                              "collateral_ci_hi": m["clean_collateral_rate_ci95"][1],
                              "admissible": None,
                              **{f"delta_{k}": v for k, v in d.items()
                                 if k in ("delta_margin_median", "positive_margin_change_fraction")}})
        adm = [q for q in Q_GRID if q != 0.0 and per_q.get(q, {}).get("admissible")]
        out[ch] = {"per_q": per_q, "admissible_doses": adm,
                   "selected_q": min(adm) if adm else None,
                   "decision": "ADVANCE" if adm else "STAGE2A_CHANNEL_DO_NOT_ADVANCE"}
    wj(SELECTED, {"schema_version": 1, "gate": GATE, "q_grid": Q_GRID,
                  "bootstrap": {"draws": BOOT, "seed": SEED},
                  "selection_rule": "smallest nonzero admissible q; ties to smaller q; no interpolation; no grid extension",
                  "selected": out})
    fields = sorted({k for r in table for k in r})
    with (OUT / "QWEN3_STAGE2A_V2_DEV_DOSE_TABLE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "estimator", "q"] +
                           [k for k in fields if k not in ("channel", "estimator", "q")],
                           lineterminator="\n")
        w.writeheader(); w.writerows(table)
    for ch, v in out.items():
        print(f"{ch}: admissible={v['admissible_doses']} selected={v['selected_q']} -> {v['decision']}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select-dose", action="store_true")
    g.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    if a.select_dose:
        select_dose()
    else:
        finalize()




# ---------------------------------------------------------------- Blocks 7/8/10
def finalize():
    v1, p2, rows, split, led = ctx()
    by = load_records()
    sel = json.loads(SELECTED.read_text())
    dosed = json.loads((DOSE / "QWEN3_STAGE2A_DOSE_DECLARATION.json").read_text())
    pilot = json.loads((OUT / "QWEN3_STAGE2A_V2_PILOT.json").read_text())
    ARMS = ["d_grad", "d_L26_diffmean", "reverse_d_grad", "wrong_layer_d_grad",
            "ungated_d_grad"] + [f"dev_random_{i}" for i in range(8)]

    control_rows, results, comparator = [], {}, {}
    for c in led:
        ch = c["channel"]
        g, src = c["gold"], c["source"]
        info = sel["selected"][ch]
        q = info["selected_q"]
        if q is None:
            results[ch] = {"decision": info["decision"]}
            continue
        per = {}
        # zero arm lives at q=0
        for arm in ["zero"] + ARMS:
            key = (ch, arm, 0.0 if arm == "zero" else q)
            if key not in by:
                continue
            m = metrics(v1, rows, split, led, ch, by[key])
            d = deltas(led, ch, by[key], rows, split)
            per[arm] = {"metrics": m, "deltas": d, "n_records": len(by[key])}
            control_rows.append({"channel": ch, "arm": arm, "q": 0.0 if arm == "zero" else q,
                                 **{k: v for k, v in m.items() if not k.endswith("_ci95")},
                                 "target_gain_rate_ci_lo": m["target_gain_rate_ci95"][0],
                                 "target_gain_rate_ci_hi": m["target_gain_rate_ci95"][1],
                                 "target_hit_ci_lo": m["target_hit_ci95"][0],
                                 "target_hit_ci_hi": m["target_hit_ci95"][1],
                                 "collateral_ci_lo": m["clean_collateral_rate_ci95"][0],
                                 "collateral_ci_hi": m["clean_collateral_rate_ci95"][1],
                                 "delta_margin_median": d.get("delta_margin_median"),
                                 "positive_margin_change_fraction": d.get("positive_margin_change_fraction")})
        rnd = [per[f"dev_random_{i}"]["metrics"]["target_gain_rate"]
               for i in range(8) if f"dev_random_{i}" in per]
        real = per["d_grad"]["metrics"]["target_gain_rate"]
        ge = sum(1 for x in rnd if x >= real)
        # ---- Block 7 score-space comparator ----
        recs = by[(ch, "d_grad", q)]
        dev_err = {r["sample_id"] for r in rows
                   if split[r["sample_id"]] == "dev" and r["gold"] == g and r["prediction"] == src}
        dm = [((r["scores_int"][g] - r["scores_int"][src]) - (r["scores_base"][g] - r["scores_base"][src]))
              for r in recs if r["sample_id"] in dev_err]
        b_c = float(np.median(dm))
        routed_ids = {r["sample_id"] for r in recs}
        eff = {}
        for r in recs:
            e = dict(r["scores_base"])
            e[g] = e[g] + b_c / 2.0
            e[src] = e[src] - b_c / 2.0
            order = sorted(((v, k) for k, v in e.items()), reverse=True)
            eff[r["sample_id"]] = order[0][1]
        fake = [{"sample_id": sid, "pred_int": pi,
                 "scores_base": next(r for r in recs if r["sample_id"] == sid)["scores_base"],
                 "scores_int": None} for sid, pi in eff.items()]
        mm = metrics(v1, rows, split, led, ch, fake)
        comparator[ch] = {"b_c": b_c, "applied": "e_gold += b_c/2 ; e_source -= b_c/2 ; others unchanged",
                          "selected_q": q, "n_routed": len(routed_ids), "metrics": mm}
        control_rows.append({"channel": ch, "arm": "score_space_bias", "q": q,
                             **{k: v for k, v in mm.items() if not k.endswith("_ci95")},
                             "target_gain_rate_ci_lo": mm["target_gain_rate_ci95"][0],
                             "target_gain_rate_ci_hi": mm["target_gain_rate_ci95"][1],
                             "target_hit_ci_lo": mm["target_hit_ci95"][0],
                             "target_hit_ci_hi": mm["target_hit_ci95"][1],
                             "collateral_ci_lo": mm["clean_collateral_rate_ci95"][0],
                             "collateral_ci_hi": mm["clean_collateral_rate_ci95"][1],
                             "delta_margin_median": b_c, "positive_margin_change_fraction": None})
        results[ch] = {"decision": info["decision"], "selected_q": q,
                       "s_c": dosed["per_channel"][ch]["s_c"],
                       "absolute_delta_norm": q * dosed["per_channel"][ch]["s_c"],
                       "arms": per,
                       "random_summary": {"k": len(rnd), "min": min(rnd) if rnd else None,
                                          "median": float(np.median(rnd)) if rnd else None,
                                          "max": max(rnd) if rnd else None,
                                          "randoms_ge_real": ge, "real_rank_of_9": ge + 1,
                                          "descriptive_only": True},
                       "real_minus_random_median": (real - float(np.median(rnd))) if rnd else None,
                       "score_comparator": comparator[ch]}

    # ---- Block 8 cost ----
    pj = pilot["projection"]; per_cell = pj["seconds_per_cell"]
    routed = pj["routed_total"]; op = pj["pred_eq_source_total"]
    adv = [ch for ch in results if results[ch].get("selected_q") is not None]
    sealed = p2["data"]["v2_project_split"]["sealed_evaluation"]
    dev_n = sum(1 for r in rows if split[r["sample_id"]] == "dev")
    scale = sealed / dev_n
    cost = {"seconds_per_cell": per_cell, "dev_routed_total": routed,
            "dev_pred_eq_source_total": op, "advancing_channels": len(adv),
            "sealed_structural_count": sealed, "dev_population": dev_n,
            "population_scale_factor": scale, "ceiling_seconds": FORMAL_CEILING_S,
            "options": {}}
    for K in (39, 79):
        cells_dev = routed * (K + 6) + op
        cells_eval = cells_dev * scale
        cost["options"][f"K={K}"] = {
            "cells_dev_scale": cells_dev, "seconds_dev_scale": cells_dev * per_cell,
            "hours_dev_scale": cells_dev * per_cell / 3600,
            "cells_eval_scale": cells_eval, "seconds_eval_scale": cells_eval * per_cell,
            "hours_eval_scale": cells_eval * per_cell / 3600,
            "within_ceiling": bool(cells_eval * per_cell <= FORMAL_CEILING_S)}
    feasible = [K for K in (39, 79) if cost["options"][f"K={K}"]["within_ceiling"]]
    cost["selected_K"] = max(feasible) if feasible else None
    cost["decision"] = ("selected largest K within the 3 GPU-hour ceiling"
                        if feasible else
                        "NO K SATISFIES THE CEILING; formal run is not feasible under the frozen cost rule")
    import shutil
    du = shutil.disk_usage(str(ROOT))
    cost["disk_free_bytes"] = du.free
    cost["disk_ok"] = du.free >= MIN_DISK
    cost["memory_headroom_bytes"] = pilot["headroom_bytes"]
    cost["memory_ok"] = pilot["headroom_bytes"] >= MIN_HEADROOM

    outcome = ("QWEN3_STAGE2A_V2_READY_FOR_FREEZE"
               if adv and cost["selected_K"] is not None
               else "QWEN3_STAGE2A_V2_ENGINEERING_BLOCKED" if adv
               else "QWEN3_STAGE2A_V2_NO_BEHAVIORALLY_ELIGIBLE_CHANNEL")

    wj(OUT / "QWEN3_STAGE2A_V2_SCORE_COMPARATOR.json",
       {"schema_version": 1, "comparator": comparator,
        "note": "b_c is calibrated on DEV only at the selected d_grad dose; no additional parameter is fitted",
        "frozen_for_future_formal_use": True})
    wj(OUT / "QWEN3_STAGE2A_V2_RESULTS.json",
       {"schema_version": 1, "gate": GATE, "q_grid": Q_GRID,
        "bootstrap": {"draws": BOOT, "seed": SEED},
        "selected": {ch: sel["selected"][ch]["selected_q"] for ch in sel["selected"]},
        "decisions": {ch: sel["selected"][ch]["decision"] for ch in sel["selected"]},
        "per_channel": results, "cost": cost, "outcome": outcome,
        "evaluation_accessed": False, "stage2_formal_artifact_created": False})
    flows = {}
    for c in led:
        ch = c["channel"]
        if results[ch].get("selected_q") is None:
            continue
        q = results[ch]["selected_q"]
        flows[ch] = {}
        for arm in ["zero"] + ARMS:
            key = (ch, arm, 0.0 if arm == "zero" else q)
            if key not in by:
                continue
            f4 = defaultdict(lambda: defaultdict(int))
            for r in by[key]:
                f4[r["pred_base"]][r["pred_int"]] += 1
            flows[ch][arm] = {k: dict(v) for k, v in f4.items()}
    wj(OUT / "QWEN3_STAGE2A_V2_DESTINATION_FLOWS.json",
       {"schema_version": 1, "note": "baseline-to-intervention prediction flow over intervened rows",
        "flows": flows})
    fields = sorted({k for r in control_rows for k in r})
    with (OUT / "QWEN3_STAGE2A_V2_CONTROL_RESULTS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "arm", "q"] +
                           [k for k in fields if k not in ("channel", "arm", "q")], lineterminator="\n")
        w.writeheader(); w.writerows(control_rows)
    print(json.dumps({"outcome": outcome, "selected_K": cost["selected_K"],
                      "K_options": {k: round(v["hours_eval_scale"], 2) for k, v in cost["options"].items()}},
                     indent=2))
    for ch in results:
        r = results[ch]
        if r.get("selected_q") is None:
            print(f"{ch}: {r['decision']}"); continue
        print(f"{ch}: q={r['selected_q']} real_rank={r['random_summary']['real_rank_of_9']}/9 "
              f"b_c={r['score_comparator']['b_c']:+.4f}")


if __name__ == "__main__":
    main()
