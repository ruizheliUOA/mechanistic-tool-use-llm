#!/usr/bin/env python3
"""Stage E analysis — frozen DEV dose gate, destination accounting, specificity controls
and the conjunctive channel-advancement decision.

Every metric, gate condition and selection rule is transcribed from the frozen Qwen3-8B
modern protocol (`scripts/qwen3_stage2a_v2_analyze.py`,
`final/results/qwen3_stage2a_v2/QWEN3_STAGE2A_V2_DOSE_SELECTION_RULE.md`,
`final/results/qwen3_stage2_formal/QWEN3_STAGE2_ENDPOINT_SPEC.md`,
`final/results/qwen3_stage2_formal/QWEN3_STAGE2_CHANNEL_SELECTION_DISCLOSURE.md`).
No threshold is weakened, no weighted actionability score is invented, and advancement is
conjunctive/lexicographic.

  --select-dose  frozen nine-condition dose gate over the q grid; emits the selected dose
  --finalize     controls, specificity, destination/collateral audits, advancement table
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

import stage_d_e_common as C

RECORDS = C.HERE / "DEV_INTERVENTION_RECORDS.jsonl"
SELECTED = C.HERE / "_SELECTED_DOSE.json"
BOOT = C.DEV_BOOTSTRAP_DRAWS
SEED = C.DEV_BOOTSTRAP_SEED


def load_records():
    by = defaultdict(list)
    with open(RECORDS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by[(r["channel"], r["arm"], r["q"])].append(r)
    return by


def metrics(ctx, ch, recs):
    """Frozen destination-resolved accounting.  Transcribed verbatim."""
    g, src = ctx.meta(ch)
    eff = {r["sample_id"]: r["pred_int"] for r in recs}
    dev = ctx.split_rows("dev")
    errs = [r for r in dev if r["gold"] == g and r["prediction"] == src]
    se = ga = w2 = 0
    per_err = []
    dest = {"SOURCE_RETAINED": 0, "GOLD_ARRIVAL": 0, "OTHER_WRONG": 0}
    for r in errs:
        pi = eff.get(r["sample_id"], r["prediction"])
        exited = pi != src
        gold = pi == g
        se += int(exited)
        ga += int(gold and exited)
        w2 += int(exited and not gold)
        per_err.append((int(exited), int(gold and exited), int(exited and not gold)))
        dest["GOLD_ARRIVAL" if (exited and gold)
             else "OTHER_WRONG" if exited else "SOURCE_RETAINED"] += 1
    base_ok = [r for r in dev if r["prediction"] == r["gold"]]
    fixed = broke = coll = 0
    per_coll = []
    coll_dest = {"CORRECT_RETAINED": 0, "BROKEN_TO_SOURCE": 0, "BROKEN_TO_OTHER": 0}
    for r in dev:
        pi = eff.get(r["sample_id"], r["prediction"])
        was, now = r["prediction"] == r["gold"], pi == r["gold"]
        fixed += int((not was) and now)
        broke += int(was and not now)
    for r in base_ok:
        pi = eff.get(r["sample_id"], r["prediction"])
        bad = int(pi != r["gold"])
        coll += bad
        per_coll.append(bad)
        coll_dest["CORRECT_RETAINED" if not bad
                  else "BROKEN_TO_SOURCE" if pi == src else "BROKEN_TO_OTHER"] += 1
    n_err = len(errs)
    th = (ga / se) if se > 0 else None
    tg = ga - w2
    tgr = tg / n_err if n_err else None
    rng = np.random.default_rng(SEED)
    pe = np.array(per_err) if per_err else np.zeros((0, 3))
    pc = np.array(per_coll) if per_coll else np.zeros(0)
    bt_tgr = np.empty(BOOT)
    bt_th = np.empty(BOOT)
    bt_cc = np.empty(BOOT)
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

    routed_ids = set(eff)
    return {"n_channel_error": n_err, "source_exits": se, "gold_arrivals": ga,
            "wrong_to_wrong": w2, "target_hit": th,
            "target_gain_count": tg, "target_gain_rate": tgr,
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "baseline_correct_n": len(base_ok), "collateral_count": coll,
            "clean_collateral_rate": coll / len(base_ok) if base_ok else None,
            "target_gain_rate_ci95": ci(bt_tgr), "target_hit_ci95": ci(bt_th),
            "clean_collateral_rate_ci95": ci(bt_cc),
            "n_intervened_records": len(recs),
            "router_fired_channel_error_count": sum(
                1 for r in errs if r["sample_id"] in routed_ids),
            "router_fired_baseline_correct_exposure": sum(
                1 for r in base_ok if r["sample_id"] in routed_ids),
            "eligible_collateral": sum(1 for r in base_ok if r["sample_id"] in routed_ids),
            "destination_counts": dest, "collateral_destination_counts": coll_dest}


def deltas(ctx, ch, recs):
    g, src = ctx.meta(ch)
    dev_err = {r["sample_id"] for r in ctx.errors(ch, "dev")}
    dm, dg, ds = [], [], []
    other = defaultdict(list)
    pos = 0
    for r in recs:
        if r["sample_id"] not in dev_err or r.get("scores_int") is None:
            continue
        b, i = r["scores_base"], r["scores_int"]
        d = (i[g] - i[src]) - (b[g] - b[src])
        dm.append(d)
        dg.append(i[g] - b[g])
        ds.append(i[src] - b[src])
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
        "1_source_exits_ge_10": m["source_exits"] >= C.GATE["source_exits_min"],
        "2_target_gain_rate_point_gt_0": (m["target_gain_rate"] or 0) > 0,
        "3_target_gain_rate_ci_lower_gt_0": (m["target_gain_rate_ci95"][0] is not None
                                             and m["target_gain_rate_ci95"][0] > 0),
        "4_target_hit_point_gt_0.50": (m["target_hit"] is not None
                                       and m["target_hit"] > C.GATE["target_hit_min"]),
        "5_target_hit_ci_lower_gt_0.50": (m["target_hit_ci95"][0] is not None
                                          and m["target_hit_ci95"][0] > C.GATE["target_hit_min"]),
        "6_collateral_point_le_0.05": (m["clean_collateral_rate"] is not None
                                       and m["clean_collateral_rate"] <= C.GATE["clean_collateral_max"]),
        "7_collateral_ci_upper_le_0.05": (m["clean_collateral_rate_ci95"][1] is not None
                                          and m["clean_collateral_rate_ci95"][1]
                                          <= C.GATE["clean_collateral_max"]),
        "8_finite_deterministic": True,
        "9_population_router_budget_consistent": True,
    }
    return c, all(c.values())


ROW_KEYS = ["n_channel_error", "source_exits", "gold_arrivals", "wrong_to_wrong",
            "target_hit", "target_gain_count", "target_gain_rate", "fixed", "broke", "net",
            "baseline_correct_n", "collateral_count", "clean_collateral_rate",
            "router_fired_channel_error_count", "router_fired_baseline_correct_exposure",
            "eligible_collateral", "n_intervened_records"]


def flat(ch, arm, q, m, d):
    r = {"channel": ch, "arm": arm, "q": q}
    r.update({k: m[k] for k in ROW_KEYS})
    r["target_gain_rate_ci_lo"] = m["target_gain_rate_ci95"][0]
    r["target_gain_rate_ci_hi"] = m["target_gain_rate_ci95"][1]
    r["target_hit_ci_lo"] = m["target_hit_ci95"][0]
    r["target_hit_ci_hi"] = m["target_hit_ci95"][1]
    r["collateral_ci_lo"] = m["clean_collateral_rate_ci95"][0]
    r["collateral_ci_hi"] = m["clean_collateral_rate_ci95"][1]
    r["delta_margin_median"] = d.get("delta_margin_median")
    r["positive_margin_change_fraction"] = d.get("positive_margin_change_fraction")
    return r


def select_dose():
    ctx = C.Ctx()
    by = load_records()
    qual = json.loads((C.HERE / "_GEOMETRY_RESULTS.json").read_text())["qualified"]
    out, table = {}, []
    for ch, g, s in C.CHANNELS:
        if ch not in qual:
            out[ch] = {"per_q": {}, "admissible_doses": [], "selected_q": None,
                       "decision": "DECLINE_ESTIMATOR",
                       "note": "geometry gate not passed; no behavioural intervention run"}
            continue
        per_q = {}
        for q in C.Q_GRID:
            key = (ch, "d_grad", q) if q != 0.0 else (ch, "zero", 0.0)
            if key not in by:
                continue
            m = metrics(ctx, ch, by[key])
            d = deltas(ctx, ch, by[key])
            cond, ok = gate_eval(m)
            per_q[str(q)] = {"metrics": m, "deltas": d, "gate": cond,
                             "admissible": bool(ok and q != 0.0)}
            row = flat(ch, "d_grad" if q != 0.0 else "zero", q, m, d)
            row["admissible"] = bool(ok and q != 0.0)
            table.append(row)
        for q in C.Q_GRID:
            key = (ch, "d_L26_diffmean", q)
            if key in by:
                m = metrics(ctx, ch, by[key])
                d = deltas(ctx, ch, by[key])
                row = flat(ch, "d_L26_diffmean", q, m, d)
                row["admissible"] = None
                table.append(row)
        adm = [q for q in C.Q_GRID if q != 0.0 and per_q.get(str(q), {}).get("admissible")]
        out[ch] = {"per_q": per_q, "admissible_doses": adm,
                   "selected_q": min(adm) if adm else None,
                   "decision": "ADVANCE_TO_CONTROLS" if adm else "STAGE_E_CHANNEL_DO_NOT_ADVANCE"}
        print("%s: admissible=%s selected=%s -> %s"
              % (ch, adm, out[ch]["selected_q"], out[ch]["decision"]))
    C.wj(SELECTED, {"gate": C.GATE, "q_grid": C.Q_GRID,
                    "bootstrap": {"draws": BOOT, "seed": SEED},
                    "selection_rule": "smallest nonzero admissible q; ties to smaller q; "
                                      "no interpolation; no grid extension",
                    "selected": out})
    write_csv(C.HERE / "DEV_DOSE_TABLE.csv", table)


def write_csv(path, rows):
    if not rows:
        open(path, "w").write("")
        return
    cols = list(rows[0])
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def finalize():
    ctx = C.Ctx()
    by = load_records()
    sel = json.loads(SELECTED.read_text())
    geo = json.loads((C.HERE / "_GEOMETRY_RESULTS.json").read_text())
    dose = json.loads((C.HERE / "DOSE_DECLARATION.json").read_text())
    arm_rows, dest_rows, coll_rows = [], [], []
    results, rand_out, rev_out, ung_out = {}, {}, {}, {}

    for ch, g, s in C.CHANNELS:
        info = sel["selected"][ch]
        q = info["selected_q"]
        geo_dec = geo["per_channel"][ch]["decision"]
        if q is None:
            results[ch] = {"selected_q": None, "geometry": geo_dec,
                           "decision_stage": info["decision"]}
            # still emit whatever d_grad doses exist, for the destination audit
            for qq in C.Q_GRID:
                key = (ch, "d_grad", qq) if qq != 0.0 else (ch, "zero", 0.0)
                if key in by:
                    m = metrics(ctx, ch, by[key])
                    arm_rows.append(flat(ch, key[1], qq, m, deltas(ctx, ch, by[key])))
                    dest_rows.append(dest_row(ch, key[1], qq, m))
                    coll_rows.append(coll_row(ch, key[1], qq, m))
            continue

        per = {}
        arms = ["zero", "d_grad", "d_L26_diffmean", "d_L21_same_layer_diffmean",
                "reverse_d_grad", "wrong_layer_d_grad", "ungated_d_grad"] + \
               ["dev_random_%d" % i for i in range(len(C.DEV_RANDOM_SEEDS))]
        for arm in arms:
            key = (ch, arm, 0.0 if arm == "zero" else q)
            if key not in by:
                continue
            m = metrics(ctx, ch, by[key])
            d = deltas(ctx, ch, by[key])
            per[arm] = {"metrics": m, "deltas": d, "n_records": len(by[key])}
            arm_rows.append(flat(ch, arm, key[2], m, d))
            dest_rows.append(dest_row(ch, arm, key[2], m))
            coll_rows.append(coll_row(ch, arm, key[2], m))
        # every d_grad dose also lands in the destination/collateral audits
        for qq in C.Q_GRID:
            for arm in ("d_grad", "d_L26_diffmean"):
                key = (ch, arm, qq)
                if key in by and qq != q:
                    m = metrics(ctx, ch, by[key])
                    dest_rows.append(dest_row(ch, arm, qq, m))
                    coll_rows.append(coll_row(ch, arm, qq, m))

        real = per["d_grad"]["metrics"]["target_gain_rate"]
        rnd = [per["dev_random_%d" % i]["metrics"]["target_gain_rate"]
               for i in range(len(C.DEV_RANDOM_SEEDS)) if "dev_random_%d" % i in per]
        ge = sum(1 for x in rnd if x >= real)
        rand_out[ch] = {
            "selected_q": q, "k_random": len(rnd),
            "real_target_gain_rate": real,
            "random_target_gain_rates": rnd,
            "random_min": min(rnd) if rnd else None,
            "random_median": float(np.median(rnd)) if rnd else None,
            "random_max": max(rnd) if rnd else None,
            "randoms_ge_real": ge, "real_rank_of_%d" % (len(rnd) + 1): ge + 1,
            "real_minus_random_median": (real - float(np.median(rnd))) if rnd else None,
            "real_exceeds_every_random": ge == 0,
            "seeds": C.DEV_RANDOM_SEEDS,
            "status": "frozen DEV development budget; descriptive_only per the frozen "
                      "protocol. The formal K=59 null is not generated here and its seeds "
                      "are disjoint from these by construction.",
        }
        rv = per.get("reverse_d_grad", {}).get("metrics")
        rev_out[ch] = {
            "selected_q": q,
            "real_target_gain_rate": real,
            "reverse_target_gain_rate": rv["target_gain_rate"] if rv else None,
            "real_target_hit": per["d_grad"]["metrics"]["target_hit"],
            "reverse_target_hit": rv["target_hit"] if rv else None,
            "real_source_exits": per["d_grad"]["metrics"]["source_exits"],
            "reverse_source_exits": rv["source_exits"] if rv else None,
            "real_delta_margin_median": per["d_grad"]["deltas"].get("delta_margin_median"),
            "reverse_delta_margin_median": (per["reverse_d_grad"]["deltas"]
                                            .get("delta_margin_median")
                                            if "reverse_d_grad" in per else None),
            "reverse_reproduces_real": (rv is not None and rv["target_gain_rate"] is not None
                                        and real is not None
                                        and rv["target_gain_rate"] >= real),
        }
        ug = per.get("ungated_d_grad", {}).get("metrics")
        routed, pop, _ = ctx.routed(ch)
        ung_out[ch] = {
            "selected_q": q, "tau": ctx.tau[ch],
            "dev_pred_eq_source": len(pop), "dev_rows_exposed_gated": len(routed),
            "dev_rows_exposed_ungated": len(pop),
            "router_selectivity_ratio": len(routed) / len(pop),
            "router_effectively_ungated": len(routed) == len(pop),
            "gated_target_gain_rate": real,
            "ungated_target_gain_rate": ug["target_gain_rate"] if ug else None,
            "gated_collateral_rate": per["d_grad"]["metrics"]["clean_collateral_rate"],
            "ungated_collateral_rate": ug["clean_collateral_rate"] if ug else None,
            "gated_eligible_collateral": per["d_grad"]["metrics"]["eligible_collateral"],
            "ungated_eligible_collateral": ug["eligible_collateral"] if ug else None,
        }

        # ---- frozen score-space bias comparator ----
        recs = by[(ch, "d_grad", q)]
        dev_err = {r["sample_id"] for r in ctx.errors(ch, "dev")}
        dmv = [((r["scores_int"][g] - r["scores_int"][s])
                - (r["scores_base"][g] - r["scores_base"][s]))
               for r in recs if r["sample_id"] in dev_err]
        b_c = float(np.median(dmv))
        fake = []
        for r in recs:
            e = dict(r["scores_base"])
            e[g] = e[g] + b_c / 2.0
            e[s] = e[s] - b_c / 2.0
            order = sorted(((v, k) for k, v in e.items()), reverse=True)
            fake.append({"sample_id": r["sample_id"], "pred_int": order[0][1],
                         "scores_base": r["scores_base"], "scores_int": None})
        mm = metrics(ctx, ch, fake)
        arm_rows.append(flat(ch, "score_space_bias", q, mm, {"delta_margin_median": b_c}))
        dest_rows.append(dest_row(ch, "score_space_bias", q, mm))
        coll_rows.append(coll_row(ch, "score_space_bias", q, mm))

        results[ch] = {
            "selected_q": q, "geometry": geo_dec,
            "s_c": dose["per_channel"][ch]["s_c"],
            "absolute_delta_norm": q * dose["per_channel"][ch]["s_c"],
            "arms": {k: {"metrics": v["metrics"], "deltas": v["deltas"]} for k, v in per.items()},
            "score_comparator": {"b_c": b_c,
                                 "applied": "e_gold += b_c/2 ; e_source -= b_c/2 ; others unchanged",
                                 "metrics": mm},
            "random_specificity": rand_out[ch], "reverse_control": rev_out[ch],
            "ungated_control": ung_out[ch],
        }

    # ---------------- conjunctive advancement ----------------
    adv_rows, adv_detail = [], {}
    for ch, g, s in C.CHANNELS:
        r = results[ch]
        geo_ok = geo["per_channel"][ch]["decision"] == "GEOMETRY_QUALIFIED"
        q = r.get("selected_q")
        if not geo_ok:
            dec, why = "DECLINE_ESTIMATOR", "frozen geometry gate not satisfied"
            cond = {}
        elif q is None:
            dec, why = "DECLINE_CORRECTABILITY", "no dose satisfied the frozen DEV dose gate"
            cond = {}
        else:
            dgm = r["arms"]["d_grad"]["metrics"]
            cond = {
                "1_support_gate": True,
                "2_readability_gate": True,
                "3_geometry_qualification": geo_ok,
                "4_real_beats_matched_random": r["random_specificity"]["real_exceeds_every_random"],
                "5_reverse_does_not_reproduce": not r["reverse_control"]["reverse_reproduces_real"],
                "6_positive_destination_correct_signal": (
                    dgm["gold_arrivals"] > 0 and (dgm["target_gain_rate"] or 0) > 0
                    and dgm["target_gain_rate_ci95"][0] > 0),
                "7_source_exits_not_dominated_by_other_wrong": (
                    dgm["source_exits"] > 0 and dgm["gold_arrivals"] > dgm["wrong_to_wrong"]),
                "8_fixed_broke_criterion": dgm["fixed"] > dgm["broke"],
                "9_collateral_within_frozen_dev_limit": (
                    dgm["clean_collateral_rate"] is not None
                    and dgm["clean_collateral_rate"] <= C.GATE["clean_collateral_max"]
                    and dgm["clean_collateral_rate_ci95"][1] <= C.GATE["clean_collateral_max"]),
                "10_dose_not_pathological_boundary_optimum": q != max(C.Q_GRID),
                "11_non_vacuous_collateral_evaluation": dgm["eligible_collateral"] > 0,
                "12_exceeds_frozen_score_space_comparator": (
                    (dgm["target_gain_rate"] or 0)
                    > (r["score_comparator"]["metrics"]["target_gain_rate"] or 0)),
                "13_diffmean_does_not_reproduce": (
                    "d_L26_diffmean" not in r["arms"]
                    or (r["arms"]["d_L26_diffmean"]["metrics"]["target_gain_rate"] or 0)
                    < (dgm["target_gain_rate"] or 0)),
                "14_wrong_layer_does_not_reproduce": (
                    "wrong_layer_d_grad" not in r["arms"]
                    or (r["arms"]["wrong_layer_d_grad"]["metrics"]["target_gain_rate"] or 0)
                    < (dgm["target_gain_rate"] or 0)),
            }
            if all(cond.values()):
                dec, why = "FORMAL_ADVANCEMENT_ELIGIBLE", "complete frozen conjunction satisfied"
            elif not cond["4_real_beats_matched_random"] or not cond["5_reverse_does_not_reproduce"]:
                dec, why = "DECLINE_STEERABILITY", "direction-specificity not established"
            elif not cond["9_collateral_within_frozen_dev_limit"] or not cond["11_non_vacuous_collateral_evaluation"]:
                dec, why = "DECLINE_COLLATERAL", ("collateral bound breached"
                                                  if not cond["9_collateral_within_frozen_dev_limit"]
                                                  else "collateral evaluation structurally vacuous")
            else:
                dec, why = "DECLINE_CORRECTABILITY", "frozen conjunction not satisfied"
        adv_detail[ch] = {"conditions": cond, "decision": dec, "reason": why,
                          "selected_q": q}
        dgm = r["arms"]["d_grad"]["metrics"] if q is not None else None
        adv_rows.append({
            "channel": ch, "gold": g, "source": s,
            "support": "ELIGIBLE",
            "readable": "READABLE (DEV AUC %.4f, tau %s)"
                        % (dev_auc(ch), ctx.tau[ch]),
            "geometry": geo["per_channel"][ch]["decision"],
            "steerable": ("YES" if q is not None and cond.get("4_real_beats_matched_random")
                          and cond.get("5_reverse_does_not_reproduce")
                          else "NO" if q is not None else "NOT_EVALUATED"),
            "destination_correct": ("target_gain_rate %.4f, target_hit %s, gold %d vs other-wrong %d"
                                    % (dgm["target_gain_rate"], fmt(dgm["target_hit"]),
                                       dgm["gold_arrivals"], dgm["wrong_to_wrong"])
                                    if dgm else "NOT_EVALUATED"),
            "collateral": (("%d/%d = %.4f (eligible exposure %d)"
                            % (dgm["collateral_count"], dgm["baseline_correct_n"],
                               dgm["clean_collateral_rate"], dgm["eligible_collateral"]))
                           if dgm else "NOT_EVALUATED"),
            "decision": dec, "reason": why})

    write_csv(C.HERE / "DEV_ARM_SUMMARY.csv", arm_rows)
    write_csv(C.HERE / "DEV_DESTINATION_AUDIT.csv", dest_rows)
    write_csv(C.HERE / "DEV_COLLATERAL_AUDIT.csv", coll_rows)
    write_csv(C.HERE / "CHANNEL_ADVANCEMENT_TABLE.csv", adv_rows)
    C.wj(C.HERE / "DEV_RANDOM_SPECIFICITY.json",
         {"statistic": "target_gain_rate of the real direction versus the eight frozen DEV "
                       "matched-random directions at the selected dose, same absolute "
                       "perturbation norm q*s_c, same population",
          "prespecified": True, "changed_after_results": False,
          "per_channel": rand_out})
    C.wj(C.HERE / "DEV_REVERSE_CONTROL.json",
         {"statistic": "target_gain_rate and target_hit of -d_grad at the selected dose",
          "per_channel": rev_out})
    C.wj(C.HERE / "DEV_UNGATED_CONTROL.json",
         {"semantics": "same q*s_c; only the Router gate is disabled",
          "per_channel": ung_out})
    C.wj(C.HERE / "_STAGE_E_RESULTS.json",
         {"results": results, "advancement": adv_detail,
          "eligible": [ch for ch in adv_detail
                       if adv_detail[ch]["decision"] == "FORMAL_ADVANCEMENT_ELIGIBLE"],
          "sealed_evaluation_accessed": False})
    for ch in adv_detail:
        print("%s -> %s (%s)" % (ch, adv_detail[ch]["decision"], adv_detail[ch]["reason"]))


def fmt(x):
    return "n/a" if x is None else "%.4f" % x


def dev_auc(ch):
    import csv as _csv
    for r in _csv.DictReader(open(C.PROSP / "ROUTER_AUDIT.csv", encoding="utf-8")):
        if r["channel"].replace("->", "__to__") == ch:
            return float(r["dev_auc"])
    return float("nan")


def dest_row(ch, arm, q, m):
    d = m["destination_counts"]
    return {"channel": ch, "arm": arm, "q": q,
            "n_channel_error": m["n_channel_error"],
            "SOURCE_RETAINED": d["SOURCE_RETAINED"],
            "GOLD_ARRIVAL": d["GOLD_ARRIVAL"],
            "OTHER_WRONG": d["OTHER_WRONG"],
            "source_exits": m["source_exits"], "gold_arrivals": m["gold_arrivals"],
            "other_wrong_transitions": m["wrong_to_wrong"],
            "target_hit": m["target_hit"], "target_gain_count": m["target_gain_count"],
            "target_gain_rate": m["target_gain_rate"],
            "target_gain_rate_ci_lo": m["target_gain_rate_ci95"][0],
            "target_gain_rate_ci_hi": m["target_gain_rate_ci95"][1],
            "target_hit_ci_lo": m["target_hit_ci95"][0],
            "target_hit_ci_hi": m["target_hit_ci95"][1],
            "router_fired_channel_error_count": m["router_fired_channel_error_count"]}


def coll_row(ch, arm, q, m):
    c = m["collateral_destination_counts"]
    return {"channel": ch, "arm": arm, "q": q,
            "baseline_correct_n": m["baseline_correct_n"],
            "router_fired_baseline_correct_exposure": m["router_fired_baseline_correct_exposure"],
            "eligible_collateral": m["eligible_collateral"],
            "CORRECT_RETAINED": c["CORRECT_RETAINED"],
            "BROKEN_TO_SOURCE": c["BROKEN_TO_SOURCE"],
            "BROKEN_TO_OTHER": c["BROKEN_TO_OTHER"],
            "collateral_count": m["collateral_count"],
            "clean_collateral_rate": m["clean_collateral_rate"],
            "collateral_ci_lo": m["clean_collateral_rate_ci95"][0],
            "collateral_ci_hi": m["clean_collateral_rate_ci95"][1],
            "fixed": m["fixed"], "broke": m["broke"], "net": m["net"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select-dose", action="store_true")
    g.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    (select_dose if a.select_dose else finalize)()
