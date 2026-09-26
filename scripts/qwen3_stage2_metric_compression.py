#!/usr/bin/env python3
"""Qwen3 SAKIKO Stage 2 — formal metric-compression audit.

Read-only. No model load, no GPU, no inference, no arm rerun, no evaluation
prompt or dataset payload access, no frozen artifact modified. Uses only the
per-sample outputs already committed by the single authorized formal run.

Quantifies two distinct forms of information compression by aggregate
correctness-family metrics:

  A1 DESTINATION COMPRESSION
      near-identical aggregate correctness can conceal different sample-level
      destinations in a multiclass response-mode space;

  A2 REPAIR-HARM CANCELLATION
      near-identical Net can conceal materially different combinations of
      repair and collateral damage.

Every analysis carries one of three status labels:
  PRESPECIFIED AND ALREADY REPORTED
  LATE-COMPUTED PRESPECIFIED SECONDARY
  POST-HOC EXPLORATORY FORMAL-RECORD ANALYSIS

Nothing here may be promoted to the formal primary endpoint, which was and
remains FORMAL_CONFIRMATORY_SUCCESS. No row identifier, UUID, prompt, label,
prediction string or example content is written to any output.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "final/results/qwen3_stage2_formal"
OUT = ROOT / "final/results/qwen3_stage2_metric_compression"

RECORDS = FORMAL / "QWEN3_STAGE2_FORMAL_RECORDS.jsonl"
RESULTS = FORMAL / "QWEN3_STAGE2_FORMAL_RESULTS.json"
HASHES = FORMAL / "QWEN3_STAGE2_FORMAL_RESULT_HASHES.json"

GOLD, SOURCE = "cannot_answer", "tool_call"
N_EVAL = 548
CATS = ["GOLD", "SOURCE", "OTHER_WRONG"]
UTILITY = {"GOLD": 1, "SOURCE": 0, "OTHER_WRONG": -1}
ALPHA = 0.05
Z = 1.959963984540054

# Locked from the previously committed paired recovery. Not recomputed here.
LOCKED_CLAIM_C = {
    "point_difference": 0.091954,
    "bootstrap_ci_includes_zero": True,
    "exact_two_sided_randomization_p": 0.3770,
    "verdict": "UNDERPOWERED / INCONCLUSIVE",
    "note": "fixed; no alternative version of Claim C is computed in this audit",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clopper_pearson(k: int, n: int, alpha: float = ALPHA):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def paired_rd_ci(b: int, c: int, n: int):
    if n == 0:
        return [None, None]
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n ** 2)
    se = max(var, 0.0) ** 0.5
    return [d - Z * se, d + Z * se]


def classify(pred: str) -> str:
    return "GOLD" if pred == GOLD else ("SOURCE" if pred == SOURCE else "OTHER_WRONG")


def main() -> None:
    # ---- Block 1: integrity
    integrity = {}
    for rel, meta in json.loads(HASHES.read_text())["artifacts"].items():
        got = sha256_file(ROOT / rel)
        integrity[rel] = {"recomputed": got, "committed": meta["sha256"], "match": got == meta["sha256"]}
        if got != meta["sha256"]:
            raise SystemExit("METRIC_COMPRESSION_PRECHECK_FAILED")
    res = json.loads(RESULTS.read_text())

    need = {"zero", "d_grad", "ungated_d_grad", "score_space_comparator"}
    arms: dict[str, dict[str, dict]] = {a: {} for a in need}
    with RECORDS.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] in need:
                arms[r["arm"]][r["sample_id"]] = r
    if any(not arms[a] for a in need):
        raise SystemExit("METRIC_COMPRESSION_UNRECONSTRUCTABLE: missing arm")

    routed = set(arms["d_grad"])
    if routed != set(arms["score_space_comparator"]) or routed != set(arms["zero"]):
        raise SystemExit("METRIC_COMPRESSION_UNRECONSTRUCTABLE: routed arms disagree")
    pes = set(arms["ungated_d_grad"])
    if not routed <= pes:
        raise SystemExit("METRIC_COMPRESSION_UNRECONSTRUCTABLE: routed not inside source-predicted set")

    # zero-arm reproduces baseline exactly on every routed row
    zero_exact = all(arms["zero"][s]["pred_int"] == arms["zero"][s]["pred_base"] for s in routed)

    base = {s: (r["gold"], r["pred_base"]) for s, r in arms["ungated_d_grad"].items()}
    err = sorted(s for s, (g, pb) in base.items() if g == GOLD and pb == SOURCE)
    correct_in_pes = sorted(s for s, (g, pb) in base.items() if g == pb)

    def dest(arm: str, sid: str) -> str:
        r = arms[arm].get(sid)
        return classify(r["pred_int"]) if r is not None else "SOURCE"

    act = [dest("d_grad", s) for s in err]
    cmp_ = [dest("score_space_comparator", s) for s in err]

    def marg(labels):
        c = Counter(labels)
        return {"gold_arrivals": c["GOLD"], "source_remains": c["SOURCE"],
                "wrong_to_wrong": c["OTHER_WRONG"],
                "source_exits": c["GOLD"] + c["OTHER_WRONG"],
                "target_gain_count": c["GOLD"] - c["OTHER_WRONG"]}

    m_act, m_cmp = marg(act), marg(cmp_)
    if (len(err) != 87
            or m_act != {"gold_arrivals": 38, "source_remains": 35, "wrong_to_wrong": 14,
                         "source_exits": 52, "target_gain_count": 24}
            or m_cmp != {"gold_arrivals": 37, "source_remains": 29, "wrong_to_wrong": 21,
                         "source_exits": 58, "target_gain_count": 16}):
        raise SystemExit("METRIC_COMPRESSION_UNRECONSTRUCTABLE: marginals disagree")

    # ---- Block 3: destination compression
    tab = {a: {c: 0 for c in CATS} for a in CATS}
    for a, c in zip(act, cmp_):
        tab[a][c] += 1
    agree = sum(tab[c][c] for c in CATS)
    disagree = 87 - agree
    if agree != 55 or disagree != 32:
        raise SystemExit("METRIC_COMPRESSION_UNRECONSTRUCTABLE: agreement disagrees with committed recovery")

    a1 = {
        "status": "POST-HOC EXPLORATORY FORMAL-RECORD ANALYSIS (3x3 table itself was already recovered)",
        "paired_table_3x3": tab, "total": 87, "agreement": agree, "disagreement": disagree,
        "destination_disagreement_proportion": disagree / 87,
        "disagreement_clopper_pearson_ci95": clopper_pearson(disagree, 87),
        "marginals": {"activation": m_act, "score_space_comparator": m_cmp},
    }

    # ---- Block 4: repair-harm cancellation, gated vs source-prediction-gated
    def arm_block(name: str, key: str) -> dict:
        s = res["primary"]["real"] if key == "d_grad" else res["secondary"][key]
        final_correct = res["primary"]["real"]["baseline_correct_n"] + s["fixed"] - s["broke"]
        return {"arm": name, "fixed": s["fixed"], "broke": s["broke"], "net": s["net"],
                "final_correct_of_548": final_correct, "final_accuracy": final_correct / N_EVAL,
                "source_exits": s["source_exits"], "gold_arrivals": s["gold_arrivals"],
                "wrong_to_wrong": s["wrong_to_wrong"],
                "source_remains": s["n_channel_error"] - s["source_exits"],
                "target_hit": s["target_hit"], "target_gain_count": s["target_gain_count"],
                "target_gain_rate": s["target_gain_rate"],
                "observed_collateral_count": s["collateral_count"],
                "frozen_collateral_denominator": s["baseline_correct_n"]}

    gated, ungated = arm_block("router_gated_d_grad", "d_grad"), arm_block("source_prediction_gated_d_grad", "ungated_d_grad")

    # eligible at-risk populations, reconstructed from committed records
    elig_ungated = len(correct_in_pes)
    elig_gated = sum(1 for s in correct_in_pes if s in routed)
    br_g = {s: (s in routed and arms["d_grad"][s]["pred_int"] != base[s][0]) for s in correct_in_pes}
    br_u = {s: arms["ungated_d_grad"][s]["pred_int"] != base[s][0] for s in correct_in_pes}
    n11 = sum(1 for s in correct_in_pes if br_g[s] and br_u[s])
    b_only = sum(1 for s in correct_in_pes if br_g[s] and not br_u[s])
    c_only = sum(1 for s in correct_in_pes if br_u[s] and not br_g[s])
    n_correct_total = res["primary"]["real"]["baseline_correct_n"]
    n00 = n_correct_total - n11 - b_only - c_only          # includes rows outside the source-predicted set
    a2 = {
        "status": "POST-HOC EXPLORATORY FORMAL-RECORD ANALYSIS",
        "arms": [gated, ungated],
        "scope_notes": [
            "the so-called ungated arm is still restricted by the baseline source-prediction condition",
            "it removes the Router gate but is not a completely unconditional intervention",
            "Net is a global 548-row correctness-family summary",
            "Net includes Broke; it is Fixed minus Broke",
            "its limitation is compression and cancellation, not complete blindness",
        ],
        "paired_2x2_over_all_baseline_correct": {
            "denominator": n_correct_total,
            "gated_break_and_ungated_break": n11,
            "gated_only_break": b_only,
            "ungated_only_break": c_only,
            "neither_breaks": n00,
            "mcnemar_exact_p_two_sided": exact_mcnemar(b_only, c_only),
            "paired_collateral_risk_difference": (b_only - c_only) / n_correct_total,
            "paired_collateral_risk_difference_ci95": paired_rd_ci(b_only, c_only, n_correct_total),
        },
        "eligible_at_risk_reconstruction": {
            "reconstructable_without_inference_or_row_disclosure": True,
            "note": "recovered from committed per-sample records; supersedes the earlier "
                    "ELIGIBLE_AT_RISK_DENOMINATOR_NOT_RECONSTRUCTABLE marker",
            "source_predicted_population_size": len(pes),
            "router_gated_population_size": len(routed),
            "eligible_at_risk_source_prediction_gated": elig_ungated,
            "eligible_at_risk_router_gated": elig_gated,
            "events_router_gated": sum(br_g.values()),
            "events_source_prediction_gated": sum(br_u.values()),
            "rate_router_gated_on_its_eligible_denominator":
                (sum(br_g.values()) / elig_gated) if elig_gated else None,
            "rate_source_prediction_gated_on_its_eligible_denominator":
                (sum(br_u.values()) / elig_ungated) if elig_ungated else None,
            "frozen_reported_rates_use_denominator": n_correct_total,
            "post_hoc_exact_bounds": {
                "note": "POST-HOC EXPLORATORY. No interval on the eligible denominator was "
                        "frozen before the results were observed. Shown only to make the "
                        "exposure difference legible.",
                "router_gated_events_over_eligible": [sum(br_g.values()), elig_gated],
                "router_gated_clopper_pearson_ci95": clopper_pearson(sum(br_g.values()), elig_gated),
                "source_prediction_gated_events_over_eligible": [sum(br_u.values()), elig_ungated],
                "source_prediction_gated_clopper_pearson_ci95":
                    clopper_pearson(sum(br_u.values()), elig_ungated),
                "frozen_denominator_events_over_all_baseline_correct":
                    [sum(br_g.values()), n_correct_total],
                "frozen_denominator_clopper_pearson_ci95":
                    clopper_pearson(sum(br_g.values()), n_correct_total),
            },
        },
    }

    # ---- Block 5: paired all-548 correctness for both contrasts
    def all548(a_key: str, b_key: str, name: str) -> dict:
        pop = set(arms[a_key]) | set(arms[b_key])
        b = c = 0
        for s in pop:
            g = base[s][0]
            pa = arms[a_key][s]["pred_int"] if s in arms[a_key] else base[s][1]
            pb = arms[b_key][s]["pred_int"] if s in arms[b_key] else base[s][1]
            ca, cb = pa == g, pb == g
            b += int(ca and not cb)
            c += int(cb and not ca)
        return {"contrast": name,
                "status": "POST-HOC EXPLORATORY FORMAL-RECORD ANALYSIS",
                "scope": "rows neither arm modified are concordant by construction",
                "first_correct_second_wrong": b, "second_correct_first_wrong": c,
                "paired_risk_difference": (b - c) / N_EVAL,
                "paired_risk_difference_ci95": paired_rd_ci(b, c, N_EVAL),
                "mcnemar_exact_p_two_sided": exact_mcnemar(b, c)}

    s_act = res["primary"]["real"]
    s_cmp = res["secondary"]["score_space_comparator"]
    s_ung = res["secondary"]["ungated_d_grad"]
    blk5 = [
        {**all548("d_grad", "score_space_comparator", "activation vs score-space"),
         "fixed_broke_net": {"activation": [s_act["fixed"], s_act["broke"], s_act["net"]],
                             "score_space": [s_cmp["fixed"], s_cmp["broke"], s_cmp["net"]]}},
        {**all548("d_grad", "ungated_d_grad", "router-gated vs source-prediction-gated"),
         "fixed_broke_net": {"router_gated": [s_act["fixed"], s_act["broke"], s_act["net"]],
                             "source_prediction_gated": [s_ung["fixed"], s_ung["broke"], s_ung["net"]]}},
    ]
    for e in blk5:
        lo, hi = e["paired_risk_difference_ci95"]
        e["post_hoc_practical_equivalence_sensitivity"] = {
            "note": "no equivalence margin was frozen before the results were observed; "
                    "this is a compatibility display only and selects no preferred margin",
            "observed_ci95_percentage_points": [lo * 100, hi * 100],
            "margins_compatible_with_the_interval_pp":
                [m for m in (0.5, 1.0, 2.0, 5.0) if max(abs(lo), abs(hi)) * 100 <= m],
            "margins_not_compatible_pp":
                [m for m in (0.5, 1.0, 2.0, 5.0) if max(abs(lo), abs(hi)) * 100 > m],
        }
        e["non_significance_is_not_equality"] = True

    out = {
        "schema_version": 1, "analysis": "QWEN3_STAGE2_METRIC_COMPRESSION",
        "not_part_of_formal_execution": True,
        "formal_primary_unchanged": res["outcome"],
        "locked_claim_C": LOCKED_CLAIM_C,
        "integrity": integrity,
        "arms_present": sorted(need), "zero_arm_reproduces_baseline_exactly": zero_exact,
        "populations": {"evaluation_rows": N_EVAL,
                        "baseline_correct": n_correct_total,
                        "source_predicted": len(pes), "router_gated": len(routed),
                        "formal_channel_errors": len(err)},
        "claim_A1_destination_compression": a1,
        "claim_A2_repair_harm_cancellation": a2,
        "block5_global_correctness_sensitivity": blk5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "QWEN3_STAGE2_METRIC_COMPRESSION.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    rows = ["contrast,cell,count"]
    for a in CATS:
        for c in CATS:
            rows.append(f"activation_vs_score,{a}|{c},{tab[a][c]}")
    p = a2["paired_2x2_over_all_baseline_correct"]
    for k in ("gated_break_and_ungated_break", "gated_only_break", "ungated_only_break", "neither_breaks"):
        rows.append(f"gated_vs_ungated_baseline_correct,{k},{p[k]}")
    (OUT / "aggregate_paired_tables.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    g = ["contrast,first_correct_second_wrong,second_correct_first_wrong,paired_risk_difference,ci_low,ci_high,mcnemar_exact_p"]
    for e in blk5:
        lo, hi = e["paired_risk_difference_ci95"]
        g.append(f'{e["contrast"]},{e["first_correct_second_wrong"]},{e["second_correct_first_wrong"]},'
                 f'{e["paired_risk_difference"]:.6g},{lo:.6g},{hi:.6g},{e["mcnemar_exact_p_two_sided"]:.6g}')
    (OUT / "global_correctness_paired_tables.csv").write_text("\n".join(g) + "\n", encoding="utf-8")

    print(json.dumps({k: out[k] for k in
                      ("populations", "zero_arm_reproduces_baseline_exactly",
                       "claim_A1_destination_compression", "claim_A2_repair_harm_cancellation",
                       "block5_global_correctness_sensitivity")}, indent=1, default=str))


if __name__ == "__main__":
    main()
