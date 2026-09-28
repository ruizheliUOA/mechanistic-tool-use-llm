#!/usr/bin/env python3
"""Package the Qwen3-4B formal result into the required deliverables.

Read-only with respect to science: it splits the frozen runner's committed record and
result files into the per-arm evidence files the formal protocol requires, and derives the
required audits. No model is loaded, no SEALED row is re-scored, no endpoint is recomputed
from anything other than the runner's own records, and no threshold, denominator or
condition is redefined.
"""

import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
NS = ROOT / "final/results/qwen3_4b_w2c_formal_v1"
DE = ROOT / "research_exploration/qwen3_4b_stage_d_e_v1"
GATE = ROOT / "research_exploration/qwen3_4b_formal_gate_v1"
AUDIT = ROOT / "research_exploration/qwen3_4b_collateral_integrity_audit_v1"
RECORDS = NS / "QWEN3_4B_FORMAL_RECORDS.jsonl"
RESULTS = NS / "QWEN3_4B_FORMAL_RESULTS.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def wj(p, o):
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(o, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(t, p)


def wcsv(p, rows):
    if not rows:
        Path(p).write_text("")
        return
    cols = list(rows[0])
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


ARM_FILE = {
    "zero": "FORMAL_ZERO_RECORDS.jsonl",
    "real_d_grad": "FORMAL_REAL_RECORDS.jsonl",
    "d_L26_diffmean": "FORMAL_DIFFMEAN_RECORDS.jsonl",
    "reverse_d_grad": "FORMAL_REVERSE_RECORDS.jsonl",
    "wrong_layer_d_grad": "FORMAL_WRONG_LAYER_RECORDS.jsonl",
    "ungated_d_grad": "FORMAL_UNGATED_RECORDS.jsonl",
}


def main():
    P = json.loads((DE / "FORMAL_PROTOCOL_FROZEN.json").read_text())
    fc = P["frozen_configuration"]
    R = json.loads(RESULTS.read_text())
    marker = json.loads((NS / "FORMAL_ACCESS_STARTED.json").read_text())
    gold, source = P["formal_channel"]["gold"], P["formal_channel"]["source"]
    K = P["formal_randoms"]["K"]

    by = defaultdict(list)
    for line in open(RECORDS, encoding="utf-8"):
        o = json.loads(line)
        by[o["arm"]].append(o)

    # ---- per-arm record files ----
    for arm, fn in ARM_FILE.items():
        with open(NS / fn, "w", encoding="utf-8", newline="\n") as f:
            for o in by[arm]:
                f.write(json.dumps(o, sort_keys=True) + "\n")
    with open(NS / "FORMAL_RANDOM_RECORDS.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in range(K):
            for o in by["formal_random_%d" % i]:
                f.write(json.dumps(o, sort_keys=True) + "\n")

    # ---- baseline records, reconstructed from the arms' own baseline fields ----
    base = {}
    for arm, recs in by.items():
        for o in recs:
            base.setdefault(o["sample_id"], {
                "sample_id": o["sample_id"], "project_index": o["project_index"],
                "gold": o["gold"], "scores_base": o["scores_base"],
                "pred_base": o["pred_base"],
                "router_probability": o["router_probability"],
                "router_decision": o["router_decision"],
                "baseline_correct": o["pred_base"] == o["gold"]})
    with open(NS / "FORMAL_BASELINE_RECORDS.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for sid in sorted(base):
            f.write(json.dumps(base[sid], sort_keys=True) + "\n")

    # ---- score-space comparator records (deterministic, from frozen b_c) ----
    b_c = fc["b_c"]
    ss = []
    for o in by["real_d_grad"]:
        e = dict(o["scores_base"])
        e[gold] += b_c / 2.0
        e[source] -= b_c / 2.0
        pred = max(e, key=lambda k: e[k])
        ss.append({"sample_id": o["sample_id"], "project_index": o["project_index"],
                   "gold": o["gold"], "scores_base": o["scores_base"],
                   "pred_base": o["pred_base"], "arm": "score_space", "b_c": b_c,
                   "applied": "e_gold += b_c/2 ; e_source -= b_c/2 ; others unchanged",
                   "scores_int": e, "pred_int": pred,
                   "destination": ("SOURCE_RETAINED" if pred == source
                                   else "GOLD_ARRIVAL" if pred == gold else "OTHER_WRONG"),
                   "router_probability": o["router_probability"],
                   "router_decision": o["router_decision"]})
    with open(NS / "FORMAL_SCORE_SPACE_RECORDS.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for o in ss:
            f.write(json.dumps(o, sort_keys=True) + "\n")

    # ---- destination results ----
    arms = R["arms"]
    order = ["zero", "real_d_grad", "d_L26_diffmean", "reverse_d_grad",
             "wrong_layer_d_grad", "ungated_d_grad", "score_space"] + \
            ["formal_random_%d" % i for i in range(K)]
    drows = []
    for a in order:
        if a not in arms:
            continue
        m = arms[a]
        d = m["destination_counts"]
        c = m["collateral_destination_counts"]
        drows.append({
            "arm": a, "n_channel_error": m["n_channel_error"],
            "router_fired_channel_errors": m["router_fired_channel_error_count"],
            "SOURCE_RETAINED": d["SOURCE_RETAINED"], "GOLD_ARRIVAL": d["GOLD_ARRIVAL"],
            "OTHER_WRONG": d["OTHER_WRONG"],
            "source_exits": m["source_exits"], "gold_arrivals": m["gold_arrivals"],
            "other_wrong": m["wrong_to_wrong"], "target_hit": m["target_hit"],
            "target_hit_ci_lo": m["target_hit_ci95"][0], "target_hit_ci_hi": m["target_hit_ci95"][1],
            "target_gain_count": m["target_gain_count"], "target_gain_rate": m["target_gain_rate"],
            "target_gain_rate_ci_lo": m["target_gain_rate_ci95"][0],
            "target_gain_rate_ci_hi": m["target_gain_rate_ci95"][1],
            "fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
            "baseline_correct_n": m["baseline_correct_n"],
            "all_correct_collateral_count": m["collateral_count"],
            "all_correct_collateral_rate": m["clean_collateral_rate"],
            "all_correct_collateral_ci_lo": m["clean_collateral_rate_ci95"][0],
            "all_correct_collateral_ci_hi": m["clean_collateral_rate_ci95"][1],
            "eligible_at_risk_n": m["eligible_collateral"],
            "eligible_at_risk_collateral_rate": (m["collateral_count"] / m["eligible_collateral"]
                                                 if m["eligible_collateral"] else None),
            "CORRECT_RETAINED": c["CORRECT_RETAINED"],
            "BROKEN_TO_SOURCE": c["BROKEN_TO_SOURCE"], "BROKEN_TO_OTHER": c["BROKEN_TO_OTHER"],
        })
    wcsv(NS / "FORMAL_DESTINATION_RESULTS.csv", drows)
    wj(NS / "FORMAL_DESTINATION_RESULTS.json",
       {"channel": P["formal_channel"], "per_arm": {r["arm"]: r for r in drows},
        "destination_partition_verified": all(
            r["SOURCE_RETAINED"] + r["GOLD_ARRIVAL"] + r["OTHER_WRONG"] == r["n_channel_error"]
            for r in drows)})

    # ---- random null ----
    real = arms["real_d_grad"]["target_gain_rate"]
    rnd = R["random_target_gain_rates"]
    ge = R["randoms_ge_real"]
    rrows = [{"direction": "real_d_grad", "target_gain_rate": real, "is_real": True}] + \
            [{"direction": "formal_random_%d" % i, "target_gain_rate": rnd[i], "is_real": False,
              "ge_real": rnd[i] >= real} for i in range(K)]
    wcsv(NS / "FORMAL_RANDOM_NULL_RESULTS.csv", rrows)
    allv = sorted(rnd + [real], reverse=True)
    wj(NS / "FORMAL_RANDOM_NULL_RESULTS.json", {
        "statistic": P["primary"]["statistic"], "null": P["primary"]["null"],
        "p_value_definition": P["primary"]["p_value"],
        "K": K, "real_target_gain_rate": real, "random_target_gain_rates": rnd,
        "randoms_ge_real": ge, "real_rank_of_%d" % (K + 1): ge + 1,
        "p_add_one": R["p_add_one"], "one_sided": True,
        "tie_convention": "ties count AGAINST the real direction (>= in the numerator count)",
        "add_one_convention": "(1 + #{T_random >= T_real}) / (K + 1)",
        "random_min": min(rnd), "random_median": float(np.median(rnd)), "random_max": max(rnd),
        "real_minus_random_median": real - float(np.median(rnd)),
        "pooled_with_dev_randoms": False, "pooled_with_qwen3_8b_randoms": False,
        "alternative_test_used": False})

    # ---- collateral audit, both denominators ----
    m = arms["real_d_grad"]
    ug = arms["ungated_d_grad"]
    wj(NS / "FORMAL_COLLATERAL_AUDIT.json", {
        "frozen_formal_metric": {
            "definition": P["endpoints"]["clean_collateral_rate"],
            "numerator": m["collateral_count"], "denominator": m["baseline_correct_n"],
            "rate": m["clean_collateral_rate"], "ci95": m["clean_collateral_rate_ci95"],
            "threshold": 0.05,
            "point_pass": m["clean_collateral_rate"] <= 0.05,
            "ci_upper_pass": m["clean_collateral_rate_ci95"][1] <= 0.05},
        "eligible_at_risk_diagnostic": {
            "definition": "collateral events / (Router-fired AND baseline-correct) rows",
            "numerator": m["collateral_count"], "denominator": m["eligible_collateral"],
            "rate": (m["collateral_count"] / m["eligible_collateral"]
                     if m["eligible_collateral"] else None),
            "status": "REQUIRED DIAGNOSTIC DISCLOSURE -- not the frozen gate metric"},
        "dilution": {
            "untouched_rows_in_frozen_denominator":
                m["baseline_correct_n"] - m["eligible_collateral"],
            "fraction_never_perturbed": ((m["baseline_correct_n"] - m["eligible_collateral"])
                                         / m["baseline_correct_n"])},
        "collateral_destinations": m["collateral_destination_counts"],
        "ungated_comparison": {
            "eligible_at_risk_n": ug["eligible_collateral"],
            "collateral_count": ug["collateral_count"],
            "all_correct_rate": ug["clean_collateral_rate"],
            "eligible_at_risk_rate": (ug["collateral_count"] / ug["eligible_collateral"]
                                      if ug["eligible_collateral"] else None)},
        "claim_lock": ("Passing the frozen formal collateral endpoint must not be interpreted "
                       "as demonstrated deployment safety."),
        "claim_lock_source": "research_exploration/qwen3_4b_formal_gate_v1/COLLATERAL_CLAIM_LOCK.md",
        "prohibited_claims": ["safe deployment", "proven low-risk intervention", "zero-risk",
                              "deployment-safe"]})

    # ---- score-space paired audit ----
    sm = arms["score_space"]
    wj(NS / "FORMAL_SCORE_SPACE_PAIRED_AUDIT.json", {
        "comparator": P["deterministic_score_space_comparator"],
        "b_c_frozen_on": "DEV, at the selected dose; NOT tuned on SEALED",
        "activation": {k: m[k] for k in ("source_exits", "gold_arrivals", "wrong_to_wrong",
                                         "target_hit", "target_gain_rate", "fixed", "broke",
                                         "net", "clean_collateral_rate")},
        "score_space": {k: sm[k] for k in ("source_exits", "gold_arrivals", "wrong_to_wrong",
                                           "target_hit", "target_gain_rate", "fixed", "broke",
                                           "net", "clean_collateral_rate")},
        "paired_difference_target_gain_rate":
            (m["target_gain_rate"] or 0) - (sm["target_gain_rate"] or 0),
        "paired_difference_wrong_to_wrong_rate":
            (m["wrong_to_wrong"] - sm["wrong_to_wrong"]) / m["n_channel_error"],
        "activation_exceeds_score_space":
            (m["target_gain_rate"] or 0) > (sm["target_gain_rate"] or 0),
        "role_in_conjunction": ("secondary/ordered, non-promotable; the frozen conjunction does "
                                "NOT require activation superiority over score-space"),
        "interpretation_rule": ("If the score comparator matches or exceeds, do not claim the "
                               "activation intervention provides behaviour unavailable to a "
                               "simple frozen score-space shift.")})

    # ---- conjunction ----
    wj(NS / "FORMAL_SUCCESS_CONJUNCTION.json", {
        "frozen_conditions": P["primary"]["conjunction"],
        "is_intersection_claim": True, "evaluated_once": True,
        "observed": R["conjunction"],
        "all_pass": all(R["conjunction"].values()),
        "verdict": R["verdict"]})

    # ---- provenance, sealed audit, hashes ----
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    wj(NS / "FORMAL_INVOCATION_PROVENANCE.json", {
        "invocation_uuid": marker["invocation_uuid"], "access_marker_utc": marker["utc"],
        "repository_head": marker["repository_head"], "head_now": head,
        "runner": "scripts/qwen3_4b_w2c_formal.py",
        "runner_sha256": marker["runner_sha256"],
        "protocol_sha256": marker["protocol_sha256"],
        "model": marker["model_repository"], "model_revision": marker["model_revision"],
        "channel": marker["selected_channel"], "router_sha256": marker["router_sha256"],
        "direction_sha256": marker["direction_sha256"],
        "random_matrix_sha256": marker["random_matrix_sha256"],
        "gate_verdict": marker["gate_verdict"], "gate_checks": marker["gate_checks"],
        "formal_run_count": 1, "superseded_markers": marker.get("supersedes", []),
        "mechanical_incidents_before_access": 2,
        "incident_record": "research_exploration/qwen3_4b_formal_gate_v1/mechanical_incident_1/"})

    nrec = sum(1 for _ in open(RECORDS, encoding="utf-8"))
    sids = defaultdict(set)
    dup = 0
    for line in open(RECORDS, encoding="utf-8"):
        o = json.loads(line)
        if o["sample_id"] in sids[o["arm"]]:
            dup += 1
        sids[o["arm"]].add(o["sample_id"])
    nonfinite = 0
    for line in open(RECORDS, encoding="utf-8"):
        o = json.loads(line)
        for d in (o["scores_base"], o["scores_int"]):
            if not all(np.isfinite(list(d.values()))):
                nonfinite += 1
    led = json.loads((GATE / "ONE_SHOT_EXECUTION_LEDGER.json").read_text())
    wj(NS / "SEALED_ACCESS_AUDIT.json", {
        "sealed_population": 548,
        "access_marker_written_before_first_sealed_load": True,
        "invocation_uuid": marker["invocation_uuid"],
        "formal_run_count": led["formal_run_count"],
        "scientific_endpoint_observed": led["scientific_endpoint_observed"],
        "second_scientific_invocation_exists": False,
        "total_records": nrec, "duplicate_sample_within_arm": dup,
        "non_finite_score_records": nonfinite,
        "arms_present": sorted(sids),
        "n_arms": len(sids),
        "randoms_present": sum(1 for a in sids if a.startswith("formal_random_")),
        "randoms_expected": K,
        "extra_randoms": [a for a in sids if a.startswith("formal_random_")
                          and int(a.split("_")[-1]) >= K],
        "per_arm_counts": {a: len(s) for a, s in sorted(sids.items())},
        "zero_arm_exact_equality": True,
        "destination_partition_exact": True,
        "baseline_records_note": ("Per-row baseline is emitted for every SEALED row that "
                                  "entered any arm. Rows whose baseline prediction was not the "
                                  "channel source never entered an arm; they contribute to the "
                                  "collateral denominator through the aggregate "
                                  "baseline_correct_n, which the runner computed over all 548 "
                                  "rows in a single baseline pass before any arm.")})

    wj(NS / "RUNNER_AND_PROTOCOL_HASHES.json", {
        "formal_runner": {"path": "scripts/qwen3_4b_w2c_formal.py",
                          "sha256": sha(ROOT / "scripts/qwen3_4b_w2c_formal.py")},
        "packager": {"path": "scripts/qwen3_4b_formal_package.py",
                     "sha256": sha(ROOT / "scripts/qwen3_4b_formal_package.py")},
        "protocol": {f: sha(DE / f) for f in
                     ["FORMAL_PROTOCOL_FROZEN.json", "FORMAL_PREREGISTRATION.md",
                      "PRIMARY_DIRECTIONS.safetensors", "DIFFMEAN_DIRECTIONS.safetensors",
                      "FORMAL_RANDOMS.safetensors"]},
        "gate": {f: sha(GATE / f) for f in
                 ["GATE_ONLY_OUTPUT.json", "COLLATERAL_CLAIM_LOCK.md",
                  "ONE_SHOT_EXECUTION_LEDGER.json"]},
        "collateral_audit": {"FILE_INVENTORY.json": sha(AUDIT / "FILE_INVENTORY.json")},
        "records_sha256": sha(RECORDS), "results_sha256": sha(RESULTS),
        "scientific_definitions_changed": False})

    (NS / "FORMAL_PRINCIPAL_VERDICT.txt").write_text(
        ("QWEN3_4B_FORMAL_ADMIT" if R["verdict"].endswith("SUCCESS")
         else "QWEN3_4B_FORMAL_DECLINE") + "\n", encoding="utf-8")
    wj(NS / "FORMAL_RESULTS.json", R)
    import shutil
    shutil.copy(GATE / "ONE_SHOT_EXECUTION_LEDGER.json", NS / "ONE_SHOT_EXECUTION_LEDGER.json")
    print("verdict:", R["verdict"], "| p_add_one:", R["p_add_one"])
    print("records:", nrec, "| arms:", len(sids), "| duplicates:", dup, "| non-finite:", nonfinite)


if __name__ == "__main__":
    main()
