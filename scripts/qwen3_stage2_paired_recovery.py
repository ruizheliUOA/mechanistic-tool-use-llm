#!/usr/bin/env python3
"""Qwen3 SAKIKO Stage 2 — paired-outcome recovery.

Read-only. Loads no model, runs no inference, touches no GPU, opens no
evaluation prompt and no source dataset payload. It reads only the per-sample
outputs already committed by the single authorized formal execution and
recovers the paired comparisons that the frozen preregistration specified but
the formal runner never emitted.

Two analysis classes are produced and are labelled distinctly everywhere:

  LATE-COMPUTED RECOVERY OF PRESPECIFIED SECONDARIES
      preregistration secondary.ordered[1]:
      "d_grad versus frozen score-space comparator: paired difference in
       target_gain_rate and wrong_to_wrong rate"

  POST-HOC EXPLORATORY SENSITIVITY ANALYSIS
      everything else computed here.

Nothing here can alter, rescue or extend the formal primary result, which was
and remains FORMAL_CONFIRMATORY_SUCCESS. The preregistration marks all
secondaries non_promotable and must_not_rescue_failed_primary.

No row identifier, UUID, prompt, label, prediction string or per-example
record is written to any output.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "final/results/qwen3_stage2_formal"
OUT = ROOT / "final/results/qwen3_stage2_paired_recovery"

RECORDS = FORMAL / "QWEN3_STAGE2_FORMAL_RECORDS.jsonl"
RESULTS = FORMAL / "QWEN3_STAGE2_FORMAL_RESULTS.json"
PREREG = FORMAL / "QWEN3_STAGE2_PREREGISTRATION.json"
HASHES = FORMAL / "QWEN3_STAGE2_FORMAL_RESULT_HASHES.json"

GOLD, SOURCE = "cannot_answer", "tool_call"
ACT_ARM, CMP_ARM, ALL_SRC_ARM = "d_grad", "score_space_comparator", "ungated_d_grad"

# ---- declared before any statistic was computed -------------------------
BOOT_SALT = "SAKIKO_QWEN3_PAIRED_RECOVERY_V1"   # distinct from the formal salt
BOOT_DRAWS = 10_000
ALPHA = 0.05
SIDEDNESS = "two-sided"   # the preregistration defines no directionality
BOOT_SEED = int.from_bytes(hashlib.sha256(BOOT_SALT.encode()).digest()[:8], "big")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- ingest
def load_arms():
    """arm -> {sample_id: (gold, pred_base, pred_int)}. Ids never leave this scope."""
    keep = {ACT_ARM, CMP_ARM, ALL_SRC_ARM}
    arms: dict[str, dict[str, tuple[str, str, str]]] = {a: {} for a in keep}
    with RECORDS.open() as fh:
        for line in fh:
            r = json.loads(line)
            a = r["arm"]
            if a in keep:
                arms[a][r["sample_id"]] = (r["gold"], r["pred_base"], r["pred_int"])
    return arms


def classify(pred: str) -> str:
    if pred == GOLD:
        return "GOLD"
    if pred == SOURCE:
        return "SOURCE"
    return "OTHER_WRONG"


UTILITY = {"GOLD": 1, "SOURCE": 0, "OTHER_WRONG": -1}
CATS = ["GOLD", "SOURCE", "OTHER_WRONG"]


# ------------------------------------------------------------ statistics
def exact_mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial sign test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def wilson_paired_rd_ci(b: int, c: int, n: int, z: float = 1.959963984540054):
    """Agresti-Min style CI for the paired risk difference (b - c)/n."""
    if n == 0:
        return [None, None]
    d = (b - c) / n
    # variance of the paired difference of proportions
    var = (b + c - (b - c) ** 2 / n) / (n ** 2)
    se = var ** 0.5
    return [d - z * se, d + z * se]


def exact_sign_randomization(d: list[int]) -> dict:
    """Exact null distribution of sum(eps_i * d_i), eps_i uniform in {-1,+1}.

    Exact enumeration by convolution over the multiset of nonzero |d_i|;
    2**n enumeration is infeasible but the integer-valued convolution is not.
    Probabilities are exact rationals over 2**m (m = number of nonzero d_i).
    """
    nz = [abs(x) for x in d if x != 0]
    m = len(nz)
    obs = sum(d)
    dist = {0: Fraction(1)}
    for a in nz:
        nxt: dict[int, Fraction] = {}
        for s, p in dist.items():
            half = p / 2
            nxt[s + a] = nxt.get(s + a, Fraction(0)) + half
            nxt[s - a] = nxt.get(s - a, Fraction(0)) + half
        dist = nxt
    p_two = sum(p for s, p in dist.items() if abs(s) >= abs(obs))
    p_ge = sum(p for s, p in dist.items() if s >= obs)
    return {
        "method": "exact within-pair arm-label sign randomization (full enumeration by convolution)",
        "monte_carlo": False,
        "n_nonzero_pairs": m,
        "support_size": len(dist),
        "observed_sum": obs,
        "p_two_sided": float(p_two),
        "p_one_sided_activation_favoured_EXPLORATORY": float(p_ge),
        "null_mean": 0.0,
        "null_sd": float(sum(a * a for a in nz) ** 0.5),
    }


def paired_bootstrap(d: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(d)
    draws = np.empty(BOOT_DRAWS)
    for i in range(BOOT_DRAWS):
        draws[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
    return {
        "method": "paired nonparametric bootstrap over the 87 paired examples, percentile",
        "seed_salt": BOOT_SALT,
        "seed": BOOT_SEED,
        "draws": BOOT_DRAWS,
        "mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ------------------------------------------------------------------ main
def main() -> None:
    committed = json.loads(HASHES.read_text())["artifacts"]
    integrity = {}
    for rel, meta in committed.items():
        p = ROOT / rel
        integrity[rel] = {"recomputed": sha256_file(p), "committed": meta["sha256"]}
        if integrity[rel]["recomputed"] != meta["sha256"]:
            raise SystemExit(f"PAIRED_RECOVERY_PRECHECK_FAILED: hash mismatch {rel}")

    res = json.loads(RESULTS.read_text())
    arms = load_arms()

    # the 87 formal-channel baseline errors: the ungated arm's population is
    # every row whose baseline prediction is the source mode, so it is a
    # superset of the channel-error set and lets us enumerate it exactly.
    err = sorted(sid for sid, (g, pb, _) in arms[ALL_SRC_ARM].items()
                 if g == GOLD and pb == SOURCE)
    routed_act = set(arms[ACT_ARM])
    routed_cmp = set(arms[CMP_ARM])
    if routed_act != routed_cmp:
        raise SystemExit("PAIRED_RECOVERY_UNRECONSTRUCTABLE: arms cover different populations")

    # rows the Router did not fire on are frozen at baseline under both arms
    def dest(arm: str, sid: str) -> str:
        rec = arms[arm].get(sid)
        return classify(rec[2]) if rec is not None else SOURCE and "SOURCE"

    act = [dest(ACT_ARM, s) for s in err]
    cmp_ = [dest(CMP_ARM, s) for s in err]

    def marginals(labels):
        c = Counter(labels)
        return {"gold_arrivals": c["GOLD"], "wrong_to_wrong": c["OTHER_WRONG"],
                "source_remains": c["SOURCE"],
                "source_exits": c["GOLD"] + c["OTHER_WRONG"],
                "target_gain_count": c["GOLD"] - c["OTHER_WRONG"]}

    m_act, m_cmp = marginals(act), marginals(cmp_)
    exp_act = {"gold_arrivals": 38, "wrong_to_wrong": 14, "source_remains": 35,
               "source_exits": 52, "target_gain_count": 24}
    exp_cmp = {"gold_arrivals": 37, "wrong_to_wrong": 21, "source_remains": 29,
               "source_exits": 58, "target_gain_count": 16}
    checks = {
        "n_channel_error_is_87": len(err) == 87,
        "activation_marginals_match_committed": m_act == exp_act,
        "comparator_marginals_match_committed": m_cmp == exp_cmp,
        "arms_share_identical_routed_population": True,
        "routed_within_channel": len(routed_act & set(err)),
        "unrouted_within_channel": len(err) - len(routed_act & set(err)),
        "committed_real": {k: res["primary"]["real"][k] for k in
                           ("n_channel_error", "source_exits", "gold_arrivals",
                            "wrong_to_wrong", "target_gain_count")},
        "committed_comparator": {k: res["secondary"][CMP_ARM][k] for k in
                                 ("n_channel_error", "source_exits", "gold_arrivals",
                                  "wrong_to_wrong", "target_gain_count")},
    }
    if not (checks["n_channel_error_is_87"] and checks["activation_marginals_match_committed"]
            and checks["comparator_marginals_match_committed"]):
        raise SystemExit("PAIRED_RECOVERY_UNRECONSTRUCTABLE: marginals disagree with committed result")

    # ---- 3x3 paired destination table
    table = {a: {c: 0 for c in CATS} for a in CATS}
    for a, c in zip(act, cmp_):
        table[a][c] += 1
    agree = sum(table[c][c] for c in CATS)

    # ---- Claim C: paired destination utility  (mean = paired target_gain_rate difference)
    d = np.array([UTILITY[a] - UTILITY[c] for a, c in zip(act, cmp_)], dtype=int)
    freq = {str(v): int((d == v).sum()) for v in range(-2, 3)}
    utility = {
        "label": "LATE-COMPUTED RECOVERY OF PRESPECIFIED SECONDARIES",
        "prereg_item": "secondary.ordered[1] - paired difference in target_gain_rate",
        "n_pairs": int(len(d)),
        "total_utility_activation": int(sum(UTILITY[x] for x in act)),
        "total_utility_comparator": int(sum(UTILITY[x] for x in cmp_)),
        "total_paired_difference": int(d.sum()),
        "mean_paired_difference": float(d.mean()),
        "equals_paired_target_gain_rate_difference": float(d.mean()),
        "n_d_gt_0": int((d > 0).sum()), "n_d_eq_0": int((d == 0).sum()),
        "n_d_lt_0": int((d < 0).sum()),
        "frequency_distribution": freq,
        "bootstrap": paired_bootstrap(d),
        "randomization": exact_sign_randomization(d.tolist()),
        "alpha": ALPHA, "sidedness": SIDEDNESS,
    }

    # ---- binary paired decompositions
    def binary(pred_a, pred_c, name, label, prereg_item):
        n11 = sum(1 for x, y in zip(pred_a, pred_c) if x and y)
        n00 = sum(1 for x, y in zip(pred_a, pred_c) if not x and not y)
        b = sum(1 for x, y in zip(pred_a, pred_c) if x and not y)     # activation-only
        c = sum(1 for x, y in zip(pred_a, pred_c) if y and not x)     # comparator-only
        return {"name": name, "label": label, "prereg_item": prereg_item,
                "both_positive": n11, "both_negative": n00,
                "activation_only": b, "comparator_only": c,
                "mcnemar_exact_p_two_sided": exact_mcnemar(b, c),
                "paired_risk_difference": (b - c) / len(pred_a),
                "paired_risk_difference_ci95": wilson_paired_rd_ci(b, c, len(pred_a))}

    binaries = [
        binary([x == "OTHER_WRONG" for x in act], [x == "OTHER_WRONG" for x in cmp_],
               "B_other_wrong_arrival", "LATE-COMPUTED RECOVERY OF PRESPECIFIED SECONDARIES",
               "secondary.ordered[1] - paired difference in wrong_to_wrong rate"),
        binary([x == "GOLD" for x in act], [x == "GOLD" for x in cmp_],
               "A_gold_arrival", "POST-HOC EXPLORATORY SENSITIVITY ANALYSIS",
               "not prespecified as a paired comparison"),
        binary([x != "SOURCE" for x in act], [x != "SOURCE" for x in cmp_],
               "C_source_exit", "POST-HOC EXPLORATORY SENSITIVITY ANALYSIS",
               "not prespecified as a paired comparison"),
    ]

    # ---- post-hoc composition test (Stuart-Maxwell, marginal homogeneity)
    # standard (k-1)-dimensional form: d_i = row_i - col_i for i < k,
    # V_ii = n_i. + n_.i - 2 n_ii ; V_ij = -(n_ij + n_ji).
    n = [[table[CATS[i]][CATS[j]] for j in range(3)] for i in range(3)]
    row = [sum(n[i]) for i in range(3)]
    col = [sum(n[i][j] for i in range(3)) for j in range(3)]
    dvec = np.array([row[i] - col[i] for i in range(2)], float)
    V = np.zeros((2, 2))
    for i in range(2):
        V[i, i] = row[i] + col[i] - 2 * n[i][i]
        for j in range(2):
            if i != j:
                V[i, j] = -(n[i][j] + n[j][i])
    try:
        stat = float(dvec @ np.linalg.solve(V, dvec))
        p_sm = float(np.exp(-stat / 2.0))   # chi-square, 2 df, survival function
    except Exception:
        stat, p_sm = None, None
    composition = {
        "label": "POST-HOC EXPLORATORY COMPOSITION TEST",
        "prereg_item": "full 3x3 destination composition was NOT prespecified",
        "stuart_maxwell_statistic": stat, "df": 2, "p_two_sided": p_sm,
        "cannot_establish_or_rescue_primary": True,
    }

    # ---- post-hoc all-548 paired correctness
    # rows neither arm touched are identical under both arms and are concordant
    b548 = c548 = 0
    for sid in routed_act:
        g, _, pia = arms[ACT_ARM][sid]
        _, _, pic = arms[CMP_ARM][sid]
        ca, cc = pia == g, pic == g
        b548 += int(ca and not cc)
        c548 += int(cc and not ca)
    all548 = {
        "label": "POST-HOC EXPLORATORY SENSITIVITY ANALYSIS",
        "prereg_item": "all-548 paired correctness was NOT prespecified as a paired comparison",
        "scope": "discordant pairs can only arise on rows at least one arm modified",
        "activation_correct_comparator_wrong": b548,
        "comparator_correct_activation_wrong": c548,
        "mcnemar_exact_p_two_sided": exact_mcnemar(b548, c548),
        "committed_activation_fixed_broke_net": [res["primary"]["real"]["fixed"],
                                                 res["primary"]["real"]["broke"],
                                                 res["primary"]["real"]["net"]],
        "committed_comparator_fixed_broke_net": [res["secondary"][CMP_ARM]["fixed"],
                                                 res["secondary"][CMP_ARM]["broke"],
                                                 res["secondary"][CMP_ARM]["net"]],
    }

    out = {
        "schema_version": 1,
        "analysis": "QWEN3_STAGE2_PAIRED_RECOVERY",
        "not_part_of_formal_execution": True,
        "formal_primary_unchanged": res["outcome"],
        "declared_before_computation": {
            "alpha": ALPHA, "sidedness": SIDEDNESS,
            "bootstrap_seed_salt": BOOT_SALT, "bootstrap_seed": BOOT_SEED,
            "bootstrap_draws": BOOT_DRAWS,
            "randomization": "exact enumeration by convolution",
            "note": "the preregistration names the endpoints but no method, alpha, "
                    "directionality or interval; these are declared here and labelled",
        },
        "integrity": integrity,
        "reconstruction_checks": checks,
        "marginals": {"activation": m_act, "score_space_comparator": m_cmp},
        "paired_table_3x3": table,
        "paired_table_totals": {"total": int(sum(sum(r.values()) for r in table.values())),
                                "agreement": agree,
                                "disagreement": 87 - agree},
        "claim_C_paired_destination_utility": utility,
        "binary_paired_decompositions": binaries,
        "composition_analysis": composition,
        "all_548_accuracy_family": all548,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "QWEN3_STAGE2_PAIRED_RECOVERY.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    rows = ["activation_destination,comparator_destination,count"]
    for a in CATS:
        for c in CATS:
            rows.append(f"{a},{c},{table[a][c]}")
    (OUT / "paired_destination_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    brows = ["analysis,label,both_positive,both_negative,activation_only,comparator_only,"
             "mcnemar_exact_p_two_sided,paired_risk_difference,ci95_low,ci95_high"]
    for x in binaries:
        lo, hi = x["paired_risk_difference_ci95"]
        brows.append(f'{x["name"]},{x["label"]},{x["both_positive"]},{x["both_negative"]},'
                     f'{x["activation_only"]},{x["comparator_only"]},'
                     f'{x["mcnemar_exact_p_two_sided"]:.6g},{x["paired_risk_difference"]:.6g},'
                     f"{lo:.6g},{hi:.6g}")
    (OUT / "binary_paired_tables.csv").write_text("\n".join(brows) + "\n", encoding="utf-8")

    print(json.dumps({k: out[k] for k in
                      ("reconstruction_checks", "marginals", "paired_table_3x3",
                       "paired_table_totals", "claim_C_paired_destination_utility",
                       "binary_paired_decompositions", "composition_analysis",
                       "all_548_accuracy_family")}, indent=1, default=str))


if __name__ == "__main__":
    main()
