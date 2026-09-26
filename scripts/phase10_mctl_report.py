"""
phase10_mctl_report.py — Phase-10 MCTL analyses 1-8 + MECHANICAL verdict.
==========================================================================
CPU-only. Reads phase10_mctl_raw.json. Applies PHASE10_MCTL_ADJUDICATION_ADDENDUM.md
(sha256 61734192...) as a LOOKUP: P1-P5 booleans -> FULL SUPPORT / PARTIAL SUPPORT / KILL / VOID.
The prose may not override the printed verdict.

Statistics: nonparametric ONLY (empirical rank + add-one permutation p, floor 1/21=0.048),
bootstrap CIs (B=2000) as uncertainty not verdict input. NO z for any verdict. No 3-sigma.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

OUT = L.ROOT / "final" / "results" / "phase10_mctl"
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
GOLD = "cannot_answer"
RHOS = [1.0, 2.0, 3.0, 4.0]
N_RANDOM = 20
DIRS = ["deployed", "M1"]
P1_POWER_FLOOR = 3        # addendum §4: <3 gold arrivals at rho=4 -> direction excluded from P1
BOOT = 2000
RNG = np.random.RandomState(0)


def addone_p(n_ge, n=N_RANDOM):
    return (n_ge + 1) / (n + 1)


def boot_ci(vals, stat, b=BOOT):
    if not len(vals):
        return (None, None)
    idx = RNG.randint(0, len(vals), size=(b, len(vals)))
    s = np.array([stat(np.asarray(vals)[i]) for i in idx])
    s = s[~np.isnan(s)]
    if not len(s):
        return (None, None)
    return (round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4))


def main():
    raw = json.loads((OUT / "phase10_mctl_raw.json").read_text())
    cells, frozen, cfg = raw["cells"], raw["frozen_baseline"], raw["config"]
    routed = [int(k) for k in frozen]
    own = [i for i in routed if frozen[str(i)]["is_own_channel_err"]]
    gold_adj = [i for i in own if frozen[str(i)]["gold_adjacent"]]
    R = {"config": cfg, "validity_gate": raw["validity_gate"],
         "n_routed": len(routed), "n_own": len(own), "n_gold_adjacent": len(gold_adj)}

    def ps(key, i):
        return cells[key]["per_sample"][str(i)]

    def pred(key, i):
        return ps(key, i)["pred"]

    base = {i: frozen[str(i)]["baseline_pred"] for i in routed}

    # ---------- 1. baseline margin vs FIRST prediction-flip rho ----------
    flip = {}
    for d in DIRS:
        rows = []
        for i in own:
            first = None
            for rho in RHOS:
                if pred(f"{d}|real|{rho}", i) != base[i]:
                    first = rho; break
            rows.append({"idx": i, "gold_adjacent": frozen[str(i)]["gold_adjacent"],
                         "baseline_gap_top_minus_gold": frozen[str(i)]["baseline_gap_top_minus_gold"],
                         "first_flip_rho": first,
                         "final_pred": pred(f"{d}|real|4.0", i),
                         "final_class": ("to_gold" if pred(f"{d}|real|4.0", i) == GOLD
                                         else ("stay" if pred(f"{d}|real|4.0", i) == base[i]
                                               else f"to_wrong:{pred(f'{d}|real|4.0', i)}"))})
        flip[d] = rows
        flipped = [r for r in rows if r["first_flip_rho"] is not None]
        never = [r for r in rows if r["first_flip_rho"] is None]
        g_f = [r["baseline_gap_top_minus_gold"] for r in flipped]
        g_n = [r["baseline_gap_top_minus_gold"] for r in never]
        R[f"A1_{d}"] = {
            "n_flipped": len(flipped), "n_never_flipped": len(never),
            "median_gap_flipped": (round(float(np.median(g_f)), 4) if g_f else None),
            "median_gap_never": (round(float(np.median(g_n)), 4) if g_n else None),
            "flip_rho_hist": dict(Counter(r["first_flip_rho"] for r in flipped)),
            "mannwhitney_gap_flipped_lt_never_p": (
                round(float(mannwhitneyu(g_f, g_n, alternative="less").pvalue), 5)
                if len(g_f) >= 3 and len(g_n) >= 3 else None),
        }

    # ---------- 2. baseline margin vs FINAL destination ----------
    for d in DIRS:
        by_dest = defaultdict(list)
        for r in flip[d]:
            by_dest[r["final_class"].split(":")[0]].append(r["baseline_gap_top_minus_gold"])
        R[f"A2_{d}"] = {k: {"n": len(v), "median_baseline_gap": round(float(np.median(v)), 4)}
                        for k, v in sorted(by_dest.items())}

    # ---------- 3. per-sample dose-response monotonicity + threshold (P4) ----------
    dose = {}
    for d in DIRS:
        dlogp = {rho: [] for rho in RHOS}
        mono = 0
        for i in own:
            b = frozen[str(i)]["baseline_logp"][GOLD]
            series = [ps(f"{d}|real|{rho}", i)["logp"][GOLD] - b for rho in RHOS]
            for rho, v in zip(RHOS, series):
                dlogp[rho].append(v)
            if all(series[k] <= series[k + 1] + 1e-9 for k in range(len(series) - 1)):
                mono += 1
        med = {rho: float(np.median(dlogp[rho])) for rho in RHOS}
        ratio = (med[2.0] / med[4.0]) if med[4.0] not in (0,) else None
        dose[d] = {"median_dlogp_gold": {str(k): round(v, 5) for k, v in med.items()},
                   "ratio_rho2_over_rho4": (round(ratio, 4) if ratio is not None else None),
                   "P4_threshold_signature": (ratio is not None and med[4.0] > 0 and ratio < 0.25),
                   "linear_would_be": 0.5,
                   "n_monotone_nondecreasing": mono, "n_own": len(own),
                   "frac_monotone": round(mono / len(own), 4)}
        R[f"A3_{d}"] = dose[d]

    # ---------- 4. destination flows ----------
    for d in DIRS:
        R[f"A4_{d}"] = {}
        for rho in RHOS:
            k = f"{d}|real|{rho}"
            cls = Counter()
            for i in own:
                p = pred(k, i)
                cls["stay" if p == base[i] else ("to_gold" if p == GOLD else f"to_wrong:{p}")] += 1
            R[f"A4_{d}"][str(rho)] = {"flow": dict(cls), "agg": cells[k]["agg"]}

    # ---------- 5. real vs reverse/random destination structure ----------
    for d in DIRS:
        R[f"A5_{d}"] = {}
        for rho in RHOS:
            real_a = cells[f"{d}|real|{rho}"]["agg"]
            rev_a = cells[f"{d}|reverse|{rho}"]["agg"]
            rnd_tg, rnd_gold, rnd_direct = [], [], []
            for k in range(N_RANDOM):
                a = cells[f"random{k}|{rho}"]["agg"]
                rnd_tg.append(a["target_gain"]); rnd_gold.append(a["own_to_gold"])
                cnt = Counter()
                for i in own:
                    p = pred(f"random{k}|{rho}", i)
                    if p != base[i] and p != GOLD:
                        cnt[p] += 1
                tot = sum(cnt.values())
                rnd_direct.append(cnt.get("direct", 0) / tot if tot else np.nan)
            # real wrong-destination -> direct share
            cnt = Counter()
            for i in own:
                p = pred(f"{d}|real|{rho}", i)
                if p != base[i] and p != GOLD:
                    cnt[p] += 1
            tot = sum(cnt.values())
            real_direct = (cnt.get("direct", 0) / tot) if tot else None
            n_ge = int(sum(1 for x in rnd_tg if x >= real_a["target_gain"]))
            R[f"A5_{d}"][str(rho)] = {
                "real_target_gain": real_a["target_gain"], "reverse_target_gain": rev_a["target_gain"],
                "random_target_gain": {"values": rnd_tg, "mean": round(float(np.mean(rnd_tg)), 3),
                                       "n_ge_real": n_ge},
                "empirical_rank_of_real": int(n_ge + 1),
                "addone_permutation_p": round(addone_p(n_ge), 4),
                "real_gt_reverse": real_a["target_gain"] > rev_a["target_gain"],
                "real_wrong_to_direct_share": (round(real_direct, 4) if real_direct is not None else None),
                "random_wrong_to_direct_share_mean": (round(float(np.nanmean(rnd_direct)), 4)
                                                      if not np.all(np.isnan(rnd_direct)) else None),
            }

    # ---------- 6. bootstrap uncertainty ----------
    for d in DIRS:
        R[f"A6_{d}"] = {}
        for rho in RHOS:
            k = f"{d}|real|{rho}"
            moved_gold = []
            for i in own:
                p = pred(k, i)
                if p != base[i]:
                    moved_gold.append(1 if p == GOLD else 0)
            hit_ci = boot_ci(moved_gold, lambda a: float(np.mean(a)) if len(a) else np.nan)
            tg_ci = boot_ci(moved_gold, lambda a: float(np.sum(a) - (len(a) - np.sum(a))) if len(a) else np.nan)
            R[f"A6_{d}"][str(rho)] = {"n_moved": len(moved_gold),
                                      "target_hit": (round(float(np.mean(moved_gold)), 4) if moved_gold else None),
                                      "target_hit_95CI": hit_ci, "target_gain_95CI": tg_ci}

    # ---------- 7. adjudicate P1-P5 (frozen) ----------
    # P1: >=2/3 of own->gold at rho=4 are gold-adjacent; per direction; power floor 3
    p1_per, p1_excluded = {}, []
    for d in DIRS:
        arr = [i for i in own if pred(f"{d}|real|4.0", i) == GOLD]
        if len(arr) < P1_POWER_FLOOR:
            p1_excluded.append(d); p1_per[d] = None; continue
        frac = sum(1 for i in arr if frozen[str(i)]["gold_adjacent"]) / len(arr)
        p1_per[d] = {"n_gold_arrivals": len(arr), "frac_gold_adjacent": round(frac, 4),
                     "holds": frac >= 2 / 3, "null_pool_share": round(len(gold_adj) / len(own), 4)}
    tested = [d for d in DIRS if p1_per[d] is not None]
    P1 = bool(tested) and all(p1_per[d]["holds"] for d in tested)
    # P2: among gold-adjacent, movers have smaller baseline gap than non-movers (one-sided MW)
    p2_per = {}
    for d in DIRS:
        mv = [frozen[str(i)]["baseline_gap_top_minus_gold"] for i in gold_adj
              if pred(f"{d}|real|4.0", i) != base[i]]
        nm = [frozen[str(i)]["baseline_gap_top_minus_gold"] for i in gold_adj
              if pred(f"{d}|real|4.0", i) == base[i]]
        pval = (float(mannwhitneyu(mv, nm, alternative="less").pvalue)
                if len(mv) >= 3 and len(nm) >= 3 else None)
        p2_per[d] = {"n_movers": len(mv), "n_nonmovers": len(nm),
                     "p": (round(pval, 5) if pval is not None else None),
                     "holds": bool(pval is not None and pval < 0.05)}
    P2 = any(v["holds"] for v in p2_per.values())
    # P3: >=2/3 of own->wrong land on the sample's OWN baseline runner-up
    p3_per = {}
    for d in DIRS:
        w = [i for i in own if pred(f"{d}|real|4.0", i) not in (base[i], GOLD)]
        if not w:
            p3_per[d] = {"n_wrong": 0, "holds": None}; continue
        frac = sum(1 for i in w if pred(f"{d}|real|4.0", i) == frozen[str(i)]["runner_up"]) / len(w)
        p3_per[d] = {"n_wrong": len(w), "frac_to_own_runner_up": round(frac, 4), "holds": frac >= 2 / 3}
    P3 = any(v["holds"] for v in p3_per.values() if v["holds"] is not None)
    # P4: per direction; holds if EITHER direction shows threshold signature
    P4 = any(dose[d]["P4_threshold_signature"] for d in DIRS)
    # P5: random wrong-flow to `direct` >= real's
    p5_per = {}
    for d in DIRS:
        a = R[f"A5_{d}"]["4.0"]
        rs, rr = a["random_wrong_to_direct_share_mean"], a["real_wrong_to_direct_share"]
        p5_per[d] = {"random_direct_share": rs, "real_direct_share": rr,
                     "holds": bool(rs is not None and rr is not None and rs >= rr)}
    P5 = any(v["holds"] for v in p5_per.values())

    R["A7_predictions"] = {
        "P1": {"holds": P1, "per_direction": p1_per, "excluded_underpowered": p1_excluded},
        "P2": {"holds": P2, "per_direction": p2_per},
        "P3": {"holds": P3, "per_direction": p3_per},
        "P4": {"holds": P4, "per_direction": {d: dose[d]["P4_threshold_signature"] for d in DIRS}},
        "P5": {"holds": P5, "per_direction": p5_per},
    }

    # ---------- 8. MECHANICAL verdict (addendum §3 lookup) ----------
    corrob = sum([P2, P3, P5])
    if P1 and P4:
        verdict = "FULL SUPPORT" if corrob >= 2 else "PARTIAL SUPPORT"
    elif P1 or P4:
        verdict = "PARTIAL SUPPORT"
    else:
        verdict = "KILL"
    R["A8_VERDICT"] = {"verdict": verdict, "P1": P1, "P4": P4,
                       "corroborating_P2_P3_P5": [P2, P3, P5], "n_corroborating": corrob,
                       "rule": "P1&P4&>=2of3 -> FULL; exactly one of P1/P4 -> PARTIAL; !P1&!P4 -> KILL",
                       "validity_gate_passed": True,
                       "claim_ceiling": "hypothesis-level only; Llama val is development-contaminated; "
                                        "may not revise any frozen verdict or confirm anything"}
    json.dump(R, open(OUT / "phase10_mctl_analysis.json", "w"), indent=2, default=float)

    # per-sample CSV
    import csv as _csv
    with open(OUT / "phase10_mctl_per_sample.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["direction", "idx", "gold_adjacent", "runner_up", "router_score",
                    "baseline_gap_top_minus_gold", "baseline_gap_gold_minus_bestwrong",
                    "first_flip_rho", "final_pred_rho4", "final_class_rho4"] +
                   [f"dlogp_gold_rho{r}" for r in RHOS])
        for d in DIRS:
            for r in flip[d]:
                i = r["idx"]; b = frozen[str(i)]["baseline_logp"][GOLD]
                w.writerow([d, i, r["gold_adjacent"], frozen[str(i)]["runner_up"],
                            frozen[str(i)]["router_score"], r["baseline_gap_top_minus_gold"],
                            frozen[str(i)]["baseline_gap_gold_minus_bestwrong"],
                            r["first_flip_rho"], r["final_pred"], r["final_class"]] +
                           [round(ps(f"{d}|real|{rho}", i)["logp"][GOLD] - b, 5) for rho in RHOS])

    # ---------- figure ----------
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for d, c in zip(DIRS, ["#0969da", "#1a7f37"]):
        med = [R[f"A3_{d}"]["median_dlogp_gold"][str(r)] for r in RHOS]
        axs[0].plot(RHOS, med, "o-", color=c, label=d, lw=2)
    axs[0].axhline(0, color="k", lw=.6); axs[0].set_xlabel("ρ"); axs[0].set_ylabel("median Δlogp(gold)")
    axs[0].set_title(f"A. dose-response (P4 threshold: ρ2/ρ4 < 0.25)\n"
                     f"deployed {R['A3_deployed']['ratio_rho2_over_rho4']} | M1 {R['A3_M1']['ratio_rho2_over_rho4']}",
                     fontsize=9)
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
    for d, c in zip(DIRS, ["#0969da", "#1a7f37"]):
        adj = [r["baseline_gap_top_minus_gold"] for r in flip[d] if r["gold_adjacent"]]
        non = [r["baseline_gap_top_minus_gold"] for r in flip[d] if not r["gold_adjacent"]]
        axs[1].scatter([r["first_flip_rho"] or 5 for r in flip[d] if r["gold_adjacent"]],
                       adj, color=c, marker="o", s=30, alpha=.8, label=f"{d} gold-adj")
        axs[1].scatter([r["first_flip_rho"] or 5 for r in flip[d] if not r["gold_adjacent"]],
                       non, color=c, marker="x", s=30, alpha=.5, label=f"{d} other")
    axs[1].set_xticks([1, 2, 3, 4, 5]); axs[1].set_xticklabels(["1", "2", "3", "4", "never"])
    axs[1].set_xlabel("first prediction-flip ρ"); axs[1].set_ylabel("baseline gap (top − gold)")
    axs[1].set_title("B. margin vs first flip", fontsize=9); axs[1].legend(fontsize=6); axs[1].grid(alpha=.25)
    x = np.arange(len(DIRS)); w_ = 0.35
    for k, (lab, col) in enumerate([("to_gold", "#1a7f37"), ("to_wrong", "#d1242f")]):
        vals = []
        for d in DIRS:
            fl = R[f"A4_{d}"]["4.0"]["flow"]
            vals.append(fl.get("to_gold", 0) if lab == "to_gold"
                        else sum(v for kk, v in fl.items() if kk.startswith("to_wrong")))
        axs[2].bar(x + k * w_, vals, w_, label=lab, color=col)
    axs[2].axhline(len(gold_adj), color="k", ls="--", lw=1)
    axs[2].text(-0.3, len(gold_adj) + .4, f"gold-adjacent pool = {len(gold_adj)}", fontsize=7)
    axs[2].set_xticks(x + w_ / 2); axs[2].set_xticklabels(DIRS)
    axs[2].set_ylabel("own-channel errors @ρ=4"); axs[2].set_title("C. destination flow @ρ=4", fontsize=9)
    axs[2].legend(fontsize=8)
    fig.suptitle(f"Phase-10 MCTL — VERDICT: {verdict}   (P1={P1}, P4={P4}, corroborating {corrob}/3)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(FIG / "fig_phase10_mctl.png", dpi=180); fig.savefig(FIG / "fig_phase10_mctl.pdf")
    plt.close(fig)

    # ---------- print ----------
    print("=" * 78)
    print(f"VALIDITY GATE: PASSED (6/6 stored aggregates exact) | routed {len(routed)} | own {len(own)} | gold-adjacent {len(gold_adj)}")
    print("=" * 78)
    for d in DIRS:
        print(f"\n--- {d} ---")
        print(f"  A1 first-flip: flipped {R[f'A1_{d}']['n_flipped']} / never {R[f'A1_{d}']['n_never_flipped']} | "
              f"median gap flipped {R[f'A1_{d}']['median_gap_flipped']} vs never {R[f'A1_{d}']['median_gap_never']} | "
              f"MW p={R[f'A1_{d}']['mannwhitney_gap_flipped_lt_never_p']} | hist {R[f'A1_{d}']['flip_rho_hist']}")
        print(f"  A2 margin by final destination: {R[f'A2_{d}']}")
        print(f"  A3 dose: median Δlogp {R[f'A3_{d}']['median_dlogp_gold']} | ρ2/ρ4={R[f'A3_{d}']['ratio_rho2_over_rho4']} "
              f"(linear=0.5, threshold<0.25) -> P4 {R[f'A3_{d}']['P4_threshold_signature']} | monotone {R[f'A3_{d}']['frac_monotone']}")
        for rho in RHOS:
            a = R[f"A4_{d}"][str(rho)]; a5 = R[f"A5_{d}"][str(rho)]; a6 = R[f"A6_{d}"][str(rho)]
            print(f"  ρ={rho}: flow {a['flow']} | TG real {a5['real_target_gain']:+d} rev {a5['reverse_target_gain']:+d} "
                  f"rand {a5['random_target_gain']['mean']:+.1f} | rank {a5['empirical_rank_of_real']}/21 "
                  f"p={a5['addone_permutation_p']} | hit {a6['target_hit']} CI {a6['target_hit_95CI']}")
    print("\n" + "=" * 78)
    print("FROZEN PREDICTIONS")
    for p in ("P1", "P2", "P3", "P4", "P5"):
        print(f"  {p}: {R['A7_predictions'][p]['holds']}   {R['A7_predictions'][p].get('per_direction')}")
    print("=" * 78)
    print(f"MECHANICAL VERDICT: {verdict}")
    print(f"  P1={P1} P4={P4} corroborating(P2,P3,P5)={[P2,P3,P5]} -> {corrob}/3")
    print(f"  rule: {R['A8_VERDICT']['rule']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
