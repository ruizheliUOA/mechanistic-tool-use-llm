"""
phase9_m1m2_report.py — Phase-9 Part C report: M1/M2 tables + figure + verdict.
================================================================================
CPU-only post-processing of phase9_m1m2_results.json (validation rows only).
Applies the FROZEN success criteria; does not re-tune anything.

Frozen success (all required, per PHASE9_LLAMA_M1M2_PROTOCOL.md §4):
  target-hit >= 0.60 ; redistribution <= 0.40 ; z(TargetGain) >= 3 ; n_ge/N <= 0.05 ;
  real > reverse ; Broke <= Fixed/2 ; collateral bounded ; NO boundary-only success
  (a pass must occur at rho in {1,2}, not only at the grid max rho=4).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L

OUT = L.ROOT / "final" / "results" / "phase9_mechanism_and_ood"
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
RES = OUT / "phase9_m1m2_results.json"
RHO_BOUNDARY = 4.0

# deployed Phase-8 DiffMean baseline on the SAME routed set (from llama_rho_curves.json)
def deployed_baseline():
    c = json.loads((L.ROOT / "final/results/phase8_prospective_llama/llama_rho_curves.json").read_text())["ca_tc"]
    out = {}
    for r in c["rows"]:
        real = r["real"]
        moved = real["own_before"] - real["own_after"]
        out[r["rho"]] = {"target_gain": real["own_to_gold"] - real["own_to_other_wrong"],
                         "target_hit": (round(real["own_to_gold"] / moved, 4) if moved else None),
                         "redist": (round(real["own_to_other_wrong"] / moved, 4) if moved else None),
                         "net": real["net"], "moved": moved,
                         "own": f"{real['own_before']}->{real['own_after']}"}
    return out


def main():
    res = json.loads(RES.read_text())
    base = deployed_baseline()
    flat = []
    for m in ("M1", "M2"):
        if m not in res:
            continue
        for r in sorted(res[m]["rows"], key=lambda x: x["rho"]):
            real, rev, rt = r["real"], r["reverse"], r["random_target_gain"]
            flat.append({
                "method": m, "rho": r["rho"], "delta_norm": r["delta_norm"],
                "n_routed": r["n_routed"], "obs": r["obs"], "inj": r["inj"], "thr": r["thr"],
                "target_gain": real["target_gain"],
                "target_hit_rate": real["target_hit_rate"],
                "redistribution_rate": real["redistribution_rate"],
                "own_before": real["own_before"], "own_after": real["own_after"],
                "own_moved": real["own_moved"], "own_to_gold": real["own_to_gold"],
                "own_to_other_wrong": real["own_to_other_wrong"],
                "fixed": real["fixed"], "broke": real["broke"], "net": real["net"],
                "reverse_target_gain": rev["target_gain"], "reverse_net": rev["net"],
                "rand_tg_mean": rt["mean"], "rand_tg_std": rt["std"], "rand_tg_max": rt["max"],
                "n_ge_real": rt["n_ge_real"], "n_random": rt["n"],
                "z_target_gain": r["z_target_gain"], "z_net": r["z_net"],
                "damage_on_correct": real["damage_on_correct"],
                "cell_pass": r["cell_pass"],
                **{f"crit_{k}": v for k, v in r["criteria"].items()},
            })
    with open(OUT / "phase9_m1m2_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys())); w.writeheader()
        for r in flat:
            w.writerow(r)

    # ---- verdicts ----
    verdicts = {}
    for m in ("M1", "M2"):
        if m not in res:
            continue
        rows = res[m]["rows"]
        interior = [r for r in rows if r["cell_pass"] and r["rho"] < RHO_BOUNDARY]
        boundary_only = [r for r in rows if r["cell_pass"] and r["rho"] >= RHO_BOUNDARY]
        # partial-signal rule (frozen §5): hit in [0.40,0.60) AND z>=3 AND TG>0 at non-boundary rho
        partial = [r for r in rows
                   if r["rho"] < RHO_BOUNDARY
                   and r["real"]["target_hit_rate"] is not None
                   and 0.40 <= r["real"]["target_hit_rate"] < 0.60
                   and r["z_target_gain"] is not None and r["z_target_gain"] >= 3.0
                   and r["real"]["target_gain"] > 0]
        verdicts[m] = {"verdict": "PASS" if interior else "FAIL",
                       "interior_pass_rhos": [r["rho"] for r in interior],
                       "boundary_only_pass_rhos": [r["rho"] for r in boundary_only],
                       "partial_signal_rhos": [r["rho"] for r in partial],
                       "best_target_hit": max((r["real"]["target_hit_rate"] or 0) for r in rows),
                       "best_target_gain": max(r["real"]["target_gain"] for r in rows)}
    json.dump({"verdicts": verdicts, "deployed_baseline": base},
              open(OUT / "phase9_m1m2_verdicts.json", "w"), indent=2, default=float)

    # ---- figure: Target Gain vs rho, real/reverse/random, per method + deployed baseline ----
    ms = [m for m in ("M1", "M2") if m in res]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.8), squeeze=False)
    for ax, m in zip(axes[0], ms):
        rows = sorted(res[m]["rows"], key=lambda x: x["rho"])
        rho = [r["rho"] for r in rows]
        rm = np.array([r["random_target_gain"]["mean"] for r in rows])
        rs = np.array([r["random_target_gain"]["std"] for r in rows])
        ax.fill_between(rho, rm - rs, rm + rs, color="#bbb", alpha=.45, label="random ±1sd")
        ax.plot(rho, rm, "s--", color="#888", ms=4, label="random mean")
        ax.plot(rho, [r["reverse"]["target_gain"] for r in rows], "v--", color="#d1242f", ms=5, label="reverse")
        ax.plot(rho, [r["real"]["target_gain"] for r in rows], "o-", color="#1a7f37", ms=6, lw=2, label="real")
        ax.plot(rho, [base.get(x, {}).get("target_gain", np.nan) for x in rho], "d:", color="#0969da",
                ms=5, label="deployed DiffMean (Ph-8)")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xscale("log", base=2); ax.set_xticks(rho); ax.set_xticklabels([str(x) for x in rho])
        ax.set_title(f"Llama ca_tc — {m}", fontsize=10); ax.set_xlabel("ρ"); ax.grid(alpha=.25)
    axes[0][0].set_ylabel("Target Gain  (own→gold − own→other-wrong)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Phase-9 M1/M2: does an explicit target-vs-wrong-destination selector aim better?\n"
                 "(validation only; 20 matched-norm randoms, seed block 4000+k)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, .88])
    fig.savefig(FIG / "fig_phase9_m1m2_target_gain.png", dpi=180)
    fig.savefig(FIG / "fig_phase9_m1m2_target_gain.pdf")
    plt.close(fig)

    print("=== DEPLOYED Phase-8 DiffMean baseline (same routed set) ===")
    for r, v in sorted(base.items()):
        print(f"  rho={r:<4} TargetGain={v['target_gain']:+4d} hit={v['target_hit']} "
              f"redist={v['redist']} net={v['net']:+3d} own {v['own']}")
    print("\n=== M1/M2 cells ===")
    print(f"{'m':>3} {'rho':>4} {'TG':>4} {'hit':>6} {'redist':>7} {'randTG':>14} {'nge':>6} {'z':>6} "
          f"{'net':>5} {'revTG':>6} {'pass':>5}")
    for r in flat:
        print(f"{r['method']:>3} {r['rho']:>4} {r['target_gain']:>+4d} {str(r['target_hit_rate']):>6} "
              f"{str(r['redistribution_rate']):>7} {r['rand_tg_mean']:>6.1f}±{r['rand_tg_std']:<6.1f} "
              f"{r['n_ge_real']:>2}/{r['n_random']:<3} {str(r['z_target_gain']):>6} {r['net']:>+5d} "
              f"{r['reverse_target_gain']:>+6d} {str(r['cell_pass']):>5}")
    print("\n=== VERDICTS (frozen criteria) ===")
    for m, v in verdicts.items():
        print(f"  {m}: {v['verdict']} | interior passes: {v['interior_pass_rhos']} | "
              f"boundary-only: {v['boundary_only_pass_rhos']} | partial-signal ρ: {v['partial_signal_rhos']} | "
              f"best hit={v['best_target_hit']} best TG={v['best_target_gain']:+d}")
    both_fail = all(v["verdict"] == "FAIL" for v in verdicts.values()) and len(verdicts) == 2
    print(f"\n  BOTH FAIL -> {both_fail}  (if True: STOP the linear-selector line; do NOT run M3)")
    print("wrote", OUT / "phase9_m1m2_results.csv", "and", FIG / "fig_phase9_m1m2_target_gain.png")


if __name__ == "__main__":
    main()
