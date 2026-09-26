"""
acebench_channel_discovery_feasibility.py — Phase 4 Step 12 (CPU only; runs ONLY if R0 allows).
================================================================================================
Applies the archived transition-discovery procedure to the locked generation predictions:
all gold->pred transitions (UNKNOWN predictions excluded — they carry no target mode), per-split
counts, archived threshold tiers (default 50/15, strict 75/25, lenient 30/10), 200-resample
bootstrap, plus the archived SAKIKO-CA intervention-scope filter (train>=30, val>=5, test>=5,
train-correct ref pool >= 30). NO activations, NO routers, NO intervention.

Outputs -> final/results/acebench_generation_readout/channel_discovery/
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "final/results/acebench_generation_readout"
CD = RES / "channel_discovery"
DET = ROOT / ".cache/acebench_phase4/generation_details_full.jsonl"

TIERS = {"default": (50, 15), "strict": (75, 25), "lenient": (30, 10)}
SCOPE = {"train_min": 30, "val_min": 5, "test_min": 5, "ref_min": 30}
SEM = {
    "tool_call": "call the provided function(s)",
    "ask_user": "ask for a missing required parameter",
    "flag_param_error": "point out an invalid provided value",
    "cannot_comply": "state that no provided function can do it",
}


def main():
    CD.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(DET, encoding="utf-8")]
    known = [r for r in recs if r["pred"] != "UNKNOWN"]
    errs = [r for r in known if r["gold"] != r["pred"]]
    by_split = lambda rs, s: [r for r in rs if r["split"] == s]

    trans = sorted({(r["gold"], r["pred"]) for r in errs})
    rng = np.random.RandomState(42)
    rows, out = [], []
    for g, p in trans:
        cnt = {s: sum(1 for r in by_split(errs, s) if r["gold"] == g and r["pred"] == p)
               for s in ("train", "val", "test")}
        total = sum(cnt.values())
        ref = sum(1 for r in by_split(known, "train") if r["gold"] == g and r["pred"] == g)
        status = {}
        for tier, (tmin, emin) in TIERS.items():
            status[tier] = ("stable" if cnt["train"] >= tmin and min(cnt["val"], cnt["test"]) >= emin
                            else "weak" if cnt["train"] >= TIERS["lenient"][0] // 1 and tier != "lenient"
                            else "weak" if cnt["train"] >= 20 else "insufficient")
        # bootstrap stability at the DEFAULT tier (archived definition), 200 resamples per split
        flags = {s: np.array([r["gold"] == g and r["pred"] == p for r in by_split(errs, s)] +
                             [False] * 0) for s in ("train", "val", "test")}
        pools = {s: np.array([1 if (r["gold"] == g and r["pred"] == p) else 0
                              for r in by_split(recs, s)]) for s in ("train", "val", "test")}
        ok = 0
        for _ in range(200):
            t = pools["train"][rng.randint(0, len(pools["train"]), len(pools["train"]))].sum()
            v = pools["val"][rng.randint(0, len(pools["val"]), len(pools["val"]))].sum()
            te = pools["test"][rng.randint(0, len(pools["test"]), len(pools["test"]))].sum()
            if t >= TIERS["default"][0] and min(v, te) >= TIERS["default"][1]:
                ok += 1
        boot_default = ok / 200
        ok = 0
        rng2 = np.random.RandomState(43)
        for _ in range(200):
            t = pools["train"][rng2.randint(0, len(pools["train"]), len(pools["train"]))].sum()
            v = pools["val"][rng2.randint(0, len(pools["val"]), len(pools["val"]))].sum()
            te = pools["test"][rng2.randint(0, len(pools["test"]), len(pools["test"]))].sum()
            if t >= SCOPE["train_min"] and v >= SCOPE["val_min"] and te >= SCOPE["test_min"]:
                ok += 1
        boot_scope = ok / 200
        scope_pass = (cnt["train"] >= SCOPE["train_min"] and cnt["val"] >= SCOPE["val_min"]
                      and cnt["test"] >= SCOPE["test_min"] and ref >= SCOPE["ref_min"])
        reject = None
        if not scope_pass:
            reject = ("insufficient split support" if cnt["train"] < SCOPE["train_min"]
                      or cnt["val"] < SCOPE["val_min"] or cnt["test"] < SCOPE["test_min"]
                      else f"reference pool too small ({ref})")
        rec = {"channel_id": f"{g}__{p}", "gold": g, "pred": p,
               "semantic": f"should {SEM[g]}, but instead did: {SEM[p]}",
               "count_all": total, "count_train": cnt["train"], "count_val": cnt["val"],
               "count_test": cnt["test"], "ref_pool_train": ref,
               "status_default": status["default"], "status_strict": status["strict"],
               "status_lenient": status["lenient"],
               "bootstrap_default_thresholds": round(boot_default, 3),
               "bootstrap_scope_filter": round(boot_scope, 3),
               "intervention_scope_pass": bool(scope_pass),
               "rejection_reason": reject,
               "new_vs_w2c_metatool": ("NEW family (flag_param_error involved)"
                                       if "flag_param_error" in (g, p) else
                                       "analogous family exists (ask/refuse/call axes)")}
        out.append(rec)
        rows.append([rec["channel_id"], total, cnt["train"], cnt["val"], cnt["test"], ref,
                     status["default"], status["lenient"], round(boot_scope, 3),
                     scope_pass, reject or ""])

    out.sort(key=lambda r: -r["count_all"])
    eligible = [r for r in out if r["intervention_scope_pass"]
                and r["bootstrap_scope_filter"] >= 0.80]
    summary = {
        "dataset": "acebench_decision (generation readout, locked)",
        "n_rows": len(recs), "n_unknown_excluded": len(recs) - len(known),
        "n_errors": len(errs), "n_transitions": len(out),
        "transitions": out,
        "n_scope_pass": sum(r["intervention_scope_pass"] for r in out),
        "n_eligible_boot80": len(eligible),
        "eligible_channels": [r["channel_id"] for r in eligible],
        "ge2_stable_channels": len(eligible) >= 2,
        "new_family_present": any("flag_param_error" in (r["gold"], r["pred"])
                                  for r in eligible),
        "scope_filter": SCOPE, "tiers": TIERS,
    }
    json.dump(summary, open(CD / "acebench_channel_discovery.json", "w"), indent=2,
              ensure_ascii=False)
    with open(CD / "acebench_channel_discovery.csv", "w") as f:
        f.write("channel,all,train,val,test,ref_pool,status_default,status_lenient,"
                "bootstrap_scope,scope_pass,rejection\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_errors", "n_transitions", "n_scope_pass",
                                              "n_eligible_boot80", "eligible_channels",
                                              "ge2_stable_channels", "new_family_present")},
                     indent=1))


if __name__ == "__main__":
    main()
