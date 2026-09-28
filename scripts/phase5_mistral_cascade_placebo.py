"""
phase5_mistral_cascade_placebo.py — cascade-level placebo for Arm F (matched to the archived
Qwen cascade placebo, so the cross-model specificity comparison is apples-to-apples).
=============================================================================================
The Qwen archive reports a CASCADE-level placebo (real / reverse / random x10 / ungated) at the
locked config. Phase-5's pilot ran PER-CHANNEL controls. This script adds the matched
cascade-level control for Mistral's Arm F (manual {ca_tc, rfi_tc, ca_direct} cascade at the
val-locked configs), changing NO selection and NO config — it only characterizes an
already-locked, already-run arm.

Usage: python scripts/phase5_mistral_cascade_placebo.py
"""
from __future__ import annotations
import gc, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L
from phase5_mistral_discovery import SHORT

ID2GP = {v: k for k, v in SHORT.items()}
PIL = L.OUT / "pilot"
PLA = L.OUT / "placebos"
N_RANDOM_CASCADE = 10          # matches the archived Qwen cascade placebo (n=10)


def main():
    ds, meta = L.load_meta()
    tr, va, te = L.load_splits()
    acts = {lyr: np.load(L.CACHE_DIR / f"acts_L{lyr}.npy") for lyr in L.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    geom = json.loads((L.OUT / "geometry" / "mistral7b_geometry.json").read_text())["geometry"]
    locked = json.loads((PIL / "locked_configs.json").read_text())
    members = json.loads((PIL / "pilot_summary.json").read_text())["manual_diag_members"]
    CH = {cid: {"gold": ID2GP[cid][0], "from_pred": ID2GP[cid][1],
                "etype": f"{ID2GP[cid][0]}__{ID2GP[cid][1]}"} for cid in geom}
    # cascade order identical to the pilot's Arm F (over-call first, then val_net desc)
    order = sorted(members, key=lambda c: (0 if ID2GP[c][1] == "tool_call" else 1,
                                           -locked[c]["val_net"]))
    print("cascade order:", order)

    model, tok, dev = L.load_model(assert_bf16=True)
    routers, dirs, tsc = {}, {}, {}
    for cid in order:
        A = acts[locked[cid]["obs"]]
        r = L.fit_router(A, tr, va, meta, CH[cid]["etype"], CH[cid]["from_pred"], "matched_pred")
        routers[cid] = (r, A)
        tsc[cid] = L.rscore(r, A, te)
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, CH[cid]["etype"], CH[cid]["gold"],
                                          locked[cid]["method"])
        dirs[cid] = (unit, mn)

    def cascade(override=None, gated=True):
        ip = dict(base_pred); routed = Counter(); touched = []
        for i in te:
            chosen = None
            for cid in order:
                gate = (base_pred[i] == CH[cid]["from_pred"]) if gated else True
                if gate and tsc[cid][i] >= locked[cid]["thr"]:
                    chosen = cid; break
            if chosen is None:
                continue
            c = locked[chosen]; unit, mn = dirs[chosen]
            uvec = override[chosen] if override else unit
            ip[i] = L.predict(model, tok, dev, ds[i],
                              (c["alpha"] * mn * uvec).astype(np.float32), c["inj"])[0]
            routed[chosen] += 1; touched.append(i)
        m = L.metrics(te, meta, base_pred, ip, CH)
        dmg = sum(1 for i in touched if meta[i]["correct"] and ip[i] != meta[i]["gold"])
        gc.collect(); torch.cuda.empty_cache()
        return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
                "acc": m["acc"], "n_touched": len(touched), "damage_on_correct": dmg,
                "nt_after": m["nt_after"], "routed": dict(routed)}

    out = {}
    D = acts[L.OBS_LAYERS[0]].shape[1]
    t0 = time.time()
    out["real"] = cascade()
    print("real:", out["real"]["net"])
    out["reverse"] = cascade(override={c: -dirs[c][0] for c in order})
    print("reverse:", out["reverse"]["net"])
    rnd = []
    for k in range(N_RANDOM_CASCADE):
        rng = np.random.RandomState(1000 + k)
        ov = {}
        for c in order:
            v = rng.randn(D).astype(np.float32); ov[c] = v / (np.linalg.norm(v) + 1e-12)
        r = cascade(override=ov)
        rnd.append(r["net"])
        print(f"  random {k+1}/{N_RANDOM_CASCADE}: {r['net']:+d}")
    real = out["real"]["net"]
    out["random"] = {"n": len(rnd), "nets": rnd, "mean": float(np.mean(rnd)),
                     "std": float(np.std(rnd)), "max": int(np.max(rnd)), "min": int(np.min(rnd)),
                     "real_net": real, "real_ge_all_random": bool(real > max(rnd)),
                     "n_random_ge_real": int(sum(1 for x in rnd if x >= real)),
                     "percentile_of_real": float(100.0 * sum(1 for x in rnd if x < real) / len(rnd)),
                     "marginal_over_random_mean": float(real - np.mean(rnd))}
    out["ungated"] = cascade(gated=False)
    print("ungated:", out["ungated"]["net"], "broke", out["ungated"]["broke"])
    out["_meta"] = {"arm": "F_manual_diag", "members": order, "n_test": len(te),
                    "note": "cascade-level placebo matched to archived Qwen protocol (n_random=10)",
                    "elapsed_min": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(PLA / "cascade_placebo_armF.json", "w"), indent=2, default=float)

    with open(PLA / "CASCADE_PLACEBO_ARMF.md", "w") as f:
        f.write("# Mistral Arm F (manual cascade) — cascade-level placebo\n\n")
        f.write("Matched to the archived Qwen cascade placebo (same protocol, n_random=10).\n\n")
        f.write("| variant | Fixed | Broke | Net |\n|---|---|---|---|\n")
        for k in ["real", "reverse", "ungated"]:
            v = out[k]
            f.write(f"| {k} | {v['fixed']} | {v['broke']} | **{v['net']:+d}** |\n")
        r = out["random"]
        f.write(f"\n**Random ({r['n']} dirs):** mean {r['mean']:.1f} ± {r['std']:.1f} "
                f"(max {r['max']}, min {r['min']}); real {r['real_net']}; "
                f"real ≥ all random: **{r['real_ge_all_random']}**; #random ≥ real: "
                f"**{r['n_random_ge_real']}**; percentile {r['percentile_of_real']:.0f}; "
                f"**marginal over random mean: {r['marginal_over_random_mean']:+.1f}**.\n")
    print("wrote cascade placebo;", out["random"])


if __name__ == "__main__":
    main()
