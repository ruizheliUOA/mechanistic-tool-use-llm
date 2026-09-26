"""
phase5_mistral_multiseed.py — Phase-5 multi-seed robustness (Mistral × W2C).
============================================================================
Runs ONLY after the seed-42 pilot gate passes. For each seed in {42,123,456,789,2024}:
  - rebuild split (deterministic stratified 70/15/15 on Mistral's own etype);
  - REDISCOVER candidate channels from that split's TRAIN counts (scope filter);
  - per candidate: R1 obs (norm gate + max ratio), R2 method (cos<0.6 -> +PCA1),
    refit router (matched_pred), re-extract direction, reduced val sweep -> lock;
  - build auto-cascade over channels that individually pass a per-seed utility check
    (val_net>0 required; on test: Net>0 and real>reverse) -> LOCKED TEST once;
  - controls: real / reverse / ungated / random (x10 for {42,123}, x1 else).

Cached seed-independent activations are reused (only the split changes). Model loaded once.
Reduced grid: inj {0,-2,-4}, alpha {2,4,6}, thr {0.4..0.8}. NO Qwen channel set is frozen on.
"""
from __future__ import annotations
import gc, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L
from phase5_mistral_discovery import SHORT, SCOPE_TRAIN_MIN, SCOPE_EVAL_MIN

ID2GP = {v: k for k, v in SHORT.items()}
GP2ID = SHORT
SEEDS = [42, 123, 456, 789, 2024]
RANDOM_FULL = {42, 123}
INJ_OFFSETS = [0, -2, -4]
ALPHAS = [2.0, 4.0, 6.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
GOLD = set(L.GOLD_CLASSES)
OUT = L.OUT / "multiseed"; OUT.mkdir(parents=True, exist_ok=True)


def sid(g, p):
    return GP2ID.get((g, p), f"{g}__{p}")


def discover_split(meta, tr, va, te):
    """Scope-passing candidate channels from this split's counts."""
    def cnt(idxs):
        return Counter((meta[i]["gold"], meta[i]["pred"]) for i in idxs
                       if not meta[i]["correct"])
    ctr, cva, cte = cnt(tr), cnt(va), cnt(te)
    cands = []
    for (g, p), n in ctr.items():
        if g in GOLD and p != g and n >= SCOPE_TRAIN_MIN \
                and cva.get((g, p), 0) >= SCOPE_EVAL_MIN and cte.get((g, p), 0) >= SCOPE_EVAL_MIN:
            cands.append(sid(g, p))
    return cands


def geom_channel(cid, meta, tr, va, acts):
    g, p = ID2GP[cid]; et = f"{g}__{p}"
    per = {}
    for lyr in L.OBS_LAYERS:
        A = acts[lyr]
        err = [i for i in tr if meta[i]["etype"] == et]
        ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == g]
        if len(err) < 10 or len(ref) < 5:
            return None
        _, dm_norm = L.diffmean(A, err, ref)
        med = float(np.median(np.linalg.norm(A[tr], axis=1)))
        router = L.fit_router(A, tr, va, meta, et, p, "matched_pred")
        cos = L.cos_dm_pc1(A, tr, meta, et, g)
        per[lyr] = {"dm_norm": dm_norm, "ratio": dm_norm / (med + 1e-9),
                    "auc": (router["val_auc"] if router else None), "cos": cos}
    surv = [lyr for lyr in L.OBS_LAYERS if per[lyr]["ratio"] >= L.DEGEN_RATIO_FLOOR]
    if not surv:
        return None
    r1 = max(surv, key=lambda lyr: (per[lyr]["ratio"], per[lyr]["auc"] or 0))
    methods = ["diffmean"] + (["pca1"] if per[r1]["cos"] is not None and per[r1]["cos"] < L.COS_THR else [])
    inj = sorted({max(0, r1 + o) for o in INJ_OFFSETS})
    return {"etype": et, "gold": g, "from_pred": p, "R1_obs": r1, "methods": methods,
            "inj": inj, "router_val_auc": per[r1]["auc"], "cos_dm_pc1": per[r1]["cos"]}


def channels_map(cids):
    return {cid: {"gold": ID2GP[cid][0], "from_pred": ID2GP[cid][1],
                  "etype": f"{ID2GP[cid][0]}__{ID2GP[cid][1]}"} for cid in cids}


def run_seed(seed, model, tok, dev, acts, ds, meta_full, base_pred):
    tr, va, te = L.gen_split(meta_full, seed)
    meta = meta_full
    cands = discover_split(meta, tr, va, te)
    CH = channels_map(cands)
    locked, sweep_auc = {}, {}
    for cid in cands:
        gc_ = geom_channel(cid, meta, tr, va, acts)
        if gc_ is None:
            continue
        A = acts[gc_["R1_obs"]]
        router = L.fit_router(A, tr, va, meta, gc_["etype"], gc_["from_pred"], "matched_pred")
        vsc = L.rscore(router, A, va)
        firing = [i for i in va if base_pred[i] == gc_["from_pred"] and vsc[i] >= min(THRESHOLDS)]
        rows = []
        for method in gc_["methods"]:
            unit, nrm, mn, _, _ = L.direction(A, tr, meta, gc_["etype"], gc_["gold"], method)
            for inj in gc_["inj"]:
                for a in ALPHAS:
                    corr = (a * mn * unit).astype(np.float32)
                    preds = {i: L.predict(model, tok, dev, ds[i], corr, inj)[0] for i in firing}
                    gc.collect(); torch.cuda.empty_cache()
                    for thr in THRESHOLDS:
                        routed = [i for i in firing if vsc[i] >= thr]
                        ip = {i: base_pred[i] for i in va}
                        for i in routed:
                            ip[i] = preds[i]
                        m = L.metrics(va, meta, base_pred, ip, CH)
                        rows.append({"method": method, "obs": gc_["R1_obs"], "inj": inj,
                                     "alpha": a, "thr": thr, "net": m["net"], "broke": m["broke"],
                                     "ch_fixed": m["chan_fixed"][cid]})
        if not rows:
            continue
        b = max(rows, key=lambda r: (r["net"], -r["broke"], r["ch_fixed"]))
        if b["net"] <= 0:
            continue  # per-seed utility: must help on val
        locked[cid] = {**b, "router_val_auc": gc_["router_val_auc"], "cos_dm_pc1": gc_["cos_dm_pc1"]}
        sweep_auc[cid] = gc_["router_val_auc"]

    # cascade order: over-call first, then val_net
    order = sorted(locked, key=lambda c: (0 if ID2GP[c][1] == "tool_call" else 1, -locked[c]["net"]))
    # precompute routers/dirs on test
    routers, dirs, tsc = {}, {}, {}
    for cid in order:
        A = acts[locked[cid]["obs"]]
        r = L.fit_router(A, tr, va, meta, CH[cid]["etype"], CH[cid]["from_pred"], "matched_pred")
        routers[cid] = (r, A); tsc[cid] = L.rscore(r, A, te)
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, CH[cid]["etype"], CH[cid]["gold"], locked[cid]["method"])
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
            ip[i] = L.predict(model, tok, dev, ds[i], (c["alpha"] * mn * uvec).astype(np.float32), c["inj"])[0]
            routed[chosen] += 1; touched.append(i)
        return ip, routed, touched

    base_acc = sum(1 for i in te if base_pred[i] == meta[i]["gold"]) / len(te)
    ip, routed, touched = cascade()
    m = L.metrics(te, meta, base_pred, ip, CH)
    dmg = sum(1 for i in touched if meta[i]["correct"] and ip[i] != meta[i]["gold"])

    # controls (cascade-level)
    D = acts[L.OBS_LAYERS[0]].shape[1]
    def variant(kind, rs=None, gated=True):
        if kind == "reverse":
            ov = {cid: -dirs[cid][0] for cid in order}
        elif kind == "random":
            rng = np.random.RandomState(rs); ov = {}
            for cid in order:
                v = rng.randn(D).astype(np.float32); ov[cid] = v / (np.linalg.norm(v) + 1e-12)
        else:
            ov = None
        ipx, _, tch = cascade(override=ov, gated=gated)
        mx = L.metrics(te, meta, base_pred, ipx, CH)
        return {"fixed": mx["fixed"], "broke": mx["broke"], "net": mx["net"], "n_touched": len(tch)}

    ctrl = {"real": {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"], "n_touched": len(touched)}}
    if order:
        ctrl["reverse"] = variant("reverse")
        ctrl["ungated"] = variant("real", gated=False)
        nr = 10 if seed in RANDOM_FULL else 1
        rn = [variant("random", rs=1000 + k)["net"] for k in range(nr)]
        ctrl["random"] = {"n": nr, "nets": rn, "limited": nr < 10, "mean": float(np.mean(rn)),
                          "std": float(np.std(rn)), "max": int(np.max(rn)), "min": int(np.min(rn)),
                          "real_ge_all": bool(m["net"] > max(rn)),
                          "n_ge_real": int(sum(1 for x in rn if x >= m["net"]))}
    row = {"seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "discovered_candidates": cands, "selected_channels": order,
           "base_acc": round(base_acc, 5), "sakiko_acc": round(m["acc"], 5),
           "delta_acc": round(m["acc"] - base_acc, 5), "fixed": m["fixed"], "broke": m["broke"],
           "net": m["net"], "nt_before": m["nt_before"], "nt_after": m["nt_after"],
           "routed": dict(routed), "n_touched": len(touched), "damage_on_correct": dmg,
           "locked": locked}
    return row, ctrl


def main():
    ds, meta = L.load_meta()
    acts = {lyr: np.load(L.CACHE_DIR / f"acts_L{lyr}.npy") for lyr in L.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    model, tok, dev = L.load_model(assert_bf16=True)
    rows, controls = [], {}
    t0 = time.time()
    for seed in SEEDS:
        print(f"===== SEED {seed} =====")
        row, ctrl = run_seed(seed, model, tok, dev, acts, ds, meta, base_pred)
        rows.append(row); controls[str(seed)] = ctrl
        print(f"  seed {seed}: sel={row['selected_channels']} net={row['net']:+d} "
              f"acc {row['base_acc']:.4f}->{row['sakiko_acc']:.4f}")
        json.dump({"per_seed": rows}, open(OUT / "mistral7b_multiseed_results.json", "w"),
                  indent=2, default=float)
        json.dump(controls, open(OUT / "mistral7b_multiseed_controls.json", "w"), indent=2, default=float)
    _report(rows, controls)
    print(f"multiseed done in {(time.time()-t0)/60:.1f} min")


def _report(rows, controls):
    nets = [r["net"] for r in rows]; deltas = [r["delta_acc"] for r in rows]
    agg = {"n_seeds": len(rows), "net_mean": float(np.mean(nets)), "net_std": float(np.std(nets)),
           "net_min": int(np.min(nets)), "net_max": int(np.max(nets)),
           "n_positive": int(sum(1 for x in nets if x > 0)), "delta_acc_mean": float(np.mean(deltas))}
    json.dump({"aggregate": agg, "per_seed": rows},
              open(OUT / "mistral7b_multiseed_results.json", "w"), indent=2, default=float)
    with open(OUT / "mistral7b_multiseed_results.csv", "w") as f:
        f.write("seed,n_test,selected_channels,base_acc,sakiko_acc,delta_acc,fixed,broke,net,real_ge_all_random\n")
        for r in rows:
            rga = controls[str(r["seed"])].get("random", {}).get("real_ge_all", "")
            f.write(f"{r['seed']},{r['n_test']},\"{'|'.join(r['selected_channels'])}\",{r['base_acc']},"
                    f"{r['sakiko_acc']},{r['delta_acc']},{r['fixed']},{r['broke']},{r['net']},{rga}\n")
    with open(OUT / "MISTRAL7B_MULTISEED_SUMMARY.md", "w") as f:
        f.write("# Mistral-7B Native SAKIKO-CA — Multi-Seed Robustness (W2C)\n\n")
        f.write(f"**Aggregate Net = {agg['net_mean']:.1f} ± {agg['net_std']:.1f} "
                f"(min {agg['net_min']}, max {agg['net_max']}); positive {agg['n_positive']}/{agg['n_seeds']}; "
                f"mean Δacc {agg['delta_acc_mean']:+.4f}.**\n\n")
        f.write("| seed | selected channels | base→SAKIKO | Δ | Fixed | Broke | Net | real≥all rand |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            c = controls[str(r["seed"])]; rga = c.get("random", {}).get("real_ge_all", "—")
            lim = " *(lim)*" if c.get("random", {}).get("limited") else ""
            f.write(f"| {r['seed']} | {', '.join(r['selected_channels']) or '(none)'} | "
                    f"{r['base_acc']:.4f}→{r['sakiko_acc']:.4f} | {r['delta_acc']:+.4f} | {r['fixed']} | "
                    f"{r['broke']} | **{r['net']:+d}** | {rga}{lim} |\n")
        f.write("\n## Controls per seed\n\n| seed | real | reverse | ungated(broke) | random mean(max) |\n|---|---|---|---|---|\n")
        for r in rows:
            c = controls[str(r["seed"])]
            rv = c.get("reverse", {}).get("net", "—"); ug = c.get("ungated", {})
            rd = c.get("random", {})
            f.write(f"| {r['seed']} | {c['real']['net']:+d} | {rv} | "
                    f"{ug.get('net','—')}({ug.get('broke','—')}) | {rd.get('mean','—')}({rd.get('max','—')}) |\n")


if __name__ == "__main__":
    main()
