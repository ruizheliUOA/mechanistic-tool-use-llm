"""
phase3_multiseed.py — Step 9: multi-seed R vs E for the pilot-passed new channel(s).
=====================================================================================
Runs only if the seed-42 extended system cleared the multiseed gate (checked by the caller /
final report; this script asserts pilot gate + extended-cascade improvement before running).

Per seed in {42,123,456,789,2024} (gen_split from the archived multiseed builder):
  R = 3-channel SAKIKO-CA arm-B recipe (R1 obs-gate + R2 method + fixed NATIVE_CFG inj-offset/alpha,
      threshold on val) over candidate obs {12,16,20} — reproduces archived gpu_pilot armB
      (per-seed Net ~104/110/125/124/112, the drift cross-check).
  E = R + each pilot-passed new channel, selected per-seed by the SAME rules over obs {12,16,20,24}
      with a reduced val grid (inj {obs,obs-2,obs-4} x alpha {2,4,6}∪pilot x thr), appended after
      ca_direct (val-selected mutual order if both passed).
Locked test once per seed per system. Reduced grids; nothing tuned on test.

Outputs: multiseed/{PHASE3_MULTISEED_SUMMARY.md, multiseed_summary.json, multiseed_table.csv,
multiseed_rows.jsonl}. Resumable (one flushed row per seed).
"""
from __future__ import annotations
import gc, json, logging, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L
import qwen7b_native_sakiko as P
import qwen7b_native_sakiko_multiseed as MS
import qwen7b_method_diagnostics as D

log = logging.getLogger("p3.ms")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

MSOUT = L.OUT / "multiseed"; MSOUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 2024]
OLD_OBS = [12, 16, 20]
NEW_OBS = [12, 16, 20, 24]
NATIVE_CFG = {"rfi_tc": (-2, 2.0), "ca_tc": (-4, 4.0), "ca_direct": (-4, 6.0)}
ALPHAS_MS = [2.0, 4.0, 6.0]
THRESHOLDS = L.THRESHOLDS
ARCHIVED_ARMB = {42: 104, 123: 110, 456: 125, 789: 124, 2024: 112}


def _dir_pools(meta, tr, etype, ref_gold):
    err = [i for i in tr if meta[i]["etype"] == etype]
    ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == ref_gold]
    return err, ref


def select_channel(ch, meta, tr, va, acts, cand_obs, native, ds, model, tok, dev,
                   base_pred, extra_alpha=None):
    """R1+R2 selection + val lock for one channel. `native`=(inj_offset,alpha) or None (grid)."""
    if ch in L.NEW_CHANNELS:
        d = L.NEW_CHANNELS[ch]; et, fp, rg, negm = d["etype"], d["from_pred"], d["ref_gold"], "matched_pred"
    else:
        pc = P.CHANNELS[ch]; et, fp, rg, negm = ch, pc["from_pred"], pc["ref_gold"], "all_correct"
    # R1
    info = {}
    for Lx in cand_obs:
        _, nrm, mn, _, _ = L.direction(acts[Lx], tr, meta, et, rg, "diffmean")
        r = L.fit_router(acts[Lx], tr, va, meta, et, fp, negm)
        info[Lx] = {"norm": nrm, "ratio": nrm / mn, "auc": r["val_auc"] or 0.0, "router": r}
    survivors = [Lx for Lx in cand_obs if info[Lx]["norm"] >= L.DEGEN_NORM] or cand_obs
    r1 = max(survivors, key=lambda Lx: (info[Lx]["ratio"], info[Lx]["auc"]))
    # R2
    err, ref = _dir_pools(meta, tr, et, rg)
    u_dm, _ = D.diffmean(acts[r1], err, ref)
    u_p1, _ = D.pca_dir(acts[r1], err, ref, k=1)
    cos = abs(float(u_dm @ u_p1))
    methods = ["diffmean", "pca1"] if cos < L.COS_THR else ["diffmean"]
    router = info[r1]["router"]; A = acts[r1]
    vsc = L.rscore(router, A, va)
    firing = [i for i in va if base_pred[i] == fp and vsc[i] >= min(THRESHOLDS)]
    # candidate (inj,alpha) set
    if native is not None:
        off, alpha = native
        inj_alpha = [(max(0, r1 + off), alpha)]
    else:
        alphas = sorted(set(ALPHAS_MS + ([extra_alpha] if extra_alpha else [])))
        inj_alpha = [(max(0, r1 + o), a) for o in L.INJ_OFFSETS for a in alphas]
    best = None
    for method in methods:
        unit, nrm, mn, _, _ = L.direction(A, tr, meta, et, rg, method)
        for inj, alpha in inj_alpha:
            corr = (alpha * mn * unit).astype(np.float32)
            preds = {i: P.predict_hooked(model, tok, dev, ds[i], corr, inj) for i in firing}
            for thr in THRESHOLDS:
                routed = [i for i in firing if vsc[i] >= thr]
                ip = dict(base_pred)
                for i in routed:
                    ip[i] = preds[i]
                m = L.metrics5(va, meta, base_pred, ip)
                key = (m["net"], -m["broke"], m["chan_fixed"][ch])
                if best is None or key > best["key"]:
                    best = {"key": key, "method": method, "obs": r1, "inj": inj,
                            "alpha": alpha, "thr": thr, "val_net": m["net"],
                            "dir_norm": round(nrm, 3), "router": router, "unit": unit,
                            "median_norm": mn, "val_auc": info[r1]["auc"], "cos_dm_pca1": cos}
            gc.collect(); torch.cuda.empty_cache()
    return best, {"from_pred": fp, "etype": et}


def cascade(idxs, chans, priority, acts, ds, model, tok, dev, base_pred):
    vs = {ch: L.rscore(chans[ch]["router"], acts[chans[ch]["obs"]], idxs) for ch in chans}
    ip = dict(base_pred); routed = Counter(); touched = []
    for i in idxs:
        chosen = None
        for ch in priority:
            if base_pred[i] == chans[ch]["from_pred"] and vs[ch][i] >= chans[ch]["thr"]:
                chosen = ch; break
        if chosen is None:
            continue
        c = chans[chosen]
        corr = (c["alpha"] * c["median_norm"] * c["unit"]).astype(np.float32)
        ip[i] = P.predict_hooked(model, tok, dev, ds[i], corr, c["inj"])
        routed[chosen] += 1; touched.append(i)
    return ip, dict(routed), touched


def main():
    pilot = json.loads((L.OUT / "pilot" / "pilot_summary.json").read_text())
    passed = [ch for ch in ["ca_rfi", "tc_rfi"] if pilot["gate"].get(ch, {}).get("pass")]
    if not passed:
        log.error("no channel passed pilot gate — multiseed must not run"); sys.exit(1)
    ext = json.loads((L.OUT / "extended_cascade" / "extended_cascade_summary.json").read_text())
    if not ext.get("built") or ext["marginal_E_minus_R"]["net"] < 3:
        log.error("seed-42 extended system did not clear the gate — multiseed must not run"); sys.exit(1)
    pilot_alpha = {ch: pilot["locked"][ch]["alpha"] for ch in passed}
    # val-selected new-channel order from the extended cascade
    new_order = ext.get("new_channel_order", passed)

    ds, meta = L.load_meta_and_dataset()
    acts = {Lx: np.load(L.CACHE_DIR / f"acts_L{Lx}.npy") for Lx in L.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    model, tok, dev = P.load_model()

    rowf = MSOUT / "multiseed_rows.jsonl"
    done = {json.loads(l)["seed"] for l in open(rowf)} if rowf.exists() else set()
    fout = open(rowf, "a")
    for seed in SEEDS:
        if seed in done:
            log.info("skip seed %d (done)", seed); continue
        tr, va, te = MS.gen_split(meta, seed)
        base_acc = sum(1 for i in te if base_pred[i] == meta[i]["gold"]) / len(te)
        # R: 3 old channels
        chansR = {}
        for ch in L.REF_PRIORITY:
            best, extra = select_channel(ch, meta, tr, va, acts, OLD_OBS, NATIVE_CFG[ch],
                                         ds, model, tok, dev, base_pred)
            chansR[ch] = {**best, **extra}
        ipR, routedR, touchR = cascade(te, chansR, L.REF_PRIORITY, acts, ds, model, tok, dev, base_pred)
        mR = L.metrics5(te, meta, base_pred, ipR)
        log.info("seed %d R: net=%+d (archived armB ~%+d) fixed=%d broke=%d",
                 seed, mR["net"], ARCHIVED_ARMB.get(seed, 0), mR["fixed"], mR["broke"])
        # E: R + new channels
        chansE = dict(chansR); new_sel = {}
        for ch in passed:
            best, extra = select_channel(ch, meta, tr, va, acts, NEW_OBS, None, ds, model, tok, dev,
                                         base_pred, extra_alpha=pilot_alpha[ch])
            chansE[ch] = {**best, **extra}
            new_sel[ch] = {"obs": best["obs"], "inj": best["inj"], "alpha": best["alpha"],
                           "thr": best["thr"], "method": best["method"],
                           "dir_norm": best["dir_norm"], "val_auc": round(best["val_auc"], 4),
                           "cos_dm_pca1": round(best["cos_dm_pca1"], 4), "val_net": best["val_net"]}
        priE = L.REF_PRIORITY + [ch for ch in new_order if ch in passed]
        ipE, routedE, touchE = cascade(te, chansE, priE, acts, ds, model, tok, dev, base_pred)
        mE = L.metrics5(te, meta, base_pred, ipE)
        dmgE = sum(1 for i in touchE if meta[i]["correct"] and ipE[i] != meta[i]["gold"])
        log.info("seed %d E: net=%+d fixed=%d broke=%d delta_vs_R=%+d",
                 seed, mE["net"], mE["fixed"], mE["broke"], mE["net"] - mR["net"])
        row = {"seed": seed, "base_acc": round(base_acc, 5),
               "R": {"net": mR["net"], "fixed": mR["fixed"], "broke": mR["broke"],
                     "acc": mR["acc"], "routed": routedR, "nt_after": mR["nt_after"]},
               "E": {"net": mE["net"], "fixed": mE["fixed"], "broke": mE["broke"], "acc": mE["acc"],
                     "routed": routedE, "nt_after": mE["nt_after"], "damage_on_correct": dmgE},
               "delta_net": mE["net"] - mR["net"], "archived_armB_net": ARCHIVED_ARMB.get(seed),
               "new_channel_selection": new_sel}
        fout.write(json.dumps(row, default=float) + "\n"); fout.flush()
        gc.collect(); torch.cuda.empty_cache()
    fout.close()
    aggregate()


def aggregate():
    rows = [json.loads(l) for l in open(MSOUT / "multiseed_rows.jsonl")]
    rows.sort(key=lambda r: SEEDS.index(r["seed"]))
    rN = [r["R"]["net"] for r in rows]; eN = [r["E"]["net"] for r in rows]
    dN = [r["delta_net"] for r in rows]
    wins = sum(1 for d in dN if d > 2); losses = sum(1 for d in dN if d < -2)
    ties = len(dN) - wins - losses
    agg = {"seeds": [r["seed"] for r in rows],
           "R_net_mean": round(float(np.mean(rN)), 2), "R_net_std": round(float(np.std(rN)), 2),
           "E_net_mean": round(float(np.mean(eN)), 2), "E_net_std": round(float(np.std(eN)), 2),
           "delta_mean": round(float(np.mean(dN)), 2), "delta_min": min(dN), "delta_max": max(dN),
           "wins": wins, "ties": ties, "losses": losses,
           "per_seed": [{"seed": r["seed"], "R_net": r["R"]["net"], "E_net": r["E"]["net"],
                         "delta": r["delta_net"], "archived_armB": r["archived_armB_net"],
                         "new_sel": r["new_channel_selection"]} for r in rows]}
    json.dump(agg, open(MSOUT / "multiseed_summary.json", "w"), indent=2, default=float)
    with open(MSOUT / "multiseed_table.csv", "w") as f:
        f.write("seed,R_net,E_net,delta,archived_armB,E_fixed,E_broke,E_dmg\n")
        for r in rows:
            f.write(f"{r['seed']},{r['R']['net']},{r['E']['net']},{r['delta_net']},"
                    f"{r['archived_armB_net']},{r['E']['fixed']},{r['E']['broke']},"
                    f"{r['E']['damage_on_correct']}\n")
    with open(MSOUT / "PHASE3_MULTISEED_SUMMARY.md", "w") as f:
        f.write("# Phase 3 — Multi-seed: Reference R vs Extended E\n\n")
        f.write(f"R Net {agg['R_net_mean']}±{agg['R_net_std']} | "
                f"E Net {agg['E_net_mean']}±{agg['E_net_std']} | "
                f"ΔNet {agg['delta_mean']} (min {agg['delta_min']}, max {agg['delta_max']}) | "
                f"wins/ties/losses {agg['wins']}/{agg['ties']}/{agg['losses']}\n\n")
        f.write("| seed | R Net | E Net | Δ | archived armB |\n|---|---|---|---|---|\n")
        for p in agg["per_seed"]:
            f.write(f"| {p['seed']} | {p['R_net']:+d} | {p['E_net']:+d} | {p['delta']:+d} "
                    f"| {p['archived_armB']} |\n")
    log.info("multiseed aggregate: E %s±%s vs R %s±%s (Δ %s, W/T/L %d/%d/%d)",
             agg["E_net_mean"], agg["E_net_std"], agg["R_net_mean"], agg["R_net_std"],
             agg["delta_mean"], agg["wins"], agg["ties"], agg["losses"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--aggregate-only":
        aggregate()
    else:
        main()
