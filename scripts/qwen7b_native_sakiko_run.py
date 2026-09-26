"""
qwen7b_native_sakiko_run.py — run stage: val sweep -> lock -> locked test -> placebo
Imported by qwen7b_native_sakiko.py --stage run.
"""
from __future__ import annotations
import json, pickle, gc, time, logging
from collections import Counter
from pathlib import Path
import numpy as np
import torch

import qwen7b_native_sakiko as P  # shared constants + helpers

log = logging.getLogger("qwen7b_sakiko.run")

INJ_OFFSETS = [0, -2, -4]   # injection at obs, obs-2, obs-4


def _metrics(idxs, meta, base_pred, int_pred):
    fixed = broke = 0
    for i in idxs:
        g = meta[i]["gold"]; bp = base_pred[i]; ip = int_pred[i]
        if bp != g and ip == g: fixed += 1
        elif bp == g and ip != g: broke += 1
    def chan(ch):
        f = sum(1 for i in idxs if meta[i]["etype"] == ch and int_pred[i] == meta[i]["gold"])
        return f
    nt = {ch: sum(1 for i in idxs if meta[i]["etype"] == ch) for ch in P.CHANNELS}
    nt_after = {ch: sum(1 for i in idxs
                        if meta[i]["gold"] == P.CHANNELS[ch]["gold"]
                        and int_pred[i] == P.CHANNELS[ch]["from_pred"]) for ch in P.CHANNELS}
    acc = sum(1 for i in idxs if int_pred[i] == meta[i]["gold"]) / len(idxs)
    return {"fixed": fixed, "broke": broke, "net": fixed - broke, "acc": round(acc, 4),
            "chan_fixed": {ch: chan(ch) for ch in P.CHANNELS},
            "nt_before": nt, "nt_after": nt_after}


def stage_run():
    ds, meta = P.load_meta_and_dataset()
    tr, va, te = P.load_splits()
    acts = {L: np.load(P.CACHE_DIR / f"acts_L{L}.npy") for L in P.OBS_LAYERS}
    routers = pickle.load(open(P.CACHE_DIR / "routers.pkl", "rb"))
    directions = pickle.load(open(P.CACHE_DIR / "directions.pkl", "rb"))
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}

    model, tok, dev = P.load_model()

    def router_score(ch, idxs):
        r = routers[ch]; A = acts[r["obs_layer"]]
        return dict(zip(idxs, r["clf"].predict_proba(r["scaler"].transform(A[idxs]))[:, 1]))

    # ── STEP 5: per-channel val sweep ──
    log.info("=== Step 5: validation sweep ===")
    sweep = {}            # ch -> list of grid rows
    corrected_cache = {}  # (ch, inj, alpha) -> {idx: pred}  (val firing-eligible)
    val_base_acc = sum(1 for i in va if base_pred[i] == meta[i]["gold"]) / len(va)

    for ch, cfg in P.CHANNELS.items():
        if ch not in directions:
            log.warning("skip channel %s (no direction)", ch); continue
        unit = directions[ch]["unit"]; mednorm = directions[ch]["median_norm"]
        obsL = directions[ch]["obs_layer"]
        vscore = router_score(ch, va)
        # gate-eligible & firing (score >= min threshold) val samples
        elig = [i for i in va if base_pred[i] == cfg["from_pred"]]
        firing = [i for i in elig if vscore[i] >= min(P.THRESHOLDS)]
        log.info("  [%s] obs L%d | val gate-eligible=%d firing(>=%.1f)=%d",
                 ch, obsL, len(elig), min(P.THRESHOLDS), len(firing))
        inj_layers = sorted({max(0, obsL + off) for off in INJ_OFFSETS})
        rows = []
        for inj in inj_layers:
            for alpha in P.ALPHAS:
                corr = (alpha * mednorm * unit).astype(np.float32)
                preds = {}
                for i in firing:
                    preds[i] = P.predict_hooked(model, tok, dev, ds[i], corr, inj)
                corrected_cache[(ch, inj, alpha)] = preds
                for thr in P.THRESHOLDS:
                    int_pred = dict(base_pred)
                    touched = 0
                    for i in firing:
                        if vscore[i] >= thr:
                            int_pred[i] = preds[i]; touched += 1
                    m = _metrics(va, meta, base_pred, int_pred)
                    rows.append({"inj_layer": inj, "alpha": alpha, "threshold": thr,
                                 "fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
                                 "ch_fixed": m["chan_fixed"][ch], "touched": touched,
                                 "val_acc_after": m["acc"]})
                gc.collect(); torch.cuda.empty_cache()
        sweep[ch] = rows
        best = max(rows, key=lambda r: (r["net"], -r["broke"], r["ch_fixed"]))
        log.info("  [%s] best: L%d a=%.1f thr=%.2f net=%+d (fix=%d broke=%d ch_fix=%d touched=%d)",
                 ch, best["inj_layer"], best["alpha"], best["threshold"], best["net"],
                 best["fixed"], best["broke"], best["ch_fixed"], best["touched"])

    json.dump(sweep, open(P.OUT / "qwen25_7b_val_sweep.json", "w"), indent=2)

    # ── STEP 6: lock cascade config (best per channel) ──
    locked = {}
    for ch in sweep:
        b = max(sweep[ch], key=lambda r: (r["net"], -r["broke"], r["ch_fixed"]))
        locked[ch] = {"obs_layer": directions[ch]["obs_layer"], "inj_layer": b["inj_layer"],
                      "alpha": b["alpha"], "threshold": b["threshold"],
                      "val_net": b["net"], "val_fixed": b["fixed"], "val_broke": b["broke"]}
    # cascade eval on val (priority order, single correction)
    def cascade_pred(idxs, use_cache, corr_override=None, gated=True):
        vsc = {ch: router_score(ch, idxs) for ch in locked}
        int_pred = dict(base_pred); routed_counts = Counter(); touched_idx = []
        for i in idxs:
            chosen = None
            for ch in P.PRIORITY:
                if ch not in locked: continue
                gate_ok = (base_pred[i] == P.CHANNELS[ch]["from_pred"]) if gated else True
                if gate_ok and vsc[ch][i] >= locked[ch]["threshold"]:
                    chosen = ch; break
            if chosen is None: continue
            inj = locked[chosen]["inj_layer"]; alpha = locked[chosen]["alpha"]
            if corr_override is not None:
                unit = corr_override[chosen]
            else:
                unit = directions[chosen]["unit"]
            mednorm = directions[chosen]["median_norm"]
            key = (chosen, inj, alpha)
            if use_cache and corr_override is None and i in corrected_cache.get(key, {}):
                int_pred[i] = corrected_cache[key][i]
            else:
                corr = (alpha * mednorm * unit).astype(np.float32)
                int_pred[i] = P.predict_hooked(model, tok, dev, ds[i], corr, inj)
            routed_counts[chosen] += 1; touched_idx.append(i)
        return int_pred, routed_counts, touched_idx

    val_int, val_routed, val_touched = cascade_pred(va, use_cache=True)
    vm = _metrics(va, meta, base_pred, val_int)
    log.info("=== cascade on val: acc %.4f->%.4f net=%+d (fix=%d broke=%d) routed=%s ===",
             val_base_acc, vm["acc"], vm["net"], vm["fixed"], vm["broke"], dict(val_routed))
    json.dump({"locked": locked, "val_cascade": {**vm, "routed": dict(val_routed),
               "base_acc": round(val_base_acc, 4), "n_touched": len(val_touched)}},
              open(P.OUT / "qwen25_7b_locked_config.json", "w"), indent=2, default=float)

    # ── STEP 7: locked test once ──
    log.info("=== Step 7: locked test ===")
    test_base_acc = sum(1 for i in te if base_pred[i] == meta[i]["gold"]) / len(te)
    test_int, test_routed, test_touched = cascade_pred(te, use_cache=False)
    tm = _metrics(te, meta, base_pred, test_int)
    # damage on originally-correct
    dmg = sum(1 for i in test_touched if meta[i]["correct"] and test_int[i] != meta[i]["gold"])
    n_touch_correct = sum(1 for i in test_touched if meta[i]["correct"])
    three_before = sum(1 for i in te if meta[i]["etype"] in P.CHANNELS)
    three_after = sum(tm["nt_after"][ch] for ch in P.CHANNELS) if False else None
    # recompute three-channel errors remaining after
    three_after = sum(1 for i in te if (
        (meta[i]["gold"] == "request_for_info" and test_int[i] == "tool_call") or
        (meta[i]["gold"] == "cannot_answer" and test_int[i] == "tool_call") or
        (meta[i]["gold"] == "cannot_answer" and test_int[i] == "direct")))
    with open(P.OUT / "qwen25_7b_locked_test_details.jsonl", "w") as f:
        for i in te:
            f.write(json.dumps({"idx": i, "uuid": meta[i]["uuid"], "gold": meta[i]["gold"],
                "base_pred": base_pred[i], "int_pred": test_int[i],
                "etype": meta[i]["etype"], "touched": i in set(test_touched)}, ensure_ascii=False) + "\n")
    test_summary = {
        "base_acc": round(test_base_acc, 5), "sakiko_acc": round(tm["acc"], 5),
        "delta_acc": round(tm["acc"] - test_base_acc, 5),
        "fixed": tm["fixed"], "broke": tm["broke"], "net": tm["net"],
        "chan_fixed": tm["chan_fixed"], "nt_before": tm["nt_before"], "nt_after": tm["nt_after"],
        "three_channel_before": three_before, "three_channel_after": three_after,
        "routed": dict(test_routed), "n_touched": len(test_touched),
        "touched_correct": n_touch_correct, "damage_on_correct": dmg,
        "n_test": len(te), "locked": locked,
    }
    json.dump(test_summary, open(P.OUT / "qwen25_7b_locked_test_summary.json", "w"),
              indent=2, default=float)
    log.info("LOCKED TEST: acc %.4f->%.4f (d%+.4f) net=%+d (fix=%d broke=%d) 3ch %d->%d routed=%s",
             test_base_acc, tm["acc"], tm["acc"]-test_base_acc, tm["net"], tm["fixed"],
             tm["broke"], three_before, three_after, dict(test_routed))

    # ── STEP 8: placebo controls (on test, locked config) ──
    log.info("=== Step 8: placebo controls ===")
    D = acts[P.OBS_LAYERS[0]].shape[1]
    def run_variant(kind, rng_seed=None, gated=True):
        if kind == "real":
            ov = None
        elif kind == "reverse":
            ov = {ch: -directions[ch]["unit"] for ch in locked}
        elif kind == "random":
            rng = np.random.RandomState(rng_seed)
            ov = {}
            for ch in locked:
                v = rng.randn(D).astype(np.float32); ov[ch] = v/(np.linalg.norm(v)+1e-12)
        else:
            ov = None
        ip, routed, touched = cascade_pred(te, use_cache=False, corr_override=ov, gated=gated)
        m = _metrics(te, meta, base_pred, ip)
        return {"fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
                "chan_fixed": m["chan_fixed"], "n_touched": len(touched)}

    placebo = {"real": run_variant("real")}
    log.info("  real: net=%+d", placebo["real"]["net"])
    placebo["reverse"] = run_variant("reverse")
    log.info("  reverse: net=%+d", placebo["reverse"]["net"])
    rnd = [run_variant("random", rng_seed=1000+k) for k in range(10)]
    rnd_nets = [r["net"] for r in rnd]
    placebo["random_distribution"] = {
        "n": len(rnd), "nets": rnd_nets,
        "mean": round(float(np.mean(rnd_nets)), 2), "std": round(float(np.std(rnd_nets)), 2),
        "max": int(np.max(rnd_nets)), "min": int(np.min(rnd_nets)),
        "real_net": placebo["real"]["net"],
        "real_ge_all_random": bool(placebo["real"]["net"] > max(rnd_nets)),
        "n_random_ge_real": int(sum(1 for x in rnd_nets if x >= placebo["real"]["net"])),
    }
    log.info("  random dist: mean=%.1f max=%d | real=%d ge_all=%s",
             placebo["random_distribution"]["mean"], placebo["random_distribution"]["max"],
             placebo["real"]["net"], placebo["random_distribution"]["real_ge_all_random"])
    placebo["ungated"] = run_variant("real", gated=False)
    log.info("  ungated: net=%+d broke=%d (vs gated broke=%d)",
             placebo["ungated"]["net"], placebo["ungated"]["broke"], placebo["real"]["broke"])
    json.dump(placebo, open(P.OUT / "qwen25_7b_placebo_controls.json", "w"), indent=2, default=float)

    _write_reports(test_summary, placebo, locked, vm, val_base_acc, val_routed)
    log.info("=== run stage complete ===")


def _write_reports(ts, placebo, locked, vm, val_base_acc, val_routed):
    # val sweep summary
    with open(P.OUT / "qwen25_7b_val_sweep_summary.md", "w") as f:
        f.write("# Qwen2.5-7B SAKIKO — Validation Sweep & Locked Config\n\n")
        f.write("## Locked per-channel config (selected on val only)\n\n")
        f.write("| channel | obs | inj | alpha | thr | val_net | val_fixed | val_broke |\n|---|---|---|---|---|---|---|---|\n")
        for ch, c in locked.items():
            f.write(f"| {ch} | L{c['obs_layer']} | L{c['inj_layer']} | {c['alpha']} | {c['threshold']} "
                    f"| {c['val_net']:+d} | {c['val_fixed']} | {c['val_broke']} |\n")
        f.write(f"\n## Cascade on val (priority {P.PRIORITY})\n\n")
        f.write(f"- acc {val_base_acc:.4f} → {vm['acc']:.4f}  | net {vm['net']:+d} "
                f"(fixed {vm['fixed']}, broke {vm['broke']}) | routed {dict(val_routed)}\n")

    # locked test summary
    with open(P.OUT / "qwen25_7b_locked_test_summary.md", "w") as f:
        f.write("# Qwen2.5-7B Native SAKIKO — Locked Test (seed=42)\n\n")
        f.write("| metric | value |\n|---|---|\n")
        f.write(f"| baseline test acc | {ts['base_acc']:.4f} |\n")
        f.write(f"| SAKIKO test acc | {ts['sakiko_acc']:.4f} |\n")
        f.write(f"| Δ acc | {ts['delta_acc']:+.4f} |\n")
        f.write(f"| Fixed | {ts['fixed']} |\n| Broke | {ts['broke']} |\n| **Net** | **{ts['net']:+d}** |\n")
        f.write(f"| 3-channel errors before→after | {ts['three_channel_before']}→{ts['three_channel_after']} |\n")
        f.write(f"| samples touched | {ts['n_touched']} |\n")
        f.write(f"| damage on originally-correct | {ts['damage_on_correct']} / {ts['touched_correct']} touched-correct |\n")
        f.write(f"| routed | {ts['routed']} |\n\n")
        f.write("## Per-channel (gold→from_pred error count before vs after)\n\n")
        f.write("| channel | before | after | fixed |\n|---|---|---|---|\n")
        for ch in P.CHANNELS:
            f.write(f"| {ch} | {ts['nt_before'][ch]} | {ts['nt_after'][ch]} | {ts['chan_fixed'][ch]} |\n")

    # placebo summary
    with open(P.OUT / "qwen25_7b_placebo_summary.md", "w") as f:
        f.write("# Qwen2.5-7B Native SAKIKO — Placebo / Causal Controls (test, locked)\n\n")
        f.write("| variant | Fixed | Broke | Net |\n|---|---|---|---|\n")
        for k in ["real", "reverse", "ungated"]:
            p = placebo[k]; f.write(f"| {k} | {p['fixed']} | {p['broke']} | **{p['net']:+d}** |\n")
        rd = placebo["random_distribution"]
        f.write(f"\n**Random distribution ({rd['n']} dirs):** net mean {rd['mean']} ± {rd['std']} "
                f"(max {rd['max']}, min {rd['min']}); real net {rd['real_net']}; "
                f"real ≥ all random: {rd['real_ge_all_random']}; #random ≥ real: {rd['n_random_ge_real']}.\n")


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stage_run()
