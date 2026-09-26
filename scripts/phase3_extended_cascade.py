"""
phase3_extended_cascade.py — Step 8: frozen 3-channel reference R vs extended E (R + passed channels).
======================================================================================================
Reference R = archived taskA updated cascade (rfi_tc L20/L18/a2/0.70, ca_tc L20/L16/a4/0.80,
ca_direct L20/L16/a6/0.40; DiffMean; priority rfi_tc->ca_tc->ca_direct). Old-channel routers use the
archived all-correct negatives; directions refit on this session's caches from train. R must reproduce
archived Net +117 within +/-3 before E is interpreted (R and E run in ONE process to cancel bf16 drift).

E appends each pilot-passed new channel after ca_direct. If BOTH pass, the A/B mutual order is chosen
on VALIDATION, locked, then test once; the swapped order is a pre-registered order-sensitivity ablation.
The three R channels are never retuned.

Outputs: extended_cascade/{EXTENDED_CASCADE_SUMMARY.md, extended_cascade_summary.json,
extended_cascade_key_table.csv}. Run once (resumable via _state.json).
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

log = logging.getLogger("p3.cascade")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

EXT = L.OUT / "extended_cascade"; EXT.mkdir(parents=True, exist_ok=True)
STATE = EXT / "_state.json"
ARCHIVED_REF_NET = 117


def build_channel(ch, cfg, acts, tr, va, meta):
    """Router + direction object for one channel. Old channels: all-correct negatives (archived);
    new channels: matched-pred negatives (protocol). ref_gold/etype resolved per channel."""
    A = acts[cfg["obs"]]
    if ch in L.NEW_CHANNELS:
        d = L.NEW_CHANNELS[ch]
        et, fp, rg = d["etype"], d["from_pred"], d["ref_gold"]
        neg_mode = "matched_pred"
    else:
        pc = P.CHANNELS[ch]
        et, fp, rg = ch, pc["from_pred"], pc["ref_gold"]
        neg_mode = "all_correct"
    router = L.fit_router(A, tr, va, meta, et, fp, neg_mode)
    unit, nrm, mn, ne, nr = L.direction(A, tr, meta, et, rg, cfg.get("method", "diffmean"))
    return {"obs": cfg["obs"], "inj": cfg["inj"], "alpha": cfg["alpha"], "thr": cfg["thr"],
            "method": cfg.get("method", "diffmean"), "from_pred": fp, "etype": et,
            "router": router, "unit": unit.astype(np.float32), "median_norm": mn,
            "dir_norm": round(nrm, 3), "val_auc": router["val_auc"]}


def cascade(idxs, chans, priority, acts, ds, model, tok, dev, base_pred, cache=None):
    """Priority cascade, single correction per sample. `cache` memoizes (ch,i)->pred to avoid
    recomputing identical (channel,sample,config) forward passes across R/E/order variants."""
    vs = {ch: L.rscore(c["router"], acts[c["obs"]], idxs) for ch, c in chans.items()}
    ip = dict(base_pred); routed = Counter(); touched = []; multi_eligible = 0
    if cache is None:
        cache = {}
    for i in idxs:
        eligible = [ch for ch in priority
                    if base_pred[i] == chans[ch]["from_pred"] and vs[ch][i] >= chans[ch]["thr"]]
        if len(eligible) > 1:
            multi_eligible += 1
        if not eligible:
            continue
        chosen = eligible[0]
        c = chans[chosen]
        key = (chosen, i, c["obs"], c["inj"], round(c["alpha"], 3), c["method"])
        if key not in cache:
            corr = (c["alpha"] * c["median_norm"] * c["unit"]).astype(np.float32)
            cache[key] = P.predict_hooked(model, tok, dev, ds[i], corr, c["inj"])
        ip[i] = cache[key]
        routed[chosen] += 1; touched.append(i)
    return ip, dict(routed), touched, multi_eligible, cache


def full_report(idxs, ip, base_pred, meta, touched):
    m = L.metrics5(idxs, meta, base_pred, ip)
    dmg = sum(1 for i in touched if meta[i]["correct"] and ip[i] != meta[i]["gold"])
    three_after = sum(1 for i in idxs if (
        (meta[i]["gold"] == "request_for_info" and ip[i] == "tool_call") or
        (meta[i]["gold"] == "cannot_answer" and ip[i] == "tool_call") or
        (meta[i]["gold"] == "cannot_answer" and ip[i] == "direct")))
    return {**{k: m[k] for k in ("acc", "macro_f1_3cls", "fixed", "broke", "net",
                                 "nt_before", "nt_after", "chan_fixed")},
            "damage_on_correct": dmg, "n_touched": len(touched),
            "three_channel_after": three_after,
            "transition_matrix": L.transition_matrix(idxs, meta, ip)}


def main():
    ds, meta = L.load_meta_and_dataset()
    tr, va, te = L.load_splits()
    acts = {Lx: np.load(L.CACHE_DIR / f"acts_L{Lx}.npy") for Lx in L.OBS_LAYERS}
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}

    pilot = json.loads((L.OUT / "pilot" / "pilot_summary.json").read_text())
    gate = pilot["gate"]; locked = pilot["locked"]
    passed = [ch for ch in ["ca_rfi", "tc_rfi"] if gate.get(ch, {}).get("pass")]
    log.info("channels passing pilot gate: %s", passed)
    if not passed:
        log.warning("no channel passed the gate — extended cascade NOT built")
        (EXT / "EXTENDED_CASCADE_SUMMARY.md").write_text(
            "# Extended Cascade — NOT BUILT\n\nNeither new channel passed the pilot gate; "
            "per PHASE3_PROTOCOL.md the cascade extension is not performed.\n")
        json.dump({"passed": [], "built": False}, open(EXT / "extended_cascade_summary.json", "w"))
        return

    model, tok, dev = P.load_model()
    cache = {}

    # reference R
    chansR = {ch: build_channel(ch, L.REF_LOCKED[ch], acts, tr, va, meta) for ch in L.REF_PRIORITY}
    ipR, routedR, touchR, multiR, cache = cascade(te, chansR, L.REF_PRIORITY, acts, ds, model,
                                                  tok, dev, base_pred, cache)
    R = full_report(te, ipR, base_pred, meta, touchR)
    R["routed"] = routedR
    repro_ok = abs(R["net"] - ARCHIVED_REF_NET) <= 3
    log.info("REFERENCE R: net=%+d (archived +%d, repro_ok=%s) fixed=%d broke=%d routed=%s",
             R["net"], ARCHIVED_REF_NET, repro_ok, R["fixed"], R["broke"], routedR)

    # extended E — order chosen on val if both passed
    new_priority = passed[:]
    order_note = "single new channel"
    if len(passed) == 2:
        best_net, best_order = None, passed
        for order in ([passed[0], passed[1]], [passed[1], passed[0]]):
            chansE = {**chansR, **{ch: build_channel(ch, locked[ch], acts, tr, va, meta)
                                   for ch in passed}}
            pri = L.REF_PRIORITY + order
            ipv, _, tv, _, cache = cascade(va, chansE, pri, acts, ds, model, tok, dev,
                                           base_pred, cache)
            netv = L.metrics5(va, meta, base_pred, ipv)["net"]
            if best_net is None or netv > best_net:
                best_net, best_order = netv, order
        new_priority = best_order
        order_note = f"val-selected order {best_order} (val net {best_net})"
    log.info("extended new-channel order: %s (%s)", new_priority, order_note)

    chansE = {**chansR, **{ch: build_channel(ch, locked[ch], acts, tr, va, meta) for ch in passed}}
    priE = L.REF_PRIORITY + new_priority
    ipE, routedE, touchE, multiE, cache = cascade(te, chansE, priE, acts, ds, model, tok, dev,
                                                  base_pred, cache)
    E = full_report(te, ipE, base_pred, meta, touchE)
    E["routed"] = routedE
    log.info("EXTENDED E: net=%+d fixed=%d broke=%d routed=%s", E["net"], E["fixed"],
             E["broke"], routedE)

    # order-sensitivity ablation (swapped), if both passed
    E_swapped = None
    if len(passed) == 2:
        pri_sw = L.REF_PRIORITY + list(reversed(new_priority))
        ipS, routedS, touchS, _, cache = cascade(te, chansE, pri_sw, acts, ds, model, tok, dev,
                                                 base_pred, cache)
        S = full_report(te, ipS, base_pred, meta, touchS)
        E_swapped = {"priority": pri_sw, "net": S["net"], "fixed": S["fixed"],
                     "broke": S["broke"], "routed": routedS}
        log.info("EXTENDED E (swapped %s): net=%+d", list(reversed(new_priority)), S["net"])

    # marginal / additivity / interference
    single_nets = {ch: pilot["arms"][("B_ca_rfi_only" if ch == "ca_rfi" else "C_tc_rfi_only")]["net"]
                   for ch in passed}
    overlap_rfi_pred = sum(1 for i in te if base_pred[i] == "request_for_info")
    both_fire = 0
    if len(passed) == 2:
        vA = L.rscore(chansE[passed[0]]["router"], acts[chansE[passed[0]]["obs"]], te)
        vB = L.rscore(chansE[passed[1]]["router"], acts[chansE[passed[1]]["obs"]], te)
        both_fire = sum(1 for i in te if base_pred[i] == "request_for_info"
                        and vA[i] >= chansE[passed[0]]["thr"] and vB[i] >= chansE[passed[1]]["thr"])

    summary = {
        "passed": passed, "built": True, "new_channel_order": new_priority,
        "order_selection": order_note,
        "reference_R": {**{k: R[k] for k in ("acc", "macro_f1_3cls", "fixed", "broke", "net",
                                             "damage_on_correct", "n_touched", "three_channel_after",
                                             "nt_after", "routed")},
                        "archived_net": ARCHIVED_REF_NET, "reproduction_ok": bool(repro_ok)},
        "extended_E": {k: E[k] for k in ("acc", "macro_f1_3cls", "fixed", "broke", "net",
                                         "damage_on_correct", "n_touched", "three_channel_after",
                                         "nt_after", "routed")},
        "marginal_E_minus_R": {"net": E["net"] - R["net"], "fixed": E["fixed"] - R["fixed"],
                               "broke": E["broke"] - R["broke"],
                               "acc": round(E["acc"] - R["acc"], 5)},
        "new_channel_single_pilot_nets": single_nets,
        "additivity_note": "old vs new gate populations are disjoint by base-pred "
                           "(tool_call/direct vs request_for_info); new-new overlap below",
        "overlap": {"test_rfi_pred_samples": overlap_rfi_pred,
                    "both_new_routers_fire": both_fire},
        "order_sensitivity": E_swapped,
        "final_transition_matrix_E": E["transition_matrix"],
    }
    json.dump(summary, open(EXT / "extended_cascade_summary.json", "w"), indent=2, default=float)
    _write_md(summary, R, E)
    log.info("extended cascade written")


def _write_md(s, R, E):
    with open(EXT / "extended_cascade_key_table.csv", "w") as f:
        f.write("system,acc,fixed,broke,net,damage_on_correct,three_channel_after\n")
        for name, d in [("reference_R", s["reference_R"]), ("extended_E", s["extended_E"])]:
            f.write(f"{name},{d['acc']},{d['fixed']},{d['broke']},{d['net']},"
                    f"{d['damage_on_correct']},{d['three_channel_after']}\n")
    with open(EXT / "EXTENDED_CASCADE_SUMMARY.md", "w") as f:
        f.write("# Extended Cascade — Reference R vs Extended E (seed 42, locked test)\n\n")
        f.write(f"Passed new channels: **{s['passed']}**; order: {s['order_selection']}.\n\n")
        f.write(f"Reference R reproduction: Net {R['net']:+d} vs archived +{ARCHIVED_REF_NET} "
                f"(within ±3: {s['reference_R']['reproduction_ok']}).\n\n")
        f.write("| system | acc | Fixed | Broke | Net | dmg/correct | 3ch after |\n|---|---|---|---|---|---|---|\n")
        for name, d in [("Reference R", s["reference_R"]), ("Extended E", s["extended_E"])]:
            f.write(f"| {name} | {d['acc']:.4f} | {d['fixed']} | {d['broke']} | {d['net']:+d} "
                    f"| {d['damage_on_correct']} | {d['three_channel_after']} |\n")
        mg = s["marginal_E_minus_R"]
        f.write(f"\n**Marginal (E − R):** Net {mg['net']:+d}, Fixed {mg['fixed']:+d}, "
                f"Broke {mg['broke']:+d}, acc {mg['acc']:+.4f}.\n\n")
        f.write(f"**Single-channel pilot Nets:** {s['new_channel_single_pilot_nets']} "
                f"(additivity reference).\n\n")
        f.write(f"**Overlap:** {s['overlap']['test_rfi_pred_samples']} test RFI-pred samples; "
                f"both new routers fire on {s['overlap']['both_new_routers_fire']}.\n")
        f.write(f"{s['additivity_note']}.\n\n")
        if s["order_sensitivity"]:
            os_ = s["order_sensitivity"]
            f.write(f"**Order sensitivity:** swapped order {os_['priority']} → Net {os_['net']:+d} "
                    f"(vs chosen {s['extended_E']['net']:+d}).\n\n")
        f.write("## Per-channel residual after E (nt_after)\n\n| channel | after |\n|---|---|\n")
        for ch, v in s["extended_E"]["nt_after"].items():
            f.write(f"| {ch} | {v} |\n")


if __name__ == "__main__":
    main()
