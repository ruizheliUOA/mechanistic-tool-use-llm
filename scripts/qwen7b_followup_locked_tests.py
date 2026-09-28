"""
qwen7b_followup_locked_tests.py — two small locked-test confirmations for the Qwen2.5-7B
method diagnostic (bf16, seed=42). NON-invasive: reuses archived helpers from
qwen7b_method_diagnostics.py / qwen7b_native_sakiko*.py; touches no W2C/thesis files.

Task A — ca_direct@L20-in-cascade locked test:
  Baseline vs archived seed=42 native SAKIKO cascade (ca_direct obs L16) vs updated cascade
  (ca_direct obs L20, inj L16, alpha 6, thr 0.40). rfi_tc/ca_tc kept at archived configs.
  The 3 channels partition by base prediction (ca_direct gates pred==direct; rfi_tc/ca_tc gate
  pred==tool_call), so the swap only affects ca_direct-routed samples.

Task B — ca_tc PCA-1 locked test:
  ca_tc DiffMean vs PCA-1 at the archived ca_tc setup (obs L20, inj L16, alpha 4). Lock the
  firing threshold on val, evaluate on test; reverse-direction control included (cheap).

Outputs -> final/results/7b_w2c_sakiko/method_diagnostics/gpu_confirmation/followup_locked_tests/
Usage:  python scripts/qwen7b_followup_locked_tests.py
"""
from __future__ import annotations
import json, pickle, gc, sys, logging
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# importing the driver installs the robust (datasets-version-proof) loader on P
import run_qwen7b_method_diag_gpu as DRV      # noqa: F401  (side effect: patches loader)
import qwen7b_native_sakiko as P
import qwen7b_method_diagnostics as D
from qwen7b_native_sakiko_run import _metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("followup")

OUT = P.ROOT / "final" / "results" / "7b_w2c_sakiko" / "method_diagnostics" / \
    "gpu_confirmation" / "followup_locked_tests"
OUT.mkdir(parents=True, exist_ok=True)
ALL_OBS = [16, 18, 20, 22]
REFG = {"rfi_tc": "request_for_info", "ca_tc": "cannot_answer", "ca_direct": "cannot_answer"}

# archived seed=42 locked cascade config (from qwen25_7b_locked_config.json)
ARCHIVED = {
    "rfi_tc":    {"obs": 20, "inj": 18, "alpha": 2.0, "thr": 0.7},
    "ca_tc":     {"obs": 20, "inj": 16, "alpha": 4.0, "thr": 0.8},
    "ca_direct": {"obs": 16, "inj": 16, "alpha": 6.0, "thr": 0.4},
}
CA_DIRECT_L20 = {"obs": 20, "inj": 16, "alpha": 6.0, "thr": 0.4}


def full_metrics(idxs, meta, base_pred, int_pred, touched):
    m = _metrics(idxs, meta, base_pred, int_pred)
    dmg = sum(1 for i in touched if meta[i]["correct"] and int_pred[i] != meta[i]["gold"])
    n_tc = sum(1 for i in touched if meta[i]["correct"])
    three_before = sum(1 for i in idxs if meta[i]["etype"] in P.CHANNELS)
    three_after = sum(1 for i in idxs if (
        (meta[i]["gold"] == "request_for_info" and int_pred[i] == "tool_call") or
        (meta[i]["gold"] == "cannot_answer" and int_pred[i] == "tool_call") or
        (meta[i]["gold"] == "cannot_answer" and int_pred[i] == "direct")))
    return {"acc": m["acc"], "fixed": m["fixed"], "broke": m["broke"], "net": m["net"],
            "chan_fixed": m["chan_fixed"], "nt_before": m["nt_before"], "nt_after": m["nt_after"],
            "three_channel_before": three_before, "three_channel_after": three_after,
            "n_touched": len(touched), "touched_correct": n_tc, "damage_on_correct": dmg}


def main():
    ds, meta = P.load_meta_and_dataset()
    tr, va, te = P.load_splits()
    acts = {L: np.load(P.CACHE_DIR / f"acts_L{L}.npy") for L in ALL_OBS}
    routers = pickle.load(open(P.CACHE_DIR / "routers.pkl", "rb"))       # archived (ca_direct@L16)
    directions = pickle.load(open(P.CACHE_DIR / "directions.pkl", "rb"))  # archived units
    base_pred = {i: meta[i]["pred"] for i in range(len(meta))}
    base_acc = {nm: sum(1 for i in S if base_pred[i] == meta[i]["gold"]) / len(S)
                for nm, S in [("val", va), ("test", te)]}

    model, tok, dev = P.load_model()

    # ── build per-channel intervention units/routers/mednorms ────────────────────
    def diffmean_unit(ch, obsL):
        err, ref = D.get_pools(meta, tr, ch, REFG[ch])
        unit, nrm = D.diffmean(acts[obsL], err, ref)
        return unit, nrm, float(np.median(np.linalg.norm(acts[obsL][tr], axis=1)))

    # archived channel bundle: router from routers.pkl, DiffMean unit from directions.pkl
    def arch_bundle(ch):
        c = ARCHIVED[ch]
        r = routers[ch]
        router = {"clf": r["clf"], "sc": r["scaler"], "feat": (lambda idxs: acts[r["obs_layer"]][idxs])}
        return {"obs": c["obs"], "inj": c["inj"], "alpha": c["alpha"], "thr": c["thr"],
                "unit": directions[ch]["unit"], "mednorm": directions[ch]["median_norm"],
                "router": router, "router_obs": r["obs_layer"]}

    # ca_direct @ L20 bundle: fresh L20 router + DiffMean@L20
    def cadir_l20_bundle():
        c = CA_DIRECT_L20
        rt = D.fit_router(acts[20], meta, tr, va, "ca_direct")   # LR@L20, pos=ca_direct neg=correct
        unit, nrm, mednorm = diffmean_unit("ca_direct", 20)
        return {"obs": 20, "inj": c["inj"], "alpha": c["alpha"], "thr": c["thr"],
                "unit": unit, "mednorm": mednorm, "router": rt, "router_obs": 20,
                "_dir_norm": nrm, "_router_val_auc": rt["val_auc"]}

    def rscore(bundle, idxs):
        return D.router_scores(bundle["router"], acts[bundle["router_obs"]], idxs)

    def cascade_pred(idxs, channels):
        vsc = {ch: rscore(channels[ch], idxs) for ch in channels}
        corr = {}
        for ch, b in channels.items():
            elig = [i for i in idxs if base_pred[i] == P.CHANNELS[ch]["from_pred"] and vsc[ch][i] >= b["thr"]]
            corr[ch] = D.precompute_corrected(model, tok, dev, ds, elig, b["unit"], b["mednorm"], b["inj"], b["alpha"])
        int_pred = dict(base_pred); routed = Counter(); touched = []
        for i in idxs:
            for ch in P.PRIORITY:
                if ch not in channels:
                    continue
                b = channels[ch]
                if base_pred[i] == P.CHANNELS[ch]["from_pred"] and vsc[ch][i] >= b["thr"]:
                    int_pred[i] = corr[ch][i]; routed[ch] += 1; touched.append(i); break
        return int_pred, dict(routed), touched

    # ════════ TASK A: ca_direct@L20-in-cascade locked test ════════
    log.info("===== TASK A: ca_direct L20 in cascade =====")
    arch_channels = {ch: arch_bundle(ch) for ch in P.CHANNELS}
    cad20 = cadir_l20_bundle()
    log.info("  ca_direct@L20: dir_norm=%.3f router_val_auc=%.4f (cf CPU 10.41 / 0.9555)",
             cad20["_dir_norm"], cad20["_router_val_auc"])
    upd_channels = {"rfi_tc": arch_channels["rfi_tc"], "ca_tc": arch_channels["ca_tc"],
                    "ca_direct": cad20}

    ip_arch, routed_arch, touched_arch = cascade_pred(te, arch_channels)
    m_arch = full_metrics(te, meta, base_pred, ip_arch, touched_arch)
    m_arch["routed"] = routed_arch
    log.info("  ARCHIVED cascade (test): acc %.4f->%.4f net=%+d (fix=%d broke=%d) cadir %d->%d routed=%s",
             base_acc["test"], m_arch["acc"], m_arch["net"], m_arch["fixed"], m_arch["broke"],
             m_arch["nt_before"]["ca_direct"], m_arch["nt_after"]["ca_direct"], routed_arch)

    ip_upd, routed_upd, touched_upd = cascade_pred(te, upd_channels)
    m_upd = full_metrics(te, meta, base_pred, ip_upd, touched_upd)
    m_upd["routed"] = routed_upd
    log.info("  UPDATED  cascade (test): acc %.4f->%.4f net=%+d (fix=%d broke=%d) cadir %d->%d routed=%s",
             base_acc["test"], m_upd["acc"], m_upd["net"], m_upd["fixed"], m_upd["broke"],
             m_upd["nt_before"]["ca_direct"], m_upd["nt_after"]["ca_direct"], routed_upd)

    taskA = {
        "baseline_test_acc": round(base_acc["test"], 5),
        "archived_cascade": m_arch, "updated_cascade": m_upd,
        "ca_direct_config": {"archived": ARCHIVED["ca_direct"], "updated": CA_DIRECT_L20,
                             "updated_dir_norm": round(cad20["_dir_norm"], 4),
                             "updated_router_val_auc": round(cad20["_router_val_auc"], 4)},
        "archived_reference_json": {"net": 79, "fixed": 92, "broke": 13, "sakiko_acc": 0.5566},
        "delta_net_updated_minus_archived": m_upd["net"] - m_arch["net"],
    }
    json.dump(taskA, open(OUT / "taskA_ca_direct_l20_cascade.json", "w"), indent=2, default=float)

    # ════════ TASK B: ca_tc PCA-1 vs DiffMean locked test ════════
    log.info("===== TASK B: ca_tc DiffMean vs PCA-1 =====")
    ch = "ca_tc"; gate = P.CHANNELS[ch]["from_pred"]; obsL, inj, alpha = 20, 16, 4.0
    err, ref = D.get_pools(meta, tr, ch, REFG[ch])
    dm_unit, dm_norm = D.diffmean(acts[obsL], err, ref)
    p1_unit, p1_norm = D.pca_dir(acts[obsL], err, ref, k=1)
    cos_dm_p1 = float(dm_unit @ p1_unit)
    mednorm = float(np.median(np.linalg.norm(acts[obsL][tr], axis=1)))
    rt = routers[ch]
    router = {"clf": rt["clf"], "sc": rt["scaler"], "feat": (lambda idxs: acts[rt["obs_layer"]][idxs])}
    vscore = D.router_scores(router, acts[rt["obs_layer"]], va)
    tscore = D.router_scores(router, acts[rt["obs_layer"]], te)

    def eval_dir(name, unit, sign=1.0):
        u = (sign * unit).astype(np.float32)
        # val: precompute firing-eligible then lock best thr
        velig = [i for i in va if base_pred[i] == gate and vscore[i] >= min(P.THRESHOLDS)]
        vcorr = D.precompute_corrected(model, tok, dev, ds, velig, u, mednorm, inj, alpha)
        best = None
        for thr in P.THRESHOLDS:
            r = D.channel_net(va, meta, base_pred, ch, gate, vscore, vcorr, thr)
            if best is None or (r["net"], -r["broke"]) > (best["net"], -best["broke"]):
                best = {"thr": thr, **r}
        # test at locked thr
        telig = [i for i in te if base_pred[i] == gate and tscore[i] >= best["thr"]]
        tcorr = D.precompute_corrected(model, tok, dev, ds, telig, u, mednorm, inj, alpha)
        rt_test = D.channel_net(te, meta, base_pred, ch, gate, tscore, tcorr, best["thr"])
        log.info("  ca_tc [%s]: locked_thr=%.2f | val net=%+d (%d->%d) | test net=%+d (%d->%d fix=%d broke=%d)",
                 name, best["thr"], best["net"], best["ch_before"], best["ch_after"],
                 rt_test["net"], rt_test["ch_before"], rt_test["ch_after"], rt_test["fixed"], rt_test["broke"])
        return {"locked_thr": best["thr"],
                "val": {"net": best["net"], "fixed": best["fixed"], "broke": best["broke"],
                        "before": best["ch_before"], "after": best["ch_after"]},
                "test": {"net": rt_test["net"], "fixed": rt_test["fixed"], "broke": rt_test["broke"],
                         "before": rt_test["ch_before"], "after": rt_test["ch_after"]}}

    resB = {
        "config": {"obs": obsL, "inj": inj, "alpha": alpha, "router_obs": rt["obs_layer"]},
        "direction_norm": {"diffmean": round(dm_norm, 4), "pca1": round(p1_norm, 4)},
        "cos_diffmean_pca1": round(cos_dm_p1, 4),
        "diffmean": eval_dir("diffmean", dm_unit),
        "pca1": eval_dir("pca1", p1_unit),
        "pca1_reverse_control": eval_dir("pca1_reverse", p1_unit, sign=-1.0),
        "diffmean_reverse_control": eval_dir("diffmean_reverse", dm_unit, sign=-1.0),
        "random_control": "TODO (per-channel random not in ready helper; reverse control provided)",
    }
    json.dump(resB, open(OUT / "taskB_ca_tc_pca_vs_diffmean.json", "w"), indent=2, default=float)
    log.info("===== follow-up locked tests done =====")


if __name__ == "__main__":
    main()
