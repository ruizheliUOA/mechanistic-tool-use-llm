#!/usr/bin/env python3
"""Stage D2 + D3 — prespecified DiffMean comparators and the frozen geometry gate.

CPU-only.  No model is loaded; this consumes the L_obs/L_inj activation caches and the
Stage D1 gradient objects.

D2 comparators, both recovered from the modern Qwen3 protocol and neither substituted:
  * `L26_diffmean_frozen`   — the original DiffMean at L_obs, the mandatory comparator
  * `L21_diffmean_committed` — the same-layer DiffMean at L_inj, preregistered as a
                               diagnostic comparator in the layer control
Both use the frozen formula unit(mean(correct gold reference) - mean(gold->source error))
over TRAIN, sign convention positive intended goldward.  No PCA, LDA, CAA, low-rank or
nonlinear estimator is computed, and neither comparator is allowed to modify d_grad.

D3 applies `qwen3_8b_stage0_1.geometry_metrics` verbatim to every DEV error sample for each
estimator, then evaluates the nine-condition Stage 1.5 advancement gate with the frozen
10000-draw paired bootstrap at seed 20260731.  The gate is conjunctive and is not
restated, reweighted or relaxed here.
"""

import csv
import json

import numpy as np

import stage_d_e_common as C

EST_ORDER = ["L21_gradient_mean", "L26_diffmean_frozen", "L21_diffmean_committed"]


def load_gradients(split):
    from safetensors import safe_open
    p = C.BIG / ("QWEN3_4B_%s_GRADIENTS.safetensors" % split.upper())
    out = {}
    with safe_open(p, framework="numpy") as f:
        keys = list(f.keys())
        for k in keys:
            out[k] = f.get_tensor(k)
    return out, C.sha_file(p)


def main():
    ctx = C.Ctx()
    v1 = ctx.v1
    assert v1.HIDDEN_SIZE == C.HIDDEN

    tr, tr_sha = load_gradients("train")
    dv, dv_sha = load_gradients("dev")

    # ---------------- D2: DiffMean comparators (TRAIN only) ----------------
    dm = {"L26_diffmean_frozen": {}, "L21_diffmean_committed": {}}
    dm_manifest = {}
    for ch, g, s in C.CHANNELS:
        for name, layer in (("L26_diffmean_frozen", C.L_OBS),
                            ("L21_diffmean_committed", C.L_INJ)):
            d, raw_norm, n_err, n_ref = C.diffmean_direction(ctx, ch, layer)
            dm[name][ch] = d
            dm_manifest.setdefault(ch, {})[name] = {
                "layer": layer, "raw_norm": raw_norm, "n_error": n_err, "n_reference": n_ref,
                "unit_norm": float(np.linalg.norm(d.astype(np.float64))),
                "sha256": C.sha_arr(d), "dtype": str(d.dtype), "shape": list(d.shape),
                "formula": "unit(mean(correct gold reference) - mean(gold->source error))",
                "sign_convention": "positive intended goldward",
                "population": "TRAIN only",
            }

    dgrad = {ch: tr["d_grad__" + ch] for ch, _, _ in C.CHANNELS}

    # ---------------- D3: DEV geometry ----------------
    results, csv_rows = {}, []
    for ch, g, p in C.CHANNELS:
        gi, pi = v1.MODES.index(g), v1.MODES.index(p)
        pop = ctx.errors(ch, "dev")
        C.assert_sealed_clean(ctx, pop)
        mats = [dv["gradient__%s__%04d" % (ch, i)] for i in range(len(pop))]
        assert len(mats) == C.EXPECTED[ch]["dev_err"]

        # DEV target-contrast axes (used only for the transfer cosine, never to build d)
        Wdev = []
        for m in mats:
            z = m[gi].astype(np.float64) - m[pi].astype(np.float64)
            Wdev.append(z / np.linalg.norm(z))
        Wdev = np.stack(Wdev)

        ests = {"L21_gradient_mean": dgrad[ch],
                "L26_diffmean_frozen": dm["L26_diffmean_frozen"][ch],
                "L21_diffmean_committed": dm["L21_diffmean_committed"][ch]}
        per = {}
        for name, dd in ests.items():
            d = dd.astype(np.float64)
            vals = [v1.geometry_metrics(m, d, g, p, "full_effect") for m in mats]
            valid = [x for x in vals if x["status"] == "VALID"]
            s = np.array([x["signed_contrast"] for x in valid])
            ac = np.array([x["a_contrast"] for x in valid])
            ad = np.array([x["a_dec"] for x in valid])
            ao = np.array([x["a_offaxis"] for x in valid])
            q = np.array([x["q_offaxis"] for x in valid if x.get("q_offaxis") is not None])
            n1 = float(np.median([x["a_contrast_rank1_null_p95"] for x in valid]))
            nr = float(np.median([x["a_dec_rank_matched_null_p95"] for x in valid]))
            per[name] = {
                "n_dev": len(vals), "n_valid": len(valid),
                "statuses": {k: sum(1 for x in vals if x["status"] == k)
                             for k in sorted({x["status"] for x in vals})},
                "mean_signed_contrast": float(s.mean()),
                "median_signed_contrast": float(np.median(s)),
                "negative_fraction": float(np.mean(s < 0)),
                "median_a_contrast": float(np.median(ac)),
                "a_contrast_rank1_null_p95": n1,
                "a_contrast_null_ratio": float(np.median(ac) / n1),
                "median_a_dec": float(np.median(ad)),
                "a_dec_rank_matched_null_p95": nr,
                "a_dec_null_ratio": float(np.median(ad) / nr),
                "Q_sum": float(ao.sum() / ad.sum()),
                "median_q_offaxis": float(np.median(q)) if len(q) else None,
                "observed_rank_distribution": {str(k): sum(1 for x in vals if x["observed_rank"] == k)
                                               for k in sorted({x["observed_rank"] for x in vals})},
                "_s": s, "_ao": ao, "_ad": ad,
            }

        wbar_dev = Wdev.mean(0)
        wbar_dev_norm = float(np.linalg.norm(wbar_dev))
        wbar_dev = wbar_dev / wbar_dev_norm
        transfer_cos = float(dgrad[ch].astype(np.float64) @ wbar_dev)

        n = per["L21_gradient_mean"]["n_valid"]
        rng = np.random.default_rng(C.GEOMETRY_BOOTSTRAP_SEED)
        keys = ("transfer", "neg_grad", "Q_grad", "dQ26", "dQ21", "dneg26", "dneg21")
        bs = {k: np.empty(C.GEOMETRY_BOOTSTRAP_DRAWS) for k in keys}
        G = per["L21_gradient_mean"]
        for b in range(C.GEOMETRY_BOOTSTRAP_DRAWS):
            i = rng.integers(0, n, n)
            wb = Wdev[i].mean(0)
            wb = wb / np.linalg.norm(wb)
            bs["transfer"][b] = dgrad[ch].astype(np.float64) @ wb
            bs["neg_grad"][b] = np.mean(G["_s"][i] < 0)
            qg = G["_ao"][i].sum() / G["_ad"][i].sum()
            bs["Q_grad"][b] = qg
            for tag, key in (("L26_diffmean_frozen", "26"), ("L21_diffmean_committed", "21")):
                o = per[tag]
                bs["dQ" + key][b] = qg - (o["_ao"][i].sum() / o["_ad"][i].sum())
                bs["dneg" + key][b] = np.mean(G["_s"][i] < 0) - np.mean(o["_s"][i] < 0)
        ci = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
              for k, v in bs.items()}

        dg = dgrad[ch]
        cond = {
            "1_finite_unit_float32": bool(
                np.all(np.isfinite(dg)) and dg.dtype == np.float32
                and dg.shape == (C.HIDDEN,)
                and abs(float(np.linalg.norm(dg.astype(np.float64))) - 1) < 1e-5),
            "2_transfer_ci_lower_gt_0": ci["transfer"][0] > 0,
            "3_negative_fraction_lt_0.5": G["negative_fraction"] < 0.5,
            "4_negative_fraction_ci_upper_lt_0.5": ci["neg_grad"][1] < 0.5,
            "5_dQ_vs_L26_ci_upper_lt_0": ci["dQ26"][1] < 0,
            "6_dQ_vs_L21_ci_upper_lt_0": ci["dQ21"][1] < 0,
            "7_dneg_vs_L26_ci_upper_lt_0": ci["dneg26"][1] < 0,
            "8_dneg_vs_L21_ci_upper_lt_0": ci["dneg21"][1] < 0,
            "9_no_inconsistency": True,
        }
        decision = "GEOMETRY_QUALIFIED" if all(cond.values()) else "DECLINE_ESTIMATOR"
        train_sum = json.loads((C.HERE / "_STAGE_D_train_SUMMARY.json").read_text())
        tc = train_sum["per_channel"][ch]
        results[ch] = {
            "gold": g, "source": p, "n_dev_error": n,
            "train_direction_norm": tc["d_grad_norm"],
            "train_wbar_norm": tc["wbar_norm"],
            "train_chance_reference": tc["chance_reference_one_over_sqrt_n"],
            "train_n": tc["n"],
            "dev_target_contrast_mean_direction_norm": wbar_dev_norm,
            "dev_target_contrast_mean_direction_sha256": C.sha_arr(wbar_dev.astype(np.float32)),
            "transfer_cosine_train_to_dev": transfer_cos,
            "transfer_cosine_ci95": ci["transfer"],
            "negative_fraction_ci95": ci["neg_grad"],
            "Q_sum_ci95": ci["Q_grad"],
            "delta_Q_vs_L26": G["Q_sum"] - per["L26_diffmean_frozen"]["Q_sum"],
            "delta_Q_vs_L26_ci95": ci["dQ26"],
            "delta_Q_vs_L21": G["Q_sum"] - per["L21_diffmean_committed"]["Q_sum"],
            "delta_Q_vs_L21_ci95": ci["dQ21"],
            "delta_negative_vs_L26": (G["negative_fraction"]
                                      - per["L26_diffmean_frozen"]["negative_fraction"]),
            "delta_negative_vs_L26_ci95": ci["dneg26"],
            "delta_negative_vs_L21": (G["negative_fraction"]
                                      - per["L21_diffmean_committed"]["negative_fraction"]),
            "delta_negative_vs_L21_ci95": ci["dneg21"],
            "cos_dgrad_L26_diffmean": float(dgrad[ch].astype(np.float64)
                                            @ dm["L26_diffmean_frozen"][ch].astype(np.float64)),
            "cos_dgrad_L21_diffmean": float(dgrad[ch].astype(np.float64)
                                            @ dm["L21_diffmean_committed"][ch].astype(np.float64)),
            "estimators": {k: {kk: vv for kk, vv in per[k].items() if not kk.startswith("_")}
                           for k in EST_ORDER},
            "gate_conditions": cond, "decision": decision,
        }
        for name in EST_ORDER:
            e = results[ch]["estimators"][name]
            csv_rows.append({
                "channel": ch, "estimator": name, "n_dev_error": e["n_valid"],
                "train_direction_norm": (tc["d_grad_norm"] if name == "L21_gradient_mean"
                                         else dm_manifest[ch][name]["unit_norm"]),
                "dev_target_contrast_mean_norm": wbar_dev_norm,
                "train_to_dev_cosine": (transfer_cos if name == "L21_gradient_mean" else None),
                "train_to_dev_cosine_ci_lo": (ci["transfer"][0] if name == "L21_gradient_mean" else None),
                "train_to_dev_cosine_ci_hi": (ci["transfer"][1] if name == "L21_gradient_mean" else None),
                "target_contrast_alignment_mean_signed": e["mean_signed_contrast"],
                "target_contrast_alignment_median_signed": e["median_signed_contrast"],
                "wrong_sign_fraction": e["negative_fraction"],
                "median_a_contrast": e["median_a_contrast"],
                "a_contrast_random_null_p95": e["a_contrast_rank1_null_p95"],
                "a_contrast_null_ratio": e["a_contrast_null_ratio"],
                "median_a_dec": e["median_a_dec"],
                "a_dec_random_null_p95": e["a_dec_rank_matched_null_p95"],
                "a_dec_null_ratio": e["a_dec_null_ratio"],
                "Q_sum_offaxis": e["Q_sum"], "median_q_offaxis": e["median_q_offaxis"],
                "decision": (decision if name == "L21_gradient_mean" else "COMPARATOR"),
            })
        print("%s: transfer_cos=%.4f CI=[%.4f,%.4f]  neg=%.4f CI_hi=%.4f  "
              "Q=%.4f dQ26=%+.4f(hi %+.4f) dQ21=%+.4f(hi %+.4f) -> %s"
              % (ch, transfer_cos, ci["transfer"][0], ci["transfer"][1],
                 G["negative_fraction"], ci["neg_grad"][1], G["Q_sum"],
                 results[ch]["delta_Q_vs_L26"], ci["dQ26"][1],
                 results[ch]["delta_Q_vs_L21"], ci["dQ21"][1], decision), flush=True)

    # ---------------- outputs ----------------
    from safetensors.numpy import save_file
    save_file({("d_grad__" + ch): dgrad[ch] for ch, _, _ in C.CHANNELS},
              C.HERE / "PRIMARY_DIRECTIONS.safetensors",
              metadata={"schema_version": "1",
                        "estimator": "unit(mean_i unit(G_i,gold - G_i,source)) over TRAIN errors",
                        "layer": str(C.L_INJ), "dimension": str(C.HIDDEN),
                        "model": "%s@%s" % (C.MODEL_REPO, C.MODEL_REVISION),
                        "sign_convention": "+d increases e_gold - e_source to first order",
                        "contains_dataset_text": "false"})
    save_file({("%s__%s" % (n, ch)): dm[n][ch] for n in dm for ch, _, _ in C.CHANNELS},
              C.HERE / "DIFFMEAN_DIRECTIONS.safetensors",
              metadata={"schema_version": "1",
                        "formula": "unit(mean(correct gold reference) - mean(gold->source error))",
                        "layers": json.dumps({"L26_diffmean_frozen": C.L_OBS,
                                              "L21_diffmean_committed": C.L_INJ}),
                        "dimension": str(C.HIDDEN), "contains_dataset_text": "false"})

    train_sum = json.loads((C.HERE / "_STAGE_D_train_SUMMARY.json").read_text())
    dev_sum = json.loads((C.HERE / "_STAGE_D_dev_SUMMARY.json").read_text())
    C.wj(C.HERE / "PRIMARY_DIRECTIONS_MANIFEST.json", {
        "estimator": "d_grad = unit(mean_i unit(G_i,gold - G_i,source)) at L_inj, TRAIN only",
        "layer": C.L_INJ, "hidden_size": C.HIDDEN,
        "model": "%s@%s" % (C.MODEL_REPO, C.MODEL_REVISION),
        "aggregation": "full-effect: sum over prompt and candidate positions",
        "site": "model.model.layers[L].mlp forward output, after MLP internals "
                "and before decoder residual addition",
        "per_channel": train_sum["per_channel"],
        "train_gradients_sha256": tr_sha, "dev_gradients_sha256": dv_sha,
        "train_runtime_seconds": train_sum["runtime_seconds"],
        "dev_runtime_seconds": dev_sum["runtime_seconds"],
        "train_peak_allocated_gib": train_sum["peak_allocated_gib"],
        "dev_peak_allocated_gib": dev_sum["peak_allocated_gib"],
        "gram_matrix": {
            "channels": [c for c, _, _ in C.CHANNELS],
            "matrix": (np.stack([dgrad[c].astype(np.float64) for c, _, _ in C.CHANNELS])
                       @ np.stack([dgrad[c].astype(np.float64)
                                   for c, _, _ in C.CHANNELS]).T).tolist()},
        "file": "PRIMARY_DIRECTIONS.safetensors",
        "file_sha256": C.sha_file(C.HERE / "PRIMARY_DIRECTIONS.safetensors"),
        "amendment": {"package": "qwen3_4b_gradient_memory_amendment_v1",
                      "hashes_sha256": C.sha_file(C.AMEND / "HASHES.json"),
                      "verdict": "QWEN3_4B_STAGE_D_AMENDMENT_READY",
                      "equivalence": "BITWISE_EQUIVALENT"},
        "sealed_evaluation_accessed": False})
    C.wj(C.HERE / "DIFFMEAN_DIRECTIONS_MANIFEST.json", {
        "role": "prespecified comparators; may not modify the primary direction",
        "per_channel": dm_manifest,
        "file": "DIFFMEAN_DIRECTIONS.safetensors",
        "file_sha256": C.sha_file(C.HERE / "DIFFMEAN_DIRECTIONS.safetensors"),
        "substituted_estimators": [],
        "note": "No PCA, LDA, CAA variant, low-rank or nonlinear direction was computed."})
    C.wj(C.HERE / "_GEOMETRY_RESULTS.json", {
        "bootstrap": {"draws": C.GEOMETRY_BOOTSTRAP_DRAWS,
                      "seed": C.GEOMETRY_BOOTSTRAP_SEED, "paired": True},
        "gate": "frozen Stage 1.5 nine-condition conjunction",
        "per_channel": results,
        "qualified": [ch for ch in results if results[ch]["decision"] == "GEOMETRY_QUALIFIED"],
        "sealed_evaluation_accessed": False})
    cols = list(csv_rows[0])
    with open(C.HERE / "DIRECTION_GEOMETRY_AUDIT.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(csv_rows)
    print("GEOMETRY_COMPLETE qualified=%s"
          % [ch for ch in results if results[ch]["decision"] == "GEOMETRY_QUALIFIED"])


if __name__ == "__main__":
    main()
