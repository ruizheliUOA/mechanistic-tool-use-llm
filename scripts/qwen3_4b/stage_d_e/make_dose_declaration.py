#!/usr/bin/env python3
"""Qwen3-4B activation-scale dose declaration.

CPU-only.  Frozen procedure, recovered from `final/results/qwen3_stage2a_dose`, applied
to the new host model.  Nothing about the procedure is chosen here:

  scaling population : all TRAIN rows where the frozen channel-c Router gate is active,
                       i.e. baseline prediction == channel source AND
                       committed Router predict_proba >= committed tau
  scale statistic    : s_c = median_x || h_L21(x) ||_2 in float64 at the frozen feature
                       position (last prompt token), at the frozen injection site
  injection          : h' = h + q * s_c * d, so || delta h ||_2 = q * s_c
  grid               : q in {0.0, 0.125, 0.25, 0.5, 1.0, 2.0}, not extendable

The population is NOT conditioned on DEV results, behavioural movement, target-hit,
intervention success, d_grad alignment, baseline margin or any evaluation information.
q = 1 is a scale unit, not a validated operating dose.

This declaration authorises no behavioural run by itself.
"""

import json

import numpy as np

import stage_d_e_common as C


def main():
    ctx = C.Ctx()
    per = {}
    for ch, g, s in C.CHANNELS:
        routed_tr, pop_tr, pr_tr = ctx.routed(ch, "train")
        C.assert_sealed_clean(ctx, routed_tr)
        H = ctx.A[C.L_INJ][[ctx.rowpos[r["project_index"]] for r in routed_tr]].astype(np.float64)
        norms = np.linalg.norm(H, axis=1)
        assert np.all(np.isfinite(norms))
        s_c = float(np.median(norms))
        per[ch] = {
            "gold": g, "source": s, "tau": ctx.tau[ch],
            "train_pred_eq_source": len(pop_tr), "router_active": len(routed_tr),
            "all_finite": True,
            "min": float(norms.min()), "q25": float(np.quantile(norms, .25)),
            "s_c": s_c, "mean": float(norms.mean()),
            "q75": float(np.quantile(norms, .75)), "max": float(norms.max()),
            "iqr": float(np.quantile(norms, .75) - np.quantile(norms, .25)),
            "population_sha256": ctx.v1.index_order_hash([r["sample_id"] for r in routed_tr]),
            "absolute_delta_norm_by_q": {str(q): q * s_c for q in C.Q_GRID},
        }
        print("%s: tau=%s  pred==source=%d  router-active=%d  s_c=%.10f"
              % (ch, ctx.tau[ch], len(pop_tr), len(routed_tr), s_c))

    C.wj(C.HERE / "DOSE_DECLARATION.json", {
        "status": "FROZEN_BEFORE_ANY_DEV_BEHAVIOURAL_INTERVENTION",
        "model": "%s@%s" % (C.MODEL_REPO, C.MODEL_REVISION),
        "layer": C.L_INJ, "feature_position": "last prompt token",
        "site": "model.model.layers[L].mlp forward output, after MLP internals "
                "and before decoder residual addition",
        "scale_statistic": "s_c = median_x ||h_L21(x)||_2 over the TRAIN Router-active population",
        "scaling_population_rule": "baseline prediction == channel source AND "
                                   "committed Router predict_proba >= committed tau; TRAIN only",
        "rho_ref": 1.0,
        "injection": "h' = h + q * s_c * d",
        "q_grid": C.Q_GRID, "grid_extendable": False,
        "matched_activation_budget": ["d_grad", "L26_diffmean_frozen", "L21_diffmean_committed",
                                      "reverse_d_grad", "matched_random"],
        "wrong_layer_semantics": "same absolute norm q*s_c; NOT renormalised by the wrong "
                                 "layer's activation norm",
        "ungated_semantics": "same q*s_c; only the Router gate is disabled",
        "conditioned_on_dev_or_evaluation": False,
        "per_channel": per,
        "sealed_evaluation_accessed": False})
    print("DOSE_DECLARATION_WRITTEN")


if __name__ == "__main__":
    main()
