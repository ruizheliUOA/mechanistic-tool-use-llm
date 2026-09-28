#!/usr/bin/env python3
"""Formal freeze preparation for the advanced Qwen3-4B channel.

CPU-only.  Prepares but does NOT execute the formal protocol.  SEALED TEST is not opened:
the evaluation partition is inherited by reference (identity pinned by the ordered sealed
uuid hash) and no sealed row is loaded, rendered, tokenised, scored or counted.

Everything frozen here is transcribed from the Qwen3-8B formal preregistration
(`final/results/qwen3_stage2_formal/QWEN3_STAGE2_PREREGISTRATION.json`, `_GATE_SPEC.md`,
`_ENDPOINT_SPEC.md`, `_VOID_SCHEMA.json`).  The only substitutions are host-model facts
(checkpoint, width), this replication's own frozen values (channel, tau, s_c, q, b_c,
direction hashes), and a fresh random salt so the formal null cannot reuse Qwen3-8B vectors.
"""

import hashlib
import json

import numpy as np

import stage_d_e_common as C

K = 59
RANDOM_SALT = "SAKIKO_QWEN3_4B_CA_TC_STAGE2_RANDOM_V1|"
BOOTSTRAP_SALT = "SAKIKO_QWEN3_4B_CA_TC_STAGE2_BOOTSTRAP_V1"
HISTORICAL_SEEDS = [20260729, 20260730, 20260731, 20260801]


def formal_randoms(dim):
    """Frozen construction, transcribed from QWEN3_STAGE2_RANDOM_MANIFEST.json."""
    vecs, seeds = [], []
    for i in range(K):
        sb = hashlib.sha256((RANDOM_SALT + str(i)).encode("utf-8")).digest()
        seed = int.from_bytes(sb[:8], "big", signed=False)
        seeds.append(seed)
        g = np.random.Generator(np.random.PCG64DXSM(seed))
        z = g.standard_normal(dim)
        vecs.append((z / np.linalg.norm(z)).astype(np.float32))
    return np.stack(vecs), seeds


def main():
    from safetensors import safe_open
    from safetensors.numpy import save_file

    ctx = C.Ctx()
    stage_e = json.loads((C.HERE / "_STAGE_E_RESULTS.json").read_text())
    geo = json.loads((C.HERE / "_GEOMETRY_RESULTS.json").read_text())
    dose = json.loads((C.HERE / "DOSE_DECLARATION.json").read_text())
    eligible = stage_e["eligible"]
    if len(eligible) != 1:
        raise RuntimeError("formal selection expects exactly one eligible channel, got %r"
                           % eligible)
    ch = eligible[0]
    g, s = ctx.meta(ch)
    r = stage_e["results"][ch]
    q = r["selected_q"]
    s_c = dose["per_channel"][ch]["s_c"]
    b_c = r["score_comparator"]["b_c"]

    with safe_open(C.HERE / "PRIMARY_DIRECTIONS.safetensors", framework="numpy") as f:
        d_grad = f.get_tensor("d_grad__" + ch)
    with safe_open(C.HERE / "DIFFMEAN_DIRECTIONS.safetensors", framework="numpy") as f:
        d26 = f.get_tensor("L26_diffmean_frozen__" + ch)
        d21 = f.get_tensor("L21_diffmean_committed__" + ch)

    R, seeds = formal_randoms(C.HIDDEN)
    assert not (set(seeds) & set(C.DEV_RANDOM_SEEDS) & set(HISTORICAL_SEEDS))
    assert len(set(seeds)) == K
    save_file({"formal_randoms": R}, C.HERE / "FORMAL_RANDOMS.safetensors",
              metadata={"schema_version": "1", "K": str(K), "salt": RANDOM_SALT,
                        "dimension": str(C.HIDDEN),
                        "construction": "seed=uint64(SHA256(salt+decimal(i))[:8]); "
                                        "PCG64DXSM; z~N(0,I) float64; normalise in "
                                        "float64; cast once to float32",
                        "no_rejection": "true", "no_orthogonalization": "true",
                        "no_cosine_filtering": "true", "no_outcome_matching": "true",
                        "contains_dataset_text": "false"})

    # descriptive only, recorded AFTER the complete set was frozen; no vector removed
    cg = (R.astype(np.float64) @ d_grad.astype(np.float64))
    c26 = (R.astype(np.float64) @ d26.astype(np.float64))

    # sealed identity, by reference only
    rows, sealed = C.load_population()
    raw = [json.loads(l) for l in open(C.RAW, encoding="utf-8")]
    perm = np.random.default_rng(C.PERM_SEED).permutation(C.TOTAL_ROWS)
    sealed_uuids = [raw[perm[i]]["uuid"] for i in sorted(sealed)]
    sealed_hash = ctx.v1.index_order_hash(sealed_uuids)

    runner_files = ["stage_d_e_common.py", "run_stage_d_gradients.py",
                    "run_stage_d_geometry.py", "make_dose_declaration.py",
                    "run_stage_e_dev.py", "analyze_stage_e.py", "make_formal_freeze.py"]

    frozen = {
        "schema_version": 1,
        "protocol_version": "QWEN3_4B_STAGE_D_E_V1",
        "status": "FROZEN_BEFORE_ANY_SEALED_ACCESS",
        "tier": "CONFIRMATORY / PREPARED_NOT_EXECUTED",
        "executed": False,
        "sealed_evaluation": {"count": C.N_SEALED, "opened": False,
                              "identity_inherited_by_reference": True,
                              "ordered_uuid_sha256": sealed_hash},
        "formal_question": ("On the untouched evaluation population, does the frozen "
                            "TRAIN-only d_grad direction produce destination-correct "
                            "behaviour that exceeds a prospectively generated "
                            "matched-random null?"),
        "formal_channel": {"channel": ch, "gold": g, "source": s, "single_channel": True},
        "frozen_configuration": {
            "model": C.MODEL_REPO, "model_revision": C.MODEL_REVISION,
            "num_hidden_layers": C.N_LAYERS, "hidden_size": C.HIDDEN,
            "L_obs": C.L_OBS, "L_inj": C.L_INJ,
            "site": "model.model.layers[L].mlp forward output, after MLP internals "
                    "and before decoder residual addition",
            "intervention_positions": "all sequence positions, matching the frozen "
                                      "full-effect aggregation",
            "dtype": "bfloat16", "attention_implementation": "eager", "batch_size": 1,
            "tau": ctx.tau[ch], "s_c": s_c, "q": q, "absolute_perturbation_norm": q * s_c,
            "b_c": b_c, "wrong_layer": C.WRONG_LAYER,
            "loader_semantics": "committed load_model(): eval(); use_cache=False; "
                                "all parameters requires_grad_(False)",
        },
        "router": {
            "file": "ROUTER_%s.npz" % ch.replace("__to__", "__"),
            "source_package": "research_exploration/qwen3_4b_w2c_prospective_replication_v1",
            "sha256": C.sha_file(C.PROSP / ("ROUTER_%s.npz" % ch.replace("__to__", "__"))),
            "tau": ctx.tau[ch], "dev_auc": 0.9237,
            "gate": "baseline prediction == source AND predict_proba >= tau",
            "refit_forbidden": True,
        },
        "directions": {
            "primary_d_grad": {"file": "PRIMARY_DIRECTIONS.safetensors",
                               "key": "d_grad__" + ch, "sha256": C.sha_arr(d_grad),
                               "norm": float(np.linalg.norm(d_grad.astype(np.float64)))},
            "L26_diffmean_frozen": {"file": "DIFFMEAN_DIRECTIONS.safetensors",
                                    "key": "L26_diffmean_frozen__" + ch,
                                    "sha256": C.sha_arr(d26)},
            "L21_same_layer_diffmean": {"key": "L21_diffmean_committed__" + ch,
                                        "sha256": C.sha_arr(d21),
                                        "role": "DEV diagnostic only; NOT promoted to the "
                                                "formal arm battery, per the frozen protocol"},
        },
        "formal_randoms": {
            "K": K, "salt": RANDOM_SALT, "dimension": C.HIDDEN,
            "construction": "seed=uint64(SHA256(salt+decimal(i))[:8]); PCG64DXSM; "
                            "z~N(0,I) float64; normalise in float64; cast once to float32",
            "seeds": seeds, "file": "FORMAL_RANDOMS.safetensors",
            "file_sha256": C.sha_file(C.HERE / "FORMAL_RANDOMS.safetensors"),
            "matrix_sha256": C.sha_arr(R),
            "no_rejection": True, "no_orthogonalization": True, "no_cosine_filtering": True,
            "no_outcome_matching": True, "no_post_hoc_selection": True, "no_regeneration": True,
            "seed_disjointness": {
                "dev_development_seeds": C.DEV_RANDOM_SEEDS,
                "other_historical_seeds": HISTORICAL_SEEDS,
                "formal_seed_min": min(seeds), "formal_seed_max": max(seeds),
                "intersection_empty": True,
                "qwen3_8b_salt_distinct": True},
            "cos_with_d_grad": {"min": float(cg.min()), "median": float(np.median(cg)),
                                "max": float(cg.max()), "abs_max": float(np.abs(cg).max())},
            "cos_with_L26_diffmean": {"min": float(c26.min()),
                                      "median": float(np.median(c26)),
                                      "max": float(c26.max())},
            "cosine_note": "recorded descriptively AFTER the complete set was frozen; "
                           "no vector was removed on their basis",
        },
        "arm_battery": ["baseline", "zero", "real TRAIN-only d_grad",
                        "K=59 fresh matched-random directions",
                        "frozen original L26 DiffMean", "reverse -d_grad",
                        "wrong-layer d_grad at L26", "ungated d_grad",
                        "frozen score-space target/source comparator"],
        "excluded_arms": ["same-layer L21 DiffMean", "another channel's d_grad",
                          "orthogonalized mismatch", "pooled or shared direction",
                          "PCA or rank-2 direction", "any new estimator",
                          "any additional dose"],
        "execution_order": ["environment and hash verification", "model load",
                            "evaluation baseline", "zero arm",
                            "exact zero-equality gate", "real d_grad",
                            "frozen L26 DiffMean", "reverse", "wrong-layer", "ungated",
                            "K formal randoms in seed order", "score comparator",
                            "endpoint computation", "formal report generation"],
        "endpoints": {
            "source_exits": "count(baseline pred = source AND intervention pred != source)",
            "gold_arrivals": "count(baseline pred = source AND intervention pred = gold)",
            "wrong_to_wrong": "count(baseline pred = source AND intervention pred != source "
                              "AND intervention pred != gold)",
            "target_hit": "gold_arrivals / source_exits",
            "target_gain_count": "gold_arrivals - wrong_to_wrong",
            "target_gain_rate": "target_gain_count / n_channel_error",
            "clean_collateral_rate": "baseline-correct rows made incorrect / "
                                     "baseline-correct rows",
            "destination_resolution": ["SOURCE_RETAINED", "GOLD_ARRIVAL", "OTHER_WRONG"],
            "collateral_resolution": ["CORRECT_RETAINED", "BROKEN_TO_SOURCE",
                                      "BROKEN_TO_OTHER"],
        },
        "populations": {
            "baseline": "single baseline over all %d evaluation rows computed first" % C.N_SEALED,
            "channel_error": "gold = %s AND baseline prediction = %s" % (g, s),
            "safety_denominator": "baseline prediction = gold",
            "never_redefined_by_intervention_results": True,
        },
        "bootstrap": {"draws": 10000, "frozen_before_execution": True,
                      "seed_salt": BOOTSTRAP_SALT,
                      "seed_derivation": "uint64 of first 8 bytes of SHA256(seed_salt)",
                      "seed": int.from_bytes(
                          hashlib.sha256(BOOTSTRAP_SALT.encode()).digest()[:8], "big")},
        "primary": {
            "statistic": "target_gain_rate = (gold_arrivals - wrong_to_wrong) / n_channel_error",
            "null": "K=59 fresh matched-random directions at the identical absolute "
                    "perturbation norm and Router gate",
            "p_value": "p_add_one = (1 + count[target_gain_rate_random >= "
                       "target_gain_rate_real]) / (K + 1)",
            "is_intersection_claim": True,
            "no_subset_substitution_after_execution": True,
            "conjunction": [
                "evaluation channel-error support n_channel_error >= 30",
                "real target_gain_rate point estimate > 0",
                "bootstrap 95% lower bound for real target_gain_rate > 0",
                "p_add_one <= 0.05 against the K fresh randoms",
                "real target_hit point estimate > 0.50",
                "bootstrap 95% lower bound for real target_hit > 0.50",
                "clean_collateral_rate point estimate <= 0.05",
                "bootstrap 95% upper bound for clean_collateral_rate <= 0.05",
                "zero-control equality passes exactly",
                "no structural or provenance failure occurs"],
        },
        "secondary": {
            "non_promotable": True, "must_not_rescue_failed_primary": True,
            "ordered": ["d_grad versus frozen L26 DiffMean: paired difference in target_gain_rate",
                        "d_grad versus frozen score-space comparator: paired difference in "
                        "target_gain_rate and wrong_to_wrong rate",
                        "target_hit", "source_exits", "gold_arrivals", "wrong_to_wrong",
                        "Fixed / Broke / Net", "continuous target-source margin change",
                        "reverse contrast", "wrong-layer contrast", "ungated collateral"],
        },
        "deterministic_score_space_comparator": {
            "b_c": b_c, "calibrated_on": "DEV, at the selected dose",
            "applied": "e_gold += b_c/2 ; e_source -= b_c/2 ; others unchanged",
            "role": "frozen comparator; d_grad must exceed it for the non-reducibility claim",
        },
        "test_access_rule": (
            "SEALED TEST may be opened only by a separate, explicitly authorised one-shot "
            "formal execution task, and only after this freeze is committed and pushed and "
            "the gate-only verification passes. This task did not open it."),
        "void_rules": ["model/tokenizer/hash mismatch", "evaluation population mismatch",
                       "zero-control failure", "arm missing or reordered",
                       "direction/vector mismatch", "Router mismatch", "dose mismatch",
                       "batch-size change", "scientific runner modification",
                       "automatic retry or resume", "endpoint modification",
                       "random-vector replacement", "incomplete all-arm execution"],
        "void_reason_codes": ["MODEL_HASH_MISMATCH", "EVAL_POPULATION_MISMATCH",
                              "ZERO_CONTROL_FAILURE", "ARM_MISSING_OR_REORDERED",
                              "DIRECTION_MISMATCH", "ROUTER_MISMATCH", "DOSE_MISMATCH",
                              "BATCH_SIZE_CHANGE", "RUNNER_MODIFIED",
                              "AUTOMATIC_RETRY_OR_RESUME", "ENDPOINT_MODIFIED",
                              "RANDOM_VECTOR_REPLACED", "INCOMPLETE_ALL_ARM_EXECUTION"],
        "void_semantics": ("A natural scientific outcome, including insufficient "
                           "evaluation support or a failed conjunction, is NOT void. VOID "
                           "is mechanical only, requires manual restart, forbids automatic "
                           "retry, and forbids computing endpoints once declared."),
        "raw_record_schema": ["immutable sample id", "gold", "baseline four-mode scores",
                              "baseline prediction", "Router probability and decision",
                              "arm identity", "direction seed/hash", "dose and achieved "
                              "perturbation norm", "intervention four-mode scores",
                              "intervention prediction", "destination transition",
                              "batch index", "global execution order"],
        "formal_runner": {
            "status": "NOT_WRITTEN_IN_THIS_TASK",
            "requirement": "the formal runner must import the committed frozen readout and "
                           "these frozen artifacts, add no estimator and no arm, and pass "
                           "gate-only verification before any sealed row is opened",
            "development_runners_sha256": {f: C.sha_file(C.HERE / f) for f in runner_files},
        },
        "is_not": ["a cross-channel predictor validation",
                   "an untouched choice among all possible channels",
                   "an independent-dataset replication",
                   "a claim that gradient-based steering is novel in general",
                   "a confirmation that all Qwen3-4B channels are correctable"],
    }
    C.wj(C.HERE / "FORMAL_PROTOCOL_FROZEN.json", frozen)
    print("FORMAL_FREEZE_WRITTEN channel=%s q=%s s_c=%.6f b_c=%.6f K=%d"
          % (ch, q, s_c, b_c, K))
    print("  cos(random, d_grad): min %.5f median %.5f max %.5f"
          % (cg.min(), np.median(cg), cg.max()))


if __name__ == "__main__":
    main()
