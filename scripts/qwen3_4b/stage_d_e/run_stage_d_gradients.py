#!/usr/bin/env python3
"""Stage D1 — frozen TRAIN-only gradient-direction estimation, plus the DEV gradient
objects the frozen geometry gate consumes.

The estimator is not reimplemented.  `gradient_object` below is the Stage 1.5 sequence
object verbatim: for each of the four frozen modes, the committed
`qwen3_8b_stage0_1.gradient_candidate` is called at L_inj and the per-position gradient is
summed over prompt and candidate positions (full-effect aggregation).  Then

    d_grad = unit( mean_i unit( G_i,gold - G_i,source ) )

over the frozen TRAIN error population, ordered by immutable UUID.

The only change relative to the harness that OOM'd is the loader: parameter gradients are
disabled exactly as the committed `load_model()` does.  That was proven bitwise-equivalent
on Qwen3-8B in `research_exploration/qwen3_4b_gradient_memory_amendment_v1`.

TRAIN gradients build the direction.  DEV gradients are computed in the same pass but are
written to a separate file and are NOT read by the direction construction.

SEALED evaluation is never touched.
"""

import argparse
import json
import time

import numpy as np

import stage_d_e_common as C


def gradient_object(v1, model, tok, device, sample):
    """The exact Stage 1.5 object: G_i,m at L_inj, full-effect positions."""
    prompt, _, _ = v1.render_prompt(tok, sample)
    G, n_tok = {}, None
    for m in v1.MODES:
        pids, cids = v1.candidate_ids(tok, prompt, sample["answers"][m])
        _score, grad, _state = v1.gradient_candidate(model, device, pids, cids)
        if n_tok is None:
            n_tok = (len(pids), grad.shape[0])
        G[m] = grad.sum(axis=0).astype(np.float32)
    return np.stack([G[m] for m in v1.MODES]).astype(np.float32), n_tok


def run(split, limit=0):
    import torch
    from safetensors.numpy import save_file

    ctx = C.Ctx(need_model=True)
    v1 = ctx.v1
    env = {}
    env["gradient_checkpointing"] = ctx.v2.install_gradient_checkpointing(ctx.model, C.L_INJ)
    env["chunk_checkpointed_attention"] = ctx.v2.install_chunk_checkpointed_attention(v1)
    env["attention_guard"] = ctx.guard
    env["amendment_param_grads_disabled"] = True
    env["n_params_requiring_grad"] = sum(1 for p in ctx.model.parameters() if p.requires_grad)

    C.BIG.mkdir(parents=True, exist_ok=True)
    runner_hash = C.sha_file(__file__)
    common_hash = C.sha_file(C.HERE / "stage_d_e_common.py")
    frozen_hash = C.sha_file(C.ROOT / "scripts/qwen3_8b_stage0_1.py")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    tensors, records, per_channel = {}, [], {}
    rec_path = C.HERE / ("_GRAD_RECORDS_%s.jsonl" % split)
    with open(rec_path, "w", encoding="utf-8", newline="\n") as fh:
        for ch, g, p in C.CHANNELS:
            gi, pi = v1.MODES.index(g), v1.MODES.index(p)
            pop = ctx.errors(ch, split)
            exp = C.EXPECTED[ch]["train_err" if split == "train" else "dev_err"]
            if len(pop) != exp:
                raise RuntimeError("population drift %s %s: %d != %d" % (ch, split, len(pop), exp))
            C.assert_sealed_clean(ctx, pop)
            if limit:
                pop = pop[:limit]
            W, nonfinite = [], 0
            for ordinal, r in enumerate(pop):
                G, (n_prompt, n_seq) = gradient_object(
                    v1, ctx.model, ctx.tok, ctx.device, ctx.sample_of[r["project_index"]])
                finite = bool(np.all(np.isfinite(G)))
                z = G[gi].astype(np.float64) - G[pi].astype(np.float64)
                nz = float(np.linalg.norm(z))
                gnorms = {m: float(np.linalg.norm(G[k].astype(np.float64)))
                          for k, m in enumerate(v1.MODES)}
                status = "VALID"
                if not finite:
                    status = "NON_FINITE_GRADIENT"
                    nonfinite += 1
                elif nz <= v1.CONTRAST_NORM_TOL:
                    status = "CONTRAST_NORM_BELOW_TOLERANCE"
                w = (z / nz) if status == "VALID" else None
                rec = {"sample_id": r["sample_id"], "project_index": r["project_index"],
                       "channel": ch, "split": split, "gold": g, "source": p,
                       "ordinal": ordinal, "status": status, "finite": finite,
                       "target_contrast_norm": nz,
                       "unit_contrast_norm": (float(np.linalg.norm(w)) if w is not None else None),
                       "per_mode_gradient_norms": gnorms,
                       "intervention_site_tensor_shape": [int(n_seq), C.HIDDEN],
                       "aggregated_object_shape": list(G.shape),
                       "prompt_tokens": int(n_prompt), "sequence_tokens": int(n_seq),
                       "layer": C.L_INJ, "runner_sha256": runner_hash,
                       "frozen_module_sha256": frozen_hash}
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                records.append(rec)
                if status != "VALID":
                    # The frozen protocol hard-stops rather than dropping a sample.
                    raise RuntimeError("%s for %s" % (status, r["sample_id"]))
                W.append(w)
                key = "gradient__%s__%04d" % (ch, ordinal)
                tensors[key] = G
                tensors["w__%s__%04d" % (ch, ordinal)] = w.astype(np.float32)
                if (ordinal + 1) % 25 == 0:
                    el = time.perf_counter() - t0
                    print("  %s %s %d/%d  %.1fs  peak %.2f GiB"
                          % (ch, split, ordinal + 1, len(pop), el,
                             torch.cuda.max_memory_allocated(0) / 2 ** 30), flush=True)
            Wm = np.stack(W)
            wbar = Wm.mean(0)
            wbar_norm = float(np.linalg.norm(wbar))
            d = (wbar / wbar_norm).astype(np.float32)
            per_channel[ch] = {
                "gold": g, "source": p, "n": len(pop), "non_finite": nonfinite,
                "wbar_norm": wbar_norm,
                "chance_reference_one_over_sqrt_n": 1.0 / float(np.sqrt(len(pop))),
                "d_grad_norm": float(np.linalg.norm(d.astype(np.float64))),
                "d_grad_sha256": C.sha_arr(d),
                "population_sha256": v1.index_order_hash([r["sample_id"] for r in pop]),
                "target_contrast_norm_median": float(np.median([x["target_contrast_norm"]
                                                                for x in records
                                                                if x["channel"] == ch])),
            }
            if split == "train":
                tensors["d_grad__" + ch] = d
            print("%s [%s]: n=%d  ||wbar||=%.6f  chance=%.6f"
                  % (ch, split, len(pop), wbar_norm, 1 / np.sqrt(len(pop))), flush=True)

    out = C.BIG / ("QWEN3_4B_%s_GRADIENTS.safetensors" % split.upper())
    save_file(tensors, out, metadata={
        "schema_version": "1", "scope": "%s_only_injection_site_gradients" % split,
        "model": "%s@%s" % (C.MODEL_REPO, C.MODEL_REVISION),
        "layer": str(C.L_INJ),
        "site": "model.model.layers[L].mlp forward output, after MLP internals "
                "and before decoder residual addition",
        "aggregation": "full-effect: sum over prompt and candidate positions",
        "mode_order": json.dumps(v1.MODES), "dimension": str(C.HIDDEN),
        "contains_dataset_text": "false"})
    C.wj(C.HERE / ("_STAGE_D_%s_SUMMARY.json" % split), {
        "split": split, "per_channel": per_channel,
        "runtime_seconds": time.perf_counter() - t0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2 ** 30,
        "gradients_file": str(out), "gradients_sha256": C.sha_file(out),
        "runner_sha256": runner_hash, "common_sha256": common_hash,
        "frozen_module_sha256": frozen_hash,
        "amendment_package_sha256": C.sha_file(AMEND_HASHES),
        "environment": env, "sealed_evaluation_accessed": False})
    print("STAGE_D_%s_COMPLETE  %.1fs  peak %.2f GiB"
          % (split.upper(), time.perf_counter() - t0,
             torch.cuda.max_memory_allocated(0) / 2 ** 30))


AMEND_HASHES = C.AMEND / "HASHES.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.split, a.limit)
