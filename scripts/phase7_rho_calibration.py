"""
phase7_rho_calibration.py — measure architecture-normalized perturbation scale (TRAIN ONLY).
=============================================================================================
Measures median_train(||h_MLP(L)||) for every observation/injection layer used by the Phase-7
development channels, and emits the exact rho <-> alpha map.

This is a MEASUREMENT of the models, not an intervention and not an experiment outcome: it
involves no steering, no random directions, no validation metric. It therefore legitimately
precedes the freezing of the rho grid (which the protocol requires to be justified by
activation norms).

FIREWALL: forward passes are run on TRAIN rows only (assert_no_test enforced).

Usage: python scripts/phase7_rho_calibration.py
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

# layers required by the development channels (obs and inj)
NEEDED = {
    "qwen25_7b":    {"cached": [16, 20, 24], "to_measure": [18]},   # ca_rfi inj=18; tc_rfi obs/inj=24
    "mistral7b_v03": {"cached": [18, 22], "to_measure": [20]},      # rfi_tc inj=20; ca_tc/ca_direct inj=18
}
OUTF = P.OUT / "rho_calibration.json"


def median_norm_from_cache(model_key, L, tr):
    p = P.MODELS[model_key]["cache"] / f"acts_L{L}.npy"
    if not p.exists():
        return None
    A = np.load(p, mmap_mode="r")
    n = np.linalg.norm(np.asarray(A[tr]), axis=1)
    return float(np.median(n)), float(np.sqrt((np.asarray(A[tr]) ** 2).mean()))


def measure_layers(model_key, layers, tr, ds):
    """One forward pass per TRAIN row; capture ||MLP output|| at the requested layers."""
    P.assert_no_test(tr, "rho_calibration/train")
    model, tok, dev = P.load_model(model_key)
    norms = {L: [] for L in layers}
    sq_sum = {L: 0.0 for L in layers}
    n_elem = {L: 0 for L in layers}
    mlps = {L: model.model.layers[L].mlp for L in layers}
    skipped = 0
    t0 = time.time()
    for i in tqdm(tr, desc=f"calib {model_key} L{layers}", ncols=90):
        ids = tok.encode(P.make_prompt_text(tok, ds[i]), add_special_tokens=False)
        if len(ids) > P.MAXTOK:
            skipped += 1; continue
        last = len(ids) - 1
        inp = torch.tensor([ids], device=dev)
        cap, handles = {}, []
        for L in layers:
            def mk(_L):
                def hook(m, _in, out, __L=_L):
                    o = out[0] if isinstance(out, tuple) else out
                    cap[__L] = o[0, last, :].detach().float().cpu().numpy()
                return hook
            handles.append(mlps[L].register_forward_hook(mk(L)))
        with torch.no_grad():
            model(inp)
        for h in handles:
            h.remove()
        for L in layers:
            v = cap[L]
            norms[L].append(float(np.linalg.norm(v)))
            sq_sum[L] += float((v ** 2).sum()); n_elem[L] += v.size
        del inp
        if len(norms[layers[0]]) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    del model
    gc.collect(); torch.cuda.empty_cache()
    P.log.info("measured %s in %.1f min (skipped %d)", model_key, (time.time() - t0) / 60, skipped)
    return {L: (float(np.median(norms[L])), float(np.sqrt(sq_sum[L] / n_elem[L]))) for L in layers}


def main():
    tr, va = P.splits()
    ds = P.load_ds_raw()
    out = {"note": "median/RMS of MLP-output at last prompt token, TRAIN rows only",
           "n_train": len(tr), "models": {}}
    for mk, req in NEEDED.items():
        M = P.MODELS[mk]
        entry = {"hidden": M["hidden"], "layers_total": M["layers"], "layer_scale": {}}
        for L in req["cached"]:
            r = median_norm_from_cache(mk, L, tr)
            if r:
                entry["layer_scale"][str(L)] = {"median_norm_train": round(r[0], 5),
                                                "rms_train": round(r[1], 6), "source": "cache"}
        need = [L for L in req["to_measure"] if str(L) not in entry["layer_scale"]]
        if need:
            meas = measure_layers(mk, need, tr, ds)
            for L, (med, rms) in meas.items():
                entry["layer_scale"][str(L)] = {"median_norm_train": round(med, 5),
                                                "rms_train": round(rms, 6),
                                                "source": "measured_phase7_train_only"}
        out["models"][mk] = entry
    json.dump(out, open(OUTF, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", OUTF)


if __name__ == "__main__":
    main()
