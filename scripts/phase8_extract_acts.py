"""
phase8_extract_acts.py — cache MLP-output activations at the frozen normalized-depth layers.
=============================================================================================
Captures the MLP output at the LAST prompt token (the avg_logp decision position) for every
layer needed by the frozen grid: obs grid round({0.40,0.55,0.70,0.85}*L) plus every injection
candidate (obs + {0,-0.06L,-0.12L}) and the wrong-layer control (~0.15L).

Rows: TRAIN+VAL only for the intervention pipeline; caches live OUTSIDE the repo.
(Test rows are NOT extracted — the Stage-6 locked test scores them live, after decisions are
frozen and hashed.)

Usage: python scripts/phase8_extract_acts.py
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_lib as L


def needed_layers():
    layers = set(L.obs_grid())
    for o in L.obs_grid():
        layers |= set(L.inj_candidates(o))
    layers.add(L.wrong_layer())
    return sorted(layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layers = needed_layers()
    paths = {lyr: L.CACHE / f"acts_L{lyr}.npy" for lyr in layers}
    if all(p.exists() for p in paths.values()) and not args.force:
        L.log.info("all caches exist: %s", layers); return

    ds = L.load_ds_raw()
    tr, va = L.splits_train_val()
    rows = sorted(set(tr) | set(va))
    L.assert_no_test(rows, "extract")            # firewall: train+val only
    n = len(ds)
    model, tok, dev = L.load_model()
    D = model.config.hidden_size
    acts = {lyr: np.zeros((n, D), dtype=np.float32) for lyr in layers}
    mlps = {lyr: model.model.layers[lyr].mlp for lyr in layers}
    skipped = []
    t0 = time.time()
    for i in tqdm(rows, desc=f"extract L{layers}", ncols=90):
        ids = tok.encode(L.make_prompt_text(tok, ds[i]), add_special_tokens=False)
        if len(ids) > L.MAXTOK:
            skipped.append(i); continue
        last = len(ids) - 1
        inp = torch.tensor([ids], device=dev)
        cap, handles = {}, []
        for lyr in layers:
            def mk(_l):
                def hook(m, _in, out, __l=_l):
                    o = out[0] if isinstance(out, tuple) else out
                    cap[__l] = o[0, last, :].detach().float().cpu().numpy()
                return hook
            handles.append(mlps[lyr].register_forward_hook(mk(lyr)))
        with torch.no_grad():
            model(inp)
        for h in handles:
            h.remove()
        for lyr in layers:
            acts[lyr][i] = cap[lyr]
        del inp
        if len(skipped) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    for lyr in layers:
        np.save(paths[lyr], acts[lyr])
    el = time.time() - t0
    L.log.info("extracted %d rows (train+val) at layers %s in %.1f min; skipped %d",
               len(rows), layers, el / 60, len(skipped))
    json.dump({"model": L.MODELS[L.MODEL_KEY]["repo"], "revision": L.MODELS[L.MODEL_KEY]["revision"],
               "layers": layers, "n_layers_total": L.n_layers(), "hidden": D,
               "normalized_depth": {str(l): round(l / L.n_layers(), 3) for l in layers},
               "rows_extracted": "train+val only (test firewalled)", "n_rows": len(rows),
               "skipped": skipped, "cache_dir": str(L.CACHE), "elapsed_min": round(el / 60, 1)},
              open(L.OUT / "llama_activation_extraction.json", "w"), indent=2)


if __name__ == "__main__":
    main()
