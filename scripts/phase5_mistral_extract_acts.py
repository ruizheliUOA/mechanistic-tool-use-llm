"""
phase5_mistral_extract_acts.py — cache Mistral MLP-output activations at obs layers.
====================================================================================
One model pass over all 3652 W2C samples; captures the MLP output at the LAST prompt
token (the decision position used by the avg_logp readout) for each obs layer in
OBS_LAYERS. Caches are float32 .npy under the OUT-OF-REPO cache dir (never committed).

Row index == seed-42-shuffled W2C order == split index == baseline-details row.

Usage:
  python scripts/phase5_mistral_extract_acts.py            # extract (skips if cached)
  python scripts/phase5_mistral_extract_acts.py --force
"""
from __future__ import annotations
import argparse, gc, json, time
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_mistral_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = {lyr: L.CACHE_DIR / f"acts_L{lyr}.npy" for lyr in L.OBS_LAYERS}
    if all(p.exists() for p in paths.values()) and not args.force:
        L.log.info("all obs-layer caches exist; skipping.")
        return

    ds = L.load_ds_raw()
    n = len(ds)
    model, tok, dev = L.load_model(assert_bf16=True)
    D = model.config.hidden_size
    acts = {lyr: np.zeros((n, D), dtype=np.float32) for lyr in L.OBS_LAYERS}
    mlps = {lyr: model.model.layers[lyr].mlp for lyr in L.OBS_LAYERS}
    skipped = []
    t0 = time.time()
    for i in tqdm(range(n), desc="extract", ncols=90):
        pt = L.make_prompt_text(tok, ds[i])
        ids = tok.encode(pt, add_special_tokens=False)
        if len(ids) > L.MAXTOK:
            skipped.append(i); continue
        last = len(ids) - 1
        inp = torch.tensor([ids], device=dev)
        cap, handles = {}, []
        for lyr in L.OBS_LAYERS:
            def mk(_lyr):
                def hook(m, _in, out, __l=_lyr):
                    o = out[0] if isinstance(out, tuple) else out
                    cap[__l] = o[0, last, :].detach().float().cpu().numpy()
                return hook
            handles.append(mlps[lyr].register_forward_hook(mk(lyr)))
        with torch.no_grad():
            model(inp)
        for h in handles:
            h.remove()
        for lyr in L.OBS_LAYERS:
            acts[lyr][i] = cap[lyr]
        del inp
        if (i + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    for lyr in L.OBS_LAYERS:
        np.save(paths[lyr], acts[lyr])
    elapsed = time.time() - t0
    L.log.info("extraction done in %.1f min; skipped=%d", elapsed / 60, len(skipped))

    OUTG = L.OUT / "geometry"; OUTG.mkdir(parents=True, exist_ok=True)
    json.dump({"model_repo": L.MODEL_REPO, "revision": L.MODEL_REVISION,
               "n": n, "hidden": D, "obs_layers": L.OBS_LAYERS,
               "normalized_depth": {str(lyr): round(lyr / L.N_LAYERS, 3) for lyr in L.OBS_LAYERS},
               "decision_position": "last_prompt_token",
               "cache_dir": str(L.CACHE_DIR), "skipped": skipped,
               "elapsed_min": round(elapsed / 60, 1)},
              open(OUTG / "mistral7b_activation_extraction.json", "w"), indent=2)


if __name__ == "__main__":
    main()
