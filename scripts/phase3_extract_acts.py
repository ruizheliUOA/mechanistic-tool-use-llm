"""
phase3_extract_acts.py — regenerate seed-invariant W2C activations at L12/16/20/24.
====================================================================================
Identical hook convention to the archived qwen7b_native_sakiko.stage_extract (MLP output,
last prompt token, fp32), extended by L24 per PHASE3_PROTOCOL.md §4. Caches are gitignored.
"""
from __future__ import annotations
import gc, json, logging, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L
import qwen7b_native_sakiko as P

log = logging.getLogger("p3.extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    paths = {Lx: L.CACHE_DIR / f"acts_L{Lx}.npy" for Lx in L.OBS_LAYERS}
    if all(p.exists() for p in paths.values()):
        log.info("all caches exist; nothing to do")
        return
    ds = L.load_ds_raw()
    n = len(ds)
    model, tok, dev = P.load_model()
    D = model.config.hidden_size
    acts = {Lx: np.zeros((n, D), dtype=np.float32) for Lx in L.OBS_LAYERS}
    mlps = {Lx: model.model.layers[Lx].mlp for Lx in L.OBS_LAYERS}
    t0 = time.time()
    skipped = []
    for i in range(n):
        prompt = P.make_prompt_text(tok, ds[i])
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) > 8192:
            skipped.append(i)
            continue
        last = len(ids) - 1
        inp = torch.tensor([ids], device=dev)
        cap, handles = {}, []
        for Lx in L.OBS_LAYERS:
            def mk(Lx):
                def hook(m, _in, out, _L=Lx):
                    o = out[0] if isinstance(out, tuple) else out
                    cap[_L] = o[0, last, :].detach().float().cpu().numpy()
                return hook
            handles.append(mlps[Lx].register_forward_hook(mk(Lx)))
        with torch.no_grad():
            model(inp)
        for h in handles:
            h.remove()
        for Lx in L.OBS_LAYERS:
            acts[Lx][i] = cap[Lx]
        del inp
        if (i + 1) % 400 == 0:
            gc.collect(); torch.cuda.empty_cache()
            log.info("%d/%d (%.1f min elapsed)", i + 1, n, (time.time() - t0) / 60)
    for Lx in L.OBS_LAYERS:
        np.save(paths[Lx], acts[Lx])
    log.info("extraction done in %.1f min; skipped=%d; layers=%s",
             (time.time() - t0) / 60, len(skipped), L.OBS_LAYERS)
    (L.OUT / "baseline_regen" / "extraction_report.json").write_text(json.dumps({
        "layers": L.OBS_LAYERS, "n": n, "hidden": D, "skipped": len(skipped),
        "dtype": "float32", "hook": "mlp_output_last_prompt_token",
        "minutes": round((time.time() - t0) / 60, 1)}, indent=2))


if __name__ == "__main__":
    main()
