#!/usr/bin/env python3
"""Qwen3-4B x When2Call — Stage A baseline on TRAIN+DEV only.

Prospective cross-model replication on the established When2Call measurement environment.
This is NOT an independent-dataset replication.

The readout is not redesigned. The committed Qwen3-8B Stage-0/1 module is imported and its
pure scoring functions are reused verbatim (`render_prompt`, `candidate_ids`,
`score_candidate`, `score_sample`, `MODES`, the long-sequence attention guard and the
determinism block). Only the model weights change.

SEALED_EVAL (548 rows) is never loaded, rendered, tokenised or scored. The runner refuses
any project_index outside the frozen TRAIN/DEV index.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path("/root/autodl-tmp/sakiko-followup")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_REPO = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_DIR = ROOT / ".cache/qwen3_4b" / MODEL_REVISION

RAW = ROOT / "data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl"
SPLIT_INDEX = ROOT / "final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"
PERM_SEED = 42
TOTAL_ROWS = 3652

# Frozen layer mapping, recomputed for Qwen3-4B rather than copied:
#   round(old / (old_n - 1) * (new_n - 1)) with the Qwen2.5 base (28 layers, L_obs 20, L_inj 16)
L_OBS = 26   # 20/27 -> 26/35, normalised depth 0.7407
L_INJ = 21   # 16/27 -> 21/35, normalised depth 0.5926


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_population():
    """Deterministic reconstruction, verified: project_index i -> raw[perm[i]]."""
    import csv
    import numpy as np
    raw = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    assert len(raw) == TOTAL_ROWS, len(raw)
    perm = np.random.default_rng(PERM_SEED).permutation(TOTAL_ROWS)
    rows = []
    for r in csv.DictReader(open(SPLIT_INDEX, encoding="utf-8")):
        pi = int(r["project_index"])
        s = raw[perm[pi]]
        assert s["uuid"] == r["sample_id"], (pi, s["uuid"], r["sample_id"])
        assert s["correct_answer"] == r["gold"]
        rows.append({"project_index": pi, "sample_id": s["uuid"], "gold": r["gold"],
                     "split": r["v2_split"], "sample": s})
    assert len(rows) == 3104, len(rows)
    allowed = {r["project_index"] for r in rows}
    sealed = sorted(set(range(TOTAL_ROWS)) - allowed)
    assert len(sealed) == 548
    return rows, set(sealed), perm, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pilot only; 0 = full TRAIN+DEV")
    a = ap.parse_args()

    import numpy as np
    import torch
    import qwen3_8b_stage0_1 as V1          # committed, hash-verified, import-safe

    rows, sealed, perm, raw = load_population()
    if a.limit:
        rows = rows[:a.limit]

    # SEALED refusal gate
    for r in rows:
        if r["project_index"] in sealed:
            print("QWEN3_4B_TEST_CONTAMINATION: %d" % r["project_index"], flush=True)
            raise SystemExit(2)

    V1.set_determinism()
    guard = V1.install_long_sequence_attention_guard()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
    model.eval()
    assert model.config.model_type == "qwen3"
    assert getattr(model.config, "_attn_implementation", None) == "eager"
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    assert L_OBS < n_layers and L_INJ < n_layers
    device = next(model.parameters()).device

    capture = {"L%d" % L_OBS: model.model.layers[L_OBS].mlp,
               "L%d" % L_INJ: model.model.layers[L_INJ].mlp}

    out_rows = HERE / "QWEN3_4B_BASELINE_ROWS.jsonl"
    A_obs = np.zeros((len(rows), hidden), dtype=np.float32)
    A_inj = np.zeros((len(rows), hidden), dtype=np.float32)
    t0 = time.perf_counter()
    with open(out_rows, "w", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(rows):
            res, cap = V1.score_sample(model, tok, device, r["sample"], capture)
            A_obs[i] = cap["L%d" % L_OBS]
            A_inj[i] = cap["L%d" % L_INJ]
            rec = {"project_index": r["project_index"], "sample_id": r["sample_id"],
                   "split": r["split"], "gold": r["gold"],
                   "prediction": res["prediction"], "runner_up": res["runner_up"],
                   "scores": res["scores"], "candidate_lengths": res["candidate_lengths"],
                   "prompt_tokens": res["prompt_tokens"],
                   "correct": res["prediction"] == r["gold"],
                   "transition": (None if res["prediction"] == r["gold"]
                                  else "%s->%s" % (r["gold"], res["prediction"])),
                   "row_order": i}
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            if (i + 1) % 100 == 0 or i == 0:
                el = time.perf_counter() - t0
                print("[%4d/%4d] %.1fs elapsed, %.2f rows/s" % (i + 1, len(rows), el, (i + 1) / el),
                      flush=True)
    np.save(HERE / ("QWEN3_4B_ACT_L%d.npy" % L_OBS), A_obs)
    np.save(HERE / ("QWEN3_4B_ACT_L%d.npy" % L_INJ), A_inj)
    env = {"model_repository": MODEL_REPO, "model_revision": MODEL_REVISION,
           "num_hidden_layers": n_layers, "hidden_size": hidden,
           "L_obs": L_OBS, "L_inj": L_INJ,
           "normalised_depth_obs": round(L_OBS / (n_layers - 1), 4),
           "normalised_depth_inj": round(L_INJ / (n_layers - 1), 4),
           "attention_guard": guard, "rows": len(rows),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    json.dump(env, open(HERE / "QWEN3_4B_STAGE_A_ENV.json", "w", encoding="utf-8", newline="\n"),
              indent=2, sort_keys=True)
    print("\nStage A complete: %d rows in %.1fs" % (len(rows), time.perf_counter() - t0))


if __name__ == "__main__":
    sys.exit(main())
