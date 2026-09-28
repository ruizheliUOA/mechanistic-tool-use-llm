"""
run_qwen7b_method_diag_gpu.py — GPU-confirmation driver for the Qwen2.5-7B method diagnostic
============================================================================================
This is a thin, non-invasive driver used to run the pending GPU confirmation of the
Qwen2.5-7B method diagnostic (ca_direct L16-vs-L20 intervention Net; rfi_tc original-vs-
balanced-negative router Net). It does NOT change any diagnostic logic; it only:

  1. Monkeypatches `qwen7b_native_sakiko.load_meta_and_dataset` so the W2C mcq split loads
     from a local copy of the raw jsonl. This is an ENVIRONMENT-COMPAT fix only: the pinned
     `datasets` version on this GPU box raises DatasetGenerationCastError on the heterogeneous
     `orig_tools` column when using `load_dataset("nvidia/When2Call","test",split="mcq")`.
     The local loader reproduces the EXACT same rows and the EXACT same `.shuffle(seed=42)`
     ordering — verified by the original uuid-alignment assertion against the archived
     baseline-details file (ds[i].uuid == baseline_details[i].uuid for all 3652 rows).
  2. Dispatches to the archived stage functions unchanged.

Stages:
  extract      : native acts at OBS_LAYERS [12,16,20]  (qwen7b_native_sakiko.stage_extract)
  analyze      : routers + DiffMean directions          (qwen7b_native_sakiko.stage_analyze)
  diag_extract : diagnostic extra acts at L18,L22        (qwen7b_method_diagnostics.stage_extract)
  diag         : Parts A/B/C intervention Net            (qwen7b_method_diagnostics.stage_diag)

Usage:
  python scripts/run_qwen7b_method_diag_gpu.py --stage extract
  python scripts/run_qwen7b_method_diag_gpu.py --stage analyze
  python scripts/run_qwen7b_method_diag_gpu.py --stage diag_extract
  python scripts/run_qwen7b_method_diag_gpu.py --stage diag
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen7b_native_sakiko as P

LOCAL_MCQ = P.ROOT / "data" / "processed" / "qwen25_7b_w2c" / "w2c_test_mcq.jsonl"


def _robust_load_meta_and_dataset():
    """Drop-in replacement for P.load_meta_and_dataset that avoids the datasets cast error.
    Builds the mcq split from the local raw jsonl, normalizes the column union, applies the
    identical shuffle(seed=SEED), then reuses the archived meta-derivation + uuid assertion."""
    from datasets import Dataset
    rows = [json.loads(l) for l in open(LOCAL_MCQ)]
    keys = sorted({k for r in rows for k in r.keys()})
    norm = [{k: r.get(k, None) for k in keys} for r in rows]
    ds = Dataset.from_list(norm).shuffle(seed=P.SEED)

    det = [json.loads(l) for l in open(P.BASE_DETAILS)]
    assert len(ds) == len(det), (len(ds), len(det))
    meta = []
    for i, (s, d) in enumerate(zip(ds, det)):
        assert s["uuid"] == d["uuid"], i  # exact-order guarantee (same as archived loader)
        gold, pred = d["gold"], d["pred"]
        correct = (pred == gold)
        if correct:
            etype = "correct"
        elif gold == "request_for_info" and pred == "tool_call":
            etype = "rfi_tc"
        elif gold == "cannot_answer" and pred == "tool_call":
            etype = "ca_tc"
        elif gold == "cannot_answer" and pred == "direct":
            etype = "ca_direct"
        else:
            etype = f"{gold}__{pred}"
        meta.append({"idx": i, "uuid": s["uuid"], "gold": gold, "pred": pred,
                     "correct": correct, "etype": etype})
    return ds, meta


# install the compat loader on the shared module (both stage modules resolve it here)
P.load_meta_and_dataset = _robust_load_meta_and_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["extract", "analyze", "diag_extract", "diag"])
    a = ap.parse_args()
    if a.stage == "extract":
        P.stage_extract(SimpleNamespace(force=False))
    elif a.stage == "analyze":
        P.stage_analyze(SimpleNamespace(force=False))
    else:
        import qwen7b_method_diagnostics as D
        # D imported P at top; ensure it sees the patched loader too
        D.P.load_meta_and_dataset = _robust_load_meta_and_dataset
        (D.stage_extract if a.stage == "diag_extract" else D.stage_diag)()


if __name__ == "__main__":
    main()
