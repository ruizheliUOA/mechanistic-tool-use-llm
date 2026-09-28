"""
phase3_baseline_regen.py — regenerate the Qwen2.5-7B W2C baseline (LFS content unavailable).
=============================================================================================
Byte-identical readout to the archived eval_w2c_7b_baseline.py (avg_logp over 4 candidates,
bf16, unbatched, same prompts), driven by the archived shim loader (local raw jsonl,
shuffle seed=42). Writes:

  data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl   (gitignored)
  final/results/w2c_qwen_new_channel_intervention/baseline_regen/
      BASELINE_REGEN_VERIFICATION.md + baseline_regen_summary.json  (small text, committable)

Blocking verification vs archived aggregates per PHASE3_PROTOCOL.md §3.
"""
from __future__ import annotations
import gc, json, logging, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_lib as L
import qwen7b_native_sakiko as P

log = logging.getLogger("p3.baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

OUTB = L.OUT / "baseline_regen"
OUTB.mkdir(parents=True, exist_ok=True)
ARCH_DECOMP = L.ROOT / "final/results/7b_w2c_baseline/qwen25_7b_error_decomposition.json"
ARCH_DISC = L.ROOT / "final/results/cross_dataset_channel_discovery/w2c_qwen_channel_discovery.json"


def main():
    ds = L.load_ds_raw()
    n = len(ds)
    model, tok, dev = P.load_model()

    done = []
    if L.BASE_DETAILS_REGEN.exists():
        done = [json.loads(l) for l in open(L.BASE_DETAILS_REGEN)]
        log.info("resume: %d rows already scored", len(done))
    f = open(L.BASE_DETAILS_REGEN, "a")
    t0 = time.time()
    for i in range(len(done), n):
        s = ds[i]
        prompt = P.make_prompt_text(tok, s)
        pids = tok.encode(prompt, add_special_tokens=False)
        if len(pids) > 8192:
            f.write(json.dumps({"uuid": s["uuid"], "gold": s["correct_answer"],
                                "pred": None, "skipped": True}) + "\n")
            continue
        prompt_ids = torch.tensor([pids], device=dev)
        scored = {}
        for lb in ["direct", "tool_call", "request_for_info", "cannot_answer"]:
            cand = s["answers"].get(lb, "")
            if not cand:
                scored[lb] = -1e9
                continue
            cids = tok.encode(cand, add_special_tokens=False)
            inp = torch.cat([prompt_ids, torch.tensor([cids], device=dev)], dim=1)
            with torch.no_grad():
                logits = model(inp).logits
            sl = logits[0, len(pids) - 1: len(pids) + len(cids) - 1, :]
            lp = torch.log_softmax(sl.float(), dim=-1)
            scored[lb] = sum(lp[j, cids[j]].item() for j in range(len(cids))) / len(cids)
            del inp, logits
        pred = max(scored, key=scored.get)
        f.write(json.dumps({"uuid": s["uuid"], "gold": s["correct_answer"], "pred": pred,
                            "correct": bool(pred == s["correct_answer"]),
                            "avg_logp": {k: round(v, 4) for k, v in scored.items()}},
                           ensure_ascii=False) + "\n")
        del prompt_ids
        if (i + 1) % 200 == 0:
            f.flush(); gc.collect(); torch.cuda.empty_cache()
            el = time.time() - t0
            log.info("%d/%d (%.2fs/sample, ETA %.0f min)", i + 1, n,
                     el / (i + 1 - len(done)), el / (i + 1 - len(done)) * (n - i - 1) / 60)
    f.close()
    verify()


def verify():
    det = [json.loads(l) for l in open(L.BASE_DETAILS_REGEN)]
    n = len(det)
    skipped = sum(1 for d in det if d.get("skipped"))
    acc = sum(1 for d in det if d.get("correct")) / n
    cm = Counter((d["gold"], d["pred"]) for d in det if not d.get("skipped"))
    arch = json.loads(ARCH_DECOMP.read_text())
    acm = arch["confusion_matrix"]
    diffs = []
    for g in ["tool_call", "request_for_info", "cannot_answer"]:
        for p in ["direct", "tool_call", "request_for_info", "cannot_answer"]:
            ours, theirs = cm.get((g, p), 0), acm[g][p]
            if ours != theirs:
                diffs.append({"gold": g, "pred": p, "regen": ours, "archived": theirs})
    # per-split channel counts vs discovery JSON
    tr, va, te = L.load_splits()
    split_of = {}
    for nm, idxs in [("train", tr), ("val", va), ("test", te)]:
        for i in idxs:
            split_of[i] = nm
    etypes = []
    for i, d in enumerate(det):
        g, p = d["gold"], d["pred"]
        etypes.append("correct" if g == p else f"{g}__{p}")
    disc = json.loads(ARCH_DISC.read_text())
    chan_check = []
    for c in disc["discovered_channels"]:
        et = f"{c['gold']}__{c['pred']}"
        cnts = Counter(split_of[i] for i, e in enumerate(etypes) if e == et)
        chan_check.append({"channel": c["channel_id"],
                           "regen": [sum(cnts.values()), cnts.get("train", 0),
                                     cnts.get("val", 0), cnts.get("test", 0)],
                           "archived": [c["count_all"], c["count_train"],
                                        c["count_val"], c["count_test"]]})
    max_cell_diff = max((abs(d["regen"] - d["archived"]) for d in diffs), default=0)
    max_chan_diff = max(max(abs(a - b) for a, b in zip(cc["regen"], cc["archived"]))
                        for cc in chan_check)
    ok = (n == 3652 and skipped == 0 and abs(acc - arch["accuracy"]) <= 0.005
          and max_cell_diff <= 5 and max_chan_diff <= 3)
    summary = {"n": n, "skipped": skipped, "accuracy": round(acc, 5),
               "archived_accuracy": arch["accuracy"],
               "confusion_diffs": diffs, "max_cell_diff": max_cell_diff,
               "per_split_channel_check": chan_check, "max_channel_split_diff": max_chan_diff,
               "verdict": "PASS" if ok else "FAIL"}
    (OUTB / "baseline_regen_summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUTB / "BASELINE_REGEN_VERIFICATION.md", "w") as f:
        f.write("# Phase 3 — Baseline Regeneration Verification\n\n")
        f.write(f"- n={n}, skipped={skipped}, accuracy={acc:.5f} "
                f"(archived {arch['accuracy']}, |d|={abs(acc-arch['accuracy']):.5f})\n")
        f.write(f"- confusion-matrix cells differing from archive: {len(diffs)} "
                f"(max |d| = {max_cell_diff}, tolerance 5)\n")
        for d in diffs:
            f.write(f"  - {d['gold']}->{d['pred']}: regen {d['regen']} vs archived {d['archived']}\n")
        f.write(f"- per-split channel counts (all/train/val/test), tolerance 3 "
                f"(max |d| = {max_chan_diff}):\n\n")
        f.write("| channel | regen | archived |\n|---|---|---|\n")
        for cc in chan_check:
            f.write(f"| {cc['channel']} | {cc['regen']} | {cc['archived']} |\n")
        f.write(f"\n**Verdict: {summary['verdict']}**\n")
    log.info("verification: %s (acc %.5f, max cell diff %d, max chan diff %d)",
             summary["verdict"], acc, max_cell_diff, max_chan_diff)
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        verify()
    else:
        main()
