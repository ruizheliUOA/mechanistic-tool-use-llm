"""
phase9_positive_ood_feasibility.py — Part A: OOD feasibility audit across ALL confirmed positives.
==================================================================================================
CPU-ONLY. Uses domain/tool identities and SUPPORT COUNTS only. No intervention is run, and no
OOD outcome is inspected during candidate selection (per the brief).

Frozen support requirements checked (Gate-v2 Stage-1, unchanged):
  - SEEN train references      >= 80   (REF_FLOOR)
  - SEEN validation channel errors >= 30   (POWER_FLOOR)
  - UNSEEN evaluation channel errors >= 8  (eval floor)
  - (SEEN train channel errors >= 30)

Selection rule (PREDEFINED, applied in a fixed order):
  Iterate confirmed-positive channels in a fixed precedence order (strength of archived
  confirmation: z desc), and choose the FIRST that satisfies ALL frozen support requirements
  under at least one group-disjoint domain split. If none qualifies -> NO CANDIDATE.

Outputs: positive_ood_feasibility.csv
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

OUT = P.ROOT / "final" / "results" / "phase9_mechanism_and_ood"
OUT.mkdir(parents=True, exist_ok=True)
REF_FLOOR, POWER_FLOOR, EVAL_FLOOR, TRAIN_FLOOR = 80, 30, 8, 30


def is_lfs_pointer(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size < 200 and p.read_bytes()[:7] == b"version"
    except Exception:
        return False


def domain_of(ds, i):
    t = ds[i]["tools"]
    if not t:
        return "NO_TOOL"
    x = t[0]
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return "RAW"
    n = (x.get("name") or "?")
    return n.split("_")[0] if "_" in n else n.split(".")[0]


# Confirmed-positive inventory. `confirmed` reflects the FROZEN rule
# (z>=2 AND n_ge/N<=0.05 AND real>reverse) applied to the ARCHIVED locked-test battery.
CANDIDATES = [
    dict(key="qwen_w2c_ca_rfi", model="Qwen2.5-7B", dataset="W2C", channel="ca_rfi",
         gold="cannot_answer", pred="request_for_info", ref_gold="cannot_answer",
         z=4.82, n_random=20, n_ge=0, confirmed=True,
         note="Phase-3 locked test; multiseed +15.8, 5/5"),
    dict(key="qwen_metatool_tc_nt", model="Qwen2.5-7B", dataset="MetaTool-Binary", channel="tc_nt",
         gold="tool_call", pred="no_tool", ref_gold="tool_call",
         z=2.39, n_random=20, n_ge=0, confirmed=True,
         note="under-call; pca1 obs24/inj22 a4.0 thr0.7 AUC .983"),
    dict(key="qwen_metatool_nt_tc", model="Qwen2.5-7B", dataset="MetaTool-Binary", channel="nt_tc",
         gold="no_tool", pred="tool_call", ref_gold="no_tool",
         z=1.92, n_random=20, n_ge=1, confirmed=False,
         note="over-call; z=1.92 < 2 -> NOT confirmed under the frozen rule"),
    dict(key="qwen_w2c_rfi_tc", model="Qwen2.5-7B", dataset="W2C", channel="rfi_tc",
         gold="request_for_info", pred="tool_call", ref_gold="request_for_info",
         z=None, n_random=0, n_ge=None, confirmed=False,
         note="v1 cascade member; NO per-channel random battery (cascade-level n=10 only)"),
    dict(key="qwen_w2c_ca_tc", model="Qwen2.5-7B", dataset="W2C", channel="ca_tc",
         gold="cannot_answer", pred="tool_call", ref_gold="cannot_answer",
         z=None, n_random=0, n_ge=None, confirmed=False,
         note="v1 cascade member; no per-channel battery; archived reverse retains 46%"),
    dict(key="qwen_w2c_ca_direct", model="Qwen2.5-7B", dataset="W2C", channel="ca_direct",
         gold="cannot_answer", pred="direct", ref_gold="cannot_answer",
         z=None, n_random=0, n_ge=None, confirmed=False,
         note="v1 cascade member; no per-channel battery"),
    dict(key="phi_w2c_rfi_tc", model="Phi-3.5-mini", dataset="W2C", channel="rfi_tc",
         gold="request_for_info", pred="tool_call", ref_gold="request_for_info",
         z=None, n_random=0, n_ge=None, confirmed=False,
         note="summary-level only; SAKIKO-v1 manual design; no per-channel controls"),
    dict(key="phi_w2c_ca_tc", model="Phi-3.5-mini", dataset="W2C", channel="ca_tc",
         gold="cannot_answer", pred="tool_call", ref_gold="cannot_answer",
         z=None, n_random=0, n_ge=None, confirmed=False, note="summary-level only"),
    dict(key="phi_w2c_ca_direct", model="Phi-3.5-mini", dataset="W2C", channel="ca_direct",
         gold="cannot_answer", pred="direct", ref_gold="cannot_answer",
         z=None, n_random=0, n_ge=None, confirmed=False, note="summary-level only"),
]

DATA = {
    "W2C/Qwen2.5-7B": Path("data/processed/qwen25_7b_w2c/cache/baseline_details_regen.jsonl"),
    "MetaTool-Binary/Qwen2.5-7B": Path("final/results/metatool_qwen7b_sakiko_ca/baseline/metatool_qwen7b_baseline_details.jsonl"),
    "W2C/Phi-3.5-mini": Path("trace_db/w2c_phi35.jsonl"),
}


def w2c_support(cand, ds, rows, tr, va, te):
    ET = f"{cand['gold']}__{cand['pred']}"
    dom_err_tr = Counter(); dom_err_va = Counter(); dom_err_all = Counter(); dom_ref_tr = Counter()
    for (i, dom, gold, pred, et, sp) in rows:
        if et == ET:
            dom_err_all[dom] += 1
            if sp == "train": dom_err_tr[dom] += 1
            if sp == "val": dom_err_va[dom] += 1
        if sp == "train" and et == "correct" and gold == cand["ref_gold"]:
            dom_ref_tr[dom] += 1
    full = dict(ref_train=sum(dom_ref_tr.values()), err_train=sum(dom_err_tr.values()),
                err_val=sum(dom_err_va.values()), err_all=sum(dom_err_all.values()))
    # candidate splits: hold out the top-k domains by channel-error count (k=1..5)
    order = [d for d, _ in dom_err_all.most_common()]
    best = None
    trials = []
    for k in range(1, 6):
        held = set(order[:k])
        s_ref = sum(c for d, c in dom_ref_tr.items() if d not in held)
        s_etr = sum(c for d, c in dom_err_tr.items() if d not in held)
        s_eva = sum(c for d, c in dom_err_va.items() if d not in held)
        u_err = sum(c for d, c in dom_err_all.items() if d in held)
        ok = (s_ref >= REF_FLOOR and s_eva >= POWER_FLOOR and u_err >= EVAL_FLOOR and s_etr >= TRAIN_FLOOR)
        trials.append(dict(k=k, held=sorted(held), seen_ref=s_ref, seen_err_train=s_etr,
                           seen_err_val=s_eva, unseen_err=u_err, all_floors_pass=ok))
        if ok and best is None:
            best = trials[-1]
    return full, trials, best, dom_ref_tr


def main():
    ds = P.load_ds_raw()
    tr, va = set(P.TRAIN_IDX), set(P.VAL_IDX); te = set(P.TEST_IDX)
    # W2C/Qwen rows (support counts only)
    w2c_rows = []
    dpath = DATA["W2C/Qwen2.5-7B"]
    det = [json.loads(l) for l in open(dpath)]
    for i, d in enumerate(det):
        et = "correct" if d["pred"] == d["gold"] else f"{d['gold']}__{d['pred']}"
        sp = "train" if i in tr else ("val" if i in va else "test")
        w2c_rows.append((i, domain_of(ds, i), d["gold"], d["pred"], et, sp))

    out_rows = []
    for c in CANDIDATES:
        key = f"{c['dataset']}/{c['model']}"
        dp = DATA.get(key)
        data_available = (dp is not None and dp.exists() and not is_lfs_pointer(dp))
        row = dict(candidate=c["key"], model=c["model"], dataset=c["dataset"], channel=c["channel"],
                   archived_z=c["z"], n_random=c["n_random"], n_ge_real=c["n_ge"],
                   confirmed_positive=c["confirmed"], per_sample_data_available=data_available,
                   note=c["note"])
        if not c["confirmed"]:
            row.update(feasible="N/A — not a confirmed positive", blocker="not confirmed under frozen rule",
                       ref_train="", err_train="", err_val="", err_all="",
                       best_split="", seen_ref="", seen_err_val="", unseen_err="")
            out_rows.append(row); continue
        if not data_available:
            row.update(feasible="NO — data unavailable",
                       blocker=f"per-sample artifact is a Git-LFS pointer (objects not fetched): {dp}",
                       ref_train="UNAVAILABLE", err_train="UNAVAILABLE", err_val="UNAVAILABLE",
                       err_all="UNAVAILABLE", best_split="", seen_ref="", seen_err_val="", unseen_err="")
            out_rows.append(row); continue
        full, trials, best, dom_ref = w2c_support(c, ds, w2c_rows, tr, va, te)
        blocker = []
        if full["ref_train"] < REF_FLOOR:
            blocker.append(f"FULL-DATA ref pool {full['ref_train']} < {REF_FLOOR} before any split")
        row.update(ref_train=full["ref_train"], err_train=full["err_train"], err_val=full["err_val"],
                   err_all=full["err_all"],
                   feasible=("YES" if best else "NO"),
                   blocker=("; ".join(blocker) if blocker else ("none" if best else "no split satisfies all floors")),
                   best_split=(json.dumps(best["held"]) if best else ""),
                   seen_ref=(best["seen_ref"] if best else max(t["seen_ref"] for t in trials)),
                   seen_err_val=(best["seen_err_val"] if best else max(t["seen_err_val"] for t in trials)),
                   unseen_err=(best["unseen_err"] if best else max(t["unseen_err"] for t in trials)))
        row["_trials"] = trials
        out_rows.append(row)

    # ---- predefined selection rule ----
    confirmed = [r for r in out_rows if r["confirmed_positive"]]
    confirmed_sorted = sorted(confirmed, key=lambda r: -(r["archived_z"] or 0))
    selected = next((r for r in confirmed_sorted if r["feasible"] == "YES"), None)

    fields = [k for k in out_rows[0].keys() if k != "_trials"]
    with open(OUT / "positive_ood_feasibility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("=== CONFIRMED-POSITIVE INVENTORY (frozen rule: z>=2, n_ge/N<=0.05, real>reverse) ===")
    for r in out_rows:
        mark = "CONFIRMED" if r["confirmed_positive"] else "not-confirmed"
        print(f"  {r['candidate']:24s} {mark:14s} z={str(r['archived_z']):5s} data={r['per_sample_data_available']}")
    print("\n=== FEASIBILITY (confirmed positives only) ===")
    for r in confirmed_sorted:
        print(f"  {r['candidate']}: feasible={r['feasible']} | blocker={r['blocker']}")
        if r.get("_trials"):
            for t in r["_trials"]:
                print(f"      held(top-{t['k']})={t['held']}: seen_ref={t['seen_ref']} "
                      f"seen_err_val={t['seen_err_val']} unseen_err={t['unseen_err']} -> "
                      f"{'PASS' if t['all_floors_pass'] else 'FAIL'}")
    print("\n=== SELECTION RULE OUTCOME ===")
    print("  precedence (z desc):", [r["candidate"] for r in confirmed_sorted])
    print("  SELECTED:", selected["candidate"] if selected else "*** NO CANDIDATE — OOD NOT FEASIBLE ***")
    print("\nwrote", OUT / "positive_ood_feasibility.csv")


if __name__ == "__main__":
    main()
