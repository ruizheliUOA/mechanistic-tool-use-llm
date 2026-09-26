"""
phase9_reference_learning_curve.py — Part B: reference-pool learning-curve audit.
=================================================================================
CPU-ONLY, DIAGNOSTIC ONLY. Uses historical train/validation artifacts (cached activations).
Does NOT change Gate v2 and does NOT reclassify any prior outcome.

Question: how does DiffMean direction quality depend on the number of reference samples?
Subject : Qwen ca_rfi (ref pool = 79, the channel the frozen REF_FLOOR=80 rejects by one).
Control : Qwen rfi_tc (ref pool = 277, a plentiful-reference channel) — shows what a converged
          curve looks like when references are NOT scarce.

Per reference subsample size n in {20,30,40,50,60,70,max}:
  - split-half direction cosine        (two disjoint halves of the n refs; err pool fixed)
  - bootstrap direction stability      (resample n refs with replacement; cos to full-pool dir)
  - direction norm convergence         (||DiffMean|| vs n)
  - cos to the DEPLOYED full-pool direction  (CPU proxy for behavioural equivalence)
  - uncertainty intervals              (2.5/97.5 percentiles over B repeats)

NOTE ON "validation behavioral response": measuring Net/z at each n requires running the model
(GPU). It is NOT computable from artifacts and is therefore reported as DEFERRED-GPU rather
than invented. `cos_to_deployed` is provided as the principled CPU proxy: if a subsampled
direction is ~collinear with the deployed one, its behavioural response is ~the deployed one's.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7_lib as P

OUT = P.ROOT / "final" / "results" / "phase9_mechanism_and_ood"
OUT.mkdir(parents=True, exist_ok=True)
B = 200
SIZES = [20, 30, 40, 50, 60, 70]

SUBJECTS = [
    dict(key="qwen_ca_rfi", model="qwen25_7b", channel="ca_rfi", gold="cannot_answer",
         pred="request_for_info", ref_gold="cannot_answer", obs=20,
         role="SUBJECT (ref=79; frozen REF_FLOOR=80 rejects by one)"),
    dict(key="qwen_rfi_tc", model="qwen25_7b", channel="rfi_tc", gold="request_for_info",
         pred="tool_call", ref_gold="request_for_info", obs=20,
         role="CONTROL (ref=277; plentiful references)"),
]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def diffmean(A, err_idx, ref_idx):
    raw = A[ref_idx].mean(0) - A[err_idx].mean(0)
    n = float(np.linalg.norm(raw))
    return raw / (n + 1e-12), n


def ci(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    meta = P.load_meta("qwen25_7b")
    tr, va = P.splits()
    P.assert_no_test(tr, "lc/tr")
    rows = []
    for S in SUBJECTS:
        A = np.load(P.MODELS[S["model"]]["cache"] / f"acts_L{S['obs']}.npy")
        et = f"{S['gold']}__{S['pred']}"
        err = [i for i in tr if meta[i]["etype"] == et]
        ref = [i for i in tr if meta[i]["correct"] and meta[i]["gold"] == S["ref_gold"]]
        n_ref = len(ref)
        d_full, norm_full = diffmean(A, err, ref)     # the deployed direction
        sizes = [n for n in SIZES if n <= n_ref] + [n_ref]
        print(f"\n=== {S['key']} — {S['role']} ===")
        print(f"    n_err_train={len(err)}  n_ref_train={n_ref}  ||d_full||={norm_full:.4f}")
        print(f"    {'n_ref':>6} {'splithalf cos':>22} {'boot cos->deployed':>24} "
              f"{'||d||':>16} {'cos_to_deployed':>22}")
        for n in sizes:
            rng = np.random.RandomState(42)
            sh, bt, nn, cd = [], [], [], []
            for _ in range(B):
                sub = list(rng.choice(ref, size=n, replace=False))
                # split-half (disjoint halves of the SAME n refs; err pool fixed)
                h = n // 2
                if h >= 2:
                    d1, _ = diffmean(A, err, sub[:h])
                    d2, _ = diffmean(A, err, sub[h:2 * h])
                    sh.append(float(d1 @ d2))
                # point estimate at this n
                d_n, norm_n = diffmean(A, err, sub)
                nn.append(norm_n); cd.append(float(d_n @ d_full))
                # bootstrap: resample n refs WITH replacement -> cos to deployed
                bs = list(rng.choice(sub, size=n, replace=True))
                d_b, _ = diffmean(A, err, bs)
                bt.append(float(d_b @ d_full))
            m_sh, lo_sh, hi_sh = ci(sh) if sh else (float("nan"),) * 3
            m_bt, lo_bt, hi_bt = ci(bt)
            m_nn, lo_nn, hi_nn = ci(nn)
            m_cd, lo_cd, hi_cd = ci(cd)
            tag = " (max/deployed)" if n == n_ref else ""
            print(f"    {n:>6}{tag:16s} {m_sh:.3f} [{lo_sh:.3f},{hi_sh:.3f}]   "
                  f"{m_bt:.3f} [{lo_bt:.3f},{hi_bt:.3f}]   {m_nn:.3f} [{lo_nn:.2f},{hi_nn:.2f}]   "
                  f"{m_cd:.3f} [{lo_cd:.3f},{hi_cd:.3f}]")
            rows.append(dict(subject=S["key"], role=S["role"], channel=S["channel"],
                             n_err_train=len(err), n_ref_available=n_ref, n_ref_used=n,
                             is_max=(n == n_ref),
                             splithalf_cos_mean=round(m_sh, 4), splithalf_ci_lo=round(lo_sh, 4),
                             splithalf_ci_hi=round(hi_sh, 4),
                             bootstrap_cos_to_deployed_mean=round(m_bt, 4),
                             bootstrap_ci_lo=round(lo_bt, 4), bootstrap_ci_hi=round(hi_bt, 4),
                             dir_norm_mean=round(m_nn, 4), dir_norm_ci_lo=round(lo_nn, 4),
                             dir_norm_ci_hi=round(hi_nn, 4), dir_norm_full=round(norm_full, 4),
                             cos_to_deployed_mean=round(m_cd, 4), cos_to_deployed_ci_lo=round(lo_cd, 4),
                             cos_to_deployed_ci_hi=round(hi_cd, 4),
                             validation_behavioral_response="DEFERRED-GPU (not computable from artifacts)",
                             n_bootstrap=B))
    with open(OUT / "phase9_reference_learning_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(OUT / "phase9_reference_learning_curve.json", "w"), indent=2)
    print("\nwrote", OUT / "phase9_reference_learning_curve.csv")


if __name__ == "__main__":
    main()
